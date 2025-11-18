"""
Graph-based retriever using precomputed GraphIndex to fetch chunks and KG evidence quickly.

Query-time flow:
  - Extract entities from query via lexicon (names/aliases) with optional fuzzy matching
  - Fetch entity neighborhoods and prelinked chunk_ids via entity_chunk_map
  - Expand up to N hops (cap edges) and collect best triples/evidence from adjacency
  - Optionally retrieve semantic clusters/topics and boost their chunks
  - Return candidate chunks and supporting triples
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from langchain_openai import OpenAIEmbeddings

from ..utils.lc_compat import Document
from ..config import settings
from .graphrag_index import GraphIndex

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    chunks: List[Document]
    triples: List[Dict[str, Any]]
    matched_entities: List[str]
    matched_clusters: List[Dict[str, Any]] = field(default_factory=list)


class GraphRetriever:
    def __init__(
        self,
        graph: Optional[GraphIndex] = None,
        max_edges: Optional[int] = None,
        max_hops: Optional[int] = None,
    ):
        self.graph = graph or GraphIndex.load()
        # Cap the total number of edges explored for efficiency
        self.max_edges: int = max_edges if max_edges is not None else settings.graph_max_edges
        # Multi-hop configuration
        self.max_hops: int = max_hops if max_hops is not None else settings.graph_retrieval_max_hops
        self.hop_decay: float = settings.graph_hop_decay

        # Optional cluster/topic retrieval state
        self._cluster_index: Dict[str, Any] = self.graph.cluster_index or {}
        self._cluster_ids: Optional[List[str]] = None
        self._cluster_embeddings: Optional[np.ndarray] = None
        self._cluster_embedder: Optional[OpenAIEmbeddings] = None
        self._init_cluster_vectors()

    def _chunk_to_document(self, chunk_id: str) -> Optional[Document]:
        meta = self.graph.hierarchy.get(chunk_id)
        if not meta:
            return None
        # Prefer stored preview if available; otherwise build a metadata-based preview
        text_preview = (self.graph.previews or {}).get(chunk_id)
        if not text_preview:
            # Fallback placeholder preview
            bt = meta.get('book_title') or meta.get('title')
            chn = meta.get('chapter_number')
            cht = meta.get('chapter_title')
            text_preview = f"[Chunk {chunk_id}] {bt} - Chapter {chn}: {cht}"
        # Attempt to backfill episode_number when missing/unknown from title
        try:
            epn = meta.get('episode_number')
            if (epn is None) or (str(epn).lower() == 'unknown'):
                title = meta.get('title') or meta.get('chapter_title') or ''
                import re
                m = re.search(r'episode\s*(\d+)', str(title), flags=re.IGNORECASE)
                if m:
                    meta = {**meta, 'episode_number': int(m.group(1))}
        except Exception:
            pass
        return Document(page_content=text_preview, metadata={"chunk_id": chunk_id, **meta})

    def _init_cluster_vectors(self) -> None:
        """Precompute cluster ID list and embedding matrix, if available."""
        if not settings.graph_enable_cluster_retrieval:
            return

        clusters = (self._cluster_index or {}).get("clusters") or {}
        if not clusters:
            return

        ids: List[str] = []
        vectors: List[List[float]] = []
        for cid, info in clusters.items():
            emb = info.get("embedding")
            if isinstance(emb, list) and emb:
                try:
                    # Ensure all entries are floats
                    vec = [float(x) for x in emb]
                    ids.append(cid)
                    vectors.append(vec)
                except Exception:
                    continue

        if not ids or not vectors:
            return

        try:
            self._cluster_ids = ids
            self._cluster_embeddings = np.asarray(vectors, dtype=float)
        except Exception:
            self._cluster_ids = None
            self._cluster_embeddings = None

    def _ensure_cluster_embedder(self) -> None:
        """Lazily initialize the embeddings model used for cluster similarity."""
        if self._cluster_embedder is not None:
            return
        try:
            self._cluster_embedder = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize cluster embedder: {exc}")
            self._cluster_embedder = None

    def _expand_neighbors(
        self,
        entity_ids: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, float], List[Dict[str, Any]]]:
        """Expand neighbors with limited multi-hop BFS.

        Returns:
            entity_hops: entity_id -> minimum hop distance from any seed (0 for matched entities)
            entity_confidence: entity_id -> max p_true observed on incoming edges
            triples: sample triple-like records for evidence, annotated with hop distance
        """
        entity_hops: Dict[str, int] = {eid: 0 for eid in entity_ids}
        entity_confidence: Dict[str, float] = {eid: 1.0 for eid in entity_ids}
        seen_edges: Set[Tuple[str, str, str]] = set()
        triples: List[Dict[str, Any]] = []
        edge_budget = max(self.max_edges, 0)

        if edge_budget == 0 or not entity_ids:
            return entity_hops, entity_confidence, triples

        queue: deque[str] = deque(entity_ids)
        while queue and edge_budget > 0:
            eid = queue.popleft()
            current_hop = entity_hops.get(eid, 0)
            if current_hop >= self.max_hops:
                continue

            for edge in self.graph.adjacency.get(eid, [])[: self.max_edges]:
                if edge_budget <= 0:
                    break
                nid = edge.get("neighbor_id")
                pred = edge.get("predicate")
                if not nid or not pred:
                    continue

                key = (eid, pred, nid)
                if key in seen_edges:
                    continue
                seen_edges.add(key)

                hop_distance = current_hop + 1
                # Track minimum hop distance
                if nid not in entity_hops or hop_distance < entity_hops[nid]:
                    entity_hops[nid] = hop_distance

                # Track confidence using max p_true over incoming edges
                p_true = float(edge.get("p_true", 1.0))
                entity_confidence[nid] = max(entity_confidence.get(nid, 0.0), p_true)

                # Capture an evidence triple with sample context and hop info
                triples.append(
                    {
                        "source": eid,
                        "predicate": pred,
                        "target": nid,
                        "p_true": p_true,
                        "hop": hop_distance,
                        "context": (edge.get("contexts") or [""])[0],
                    }
                )
                edge_budget -= 1

                # Only continue BFS if we have not yet exceeded hop limit
                if hop_distance < self.max_hops:
                    queue.append(nid)

        return entity_hops, entity_confidence, triples

    def _cluster_candidates_for_query(
        self,
        query: str,
        k_clusters: int = 5,
        k_chunks: int = 20,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Retrieve cluster-based candidate chunks for a query.

        Returns:
            matched_clusters: list of cluster metadata dicts
            chunk_boosts: chunk_id -> additional score derived from cluster similarity
        """
        if (
            not settings.graph_enable_cluster_retrieval
            or self._cluster_ids is None
            or self._cluster_embeddings is None
        ):
            return [], {}

        self._ensure_cluster_embedder()
        if self._cluster_embedder is None:
            return [], {}

        try:
            query_vec = np.asarray(self._cluster_embedder.embed_query(query), dtype=float)
        except Exception as exc:
            logger.warning(f"Cluster query embedding failed: {exc}")
            return [], {}

        if not np.any(query_vec):
            return [], {}

        # Cosine similarity between query and cluster centroids
        cluster_matrix = self._cluster_embeddings
        denom = (np.linalg.norm(cluster_matrix, axis=1) * np.linalg.norm(query_vec))
        # Avoid division by zero
        denom = np.where(denom == 0, 1e-9, denom)
        sims = np.dot(cluster_matrix, query_vec) / denom

        # Select top clusters by similarity
        k_clusters = max(1, min(k_clusters, len(self._cluster_ids)))
        top_indices = np.argsort(sims)[-k_clusters:][::-1]

        matched_clusters: List[Dict[str, Any]] = []
        chunk_boosts: Dict[str, float] = {}
        clusters = (self._cluster_index or {}).get("clusters") or {}

        # Normalize similarities into a small boost range
        for idx in top_indices:
            cid = self._cluster_ids[idx]
            sim = float(sims[idx])
            info = clusters.get(cid, {})
            chunk_ids = info.get("chunk_ids") or []
            if not chunk_ids:
                continue

            # Store cluster match metadata for diagnostics
            matched_clusters.append(
                {
                    "cluster_id": cid,
                    "similarity": sim,
                    "summary": info.get("summary"),
                    "chunk_count": len(chunk_ids),
                }
            )

            # Soft boost for chunks belonging to this cluster
            # Scale similarity into [0.3, 1.0] before combining with other scores
            normalized = max(0.0, sim)
            boost = 0.3 + 0.7 * normalized
            for cid_chunk in chunk_ids[:k_chunks]:
                chunk_boosts[cid_chunk] = max(chunk_boosts.get(cid_chunk, 0.0), boost)

        return matched_clusters, chunk_boosts

    def retrieve(self, query: str, k: int = 20) -> GraphRetrievalResult:
        matched = self.graph.find_entities_in_text(query)
        logger.info(f"GraphRetriever matched entities: {matched}")
        if not matched:
            # Cluster-based retrieval can still provide useful book/section chunks
            matched_clusters, cluster_boosts = self._cluster_candidates_for_query(query, k_clusters=5, k_chunks=k * 3)
            if not matched_clusters:
                return GraphRetrievalResult(chunks=[], triples=[], matched_entities=[], matched_clusters=[])

            # Build documents solely from cluster-based candidates
            chunk_scores: Dict[str, float] = {}

            def bump_only(chunk_id: str, score: float) -> None:
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score

            for cid, boost in cluster_boosts.items():
                bump_only(cid, boost)

            ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            docs: List[Document] = []
            for chunk_id, _ in ranked[:k]:
                doc = self._chunk_to_document(chunk_id)
                if doc:
                    docs.append(doc)

            return GraphRetrievalResult(
                chunks=docs,
                triples=[],
                matched_entities=[],
                matched_clusters=matched_clusters,
            )

        # Expand neighborhoods 1-hop
        entity_hops, entity_conf, triples = self._expand_neighbors(matched)

        # Collect chunks linked to matched and expanded entities
        chunk_scores: Dict[str, float] = {}

        def bump(chunk_id: str, score: float) -> None:
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score

        # Direct entity matches get a higher base boost
        for eid in matched:
            for cid in self.graph.entity_chunk_map.get(eid, [])[: k * 3]:
                confidence = entity_conf.get(eid, 1.0)
                bump(cid, 2.0 * confidence)  # direct match boost scaled by confidence

        # Neighbor entities (multi-hop) get decayed boosts by hop distance
        for eid, hop in entity_hops.items():
            if eid in matched:
                continue
            hop_factor = self.hop_decay**max(hop, 1)
            confidence = entity_conf.get(eid, 1.0)
            neighbor_boost = 1.0 * hop_factor * confidence
            for cid in self.graph.entity_chunk_map.get(eid, [])[: k * 2]:
                bump(cid, neighbor_boost)

        # Cluster/topic-based boosts (optional) to capture semantic neighborhoods
        matched_clusters: List[Dict[str, Any]] = []
        try:
            matched_clusters, cluster_boosts = self._cluster_candidates_for_query(
                query,
                k_clusters=5,
                k_chunks=k * 3,
            )
            for cid, boost in cluster_boosts.items():
                bump(cid, boost)
        except Exception as exc:
            logger.debug(f"Cluster-based graph boosts skipped due to error: {exc}")

        # Co-occurrence bonus: favor chunks that contain multiple matched entities
        try:
            # Build entity -> type map for simple type-aware boosts
            ent_types = {}
            for item in self.graph.entities_lexicon.get("entities", []):
                ent_types[item.get("id")] = (item.get("type") or "").upper()
            matched_set = set(matched)
            for chunk_id in list(chunk_scores.keys()):
                ents_in_chunk = set(self.graph.chunk_entity_map.get(chunk_id, []))
                co = matched_set.intersection(ents_in_chunk)
                if len(co) >= 2:
                    # Stronger co-occurrence weighting: scale with count and slightly amplify existing score
                    base = chunk_scores.get(chunk_id, 0.0)
                    co_count = len(co)
                    # Additive bonus + multiplicative factor
                    bonus = 1.2 + 0.4 * (co_count - 2)
                    factor = 1.0 + 0.15 * (co_count - 1)
                    types = {ent_types.get(eid, "") for eid in co}
                    if "PERSON" in types and "ORGANIZATION" in types:
                        bonus += 0.8
                        factor += 0.15
                    chunk_scores[chunk_id] = base * factor + bonus
        except Exception as e:
            logger.debug(f"Co-occurrence bonus skipped due to error: {e}")

        # Deprioritize any legacy 'epunknown' chunks to favor clean episode labeling
        try:
            for chunk_id in list(chunk_scores.keys()):
                if isinstance(chunk_id, str) and chunk_id.startswith('epunknown:'):
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) - 2.0
        except Exception:
            pass

        # Rank and convert to Documents
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        docs: List[Document] = []
        for chunk_id, _ in ranked[:k]:
            doc = self._chunk_to_document(chunk_id)
            if doc:
                docs.append(doc)

        return GraphRetrievalResult(
            chunks=docs,
            triples=triples[: k * 2],
            matched_entities=matched,
            matched_clusters=matched_clusters,
        )

    def cluster_search(self, query: str, k: int = 20) -> List[Document]:
        """Public helper to retrieve chunks purely via semantic clusters.

        This is primarily intended for diagnostics and offline evaluation, and
        does not depend on entity lexical matching.
        """
        matched_clusters, chunk_boosts = self._cluster_candidates_for_query(
            query,
            k_clusters=5,
            k_chunks=k * 3,
        )
        if not matched_clusters or not chunk_boosts:
            return []

        ranked = sorted(chunk_boosts.items(), key=lambda x: x[1], reverse=True)
        docs: List[Document] = []
        for chunk_id, _ in ranked[:k]:
            doc = self._chunk_to_document(chunk_id)
            if doc:
                docs.append(doc)
        return docs
