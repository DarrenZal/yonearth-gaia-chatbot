"""
Graph-based retriever using precomputed GraphIndex to fetch chunks and KG evidence quickly.

Query-time flow:
  - Extract entities from query via lexicon (names/aliases)
  - Fetch entity neighborhoods and prelinked chunk_ids via entity_chunk_map
  - Expand 1-hop (cap edges) and collect best triples/evidence from adjacency
  - Return candidate chunks and supporting triples
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Set

from langchain_core.documents import Document

from .graphrag_index import GraphIndex

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    chunks: List[Document]
    triples: List[Dict[str, Any]]
    matched_entities: List[str]


class GraphRetriever:
    def __init__(self, graph: Optional[GraphIndex] = None, max_edges: int = 50):
        self.graph = graph or GraphIndex.load()
        self.max_edges = max_edges

    def _chunk_to_document(self, chunk_id: str) -> Optional[Document]:
        meta = self.graph.hierarchy.get(chunk_id)
        if not meta:
            return None
        # We don't persist chunk text in the index to keep it light; use a placeholder preview
        preview = f"[Chunk {chunk_id}] {meta.get('book_title')} - Chapter {meta.get('chapter_number')}: {meta.get('chapter_title')}"
        return Document(page_content=preview, metadata={"chunk_id": chunk_id, **meta})

    def _expand_neighbors(self, entity_ids: List[str]) -> Tuple[Set[str], List[Dict[str, Any]]]:
        """Expand 1-hop neighbors capped by max_edges; collect triple-like records for evidence."""
        seen: Set[Tuple[str, str, str]] = set()
        triples: List[Dict[str, Any]] = []
        expanded: Set[str] = set(entity_ids)
        edge_budget = self.max_edges
        for eid in entity_ids:
            for edge in self.graph.adjacency.get(eid, [])[: self.max_edges]:
                if edge_budget <= 0:
                    break
                nid = edge.get("neighbor_id")
                pred = edge.get("predicate")
                key = (eid, pred, nid)
                if key in seen:
                    continue
                seen.add(key)
                expanded.add(nid)
                # capture an evidence triple with sample context
                triples.append({
                    "source": eid,
                    "predicate": pred,
                    "target": nid,
                    "p_true": edge.get("p_true", 1.0),
                    "context": (edge.get("contexts") or [""])[0],
                })
                edge_budget -= 1
        return expanded, triples

    def retrieve(self, query: str, k: int = 20) -> GraphRetrievalResult:
        matched = self.graph.find_entities_in_text(query)
        logger.info(f"GraphRetriever matched entities: {matched}")
        if not matched:
            return GraphRetrievalResult(chunks=[], triples=[], matched_entities=[])

        # Expand neighborhoods 1-hop
        expanded_entities, triples = self._expand_neighbors(matched)

        # Collect chunks linked to matched and expanded entities
        chunk_scores: Dict[str, float] = {}
        def bump(chunk_id: str, score: float):
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score

        for eid in matched:
            for cid in self.graph.entity_chunk_map.get(eid, [])[: k * 3]:
                bump(cid, 2.0)  # direct match boost

        for eid in expanded_entities:
            if eid in matched:
                continue
            for cid in self.graph.entity_chunk_map.get(eid, [])[: k * 2]:
                bump(cid, 1.0)  # neighbor boost

        # Rank and convert to Documents
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        docs: List[Document] = []
        for chunk_id, _ in ranked[:k]:
            doc = self._chunk_to_document(chunk_id)
            if doc:
                docs.append(doc)

        return GraphRetrievalResult(chunks=docs, triples=triples[: k * 2], matched_entities=matched)

