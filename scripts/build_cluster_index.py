#!/usr/bin/env python3
"""
Build multi-level semantic cluster index for GraphRAG.

This offline script reads the existing GraphRAG index under:
  - data/graph_index/hierarchy.json
  - data/graph_index/previews.json
  - data/graph_index/chunk_entity_map.json

It then:
  - Computes embeddings for chunk-level previews using OpenAIEmbeddings
    (settings.openai_embedding_model)
  - Clusters chunks into a coarse level of topic clusters
  - For sufficiently large coarse clusters, creates finer-grained subclusters
  - Computes a centroid embedding for each cluster (coarse and fine)
  - Generates a short natural-language summary for each cluster using
    the same chat model Gaia uses (settings.openai_model)

Output:
  - data/graph_index/cluster_index.json

Expected JSON shape (backwards-compatible with GraphRetriever):

{
  "levels": {
    "coarse": ["coarse_000", "coarse_001", "..."],
    "fine": ["fine_000_00", "fine_000_01", "..."]
  },
  "clusters": {
    "<cluster_id>": {
      "chunk_ids": ["chunk_id_1", "chunk_id_2", "..."],
      "summary": "Short human-readable summary of this topic cluster.",
      "embedding": [0.123, -0.456, ...],
      "level": "coarse" | "fine",
      "parent": "coarse_000",          # for fine clusters (optional)
      "children": ["fine_000_00", ...] # for coarse clusters (optional)
    },
    ...
  },
  "meta": {
    "n_chunks": 12345,
    "n_clusters_total": 200,
    "n_clusters_coarse": 60,
    "n_clusters_fine": 140,
    "embedding_model": "...",
    "chat_model": "...",
    "created_at": "...",
    "params": {
      "coarse_k": 60,
      "min_fine_per_coarse": 2,
      "max_fine_per_coarse": 8,
      "min_cluster_size": 25
    }
  }
}

GraphRetriever currently uses only the "clusters" mapping and expects each
cluster entry to expose "chunk_ids", "summary", and "embedding". The extra
fields ("levels", "level", "parent", "children", "meta") are additive and
safe to ignore at query time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.cluster import KMeans

# Add project root to path so we can import src.*
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings  # type: ignore
from src.rag.graphrag_index import GraphIndex  # type: ignore
from src.utils.lc_compat import HumanMessage  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for multi-level clustering."""

    coarse_k: Optional[int] = None
    min_coarse: int = 50
    max_coarse: int = 100
    coarse_divisor: int = 800  # heuristic: ~N/800 coarse clusters

    min_cluster_size: int = 20  # do not keep clusters below this size

    min_fine_per_coarse: int = 2
    max_fine_per_coarse: int = 8
    fine_divisor: int = 200  # heuristic: ~size/200 fine clusters per coarse
    min_fine_cluster_size: int = 10

    summary_max_examples: int = 8
    summary_max_child_summaries: int = 5
    summary_snippet_chars: int = 220


class ClusterIndexBuilder:
    """Offline builder for multi-level GraphRAG cluster_index.json."""

    def __init__(
        self,
        config: Optional[ClusterConfig] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        self.config = config or ClusterConfig()

        # Load GraphIndex from existing JSON files
        self.graph: GraphIndex = GraphIndex.load()

        # Paths
        self.output_path: Path = output_path or (
            settings.data_dir / "graph_index" / "cluster_index.json"
        )

        # LLM + embeddings
        self.embedder = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.summarizer = ChatOpenAI(
            model_name=settings.openai_model,
            temperature=0.2,
            max_tokens=256,
            openai_api_key=settings.openai_api_key,
        )

    # ------------------------------------------------------------------
    # Chunk preparation and embedding
    # ------------------------------------------------------------------
    def _build_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Construct a text snippet for a chunk using previews + hierarchy."""
        meta = self.graph.hierarchy.get(chunk_id) or {}
        preview = (self.graph.previews or {}).get(chunk_id)
        if preview:
            return str(preview)

        # Fallback lightweight preview from metadata
        book_title = meta.get("book_title") or meta.get("title") or ""
        chapter_number = meta.get("chapter_number")
        chapter_title = meta.get("chapter_title") or ""

        pieces: List[str] = []
        if book_title:
            pieces.append(str(book_title))
        if chapter_number is not None or chapter_title:
            ch = f"Chapter {chapter_number}" if chapter_number is not None else ""
            if chapter_title:
                if ch:
                    ch += f": {chapter_title}"
                else:
                    ch = str(chapter_title)
            if ch:
                pieces.append(ch)

        if not pieces:
            return None

        return " - ".join(pieces)

    def _prepare_chunks(self) -> Tuple[List[str], List[str]]:
        """Return sorted chunk_ids and their text previews."""
        chunk_ids: List[str] = []
        texts: List[str] = []
        for cid in sorted(self.graph.hierarchy.keys()):
            text = self._build_chunk_text(cid)
            if not text:
                continue
            chunk_ids.append(cid)
            texts.append(text)

        logger.info("Prepared %d chunks with text for clustering", len(chunk_ids))
        return chunk_ids, texts

    def _embed_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed chunk texts into a dense matrix, batching to limit memory usage.

        This version intentionally avoids keeping multiple copies of all
        embeddings in memory (no list-of-floats cache), which helps prevent
        the OS from killing the process on large corpora.
        """
        n_chunks = len(chunk_ids)
        if n_chunks == 0:
            raise RuntimeError("No chunks available for clustering.")

        logger.info("Embedding %d chunks for clustering (batch_size=%d)", n_chunks, batch_size)

        matrix: Optional[np.ndarray] = None
        row = 0

        for start in range(0, n_chunks, batch_size):
            end = min(start + batch_size, n_chunks)
            batch_texts = texts[start:end]
            try:
                vectors = self.embedder.embed_documents(batch_texts)
            except Exception as exc:
                logger.error("Failed to embed batch starting at %d: %s", start, exc)
                raise

            if not vectors:
                continue

            if matrix is None:
                dim = len(vectors[0])
                if dim <= 0:
                    raise RuntimeError("Embedding model returned zero-dimension vectors.")
                matrix = np.zeros((n_chunks, dim), dtype=np.float32)

            for local_idx, vec in enumerate(vectors):
                matrix[row + local_idx] = np.asarray(vec, dtype=np.float32)

            row += len(vectors)

        if matrix is None or row == 0:
            raise RuntimeError("No embeddings available for clustering.")

        if row != n_chunks:
            matrix = matrix[:row, :]

        logger.info("Final embedding matrix has shape %s", matrix.shape)
        return matrix

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def _auto_coarse_k(self, n_chunks: int) -> int:
        if self.config.coarse_k is not None:
            return max(1, self.config.coarse_k)
        k = max(1, n_chunks // max(self.config.coarse_divisor, 1))
        k = max(self.config.min_coarse, k)
        k = min(self.config.max_coarse, k)
        return k

    def _cluster_coarse(
        self,
        chunk_ids: List[str],
        matrix: np.ndarray,
    ) -> Tuple[Dict[int, List[int]], np.ndarray]:
        """Run coarse-level KMeans clustering over all chunks."""
        n_chunks = len(chunk_ids)
        if n_chunks != matrix.shape[0]:
            raise ValueError("Number of chunk_ids and embedding rows must match.")

        k = self._auto_coarse_k(n_chunks)
        logger.info("Running coarse KMeans with k=%d on %d chunks", k, n_chunks)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)
        centers = kmeans.cluster_centers_

        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)

        # Filter out very small clusters (optional)
        filtered_clusters: Dict[int, List[int]] = {}
        for label, indices in clusters.items():
            if len(indices) < self.config.min_cluster_size:
                logger.debug(
                    "Dropping tiny coarse cluster %d with %d items (< %d)",
                    label,
                    len(indices),
                    self.config.min_cluster_size,
                )
                continue
            filtered_clusters[label] = indices

        logger.info(
            "Coarse clustering produced %d clusters (k=%d, min_size=%d)",
            len(filtered_clusters),
            k,
            self.config.min_cluster_size,
        )
        return filtered_clusters, centers

    def _cluster_fine_for_coarse(
        self,
        coarse_label: int,
        member_indices: List[int],
        matrix: np.ndarray,
    ) -> Dict[int, List[int]]:
        """Sub-cluster a single coarse cluster into finer clusters."""
        size = len(member_indices)
        if size < max(self.config.min_fine_cluster_size, self.config.min_fine_per_coarse):
            logger.debug(
                "Skipping fine clustering for coarse %d: size=%d < min_fine_cluster_size=%d",
                coarse_label,
                size,
                self.config.min_fine_cluster_size,
            )
            return {}

        # Heuristic: number of fine clusters for this coarse cluster
        n_fine = max(
            self.config.min_fine_per_coarse,
            min(self.config.max_fine_per_coarse, size // max(self.config.fine_divisor, 1)),
        )
        n_fine = min(n_fine, size)
        if n_fine <= 1:
            logger.debug(
                "Skipping fine clustering for coarse %d: n_fine=%d (size=%d)",
                coarse_label,
                n_fine,
                size,
            )
            return {}

        sub_matrix = matrix[member_indices, :]
        logger.debug(
            "Running fine KMeans for coarse %d with k=%d on %d chunks",
            coarse_label,
            n_fine,
            size,
        )
        kmeans = KMeans(n_clusters=n_fine, random_state=42, n_init=10)
        labels = kmeans.fit_predict(sub_matrix)

        fine_clusters: Dict[int, List[int]] = defaultdict(list)
        for offset, label in enumerate(labels):
            # Map back to global row index
            fine_clusters[int(label)].append(member_indices[offset])

        # Ensure there are no degenerate empty clusters
        fine_clusters = {lbl: idxs for lbl, idxs in fine_clusters.items() if idxs}
        return fine_clusters

    def _build_clusters(
        self,
        chunk_ids: List[str],
        matrix: np.ndarray,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]]:
        """Run multi-level clustering and build raw cluster structures."""
        coarse_clusters, _ = self._cluster_coarse(chunk_ids, matrix)

        clusters: Dict[str, Dict[str, Any]] = {}
        levels: Dict[str, List[str]] = {"coarse": [], "fine": []}

        # Statistics helpers
        coarse_sizes: List[int] = []
        fine_sizes: List[int] = []

        for coarse_idx, member_indices in coarse_clusters.items():
            coarse_chunk_ids = [chunk_ids[i] for i in member_indices]
            coarse_vectors = matrix[member_indices, :]
            coarse_centroid = coarse_vectors.mean(axis=0)

            coarse_id = f"coarse_{coarse_idx:03d}"
            clusters[coarse_id] = {
                "chunk_ids": coarse_chunk_ids,
                "embedding": coarse_centroid.tolist(),
                "level": "coarse",
                "children": [],
            }
            levels["coarse"].append(coarse_id)
            coarse_sizes.append(len(coarse_chunk_ids))

            # Fine-level clusters inside this coarse cluster
            fine_clusters = self._cluster_fine_for_coarse(coarse_idx, member_indices, matrix)
            for fine_local_idx, fine_member_indices in fine_clusters.items():
                fine_chunk_ids = [chunk_ids[i] for i in fine_member_indices]
                fine_vectors = matrix[fine_member_indices, :]
                fine_centroid = fine_vectors.mean(axis=0)

                fine_id = f"fine_{coarse_idx:03d}_{fine_local_idx:02d}"
                clusters[fine_id] = {
                    "chunk_ids": fine_chunk_ids,
                    "embedding": fine_centroid.tolist(),
                    "level": "fine",
                    "parent": coarse_id,
                }
                levels["fine"].append(fine_id)
                fine_sizes.append(len(fine_chunk_ids))
                # Link from parent
                clusters[coarse_id]["children"].append(fine_id)

        # Basic stats for logging and meta
        stats: Dict[str, Any] = {
            "n_chunks": len(chunk_ids),
            "n_clusters_total": len(clusters),
            "n_clusters_coarse": len(levels["coarse"]),
            "n_clusters_fine": len(levels["fine"]),
            "avg_coarse_size": float(np.mean(coarse_sizes)) if coarse_sizes else 0.0,
            "median_coarse_size": float(np.median(coarse_sizes)) if coarse_sizes else 0.0,
            "avg_fine_size": float(np.mean(fine_sizes)) if fine_sizes else 0.0,
            "median_fine_size": float(np.median(fine_sizes)) if fine_sizes else 0.0,
        }

        logger.info(
            "Built %d clusters (%d coarse, %d fine). "
            "Avg coarse size=%.1f (median=%.1f), avg fine size=%.1f (median=%.1f)",
            stats["n_clusters_total"],
            stats["n_clusters_coarse"],
            stats["n_clusters_fine"],
            stats["avg_coarse_size"],
            stats["median_coarse_size"],
            stats["avg_fine_size"],
            stats["median_fine_size"],
        )

        return clusters, levels, stats

    # ------------------------------------------------------------------
    # Summarization helpers
    # ------------------------------------------------------------------
    def _cluster_metadata_summary(self, chunk_ids: Iterable[str]) -> Tuple[str, str]:
        """Build lightweight metadata summary (books/chapters, entities) for prompts."""
        book_counter: Counter[str] = Counter()
        chapter_counter: Counter[str] = Counter()
        entity_counter: Counter[str] = Counter()

        for cid in chunk_ids:
            meta = self.graph.hierarchy.get(cid) or {}
            book_title = meta.get("book_title") or meta.get("title")
            if book_title:
                book_counter[str(book_title)] += 1
            chapter_title = meta.get("chapter_title")
            if chapter_title:
                chapter_counter[str(chapter_title)] += 1
            for eid in self.graph.chunk_entity_map.get(cid, []):
                entity_counter[str(eid)] += 1

        def _top(counter: Counter[str], limit: int = 5) -> List[str]:
            return [name for name, _ in counter.most_common(limit)]

        book_summary = ", ".join(_top(book_counter, 5)) or "Mixed sources"
        entity_summary = ", ".join(_top(entity_counter, 8))
        if not entity_summary:
            entity_summary = "various named people, organizations, and concepts"

        return book_summary, entity_summary

    def _cluster_example_snippets(self, chunk_ids: Iterable[str]) -> List[str]:
        snippets: List[str] = []
        for cid in chunk_ids:
            if len(snippets) >= self.config.summary_max_examples:
                break
            text = (self.graph.previews or {}).get(cid)
            if not text:
                text = self._build_chunk_text(cid)
            if not text:
                continue
            snippet = str(text).strip()
            if len(snippet) > self.config.summary_snippet_chars:
                snippet = snippet[: self.config.summary_snippet_chars].rstrip() + "..."
            snippets.append(snippet)
        return snippets

    def _summarize_cluster(
        self,
        cluster_id: str,
        cluster_info: Dict[str, Any],
        child_summaries: Optional[List[str]] = None,
    ) -> str:
        """Generate a short natural-language summary for a cluster."""
        chunk_ids = cluster_info.get("chunk_ids") or []
        if not chunk_ids:
            return "Cluster with no associated chunks."

        books, entities = self._cluster_metadata_summary(chunk_ids)
        snippets = self._cluster_example_snippets(chunk_ids)

        child_section = ""
        if child_summaries:
            limited_children = child_summaries[: self.config.summary_max_child_summaries]
            child_bullets = "\n".join(f"- {s}" for s in limited_children)
            child_section = (
                "\n\nBelow are summaries of sub-clusters that roll up into this topic. "
                "Use them as hints, but respond with a single, coherent summary:\n"
                f"{child_bullets}\n"
            )

        examples_section = ""
        if snippets:
            bullet_snips = "\n".join(f"{i+1}. {s}" for i, s in enumerate(snippets))
            examples_section = (
                "\n\nHere are example excerpts from this cluster:\n"
                f"{bullet_snips}\n"
            )

        prompt = (
            "You are helping to build a semantic search index over a corpus of "
            "podcast and book content about regeneration, sustainability, and related topics.\n\n"
            f"Summarize the main themes, topics, and questions covered by this cluster of chunks.\n"
            "Write a concise 2–3 sentence summary in plain English, oriented to what a user might "
            "search for. Avoid referencing 'this cluster' or 'these chunks' explicitly.\n\n"
            f"Books and sections represented: {books}\n"
            f"Frequently mentioned entities and concepts: {entities}\n"
            f"{child_section}"
            f"{examples_section}"
            "Return only the summary text."
        )

        try:
            response = self.summarizer([HumanMessage(content=prompt)])
            summary = getattr(response, "content", "").strip()
            if not summary:
                raise ValueError("Empty summary from LLM")
            return summary
        except Exception as exc:
            logger.warning(
                "Failed to summarize cluster %s (%s), falling back to heuristic summary.",
                cluster_id,
                exc,
            )
            # Heuristic fallback summary
            return (
                f"A topic cluster covering {books}, focusing on {entities}. "
                "It groups related passages from the corpus around these themes."
            )

    def _add_summaries(self, clusters: Dict[str, Dict[str, Any]], levels: Dict[str, List[str]]) -> None:
        """Attach summaries to all clusters, coarse and fine."""
        # Summarize fine clusters first (no children)
        logger.info("Summarizing %d fine clusters", len(levels.get("fine") or []))
        for cid in levels.get("fine") or []:
            clusters[cid]["summary"] = self._summarize_cluster(cid, clusters[cid])

        # Then summarize coarse clusters, optionally using child summaries
        logger.info("Summarizing %d coarse clusters", len(levels.get("coarse") or []))
        for cid in levels.get("coarse") or []:
            children = clusters[cid].get("children") or []
            child_summaries = [clusters[ch].get("summary") for ch in children if clusters.get(ch)]
            child_summaries = [s for s in child_summaries if s]
            clusters[cid]["summary"] = self._summarize_cluster(cid, clusters[cid], child_summaries)

        # Log a few sample summaries for inspection
        sample_ids = (levels.get("coarse") or [])[:3] + (levels.get("fine") or [])[:3]
        logger.info("Sample cluster summaries:")
        for sid in sample_ids:
            info = clusters.get(sid)
            if not info:
                continue
            logger.info(
                "  - %s (%s, %d chunks): %s",
                sid,
                info.get("level"),
                len(info.get("chunk_ids") or []),
                (info.get("summary") or "")[:200].replace("\n", " "),
            )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def build(self, dry_run: bool = False) -> Dict[str, Any]:
        """Build the cluster index and optionally write it to disk."""
        chunk_ids, texts = self._prepare_chunks()
        matrix = self._embed_chunks(chunk_ids, texts)
        clusters, levels, stats = self._build_clusters(chunk_ids, matrix)

        # Attach summaries
        self._add_summaries(clusters, levels)

        # Assemble final index
        cluster_index: Dict[str, Any] = {
            "levels": levels,
            "clusters": clusters,
            "meta": {
                **stats,
                "embedding_model": settings.openai_embedding_model,
                "chat_model": settings.openai_model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "params": {
                    "coarse_k": self.config.coarse_k,
                    "min_coarse": self.config.min_coarse,
                    "max_coarse": self.config.max_coarse,
                    "coarse_divisor": self.config.coarse_divisor,
                    "min_cluster_size": self.config.min_cluster_size,
                    "min_fine_per_coarse": self.config.min_fine_per_coarse,
                    "max_fine_per_coarse": self.config.max_fine_per_coarse,
                    "fine_divisor": self.config.fine_divisor,
                    "min_fine_cluster_size": self.config.min_fine_cluster_size,
                },
            },
        }

        if dry_run:
            logger.info("Dry run enabled; not writing cluster_index.json")
            return cluster_index

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(cluster_index, f, ensure_ascii=False)
        logger.info("Wrote cluster index to %s", self.output_path)
        return cluster_index


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-level semantic topic clusters for GraphRAG.",
    )
    parser.add_argument(
        "--coarse-k",
        type=int,
        default=None,
        help="Number of coarse clusters (default: auto, ~N/800 capped to [50, 100]).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=20,
        help="Minimum size of a cluster to keep at the coarse level.",
    )
    parser.add_argument(
        "--min-fine-per-coarse",
        type=int,
        default=2,
        help="Minimum fine clusters per coarse cluster (when sub-clustering).",
    )
    parser.add_argument(
        "--max-fine-per-coarse",
        type=int,
        default=8,
        help="Maximum fine clusters per coarse cluster.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run clustering and summarization but do not write cluster_index.json.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args(argv)
    config = ClusterConfig(
        coarse_k=args.coarse_k,
        min_cluster_size=args.min_cluster_size,
        min_fine_per_coarse=args.min_fine_per_coarse,
        max_fine_per_coarse=args.max_fine_per_coarse,
    )

    logger.info("Starting cluster_index build with config: %s", config)
    builder = ClusterIndexBuilder(config=config)

    try:
        builder.build(dry_run=args.dry_run)
    except Exception as exc:
        logger.error("Cluster index build failed: %s", exc, exc_info=True)
        return 1

    logger.info("Cluster index build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
