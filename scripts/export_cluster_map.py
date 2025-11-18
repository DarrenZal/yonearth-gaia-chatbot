#!/usr/bin/env python3
"""
Export a lightweight 3D cluster map for visualization.

This script reads:
  - data/graph_index/cluster_index.json
  - data/graph_index/hierarchy.json

It then:
  - Collects all coarse and fine clusters, along with their centroids
  - Projects cluster embeddings into 3D space using PCA
  - Derives simple metadata per cluster:
      * size (number of member chunks)
      * content_type: "book" | "episode" | "mixed" | "unknown"
      * top_books: most frequent book titles in the cluster
      * level, parent, children, summary (from cluster_index)

Output:
  - data/graph_index/cluster_map_3d.json

This file is intended for use by a web-based 3D viewer (e.g. Three.js +
3d-force-graph) to provide a semantic "zoomable" view over topic clusters.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA

# Add project root to path so we can import src.*
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings  # type: ignore
from src.rag.graphrag_index import GraphIndex  # type: ignore


logger = logging.getLogger(__name__)


def _infer_chunk_content_type(meta: Dict[str, Any]) -> str:
    """Best-effort content_type inference for a single chunk."""
    ct = (meta.get("content_type") or "").strip().lower()
    if ct in {"book", "episode"}:
        return ct

    # Heuristic based on metadata keys, mirroring other scripts
    if meta.get("book_title") or meta.get("book_slug") or meta.get("chapter_title"):
        return "book"
    if meta.get("episode_number") is not None or meta.get("title"):
        return "episode"
    return "unknown"


def _derive_cluster_metadata(
    chunk_ids: List[str],
    hierarchy: Dict[str, Any],
) -> Tuple[int, str, List[str]]:
    """Compute size, dominant content_type, and top book titles for a cluster."""
    size = len(chunk_ids)
    if size == 0:
        return 0, "unknown", []

    content_counts: Counter[str] = Counter()
    book_counts: Counter[str] = Counter()

    for cid in chunk_ids:
        meta = hierarchy.get(cid) or {}
        ct = _infer_chunk_content_type(meta)
        content_counts[ct] += 1

        bt = meta.get("book_title")
        if bt:
            book_counts[str(bt)] += 1

    # Determine aggregate content_type
    books = content_counts["book"]
    episodes = content_counts["episode"]
    if books and episodes:
        content_type = "mixed"
    elif books:
        content_type = "book"
    elif episodes:
        content_type = "episode"
    else:
        content_type = "unknown"

    top_books = [title for title, _ in book_counts.most_common(3)]
    return size, content_type, top_books


def export_cluster_map(output_path: Path | None = None) -> Path:
    """Export a 3D cluster map JSON for visualization."""
    # Load core graph index (reuses existing loader)
    graph = GraphIndex.load()
    base_dir = settings.data_dir / "graph_index"
    cluster_index_path = base_dir / "cluster_index.json"
    if not cluster_index_path.exists():
        raise FileNotFoundError(f"cluster_index.json not found at {cluster_index_path}")

    with open(cluster_index_path, "r", encoding="utf-8") as f:
        cluster_index: Dict[str, Any] = json.load(f)

    clusters = (cluster_index or {}).get("clusters") or {}
    if not clusters:
        raise ValueError("cluster_index.json has no 'clusters' entry.")

    # Collect clusters with valid embeddings
    cluster_ids: List[str] = []
    vectors: List[List[float]] = []
    for cid, info in clusters.items():
        emb = info.get("embedding")
        if isinstance(emb, list) and emb:
            try:
                vec = [float(x) for x in emb]
            except Exception:
                continue
            cluster_ids.append(cid)
            vectors.append(vec)

    if not cluster_ids or not vectors:
        raise ValueError("No clusters with valid embeddings found in cluster_index.json.")

    matrix = np.asarray(vectors, dtype=np.float32)
    logger.info("Loaded %d clusters with embedding dim %d", matrix.shape[0], matrix.shape[1])

    # Project embeddings to 3D using PCA
    logger.info("Running PCA projection to 3D for cluster embeddings")
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(matrix)

    # Derive per-cluster metadata and assemble nodes
    nodes: Dict[str, Dict[str, Any]] = {}
    for idx, cid in enumerate(cluster_ids):
        info = clusters.get(cid, {})
        chunk_ids = info.get("chunk_ids") or []

        size, content_type, top_books = _derive_cluster_metadata(chunk_ids, graph.hierarchy)

        node = {
            "id": cid,
            "level": info.get("level"),
            "parent": info.get("parent"),
            "children": info.get("children") or [],
            "position": [float(x) for x in coords[idx]],
            "size": size,
            "content_type": content_type,
            "top_books": top_books,
            "summary": info.get("summary") or "",
        }
        nodes[cid] = node

    # Basic projection metadata
    mins = coords.min(axis=0).tolist()
    maxs = coords.max(axis=0).tolist()

    output: Dict[str, Any] = {
        "levels": cluster_index.get("levels") or {},
        "nodes": nodes,
        "meta": {
            "projection": "pca_3d",
            "cluster_count": len(cluster_ids),
            "embedding_dim": int(matrix.shape[1]),
            "coord_min": mins,
            "coord_max": maxs,
            "source_cluster_index_meta": cluster_index.get("meta") or {},
        },
    }

    out_path = output_path or (base_dir / "cluster_map_3d.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    logger.info("Wrote cluster map to %s", out_path)
    return out_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        export_cluster_map()
    except Exception as exc:
        logger.error("Cluster map export failed: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

