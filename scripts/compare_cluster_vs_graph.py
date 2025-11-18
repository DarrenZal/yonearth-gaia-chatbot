#!/usr/bin/env python3
"""
Small diagnostic utility to compare cluster-based retrieval vs. full GraphRAG
retrieval on the shared prompt set in data/search_eval_prompts.json.

This is intended for offline evaluation and tuning. It relies on the optional
cluster_index.json file under data/graph_index when present.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings  # type: ignore
from src.rag.graph_retriever import GraphRetriever  # type: ignore


PROMPTS_FILE = (
    Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    / "data"
    / "search_eval_prompts.json"
)


def load_test_queries() -> list[str]:
    """Load test queries from the shared JSON prompt file."""
    try:
        if PROMPTS_FILE.exists():
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            prompts = data.get("prompts", [])
            queries = [p.get("query") for p in prompts if p.get("query")]
            if queries:
                print(f"✅ Loaded {len(queries)} test queries from {PROMPTS_FILE}")
                return queries
            else:
                print(
                    f"⚠️  No prompts with 'query' field in {PROMPTS_FILE}, "
                    "nothing to evaluate."
                )
        else:
            print(f"⚠️  Prompt file not found at {PROMPTS_FILE}, nothing to evaluate.")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"⚠️  Failed to load prompts from {PROMPTS_FILE}: {exc}.")

    return []


def _format_label(doc) -> str:
    """Human-friendly label for a doc, reusing the episode/book heuristics."""
    md = getattr(doc, "metadata", {}) or {}
    content_type = md.get("content_type")
    if not content_type:
        if md.get("book_title") or md.get("book_slug") or md.get("chapter_title"):
            content_type = "book"
        else:
            content_type = "episode"

    if content_type == "book":
        title = md.get("book_title", "unknown_book")
        chapter = md.get("chapter_title") or md.get("chapter_number")
        base = f"book:{title}"
        if chapter:
            base += f"/{chapter}"
    else:
        ep = md.get("episode_number", "unknown")
        base = f"ep:{ep}"

    cid = md.get("chunk_id")
    if cid and "unknown" in base:
        base += f"#{cid}"
    return base


def _summarize_docs(docs) -> dict[str, list[str]]:
    """Map label -> list of chunk_ids for quick comparison."""
    out: dict[str, list[str]] = {}
    for doc in docs:
        md = getattr(doc, "metadata", {}) or {}
        label = _format_label(doc)
        cid = md.get("chunk_id") or "n/a"
        out.setdefault(label, []).append(str(cid))
    return out


def compare_cluster_vs_graph(gr: GraphRetriever, query: str, k: int = 10) -> None:
    """Print a side-by-side comparison for a single query."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80 + "\n")

    # Full graph retrieval (entities + multi-hop + optional clusters)
    full = gr.retrieve(query, k=k)
    full_docs = full.chunks

    # Cluster-only retrieval
    cluster_docs = gr.cluster_search(query, k=k)

    print(
        f"Graph entities matched: {len(full.matched_entities)} "
        f"{full.matched_entities[:5]}"
    )
    print(
        f"Graph clusters matched: {len(full.matched_clusters)} "
        f"{[c.get('cluster_id') for c in full.matched_clusters[:5]]}"
    )
    print(f"\nFull graph docs (k={k}): {len(full_docs)}")
    print(f"Cluster-only docs (k={k}): {len(cluster_docs)}")

    full_summary = _summarize_docs(full_docs)
    cluster_summary = _summarize_docs(cluster_docs)

    full_labels = set(full_summary.keys())
    cluster_labels = set(cluster_summary.keys())

    print("\nLabels (full graph):")
    print(f"  {sorted(full_labels)}")
    print("Labels (cluster-only):")
    print(f"  {sorted(cluster_labels)}")

    only_graph = sorted(full_labels - cluster_labels)
    only_cluster = sorted(cluster_labels - full_labels)
    overlap = sorted(full_labels & cluster_labels)

    print("\nOverlap / differences:")
    print(f"  Overlap labels: {overlap}")
    print(f"  Only full graph: {only_graph}")
    print(f"  Only cluster-only: {only_cluster}")


def main() -> None:
    """Run cluster-vs-graph comparison over the prompt set."""
    # For this diagnostic we explicitly enable cluster retrieval so that
    # GraphRetriever.cluster_search will exercise the cluster index when present.
    settings.graph_enable_cluster_retrieval = True  # type: ignore[attr-defined]

    queries = load_test_queries()
    if not queries:
        return

    gr = GraphRetriever()
    print(
        f"\n✅ GraphRetriever ready with "
        f"{len(gr.graph.entities_lexicon.get('alias_index', {}))} entity aliases "
        f"and {len((gr.graph.cluster_index or {}).get('clusters', {}))} clusters"
    )

    for q in queries:
        compare_cluster_vs_graph(gr, q, k=10)


if __name__ == "__main__":
    main()

