#!/usr/bin/env python3
"""
Compare search results WITH and WITHOUT graph enhancement

This will help us understand if graph results are actually improving search quality
or if they're being completely dominated by BM25/semantic signals.

Test prompts are loaded from data/search_eval_prompts.json so they can be shared
and versioned separately from this script.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.bm25_hybrid_retriever import BM25HybridRetriever
from src.rag.vectorstore import create_vectorstore
import logging

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PROMPTS_FILE = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) / "data" / "search_eval_prompts.json"


def load_test_queries():
    """
    Load test queries from the shared JSON prompt file.

    Falls back to a small built-in list if the file is missing or invalid,
    so the script remains usable during development.
    """
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
                print(f"⚠️  No prompts with 'query' field in {PROMPTS_FILE}, using built-in defaults.")
        else:
            print(f"⚠️  Prompt file not found at {PROMPTS_FILE}, using built-in defaults.")
    except Exception as e:
        print(f"⚠️  Failed to load prompts from {PROMPTS_FILE}: {e}. Using built-in defaults.")

    # Built-in fallback prompts (kept small and representative)
    return [
        "Tell me about Vandana Shiva's work on biodiversity",
        "What did Paul Hawken say about regenerative agriculture",
        "Interviews with Joel Salatin",
        "Tell me about permaculture design principles",
        "What is biochar and how is it made",
        "How does composting work",
        "Tell me about Regeneration International",
        "What is the Buckminster Fuller Institute",
]


def _format_doc_label(doc):
    """Human-friendly label for a doc including content type."""
    md = getattr(doc, "metadata", {}) or {}
    content_type = md.get("content_type", "episode")
    if content_type == "book":
        title = md.get("book_title", "unknown_book")
        chapter = md.get("chapter_title") or md.get("chapter_number")
        base = f"book:{title}"
        if chapter:
            base += f"/{chapter}"
    elif md.get("content_type") == "kg_evidence":
        base = "kg_evidence"
    else:
        ep = md.get("episode_number", "unknown")
        base = f"ep:{ep}"
    # Disambiguate unknowns a bit with chunk_id
    cid = md.get("chunk_id")
    if cid and ("unknown" in base):
        base += f"#{cid}"
    return base


def get_result_summary(docs_or_tuples):
    """Get a summary of docs keyed by formatted label"""
    labels = {}
    for item in docs_or_tuples:
        # Handle both Document objects and (Document, score) tuples
        if isinstance(item, tuple):
            doc = item[0]
        else:
            doc = item

        label = _format_doc_label(doc)
        title = doc.metadata.get("title") or doc.metadata.get("chapter_title") or doc.metadata.get("book_title") or "N/A"
        if label not in labels:
            labels[label] = title
    return labels


def get_doc_details(docs_or_tuples):
    """Get detailed metadata for docs keyed by formatted label.

    Each label maps to a list of dicts with chunk_id and content_type so we can
    understand exactly which chunks are being added by graph-only paths.
    """
    details = {}
    for item in docs_or_tuples:
        if isinstance(item, tuple):
            doc = item[0]
        else:
            doc = item
        md = getattr(doc, "metadata", {}) or {}
        label = _format_doc_label(doc)
        content_type = md.get("content_type")
        if not content_type:
            if md.get("book_title") or md.get("book_slug") or md.get("chapter_title"):
                content_type = "book"
            else:
                content_type = "episode"
        info = {
            "chunk_id": md.get("chunk_id"),
            "content_type": content_type,
            "title": md.get("title") or md.get("chapter_title") or md.get("book_title"),
        }
        details.setdefault(label, []).append(info)
    return details


def build_backbone_docs(retriever: BM25HybridRetriever, query: str, k: int = 10, category_threshold: float = 0.7):
    """
    Build a backbone fusion (BM25 + semantic + categories) with NO graph contribution.
    Mirrors the fusion logic inside BM25HybridRetriever.hybrid_search but with graph_weight=0.
    """
    keyword_results = retriever.bm25_search(query, k=20)
    semantic_results = retriever.semantic_search(query, k=20)
    category_results = retriever.category_search(query, k=20, category_threshold=category_threshold)

    if retriever.category_first_mode and category_results:
        fused = retriever.category_first_fusion(
            keyword_results,
            semantic_results,
            category_results,
            graph_results=None,
            k=60,
            graph_weight=0.0
        )
    else:
        fused = retriever.reciprocal_rank_fusion(
            keyword_results,
            semantic_results,
            graph_results=None,
            k=60,
            graph_weight=0.0
        )
    return fused[:k], keyword_results, semantic_results, category_results


def compare_searches(query, retriever, k=10):
    """Compare backbone (bm25+semantic+category) vs hybrid (adds graph) for a query.

    Returns a small stats dict so the caller can aggregate how often graph helps
    and whether backbone coverage is preserved.
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    # Backbone components
    backbone_docs, keyword_results, semantic_results, category_results = build_backbone_docs(retriever, query, k=k)
    backbone_episodes = get_result_summary(backbone_docs)
    backbone_details = get_doc_details(backbone_docs)

    print("🔍 Backbone components (no graph):")
    print("-" * 80)

    bm25_episodes = get_result_summary(keyword_results)
    print(f"📊 BM25 Search: {len(keyword_results)} results from {len(bm25_episodes)} labels")
    print(f"   Labels: {list(bm25_episodes.keys())[:10]}")

    semantic_episodes = get_result_summary(semantic_results)
    print(f"🧠 Semantic Search: {len(semantic_results)} results from {len(semantic_episodes)} labels")
    print(f"   Labels: {list(semantic_episodes.keys())[:10]}")

    category_episodes = get_result_summary(category_results)
    print(f"🏷️  Category Search: {len(category_results)} results from {len(category_episodes)} labels")
    print(f"   Labels: {list(category_episodes.keys())[:10]}")

    print(f"\n🧱 Backbone Fusion (bm25+semantic+category, no graph): {len(backbone_docs)} docs from {len(backbone_episodes)} labels")
    print(f"   Labels: {list(backbone_episodes.keys())}")

    # Graph results
    graph_results = retriever.graph_search(query, k=k)
    if graph_results:
        graph_episodes = get_result_summary(graph_results)  # get_result_summary handles tuples now
        graph_details = get_doc_details(graph_results)
        print(f"🕸️  Graph Search: {len(graph_results)} results from {len(graph_episodes)} labels")
        print(f"   Labels: {list(graph_episodes.keys())[:10]}")

        # Check if graph found unique episodes
        bm25_ep_set = set(bm25_episodes.keys())
        semantic_ep_set = set(semantic_episodes.keys())
        graph_ep_set = set(graph_episodes.keys())

        unique_to_graph = graph_ep_set - (bm25_ep_set | semantic_ep_set)
        if unique_to_graph:
            print(f"   ✨ UNIQUE episodes found by graph: {unique_to_graph}")
            # Log chunk_ids and content_type for graph-only additions
            for label in sorted(unique_to_graph):
                for info in graph_details.get(label, []):
                    print(
                        f"      - {label} :: chunk_id={info['chunk_id']} "
                        f"content_type={info['content_type']} title={info['title']}"
                    )
        else:
            print(f"   ⚠️  No unique episodes (all overlap with BM25/semantic)")
    else:
        print(f"🕸️  Graph Search: NO RESULTS (no entities or clusters matched)")
        graph_episodes = {}

    # Hybrid results (backbone + graph)
    print(f"\n🔗 Hybrid Search Results:")
    print("-" * 80)
    hybrid_results = retriever.hybrid_search(query, k=k)
    hybrid_episodes = get_result_summary(hybrid_results)
    hybrid_details = get_doc_details(hybrid_results)
    print(f"Final hybrid: {len(hybrid_results)} results from {len(hybrid_episodes)} labels")
    print(f"Labels: {list(hybrid_episodes.keys())}")

    # Backbone vs hybrid comparison
    backbone_ep_set = set(backbone_episodes.keys())
    hybrid_ep_set = set(hybrid_episodes.keys())
    missing_from_hybrid = backbone_ep_set - hybrid_ep_set
    added_by_graph = hybrid_ep_set - backbone_ep_set

    print(f"\n🧪 Backbone vs Hybrid (episode sets):")
    print(f"   Backbone episodes: {backbone_ep_set}")
    print(f"   Hybrid episodes:   {hybrid_ep_set}")
    if missing_from_hybrid:
        print(f"   ⚠️ Backbone episodes missing in hybrid: {missing_from_hybrid}")
    else:
        print(f"   ✅ Backbone coverage preserved (no drops).")
    if added_by_graph:
        print(f"   ✨ Episodes added in hybrid (graph-path additions): {added_by_graph}")
        # Log chunk_ids and content_type for graph-only additions in final hybrid
        for label in sorted(added_by_graph):
            for info in hybrid_details.get(label, []):
                print(
                    f"      - {label} :: chunk_id={info['chunk_id']} "
                    f"content_type={info['content_type']} title={info['title']}"
                )
    else:
        print(f"   ⚠️ No extra episodes added by graph in final hybrid.")

    # Check if any graph-only episodes made it to final results
    graph_helps = False
    if graph_results:
        graph_ep_set = set(graph_episodes.keys())
        hybrid_ep_set = set(hybrid_episodes.keys())
        graph_in_hybrid = graph_ep_set & hybrid_ep_set

        print(f"\n📊 Graph Contribution Analysis:")
        print(f"   Graph found: {len(graph_ep_set)} episodes")
        print(f"   Hybrid final: {len(hybrid_ep_set)} episodes")
        print(f"   Graph episodes in final: {len(graph_in_hybrid)}/{len(graph_ep_set)} ({100*len(graph_in_hybrid)/len(graph_ep_set):.1f}%)")

        if unique_to_graph:
            unique_in_final = unique_to_graph & hybrid_ep_set
            if unique_in_final:
                print(f"   ✨ UNIQUE graph episodes in final: {unique_in_final}")
                print(f"   🎯 Graph is adding NEW episodes to results!")
                graph_helps = True
            else:
                print(f"   ⚠️  Unique graph episodes NOT in final results")
        else:
            print(f"   ⚠️  Graph found no unique episodes")
    else:
        print(f"\n⚠️  No graph results to analyze")

    backbone_ok = not missing_from_hybrid
    return {
        "graph_helps": graph_helps,
        "backbone_ok": backbone_ok,
    }


def main():
    """Run comparison tests"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit number of queries to test')
    parser.add_argument(
        '--strict-backbone',
        action='store_true',
        help='Exit with non-zero status if any query drops backbone coverage.',
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("GRAPH VS NO-GRAPH COMPARISON TEST")
    print("="*80)

    test_queries = load_test_queries()
    if args.limit:
        test_queries = test_queries[:args.limit]
        print(f"⚠️  Limited to first {args.limit} queries for testing\n")

    vectorstore = create_vectorstore()
    retriever = BM25HybridRetriever(vectorstore)

    if not retriever.graph_retriever:
        print("❌ GraphRetriever not available!")
        return

    print(f"✅ GraphRetriever loaded with {len(retriever.graph_retriever.graph.entities_lexicon.get('alias_index', {}))} entity aliases")

    results = {}

    for query in test_queries:
        stats = compare_searches(query, retriever, k=10)
        results[query] = stats

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Does Graph Enhancement Help?")
    print("="*80)

    helped_count = sum(1 for r in results.values() if r["graph_helps"])
    total_count = len(results)

    print(f"\nQueries where graph added NEW episodes: {helped_count}/{total_count} ({100*helped_count/total_count:.1f}%)")

    if helped_count > 0:
        print("\n✅ Graph IS helping by finding unique episodes:")
        for query, stats in results.items():
            if stats["graph_helps"]:
                print(f"   ✓ {query[:60]}...")
    else:
        print("\n⚠️  Graph is NOT adding unique value - all graph results overlap with BM25/semantic")

    print("\n❌ Queries where graph didn't help:")
    for query, stats in results.items():
        if not stats["graph_helps"]:
            print(f"   ✗ {query[:60]}...")

    # Backbone coverage summary
    backbone_failures = [q for q, stats in results.items() if not stats["backbone_ok"]]
    if not backbone_failures:
        print("\n✅ Backbone coverage preserved for all evaluated queries.")
    else:
        print("\n⚠️  Backbone coverage dropped for the following queries:")
        for q in backbone_failures:
            print(f"   - {q[:60]}...")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if helped_count == 0:
        print("⚠️  Graph integration may not be providing value yet.")
        print("    Consider:")
        print("    1. Tuning adaptive graph weights or min/max bounds")
        print("    2. Improving entity/cluster extraction to match more queries")
        print("    3. Expanding knowledge graph coverage")
    elif helped_count < total_count * 0.3:
        print("⚠️  Graph helps occasionally but not consistently.")
        print("    Graph weight tuning may help.")
    else:
        print("✅ Graph is providing meaningful value!")
        print("   Graph-enhanced search is finding unique episodes.")

    if args.strict_backbone and backbone_failures:
        import sys
        print(
            "\n❌ Strict-backbone mode: one or more queries lost backbone coverage. "
            "Failing with non-zero exit status."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
