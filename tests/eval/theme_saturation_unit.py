"""
Offline unit check for theme_saturation_search — runs without OpenAI/Pinecone.

Builds a BM25HybridRetriever-like object with a stub vectorstore and a small
synthetic document set, then verifies that theme-tagged episodes get pulled
even when their lexical relevance to the query is weak.

Use this when local OpenAI quota is exhausted but we still want fast feedback
on the retrieval composition logic.

Run:
    python -m tests.eval.theme_saturation_unit
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from langchain.schema import Document  # noqa: E402

from src.rag.bm25_hybrid_retriever import BM25HybridRetriever  # noqa: E402
from src.rag.episode_categorizer import EpisodeCategorizer  # noqa: E402


class _StubVectorstore:
    """No-op stand-in for YonEarthVectorStore — we never call its methods here."""
    def similarity_search_with_score(self, *args, **kwargs):
        return []


def _make_synthetic_documents(categorizer: EpisodeCategorizer):
    """Two chunks per IMPACT INVESTING episode + one chunk for each book."""
    docs = []
    ii_episodes = sorted(categorizer.get_episodes_by_category("IMPACT INVESTING"))
    for ep_id in ii_episodes:
        for i in range(2):
            docs.append(Document(
                page_content=f"Episode {ep_id} chunk {i}: exploring impact investing principles.",
                metadata={
                    "content_type": "episode",
                    "episode_id": str(ep_id),
                    "episode_number": str(ep_id),
                    "title": f"Episode {ep_id}",
                    "chunk_id": f"ep_{ep_id}_chunk_{i}",
                },
            ))
    # And a few non-finance episodes that won't be theme-tagged
    for ep_id in [1, 2, 3]:
        docs.append(Document(
            page_content=f"Episode {ep_id} chunk: unrelated content.",
            metadata={
                "content_type": "episode",
                "episode_id": str(ep_id),
                "episode_number": str(ep_id),
                "title": f"Episode {ep_id}",
                "chunk_id": f"ep_{ep_id}_chunk_0",
            },
        ))
    # Books
    for title in categorizer.books.keys():
        docs.append(Document(
            page_content=f"Book chunk from {title}: regenerative practices.",
            metadata={
                "content_type": "book",
                "book_title": title,
                "chapter_number": 1,
                "chapter_title": "Intro",
                "author": "Aaron William Perry",
                "chunk_id": f"book_{title}_ch1",
            },
        ))
    return docs


def main():
    print("Loading categorizer (with books)...")
    categorizer = EpisodeCategorizer()
    print(f"  {len(categorizer.episodes)} episodes, {len(categorizer.books)} books")
    print(f"  IMPACT INVESTING episodes: {sorted(categorizer.get_episodes_by_category('IMPACT INVESTING'))}")
    print(f"  SOIL books: {categorizer.get_books_by_category('SOIL')}")
    print()

    print("Building retriever with stub vectorstore (no Pinecone)...")
    retriever = BM25HybridRetriever(
        _StubVectorstore(),
        use_reranker=False,
        category_first_mode=False,
    )
    # Override loaded docs and rebuild BM25 index against the synthetic set
    docs = _make_synthetic_documents(categorizer)
    retriever.documents = docs
    retriever.tokenized_docs = [retriever._tokenize_document(d.page_content) for d in docs]
    from rank_bm25 import BM25Okapi
    retriever.bm25 = BM25Okapi(retriever.tokenized_docs)
    print(f"  Index size: {len(docs)} chunks")
    print()

    print("--- Test: 'finance' should pull all IMPACT INVESTING episodes ---")
    out = retriever.theme_saturation_search("finance", category_threshold=0.55, max_chunks=30)
    cited_ep_ids = sorted({
        int(d.metadata.get("episode_number", 0))
        for d in out
        if d.metadata.get("content_type") == "episode"
    })
    expected = sorted(categorizer.get_episodes_by_category("IMPACT INVESTING"))
    print(f"  episodes cited: {cited_ep_ids}")
    print(f"  expected     : {expected}")
    assert cited_ep_ids == expected, f"FAIL: missing {set(expected) - set(cited_ep_ids)}"
    print("  PASS — every IMPACT INVESTING episode contributed a chunk")
    print()

    print("--- Test: 'biochar finance' should pull BIOCHAR + IMPACT INVESTING + Soil book ---")
    out = retriever.theme_saturation_search("biochar finance", category_threshold=0.55, max_chunks=40)
    cited_books = {d.metadata.get("book_title") for d in out if d.metadata.get("content_type") == "book"}
    cited_eps = {int(d.metadata.get("episode_number", 0)) for d in out if d.metadata.get("content_type") == "episode"}
    print(f"  episodes cited: {sorted(cited_eps)}")
    print(f"  books cited   : {cited_books}")
    assert "Soil Stewardship Handbook" in cited_books, "FAIL: Soil Stewardship Handbook not cited (expected via SOIL/BIOCHAR theme)"
    assert any(ep in cited_eps for ep in expected), "FAIL: no IMPACT INVESTING episodes"
    print("  PASS — Soil Stewardship Handbook + IMPACT INVESTING episodes present")
    print()

    print("--- Test: 'random unrelated query' should produce no theme matches ---")
    out = retriever.theme_saturation_search("xyzzy plugh", category_threshold=0.55, max_chunks=30)
    print(f"  theme-saturation chunks for unrelated query: {len(out)}")
    assert len(out) == 0, f"FAIL: expected empty, got {len(out)}"
    print("  PASS — no spurious theme matches")
    print()

    print("All theme-saturation unit checks passed.")


if __name__ == "__main__":
    main()
