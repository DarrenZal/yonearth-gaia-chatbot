"""
Extract and display full permaculture content from Y on Earth book
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.vectorstore import YonEarthVectorStore
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def extract_permaculture_content():
    """Extract full permaculture content from Y on Earth"""

    # Initialize vectorstore
    vectorstore = YonEarthVectorStore()

    # Filter for Y on Earth book
    filter_y_on_earth = {
        "book_title": "Y on Earth: Get Smarter, Feel Better, Heal the Planet"
    }

    # Search for permaculture
    results = vectorstore.similarity_search_with_score(
        query="permaculture",
        k=50,
        filter=filter_y_on_earth
    )

    # Filter for only chunks containing permaculture
    perm_chunks = [(doc, score) for doc, score in results if 'permaculture' in doc.page_content.lower()]

    print("\n" + "="*80)
    print("FULL PERMACULTURE CONTENT FROM Y ON EARTH BOOK")
    print("="*80 + "\n")

    print(f"Found {len(perm_chunks)} chunks containing 'permaculture'\n")

    # Sort by chapter number
    perm_chunks_sorted = sorted(perm_chunks, key=lambda x: float(x[0].metadata.get('chapter_number', 0)))

    for i, (doc, score) in enumerate(perm_chunks_sorted, 1):
        metadata = doc.metadata
        print(f"\n{'='*80}")
        print(f"CHUNK {i}")
        print(f"{'='*80}")
        print(f"Chapter: {metadata.get('chapter_number')} - {metadata.get('chapter_title')}")
        print(f"Pages: {metadata.get('page_start')}-{metadata.get('page_end')}")
        print(f"Relevance Score: {score:.4f}")
        print(f"\nFULL CONTENT:")
        print("-" * 80)
        print(doc.page_content)
        print("-" * 80)

if __name__ == "__main__":
    extract_permaculture_content()
