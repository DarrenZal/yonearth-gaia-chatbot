"""
Search Y on Earth book in Pinecone for permaculture content
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.vectorstore import YonEarthVectorStore
from src.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_y_on_earth_permaculture():
    """Search for permaculture content in Y on Earth book"""

    # Initialize vectorstore
    vectorstore = YonEarthVectorStore()

    # Search with filter for Y on Earth book only
    filter_y_on_earth = {
        "book_title": "Y on Earth: Get Smarter, Feel Better, Heal the Planet"
    }

    print("\n" + "="*80)
    print("SEARCHING Y ON EARTH BOOK FOR 'PERMACULTURE'")
    print("="*80 + "\n")

    # Search for permaculture
    results = vectorstore.similarity_search_with_score(
        query="permaculture sustainable design principles agriculture",
        k=20,  # Get more results to see what's available
        filter=filter_y_on_earth
    )

    print(f"Found {len(results)} results\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"RESULT {i} - Relevance Score: {score:.4f}")
        print(f"{'='*80}")

        metadata = doc.metadata
        print(f"\nBook: {metadata.get('book_title', 'N/A')}")
        print(f"Chapter: {metadata.get('chapter_number', 'N/A')} - {metadata.get('chapter_title', 'N/A')}")
        print(f"Pages: {metadata.get('page_start', 'N/A')}-{metadata.get('page_end', 'N/A')}")
        print(f"Content Type: {metadata.get('content_type', 'N/A')}")
        print(f"Chunk: {metadata.get('chunk_index', 'N/A')} of {metadata.get('chunk_total', 'N/A')}")

        print(f"\nContent Preview ({len(doc.page_content)} chars):")
        print("-" * 80)
        # Show first 500 characters
        preview = doc.page_content[:500]
        print(preview)
        if len(doc.page_content) > 500:
            print(f"... ({len(doc.page_content) - 500} more characters)")
        print("-" * 80)

        # Check if "permaculture" appears in the content
        if "permaculture" in doc.page_content.lower():
            print("\n🎯 CONTAINS 'PERMACULTURE'!")
            # Show context around the word
            content_lower = doc.page_content.lower()
            perm_index = content_lower.find("permaculture")
            start = max(0, perm_index - 100)
            end = min(len(doc.page_content), perm_index + 200)
            context = doc.page_content[start:end]
            print("\nContext around 'permaculture':")
            print(f"...{context}...")

    # Also search without book filter to see if there are any results at all
    print("\n\n" + "="*80)
    print("SEARCHING ALL CONTENT FOR 'PERMACULTURE' (NO BOOK FILTER)")
    print("="*80 + "\n")

    all_results = vectorstore.similarity_search_with_score(
        query="permaculture sustainable design principles agriculture",
        k=10
    )

    print(f"Found {len(all_results)} results across all content\n")

    for i, (doc, score) in enumerate(all_results, 1):
        metadata = doc.metadata
        content_type = metadata.get('content_type', 'episode')

        if content_type == 'book':
            book_title = metadata.get('book_title', 'Unknown')
            print(f"{i}. [{content_type.upper()}] {book_title} - Score: {score:.4f}")
            print(f"   Chapter: {metadata.get('chapter_number', 'N/A')} - {metadata.get('chapter_title', 'N/A')}")
        else:
            episode_num = metadata.get('episode_number', 'N/A')
            title = metadata.get('title', 'Unknown')
            print(f"{i}. [EPISODE {episode_num}] {title} - Score: {score:.4f}")

        # Check for actual word
        if "permaculture" in doc.page_content.lower():
            print(f"   🎯 Contains 'permaculture'")

        print()

if __name__ == "__main__":
    search_y_on_earth_permaculture()
