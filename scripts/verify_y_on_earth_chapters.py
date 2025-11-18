"""
Verify specific chapter references for Y on Earth permaculture content
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.vectorstore import YonEarthVectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_chapter_references():
    """Verify the specific chapters cited for permaculture"""

    # Initialize vectorstore
    vectorstore = YonEarthVectorStore()

    # Filter for Y on Earth book
    filter_y_on_earth = {
        "book_title": "Y on Earth: Get Smarter, Feel Better, Heal the Planet"
    }

    print("\n" + "="*80)
    print("VERIFYING SPECIFIC CHAPTER REFERENCES")
    print("="*80 + "\n")

    # Search for all Y on Earth book chunks
    all_results = vectorstore.similarity_search_with_score(
        query="permaculture",
        k=100,  # Get many results
        filter=filter_y_on_earth
    )

    print(f"Total Y on Earth chunks retrieved: {len(all_results)}\n")

    # Filter for specific chapters
    part_four_chunks = []
    heal_chunks = []
    permaculture_chunks = []

    for doc, score in all_results:
        metadata = doc.metadata
        chapter_title = metadata.get('chapter_title', '')
        content = doc.page_content.lower()

        # Check if it's Part Four
        if 'part four' in chapter_title.lower() or 'creating the culture' in chapter_title.lower():
            part_four_chunks.append((doc, score))

        # Check if it's a Heal chapter
        if 'heal' in chapter_title.lower():
            heal_chunks.append((doc, score))

        # Check if contains permaculture
        if 'permaculture' in content:
            permaculture_chunks.append((doc, score))

    print(f"\n{'='*80}")
    print(f"PART FOUR CHAPTERS (Creating The Culture & Future We Really Want)")
    print(f"{'='*80}")
    print(f"Found {len(part_four_chunks)} chunks from Part Four\n")

    for i, (doc, score) in enumerate(part_four_chunks[:10], 1):
        metadata = doc.metadata
        print(f"{i}. Chapter {metadata.get('chapter_number')}: {metadata.get('chapter_title')}")
        print(f"   Pages: {metadata.get('page_start')}-{metadata.get('page_end')}")
        print(f"   Score: {score:.4f}")
        if 'permaculture' in doc.page_content.lower():
            print(f"   ✅ CONTAINS PERMACULTURE")
            # Show context
            content_lower = doc.page_content.lower()
            perm_index = content_lower.find("permaculture")
            start = max(0, perm_index - 150)
            end = min(len(doc.page_content), perm_index + 150)
            context = doc.page_content[start:end]
            print(f"   Context: ...{context}...")
        print()

    print(f"\n{'='*80}")
    print(f"'HEAL' CHAPTERS")
    print(f"{'='*80}")
    print(f"Found {len(heal_chunks)} chunks from Heal chapters\n")

    for i, (doc, score) in enumerate(heal_chunks[:10], 1):
        metadata = doc.metadata
        print(f"{i}. Chapter {metadata.get('chapter_number')}: {metadata.get('chapter_title')}")
        print(f"   Pages: {metadata.get('page_start')}-{metadata.get('page_end')}")
        print(f"   Score: {score:.4f}")
        if 'permaculture' in doc.page_content.lower():
            print(f"   ✅ CONTAINS PERMACULTURE")
            # Show context
            content_lower = doc.page_content.lower()
            perm_index = content_lower.find("permaculture")
            start = max(0, perm_index - 150)
            end = min(len(doc.page_content), perm_index + 150)
            context = doc.page_content[start:end]
            print(f"   Context: ...{context}...")
        print()

    print(f"\n{'='*80}")
    print(f"ALL CHUNKS CONTAINING 'PERMACULTURE'")
    print(f"{'='*80}")
    print(f"Found {len(permaculture_chunks)} chunks containing 'permaculture'\n")

    # Group by chapter
    chapters_with_perm = {}
    for doc, score in permaculture_chunks:
        metadata = doc.metadata
        chapter_num = metadata.get('chapter_number')
        chapter_title = metadata.get('chapter_title')
        key = f"{chapter_num} - {chapter_title}"

        if key not in chapters_with_perm:
            chapters_with_perm[key] = {
                'count': 0,
                'pages': set(),
                'metadata': metadata,
                'best_score': score
            }

        chapters_with_perm[key]['count'] += 1
        chapters_with_perm[key]['pages'].add(f"{metadata.get('page_start')}-{metadata.get('page_end')}")
        chapters_with_perm[key]['best_score'] = max(chapters_with_perm[key]['best_score'], score)

    # Sort by best score
    sorted_chapters = sorted(chapters_with_perm.items(), key=lambda x: x[1]['best_score'], reverse=True)

    for i, (chapter_key, info) in enumerate(sorted_chapters, 1):
        print(f"{i}. {chapter_key}")
        print(f"   Chunks with 'permaculture': {info['count']}")
        print(f"   Pages: {', '.join(sorted(info['pages']))}")
        print(f"   Best relevance score: {info['best_score']:.4f}")
        print()

if __name__ == "__main__":
    verify_chapter_references()
