#!/usr/bin/env python3
"""
Script to check metadata stored in Pinecone for book entries
Specifically looking at the author field format
"""
import logging
import os
import sys
from collections import defaultdict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag.pinecone_setup import PineconeManager
from src.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_book_metadata():
    """Check metadata for book entries in Pinecone"""
    
    # Initialize Pinecone manager
    manager = PineconeManager()
    index = manager.get_index()
    
    # Get index stats first
    stats = manager.get_index_stats()
    logger.info(f"Total vectors in index: {stats.get('total_vector_count', 0)}")
    
    # Query for book entries
    # We'll do a dummy query and filter by content_type
    logger.info("Querying for book entries...")
    
    try:
        # Query with a generic term to get results, filtering for books
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            filter={"content_type": "book"},
            top_k=100,  # Get up to 100 book entries
            include_metadata=True
        )
        
        if not results.matches:
            logger.info("No book entries found in the index")
            return
        
        logger.info(f"Found {len(results.matches)} book entries")
        
        # Analyze author field format
        author_formats = defaultdict(int)
        unique_authors = set()
        sample_entries = []
        
        for i, match in enumerate(results.matches):
            metadata = match.metadata
            
            # Check author field
            author = metadata.get('author', 'No author field')
            author_formats[author.startswith('Author: ')] += 1
            unique_authors.add(author)
            
            # Collect sample entries
            if i < 5:  # First 5 entries as samples
                sample_entries.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': metadata
                })
        
        # Print analysis results
        print("\n" + "="*60)
        print("BOOK METADATA ANALYSIS")
        print("="*60)
        
        print(f"\nTotal book entries found: {len(results.matches)}")
        print(f"Unique authors: {len(unique_authors)}")
        
        print("\nAuthor field format analysis:")
        print(f"  - With 'Author: ' prefix: {author_formats[True]} entries")
        print(f"  - Without 'Author: ' prefix: {author_formats[False]} entries")
        
        print("\nUnique authors found:")
        for author in sorted(unique_authors):
            print(f"  - {author}")
        
        print("\nSample book entries:")
        print("-"*60)
        
        for i, entry in enumerate(sample_entries, 1):
            print(f"\nEntry {i}:")
            print(f"  ID: {entry['id']}")
            print(f"  Score: {entry['score']}")
            print("  Metadata:")
            for key, value in entry['metadata'].items():
                if key == 'text':
                    # Truncate text content
                    text_preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"    {key}: {text_preview}")
                else:
                    print(f"    {key}: {value}")
        
        # Additional analysis - check all metadata keys
        all_keys = set()
        for match in results.matches:
            all_keys.update(match.metadata.keys())
        
        print("\nAll metadata fields found in book entries:")
        for key in sorted(all_keys):
            print(f"  - {key}")
            
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        raise


def main():
    """Main function"""
    try:
        check_book_metadata()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()