#!/usr/bin/env python3
"""
Script to inspect raw Pinecone metadata for book entries
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinecone import Pinecone
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def inspect_book_entries():
    """Query Pinecone for book entries and display raw metadata"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("yonearth-episodes")
    
    print("Inspecting Pinecone book entries...")
    print("=" * 80)
    
    # Query for book entries using filter
    try:
        # First, get some book entries
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            top_k=10,
            include_metadata=True,
            filter={"content_type": "book"}
        )
        
        print(f"\nFound {len(results['matches'])} book entries (showing first 10)")
        print("-" * 80)
        
        for i, match in enumerate(results['matches'][:5], 1):
            print(f"\nEntry {i}:")
            print(f"ID: {match['id']}")
            print(f"Score: {match['score']}")
            print("\nRaw Metadata:")
            
            # Pretty print the metadata
            metadata = match.get('metadata', {})
            for key, value in metadata.items():
                if key == 'text':
                    # Truncate text for readability
                    text_preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"  {key}: {text_preview}")
                else:
                    print(f"  {key}: {value}")
            
            print("-" * 40)
        
        # Now let's search for a specific text to see the formatting
        print("\n" + "=" * 80)
        print("Searching for 'VIRIDITAS' in book content...")
        print("=" * 80)
        
        # Create a simple embedding (this won't be accurate but will return results)
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="VIRIDITAS healing"
        )
        query_embedding = response.data[0].embedding
        
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"content_type": "book"}
        )
        
        print(f"\nFound {len(results['matches'])} matches for 'VIRIDITAS'")
        print("-" * 80)
        
        for i, match in enumerate(results['matches'][:3], 1):
            print(f"\nMatch {i}:")
            print(f"ID: {match['id']}")
            print(f"Score: {match['score']}")
            
            metadata = match.get('metadata', {})
            print(f"\nFormatted fields:")
            print(f"  title: {metadata.get('title', 'N/A')}")
            print(f"  guest_name: {metadata.get('guest_name', 'N/A')}")
            print(f"  content_type: {metadata.get('content_type', 'N/A')}")
            print(f"  chapter_title: {metadata.get('chapter_title', 'N/A')}")
            print(f"  author: {metadata.get('author', 'N/A')}")
            
            # Show text preview
            text = metadata.get('text', '')
            text_preview = text[:150] + "..." if len(text) > 150 else text
            print(f"\nText preview: {text_preview}")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        import traceback
        traceback.print_exc()
    
    # Also check the index stats
    try:
        print("\n" + "=" * 80)
        print("Index Statistics:")
        print("=" * 80)
        
        stats = index.describe_index_stats()
        print(json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"Error getting index stats: {e}")

if __name__ == "__main__":
    inspect_book_entries()