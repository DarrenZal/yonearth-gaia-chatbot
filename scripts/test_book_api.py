#!/usr/bin/env python3
"""
Test script to check book formatting in API responses
"""

import requests
import json

def test_book_query():
    """Test a query that should return book results"""
    
    url = "http://152.53.194.214/api/bm25/chat"
    
    payload = {
        "message": "what is viriditas?",
        "search_method": "hybrid",
        "personality": "warm_mother"
    }
    
    print("Testing book query...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("=" * 80)
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\nResponse structure:")
            print(f"- response: {data.get('response', 'N/A')[:100]}...")
            print(f"- sources: {len(data.get('sources', []))} sources")
            
            print("\nSources analysis:")
            for i, source in enumerate(data.get('sources', [])[:3], 1):
                print(f"\nSource {i}:")
                print(f"  content_type: {source.get('content_type', 'N/A')}")
                print(f"  title: {source.get('title', 'N/A')}")
                print(f"  guest_name: {source.get('guest_name', 'N/A')}")
                print(f"  episode_number: {source.get('episode_number', 'N/A')}")
                
                if source.get('content_type') == 'book':
                    print(f"  book_title: {source.get('book_title', 'N/A')}")
                    print(f"  author: {source.get('author', 'N/A')}")
                    print(f"  chapter_number: {source.get('chapter_number', 'N/A')}")
                    print(f"  chapter_title: {source.get('chapter_title', 'N/A')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_book_query()