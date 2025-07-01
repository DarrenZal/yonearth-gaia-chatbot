#!/usr/bin/env python3
"""
Quick test script for the API endpoints
"""
import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing YonEarth Gaia Chatbot API")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   RAG Initialized: {data.get('rag_initialized')}")
            print(f"   Vector Count: {data.get('vectorstore_stats', {}).get('total_vector_count', 0)}")
            print("   ✅ Health check passed")
        else:
            print(f"   ❌ Health check failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    print()
    
    # Test chat endpoint
    print("2. Testing chat endpoint...")
    try:
        chat_data = {
            "message": "Tell me about regenerative agriculture",
            "session_id": "test-session"
        }
        response = requests.post(f"{base_url}/chat", json=chat_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response length: {len(data.get('response', ''))}")
            print(f"   Citations: {len(data.get('citations', []))}")
            print(f"   Retrieved docs: {data.get('retrieval_count', 0)}")
            print("   ✅ Chat endpoint working")
            
            # Print first 100 chars of response
            response_text = data.get('response', '')
            if response_text:
                print(f"   Preview: {response_text[:100]}...")
        else:
            print(f"   ❌ Chat failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Chat error: {e}")
    
    print()
    
    # Test recommendations endpoint
    print("3. Testing recommendations endpoint...")
    try:
        rec_data = {"query": "soil health"}
        response = requests.post(f"{base_url}/recommendations", json=rec_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Recommendations: {len(data.get('recommendations', []))}")
            print("   ✅ Recommendations working")
        else:
            print(f"   ❌ Recommendations failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Recommendations error: {e}")
    
    print("\n🎉 API testing complete!")

if __name__ == "__main__":
    test_api()