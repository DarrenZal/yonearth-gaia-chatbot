#!/usr/bin/env python3
"""
Test voice integration through the API
"""
import requests
import json
import base64
from pathlib import Path

# API configuration
API_URL = "http://localhost:80"

def test_voice_api():
    """Test voice generation through the chat API"""
    
    print("🎤 Testing Voice API Integration")
    print("=" * 50)
    
    # Test payload with voice enabled
    payload = {
        "message": "Tell me about the importance of soil health in just two sentences.",
        "enable_voice": True,
        "personality": "warm_mother",
        "max_results": 3
    }
    
    print("📤 Sending request to /api/chat with voice enabled...")
    print(f"   Message: {payload['message']}")
    
    try:
        response = requests.post(f"{API_URL}/api/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ Response received successfully!")
            print(f"📝 Text response: {data['response'][:100]}...")
            print(f"📚 Citations: {len(data.get('citations', []))} episodes referenced")
            
            if data.get('audio_data'):
                print("\n🔊 Voice data received!")
                audio_size = len(data['audio_data'])
                print(f"   Base64 size: {audio_size:,} characters")
                print(f"   Estimated audio size: ~{audio_size * 3 // 4 // 1024} KB")
                
                # Save the audio for testing
                audio_bytes = base64.b64decode(data['audio_data'])
                output_path = Path("/root/yonearth-gaia-chatbot/data/api_test_voice.mp3")
                output_path.write_bytes(audio_bytes)
                print(f"✅ Saved API voice response to: {output_path}")
            else:
                print("\n❌ No voice data in response")
        else:
            print(f"\n❌ API request failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Make sure the server is running:")
        print("   python3 scripts/start_local.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    # Test BM25 endpoint
    print("\n" + "=" * 50)
    print("📤 Testing BM25 endpoint with voice...")
    
    bm25_payload = {
        "message": "What episodes discuss composting?",
        "enable_voice": True,
        "search_method": "hybrid",
        "k": 5,
        "gaia_personality": "wise_guide"
    }
    
    try:
        response = requests.post(f"{API_URL}/api/bm25/chat", json=bm25_payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ BM25 response received!")
            print(f"🔍 Search method used: {data.get('search_method_used', 'Unknown')}")
            print(f"📊 Documents retrieved: {data.get('documents_retrieved', 0)}")
            
            if data.get('audio_data'):
                print("🔊 Voice data included in BM25 response!")
            else:
                print("❌ No voice data in BM25 response")
        else:
            print(f"❌ BM25 request failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ BM25 test error: {e}")
    
    print("\n✨ Voice API test complete!")


if __name__ == "__main__":
    test_voice_api()