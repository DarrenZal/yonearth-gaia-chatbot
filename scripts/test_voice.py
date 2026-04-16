#!/usr/bin/env python3
"""
Test script for voice integration
"""
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.voice.elevenlabs_client import ElevenLabsVoiceClient
from src.config import settings


async def test_voice_generation():
    """Test voice generation functionality"""
    
    print("🔊 Testing Voice Integration")
    print("=" * 50)
    
    # Check if API key is configured
    if not settings.elevenlabs_api_key:
        print("❌ ElevenLabs API key not configured in .env file")
        return
    
    print(f"✅ API Key configured: {settings.elevenlabs_api_key[:10]}...")
    print(f"✅ Voice ID: {settings.elevenlabs_voice_id}")
    print(f"✅ Model ID: {settings.elevenlabs_model_id}")
    
    # Initialize client
    try:
        voice_client = ElevenLabsVoiceClient(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model_id
        )
        print("✅ Voice client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize voice client: {e}")
        return
    
    # Test connection
    print("\n📡 Testing API connection...")
    if voice_client.test_connection():
        print("✅ Successfully connected to ElevenLabs API")
    else:
        print("❌ Failed to connect to ElevenLabs API")
        return
    
    # Test text preprocessing
    print("\n📝 Testing text preprocessing...")
    test_text = "Hello, I am Gaia - the spirit of Earth. Welcome to this beautiful day!"
    processed = voice_client.preprocess_text_for_speech(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")
    
    # Test voice generation
    print("\n🎤 Testing voice generation...")
    try:
        audio_base64 = voice_client.generate_speech_base64(processed)
        if audio_base64:
            audio_size = len(audio_base64)
            print(f"✅ Generated audio successfully!")
            print(f"   Base64 size: {audio_size:,} characters")
            print(f"   Estimated audio size: ~{audio_size * 3 // 4 // 1024} KB")
            
            # Save sample audio for manual testing
            import base64
            audio_bytes = base64.b64decode(audio_base64)
            output_path = Path("/root/yonearth-gaia-chatbot/data/test_voice.mp3")
            output_path.write_bytes(audio_bytes)
            print(f"✅ Saved test audio to: {output_path}")
        else:
            print("❌ Failed to generate audio")
    except Exception as e:
        print(f"❌ Voice generation error: {e}")
    
    # Test streaming
    print("\n📊 Testing text splitting for streaming...")
    long_text = """
    Dear one, let me share with you the wisdom of regenerative agriculture. 
    It is a practice that goes beyond sustainability, actively healing the Earth 
    while producing abundant food. Through techniques like cover cropping, 
    composting, and holistic grazing, we can rebuild soil health, increase 
    biodiversity, and sequester carbon from the atmosphere. This is not just 
    farming; it's a partnership with nature, a dance with the seasons, and 
    a promise to future generations.
    """
    chunks = voice_client.split_text_for_streaming(long_text.strip())
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk)} chars - {chunk[:50]}...")
    
    print("\n✨ Voice integration test complete!")


if __name__ == "__main__":
    asyncio.run(test_voice_generation())