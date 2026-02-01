"""
ElevenLabs Text-to-Speech client for Gaia voice generation
"""
import os
import base64
import logging
from typing import Optional, Dict, Any
from io import BytesIO

from elevenlabs import ElevenLabs, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs as ElevenLabsClient

logger = logging.getLogger(__name__)


class ElevenLabsVoiceClient:
    """Client for ElevenLabs text-to-speech API"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2"
    ):
        """
        Initialize ElevenLabs client
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use for generation
            model_id: Model ID for voice generation
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "YcVr5DmTjJ2cEVwNiuhU")
        self.model_id = model_id or os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
        
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        # Initialize client
        self.client = ElevenLabs(api_key=self.api_key)
        logger.info(f"ElevenLabs client initialized with voice ID: {self.voice_id}")
    
    def generate_speech(
        self, 
        text: str,
        voice_settings: Optional[Dict[str, float]] = None,
        output_format: str = "mp3_44100_128"
    ) -> Optional[bytes]:
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            voice_settings: Voice settings (stability, similarity_boost, etc.)
            output_format: Audio format (default: mp3_44100_128)
            
        Returns:
            Audio data as bytes or None if generation fails
        """
        try:
            # Default voice settings optimized for Gaia
            if voice_settings is None:
                voice_settings = {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            
            # Generate audio
            logger.debug(f"Generating speech for text length: {len(text)}")
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=output_format,
                voice_settings=VoiceSettings(**voice_settings)
            )
            
            # Convert generator to bytes
            audio_bytes = BytesIO()
            for chunk in audio:
                audio_bytes.write(chunk)
            
            audio_data = audio_bytes.getvalue()
            logger.info(f"Generated audio: {len(audio_data)} bytes")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None
    
    def generate_speech_base64(
        self, 
        text: str,
        voice_settings: Optional[Dict[str, float]] = None,
        output_format: str = "mp3_44100_128"
    ) -> Optional[str]:
        """
        Generate speech and return as base64 encoded string
        
        Args:
            text: Text to convert to speech
            voice_settings: Voice settings
            output_format: Audio format
            
        Returns:
            Base64 encoded audio string or None if generation fails
        """
        audio_data = self.generate_speech(text, voice_settings, output_format)
        
        if audio_data:
            return base64.b64encode(audio_data).decode('utf-8')
        
        return None
    
    def preprocess_text_for_speech(self, text: str) -> str:
        """
        Preprocess text for better speech generation

        Args:
            text: Original text

        Returns:
            Preprocessed text
        """
        import re

        # Pronunciation fixes
        text = re.sub(r'yonearth', 'Y on Earth', text, flags=re.IGNORECASE)

        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'#{1,6}\s*(.*?)\n', r'\1. ', text)  # Headers
        
        # Remove citations like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove excessive line breaks
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        
        # Add pauses for better speech rhythm
        text = text.replace(' - ', ' ... ')
        text = text.replace(':', ': ')
        
        # Ensure proper sentence endings
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def split_text_for_streaming(self, text: str, max_chunk_size: int = 500) -> list[str]:
        """
        Split text into chunks for streaming generation
        
        Args:
            text: Text to split
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def test_connection(self) -> bool:
        """
        Test connection to ElevenLabs API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to generate a short test phrase
            test_audio = self.generate_speech("Hello, I am Gaia.")
            return test_audio is not None
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False