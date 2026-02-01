"""
Piper Text-to-Speech client for Gaia voice generation
Using the en_US-kristin-medium voice model
"""
import os
import base64
import logging
import tempfile
import wave
from typing import Optional, Dict, Any
from io import BytesIO
import subprocess
import json

logger = logging.getLogger(__name__)


class PiperVoiceClient:
    """Client for Piper text-to-speech"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        voice_name: str = "en_US-kristin-medium"
    ):
        """
        Initialize Piper client

        Args:
            model_path: Path to the ONNX model file
            config_path: Path to the model config JSON file
            voice_name: Voice model name
        """
        base_path = "/home/claudeuser/yonearth-gaia-chatbot/data/voices"

        self.model_path = model_path or os.path.join(base_path, f"{voice_name}.onnx")
        self.config_path = config_path or os.path.join(base_path, f"{voice_name}.onnx.json")
        self.voice_name = voice_name

        # Verify model files exist
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.config_path):
            raise ValueError(f"Config file not found: {self.config_path}")

        # Load config to get voice parameters
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        logger.info(f"Piper client initialized with voice: {self.voice_name}")
        logger.info(f"Sample rate: {self.config['audio']['sample_rate']}Hz")

    def generate_speech(
        self,
        text: str,
        voice_settings: Optional[Dict[str, float]] = None,
        output_format: str = "wav"
    ) -> Optional[bytes]:
        """
        Generate speech from text using Piper

        Args:
            text: Text to convert to speech
            voice_settings: Voice settings (for compatibility, not used by Piper)
            output_format: Audio format (Piper outputs WAV by default)

        Returns:
            Audio data as bytes or None if generation fails
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text_for_speech(text)

            logger.debug(f"Generating speech for text length: {len(processed_text)}")

            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Use Piper CLI to generate speech
                # Note: Piper can be used as a Python module or CLI
                # Using CLI for simplicity and compatibility
                process = subprocess.Popen(
                    ['piper', '--model', self.model_path, '--config', self.config_path, '--output_file', temp_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                stdout, stderr = process.communicate(input=processed_text)

                if process.returncode != 0:
                    logger.error(f"Piper generation failed: {stderr}")
                    return None

                # Read the generated audio file
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()

                logger.info(f"Generated audio: {len(audio_data)} bytes")
                return audio_data

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None

    def generate_speech_base64(
        self,
        text: str,
        voice_settings: Optional[Dict[str, float]] = None,
        output_format: str = "wav"
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

        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'#{1,6}\s*(.*?)\n', r'\1. ', text)  # Headers

        # Remove citations like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Fix pronunciation: YonEarth -> "Y on Earth" (spoken naturally)
        text = re.sub(r'yonearth', 'Y on Earth', text, flags=re.IGNORECASE)

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
        Test Piper TTS functionality

        Returns:
            True if generation successful, False otherwise
        """
        try:
            # Try to generate a short test phrase
            test_audio = self.generate_speech("Hello, I am Gaia.")
            return test_audio is not None
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def generate_speech_python_api(
        self,
        text: str,
        voice_settings: Optional[Dict[str, float]] = None,
        output_format: str = "wav"
    ) -> Optional[bytes]:
        """
        Alternative: Generate speech using Piper Python API directly

        Args:
            text: Text to convert to speech
            voice_settings: Voice settings (not used)
            output_format: Audio format

        Returns:
            Audio data as bytes or None if generation fails
        """
        try:
            from piper import PiperVoice

            # Preprocess text
            processed_text = self.preprocess_text_for_speech(text)

            logger.debug(f"Generating speech using Python API for text length: {len(processed_text)}")

            # Initialize Piper voice
            voice = PiperVoice.load(self.model_path, config_path=self.config_path)

            # Generate audio (sentence_silence parameter may not be available in all versions)
            try:
                audio_generator = voice.synthesize(processed_text, sentence_silence=0.1)
            except TypeError:
                # Fallback without sentence_silence parameter
                audio_generator = voice.synthesize(processed_text)

            # Collect audio bytes
            audio_bytes = BytesIO()

            # Write WAV header
            sample_rate = self.config['audio']['sample_rate']
            num_channels = 1  # Mono
            sample_width = 2  # 16-bit

            # Collect all audio samples first to know the size
            # Handle AudioChunk objects if that's what's returned
            audio_samples = []
            for chunk in audio_generator:
                if hasattr(chunk, 'audio'):
                    # AudioChunk object
                    audio_samples.append(chunk.audio)
                else:
                    # Raw bytes
                    audio_samples.append(chunk)

            audio_samples = b''.join(audio_samples)

            # Create WAV file in memory
            with wave.open(audio_bytes, 'wb') as wav_file:
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_samples)

            audio_data = audio_bytes.getvalue()
            logger.info(f"Generated audio: {len(audio_data)} bytes")

            return audio_data

        except ImportError:
            logger.warning("Piper Python API not available, falling back to CLI")
            return self.generate_speech(text, voice_settings, output_format)
        except Exception as e:
            logger.error(f"Error generating speech with Python API: {str(e)}")
            return None