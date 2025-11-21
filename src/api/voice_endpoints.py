"""
Voice endpoints for text-to-speech functionality
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import settings
from ..voice.piper_client import PiperVoiceClient
from .models import VoiceGenerationRequest, VoiceGenerationResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/voice", tags=["voice"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global voice client
voice_client: Optional[PiperVoiceClient] = None


def get_voice_client():
    """Get or create voice client"""
    global voice_client

    if voice_client is None:
        try:
            voice_client = PiperVoiceClient(
                voice_name="en_US-kristin-medium"
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Voice service initialization failed: {str(e)}"
            )

    return voice_client


@router.post("/generate", response_model=VoiceGenerationResponse)
@limiter.limit("10/minute")
async def generate_voice(
    request: VoiceGenerationRequest,
    client: PiperVoiceClient = Depends(get_voice_client)
) -> VoiceGenerationResponse:
    """
    Generate voice audio from text
    
    Args:
        request: Voice generation request with text and settings
        
    Returns:
        Base64 encoded audio data
    """
    try:
        # Preprocess text for better speech
        processed_text = client.preprocess_text_for_speech(request.text)
        
        # Generate audio
        audio_base64 = client.generate_speech_base64(
            text=processed_text,
            voice_settings=request.voice_settings,
            output_format=request.output_format
        )
        
        if not audio_base64:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate audio"
            )
        
        return VoiceGenerationResponse(
            audio_data=audio_base64,
            text_length=len(request.text)
        )
        
    except Exception as e:
        logger.error(f"Voice generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice generation failed: {str(e)}"
        )


@router.get("/test")
async def test_voice_connection(
    client: PiperVoiceClient = Depends(get_voice_client)
) -> dict:
    """
    Test voice service connectivity

    Returns:
        Connection status
    """
    try:
        success = client.test_connection()

        return {
            "connected": success,
            "voice_name": "en_US-kristin-medium",
            "tts_engine": "Piper"
        }

    except Exception as e:
        logger.error(f"Voice test error: {str(e)}")
        return {
            "connected": False,
            "error": str(e)
        }