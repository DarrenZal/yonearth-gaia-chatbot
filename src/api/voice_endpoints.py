"""
Voice endpoints for text-to-speech functionality
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import settings
from ..voice.elevenlabs_client import ElevenLabsVoiceClient
from .models import VoiceGenerationRequest, VoiceGenerationResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/voice", tags=["voice"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global voice client
voice_client: Optional[ElevenLabsVoiceClient] = None


def get_voice_client():
    """Get or create voice client"""
    global voice_client
    
    if not settings.elevenlabs_api_key:
        raise HTTPException(
            status_code=503,
            detail="Voice service not configured"
        )
    
    if voice_client is None:
        voice_client = ElevenLabsVoiceClient(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model_id
        )
    
    return voice_client


@router.post("/generate", response_model=VoiceGenerationResponse)
@limiter.limit("10/minute")
async def generate_voice(
    request: VoiceGenerationRequest,
    client: ElevenLabsVoiceClient = Depends(get_voice_client)
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
    client: ElevenLabsVoiceClient = Depends(get_voice_client)
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
            "voice_id": settings.elevenlabs_voice_id,
            "model_id": settings.elevenlabs_model_id
        }
        
    except Exception as e:
        logger.error(f"Voice test error: {str(e)}")
        return {
            "connected": False,
            "error": str(e)
        }