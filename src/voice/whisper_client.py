"""
Thin OpenAI Whisper wrapper for speech-to-text.

Returns transcript text + duration (seconds) from the verbose_json response
so the budget guard can reconcile actual cost without ffmpeg.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO

from openai import OpenAI

from ..config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def transcribe(
    file: BinaryIO,
    filename: str = "audio.webm",
    content_type: str = "audio/webm",
) -> dict:
    """Transcribe audio via OpenAI Whisper.

    Returns:
        {"text": str, "duration": float | None}

    duration is in seconds, read from Whisper's verbose_json response.
    None if the response omits it (caller should fall back to byte-rate).
    """
    client = _get_client()

    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, file, content_type),
        response_format="verbose_json",
    )

    # openai SDK returns a Transcription object; access attrs directly
    text = getattr(resp, "text", "") or ""
    duration = getattr(resp, "duration", None)

    if duration is not None:
        try:
            duration = float(duration)
            if duration <= 0:
                logger.warning("Whisper returned non-positive duration: %s", duration)
                duration = None
        except (TypeError, ValueError):
            logger.warning("Whisper returned unparseable duration: %r", duration)
            duration = None

    return {"text": text, "duration": duration}
