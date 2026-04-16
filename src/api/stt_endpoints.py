"""
Speech-to-text endpoints — OpenAI Whisper integration.

Two routers:
- stt_status_router: unconditionally mounted, serves GET /api/stt/status
  so the frontend can discover whether STT is available before showing
  the mic button.
- stt_router: conditionally mounted when WHISPER_ENABLED=true, serves
  POST /api/stt (the actual transcription endpoint, added in a later PR).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from ..config import settings

logger = logging.getLogger(__name__)

# --- Unconditionally mounted (always tells the frontend the flag state) ---

stt_status_router = APIRouter(prefix="/api/stt", tags=["stt"])


@stt_status_router.get("/status")
async def stt_status():
    """Return whether the STT POST endpoint is mounted.

    Contract:
      GET /api/stt/status → 200 {"enabled": true}   (flag on, POST route mounted)
      GET /api/stt/status → 200 {"enabled": false}   (flag off, POST route NOT mounted)
    """
    return {"enabled": settings.whisper_enabled}


# --- Conditionally mounted (only when WHISPER_ENABLED=true) ---

stt_router = APIRouter(prefix="/api", tags=["stt"])

# POST /api/stt is added in voice-stt-backend PR.
# Placeholder so the router has at least one route when mounted;
# removed once the real handler lands.
