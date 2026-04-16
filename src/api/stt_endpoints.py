"""
Speech-to-text endpoints — OpenAI Whisper integration.

Two routers:
- stt_status_router: unconditionally mounted, serves GET /api/stt/status
  so the frontend can discover whether STT is available before showing
  the mic button.
- stt_router: conditionally mounted when WHISPER_ENABLED=true, serves
  POST /api/stt (multipart audio → Whisper transcript).

Audio duration enforcement — Path A (MVP):
  Client-side only (60s MediaRecorder auto-stop). Server trusts the 10 MB
  nginx size cap as a proxy. The cost guard bills using Whisper's reported
  duration, so a malicious long-upload gets billed accurately and hits
  STT_DAILY_BUDGET after 1–2 requests. No ffmpeg dependency needed.
"""
from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File

from ..config import settings
from ..voice import stt_budget
from ..voice.stt_budget import BudgetDeniedError, BudgetUnavailableError
from ..voice.whisper_client import transcribe as whisper_transcribe

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_TYPES = frozenset({
    "audio/webm",
    "audio/wav",
    "audio/mpeg",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a",
})

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


# --- Multipart guard (runs as Depends BEFORE FastAPI resolves File()) ---

async def _require_multipart(request: Request):
    """Reject non-multipart requests with 415 before FastAPI's File() binding.

    Without this, a raw-body POST with Content-Type: audio/webm gets a 422
    from FastAPI's auto-validation ("field required: file"). The plan
    requires an explicit 415 with a clear message.
    """
    ct = request.headers.get("content-type", "")
    if not ct.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=415,
            detail="STT requires multipart/form-data with a 'file' field",
        )


# --- Conditionally mounted (only when WHISPER_ENABLED=true) ---

stt_router = APIRouter(prefix="/api", tags=["stt"])


@stt_router.post("/stt", dependencies=[Depends(_require_multipart)])
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    """Transcribe uploaded audio via OpenAI Whisper.

    Request contract: multipart/form-data with a 'file' field containing the
    audio blob. Raw audio/* bodies are rejected with 415 by the
    _require_multipart dependency before FastAPI's auto-binding kicks in.

    Returns: {"transcript": str}
    """
    # --- Part-level Content-Type must be an allowed audio type ---
    part_ct = (file.content_type or "").lower().strip()
    if part_ct not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{part_ct}'. Accepted: {sorted(ALLOWED_AUDIO_TYPES)}",
        )

    # --- Budget check (atomic Lua reserve) ---
    try:
        stt_budget.reserve()
    except BudgetDeniedError:
        raise HTTPException(
            status_code=429,
            detail="Voice input temporarily unavailable — daily budget reached",
        )
    except BudgetUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Voice input temporarily unavailable",
        )

    # --- Read upload (needed for byte-rate fallback + Whisper call) ---
    audio_bytes = await file.read()
    upload_size = len(audio_bytes)

    # --- Call Whisper ---
    try:
        result = whisper_transcribe(
            file=io.BytesIO(audio_bytes),
            filename=file.filename or "audio.webm",
            content_type=part_ct,
        )
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        stt_budget.refund()
        raise HTTPException(status_code=502, detail="Transcription failed") from exc

    # --- Reconcile budget with actual duration ---
    actual_cost = stt_budget.reconcile(
        duration_seconds=result.get("duration"),
        upload_size_bytes=upload_size,
    )
    logger.info(
        "STT complete: %.1fs duration, $%.4f cost, %d bytes",
        result.get("duration") or 0,
        actual_cost,
        upload_size,
    )

    return {"transcript": result["text"]}
