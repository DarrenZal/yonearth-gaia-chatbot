#!/usr/bin/env python3
"""
One-shot probe: verify the OPENAI_API_KEY has Whisper (audio transcription) access.

Run before implementing §5 STT — if this returns 401/403 the key lacks Whisper
permission and the /api/stt endpoint should stay disabled (WHISPER_ENABLED=false).

Usage:
  python scripts/probe_whisper.py            # uses OPENAI_API_KEY from env / .env
  python scripts/probe_whisper.py --key sk-…  # explicit key override

Exit codes:
  0 — Whisper access confirmed (transcript returned)
  1 — access denied (401/403) or other API error
  2 — missing key
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys
from pathlib import Path

# Load .env if available
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)


def make_silent_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal silent WAV file in memory (no external deps)."""
    num_samples = int(sample_rate * duration_s)
    data_size = num_samples * 2  # 16-bit mono
    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)
    return buf.getvalue()


def probe(api_key: str) -> bool:
    """Send a 1-second silent WAV to Whisper. Return True if transcript comes back."""
    try:
        import httpx
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
        import httpx

    wav_bytes = make_silent_wav()
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("probe.wav", wav_bytes, "audio/wav")}
    data = {"model": "whisper-1"}

    print(f"Probing Whisper API with 1s silent WAV…")
    resp = httpx.post(url, headers=headers, files=files, data=data, timeout=30)

    print(f"HTTP {resp.status_code}")
    if resp.status_code == 200:
        body = resp.json()
        print(f"Transcript: {body.get('text', '(empty)')!r}")
        print("✅ Whisper access confirmed — WHISPER_ENABLED=true is safe to set.")
        return True
    else:
        print(f"Response: {resp.text[:500]}")
        if resp.status_code in (401, 403):
            print("❌ Key lacks Whisper permission. Upgrade in OpenAI dashboard before enabling STT.")
        else:
            print(f"❌ Unexpected status {resp.status_code}.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenAI Whisper access")
    parser.add_argument("--key", help="OpenAI API key (default: OPENAI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key. Set OPENAI_API_KEY or pass --key.", file=sys.stderr)
        return 2

    return 0 if probe(api_key) else 1


if __name__ == "__main__":
    sys.exit(main())
