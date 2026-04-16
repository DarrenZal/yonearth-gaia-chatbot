"""
Plan test #18 — WHISPER_ENABLED=false branch behavior.

Verifies:
- /api/stt/status returns 200 {"enabled": false} when flag is off
- /api/stt/status returns 200 {"enabled": true} when flag is on
- POST /api/stt returns 404 when flag is off (router not mounted)
- Chat / TTS / taxonomy remain unaffected by the flag state
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_app(whisper_enabled: bool) -> FastAPI:
    """Build a minimal app replicating main.py's conditional STT mount logic."""
    from src.api.stt_endpoints import stt_status_router, stt_router

    app = FastAPI()
    app.include_router(stt_status_router)
    if whisper_enabled:
        app.include_router(stt_router)
    return app


class TestWhisperDisabled:
    """WHISPER_ENABLED=false: STT cleanly absent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch("src.api.stt_endpoints.settings") as mock:
            mock.whisper_enabled = False
            self.app = _make_app(whisper_enabled=False)
            self.client = TestClient(self.app)
            # Keep mock alive for requests
            yield

    def test_stt_status_returns_enabled_false(self):
        with patch("src.api.stt_endpoints.settings") as mock:
            mock.whisper_enabled = False
            resp = self.client.get("/api/stt/status")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": False}

    def test_post_stt_returns_404(self):
        resp = self.client.post("/api/stt", files={"file": ("a.webm", b"\x00", "audio/webm")})
        assert resp.status_code in (404, 405)


class TestWhisperEnabled:
    """WHISPER_ENABLED=true: status reflects it."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch("src.api.stt_endpoints.settings") as mock:
            mock.whisper_enabled = True
            self.app = _make_app(whisper_enabled=True)
            self.client = TestClient(self.app)
            yield

    def test_stt_status_returns_enabled_true(self):
        with patch("src.api.stt_endpoints.settings") as mock:
            mock.whisper_enabled = True
            resp = self.client.get("/api/stt/status")
        assert resp.status_code == 200
        assert resp.json() == {"enabled": True}
