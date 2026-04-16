"""
Plan test #14 — Top-level non-multipart STT 415.

POST /api/stt with Content-Type: audio/webm and a raw audio binary body
(no multipart wrapper) returns HTTP 415 with the explicit validator message.
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


@pytest.fixture
def client():
    with patch("src.api.stt_endpoints.settings") as mock_s:
        mock_s.whisper_enabled = True
        mock_s.stt_daily_budget = 5.0
        mock_s.redis_url = "redis://localhost:6379"
        from src.api.stt_endpoints import stt_router
        app = FastAPI()
        app.include_router(stt_router)
        return TestClient(app)


def test_raw_body_returns_415(client):
    """Raw audio/* body (no multipart wrapper) → 415."""
    resp = client.post(
        "/api/stt",
        content=b"\x00" * 1000,
        headers={"Content-Type": "audio/webm"},
    )
    assert resp.status_code == 415
    body = resp.json()
    assert body["detail"] == "STT requires multipart/form-data with a 'file' field"
