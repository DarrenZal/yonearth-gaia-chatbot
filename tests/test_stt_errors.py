"""
Plan test #2 (app-level) — STT error paths.

Tests via FastAPI TestClient (no nginx):
- Unsupported multipart-part Content-Type → 415
- Over-budget Redis state → 429
- Missing 'file' field → 422
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def app():
    with patch("src.api.stt_endpoints.settings") as mock_s:
        mock_s.whisper_enabled = True
        mock_s.stt_daily_budget = 5.0
        mock_s.redis_url = "redis://localhost:6379"
        from src.api.stt_endpoints import stt_router
        app = FastAPI()
        app.include_router(stt_router)
        yield app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_unsupported_part_content_type_returns_415(client):
    """image/png part → 415 even though the top-level is multipart/form-data."""
    resp = client.post(
        "/api/stt",
        files={"file": ("test.png", b"\x89PNG\r\n", "image/png")},
    )
    assert resp.status_code == 415
    assert "Unsupported audio type" in resp.json()["detail"]


def test_over_budget_returns_429(client):
    """With budget exhausted, POST returns 429."""
    from src.voice.stt_budget import BudgetDeniedError
    with patch("src.api.stt_endpoints.stt_budget.reserve", side_effect=BudgetDeniedError("exhausted")):
        resp = client.post(
            "/api/stt",
            files={"file": ("a.webm", b"\x00" * 100, "audio/webm")},
        )
    assert resp.status_code == 429
    assert "daily budget" in resp.json()["detail"]


def test_missing_file_field_returns_422(client):
    """Multipart POST with wrong field name → 422 (FastAPI's built-in validation)."""
    resp = client.post(
        "/api/stt",
        files={"wrong_field": ("a.webm", b"\x00", "audio/webm")},
    )
    assert resp.status_code == 422
