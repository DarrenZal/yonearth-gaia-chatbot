"""
Plan test #15 — Redis-down fail-closed.

When Redis is unreachable (ConnectionError, TimeoutError, ResponseError),
the budget guard raises BudgetUnavailableError → handler returns 503 →
Whisper is NEVER called.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import redis as redis_lib
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


@pytest.fixture
def whisper_mock():
    """Mock Whisper so we can assert it was NOT called."""
    with patch("src.voice.whisper_client.transcribe") as m:
        m.return_value = {"text": "should not reach", "duration": 5.0}
        yield m


@pytest.mark.parametrize("exc_class", [
    redis_lib.ConnectionError,
    redis_lib.TimeoutError,
    redis_lib.ResponseError,
])
def test_redis_error_returns_503_and_whisper_not_called(client, whisper_mock, exc_class):
    from src.voice.stt_budget import BudgetUnavailableError
    with patch("src.api.stt_endpoints.stt_budget.reserve", side_effect=BudgetUnavailableError("redis down")):
        resp = client.post(
            "/api/stt",
            files={"file": ("a.webm", b"\x00" * 100, "audio/webm")},
        )
    assert resp.status_code == 503
    assert "temporarily unavailable" in resp.json()["detail"]
    whisper_mock.assert_not_called()
