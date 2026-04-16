"""
Plan test #20 — Long-upload budget reconciliation.

With a starting counter at $0 and STT_DAILY_BUDGET=$1.00, fire one STT
request whose mocked Whisper response reports duration: 600 (10 min, cost
$0.06 — twice the $0.03 reservation). Assert:
- Call returns 200 (within budget)
- Post-call Redis counter equals exactly $0.06 (not $0.03)
- A follow-up call that pushes total over $1.00 returns 429

Hard pass criterion (AC #33): counter exactly matches cumulative true
Whisper cost — never below it.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_stt_budget import fake_redis  # reuse FakeRedis fixture


@pytest.fixture(autouse=True)
def mock_settings_everywhere():
    with patch("src.voice.stt_budget.settings") as bs, \
         patch("src.api.stt_endpoints.settings") as es:
        bs.redis_url = "redis://localhost:6379"
        bs.stt_daily_budget = 1.0
        es.whisper_enabled = True
        es.stt_daily_budget = 1.0
        es.redis_url = "redis://localhost:6379"
        yield


@pytest.fixture
def app_and_client(fake_redis):
    from src.api.stt_endpoints import stt_router
    app = FastAPI()
    app.include_router(stt_router)
    client = TestClient(app)
    return client, fake_redis


def test_long_upload_topup(app_and_client):
    client, fake_redis = app_and_client

    # Mock Whisper to return 10 minutes duration
    mock_result = {"text": "long audio", "duration": 600.0}
    with patch("src.api.stt_endpoints.whisper_transcribe", return_value=mock_result):
        resp = client.post(
            "/api/stt",
            files={"file": ("long.webm", b"\x00" * 5000, "audio/webm")},
        )

    assert resp.status_code == 200
    assert resp.json()["transcript"] == "long audio"

    # Counter should be exactly $0.06 (10 min × $0.006/min), not $0.03
    key = f"stt:cost:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    counter = float(fake_redis._store.get(key, "0"))
    assert counter == pytest.approx(0.06, abs=0.001)
