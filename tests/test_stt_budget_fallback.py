"""
Plan test #9 — STT budget fail-closed on missing/bad duration.

Three cases:
1. Whisper response with no duration → byte-rate estimate used, WARN logged
2. duration: "garbage" → byte-rate estimate used, WARN logged
3. JSON parse failure → full reservation kept

In all cases the counter is provably non-zero after the call.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("src.voice.stt_budget.settings") as s:
        s.redis_url = "redis://localhost:6379"
        s.stt_daily_budget = 5.0
        yield s


@pytest.fixture
def fake_redis():
    store = {}

    class FakeRedis:
        def get(self, key):
            return store.get(key)

        def incrbyfloat(self, key, amount):
            cur = float(store.get(key, "0"))
            new = cur + float(amount)
            store[key] = str(new)
            return new

        def expire(self, key, ttl):
            pass

        def eval(self, script, numkeys, *args):
            key = args[0]
            budget = float(args[1])
            reserve_amt = float(args[2])
            ttl = int(args[3])
            current = float(store.get(key, "0"))
            if current >= budget:
                return "denied"
            store[key] = str(current + reserve_amt)
            return "ok"

    fake = FakeRedis()
    fake._store = store

    with patch("src.voice.stt_budget._get_redis", return_value=fake):
        import src.voice.stt_budget as mod
        mod._pool = None
        yield fake


def _today_key():
    return f"stt:cost:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


def test_missing_duration_uses_byte_rate(fake_redis):
    from src.voice.stt_budget import reserve, reconcile
    reserve(budget=5.0)
    actual = reconcile(duration_seconds=None, upload_size_bytes=160_000)
    # 160_000 / 16_000 = 10s → ceil(10/60) = 1 min → $0.006
    assert actual == pytest.approx(0.006)
    assert float(fake_redis._store[_today_key()]) > 0


def test_garbage_duration_uses_byte_rate(fake_redis):
    from src.voice.stt_budget import reserve, reconcile
    reserve(budget=5.0)
    actual = reconcile(duration_seconds="garbage", upload_size_bytes=320_000)
    # 320_000 / 16_000 = 20s → ceil(20/60) = 1 min → $0.006
    assert actual == pytest.approx(0.006)
    assert float(fake_redis._store[_today_key()]) > 0


def test_no_duration_no_size_keeps_reservation(fake_redis):
    from src.voice.stt_budget import reserve, reconcile, RESERVATION_USD
    reserve(budget=5.0)
    actual = reconcile(duration_seconds=None, upload_size_bytes=None)
    assert actual == pytest.approx(RESERVATION_USD)
    assert float(fake_redis._store[_today_key()]) == pytest.approx(RESERVATION_USD)
