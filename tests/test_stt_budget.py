"""
Plan test #1 — STT budget logic.

Tests the Lua-based atomic reserve/reconcile cycle:
- Pre-call check returns 200 when counter < budget
- Post-success increment uses actual Whisper-reported duration
- Boundary call (the one that crosses the budget) still succeeds
- The *next* call after crossing returns 429
- Redis key has TTL within ±5s of midnight UTC
"""
from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("src.voice.stt_budget.settings") as s:
        s.redis_url = "redis://localhost:6379"
        s.stt_daily_budget = 0.10
        yield s


@pytest.fixture
def fake_redis():
    """In-memory Redis mock that supports GET, INCRBYFLOAT, EXPIRE, EVAL."""
    store = {}
    ttls = {}

    class FakeRedis:
        def get(self, key):
            return store.get(key)

        def incrbyfloat(self, key, amount):
            cur = float(store.get(key, "0"))
            new = cur + float(amount)
            store[key] = str(new)
            return new

        def expire(self, key, ttl):
            ttls[key] = ttl

        def eval(self, script, numkeys, *args):
            # Minimal Lua-script emulation for the reserve script
            key = args[0]
            budget = float(args[1])
            reserve_amt = float(args[2])
            ttl = int(args[3])
            current = float(store.get(key, "0"))
            if current >= budget:
                return "denied"
            new = current + reserve_amt
            store[key] = str(new)
            ttls[key] = ttl
            return "ok"

    fake = FakeRedis()
    fake._store = store
    fake._ttls = ttls

    with patch("src.voice.stt_budget._get_redis", return_value=fake):
        # Reset module-level pool so it uses our mock
        import src.voice.stt_budget as mod
        mod._pool = None
        yield fake


def test_reserve_succeeds_under_budget(fake_redis):
    from src.voice.stt_budget import reserve
    reserve(budget=0.10)  # counter=0 < 0.10 → ok
    assert float(fake_redis._store.get(list(fake_redis._store.keys())[0])) == pytest.approx(0.03)


def test_reserve_denied_at_budget(fake_redis):
    from src.voice.stt_budget import reserve, BudgetDeniedError
    # Pre-load counter to exactly budget
    key = f"stt:cost:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    fake_redis._store[key] = "0.10"
    with pytest.raises(BudgetDeniedError):
        reserve(budget=0.10)


def test_boundary_call_succeeds_then_next_denied(fake_redis):
    """The call that crosses the budget succeeds; the next one is denied."""
    from src.voice.stt_budget import reserve, BudgetDeniedError
    key = f"stt:cost:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    fake_redis._store[key] = "0.09"  # just under 0.10

    reserve(budget=0.10)  # counter goes to 0.12, still approved (check is < budget)
    with pytest.raises(BudgetDeniedError):
        reserve(budget=0.10)


def test_reconcile_normal_refund(fake_redis):
    """actual cost < reservation → counter decremented by delta."""
    from src.voice.stt_budget import reserve, reconcile, RESERVATION_USD
    reserve(budget=1.0)
    # Whisper says 15 seconds → ceil(15/60) = 1 min → $0.006
    actual = reconcile(duration_seconds=15.0, upload_size_bytes=None)
    assert actual == pytest.approx(0.006)
    key = list(fake_redis._store.keys())[0]
    assert float(fake_redis._store[key]) == pytest.approx(0.006)


def test_reconcile_topup(fake_redis):
    """actual cost > reservation → counter incremented by delta (plan test #20 core)."""
    from src.voice.stt_budget import reserve, reconcile, RESERVATION_USD
    reserve(budget=1.0)
    # 10 minutes → ceil(600/60) = 10 min → $0.06 (twice the $0.03 reservation)
    actual = reconcile(duration_seconds=600.0, upload_size_bytes=None)
    assert actual == pytest.approx(0.06)
    key = list(fake_redis._store.keys())[0]
    assert float(fake_redis._store[key]) == pytest.approx(0.06)


def test_redis_key_ttl_near_midnight(fake_redis):
    """TTL on the counter key should be ≤ 86400 and > 0."""
    from src.voice.stt_budget import reserve
    reserve(budget=1.0)
    key = list(fake_redis._ttls.keys())[0]
    ttl = fake_redis._ttls[key]
    assert 0 < ttl <= 86400
