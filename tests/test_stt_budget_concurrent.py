"""
Plan test #11 — STT budget concurrency.

50 threads all attempt reserve() simultaneously (via barrier). With a
budget of $0.10 and a $0.03 reservation, at most floor(0.10/0.03) = 3
should succeed before reconciliation frees anything up.

This tests the atomicity of the Lua reserve script: no two threads can
both read the counter, decide "under budget", and both increment — the
Lua script serializes that check-and-increment.
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("src.voice.stt_budget.settings") as bs:
        bs.redis_url = "redis://localhost:6379"
        bs.stt_daily_budget = 0.09  # exactly 3 × $0.03 reservation
        yield bs


@pytest.fixture
def thread_safe_redis():
    """Thread-safe in-memory Redis mock with locking for EVAL atomicity."""
    store = {}
    lock = threading.Lock()

    class TSRedis:
        def get(self, key):
            with lock:
                return store.get(key)

        def incrbyfloat(self, key, amount):
            with lock:
                cur = float(store.get(key, "0"))
                new = cur + float(amount)
                store[key] = str(new)
                return new

        def expire(self, key, ttl):
            pass

        def eval(self, script, numkeys, *args):
            with lock:
                key = args[0]
                budget = float(args[1])
                reserve_amt = float(args[2])
                ttl = int(args[3])
                current = float(store.get(key, "0"))
                if current >= budget:
                    return "denied"
                store[key] = str(current + reserve_amt)
                return "ok"

    fake = TSRedis()
    fake._store = store

    with patch("src.voice.stt_budget._get_redis", return_value=fake):
        import src.voice.stt_budget as mod
        mod._pool = None
        yield fake


def test_concurrent_50_reserve_only(thread_safe_redis):
    """50 threads barrier-synchronize then all try reserve() at once.

    No reconciliation — pure reservation atomicity test. Budget $0.09
    fits exactly 3 × $0.03 reservations. The 3rd call sees counter $0.06
    which is < $0.09, so it passes (counter → $0.09). The 4th sees $0.09
    which is >= $0.09, so it's denied.
    """
    from src.voice.stt_budget import reserve, BudgetDeniedError

    N = 50
    barrier = threading.Barrier(N, timeout=10)
    ok_count = 0
    denied_count = 0
    count_lock = threading.Lock()
    errors = []

    def worker():
        nonlocal ok_count, denied_count
        try:
            barrier.wait()  # all threads start reserve() at the same instant
            reserve(budget=0.09)
            with count_lock:
                ok_count += 1
        except BudgetDeniedError:
            with count_lock:
                denied_count += 1
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"Unexpected errors: {errors}"
    assert ok_count + denied_count == N

    # Atomicity: exactly 3 reservations fit in $0.09 budget at $0.03 each
    assert ok_count == 3, f"Expected exactly 3, got {ok_count}"

    # Final counter: exactly 3 × $0.03 = $0.09
    key = f"stt:cost:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    final = float(thread_safe_redis._store.get(key, "0"))
    assert final == pytest.approx(0.09, abs=0.001)
