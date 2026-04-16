"""
Atomic STT daily budget guard backed by Redis + Lua.

Design (plan §5):
- Redis key: stt:cost:YYYY-MM-DD (float counter in USD, auto-expires at midnight UTC)
- Reserve: atomic Lua EVAL reads counter, rejects if >= budget, else increments by
  RESERVATION ($0.03 = 5 min of Whisper). Returns "ok" or "denied".
- Reconcile: after Whisper returns, adjust counter so it reflects true cost:
  - actual < reservation → refund delta
  - actual > reservation → top-up delta (never absorb — AC #33)
  - actual == reservation → no-op
- Refund: on Whisper failure, decrement the full reservation (request never happened).
- Fail-closed: any Redis error → raise BudgetUnavailableError (handler returns 503,
  Whisper is never called).
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

import redis

from ..config import settings

logger = logging.getLogger(__name__)

RESERVATION_USD = 0.03  # 5 minutes × $0.006/min
COST_PER_MINUTE = 0.006
# Conservative byte-rate estimate when duration is missing:
# 16-kHz mono 16-bit PCM = 32 kB/s; WebM Opus is much smaller but we
# overestimate intentionally (fail-closed on cost).
FALLBACK_BYTES_PER_SECOND = 16_000

_pool: redis.ConnectionPool | None = None


class BudgetDeniedError(Exception):
    """Raised when the daily budget is exhausted."""


class BudgetUnavailableError(Exception):
    """Raised when Redis is unreachable — fail-closed, do NOT call Whisper."""


def _get_redis() -> redis.Redis:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(settings.redis_url, decode_responses=True)
    return redis.Redis(connection_pool=_pool)


def _key_and_ttl() -> tuple[str, int]:
    """Return today's counter key and seconds until midnight UTC."""
    now = datetime.now(timezone.utc)
    key = f"stt:cost:{now.strftime('%Y-%m-%d')}"
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    midnight_tomorrow = midnight.timestamp() + 86400
    ttl = max(int(midnight_tomorrow - now.timestamp()), 1)
    return key, ttl


# ---- Lua scripts (executed atomically inside Redis) ----

_LUA_RESERVE = """
local key     = KEYS[1]
local budget  = tonumber(ARGV[1])
local reserve = tonumber(ARGV[2])
local ttl     = tonumber(ARGV[3])

local current = tonumber(redis.call('GET', key) or '0') or 0
if current >= budget then
    return 'denied'
end
redis.call('INCRBYFLOAT', key, reserve)
redis.call('EXPIRE', key, ttl)
return 'ok'
"""


def reserve(budget: float | None = None) -> None:
    """Atomically check budget and reserve $0.03. Raises on denial or Redis error."""
    if budget is None:
        budget = settings.stt_daily_budget
    key, ttl = _key_and_ttl()
    try:
        r = _get_redis()
        result = r.eval(_LUA_RESERVE, 1, key, str(budget), str(RESERVATION_USD), str(ttl))
    except redis.RedisError as exc:
        logger.error("Redis error during STT budget reserve: %s", exc)
        raise BudgetUnavailableError(str(exc)) from exc

    if result == "denied":
        raise BudgetDeniedError(f"STT daily budget ${budget:.2f} exhausted")


def reconcile(duration_seconds: float | None, upload_size_bytes: int | None) -> float:
    """Adjust counter to reflect true Whisper cost. Returns actual cost charged.

    Three paths:
    1. duration present → actual_cost = 0.006 × ceil(duration/60)
       - actual < reservation → refund delta
       - actual > reservation → top-up delta
    2. duration missing/invalid → byte-rate fallback + WARN
    3. can't even estimate → keep full reservation (never decrement)
    """
    actual_cost = _compute_cost(duration_seconds, upload_size_bytes)
    delta = actual_cost - RESERVATION_USD

    if abs(delta) < 0.0001:
        return actual_cost  # close enough, no adjustment

    key, ttl = _key_and_ttl()
    try:
        r = _get_redis()
        r.incrbyfloat(key, delta)
        r.expire(key, ttl)
    except redis.RedisError as exc:
        # Fail-closed: if we can't reconcile, the reservation stays (overcount
        # is acceptable; undercount is a hard failure).
        logger.error("Redis error during STT budget reconcile: %s", exc)

    return actual_cost


def refund() -> None:
    """Undo a full reservation (Whisper call failed or was never made)."""
    key, ttl = _key_and_ttl()
    try:
        r = _get_redis()
        r.incrbyfloat(key, -RESERVATION_USD)
        r.expire(key, ttl)
    except redis.RedisError as exc:
        logger.error("Redis error during STT budget refund: %s", exc)


def _compute_cost(duration_seconds: float | None, upload_size_bytes: int | None) -> float:
    """Derive the Whisper cost in USD from duration or byte-rate fallback."""
    # Path 1: duration from Whisper verbose_json
    if duration_seconds is not None:
        try:
            dur = float(duration_seconds)
            if dur > 0:
                minutes = math.ceil(dur / 60)
                return COST_PER_MINUTE * minutes
        except (TypeError, ValueError):
            pass
        logger.warning("Invalid Whisper duration %r — falling back to byte-rate estimate", duration_seconds)

    # Path 2: byte-rate fallback
    if upload_size_bytes is not None and upload_size_bytes > 0:
        estimated_seconds = upload_size_bytes / FALLBACK_BYTES_PER_SECOND
        minutes = math.ceil(estimated_seconds / 60)
        cost = COST_PER_MINUTE * minutes
        logger.warning("Using byte-rate STT cost estimate: %d bytes → %.0fs → $%.4f", upload_size_bytes, estimated_seconds, cost)
        return cost

    # Path 3: can't estimate — keep full reservation
    logger.warning("Cannot estimate STT cost (no duration, no upload size) — keeping $%.2f reservation", RESERVATION_USD)
    return RESERVATION_USD


def get_counter() -> float:
    """Read today's counter value (for tests / audit). Returns 0.0 if unset."""
    key, _ = _key_and_ttl()
    try:
        r = _get_redis()
        val = r.get(key)
        return float(val) if val is not None else 0.0
    except redis.RedisError:
        return 0.0
