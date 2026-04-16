#!/usr/bin/env bash
# Plan test #2 (nginx-level) — STT abuse limits enforced by nginx.
#
# Runs against a live staging URL (not pytest — nginx behaviors aren't
# observable from FastAPI TestClient).
#
# Usage:
#   bash tests/smoke/test_stt_nginx.sh https://earthdo.me
#
# Pass criterion: every assertion exits 0 against the staging URL.
set -euo pipefail

BASE_URL="${1:?Usage: $0 <base-url>}"
PASS=0
FAIL=0

pass() { echo "  ✅ PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  ❌ FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== STT nginx smoke tests against ${BASE_URL} ==="

# --- 1. Size cap: 11 MB upload → 413 ---
echo ""
echo "--- Test: 11 MB upload → HTTP 413 ---"
dd if=/dev/urandom of=/tmp/stt-big.bin bs=1048576 count=11 2>/dev/null
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE_URL}/api/stt" \
    -F "file=@/tmp/stt-big.bin;type=audio/webm" --max-time 30)
rm -f /tmp/stt-big.bin
if [ "$STATUS" = "413" ]; then
    pass "11 MB upload returned $STATUS"
else
    fail "11 MB upload returned $STATUS (expected 413)"
fi

# --- 2. Rate limit: 31 requests in 60s → #31 returns 429 ---
# Delegates to the deterministic pacer script (plan AC #26).
echo ""
echo "--- Test: rate limit (delegating to test_stt_ratelimit.sh) ---"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "${SCRIPT_DIR}/test_stt_ratelimit.sh" ] || [ -f "${SCRIPT_DIR}/test_stt_ratelimit.sh" ]; then
    if bash "${SCRIPT_DIR}/test_stt_ratelimit.sh" "${BASE_URL}"; then
        pass "Rate limit pacer passed"
    else
        fail "Rate limit pacer failed"
    fi
else
    echo "  ⚠️  SKIP: test_stt_ratelimit.sh not found"
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ]
