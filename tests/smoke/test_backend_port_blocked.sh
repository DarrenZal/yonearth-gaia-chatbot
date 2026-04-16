#!/usr/bin/env bash
# Plan test #17 — Backend port :8000 not externally reachable.
#
# Verifies that the FastAPI backend on port 8000 is not accessible from
# the public network. Must be run from a non-server host.
#
# Usage:
#   bash tests/smoke/test_backend_port_blocked.sh
#
# Tests both IP and domain name to catch split-horizon DNS edge cases.
set -euo pipefail

SERVER_IP="152.53.194.214"
SERVER_HOST="earthdo.me"
PASS=0
FAIL=0

pass() { echo "  ✅ PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  ❌ FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== Backend port :8000 reachability test ==="
echo "Run from: $(hostname) (must NOT be the server itself)"
echo ""

# --- Test 1: nc against IP ---
echo "--- nc -zv ${SERVER_IP} 8000 (expect refused/timeout) ---"
if nc -zv -w 5 ${SERVER_IP} 8000 2>&1 | grep -qiE 'refused|timed? out'; then
    pass "nc ${SERVER_IP}:8000 → refused/timeout"
elif ! nc -zv -w 5 ${SERVER_IP} 8000 2>/dev/null; then
    pass "nc ${SERVER_IP}:8000 → connection failed"
else
    fail "nc ${SERVER_IP}:8000 → port appears open!"
fi

# --- Test 2: curl against IP ---
echo "--- curl http://${SERVER_IP}:8000/api/taxonomy (expect fail) ---"
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://${SERVER_IP}:8000/api/taxonomy" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "000" ]; then
    pass "curl ${SERVER_IP}:8000 → connection failed (000)"
else
    fail "curl ${SERVER_IP}:8000 → got HTTP ${HTTP_CODE} (expected connection failure)"
fi

# --- Test 3: curl against domain ---
echo "--- curl http://${SERVER_HOST}:8000/api/taxonomy (expect fail) ---"
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://${SERVER_HOST}:8000/api/taxonomy" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "000" ]; then
    pass "curl ${SERVER_HOST}:8000 → connection failed (000)"
else
    fail "curl ${SERVER_HOST}:8000 → got HTTP ${HTTP_CODE} (expected connection failure)"
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ]
