#!/usr/bin/env bash
# Plan AC #26 — Deterministic rate-limit pacer.
#
# Fires exactly one POST /api/stt every 1.95 seconds (so 30 requests fit
# in 58.5s, well inside the 60-second window). Request #31 fires at
# t=58.6s and must return HTTP 429.
#
# Usage:
#   bash tests/smoke/test_stt_ratelimit.sh https://earthdo.me
#
# A small (100-byte) audio stub is used — the point is to test nginx's
# rate limit, not Whisper. Expect 415 or 200 for valid requests depending
# on whether WHISPER_ENABLED is true and the file is real audio.
# The key assertion is that request #31 returns 429.
set -euo pipefail

BASE_URL="${1:?Usage: $0 <base-url>}"

echo "=== STT rate-limit pacer: 31 requests at 1.95s intervals ==="
echo "Target: ${BASE_URL}/api/stt"

# Create minimal stub file
printf '\x00%.0s' {1..100} > /tmp/stt-stub.bin

CODES=()
START=$(date +%s)

for i in $(seq 1 31); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE_URL}/api/stt" \
        -F "file=@/tmp/stt-stub.bin;type=audio/webm" --max-time 10)
    ELAPSED=$(( $(date +%s) - START ))
    echo "  req #${i}: HTTP ${CODE}  (t=${ELAPSED}s)"
    CODES+=("$CODE")
    if [ "$i" -lt 31 ]; then
        sleep 1.95
    fi
done

rm -f /tmp/stt-stub.bin

# Assert: requests 1-30 should NOT be 429
EARLY_429=0
for i in $(seq 0 29); do
    if [ "${CODES[$i]}" = "429" ]; then
        EARLY_429=$((EARLY_429 + 1))
    fi
done

# Assert: request 31 must be 429
LAST="${CODES[30]}"

echo ""
if [ "$EARLY_429" -gt 0 ]; then
    echo "❌ FAIL: ${EARLY_429} of the first 30 requests returned 429 (rate limit too tight or burst config)"
    exit 1
fi

if [ "$LAST" = "429" ]; then
    echo "✅ PASS: request #31 returned 429 — rate limit enforced"
    exit 0
else
    echo "❌ FAIL: request #31 returned ${LAST} (expected 429)"
    exit 1
fi
