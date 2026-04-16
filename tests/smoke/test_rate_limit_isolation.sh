#!/usr/bin/env bash
# Plan test #16 — Per-user rate-limit isolation through nginx.
#
# Must be run from a host whose source IP is in set_real_ip_from (typically
# yonearth.org's server). Running from an untrusted IP will cause the XFF
# headers to be ignored and the test to silently false-pass.
#
# Usage:
#   bash tests/smoke/test_rate_limit_isolation.sh https://earthdo.me
#
# Logic:
#   1. Fire 30 req from IP_A (via X-Forwarded-For) — expect all succeed
#   2. Fire 30 req from IP_B — expect all succeed
#   3. Fire 1 more from IP_A — expect 429
#   4. Fire 1 more from IP_B — expect 200 (IP_B has its own bucket)
set -euo pipefail

BASE_URL="${1:?Usage: $0 <base-url>}"
IP_A="198.51.100.1"
IP_B="203.0.113.1"

echo "=== Rate-limit isolation test ==="
echo "Base: ${BASE_URL}"
echo "IP_A: ${IP_A}  IP_B: ${IP_B}"
echo "Source IP: $(curl -s https://ifconfig.me || echo unknown)"
echo ""

printf '\x00%.0s' {1..100} > /tmp/stt-isolation.bin

fire() {
    local xff="$1"
    curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE_URL}/api/stt" \
        -F "file=@/tmp/stt-isolation.bin;type=audio/webm" \
        --header "X-Forwarded-For: ${xff}" --max-time 10
}

FAIL=0

# Step 1+2: 30 from each IP
echo "--- Step 1: 30 requests from IP_A ---"
for i in $(seq 1 30); do
    CODE=$(fire "$IP_A")
    if [ "$CODE" = "429" ]; then
        echo "  ❌ IP_A req #${i} returned 429 early"
        FAIL=1
    fi
done
echo "  All 30 sent."

echo "--- Step 2: 30 requests from IP_B ---"
for i in $(seq 1 30); do
    CODE=$(fire "$IP_B")
    if [ "$CODE" = "429" ]; then
        echo "  ❌ IP_B req #${i} returned 429 early"
        FAIL=1
    fi
done
echo "  All 30 sent."

# Step 3: #31 from IP_A → 429
echo "--- Step 3: IP_A request #31 ---"
CODE_A=$(fire "$IP_A")
echo "  IP_A #31: HTTP ${CODE_A}"
if [ "$CODE_A" != "429" ]; then
    echo "  ❌ FAIL: expected 429, got ${CODE_A}"
    FAIL=1
fi

# Step 4: IP_B should still have room → 200 (or any non-429)
echo "--- Step 4: IP_B request #31 ---"
CODE_B=$(fire "$IP_B")
echo "  IP_B #31: HTTP ${CODE_B}"
if [ "$CODE_B" = "429" ]; then
    echo "  ❌ FAIL: IP_B cross-throttled by IP_A — isolation broken"
    FAIL=1
fi

rm -f /tmp/stt-isolation.bin

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "✅ PASS: rate-limit isolation confirmed"
    exit 0
else
    echo "❌ FAIL: see errors above"
    exit 1
fi
