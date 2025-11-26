#!/bin/bash
#
# Production-safe GraphRAG builder with nohup
# Handles SSH disconnects, logging, and environment setup
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/graphrag_build_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Export it first: export OPENAI_API_KEY=your-key-here"
    exit 1
fi

# Export auto-confirm for background runs
export AUTO_CONFIRM_LLM=yes

echo "========================================"
echo "GraphRAG Builder - Production Mode"
echo "========================================"
echo "Script:  $SCRIPT_DIR/build_proper_graphrag.py"
echo "Log:     $LOG_FILE"
echo "PID:     $$"
echo ""
echo "Safeguards enabled:"
echo "  ✓ Lockfile prevents concurrent runs"
echo "  ✓ Checkpoint validation before start"
echo "  ✓ Progress checkpoints every 50 communities"
echo "  ✓ Heartbeat logging every 5 minutes"
echo "  ✓ Memory monitoring (20GB limit)"
echo "  ✓ Failed community tracking"
echo "  ✓ fsync on all checkpoint writes"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Run with nohup to survive SSH disconnect
cd "$ROOT_DIR"
nohup python3 "$SCRIPT_DIR/build_proper_graphrag.py" >> "$LOG_FILE" 2>&1 &

PID=$!
echo ""
echo "✅ Started GraphRAG builder (PID $PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "  ps -fp $PID"
echo ""
echo "Stop process:"
echo "  kill $PID"
echo ""
echo "Lockfile location:"
echo "  $ROOT_DIR/data/graphrag_hierarchy/graphrag.lock"
echo ""
