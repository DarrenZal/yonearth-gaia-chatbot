#!/bin/bash
#
# Launch book knowledge graph extraction for "Our Biggest Deal"
# Runs in background with separate API key (OPENAI_API_KEY_2)
#

set -e

echo "================================================================================"
echo "📖 KNOWLEDGE GRAPH v3.2.2 - BOOK EXTRACTION: OUR BIGGEST DEAL"
echo "================================================================================"
echo ""

# Change to project directory
cd /home/claudeuser/yonearth-gaia-chatbot

# Load environment variables
if [ -f .env ]; then
    echo "✓ Loading environment from .env"
    set -a
    source .env
    set +a
else
    echo "❌ ERROR: .env file not found!"
    exit 1
fi

# Check OPENAI_API_KEY_2
if [ -z "$OPENAI_API_KEY_2" ]; then
    echo "❌ ERROR: OPENAI_API_KEY_2 not set in .env"
    exit 1
fi
echo "✓ OPENAI_API_KEY_2 loaded (separate rate limit)"
echo ""

# Create log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/kg_extraction_book_${TIMESTAMP}.log"

echo "📖 Book: Our Biggest Deal"
echo "📁 Output directory: data/knowledge_graph_books_v3_2_2/"
echo "📝 Log file: $LOG_FILE"
echo ""

echo "🚀 Starting extraction in background..."
echo ""

# Run in background with nohup
nohup python3 scripts/extract_kg_v3_2_2_book.py > "$LOG_FILE" 2>&1 &

PID=$!

echo "✅ Book extraction started!"
echo ""
echo "📊 Process ID: $PID"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "================================================================================"
echo "MONITORING COMMANDS:"
echo "================================================================================"
echo ""
echo "# Watch progress (live tail):"
echo "  tail -f $LOG_FILE"
echo ""
echo "# Check recent output:"
echo "  tail -50 $LOG_FILE"
echo ""
echo "# Check if process is still running:"
echo "  ps aux | grep $PID"
echo ""
echo "================================================================================"
echo ""

# Show initial output
sleep 3
echo "📄 Initial output:"
echo "--------------------------------------------------------------------------------"
head -30 "$LOG_FILE"
echo "--------------------------------------------------------------------------------"
echo ""
echo "✅ Running in background. Use 'tail -f $LOG_FILE' to monitor."
