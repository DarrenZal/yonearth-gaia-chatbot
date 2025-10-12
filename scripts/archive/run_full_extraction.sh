#!/bin/bash
#
# Launch full v3.2.2 knowledge graph extraction for all 172 episodes
# Runs in background with nohup for stability
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "🚀 KNOWLEDGE GRAPH v3.2.2 - FULL EXTRACTION LAUNCHER"
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

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY not set in .env"
    exit 1
fi
echo "✓ OPENAI_API_KEY loaded"
echo ""

# Create log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/kg_extraction_full_${TIMESTAMP}.log"

echo "📁 Output directory: data/knowledge_graph_v3_2_2/"
echo "📝 Log file: $LOG_FILE"
echo ""

# Check status
PROCESSED=$(ls data/knowledge_graph_v3_2_2/episode_*_v3_2_2.json 2>/dev/null | wc -l)
REMAINING=$((172 - PROCESSED))

echo "📊 STATUS:"
echo "   Already processed: $PROCESSED episodes"
echo "   Remaining: $REMAINING episodes"
echo "   Estimated time: ~$(echo "scale=1; $REMAINING * 12.8 / 60" | bc) hours"
echo ""

# Confirmation
echo -e "${YELLOW}⚠️  This will process $REMAINING episodes (~35 hours)${NC}"
echo -e "${YELLOW}   The process will run in the background (nohup)${NC}"
echo ""
read -p "Continue? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Cancelled by user"
    exit 0
fi

echo "🚀 Starting extraction in background..."
echo ""

# Run in background with nohup
nohup python3 scripts/extract_kg_v3_2_2_full.py > "$LOG_FILE" 2>&1 &

PID=$!

echo -e "${GREEN}✅ Extraction started!${NC}"
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
echo "# Check how many episodes completed:"
echo "  ls data/knowledge_graph_v3_2_2/episode_*_v3_2_2.json | wc -l"
echo ""
echo "# Check if process is still running:"
echo "  ps aux | grep $PID"
echo ""
echo "# Stop the process (if needed):"
echo "  kill $PID"
echo ""
echo "================================================================================"
echo ""
echo "💡 TIP: Re-run this script anytime to resume - it auto-skips processed episodes"
echo ""

# Show initial output
sleep 3
echo "📄 Initial output:"
echo "--------------------------------------------------------------------------------"
head -30 "$LOG_FILE"
echo "--------------------------------------------------------------------------------"
echo ""
echo "✅ Running in background. Use 'tail -f $LOG_FILE' to monitor."
