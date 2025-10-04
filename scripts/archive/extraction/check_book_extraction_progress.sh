#!/bin/bash
# Book Knowledge Graph Extraction Progress Monitor

LOG_FILE="/home/claudeuser/yonearth-gaia-chatbot/logs/book_kg_extraction.log"
OUTPUT_DIR="/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/books"

echo "========================================="
echo "BOOK KNOWLEDGE GRAPH EXTRACTION PROGRESS"
echo "========================================="
echo ""

# Check if process is running
if pgrep -f "extract_books_knowledge_graph.py" > /dev/null; then
    echo "Status: RUNNING ✓"
else
    echo "Status: NOT RUNNING ✗"
fi

echo ""
echo "Latest Log Entries:"
echo "-------------------"
tail -10 "$LOG_FILE" 2>/dev/null || echo "No log file found"

echo ""
echo "Processing Statistics:"
echo "----------------------"

# Count completed chapters
COMPLETED_CHAPTERS=$(grep -c "Chapter.*complete:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Completed Chapters: $COMPLETED_CHAPTERS"

# Count completed books
COMPLETED_BOOKS=$(grep -c "Book complete:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Completed Books: $COMPLETED_BOOKS"

# Count total entities extracted
TOTAL_ENTITIES=$(grep "Book complete:" "$LOG_FILE" 2>/dev/null | grep -oP '\d+(?= total unique entities)' | awk '{s+=$1} END {print s}')
echo "Total Entities Extracted: ${TOTAL_ENTITIES:-0}"

# Count total relationships extracted
TOTAL_RELATIONSHIPS=$(grep "Book complete:" "$LOG_FILE" 2>/dev/null | grep -oP '\d+(?= total unique relationships)' | awk '{s+=$1} END {print s}')
echo "Total Relationships Extracted: ${TOTAL_RELATIONSHIPS:-0}"

echo ""
echo "Output Files:"
echo "-------------"
if [ -d "$OUTPUT_DIR" ]; then
    ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "No output files yet"
else
    echo "Output directory not found"
fi

echo ""
echo "To follow live progress:"
echo "  tail -f $LOG_FILE"
echo ""
