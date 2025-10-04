#!/bin/bash
# Quick progress check for knowledge graph extraction

echo "=== Knowledge Graph Extraction Progress ==="
echo ""
echo "Time: $(date)"
echo ""

# Check if process is running
if pgrep -f "process_knowledge_graph_episodes.py" > /dev/null; then
    echo "Status: ✓ RUNNING"
else
    echo "Status: ✗ NOT RUNNING"
fi

echo ""

# Count completed episodes
completed=$(ls -1 /home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/entities/episode_*_extraction.json 2>/dev/null | wc -l)
echo "Episodes Completed: $completed / 44"

# Show progress percentage
progress=$((completed * 100 / 44))
echo "Progress: $progress%"

echo ""

# Show last few log lines
echo "Recent Activity:"
tail -10 /tmp/kg_extraction_progress.log 2>/dev/null | grep -E "Processing Episode|Found.*entities|Extracted.*entities"

echo ""
echo "Full log: /tmp/kg_extraction_progress.log"
echo "Results dir: /home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/entities/"
