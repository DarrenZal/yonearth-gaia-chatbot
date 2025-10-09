#!/bin/bash

echo "================================"
echo "EXTRACTION PROGRESS MONITOR"
echo "================================"
echo

while true; do
    count=$(ls data/knowledge_graph/relationships/episode_*_extraction.json 2>/dev/null | wc -l)
    process=$(ps aux | grep extract_all_relationships_comprehensive | grep -v grep | wc -l)

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Files: $count/172 - Process running: $process"

    if [ "$process" -eq 0 ]; then
        echo "Process completed or stopped!"
        break
    fi

    sleep 10
done

echo
echo "Final count: $count extraction files"
echo "================================"
