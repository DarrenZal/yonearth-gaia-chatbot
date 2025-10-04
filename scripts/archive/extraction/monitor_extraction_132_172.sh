#!/bin/bash

# Monitor extraction progress for episodes 132-172

TARGET_DIR="/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/entities"
LOG_FILE="/tmp/monitor_extraction_132_172.log"

echo "Monitoring extraction for episodes 132-172" | tee -a $LOG_FILE
echo "Started at: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

while true; do
    # Count completed episodes
    count=$(ls $TARGET_DIR/episode_*.json 2>/dev/null | grep -E "episode_1(3[2-9]|[4-6][0-9]|7[0-2])" | wc -l)
    
    # Get latest episode
    latest=$(ls -t $TARGET_DIR/episode_1*.json 2>/dev/null | grep -E "episode_1(3[2-9]|[4-6][0-9]|7[0-2])" | head -1 | grep -oE "[0-9]+" | head -1)
    
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Episodes 132-172 completed: $count/41 (Latest: $latest)" | tee -a $LOG_FILE
    
    # Check if process is still running
    if ! ps aux | grep "process_episodes_132_172.py" | grep -v grep > /dev/null; then
        echo "[$timestamp] Process completed or stopped!" | tee -a $LOG_FILE
        break
    fi
    
    # Check if all 41 episodes are done
    if [ "$count" -ge 41 ]; then
        echo "[$timestamp] All 41 episodes completed!" | tee -a $LOG_FILE
        break
    fi
    
    sleep 120  # Check every 2 minutes
done

echo "========================================" | tee -a $LOG_FILE
echo "Monitoring ended at: $(date)" | tee -a $LOG_FILE

# Generate final statistics
echo "" | tee -a $LOG_FILE
echo "FINAL STATISTICS:" | tee -a $LOG_FILE
echo "----------------" | tee -a $LOG_FILE

final_count=$(ls $TARGET_DIR/episode_*.json 2>/dev/null | grep -E "episode_1(3[2-9]|[4-6][0-9]|7[0-2])" | wc -l)
echo "Total episodes processed: $final_count/41" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "Processed episodes:" | tee -a $LOG_FILE
ls $TARGET_DIR/episode_*.json 2>/dev/null | grep -E "episode_1(3[2-9]|[4-6][0-9]|7[0-2])" | sort -V | xargs -I {} basename {} | tee -a $LOG_FILE
