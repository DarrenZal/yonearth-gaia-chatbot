#!/bin/bash
cd /home/claudeuser/yonearth-gaia-chatbot
set -a  # Auto-export all variables
source .env
set +a
export AUTO_CONFIRM_LLM=yes
python3 scripts/build_proper_graphrag.py
