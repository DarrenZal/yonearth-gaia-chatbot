#!/bin/bash

cd /home/claudeuser/yonearth-gaia-chatbot
set -a
source .env
set +a
python3 scripts/extract_remaining_7_episodes.py
