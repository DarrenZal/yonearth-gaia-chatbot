#!/bin/bash
# Wrapper script to run overnight extraction with proper environment

# Load environment variables
set -a
source /home/claudeuser/yonearth-gaia-chatbot/.env
set +a

# Run the extraction
cd /home/claudeuser/yonearth-gaia-chatbot
python3 scripts/overnight_fresh_extraction.py
