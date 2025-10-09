#!/bin/bash

# Load environment variables from .env file
set -a
source /home/claudeuser/yonearth-gaia-chatbot/.env
set +a

# Echo confirmation
echo "Environment loaded. Starting extraction..."

# Run with --yes flag to skip confirmation
python3 /home/claudeuser/yonearth-gaia-chatbot/scripts/extract_all_relationships_comprehensive.py --yes
