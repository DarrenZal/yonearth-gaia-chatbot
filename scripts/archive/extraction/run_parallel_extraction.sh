#!/bin/bash

# Load environment variables
source /home/claudeuser/yonearth-gaia-chatbot/.env

# Export the API key
export OPENAI_API_KEY

echo "Starting parallel extraction with API key: ${OPENAI_API_KEY:0:15}..."

# Run the parallel extraction with any passed arguments
python3 /home/claudeuser/yonearth-gaia-chatbot/scripts/extract_relationships_parallel.py "$@"