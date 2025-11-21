#!/bin/bash
cd /root/yonearth-gaia-chatbot

# Export environment variables
export $(grep -v '^#' .env | xargs)

# Run uvicorn with proper settings
exec uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 4
