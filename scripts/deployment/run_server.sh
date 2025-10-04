#!/bin/bash
cd /root/yonearth-gaia-chatbot

# Export environment variables
export $(grep -v '^#' .env | xargs)

# Run the server
exec python3 simple_server.py