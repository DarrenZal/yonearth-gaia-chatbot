#!/bin/bash
# Simple health check script for YonEarth Gaia server

URL="http://localhost/health"
LOG_FILE="/var/log/yonearth-monitor.log"

# Check if the server responds
if curl -f -s "$URL" > /dev/null; then
    echo "$(date): Server is healthy" >> "$LOG_FILE"
else
    echo "$(date): Server health check failed, restarting service" >> "$LOG_FILE"
    systemctl restart yonearth-gaia.service
fi