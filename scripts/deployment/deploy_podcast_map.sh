#!/bin/bash
# Deploy podcast map visualization to production

echo "======================================"
echo "Deploying Podcast Map Visualization"
echo "======================================"

# Copy updated files to root directory
echo ""
echo "1. Copying updated server files..."
sudo cp /home/claudeuser/yonearth-gaia-chatbot/simple_server.py /root/yonearth-gaia-chatbot/simple_server.py
echo "✓ simple_server.py copied"

# Copy web files
echo ""
echo "2. Copying web interface files..."
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/PodcastMap.html /root/yonearth-gaia-chatbot/web/PodcastMap.html
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/PodcastMap.js /root/yonearth-gaia-chatbot/web/PodcastMap.js
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/PodcastMap.css /root/yonearth-gaia-chatbot/web/PodcastMap.css
echo "✓ Web files copied"

# Copy map data
echo ""
echo "3. Copying generated map data..."
sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/processed/podcast_map_data.json /root/yonearth-gaia-chatbot/data/processed/podcast_map_data.json
echo "✓ Map data copied"

# Restart service
echo ""
echo "4. Restarting service..."
sudo systemctl restart yonearth-gaia
sleep 3

# Check status
echo ""
echo "5. Checking service status..."
sudo systemctl status yonearth-gaia --no-pager | head -15

# Test endpoints
echo ""
echo "6. Testing endpoints..."
echo ""
echo "Testing /podcast-map page..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost/podcast-map

echo ""
echo "Testing /api/map_data endpoint..."
curl -s http://localhost/api/map_data | head -c 200
echo ""
echo "..."

echo ""
echo "======================================"
echo "✓ Deployment Complete!"
echo "======================================"
echo ""
echo "Visit: http://152.53.194.214/podcast-map"
echo ""
