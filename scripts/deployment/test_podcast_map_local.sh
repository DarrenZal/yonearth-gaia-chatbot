#!/bin/bash
# Test the podcast map visualization locally

echo "Starting test server on port 8001..."
cd /home/claudeuser/yonearth-gaia-chatbot
python3 simple_server.py 8001 &
SERVER_PID=$!

echo "Server started with PID $SERVER_PID"
echo "Waiting for server to initialize..."
sleep 3

echo ""
echo "Testing endpoints..."
echo "=================="

echo ""
echo "1. Testing /podcast-map page..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8001/podcast-map

echo ""
echo "2. Testing /api/map_data endpoint..."
curl -s http://localhost:8001/api/map_data | jq -r '.total_points, .clusters[].name' 2>/dev/null || echo "Error fetching map data"

echo ""
echo "3. Testing /api/map_data/episodes endpoint..."
curl -s http://localhost:8001/api/map_data/episodes | jq 'length' 2>/dev/null || echo "Error fetching episodes"

echo ""
echo "4. Testing /api/map_data/clusters endpoint..."
curl -s http://localhost:8001/api/map_data/clusters | jq -r '.[].name' 2>/dev/null || echo "Error fetching clusters"

echo ""
echo "=================="
echo "Test complete!"
echo ""
echo "Visit http://localhost:8001/podcast-map in your browser to see the visualization"
echo ""
echo "Press Enter to stop the test server..."
read

echo "Stopping server..."
kill $SERVER_PID
echo "Done!"
