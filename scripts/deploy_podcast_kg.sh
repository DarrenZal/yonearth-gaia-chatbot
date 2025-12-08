#!/bin/bash
# Deploy podcast knowledge graph integration to production

set -e  # Exit on error

VERSION=$(date +%Y%m%d%H%M)  # Generate version from timestamp
DEV_DIR="/home/claudeuser/yonearth-gaia-chatbot"
PROD_DIR="/var/www/symbiocenelabs/YonEarth/podcast"

echo "🚀 Deploying Podcast Knowledge Graph Integration (v${VERSION})"
echo "=================================================="

# Step 1: Validate source files exist
echo "✓ Validating source files..."
[ -f "${DEV_DIR}/web/podcast/index.html" ] || { echo "❌ index.html not found"; exit 1; }
[ -f "${DEV_DIR}/web/podcast/PodcastMap3D.js" ] || { echo "❌ PodcastMap3D.js not found"; exit 1; }
[ -f "${DEV_DIR}/web/podcast/PodcastMap3D.css" ] || { echo "❌ PodcastMap3D.css not found"; exit 1; }

# Step 2: Update cache-busting version in index.html
echo "✓ Updating cache-busting version to ${VERSION}..."
sed -i "s/PodcastMap3D\\.js?v=[0-9]*/PodcastMap3D.js?v=${VERSION}/" "${DEV_DIR}/web/podcast/index.html"
sed -i "s/PodcastMap3D\\.css?v=[0-9]*/PodcastMap3D.css?v=${VERSION}/" "${DEV_DIR}/web/podcast/index.html"

# Step 3: Copy frontend files to production
echo "✓ Copying frontend files to production..."
sudo cp "${DEV_DIR}/web/podcast/index.html" "${PROD_DIR}/"
sudo cp "${DEV_DIR}/web/podcast/PodcastMap3D.js" "${PROD_DIR}/"
sudo cp "${DEV_DIR}/web/podcast/PodcastMap3D.css" "${PROD_DIR}/"

# Step 4: Set correct permissions
echo "✓ Setting file permissions..."
sudo chown www-data:www-data "${PROD_DIR}"/*.{html,js,css}
sudo chmod 644 "${PROD_DIR}"/*.{html,js,css}

# Step 5: Deploy backend API changes
echo "✓ Deploying backend API..."
if [ -f "${DEV_DIR}/src/api/graph_endpoints.py" ]; then
    # Check which backend location exists
    if [ -d "/root/yonearth-gaia-chatbot-migrated/src/api" ]; then
        sudo cp "${DEV_DIR}/src/api/graph_endpoints.py" /root/yonearth-gaia-chatbot-migrated/src/api/
        echo "✓ Backend API file copied to migrated location"
        sudo systemctl restart yonearth-api-migrated
        echo "✓ Backend API restarted"
    elif [ -d "/root/yonearth-gaia-chatbot/src/api" ]; then
        sudo cp "${DEV_DIR}/src/api/graph_endpoints.py" /root/yonearth-gaia-chatbot/src/api/
        echo "✓ Backend API file copied"
        sudo systemctl restart yonearth-api
        echo "✓ Backend API restarted"
    else
        echo "⚠️  Backend directory not found, skipping API deployment"
    fi
else
    echo "⚠️  graph_endpoints.py not found, skipping backend deployment"
fi

# Step 6: Reload nginx
echo "✓ Reloading nginx..."
sudo systemctl reload nginx

# Step 7: Verify deployment
echo "✓ Verifying deployment..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://gaiaai.xyz/YonEarth/podcast/)
if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Podcast visualizer is accessible (HTTP 200)"
else
    echo "❌ Podcast visualizer returned HTTP $HTTP_CODE"
fi

# Step 8: Test API endpoint
echo "✓ Testing API endpoint..."
API_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://gaiaai.xyz/api/graph/episode/5)
if [ "$API_CODE" = "200" ]; then
    echo "✓ Knowledge graph API is responding (HTTP 200)"
else
    echo "⚠️  Knowledge graph API returned HTTP $API_CODE"
fi

echo "=================================================="
echo "✅ Deployment complete! Version: ${VERSION}"
echo "🌐 URL: https://gaiaai.xyz/YonEarth/podcast/"
echo ""
echo "Next steps:"
echo "  1. Test entity pills display when selecting an episode"
echo "  2. Verify relationships load correctly"
echo "  3. Check mobile responsiveness"
echo "  4. Monitor API logs: sudo journalctl -u yonearth-api -f"
