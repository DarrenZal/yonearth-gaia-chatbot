#!/bin/bash
# Deploy GraphRAG data to gaiaai.xyz
# Run this after regenerating clusters to ensure all files stay in sync

set -e  # Exit on error

SOURCE_DIR="/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy"
DEPLOY_DIR="/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy"
DEPLOY_DATA_DIR="/var/www/symbiocenelabs/YonEarth/graph/data"

echo "=== GraphRAG Deployment Script ==="
echo ""

# Step 1: Check source file exists
if [ ! -f "$SOURCE_DIR/graphrag_hierarchy.json" ]; then
    echo "ERROR: Source file not found: $SOURCE_DIR/graphrag_hierarchy.json"
    exit 1
fi

echo "1. Deploying graphrag_hierarchy.json..."
sudo cp "$SOURCE_DIR/graphrag_hierarchy.json" "$DEPLOY_DIR/graphrag_hierarchy.json"

# Step 2: Generate community_id_mapping.json from the hierarchy
echo "2. Generating community_id_mapping.json from hierarchy..."
sudo python3 << 'PYTHON_SCRIPT'
import json

# Load hierarchy
with open('/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/graphrag_hierarchy.json', 'r') as f:
    data = json.load(f)

clusters = data.get('clusters', {})
mapping = {}

# Add level_0 cluster titles (these are the 3 mega-clusters: level_0_0, level_0_1, level_0_20)
level1 = clusters.get('level_1', {})
for cluster_id, cluster_data in level1.items():
    if cluster_id.startswith('level_0_'):
        num = cluster_id.split('_')[-1]
        title = cluster_data.get('title', cluster_data.get('name', cluster_id))
        mapping[num] = title

# Add level_1 cluster titles (the 73 L2 clusters: level_1_574 through level_1_646)
level2 = clusters.get('level_2', {})
for cluster_id, cluster_data in level2.items():
    if cluster_id.startswith('level_1_'):
        num = cluster_id.split('_')[-1]
        title = cluster_data.get('title', cluster_data.get('name', cluster_id))
        mapping[num] = title

# Save to BOTH locations
for path in [
    '/var/www/symbiocenelabs/YonEarth/graph/data/community_id_mapping.json',
    '/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/community_id_mapping.json'
]:
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"   Written: {path}")

print(f"   Total mappings: {len(mapping)}")
PYTHON_SCRIPT

# Step 3: Deploy other layout files if they exist
echo "3. Deploying layout files (if present)..."
for file in graphsage_layout.json cluster_registry.json; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        sudo cp "$SOURCE_DIR/$file" "$DEPLOY_DIR/$file"
        echo "   Deployed: $file"
    fi
done

# Step 4: Update cache buster in JS file
echo "4. Updating cache buster..."
TIMESTAMP=$(date +%Y%m%d%H%M)
JS_FILE="/home/claudeuser/yonearth-gaia-chatbot/web/graph/GraphRAG3D_EmbeddingView.js"
sed -i "s/const cacheBuster = 'v=[^']*'/const cacheBuster = 'v=$TIMESTAMP'/" "$JS_FILE"
sudo cp "$JS_FILE" "/var/www/symbiocenelabs/YonEarth/graph/GraphRAG3D_EmbeddingView.js"
echo "   Cache buster updated to: v=$TIMESTAMP"

# Step 5: Reload nginx
echo "5. Reloading nginx..."
sudo systemctl reload nginx

echo ""
echo "=== Deployment Complete ==="
echo "Hard refresh the browser (Ctrl+Shift+R) to see changes."
