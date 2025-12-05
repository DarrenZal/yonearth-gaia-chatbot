# GraphRAG 2D Organic Views - Implementation Documentation

## Overview

2D organic visualization modes for the YonEarth Knowledge Graph featuring **Circle Packing** and **Voronoi Treemap** views.

**URL**: `https://gaiaai.xyz/YonEarth/graph/GraphRAG2D_OrganicViews.html`

## Deployment

### After Regenerating Clusters

**Always use the deployment script:**
```bash
./scripts/deploy_graphrag.sh
```

This script:
1. Deploys `graphrag_hierarchy.json` to the server
2. **Auto-generates `community_id_mapping.json`** from the hierarchy (critical for correct titles!)
3. Writes mapping to BOTH locations (primary + fallback)
4. Updates cache buster in JS file
5. Reloads nginx

### Key Files

| File | Location | Purpose |
|------|----------|---------|
| `graphrag_hierarchy.json` | `/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/` | Main cluster data |
| `community_id_mapping.json` | `/var/www/symbiocenelabs/YonEarth/graph/data/` (PRIMARY) | Cluster ID → Title mapping |
| `community_id_mapping.json` | `/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/` (FALLBACK) | Backup mapping |
| `GraphRAG3D_EmbeddingView.js` | `/var/www/symbiocenelabs/YonEarth/graph/` | Main visualization code |

### Common Issues

**Wrong cluster titles showing in voronoi view:**
- **Cause**: `community_id_mapping.json` out of sync with `graphrag_hierarchy.json`
- **Fix**: Run `./scripts/deploy_graphrag.sh` which regenerates the mapping automatically

**Browser showing old data after deployment:**
- **Fix**: Hard refresh (Ctrl+Shift+R) - the deploy script updates the cache buster

## Data Structure

### Hierarchy Levels
```
level_0 (26,219 entities) → level_1 (573 clusters) → level_2 (73 L2 clusters) → 3 mega-clusters
```

### Mega-clusters (shown in voronoi top level)
- `level_0_0`: **Regenerative Soil Health** (22 L2 children)
- `level_0_1`: **Sustainability & Regeneration** (27 L2 children)
- `level_0_20`: **Eco-Tech Fiction Nexus** (24 L2 children)

### ID Mapping Logic

The `getCommunityTitle()` function extracts numeric IDs:
- `level_0_0` → looks up `mapping["0"]` → "Regenerative Soil Health"
- `level_1_613` → looks up `mapping["613"]` → "Regenerative Economy Network"

## Visualization Modes

### Circle Packing
- Hierarchical circle layout with semantic zoom
- Click to zoom into clusters
- Color modes: by category OR entity count

### Voronoi Treemap
- Organic cell-like appearance
- 3 mega-clusters at top level
- Hover to reveal L2 sub-clusters inside each mega-cluster
- Click for cluster details

## Manual Deployment (if needed)

```bash
# Deploy JS files
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/graph/GraphRAG3D_EmbeddingView.js /var/www/symbiocenelabs/YonEarth/graph/

# Deploy data
sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/

# CRITICAL: Regenerate community_id_mapping.json (or just run deploy_graphrag.sh)
sudo systemctl reload nginx
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Wrong cluster names | `community_id_mapping.json` has old titles | Run `./scripts/deploy_graphrag.sh` |
| Old data after deploy | Browser cache | Hard refresh (Ctrl+Shift+R) |
| Visualization not rendering | JS error or data mismatch | Check browser console (F12) |
| Summaries not showing | Missing summary data | Verify `summaries_progress.json` exists |
