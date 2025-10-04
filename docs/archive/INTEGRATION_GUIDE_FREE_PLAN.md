# Podcast Map Visualization - Free Plan Setup Guide

## TL;DR - Best Option for Free Plan

**You don't need Nomic at all!** Use your existing Pinecone data to generate the visualization.

## Quick Start (Recommended)

### Step 1: Install Additional Dependencies

```bash
# Activate your venv
source venv/bin/activate

# Install required packages
pip install scikit-learn  # For t-SNE and clustering
pip install numpy

# Update requirements
pip freeze > requirements.txt
```

### Step 2: Generate Map Data from Pinecone

```bash
# Run the generator script
python scripts/generate_map_from_pinecone.py
```

This will:
- ✅ Fetch all your podcast chunks from Pinecone (18,764+ vectors)
- ✅ Use t-SNE to reduce 1536D embeddings to 2D coordinates
- ✅ Cluster chunks into 7 topic groups using K-means
- ✅ Generate `data/processed/podcast_map_data.json`
- ✅ No Nomic required!
- ✅ No API limits!

**Time:** ~5-10 minutes depending on data size

### Step 3: Use Local Data Route

Edit your main application file to use the local data version:

```python
# In simple_server.py or src/api/main.py
from src.api import podcast_map_route_local  # Use local version

# Add the routes
app.include_router(podcast_map_route_local.router)
```

### Step 4: Install Frontend Dependencies

```bash
npm install d3 d3-delaunay
```

### Step 5: Add Route to Serve the Page

```python
# In your main app file
from fastapi.responses import FileResponse

@app.get("/podcast-map")
async def podcast_map_page():
    return FileResponse("web/PodcastMap.html")
```

### Step 6: Test It!

```bash
# Start your server
python scripts/start_local.py

# Visit
http://localhost:8000/podcast-map
```

## Architecture Comparison

### Option A: With Nomic (Requires Paid Plan)
```
User → FastAPI → Nomic API → 2D Coordinates → D3.js Visualization
                  💰 $20-50/mo
```

### Option B: With Pinecone (Free! You Already Have This)
```
User → FastAPI → Local JSON File → D3.js Visualization
                  ✅ One-time generation
                  ✅ No ongoing costs
```

## Detailed Steps

### Understanding Your Existing Data

You already have:
- ✅ 18,764+ vectors in Pinecone
- ✅ Episode metadata
- ✅ Text chunks with embeddings
- ✅ Everything needed for visualization!

The script does:
1. **Fetches** all vectors from Pinecone
2. **Reduces** 1536 dimensions → 2D using t-SNE
3. **Clusters** chunks into topic groups
4. **Exports** to JSON file

### Customizing Cluster Names

Edit `scripts/generate_map_from_pinecone.py`:

```python
CLUSTER_NAMES = [
    "Your Custom Topic 1",
    "Your Custom Topic 2",
    "Your Custom Topic 3",
    # ... etc
]
```

Run the script again to regenerate with new names.

### Updating the Data

When you add new episodes:

```bash
# Process new episodes
EPISODES_TO_PROCESS=172 python3 -m src.ingestion.process_episodes

# Regenerate map data
python scripts/generate_map_from_pinecone.py
```

## Troubleshooting

### "Can't connect to Pinecone"
- Check `.env` has `PINECONE_API_KEY`
- Verify `PINECONE_INDEX_NAME=yonearth-episodes`

### "t-SNE taking forever"
- Normal for large datasets
- Expected time: 5-10 minutes for 18k vectors
- You'll see progress output

### "Out of memory"
- Reduce batch size in script
- Or run on a machine with more RAM
- t-SNE needs ~2-4GB for 18k vectors

### "Clusters don't make sense"
- Try different `n_clusters` value (5-10 works well)
- Re-run the script - clustering has randomness
- Or manually assign clusters based on episode topics

## Advanced: Manual Clustering

If automatic clustering doesn't match your episodes well, you can manually assign clusters:

```python
# Create a CSV mapping
# data/episode_clusters.csv
episode_id,cluster_id,cluster_name
1,0,Regenerative Agriculture
2,0,Regenerative Agriculture
3,1,Climate Action
# etc...
```

Then modify the script to use this mapping instead of K-means.

## Comparison: Free vs Paid Nomic

| Feature | Pinecone (Free) | Nomic (Paid) |
|---------|----------------|--------------|
| Cost | ✅ Free | 💰 $20-50/mo |
| Setup time | ⏱️ 10 min | ⏱️ 5 min |
| Data freshness | Manual update | Real-time |
| Customization | ✅ Full control | Limited |
| API limits | ✅ None | Plan-based |
| Your data | ✅ Already have it | Need upload |

## When to Upgrade to Nomic

Consider Nomic if you need:
- Real-time data updates without regenerating
- Team collaboration on the visualization
- Image/multimodal datasets
- Their advanced exploration UI
- Retrieval/query API features

But for this visualization, **Pinecone is sufficient!**

## Summary

✅ Use `scripts/generate_map_from_pinecone.py`
✅ Use `src/api/podcast_map_route_local.py`
✅ No Nomic required
✅ No ongoing costs
✅ Works with your existing data

You're ready to go!