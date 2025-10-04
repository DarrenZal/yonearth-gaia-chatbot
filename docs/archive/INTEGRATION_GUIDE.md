# Podcast Map Visualization Integration Guide

This guide provides step-by-step instructions for integrating the new podcast map visualization feature into your existing YonEarth Gaia Chatbot application.

## Overview

The podcast map visualization replicates the functionality of askfuturefossils.com, providing:
- Interactive 2D visualization of podcast chunks using D3.js
- Voronoi diagram background showing topic clusters
- Episode trajectory visualization
- Audio playback at specific timestamps
- Hover tooltips and click interactions
- Responsive design matching the original aesthetic

## Prerequisites

- Python 3.8+ with activated virtual environment
- Node.js 14+ and npm
- Existing YonEarth Gaia Chatbot application
- Nomic Atlas account with API access

## Step 1: Configuration

### 1.1 Environment Variables

Add the following to your `.env` file:

```bash
# Nomic Atlas Configuration
NOMIC_API_KEY=your-nomic-api-key-here
NOMIC_PROJECT_ID=your-nomic-project-id-here
```

To get these values:
1. Sign up at https://atlas.nomic.ai/
2. Create a new project or use an existing one
3. Find your API key in account settings
4. Get the project ID from your project URL

## Step 2: Backend Integration

### 2.1 Install Python Dependencies

Activate your virtual environment and install new dependencies:

```bash
# Activate your venv (if not already active)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install new dependencies
pip install nomic

# Update requirements.txt
pip freeze > requirements.txt
```

### 2.2 Register the New Route

#### For FastAPI Applications

Edit your main FastAPI application file (likely `src/api/main.py` or `simple_server.py`):

```python
from fastapi import FastAPI
from src.api import podcast_map_route  # Import the new route module

app = FastAPI()

# Your existing routes...

# Add the podcast map routes
app.include_router(podcast_map_route.router)
```

#### For Flask Applications

If using Flask, adapt the route file slightly and register it:

```python
from flask import Flask
from src.api.podcast_map_route import get_map_data, get_episodes, get_clusters

app = Flask(__name__)

# Your existing routes...

# Add podcast map routes
@app.route('/api/map_data')
def map_data():
    return get_map_data()

@app.route('/api/map_data/episodes')
def map_episodes():
    return get_episodes()

@app.route('/api/map_data/clusters')
def map_clusters():
    return get_clusters()
```

## Step 3: Frontend Integration

### 3.1 Install Frontend Dependencies

```bash
# Install D3.js and D3-Delaunay
npm install d3 d3-delaunay

# This automatically updates package.json
```

### 3.2 Add Route to Your Frontend Router

The integration depends on your frontend framework:

#### For Plain HTML/JavaScript

If your application serves static HTML files, simply add a link to the new page:

```html
<!-- In your main index.html or navigation -->
<a href="/podcast-map">Podcast Map</a>
```

Configure your web server to serve the new page:

```python
# In your FastAPI/Flask app
from fastapi.responses import FileResponse

@app.get("/podcast-map")
async def podcast_map_page():
    return FileResponse("web/PodcastMap.html")
```

#### For React Applications

Create a wrapper component:

```jsx
// src/components/PodcastMap.jsx
import React, { useEffect } from 'react';

const PodcastMap = () => {
    useEffect(() => {
        // Load the visualization script
        const script = document.createElement('script');
        script.src = '/PodcastMap.js';
        document.body.appendChild(script);

        return () => {
            document.body.removeChild(script);
        };
    }, []);

    return (
        <div>
            <link rel="stylesheet" href="/PodcastMap.css" />
            <div id="podcast-map-container">
                <div id="map-svg-container"></div>
                {/* Include other HTML elements from PodcastMap.html */}
            </div>
        </div>
    );
};

export default PodcastMap;
```

Add to your router:

```jsx
import PodcastMap from './components/PodcastMap';

// In your router configuration
<Route path="/podcast-map" component={PodcastMap} />
```

#### For Vue Applications

Create a wrapper component:

```vue
<!-- src/components/PodcastMap.vue -->
<template>
    <div id="podcast-map-container">
        <div id="map-svg-container"></div>
        <!-- Include other HTML elements -->
    </div>
</template>

<script>
export default {
    name: 'PodcastMap',
    mounted() {
        // Load CSS
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = '/PodcastMap.css';
        document.head.appendChild(link);

        // Load D3 visualization
        const script = document.createElement('script');
        script.src = '/PodcastMap.js';
        document.body.appendChild(script);
    }
}
</script>
```

Add to your router:

```javascript
{
    path: '/podcast-map',
    name: 'PodcastMap',
    component: () => import('./components/PodcastMap.vue')
}
```

## Step 4: Static Assets Setup

### 4.1 Audio Files

Place your podcast audio files in a publicly accessible directory:

```bash
# Create audio directory
mkdir -p web/audio/episodes

# Place your audio files here with naming convention:
# ep-1.mp3, ep-2.mp3, etc.
# Or use episode IDs from your data
```

### 4.2 Icons (Optional)

If you want the magic/chat button, add an icon:

```bash
# Create assets directory
mkdir -p web/assets

# Add a magic icon SVG file
# web/assets/magic-icon.svg
```

### 4.3 Configure Static File Serving

#### FastAPI

```python
from fastapi.staticfiles import StaticFiles

# Mount static directories
app.mount("/audio", StaticFiles(directory="web/audio"), name="audio")
app.mount("/assets", StaticFiles(directory="web/assets"), name="assets")
app.mount("/", StaticFiles(directory="web"), name="web")
```

#### Flask

```python
from flask import send_from_directory

@app.route('/audio/<path:path>')
def serve_audio(path):
    return send_from_directory('web/audio', path)

@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory('web/assets', path)
```

## Step 5: Data Preparation

### 5.1 Upload Your Data to Nomic Atlas

1. Prepare your podcast transcript data in CSV or JSON format with columns:
   - `text`: The transcript chunk text
   - `episode_id`: Unique episode identifier
   - `episode_title`: Human-readable episode title
   - `timestamp`: Audio timestamp in seconds
   - `cluster`: Topic cluster ID (0-6)

2. Upload to Nomic Atlas:

```python
import nomic
import pandas as pd

# Login
nomic.login("your-api-key")

# Load your data
df = pd.read_csv("your_podcast_data.csv")

# Create Atlas dataset
dataset = nomic.AtlasDataset(
    name="YonEarth Podcast Episodes",
    description="Podcast transcript chunks for visualization"
)

# Add data
dataset.add_data(df)

# Create map with embeddings
dataset.create_index(
    name="podcast_map",
    indexed_field="text",
    modality="text"
)
```

### 5.2 Update Backend Configuration

Modify `src/api/podcast_map_route.py` if needed to match your data schema:

- Update cluster names in `_extract_clusters()` method
- Adjust field names in `fetch_map_data()` method
- Customize colors to match your branding

## Step 6: Testing

### 6.1 Run Backend Tests

```bash
# Test the API endpoint directly
curl http://localhost:8000/api/map_data

# Should return JSON with points, clusters, and episodes
```

### 6.2 Run Playwright Tests

```bash
# Install Playwright if not already installed
pip install playwright
playwright install chromium

# Run tests
python tests/test_podcast_map.py
```

### 6.3 Manual Testing

1. Start your development server:
   ```bash
   python scripts/start_local.py
   # or
   uvicorn src.api.main:app --reload
   ```

2. Navigate to http://localhost:8000/podcast-map

3. Verify:
   - Visualization loads with data points
   - Hover shows tooltips
   - Click plays audio
   - Episode selector filters the view
   - Responsive design works on different screen sizes

## Step 7: Production Deployment

### 7.1 Update Production Environment

```bash
# On your production server
cd /path/to/yonearth-gaia-chatbot

# Pull latest code
git pull

# Activate venv
source venv/bin/activate

# Install new dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart yonearth-gaia
```

### 7.2 Verify Deployment

```bash
# Check service status
sudo systemctl status yonearth-gaia

# Check logs
sudo journalctl -u yonearth-gaia -f

# Test the endpoint
curl https://your-domain.com/api/map_data
```

## Troubleshooting

### Common Issues

1. **"NOMIC_API_KEY environment variable not set"**
   - Ensure `.env` file contains the key
   - Restart your application after adding environment variables

2. **No data points appear in visualization**
   - Check browser console for errors
   - Verify API endpoint returns data: `curl localhost:8000/api/map_data`
   - Ensure Nomic project has data uploaded

3. **Audio doesn't play**
   - Verify audio files are in correct directory
   - Check file naming matches episode IDs
   - Ensure static file serving is configured

4. **Visualization doesn't render**
   - Check that D3.js loaded correctly
   - Verify SVG container exists in DOM
   - Look for JavaScript errors in console

5. **TypeError in Nomic client**
   - Ensure you're using compatible version: `pip install nomic==2.0.0`
   - Check that project ID is correct

### Debug Mode

Enable debug logging:

```python
# In podcast_map_route.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Customization Options

### Colors and Theming

Edit `web/PodcastMap.css`:
- Background color: Line 15 (`background-color: #242c46;`)
- Cluster colors: Update in both CSS and `podcast_map_route.py`

### Visualization Behavior

Edit `web/PodcastMap.js`:
- Point size: Line 246 (`attr('r', 3)`)
- Animation duration: Line 34 (`this.transitionDuration = 300`)
- Tooltip content: Lines 259-264

### API Response Format

Modify `src/api/podcast_map_route.py`:
- Add fields to `MapPoint` model
- Customize cluster information
- Filter or transform data before returning

## Performance Optimization

For large datasets (>10,000 points):

1. **Implement pagination**:
   ```python
   @router.get("/api/map_data")
   async def get_map_data(
       limit: int = Query(1000, le=5000),
       offset: int = Query(0, ge=0)
   ):
       # Return subset of data
   ```

2. **Add caching**:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1)
   def get_cached_map_data():
       # Cache the Nomic data
   ```

3. **Use WebGL for rendering** (for >5000 points):
   - Consider using deck.gl or three.js instead of SVG

## Support

For issues specific to:
- Nomic Atlas: https://docs.nomic.ai/
- D3.js: https://d3js.org/
- Your application: Check existing documentation in CLAUDE.md

## Next Steps

1. Customize cluster names and colors to match your content
2. Add search functionality to find specific topics
3. Implement filtering by date range or guest
4. Add export functionality for visualizations
5. Integrate with existing chat interface for context-aware responses

---

## Summary

You've successfully integrated a professional-grade podcast visualization that:
- ✅ Fetches data from Nomic Atlas API
- ✅ Renders interactive D3.js visualization
- ✅ Matches the design of askfuturefossils.com
- ✅ Supports audio playback at specific timestamps
- ✅ Shows episode trajectories
- ✅ Provides responsive design
- ✅ Includes comprehensive testing

The visualization is now ready for production use and can be extended with additional features as needed.