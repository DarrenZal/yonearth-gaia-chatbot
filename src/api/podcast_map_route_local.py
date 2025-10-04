"""
Podcast Map Visualization API Route - Local Data Version
Serves pre-exported data from local JSON file (for free Nomic plans)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api", tags=["podcast_map"])


class MapPoint(BaseModel):
    """Single data point in the visualization"""
    id: str = Field(..., description="Unique identifier for the point")
    text: str = Field(..., description="Text content of the chunk")
    x: float = Field(..., description="X coordinate in 2D space")
    y: float = Field(..., description="Y coordinate in 2D space")
    episode_id: str = Field(..., description="Episode identifier")
    episode_title: Optional[str] = Field(None, description="Episode title")
    timestamp: Optional[float] = Field(None, description="Timestamp in the audio")
    cluster: int = Field(..., description="Topic cluster ID")
    cluster_name: Optional[str] = Field(None, description="Human-readable cluster name")


class MapDataResponse(BaseModel):
    """Response containing all map data"""
    points: List[MapPoint] = Field(..., description="All data points for visualization")
    clusters: List[Dict[str, Any]] = Field(..., description="Cluster metadata")
    total_points: int = Field(..., description="Total number of points")
    episodes: List[Dict[str, Any]] = Field(..., description="Episode metadata")


class LocalDataLoader:
    """Loads map data from local JSON file"""

    def __init__(self, data_path: str = "data/processed/podcast_map_data.json"):
        self.data_path = data_path
        self._cached_data = None

    def load_data(self) -> Dict[str, Any]:
        """Load data from local JSON file"""
        if self._cached_data is not None:
            return self._cached_data

        data_file = os.path.join(os.path.dirname(__file__), "../..", self.data_path)

        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            raise FileNotFoundError(
                f"Map data file not found. Please export your Nomic data to {data_file}"
            )

        try:
            with open(data_file, 'r') as f:
                self._cached_data = json.load(f)
            logger.info(f"Loaded {len(self._cached_data['points'])} points from local file")
            return self._cached_data
        except Exception as e:
            logger.error(f"Error loading data file: {str(e)}")
            raise

    def get_clusters(self) -> List[Dict[str, Any]]:
        """Extract cluster information"""
        data = self.load_data()
        return data.get("clusters", [])

    def get_episodes(self) -> List[Dict[str, Any]]:
        """Extract episode information"""
        data = self.load_data()
        return data.get("episodes", [])


# Create global loader instance
data_loader = LocalDataLoader()
umap_data_loader = LocalDataLoader(data_path="data/processed/podcast_map_umap_data.json")
hierarchical_data_loader = LocalDataLoader(data_path="data/processed/podcast_map_hierarchical_data.json")
nomic_data_loader = LocalDataLoader(data_path="data/processed/nomic_projections.json")


@router.get("/map_data", response_model=MapDataResponse)
async def get_map_data(
    episode_filter: Optional[str] = Query(None, description="Filter by episode ID")
) -> MapDataResponse:
    """
    Fetch visualization data from local JSON file

    Args:
        episode_filter: Optional episode ID to filter data

    Returns:
        Complete map data for visualization
    """
    try:
        data = data_loader.load_data()

        # Apply episode filter if provided
        points = data["points"]
        if episode_filter:
            points = [p for p in points if p.get("episode_id") == episode_filter]

        return MapDataResponse(
            points=points,
            clusters=data.get("clusters", []),
            episodes=data.get("episodes", []),
            total_points=len(points)
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching map data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching visualization data: {str(e)}"
        )


@router.get("/map_data/episodes")
async def get_episodes() -> List[Dict[str, Any]]:
    """
    Get list of all available episodes

    Returns:
        List of episode metadata
    """
    try:
        return data_loader.get_episodes()
    except Exception as e:
        logger.error(f"Error fetching episodes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching episode list: {str(e)}"
        )


@router.get("/map_data/clusters")
async def get_clusters() -> List[Dict[str, Any]]:
    """
    Get cluster information

    Returns:
        List of cluster metadata with colors and names
    """
    try:
        return data_loader.get_clusters()
    except Exception as e:
        logger.error(f"Error fetching clusters: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching cluster information: {str(e)}"
        )


@router.get("/map_data_umap", response_model=MapDataResponse)
async def get_map_data_umap(
    episode_filter: Optional[str] = Query(None, description="Filter by episode ID")
) -> MapDataResponse:
    """
    Fetch UMAP visualization data from local JSON file

    Args:
        episode_filter: Optional episode ID to filter data

    Returns:
        Complete map data for UMAP visualization
    """
    try:
        data = umap_data_loader.load_data()

        # Apply episode filter if provided
        points = data["points"]
        if episode_filter:
            points = [p for p in points if p.get("episode_id") == episode_filter]

        return MapDataResponse(
            points=points,
            clusters=data.get("clusters", []),
            episodes=data.get("episodes", []),
            total_points=len(points)
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching UMAP map data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching UMAP visualization data: {str(e)}"
        )


@router.get("/map_data_hierarchical")
async def get_map_data_hierarchical(
    episode_filter: Optional[str] = Query(None, description="Filter by episode ID")
):
    """
    Fetch hierarchical visualization data from local JSON file
    Returns data with 3-level hierarchy (l1, l2, l3)

    Args:
        episode_filter: Optional episode ID to filter data

    Returns:
        Complete hierarchical map data for visualization
    """
    try:
        data = hierarchical_data_loader.load_data()

        # Apply episode filter if provided
        points = data["points"]
        if episode_filter:
            points = [p for p in points if p.get("episode_id") == episode_filter]

        return {
            "points": points,
            "episodes": data.get("episodes", []),
            "hierarchy": data.get("hierarchy", {}),
            "levels": data.get("levels", {}),
            "total_points": len(points),
            "generated_with": data.get("generated_with", "Hierarchical"),
            "default_level": data.get("default_level", "l2")
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching hierarchical map data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching hierarchical visualization data: {str(e)}"
        )


@router.get("/map_data_nomic")
async def get_map_data_nomic(
    episode_filter: Optional[str] = Query(None, description="Filter by episode ID")
):
    """
    Fetch Nomic Atlas projection data from local JSON file
    Returns data with Nomic's UMAP projections and topic assignments

    Args:
        episode_filter: Optional episode ID to filter data

    Returns:
        Complete Nomic map data for visualization
    """
    try:
        data = nomic_data_loader.load_data()

        # Apply episode filter if provided
        points = data["points"]
        if episode_filter:
            points = [p for p in points if p.get("episode_id") == episode_filter]

        return {
            "points": points,
            "metadata": data.get("metadata", {}),
            "total_points": len(points),
            "generated_with": "Nomic Atlas UMAP",
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching Nomic map data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching Nomic visualization data: {str(e)}"
        )


@router.post("/regenerate_umap")
async def regenerate_umap(
    n_points: int = Query(6000, ge=1000, le=10000, description="Number of data points to use"),
    min_dist: float = Query(0.1, ge=0.0, le=0.99, description="UMAP min_dist parameter"),
    n_neighbors: int = Query(15, ge=5, le=200, description="UMAP n_neighbors parameter")
):
    """
    Regenerate UMAP visualization with custom parameters

    Args:
        n_points: Number of vectors to fetch from Pinecone
        min_dist: UMAP minimum distance between points (0.0-0.99)
        n_neighbors: UMAP number of neighbors (5-200)

    Returns:
        Status message indicating generation has started
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import time

    try:
        # Run UMAP generation in background thread (it's CPU-intensive)
        def run_umap_generation():
            import subprocess
            import sys

            # Call the UMAP generation script with custom parameters
            script_path = "/home/claudeuser/yonearth-gaia-chatbot/scripts/archive/visualization/generate_map_umap_topics.py"

            # Run the script with environment variables to pass parameters
            env = os.environ.copy()
            env['MAX_VECTORS'] = str(n_points)
            env['UMAP_MIN_DIST'] = str(min_dist)
            env['UMAP_N_NEIGHBORS'] = str(n_neighbors)

            # Write start timestamp
            status_file = '/tmp/umap_generation_status.json'
            import json
            with open(status_file, 'w') as f:
                json.dump({
                    'status': 'running',
                    'start_time': time.time(),
                    'parameters': {
                        'n_points': n_points,
                        'min_dist': min_dist,
                        'n_neighbors': n_neighbors
                    }
                }, f)

            result = subprocess.run(
                [sys.executable, script_path],
                env=env,
                capture_output=True,
                text=True
            )

            # Write completion status
            with open(status_file, 'w') as f:
                json.dump({
                    'status': 'completed' if result.returncode == 0 else 'failed',
                    'end_time': time.time(),
                    'success': result.returncode == 0,
                    'stdout': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else '',
                    'parameters': {
                        'n_points': n_points,
                        'min_dist': min_dist,
                        'n_neighbors': n_neighbors
                    }
                }, f)

            return result.returncode == 0

        # Start generation in background
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()

        # Start the task but don't wait for it
        loop.run_in_executor(executor, run_umap_generation)

        logger.info(f"Starting UMAP generation with n_points={n_points}, min_dist={min_dist}, n_neighbors={n_neighbors}")

        return {
            "status": "started",
            "message": f"UMAP generation started with {n_points} points, min_dist={min_dist}, n_neighbors={n_neighbors}",
            "parameters": {
                "n_points": n_points,
                "min_dist": min_dist,
                "n_neighbors": n_neighbors
            }
        }

    except Exception as e:
        logger.error(f"Error starting UMAP generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting UMAP generation: {str(e)}"
        )


@router.get("/umap_generation_status")
async def get_umap_generation_status():
    """
    Check the status of UMAP generation

    Returns:
        Current generation status
    """
    import json
    status_file = '/tmp/umap_generation_status.json'

    try:
        if not os.path.exists(status_file):
            return {
                "status": "idle",
                "message": "No generation in progress"
            }

        with open(status_file, 'r') as f:
            status_data = json.load(f)

        return status_data

    except Exception as e:
        logger.error(f"Error reading UMAP generation status: {str(e)}")
        return {
            "status": "unknown",
            "error": str(e)
        }