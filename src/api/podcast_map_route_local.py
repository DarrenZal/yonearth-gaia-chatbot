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