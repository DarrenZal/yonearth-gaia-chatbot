"""
Podcast Map Visualization API Route
Integrates with Nomic Atlas to provide map data for visualization
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import nomic
from nomic import AtlasDataset

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


class NomicAPIClient:
    """Client for interacting with Nomic Atlas API"""

    def __init__(self):
        self.api_key = os.getenv("NOMIC_API_KEY")
        self.project_id = os.getenv("NOMIC_PROJECT_ID")

        if not self.api_key:
            raise ValueError("NOMIC_API_KEY environment variable not set")
        if not self.project_id:
            raise ValueError("NOMIC_PROJECT_ID environment variable not set")

        # Login to Nomic
        nomic.login(self.api_key)

    def fetch_map_data(self) -> Dict[str, Any]:
        """
        Fetch the complete map dataset from Nomic Atlas

        Returns:
            Dictionary containing map data with points, embeddings, and metadata
        """
        try:
            # Access the Atlas dataset
            dataset = AtlasDataset(self.project_id)

            # Get the map data
            map_data = dataset.maps[0]  # Assuming first map

            # Get embeddings (2D coordinates)
            embeddings = map_data.embeddings.latent

            # Get metadata
            metadata = map_data.data.df

            # Combine into structured format
            points = []
            for idx, row in metadata.iterrows():
                point = {
                    "id": str(row.get("id", idx)),
                    "text": row.get("text", ""),
                    "x": float(embeddings[idx][0]),
                    "y": float(embeddings[idx][1]),
                    "episode_id": str(row.get("episode_id", "")),
                    "episode_title": row.get("episode_title"),
                    "timestamp": float(row.get("timestamp", 0)) if row.get("timestamp") else None,
                    "cluster": int(row.get("cluster", 0)),
                    "cluster_name": row.get("cluster_name")
                }
                points.append(point)

            # Get cluster information
            clusters = self._extract_clusters(metadata)

            # Get episode information
            episodes = self._extract_episodes(metadata)

            return {
                "points": points,
                "clusters": clusters,
                "episodes": episodes,
                "total_points": len(points)
            }

        except Exception as e:
            logger.error(f"Error fetching Nomic data: {str(e)}")
            raise

    def _extract_clusters(self, metadata) -> List[Dict[str, Any]]:
        """Extract unique cluster information from metadata"""
        clusters = []
        cluster_groups = metadata.groupby("cluster")

        # Define cluster colors (matching the original site)
        colors = [
            "#1f77b4",  # Media Evolution and Digital Democracy
            "#ff7f0e",  # Ethics and Responsible Innovation
            "#2ca02c",  # Scale, Time and Ecological Intelligence
            "#d62728",  # Organizational Design and Value Systems
            "#9467bd",  # Attention Economics and Behavioral Adaptation
            "#8c564b",  # Consciousness Transformation
            "#e377c2"   # Digital Literacy and Educational Evolution
        ]

        # Define cluster names (you can customize these based on your data)
        cluster_names = [
            "Media Evolution and Digital Democracy",
            "Ethics and Responsible Innovation",
            "Scale, Time and Ecological Intelligence",
            "Organizational Design and Value Systems",
            "Attention Economics and Behavioral Adaptation",
            "Consciousness Transformation and Contemplative Technologies",
            "Digital Literacy and Educational Evolution"
        ]

        for cluster_id, group in cluster_groups:
            cluster_info = {
                "id": int(cluster_id),
                "name": cluster_names[cluster_id % len(cluster_names)],
                "color": colors[cluster_id % len(colors)],
                "count": len(group)
            }
            clusters.append(cluster_info)

        return clusters

    def _extract_episodes(self, metadata) -> List[Dict[str, Any]]:
        """Extract unique episode information from metadata"""
        episodes = []
        episode_groups = metadata.groupby("episode_id")

        for episode_id, group in episode_groups:
            episode_info = {
                "id": str(episode_id),
                "title": group.iloc[0].get("episode_title", f"Episode {episode_id}"),
                "chunk_count": len(group),
                "audio_url": group.iloc[0].get("audio_url", "")  # If available
            }
            episodes.append(episode_info)

        return episodes


# Create global client instance
nomic_client = None


def get_nomic_client() -> NomicAPIClient:
    """Get or create Nomic client instance"""
    global nomic_client
    if nomic_client is None:
        nomic_client = NomicAPIClient()
    return nomic_client


@router.get("/map_data", response_model=MapDataResponse)
async def get_map_data(
    episode_filter: Optional[str] = Query(None, description="Filter by episode ID")
) -> MapDataResponse:
    """
    Fetch visualization data from Nomic Atlas

    Args:
        episode_filter: Optional episode ID to filter data

    Returns:
        Complete map data for visualization
    """
    try:
        client = get_nomic_client()
        data = client.fetch_map_data()

        # Apply episode filter if provided
        if episode_filter:
            data["points"] = [
                p for p in data["points"]
                if p["episode_id"] == episode_filter
            ]
            data["total_points"] = len(data["points"])

        return MapDataResponse(**data)

    except ValueError as ve:
        logger.error(f"Configuration error: {str(ve)}")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error. Please check environment variables."
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
        client = get_nomic_client()
        data = client.fetch_map_data()
        return data["episodes"]
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
        client = get_nomic_client()
        data = client.fetch_map_data()
        return data["clusters"]
    except Exception as e:
        logger.error(f"Error fetching clusters: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching cluster information: {str(e)}"
        )