"""
GraphRAG API endpoint for fetching cluster data on demand.
Add this to your FastAPI application.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import json
import os

router = APIRouter(prefix="/api/graphrag", tags=["graphrag"])

# Cache the full hierarchy in memory
_hierarchy_cache = None

def load_hierarchy():
    """Load the full GraphRAG hierarchy into memory (once)."""
    global _hierarchy_cache
    if _hierarchy_cache is None:
        hierarchy_path = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"
        with open(hierarchy_path, 'r') as f:
            _hierarchy_cache = json.load(f)
    return _hierarchy_cache

@router.get("/cluster/{level}/{cluster_id}")
async def get_cluster(level: int, cluster_id: str):
    """
    Fetch a specific cluster with all its entities and relationships.

    Args:
        level: Cluster level (1, 2, or 3)
        cluster_id: Cluster ID (e.g., 'l2_16')

    Returns:
        Cluster data with entities, relationships, and metadata
    """
    try:
        hierarchy = load_hierarchy()
        level_key = f"level_{level}"

        if level_key not in hierarchy.get('clusters', {}):
            raise HTTPException(status_code=404, detail=f"Level {level} not found")

        clusters = hierarchy['clusters'][level_key]

        # Handle both dict and array formats
        if isinstance(clusters, dict):
            cluster = clusters.get(cluster_id)
        else:
            cluster = next((c for c in clusters if c.get('id') == cluster_id), None)

        if not cluster:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        # Get entity nodes for this cluster
        entity_ids = set(cluster.get('entities', []))
        entity_nodes = []

        # Load entity data from level_0
        level_0 = hierarchy['clusters'].get('level_0', {})

        # level_0 is a dict where keys are entity names
        for entity_name, entity_data in level_0.items():
            if entity_name in entity_ids:
                entity_nodes.append({
                    'id': entity_name,
                    'name': entity_name,
                    'type': entity_data.get('type', 'ENTITY'),
                    'position': entity_data.get('position') or entity_data.get('umap_position'),
                    'embedding': None  # Don't send full embeddings
                })

        # Get relationships between entities in this cluster
        relationships = []
        all_relationships = hierarchy.get('relationships', [])

        for rel in all_relationships:
            source = rel.get('source')
            target = rel.get('target')
            if source in entity_ids and target in entity_ids:
                relationships.append({
                    'source': source,
                    'target': target,
                    'type': rel.get('type', 'RELATED_TO'),
                    'weight': rel.get('weight', 1.0)
                })

        return {
            'cluster': {
                'id': cluster_id,
                'level': level,
                'name': cluster.get('name', cluster_id),
                'title': cluster.get('title', ''),
                'description': cluster.get('description', ''),
                'center': cluster.get('center', [0, 0, 0]),
                'size': len(entity_ids)
            },
            'entities': entity_nodes,
            'relationships': relationships
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_clusters(q: str, limit: int = 20):
    """
    Search clusters by name/description.
    Returns lightweight cluster info for autocomplete.
    """
    try:
        hierarchy = load_hierarchy()
        results = []

        query = q.lower()

        for level in [1, 2, 3]:
            level_key = f"level_{level}"
            clusters = hierarchy['clusters'].get(level_key, {})

            cluster_list = list(clusters.values()) if isinstance(clusters, dict) else clusters

            for cluster in cluster_list:
                name = cluster.get('name', cluster.get('id', ''))
                description = cluster.get('description', '')
                search_text = f"{name} {description}".lower()

                if query in search_text:
                    results.append({
                        'id': cluster.get('id'),
                        'name': name,
                        'description': description[:100],  # Truncate
                        'level': level,
                        'entity_count': len(cluster.get('entities', [])),
                        'center': cluster.get('center', [0, 0, 0])
                    })

                    if len(results) >= limit:
                        break

            if len(results) >= limit:
                break

        return {'results': results[:limit]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
