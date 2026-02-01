"""
Graph inspection endpoints for entity neighborhood exploration
"""
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])

# Load unified graph data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/knowledge_graph_unified"

def load_graph_data():
    """Load unified graph and adjacency data"""
    try:
        with open(DATA_DIR / "unified.json", 'r') as f:
            unified = json.load(f)

        with open(DATA_DIR / "adjacency.json", 'r') as f:
            adjacency = json.load(f)

        return unified, adjacency
    except Exception as e:
        logger.error(f"Failed to load graph data: {e}")
        return None, None

# Cache loaded data
UNIFIED, ADJACENCY = load_graph_data()

@router.get("/entity/{entity_id}")
async def get_entity_neighborhood(
    entity_id: str,
    depth: int = Query(1, ge=1, le=3, description="Depth of neighborhood to retrieve"),
    max_neighbors: int = Query(50, ge=1, le=200, description="Maximum neighbors per level")
) -> Dict[str, Any]:
    """
    Get entity neighborhood from unified knowledge graph

    Args:
        entity_id: Entity name/ID to explore
        depth: How many hops to traverse (1-3)
        max_neighbors: Maximum neighbors per level
    """
    if not UNIFIED or not ADJACENCY:
        raise HTTPException(status_code=503, detail="Graph data not available")

    # Check if entity exists
    if entity_id not in UNIFIED.get('entities', {}):
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    # Get entity info
    entity_info = UNIFIED['entities'][entity_id]

    # Build neighborhood
    neighborhood = {
        "entity": {
            "id": entity_id,
            "type": entity_info.get('type'),
            "description": entity_info.get('description'),
            "sources": entity_info.get('sources', [])
        },
        "edges": [],
        "neighbors": {},
        "stats": {
            "depth": depth,
            "total_edges": 0,
            "unique_neighbors": 0,
            "predicates": set()
        }
    }

    # Traverse graph to specified depth
    visited = {entity_id}
    current_level = {entity_id}

    for level in range(1, depth + 1):
        next_level = set()

        for current_entity in current_level:
            # Get outgoing edges
            entity_edges = ADJACENCY.get(current_entity, {})

            for predicate, targets in entity_edges.items():
                # Limit neighbors per predicate
                for target in targets[:max_neighbors]:
                    if target not in visited:
                        next_level.add(target)
                        visited.add(target)

                        # Add edge
                        edge = {
                            "source": current_entity,
                            "predicate": predicate,
                            "target": target,
                            "level": level
                        }
                        neighborhood["edges"].append(edge)
                        neighborhood["stats"]["predicates"].add(predicate)

                        # Add neighbor info
                        if target in UNIFIED.get('entities', {}):
                            neighbor_info = UNIFIED['entities'][target]
                            neighborhood["neighbors"][target] = {
                                "type": neighbor_info.get('type'),
                                "description": neighbor_info.get('description', ''),
                                "level": level
                            }

            # Also check for incoming edges (where entity is target)
            for source_entity, predicates in ADJACENCY.items():
                if source_entity in visited:
                    continue

                for predicate, targets in predicates.items():
                    if entity_id in targets or current_entity in targets:
                        if source_entity not in visited:
                            next_level.add(source_entity)
                            visited.add(source_entity)

                            edge = {
                                "source": source_entity,
                                "predicate": predicate,
                                "target": current_entity,
                                "level": level,
                                "incoming": True
                            }
                            neighborhood["edges"].append(edge)
                            neighborhood["stats"]["predicates"].add(predicate)

                            if source_entity in UNIFIED.get('entities', {}):
                                neighbor_info = UNIFIED['entities'][source_entity]
                                neighborhood["neighbors"][source_entity] = {
                                    "type": neighbor_info.get('type'),
                                    "description": neighbor_info.get('description', ''),
                                    "level": level
                                }

        current_level = next_level

        # Stop if we've hit the max
        if len(neighborhood["edges"]) > max_neighbors * 3:
            break

    # Update stats
    neighborhood["stats"]["total_edges"] = len(neighborhood["edges"])
    neighborhood["stats"]["unique_neighbors"] = len(neighborhood["neighbors"])
    neighborhood["stats"]["predicates"] = list(neighborhood["stats"]["predicates"])

    return neighborhood

@router.get("/search")
async def search_entities(
    query: str = Query(..., min_length=2, description="Search query"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
) -> List[Dict[str, Any]]:
    """
    Search entities by name or description

    Args:
        query: Search string
        entity_type: Optional type filter (PERSON, ORGANIZATION, CONCEPT, etc.)
        limit: Maximum number of results
    """
    if not UNIFIED:
        raise HTTPException(status_code=503, detail="Graph data not available")

    results = []
    query_lower = query.lower()

    for entity_id, entity_info in UNIFIED.get('entities', {}).items():
        # Filter by type if specified
        if entity_type and entity_info.get('type') != entity_type.upper():
            continue

        # Search in entity name and description
        if (query_lower in entity_id.lower() or
            query_lower in entity_info.get('description', '').lower()):

            results.append({
                "id": entity_id,
                "type": entity_info.get('type'),
                "description": entity_info.get('description'),
                "sources": entity_info.get('sources', []),
                "match_score": 1.0 if query_lower in entity_id.lower() else 0.5
            })

        if len(results) >= limit:
            break

    # Sort by match score
    results.sort(key=lambda x: x['match_score'], reverse=True)

    return results

@router.get("/stats")
async def get_graph_stats() -> Dict[str, Any]:
    """Get statistics about the knowledge graph"""
    try:
        with open(DATA_DIR / "stats.json", 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        logger.error(f"Failed to load graph stats: {e}")
        raise HTTPException(status_code=503, detail="Stats not available")

@router.get("/predicates")
async def get_predicates() -> Dict[str, int]:
    """Get all predicates and their frequencies"""
    if not UNIFIED:
        raise HTTPException(status_code=503, detail="Graph data not available")

    predicate_counts = {}

    for rel in UNIFIED.get('relationships', []):
        pred = rel.get('predicate', 'unknown')
        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

    # Sort by frequency
    sorted_predicates = dict(sorted(predicate_counts.items(),
                                  key=lambda x: x[1],
                                  reverse=True))

    return sorted_predicates

@router.get("/types")
async def get_entity_types() -> Dict[str, int]:
    """Get all entity types and their frequencies"""
    if not UNIFIED:
        raise HTTPException(status_code=503, detail="Graph data not available")

    type_counts = {}

    for entity_info in UNIFIED.get('entities', {}).values():
        entity_type = entity_info.get('type', 'UNKNOWN')
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

    # Sort by frequency
    sorted_types = dict(sorted(type_counts.items(),
                              key=lambda x: x[1],
                              reverse=True))

    return sorted_types
