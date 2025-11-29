#!/usr/bin/env python3
"""
Generate Voronoi 2 "Semantic Landscape" Layout using Node2Vec + UMAP.

This creates a distinct visualization where node positions are determined by
graph topology (structural embeddings) rather than semantic (text) embeddings.
Connected clusters naturally stay together because Node2Vec captures graph structure.

Pipeline:
1. Load GraphRAG hierarchy data (entities + relationships)
2. Build NetworkX graph from relationships
3. Generate Node2Vec structural embeddings (64-dim)
4. Reduce to 2D using UMAP (cosine metric)
5. Compute Voronoi tessellation from community centroids
6. Output voronoi_2_layout.json matching existing schema

Input:
  data/graphrag_hierarchy/graphrag_hierarchy.json

Output:
  data/graphrag_hierarchy/voronoi_2_layout.json
"""

import json
import os
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
from collections import defaultdict
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# --------------------------------------------------------------------------------------
# Paths & Parameters
# --------------------------------------------------------------------------------------

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
HIERARCHY_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
OUTPUT_PATH = ROOT / "data/graphrag_hierarchy/voronoi_2_layout.json"

# Node2Vec parameters
NODE2VEC_DIMENSIONS = 64
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_NUM_WALKS = 10
NODE2VEC_P = 1.0  # Return parameter (1 = balanced)
NODE2VEC_Q = 1.0  # In-out parameter (1 = balanced)

# UMAP parameters - relaxed for better spread
UMAP_N_NEIGHBORS = 50   # More global structure (was 15)
UMAP_MIN_DIST = 0.5     # Force nodes apart (was 0.1)
UMAP_SPREAD = 2.0       # Additional spread parameter
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42

# Number of CPU cores to use
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)


def load_hierarchy_data(path: Path) -> dict:
    """Load the existing GraphRAG hierarchy JSON."""
    print(f"Loading hierarchy data from {path}")
    with path.open() as f:
        data = json.load(f)

    entities = data.get('entities', {})
    relationships = data.get('relationships', [])
    clusters = data.get('clusters', {})

    print(f"  Loaded {len(entities):,} entities")
    print(f"  Loaded {len(relationships):,} relationships")
    print(f"  Cluster levels: {list(clusters.keys())}")

    return data


def build_graph(entities: dict, relationships: list) -> nx.Graph:
    """
    Build a NetworkX graph from entities and relationships.
    Nodes are entity names, edges are relationships.
    """
    print("\nBuilding NetworkX graph...")
    G = nx.Graph()

    # Add all entities as nodes
    for entity_name in entities.keys():
        G.add_node(entity_name)

    # Add relationships as edges
    edge_count = 0
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')

        if source and target and source in entities and target in entities:
            # Use relationship strength as edge weight if available
            weight = rel.get('strength', 1.0)
            if G.has_edge(source, target):
                # Increase weight for multiple relationships
                G[source][target]['weight'] += weight
            else:
                G.add_edge(source, target, weight=weight)
                edge_count += 1

    print(f"  Graph has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")

    # Handle disconnected components - connect them to largest component
    components = list(nx.connected_components(G))
    print(f"  Found {len(components)} connected components")

    if len(components) > 1:
        # Sort by size, largest first
        components = sorted(components, key=len, reverse=True)
        main_component = components[0]
        main_node = list(main_component)[0]

        # Connect small components to main component
        for component in components[1:]:
            comp_node = list(component)[0]
            G.add_edge(main_node, comp_node, weight=0.1)

        print(f"  Connected {len(components)-1} isolated components to main graph")

    return G


def generate_node2vec_embeddings(G: nx.Graph) -> Tuple[np.ndarray, List[str]]:
    """
    Generate Node2Vec structural embeddings for all nodes.

    Node2Vec learns embeddings that capture graph topology by performing
    biased random walks and using skip-gram to learn node representations.
    Connected nodes will have similar embeddings.
    """
    print(f"\nGenerating Node2Vec embeddings (dim={NODE2VEC_DIMENSIONS})...")
    print(f"  Walk length: {NODE2VEC_WALK_LENGTH}, Num walks: {NODE2VEC_NUM_WALKS}")
    print(f"  Using {NUM_WORKERS} CPU cores")

    # Get ordered list of nodes
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Try PecanPy first (faster C implementation), fall back to node2vec
    use_pecanpy = False
    use_node2vec = False

    try:
        from pecanpy import pecanpy
        print("  Using PecanPy 2.x (fast implementation)")
        use_pecanpy = True
    except ImportError:
        try:
            from node2vec import Node2Vec
            print("  Using node2vec library")
            use_node2vec = True
        except ImportError:
            print("  WARNING: Neither pecanpy nor node2vec installed!")
            print("  Install with: pip install pecanpy  OR  pip install node2vec")
            print("  Falling back to simple random walk embedding...")
            return generate_simple_walk_embeddings(G, node_list)

    if use_pecanpy:
        # PecanPy 2.x API: requires edge list format
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.edg', delete=False) as f:
            edgelist_path = f.name
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1.0)
                # Write node indices for pecanpy
                f.write(f"{node_to_idx[u]}\t{node_to_idx[v]}\t{weight}\n")

        try:
            # PecanPy 2.x API - use SparseOTF for large graphs
            print(f"  Loading edge list from {edgelist_path}...")

            # Create SparseOTF graph object
            g = pecanpy.SparseOTF(
                p=NODE2VEC_P,
                q=NODE2VEC_Q,
                workers=NUM_WORKERS,
                verbose=True
            )

            # Read the edge list
            g.read_edg(edgelist_path, weighted=True, directed=False)

            print(f"  Running Node2Vec random walks...")
            # Generate embeddings using PecanPy 2.x embed method
            embeddings = g.embed(
                dim=NODE2VEC_DIMENSIONS,
                num_walks=NODE2VEC_NUM_WALKS,
                walk_length=NODE2VEC_WALK_LENGTH
            )

            # PecanPy returns embeddings indexed by node ID (integer)
            # Map back to original node order
            embedding_matrix = np.zeros((len(node_list), NODE2VEC_DIMENSIONS))
            for idx in range(len(node_list)):
                if idx < len(embeddings):
                    embedding_matrix[idx] = embeddings[idx]

        except Exception as e:
            print(f"  PecanPy error: {e}")
            print("  Falling back to simple random walk embedding...")
            os.unlink(edgelist_path)
            return generate_simple_walk_embeddings(G, node_list)
        finally:
            if os.path.exists(edgelist_path):
                os.unlink(edgelist_path)

    elif use_node2vec:
        # Use node2vec library
        node2vec = Node2Vec(
            G,
            dimensions=NODE2VEC_DIMENSIONS,
            walk_length=NODE2VEC_WALK_LENGTH,
            num_walks=NODE2VEC_NUM_WALKS,
            p=NODE2VEC_P,
            q=NODE2VEC_Q,
            workers=NUM_WORKERS,
            quiet=False
        )

        # Learn embeddings
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Extract embeddings in node order
        embedding_matrix = np.zeros((len(node_list), NODE2VEC_DIMENSIONS))
        for idx, node in enumerate(node_list):
            if node in model.wv:
                embedding_matrix[idx] = model.wv[node]

    print(f"  Generated embeddings shape: {embedding_matrix.shape}")
    return embedding_matrix, node_list


def generate_simple_walk_embeddings(G: nx.Graph, node_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Fallback: Simple random walk + SVD embedding when Node2Vec not available.
    Less sophisticated but captures basic graph structure.
    """
    import random
    from sklearn.decomposition import TruncatedSVD

    print("  Using simple random walk fallback...")

    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    n_nodes = len(node_list)

    # Build co-occurrence matrix from random walks
    cooccurrence = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    n_walks = 5
    walk_length = 20
    window_size = 5

    for _ in range(n_walks):
        for start_node in node_list:
            walk = [start_node]
            current = start_node

            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)

            # Update co-occurrence within window
            for i, node in enumerate(walk):
                node_idx = node_to_idx[node]
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        other_idx = node_to_idx[walk[j]]
                        cooccurrence[node_idx, other_idx] += 1

    # Apply SVD to get embeddings
    print(f"  Applying SVD to {cooccurrence.shape} co-occurrence matrix...")
    svd = TruncatedSVD(n_components=NODE2VEC_DIMENSIONS, random_state=42)
    embeddings = svd.fit_transform(cooccurrence)

    print(f"  Generated embeddings shape: {embeddings.shape}")
    return embeddings, node_list


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce Node2Vec embeddings to 2D using UMAP.
    Uses cosine metric and tight min_dist for clear cluster boundaries.
    """
    print(f"\nReducing {embeddings.shape[1]}D embeddings to 2D with UMAP...")
    print(f"  n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, spread={UMAP_SPREAD}, metric={UMAP_METRIC}")

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True
    )

    coords_2d = reducer.fit_transform(embeddings)

    # Normalize to [0, 1] range for easier visualization
    coords_2d[:, 0] = (coords_2d[:, 0] - coords_2d[:, 0].min()) / (coords_2d[:, 0].max() - coords_2d[:, 0].min())
    coords_2d[:, 1] = (coords_2d[:, 1] - coords_2d[:, 1].min()) / (coords_2d[:, 1].max() - coords_2d[:, 1].min())

    print(f"  2D coordinates range: x=[{coords_2d[:, 0].min():.3f}, {coords_2d[:, 0].max():.3f}], y=[{coords_2d[:, 1].min():.3f}, {coords_2d[:, 1].max():.3f}]")

    return coords_2d


def compute_community_centroids(
    clusters: dict,
    node_positions: Dict[str, Tuple[float, float]],
    level: str = 'level_1'
) -> Dict[str, Dict]:
    """
    Calculate centroid for each community based on member node positions.
    Returns dict with centroid, member count, and polygon placeholder.
    """
    print(f"\nComputing centroids for {level} communities...")

    level_clusters = clusters.get(level, {})
    centroids = {}

    for cluster_id, cluster_data in level_clusters.items():
        # Get member entities
        members = cluster_data.get('children', [])
        if not members:
            members = cluster_data.get('entities', [])

        # Collect positions of members that exist in our node_positions
        positions = []
        for member in members:
            if member in node_positions:
                positions.append(node_positions[member])

        if positions:
            positions = np.array(positions)
            centroid = positions.mean(axis=0)

            centroids[cluster_id] = {
                'id': cluster_id,
                'name': cluster_data.get('name', cluster_data.get('title', cluster_id)),
                'title': cluster_data.get('title', cluster_data.get('name', cluster_id)),
                'centroid': centroid.tolist(),
                'member_count': len(positions),
                'total_entities': len(members)
            }

    print(f"  Computed {len(centroids)} centroids")
    return centroids


def generate_voronoi_polygons(
    centroids: Dict[str, Dict],
    bounds: Tuple[float, float, float, float] = (0, 0, 1, 1)
) -> Dict[str, List[List[float]]]:
    """
    Generate Voronoi polygons from community centroids.
    Clips to specified bounds and handles edge cases.
    """
    print("\nGenerating Voronoi tessellation...")

    if len(centroids) < 4:
        print("  Warning: Too few centroids for Voronoi, using simple rectangles")
        return {cid: [] for cid in centroids.keys()}

    # Extract centroid points
    cluster_ids = list(centroids.keys())
    points = np.array([centroids[cid]['centroid'] for cid in cluster_ids])

    # Add boundary points to ensure all cells are finite
    min_x, min_y, max_x, max_y = bounds
    margin = 2.0  # Large margin to capture edge cells
    boundary_points = np.array([
        [min_x - margin, min_y - margin],
        [max_x + margin, min_y - margin],
        [max_x + margin, max_y + margin],
        [min_x - margin, max_y + margin],
        [(min_x + max_x) / 2, min_y - margin],
        [(min_x + max_x) / 2, max_y + margin],
        [min_x - margin, (min_y + max_y) / 2],
        [max_x + margin, (min_y + max_y) / 2],
    ])

    all_points = np.vstack([points, boundary_points])

    # Compute Voronoi diagram
    vor = Voronoi(all_points)

    # Extract polygons for each cluster (excluding boundary points)
    polygons = {}
    for idx, cluster_id in enumerate(cluster_ids):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            # Infinite region, create bounding polygon
            polygons[cluster_id] = []
            continue

        # Get polygon vertices
        polygon = [vor.vertices[i].tolist() for i in region]

        # Clip to bounds
        clipped = clip_polygon_to_bounds(polygon, bounds)
        polygons[cluster_id] = clipped

    valid_polygons = sum(1 for p in polygons.values() if len(p) > 2)
    print(f"  Generated {valid_polygons} valid Voronoi polygons")

    return polygons


def generate_l2_polygons_from_entity_voronoi(
    node_positions: Dict[str, Tuple[float, float]],
    clusters: dict,
    bounds: Tuple[float, float, float, float] = (0, 0, 1, 1)
) -> Dict[str, List[List[float]]]:
    """
    Generate L2 cluster polygons by computing Voronoi cells for ALL entities,
    then merging cells that belong to the same L2 cluster.

    This approach ensures polygons cover the full visualization area (0-1 range)
    rather than just clustering around centroids.

    Args:
        node_positions: Dict mapping entity names to (x, y) positions
        clusters: Original hierarchy clusters dict
        bounds: Bounding box for clipping polygons

    Returns:
        Dict mapping L2 cluster IDs to merged polygon coordinates
    """
    print("\nGenerating L2 polygons from entity-based Voronoi tessellation...")

    # Build entity -> L2 cluster mapping
    # L2 clusters (level_2) contain entity names in 'children' or 'entities' field
    level_2 = clusters.get('level_2', {})
    entity_to_l2 = {}
    for l2_id, l2_data in level_2.items():
        members = l2_data.get('children', []) or l2_data.get('entities', [])
        for entity in members:
            if entity in node_positions:
                entity_to_l2[entity] = l2_id

    print(f"  Mapped {len(entity_to_l2)} entities to L2 clusters")

    if len(entity_to_l2) < 4:
        print("  Warning: Too few entities for Voronoi")
        return {l2_id: [] for l2_id in level_2.keys()}

    # Get list of entities that have positions and L2 membership
    entity_list = list(entity_to_l2.keys())
    points = np.array([node_positions[e] for e in entity_list])

    # Add boundary points to ensure all Voronoi cells are finite
    min_x, min_y, max_x, max_y = bounds
    margin = 2.0
    boundary_points = np.array([
        [min_x - margin, min_y - margin],
        [max_x + margin, min_y - margin],
        [max_x + margin, max_y + margin],
        [min_x - margin, max_y + margin],
        [(min_x + max_x) / 2, min_y - margin],
        [(min_x + max_x) / 2, max_y + margin],
        [min_x - margin, (min_y + max_y) / 2],
        [max_x + margin, (min_y + max_y) / 2],
    ])

    all_points = np.vstack([points, boundary_points])

    # Compute Voronoi diagram
    print(f"  Computing Voronoi for {len(entity_list)} entities...")
    vor = Voronoi(all_points)

    # Extract polygons for each entity
    entity_polygons = {}
    for idx, entity in enumerate(entity_list):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            continue

        # Get polygon vertices
        polygon = [vor.vertices[i].tolist() for i in region]

        # Clip to bounds
        clipped = clip_polygon_to_bounds(polygon, bounds)
        if len(clipped) >= 3:
            entity_polygons[entity] = clipped

    print(f"  Generated {len(entity_polygons)} valid entity Voronoi cells")

    # Group polygons by L2 cluster and merge using unary_union
    l2_child_polygons = defaultdict(list)
    for entity, polygon_coords in entity_polygons.items():
        l2_id = entity_to_l2.get(entity)
        if l2_id:
            try:
                shapely_poly = ShapelyPolygon(polygon_coords)
                if shapely_poly.is_valid:
                    l2_child_polygons[l2_id].append(shapely_poly)
                else:
                    fixed = shapely_poly.buffer(0)
                    if fixed.is_valid and not fixed.is_empty:
                        l2_child_polygons[l2_id].append(fixed)
            except Exception:
                pass

    print(f"  Merging polygons for {len(l2_child_polygons)} L2 clusters...")

    # Merge entity polygons into L2 cluster polygons
    l2_polygons = {}
    for l2_id, child_polys in l2_child_polygons.items():
        if not child_polys:
            l2_polygons[l2_id] = []
            continue

        try:
            merged = unary_union(child_polys)

            if merged.is_empty:
                l2_polygons[l2_id] = []
            elif merged.geom_type == 'Polygon':
                coords = list(merged.exterior.coords)
                l2_polygons[l2_id] = [[float(x), float(y)] for x, y in coords]
            elif merged.geom_type == 'MultiPolygon':
                # Take the largest polygon
                largest = max(merged.geoms, key=lambda p: p.area)
                coords = list(largest.exterior.coords)
                l2_polygons[l2_id] = [[float(x), float(y)] for x, y in coords]
            else:
                l2_polygons[l2_id] = []
        except Exception as e:
            print(f"    Error merging {l2_id}: {e}")
            l2_polygons[l2_id] = []

    # Include L2 clusters with no entities
    for l2_id in level_2.keys():
        if l2_id not in l2_polygons:
            l2_polygons[l2_id] = []

    valid_count = sum(1 for p in l2_polygons.values() if len(p) > 2)
    print(f"  Generated {valid_count} valid L2 cluster polygons via entity union")

    return l2_polygons


def clip_polygon_to_bounds(polygon: List[List[float]], bounds: Tuple[float, float, float, float]) -> List[List[float]]:
    """
    Clip a polygon to rectangular bounds using Sutherland-Hodgman algorithm.
    """
    if not polygon:
        return []

    min_x, min_y, max_x, max_y = bounds

    def inside(p, edge):
        x, y = p
        if edge == 'left':
            return x >= min_x
        elif edge == 'right':
            return x <= max_x
        elif edge == 'bottom':
            return y >= min_y
        elif edge == 'top':
            return y <= max_y

    def intersect(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2

        if edge == 'left':
            t = (min_x - x1) / (x2 - x1) if x2 != x1 else 0
            return [min_x, y1 + t * (y2 - y1)]
        elif edge == 'right':
            t = (max_x - x1) / (x2 - x1) if x2 != x1 else 0
            return [max_x, y1 + t * (y2 - y1)]
        elif edge == 'bottom':
            t = (min_y - y1) / (y2 - y1) if y2 != y1 else 0
            return [x1 + t * (x2 - x1), min_y]
        elif edge == 'top':
            t = (max_y - y1) / (y2 - y1) if y2 != y1 else 0
            return [x1 + t * (x2 - x1), max_y]

    output = polygon
    for edge in ['left', 'right', 'bottom', 'top']:
        if not output:
            break

        input_poly = output
        output = []

        for i in range(len(input_poly)):
            current = input_poly[i]
            next_p = input_poly[(i + 1) % len(input_poly)]

            if inside(current, edge):
                if inside(next_p, edge):
                    output.append(next_p)
                else:
                    output.append(intersect(current, next_p, edge))
            elif inside(next_p, edge):
                output.append(intersect(current, next_p, edge))
                output.append(next_p)

    return output


def compute_l3_centroids_from_l2(
    l2_centroids: Dict[str, Dict],
    clusters: dict
) -> Dict[str, Dict]:
    """
    Compute L3 (top-level) centroids based on the centroids of their L2 children.

    L3 clusters in the hierarchy have 'children' arrays containing L2 cluster IDs,
    not entity names. So we compute L3 centroids as the mean of child L2 centroids.

    Args:
        l2_centroids: Dict mapping L2 cluster IDs to their centroid data
        clusters: Original hierarchy clusters dict

    Returns:
        Dict mapping L3 cluster IDs to their centroid data
    """
    print("\nComputing L3 centroids from L2 child centroids...")

    l3_clusters = clusters.get('level_3', {})
    l3_centroids = {}

    for l3_id, l3_data in l3_clusters.items():
        children = l3_data.get('children', [])

        # Collect centroids of L2 children that exist
        child_centroids = []
        for child_id in children:
            if child_id in l2_centroids:
                centroid = l2_centroids[child_id].get('centroid')
                if centroid:
                    child_centroids.append(centroid)

        if child_centroids:
            # Compute mean of child centroids
            centroids_arr = np.array(child_centroids)
            mean_centroid = centroids_arr.mean(axis=0).tolist()

            l3_centroids[l3_id] = {
                'id': l3_id,
                'name': l3_data.get('name', l3_data.get('title', l3_id)),
                'title': l3_data.get('title', l3_data.get('name', l3_id)),
                'centroid': mean_centroid,
                'member_count': len(child_centroids),
                'total_children': len(children)
            }

    print(f"  Computed {len(l3_centroids)} L3 centroids from L2 children")
    return l3_centroids


def generate_l3_polygons_from_l2_union(
    l2_polygons: Dict[str, List[List[float]]],
    clusters: dict
) -> Dict[str, List[List[float]]]:
    """
    Generate L3 (top-level) polygons by merging child L2 polygons using Shapely unary_union.

    This approach ensures:
    - No overlapping regions between L3 clusters
    - Perfect containment (L3 polygon exactly covers its L2 children)
    - Clean, non-self-intersecting boundaries

    Args:
        l2_polygons: Dict mapping L2 cluster IDs (level_1_*) to polygon coordinates
        clusters: Original hierarchy clusters dict containing level_3 parent-child info

    Returns:
        Dict mapping L3 cluster IDs (level_2_*) to merged polygon coordinates
    """
    print("\nGenerating L3 polygons via geometric union of L2 children...")

    # Get L3 cluster data from hierarchy (these are named level_2_* in the hierarchy)
    l3_clusters = clusters.get('level_3', {})

    # Build reverse mapping: L2 cluster ID -> parent L3 cluster ID
    # L3 clusters in hierarchy have 'children' arrays with L2 IDs (level_1_*)
    l2_to_l3_parent = {}
    for l3_id, l3_data in l3_clusters.items():
        children = l3_data.get('children', [])
        for child_id in children:
            l2_to_l3_parent[child_id] = l3_id

    print(f"  Found {len(l2_to_l3_parent)} L2->L3 parent mappings")

    # Group L2 polygons by their L3 parent
    l3_child_polygons = defaultdict(list)
    for l2_id, polygon_coords in l2_polygons.items():
        if len(polygon_coords) < 3:
            continue  # Skip invalid polygons

        parent_l3 = l2_to_l3_parent.get(l2_id)
        if parent_l3:
            try:
                shapely_poly = ShapelyPolygon(polygon_coords)
                if shapely_poly.is_valid:
                    l3_child_polygons[parent_l3].append(shapely_poly)
                else:
                    # Try to fix invalid polygon
                    fixed_poly = shapely_poly.buffer(0)
                    if fixed_poly.is_valid and not fixed_poly.is_empty:
                        l3_child_polygons[parent_l3].append(fixed_poly)
            except Exception as e:
                print(f"    Warning: Could not create polygon for {l2_id}: {e}")

    print(f"  L3 clusters with valid child polygons: {len(l3_child_polygons)}")

    # Compute union for each L3 cluster
    l3_polygons = {}
    for l3_id, child_polys in l3_child_polygons.items():
        if not child_polys:
            l3_polygons[l3_id] = []
            continue

        try:
            # Merge all child polygons into one
            merged = unary_union(child_polys)

            # Handle different geometry types that can result from union
            if merged.is_empty:
                l3_polygons[l3_id] = []
            elif merged.geom_type == 'Polygon':
                # Single polygon - extract exterior coordinates
                coords = list(merged.exterior.coords)
                l3_polygons[l3_id] = [[float(x), float(y)] for x, y in coords]
            elif merged.geom_type == 'MultiPolygon':
                # Multiple disconnected polygons - take the largest one
                largest = max(merged.geoms, key=lambda p: p.area)
                coords = list(largest.exterior.coords)
                l3_polygons[l3_id] = [[float(x), float(y)] for x, y in coords]
            else:
                # Unexpected geometry type
                l3_polygons[l3_id] = []
                print(f"    Warning: Unexpected geometry type for {l3_id}: {merged.geom_type}")
        except Exception as e:
            print(f"    Error computing union for {l3_id}: {e}")
            l3_polygons[l3_id] = []

    # Also include L3 clusters that have no children (empty polygons)
    for l3_id in l3_clusters.keys():
        if l3_id not in l3_polygons:
            l3_polygons[l3_id] = []

    valid_count = sum(1 for p in l3_polygons.values() if len(p) > 2)
    print(f"  Generated {valid_count} valid L3 polygons via union")

    return l3_polygons


def build_voronoi2_layout(
    hierarchy_data: dict,
    node_positions: Dict[str, Tuple[float, float]],
    node_list: List[str]
) -> dict:
    """
    Build the complete Voronoi 2 layout JSON matching the existing schema.
    """
    print("\nBuilding Voronoi 2 layout data structure...")

    clusters = hierarchy_data.get('clusters', {})
    entities = hierarchy_data.get('entities', {})

    # Compute centroids for each cluster level
    l1_centroids = compute_community_centroids(clusters, node_positions, 'level_1')
    l2_centroids = compute_community_centroids(clusters, node_positions, 'level_2')

    # L3 centroids are computed from L2 centroids (not entity positions)
    # because L3 children are L2 cluster IDs, not entity names
    l3_centroids = compute_l3_centroids_from_l2(l2_centroids, clusters)

    # Generate L2 polygons using entity-based Voronoi tessellation
    # This ensures polygons cover the full visualization area (0-1 range)
    # rather than clustering around centroids
    l2_polygons = generate_l2_polygons_from_entity_voronoi(node_positions, clusters)

    # Generate L3 polygons via geometric union of L2 child polygons (NOT Voronoi)
    # This ensures L3 shapes exactly cover their L2 children without overlaps
    l3_polygons = generate_l3_polygons_from_l2_union(l2_polygons, clusters)

    # Build L2->L3 parent relationships based on hierarchy
    # Hierarchy: level_3 clusters have 'children' arrays with level_1_* IDs (which are our L2)
    print("\nBuilding L2->L3 parent relationships...")
    hierarchy_l3 = clusters.get('level_3', {})

    # Map from L2 cluster ID to parent L3 cluster ID
    l2_to_l3_parent = {}
    # Map from L3 cluster ID to list of child L2 cluster IDs
    l3_children = {l3_id: [] for l3_id in l3_centroids.keys()}

    for l3_id, l3_data in hierarchy_l3.items():
        # Only process L3 clusters that exist in our output
        if l3_id not in l3_centroids:
            continue

        # Get children from hierarchy (these are level_1_* IDs = our L2 clusters)
        children = l3_data.get('children', [])
        for child_id in children:
            # Check if this child exists in our L2 centroids
            if child_id in l2_centroids:
                l2_to_l3_parent[child_id] = l3_id
                l3_children[l3_id].append(child_id)

    print(f"  Mapped {len(l2_to_l3_parent)} L2->L3 parent relationships")
    print(f"  L3 clusters with children: {sum(1 for c in l3_children.values() if c)}")

    # Build output structure
    layout = {
        "metadata": {
            "view_name": "voronoi-2",
            "display_name": "Voronoi 2 (Semantic Landscape)",
            "description": "Graph topology-based layout using Node2Vec structural embeddings",
            "algorithm": "Node2Vec + UMAP",
            "parameters": {
                "node2vec_dimensions": NODE2VEC_DIMENSIONS,
                "node2vec_walk_length": NODE2VEC_WALK_LENGTH,
                "node2vec_num_walks": NODE2VEC_NUM_WALKS,
                "umap_n_neighbors": UMAP_N_NEIGHBORS,
                "umap_min_dist": UMAP_MIN_DIST,
                "umap_spread": UMAP_SPREAD,
                "umap_metric": UMAP_METRIC
            },
            "node_count": len(node_list),
            "cluster_counts": {
                "level_1": len(l1_centroids),
                "level_2": len(l2_centroids),
                "level_3": len(l3_centroids)
            }
        },
        "node_positions": {
            name: {"x": pos[0], "y": pos[1]}
            for name, pos in node_positions.items()
        },
        "clusters": {
            "level_1": {
                cid: {
                    **data,
                    "polygon": []  # L1 doesn't get Voronoi, too granular
                }
                for cid, data in l1_centroids.items()
            },
            "level_2": {
                cid: {
                    **data,
                    "polygon": l2_polygons.get(cid, []),
                    "parentL3": l2_to_l3_parent.get(cid)  # Parent L3 cluster ID
                }
                for cid, data in l2_centroids.items()
            },
            "level_3": {
                cid: {
                    **data,
                    "polygon": l3_polygons.get(cid, []),
                    "children": l3_children.get(cid, [])  # List of child L2 cluster IDs
                }
                for cid, data in l3_centroids.items()
            }
        }
    }

    return layout


def main():
    """Main entry point for generating Voronoi 2 layout."""
    print("=" * 70)
    print("Voronoi 2 Semantic Landscape Generator")
    print("=" * 70)

    # Load existing hierarchy data
    hierarchy_data = load_hierarchy_data(HIERARCHY_PATH)
    entities = hierarchy_data.get('entities', {})
    relationships = hierarchy_data.get('relationships', [])

    # Build graph
    G = build_graph(entities, relationships)

    # Generate Node2Vec embeddings
    embeddings, node_list = generate_node2vec_embeddings(G)

    # Reduce to 2D with UMAP
    coords_2d = reduce_to_2d(embeddings)

    # Create node position lookup
    node_positions = {
        node: (float(coords_2d[idx, 0]), float(coords_2d[idx, 1]))
        for idx, node in enumerate(node_list)
    }

    # Build complete layout
    layout = build_voronoi2_layout(hierarchy_data, node_positions, node_list)

    # Save output
    print(f"\nSaving Voronoi 2 layout to {OUTPUT_PATH}...")
    with OUTPUT_PATH.open('w') as f:
        json.dump(layout, f, indent=2)

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved {file_size_mb:.2f} MB")

    print("\n" + "=" * 70)
    print("Voronoi 2 generation complete!")
    print("=" * 70)
    print(f"\nTo use this view, update the frontend to load:")
    print(f"  data/graphrag_hierarchy/voronoi_2_layout.json")
    print(f"\nAdd 'voronoi-2' to validModes array in GraphRAG3D_EmbeddingView.js")


if __name__ == "__main__":
    main()
