#!/usr/bin/env python3
"""
Generate Voronoi 5 Strict Treemap Layout

This script creates a top-down strict treemap visualization where:
1. L3 clusters are positioned based on cluster-to-cluster connectivity (not semantics)
2. L2 clusters are constrained within their parent L3 polygon
3. L1 clusters are constrained within their parent L2 polygon
4. Entities are placed within their parent L1 polygon

The algorithm ensures:
- Contiguous clusters at every level (no islands/holes)
- Layout determined by relationship connectivity, not node semantics
- Proper nesting: L3 → L2 → L1 → entities
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import networkx as nx

# Paths
DATA_DIR = Path('/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy')
OUTPUT_FILE = DATA_DIR / 'voronoi5_strict_treemap.json'


def load_data():
    """Load the graphrag hierarchy and build entity-cluster mappings."""
    print("Loading graphrag_hierarchy.json...")
    with open(DATA_DIR / 'graphrag_hierarchy.json', 'r') as f:
        data = json.load(f)

    entities = data['entities']
    relationships = data['relationships']
    clusters = data['clusters']

    print(f"  Loaded {len(entities)} entities, {len(relationships)} relationships")
    print(f"  L0: {len(clusters['level_0'])}, L1: {len(clusters['level_1'])}, L2: {len(clusters['level_2'])}, L3: {len(clusters['level_3'])}")

    return entities, relationships, clusters


def build_entity_cluster_map(clusters):
    """Build mappings from entity -> L1 -> L2 -> L3 cluster."""
    entity_to_l1 = {}
    entity_to_l2 = {}
    entity_to_l3 = {}
    l1_to_l2 = {}
    l2_to_l3 = {}

    # Clusters can be dict (keyed by ID) or list
    def get_cluster_items(level_data):
        if isinstance(level_data, dict):
            return level_data.values()
        return level_data

    # Map entities to L1 clusters
    for l1 in get_cluster_items(clusters['level_1']):
        l1_id = l1['id']
        for entity_name in l1.get('entities', []):
            entity_to_l1[entity_name] = l1_id

    # Map L1 to L2 clusters
    for l2 in get_cluster_items(clusters['level_2']):
        l2_id = l2['id']
        for child_id in l2.get('children', []):
            l1_to_l2[child_id] = l2_id
        for entity_name in l2.get('entities', []):
            entity_to_l2[entity_name] = l2_id

    # Map L2 to L3 clusters
    for l3 in get_cluster_items(clusters['level_3']):
        l3_id = l3['id']
        for child_id in l3.get('children', []):
            l2_to_l3[child_id] = l3_id

    # Propagate mappings upward
    for entity, l1_id in entity_to_l1.items():
        if l1_id in l1_to_l2:
            entity_to_l2[entity] = l1_to_l2[l1_id]
            l2_id = l1_to_l2[l1_id]
            if l2_id in l2_to_l3:
                entity_to_l3[entity] = l2_to_l3[l2_id]

    print(f"  Entity mappings: {len(entity_to_l1)} to L1, {len(entity_to_l2)} to L2, {len(entity_to_l3)} to L3")

    return entity_to_l1, entity_to_l2, entity_to_l3, l1_to_l2, l2_to_l3


def build_cluster_connectivity_graph(relationships, entity_to_l3, entity_to_l2, entity_to_l1):
    """Build connectivity graphs between clusters based on relationship counts."""

    print("Building cluster connectivity graphs...")

    # L3 connectivity: count relationships between entities in different L3 clusters
    l3_edges = defaultdict(int)
    l2_edges = defaultdict(int)
    l1_edges = defaultdict(int)

    for rel in relationships:
        # Handle different relationship formats
        source = rel.get('source') or rel.get('source_entity')
        target = rel.get('target') or rel.get('target_entity')

        if not source or not target:
            continue

        # L3 connectivity
        src_l3 = entity_to_l3.get(source)
        tgt_l3 = entity_to_l3.get(target)
        if src_l3 and tgt_l3 and src_l3 != tgt_l3:
            edge = tuple(sorted([src_l3, tgt_l3]))
            l3_edges[edge] += 1

        # L2 connectivity
        src_l2 = entity_to_l2.get(source)
        tgt_l2 = entity_to_l2.get(target)
        if src_l2 and tgt_l2 and src_l2 != tgt_l2:
            edge = tuple(sorted([src_l2, tgt_l2]))
            l2_edges[edge] += 1

        # L1 connectivity
        src_l1 = entity_to_l1.get(source)
        tgt_l1 = entity_to_l1.get(target)
        if src_l1 and tgt_l1 and src_l1 != tgt_l1:
            edge = tuple(sorted([src_l1, tgt_l1]))
            l1_edges[edge] += 1

    print(f"  L3 edges: {len(l3_edges)}, L2 edges: {len(l2_edges)}, L1 edges: {len(l1_edges)}")

    return dict(l3_edges), dict(l2_edges), dict(l1_edges)


def compute_cluster_sizes(clusters, entity_to_l1, entity_to_l2, entity_to_l3):
    """Compute the number of entities in each cluster (for sizing)."""
    l3_sizes = defaultdict(int)
    l2_sizes = defaultdict(int)
    l1_sizes = defaultdict(int)

    for entity in entity_to_l3:
        l3_id = entity_to_l3[entity]
        l3_sizes[l3_id] += 1

    for entity in entity_to_l2:
        l2_id = entity_to_l2[entity]
        l2_sizes[l2_id] += 1

    for entity in entity_to_l1:
        l1_id = entity_to_l1[entity]
        l1_sizes[l1_id] += 1

    return dict(l3_sizes), dict(l2_sizes), dict(l1_sizes)


def force_directed_layout(nodes, edges, sizes, iterations=100, k=1.0):
    """
    Run force-directed layout on nodes with connectivity-based attraction.

    Args:
        nodes: List of node IDs
        edges: Dict of (node1, node2) -> weight
        sizes: Dict of node_id -> size (number of entities)
        iterations: Number of simulation iterations
        k: Optimal distance factor

    Returns:
        Dict of node_id -> (x, y) position
    """
    if not nodes:
        return {}

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for (n1, n2), weight in edges.items():
        if n1 in nodes and n2 in nodes:
            G.add_edge(n1, n2, weight=weight)

    # Initial random positions
    np.random.seed(42)
    pos = {n: np.random.rand(2) * 10 for n in nodes}

    if len(nodes) == 1:
        return {nodes[0]: (5.0, 5.0)}

    # Use spring layout with size-based node weights
    # Larger nodes get more space
    node_sizes = {n: max(1, sizes.get(n, 1)) for n in nodes}
    total_size = sum(node_sizes.values())

    try:
        # Use NetworkX spring layout with edge weights
        pos = nx.spring_layout(
            G,
            k=k * np.sqrt(len(nodes)),
            iterations=iterations,
            weight='weight',
            seed=42,
            scale=10
        )

        # Convert to regular dict of tuples
        pos = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}
    except Exception as e:
        print(f"    Warning: spring_layout failed ({e}), using random positions")
        pos = {n: (np.random.rand() * 10, np.random.rand() * 10) for n in nodes}

    return pos


def generate_voronoi_polygons(positions, sizes, bounds=None, lloyd_iterations=5):
    """
    Generate Voronoi polygons from positions with Lloyd's relaxation.

    Args:
        positions: Dict of id -> (x, y)
        sizes: Dict of id -> size
        bounds: Optional (minx, miny, maxx, maxy) bounding box
        lloyd_iterations: Number of Lloyd relaxation iterations for smoother cells

    Returns:
        Dict of id -> Polygon
    """
    if not positions:
        return {}

    ids = list(positions.keys())
    points = np.array([positions[i] for i in ids])

    # Determine bounds
    if bounds:
        minx, miny, maxx, maxy = bounds
    else:
        minx, miny = points.min(axis=0) - 5
        maxx, maxy = points.max(axis=0) + 5

    # Create bounding polygon
    bounding_poly = Polygon([
        (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)
    ])

    if len(points) < 3:
        # Not enough points for Voronoi - create hexagonal-ish polygons instead of squares
        polygons = {}
        for i, pid in enumerate(ids):
            x, y = positions[pid]
            size = max(0.5, np.sqrt(sizes.get(pid, 1)) * 0.3)
            # Create a hexagon instead of square for more organic look
            n_sides = 6
            angles = [2 * np.pi * k / n_sides + np.pi / 6 for k in range(n_sides)]
            hex_coords = [(x + size * np.cos(a), y + size * np.sin(a)) for a in angles]
            poly = Polygon(hex_coords)
            poly = poly.intersection(bounding_poly)
            if poly.is_valid and not poly.is_empty:
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)
                polygons[pid] = poly
        return polygons

    # Apply Lloyd's relaxation for more regular cells
    relaxed_points = points.copy()

    for iteration in range(lloyd_iterations):
        # Add far-away points to bound the Voronoi diagram
        margin = max(maxx - minx, maxy - miny) * 2
        bounding_points = np.array([
            [minx - margin, miny - margin],
            [maxx + margin, miny - margin],
            [maxx + margin, maxy + margin],
            [minx - margin, maxy + margin]
        ])

        all_points = np.vstack([relaxed_points, bounding_points])

        try:
            vor = Voronoi(all_points)
        except Exception as e:
            print(f"    Warning: Voronoi failed iteration {iteration} ({e})")
            break

        # Move each point towards its cell's centroid (Lloyd's relaxation)
        new_points = []
        for i in range(len(ids)):
            region_idx = vor.point_region[i]
            region_vertices = vor.regions[region_idx]

            if -1 in region_vertices or len(region_vertices) < 3:
                new_points.append(relaxed_points[i])
                continue

            try:
                poly_coords = [vor.vertices[v] for v in region_vertices]
                poly = Polygon(poly_coords)
                poly = poly.intersection(bounding_poly)

                if poly.is_valid and not poly.is_empty:
                    if isinstance(poly, MultiPolygon):
                        poly = max(poly.geoms, key=lambda p: p.area)
                    # Use weighted centroid based on size
                    centroid = poly.centroid
                    weight = 0.7  # How much to move toward centroid
                    new_x = relaxed_points[i][0] * (1 - weight) + centroid.x * weight
                    new_y = relaxed_points[i][1] * (1 - weight) + centroid.y * weight
                    new_points.append([new_x, new_y])
                else:
                    new_points.append(relaxed_points[i])
            except Exception:
                new_points.append(relaxed_points[i])

        relaxed_points = np.array(new_points)

    # Final Voronoi computation with relaxed points
    margin = max(maxx - minx, maxy - miny) * 2
    bounding_points = np.array([
        [minx - margin, miny - margin],
        [maxx + margin, miny - margin],
        [maxx + margin, maxy + margin],
        [minx - margin, maxy + margin]
    ])

    all_points = np.vstack([relaxed_points, bounding_points])

    try:
        vor = Voronoi(all_points)
    except Exception as e:
        print(f"    Warning: Final Voronoi failed ({e})")
        return {}

    polygons = {}
    for i, pid in enumerate(ids):
        region_idx = vor.point_region[i]
        region_vertices = vor.regions[region_idx]

        if -1 in region_vertices or len(region_vertices) < 3:
            # Unbounded region - create a small hexagon around the point
            x, y = relaxed_points[i]
            size = max(0.3, np.sqrt(sizes.get(pid, 1)) * 0.2)
            n_sides = 6
            angles = [2 * np.pi * k / n_sides + np.pi / 6 for k in range(n_sides)]
            hex_coords = [(x + size * np.cos(a), y + size * np.sin(a)) for a in angles]
            poly = Polygon(hex_coords)
            poly = poly.intersection(bounding_poly)
            if poly.is_valid and not poly.is_empty:
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)
                polygons[pid] = poly
            continue

        try:
            poly_coords = [vor.vertices[v] for v in region_vertices]
            poly = Polygon(poly_coords)

            # Clip to bounding box
            poly = poly.intersection(bounding_poly)

            if poly.is_valid and not poly.is_empty:
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)
                polygons[pid] = poly
        except Exception:
            pass

    return polygons


def constrained_layout_within_polygon(child_ids, child_edges, child_sizes, parent_polygon, iterations=50):
    """
    Run force-directed layout constrained within a parent polygon.

    Returns Dict of child_id -> (x, y) position
    """
    if not child_ids:
        return {}

    if len(child_ids) == 1:
        centroid = parent_polygon.centroid
        return {child_ids[0]: (centroid.x, centroid.y)}

    # Get parent bounds
    minx, miny, maxx, maxy = parent_polygon.bounds

    # Run unconstrained layout first
    positions = force_directed_layout(child_ids, child_edges, child_sizes, iterations=iterations)

    if not positions:
        # Fallback: distribute evenly
        n = len(child_ids)
        for i, cid in enumerate(child_ids):
            angle = 2 * np.pi * i / n
            r = min(maxx - minx, maxy - miny) * 0.3
            cx, cy = parent_polygon.centroid.x, parent_polygon.centroid.y
            positions[cid] = (cx + r * np.cos(angle), cy + r * np.sin(angle))

    # Scale and translate to fit within parent
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]

    if xs and ys:
        src_minx, src_maxx = min(xs), max(xs)
        src_miny, src_maxy = min(ys), max(ys)

        # Add padding
        padding = 0.1
        tgt_minx = minx + (maxx - minx) * padding
        tgt_maxx = maxx - (maxx - minx) * padding
        tgt_miny = miny + (maxy - miny) * padding
        tgt_maxy = maxy - (maxy - miny) * padding

        # Scale to fit
        src_w = max(src_maxx - src_minx, 0.001)
        src_h = max(src_maxy - src_miny, 0.001)
        tgt_w = tgt_maxx - tgt_minx
        tgt_h = tgt_maxy - tgt_miny

        scale = min(tgt_w / src_w, tgt_h / src_h)

        # Center offset
        cx_src = (src_minx + src_maxx) / 2
        cy_src = (src_miny + src_maxy) / 2
        cx_tgt = (tgt_minx + tgt_maxx) / 2
        cy_tgt = (tgt_miny + tgt_maxy) / 2

        # Transform
        new_positions = {}
        for cid, (x, y) in positions.items():
            nx = cx_tgt + (x - cx_src) * scale
            ny = cy_tgt + (y - cy_src) * scale

            # Ensure within parent polygon
            pt = Point(nx, ny)
            if not parent_polygon.contains(pt):
                # Move to nearest point inside
                nearest = parent_polygon.centroid
                nx, ny = nearest.x, nearest.y

            new_positions[cid] = (nx, ny)

        return new_positions

    return positions


def clip_polygon_to_parent(child_poly, parent_poly):
    """Clip a child polygon to fit within the parent polygon."""
    try:
        clipped = child_poly.intersection(parent_poly)
        if clipped.is_empty:
            return None
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda p: p.area)
        return clipped
    except Exception:
        return None


def place_entities_in_polygon(entity_names, polygon, entity_data):
    """
    Place entities within a polygon using a simple grid/packing algorithm.

    Returns list of {id, x, y} dicts
    """
    if not entity_names or polygon is None:
        return []

    minx, miny, maxx, maxy = polygon.bounds
    cx, cy = polygon.centroid.x, polygon.centroid.y

    n = len(entity_names)
    entities = []

    if n == 1:
        entities.append({
            'id': entity_names[0],
            'x': cx,
            'y': cy
        })
    elif n <= 4:
        # Place in small circle around centroid
        for i, name in enumerate(entity_names):
            angle = 2 * np.pi * i / n
            r = min(maxx - minx, maxy - miny) * 0.2
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)

            # Ensure inside polygon
            pt = Point(x, y)
            if not polygon.contains(pt):
                x, y = cx, cy

            entities.append({'id': name, 'x': x, 'y': y})
    else:
        # Grid layout with jitter
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        dx = (maxx - minx) * 0.8 / max(cols, 1)
        dy = (maxy - miny) * 0.8 / max(rows, 1)

        start_x = minx + (maxx - minx) * 0.1
        start_y = miny + (maxy - miny) * 0.1

        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break

                x = start_x + col * dx + np.random.uniform(-dx*0.2, dx*0.2)
                y = start_y + row * dy + np.random.uniform(-dy*0.2, dy*0.2)

                # Ensure inside polygon
                pt = Point(x, y)
                if not polygon.contains(pt):
                    x, y = cx, cy

                entities.append({'id': entity_names[idx], 'x': x, 'y': y})
                idx += 1

    return entities


def generate_strict_treemap(entities, relationships, clusters):
    """
    Generate the strict treemap layout.

    Returns the hierarchy data structure for JSON output.
    """
    print("\n=== GENERATING STRICT TREEMAP LAYOUT ===\n")

    # Build mappings
    entity_to_l1, entity_to_l2, entity_to_l3, l1_to_l2, l2_to_l3 = build_entity_cluster_map(clusters)

    # Build connectivity graphs
    l3_edges, l2_edges, l1_edges = build_cluster_connectivity_graph(
        relationships, entity_to_l3, entity_to_l2, entity_to_l1
    )

    # Compute sizes
    l3_sizes, l2_sizes, l1_sizes = compute_cluster_sizes(
        clusters, entity_to_l1, entity_to_l2, entity_to_l3
    )

    # Get cluster info - handle both dict and list formats
    def to_id_dict(level_data):
        if isinstance(level_data, dict):
            # Already keyed by ID - ensure each item has 'id' field
            return {k: {**v, 'id': k} if 'id' not in v else v for k, v in level_data.items()}
        return {c['id']: c for c in level_data}

    l3_clusters = to_id_dict(clusters['level_3'])
    l2_clusters = to_id_dict(clusters['level_2'])
    l1_clusters = to_id_dict(clusters['level_1'])

    # === STEP 1: L3 Layout (Root Level) ===
    print("Step 1: Building L3 (root) layout based on connectivity...")

    l3_ids = list(l3_clusters.keys())
    print(f"  L3 clusters: {len(l3_ids)}")

    # Filter edges to only include L3 clusters we have
    l3_edges_filtered = {k: v for k, v in l3_edges.items()
                        if k[0] in l3_ids and k[1] in l3_ids}

    l3_positions = force_directed_layout(l3_ids, l3_edges_filtered, l3_sizes, iterations=150)
    print(f"  Computed positions for {len(l3_positions)} L3 clusters")

    # Generate L3 Voronoi polygons
    l3_polygons = generate_voronoi_polygons(l3_positions, l3_sizes)
    print(f"  Generated {len(l3_polygons)} L3 polygons")

    # === STEP 2: L2 Layout (within L3) ===
    print("\nStep 2: Building L2 layouts within L3 polygons...")

    l2_positions = {}
    l2_polygons = {}

    for l3_id, l3_poly in l3_polygons.items():
        l3_cluster = l3_clusters[l3_id]
        l2_children = l3_cluster.get('children', [])

        if not l2_children:
            continue

        # Get L2 edges within this L3
        l2_ids_in_l3 = [c for c in l2_children if c in l2_clusters]
        l2_edges_filtered = {k: v for k, v in l2_edges.items()
                           if k[0] in l2_ids_in_l3 and k[1] in l2_ids_in_l3}
        l2_sizes_filtered = {k: v for k, v in l2_sizes.items() if k in l2_ids_in_l3}

        # Constrained layout
        l2_pos = constrained_layout_within_polygon(
            l2_ids_in_l3, l2_edges_filtered, l2_sizes_filtered, l3_poly
        )
        l2_positions.update(l2_pos)

        # Generate L2 Voronoi within L3
        if l2_pos:
            l2_polys = generate_voronoi_polygons(l2_pos, l2_sizes_filtered, bounds=l3_poly.bounds)

            # Clip to parent
            for l2_id, l2_poly in l2_polys.items():
                clipped = clip_polygon_to_parent(l2_poly, l3_poly)
                if clipped:
                    l2_polygons[l2_id] = clipped

    print(f"  Positioned {len(l2_positions)} L2 clusters")
    print(f"  Generated {len(l2_polygons)} L2 polygons")

    # === STEP 3: L1 Layout (within L2) ===
    print("\nStep 3: Building L1 layouts within L2 polygons...")

    l1_positions = {}
    l1_polygons = {}

    for l2_id, l2_poly in l2_polygons.items():
        l2_cluster = l2_clusters.get(l2_id)
        if not l2_cluster:
            continue

        l1_children = l2_cluster.get('children', [])
        if not l1_children:
            continue

        l1_ids_in_l2 = [c for c in l1_children if c in l1_clusters]
        l1_edges_filtered = {k: v for k, v in l1_edges.items()
                           if k[0] in l1_ids_in_l2 and k[1] in l1_ids_in_l2}
        l1_sizes_filtered = {k: v for k, v in l1_sizes.items() if k in l1_ids_in_l2}

        l1_pos = constrained_layout_within_polygon(
            l1_ids_in_l2, l1_edges_filtered, l1_sizes_filtered, l2_poly
        )
        l1_positions.update(l1_pos)

        if l1_pos:
            l1_polys = generate_voronoi_polygons(l1_pos, l1_sizes_filtered, bounds=l2_poly.bounds)
            for l1_id, l1_poly in l1_polys.items():
                clipped = clip_polygon_to_parent(l1_poly, l2_poly)
                if clipped:
                    l1_polygons[l1_id] = clipped

    print(f"  Positioned {len(l1_positions)} L1 clusters")
    print(f"  Generated {len(l1_polygons)} L1 polygons")

    # === STEP 4: Place entities within L1 polygons ===
    print("\nStep 4: Placing entities within L1 polygons...")

    entity_positions = {}

    for l1_id, l1_poly in l1_polygons.items():
        l1_cluster = l1_clusters.get(l1_id)
        if not l1_cluster:
            continue

        entity_names = l1_cluster.get('entities', [])
        placed = place_entities_in_polygon(entity_names, l1_poly, entities)

        for e in placed:
            entity_positions[e['id']] = (e['x'], e['y'])

    print(f"  Placed {len(entity_positions)} entities in L1 polygons")

    # Also place entities directly in L2 polygons that have no L1 children
    print("\nStep 4b: Placing entities in L2 polygons without L1 children...")
    l2_direct_entities = 0

    for l2_id, l2_poly in l2_polygons.items():
        l2_cluster = l2_clusters.get(l2_id)
        if not l2_cluster:
            continue

        # Check if this L2 has L1 children
        l1_children = l2_cluster.get('children', [])
        has_l1 = any(c in l1_polygons for c in l1_children)

        if not has_l1:
            # Place entities directly in L2
            entity_names = l2_cluster.get('entities', [])
            # Only place entities not already positioned
            entity_names = [n for n in entity_names if n not in entity_positions]

            if entity_names:
                placed = place_entities_in_polygon(entity_names, l2_poly, entities)
                for e in placed:
                    entity_positions[e['id']] = (e['x'], e['y'])
                l2_direct_entities += len(placed)

    print(f"  Placed {l2_direct_entities} entities directly in L2 polygons")
    print(f"  Total entities placed: {len(entity_positions)}")

    # === BUILD OUTPUT STRUCTURE ===
    print("\nBuilding output hierarchy...")

    hierarchy = []

    for l3_id in l3_polygons:
        l3_cluster = l3_clusters[l3_id]
        l3_poly = l3_polygons[l3_id]

        l3_item = {
            'id': l3_id,
            'level': 3,
            'title': l3_cluster.get('title') or l3_cluster.get('name') or l3_id,
            'polygon_coords': list(l3_poly.exterior.coords) if l3_poly else [],
            'children': [],
            'entity_count': l3_sizes.get(l3_id, 0)
        }

        # Add L2 children
        for l2_id in l3_cluster.get('children', []):
            if l2_id not in l2_polygons:
                continue

            l2_cluster = l2_clusters.get(l2_id, {})
            l2_poly = l2_polygons[l2_id]

            l2_item = {
                'id': l2_id,
                'level': 2,
                'title': l2_cluster.get('title') or l2_cluster.get('name') or l2_id,
                'polygon_coords': list(l2_poly.exterior.coords) if l2_poly else [],
                'children': [],
                'entities': [],  # Direct entities for L2 without L1 children
                'entity_count': l2_sizes.get(l2_id, 0)
            }

            # Add L1 children if they exist
            l1_child_ids = l2_cluster.get('children', [])
            has_l1_children = False

            for l1_id in l1_child_ids:
                if l1_id not in l1_polygons:
                    continue

                has_l1_children = True
                l1_cluster = l1_clusters.get(l1_id, {})
                l1_poly = l1_polygons[l1_id]

                # Get entities for this L1
                entity_names = l1_cluster.get('entities', [])
                entity_list = []
                for name in entity_names:
                    if name in entity_positions:
                        x, y = entity_positions[name]
                        entity_list.append({'id': name, 'x': x, 'y': y})

                l1_item = {
                    'id': l1_id,
                    'level': 1,
                    'title': l1_cluster.get('title') or l1_cluster.get('name') or l1_id,
                    'polygon_coords': list(l1_poly.exterior.coords) if l1_poly else [],
                    'entities': entity_list,
                    'entity_count': len(entity_list)
                }

                l2_item['children'].append(l1_item)

            # If L2 has no L1 children, place entities directly in L2
            if not has_l1_children:
                l2_entity_names = l2_cluster.get('entities', [])
                for name in l2_entity_names:
                    if name in entity_positions:
                        x, y = entity_positions[name]
                        l2_item['entities'].append({'id': name, 'x': x, 'y': y})

            l3_item['children'].append(l2_item)

        hierarchy.append(l3_item)

    return hierarchy, l3_polygons


def main():
    print("=" * 70)
    print("STRICT TREEMAP LAYOUT GENERATOR")
    print("Connectivity-based clustering with contiguous nested polygons")
    print("=" * 70)

    # Load data
    entities, relationships, clusters = load_data()

    # Generate layout
    hierarchy, l3_polygons = generate_strict_treemap(entities, relationships, clusters)

    # Compute bounds
    all_coords = []
    for l3_item in hierarchy:
        all_coords.extend(l3_item.get('polygon_coords', []))
        for l2 in l3_item.get('children', []):
            all_coords.extend(l2.get('polygon_coords', []))
            for l1 in l2.get('children', []):
                all_coords.extend(l1.get('polygon_coords', []))

    if all_coords:
        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        bounds = [min(xs), max(xs), min(ys), max(ys)]
    else:
        bounds = [0, 20, 0, 20]

    # Count totals
    total_l2 = sum(len(l3.get('children', [])) for l3 in hierarchy)
    total_l1 = sum(
        len(l2.get('children', []))
        for l3 in hierarchy
        for l2 in l3.get('children', [])
    )
    total_entities = sum(
        len(l1.get('entities', []))
        for l3 in hierarchy
        for l2 in l3.get('children', [])
        for l1 in l2.get('children', [])
    )

    # Build output
    output = {
        'metadata': {
            'layout_type': 'strict_treemap',
            'description': 'Top-down strict treemap with connectivity-based positioning',
            'l3_clusters': len(hierarchy),
            'l2_clusters': total_l2,
            'l1_clusters': total_l1,
            'total_entities': total_entities,
            'bounds': bounds
        },
        'hierarchy': hierarchy
    }

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  L3 clusters: {len(hierarchy)}")
    print(f"  L2 clusters: {total_l2}")
    print(f"  L1 clusters: {total_l1}")
    print(f"  Entities: {total_entities}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == '__main__':
    main()
