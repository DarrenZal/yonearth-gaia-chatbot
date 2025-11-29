#!/usr/bin/env python3
"""
Hierarchical Voronoi Layout Generator v3

Creates a 4-level hierarchical layout based on Leiden community detection:
- L3 clusters (57) as super-cluster containers (union of L2 children)
- L2 clusters (681) as mid-level containers (union of L1 children)
- L1 clusters (2,073) as base Voronoi tiles
- All 17,280 entities assigned and constrained within L1 tiles

The Leiden algorithm creates nested partitions via children field:
- L3 (level_3) has children pointing to L2 (level_1_*) cluster IDs
- L2 (level_2) has children pointing to L1 (level_0_*) cluster IDs
- L1 (level_1) has direct entities

Pipeline:
- Step 1: Load UMAP coordinates and hierarchy
- Step 2: Build L2→L3, L1→L2 mappings from children fields
- Step 3: Assign entities to L1 clusters (with Unclassified fallback)
- Step 4: Compute L1 centroids and generate L1 Voronoi tiles
- Step 5: Union L1 tiles into L2 containers
- Step 6: Union L2 tiles into L3 containers
- Step 7: Constrain entities inside their L1 polygon
- Step 8: Generate nested hierarchical JSON (L3→L2→L1→Entities)

Output: data/graphrag_hierarchy/hierarchical_voronoi.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
HIERARCHY_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
COORDS_CSV = ROOT / "data/graphrag_hierarchy/node_layout_coordinates.csv"
OUTPUT_JSON = ROOT / "data/graphrag_hierarchy/hierarchical_voronoi.json"
DEBUG_PNG = ROOT / "data/graphrag_hierarchy/hierarchical_voronoi_debug.png"

VORONOI_MARGIN = 5.0


def load_data() -> Tuple[dict, dict]:
    """Load hierarchy and UMAP coordinates."""
    print("=" * 70)
    print("HIERARCHICAL VORONOI LAYOUT GENERATOR v3")
    print("4-Level Hierarchy: L3 → L2 → L1 → Entities")
    print("=" * 70)

    print(f"\nLoading hierarchy from {HIERARCHY_PATH}...")
    with open(HIERARCHY_PATH) as f:
        hierarchy = json.load(f)

    entities = hierarchy.get('entities', {})
    clusters = hierarchy.get('clusters', {})
    print(f"  Entities: {len(entities):,}")
    print(f"  Cluster levels: {list(clusters.keys())}")
    for level in ['level_1', 'level_2', 'level_3']:
        if level in clusters:
            print(f"    {level}: {len(clusters[level]):,} clusters")

    print(f"\nLoading UMAP coordinates from {COORDS_CSV}...")
    import pandas as pd
    coords_df = pd.read_csv(COORDS_CSV)
    print(f"  Loaded {len(coords_df):,} coordinate rows")

    node_positions = {}
    for _, row in coords_df.iterrows():
        node_positions[row['node_id']] = {
            'x': float(row['x']),
            'y': float(row['y'])
        }

    return hierarchy, node_positions


def build_hierarchy_mappings(clusters: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build hierarchy mappings from children fields:
    - L2 → L3 mapping (from L3's children which are level_1_* IDs = L2 clusters)
    - L1 → L2 mapping (from L2's children which are level_0_* IDs = L1 clusters)
    """
    print("\nBuilding hierarchy mappings from children fields...")

    l3_clusters = clusters.get('level_3', {})
    l2_clusters = clusters.get('level_2', {})

    # L2 → L3 mapping (L3 children are L2 cluster IDs with level_1_* prefix)
    l2_to_l3 = {}
    for l3_id, l3_data in l3_clusters.items():
        children = l3_data.get('children', [])
        for l2_id in children:
            if l2_id.startswith('level_1_'):
                l2_to_l3[l2_id] = l3_id

    print(f"  L2 → L3 mappings: {len(l2_to_l3):,} / {len(l2_clusters)} L2 clusters mapped")

    # L1 → L2 mapping (L2 children are L1 cluster IDs with level_0_* prefix)
    l1_to_l2 = {}
    for l2_id, l2_data in l2_clusters.items():
        children = l2_data.get('children', [])
        for l1_id in children:
            if l1_id.startswith('level_0_'):
                l1_to_l2[l1_id] = l2_id

    l1_clusters = clusters.get('level_1', {})
    print(f"  L1 → L2 mappings: {len(l1_to_l2):,} / {len(l1_clusters)} L1 clusters mapped")

    return l2_to_l3, l1_to_l2


def assign_entities_to_l1(node_positions: dict, clusters: dict) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Assign all entities to L1 clusters.
    Entities not in any L1 cluster go to 'l1_unclassified'.

    Returns:
        entity_to_l1: entity → L1 cluster ID
        l1_to_entities: L1 cluster ID → list of entities
    """
    print("\nAssigning entities to L1 clusters...")

    l1_clusters = clusters.get('level_1', {})

    # Build reverse mapping: entity → L1
    entity_to_l1 = {}
    for l1_id, l1_data in l1_clusters.items():
        for entity in l1_data.get('entities', []):
            entity_to_l1[entity] = l1_id

    # Find unassigned entities
    all_entities = set(node_positions.keys())
    assigned = set(entity_to_l1.keys())
    unassigned = all_entities - assigned

    print(f"  Entities with L1 assignment: {len(assigned):,} ({100*len(assigned)/len(all_entities):.1f}%)")
    print(f"  Entities unassigned: {len(unassigned):,} ({100*len(unassigned)/len(all_entities):.1f}%)")

    # Assign unassigned to 'l1_unclassified'
    for entity in unassigned:
        entity_to_l1[entity] = 'l1_unclassified'

    # Build forward mapping: L1 → entities
    l1_to_entities = defaultdict(list)
    for entity, l1_id in entity_to_l1.items():
        l1_to_entities[l1_id].append(entity)

    print(f"  Total L1 clusters with entities: {len(l1_to_entities)}")
    print(f"  Entities in 'l1_unclassified': {len(l1_to_entities.get('l1_unclassified', []))}")

    return entity_to_l1, dict(l1_to_entities)


def compute_l1_centroids(node_positions: dict, l1_to_entities: Dict[str, List[str]]) -> Dict[str, Tuple[float, float]]:
    """Compute centroid for each L1 cluster from member entity positions."""
    print("\nComputing L1 cluster centroids...")

    l1_centroids = {}
    for l1_id, entities in l1_to_entities.items():
        positions = [node_positions[e] for e in entities if e in node_positions]
        if positions:
            xs = [p['x'] for p in positions]
            ys = [p['y'] for p in positions]
            l1_centroids[l1_id] = (np.mean(xs), np.mean(ys))

    print(f"  Computed {len(l1_centroids):,} L1 centroids")

    return l1_centroids


def generate_voronoi_polygons(centroids: Dict[str, Tuple[float, float]],
                              bounds: Tuple[float, float, float, float]) -> Dict[str, List]:
    """Generate Voronoi polygons from centroids, clipped to bounding box."""
    print("\nGenerating Voronoi polygons...")

    min_x, max_x, min_y, max_y = bounds
    min_x -= VORONOI_MARGIN
    max_x += VORONOI_MARGIN
    min_y -= VORONOI_MARGIN
    max_y += VORONOI_MARGIN

    bbox = Polygon([
        (min_x, min_y), (max_x, min_y),
        (max_x, max_y), (min_x, max_y)
    ])

    cluster_ids = list(centroids.keys())
    points = np.array([centroids[cid] for cid in cluster_ids])

    if len(points) < 4:
        print(f"  ERROR: Need at least 4 points for Voronoi, got {len(points)}")
        return {}

    # Add far corner points to handle infinite regions
    far = max(max_x - min_x, max_y - min_y) * 10
    corner_points = np.array([
        [min_x - far, min_y - far],
        [max_x + far, min_y - far],
        [max_x + far, max_y + far],
        [min_x - far, max_y + far]
    ])
    all_points = np.vstack([points, corner_points])

    vor = Voronoi(all_points)

    polygons = {}
    for idx, cluster_id in enumerate(cluster_ids):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if -1 in region or len(region) < 3:
            continue

        vertices = [vor.vertices[i] for i in region]

        try:
            poly = Polygon(vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)

            clipped = poly.intersection(bbox)

            if clipped.is_empty:
                continue

            if isinstance(clipped, MultiPolygon):
                clipped = max(clipped.geoms, key=lambda g: g.area)

            if isinstance(clipped, Polygon) and clipped.exterior:
                coords = list(clipped.exterior.coords)
                polygons[cluster_id] = [[float(x), float(y)] for x, y in coords]
        except Exception as e:
            continue

    print(f"  Generated {len(polygons):,} valid polygons from {len(cluster_ids)} centroids")

    return polygons


def union_polygons_to_parent(child_polygons: Dict[str, List],
                             child_to_parent: Dict[str, str],
                             parent_name: str) -> Dict[str, List]:
    """Union child polygons into parent containers."""
    print(f"\nUnioning child polygons into {parent_name} containers...")

    # Group children by parent
    parent_children = defaultdict(list)
    for child_id, parent_id in child_to_parent.items():
        if child_id in child_polygons:
            parent_children[parent_id].append(child_id)

    parent_polygons = {}
    for parent_id, child_ids in parent_children.items():
        try:
            shapely_polys = []
            for child_id in child_ids:
                coords = child_polygons[child_id]
                poly = Polygon(coords)
                if poly.is_valid:
                    shapely_polys.append(poly)
                else:
                    shapely_polys.append(poly.buffer(0))

            if not shapely_polys:
                continue

            if len(shapely_polys) == 1:
                unioned = shapely_polys[0]
            else:
                unioned = unary_union(shapely_polys)

            # For disconnected groups, use convex hull
            if isinstance(unioned, MultiPolygon):
                unioned = unioned.convex_hull

            if isinstance(unioned, Polygon) and unioned.exterior:
                coords = list(unioned.exterior.coords)
                parent_polygons[parent_id] = [[float(x), float(y)] for x, y in coords]

        except Exception as e:
            continue

    print(f"  Created {len(parent_polygons):,} {parent_name} polygons")

    return parent_polygons


def constrain_entities_to_polygons(node_positions: dict,
                                   entity_to_l1: Dict[str, str],
                                   l1_polygons: Dict[str, List]) -> dict:
    """Move entities outside their L1 polygon to be inside."""
    print("\nConstraining entities inside their L1 polygons...")

    # Convert L1 polygons to Shapely
    shapely_l1 = {}
    for l1_id, coords in l1_polygons.items():
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            shapely_l1[l1_id] = poly
        except:
            continue

    moved_count = 0
    no_polygon_count = 0
    constrained_positions = {}

    for entity, pos in node_positions.items():
        x, y = pos['x'], pos['y']
        l1_id = entity_to_l1.get(entity, 'l1_unclassified')

        if l1_id not in shapely_l1:
            constrained_positions[entity] = {'x': x, 'y': y, 'l1_cluster': l1_id}
            no_polygon_count += 1
            continue

        poly = shapely_l1[l1_id]
        point = Point(x, y)

        if poly.contains(point):
            constrained_positions[entity] = {'x': x, 'y': y, 'l1_cluster': l1_id}
        else:
            # Move toward centroid
            centroid = poly.centroid
            cx, cy = centroid.x, centroid.y

            new_x, new_y = x, y
            for _ in range(50):
                mid_x = (new_x + cx) / 2
                mid_y = (new_y + cy) / 2
                if poly.contains(Point(mid_x, mid_y)):
                    new_x, new_y = mid_x, mid_y
                    break
                new_x, new_y = mid_x, mid_y

            if not poly.contains(Point(new_x, new_y)):
                new_x, new_y = cx, cy

            constrained_positions[entity] = {'x': new_x, 'y': new_y, 'l1_cluster': l1_id}
            moved_count += 1

    print(f"  Entities moved: {moved_count:,}")
    print(f"  Entities without polygon: {no_polygon_count:,}")
    print(f"  Total: {len(constrained_positions):,}")

    return constrained_positions


def build_4level_hierarchy(clusters: dict,
                           constrained_positions: dict,
                           entity_to_l1: Dict[str, str],
                           l1_to_l2: Dict[str, str],
                           l2_to_l3: Dict[str, str],
                           l1_polygons: Dict[str, List],
                           l2_polygons: Dict[str, List],
                           l3_polygons: Dict[str, List]) -> List[dict]:
    """Build nested 4-level hierarchical JSON structure: L3 → L2 → L1 → Entities"""
    print("\nBuilding 4-level nested hierarchy JSON...")

    l1_clusters = clusters.get('level_1', {})
    l2_clusters = clusters.get('level_2', {})
    l3_clusters = clusters.get('level_3', {})

    # Group entities by L1
    l1_entities = defaultdict(list)
    for entity, pos in constrained_positions.items():
        l1_id = entity_to_l1.get(entity, 'l1_unclassified')
        l1_entities[l1_id].append({
            'id': entity,
            'x': pos['x'],
            'y': pos['y']
        })

    # Build L1 nodes
    l1_nodes = {}
    all_l1_ids = set(l1_to_l2.keys()) | set(l1_entities.keys())

    for l1_id in all_l1_ids:
        l1_data = l1_clusters.get(l1_id, {})
        l1_nodes[l1_id] = {
            'id': l1_id,
            'level': 1,
            'title': l1_data.get('title', l1_id.replace('level_0_', 'Micro-Cluster ')),
            'polygon_coords': l1_polygons.get(l1_id, []),
            'entities': l1_entities.get(l1_id, []),
            'entity_count': len(l1_entities.get(l1_id, []))
        }

    # Group L1s by L2
    l2_l1_children = defaultdict(list)
    for l1_id, l2_id in l1_to_l2.items():
        if l1_id in l1_nodes:
            l2_l1_children[l2_id].append(l1_nodes[l1_id])

    # Find orphan L1s (no L2 parent)
    orphan_l1s = []
    for l1_id in l1_nodes:
        if l1_id not in l1_to_l2:
            orphan_l1s.append(l1_nodes[l1_id])

    # Build L2 nodes
    l2_nodes = {}
    all_l2_ids = set(l2_to_l3.keys()) | set(l2_l1_children.keys())

    for l2_id in all_l2_ids:
        l2_data = l2_clusters.get(l2_id, {})
        children = l2_l1_children.get(l2_id, [])
        l2_nodes[l2_id] = {
            'id': l2_id,
            'level': 2,
            'title': l2_data.get('title', l2_id.replace('level_1_', 'Cluster ')),
            'polygon_coords': l2_polygons.get(l2_id, []),
            'children': children,
            'entity_count': sum(c['entity_count'] for c in children)
        }

    # Group L2s by L3
    l3_l2_children = defaultdict(list)
    for l2_id, l3_id in l2_to_l3.items():
        if l2_id in l2_nodes:
            l3_l2_children[l3_id].append(l2_nodes[l2_id])

    # Find orphan L2s (no L3 parent)
    orphan_l2s = []
    for l2_id in l2_nodes:
        if l2_id not in l2_to_l3:
            orphan_l2s.append(l2_nodes[l2_id])

    # Build L3 nodes
    hierarchy_root = []
    for l3_id, l3_data in l3_clusters.items():
        children = l3_l2_children.get(l3_id, [])
        l3_node = {
            'id': l3_id,
            'level': 3,
            'title': l3_data.get('title', l3_id.replace('level_2_', 'Super-Cluster ')),
            'polygon_coords': l3_polygons.get(l3_id, []),
            'children': children,
            'entity_count': sum(c['entity_count'] for c in children)
        }
        hierarchy_root.append(l3_node)

    # Add orphan L1s wrapped in synthetic L2, then L3
    if orphan_l1s:
        print(f"  Adding {len(orphan_l1s)} orphan L1 clusters (including unclassified)")
        # Wrap orphan L1s in synthetic L2
        synthetic_l2 = {
            'id': 'l2_unclassified',
            'level': 2,
            'title': 'Unclassified (L2)',
            'polygon_coords': [],
            'children': orphan_l1s,
            'entity_count': sum(c['entity_count'] for c in orphan_l1s)
        }
        orphan_l2s.append(synthetic_l2)

    # Add orphan L2s wrapped in synthetic L3
    if orphan_l2s:
        print(f"  Adding {len(orphan_l2s)} orphan L2 clusters")
        synthetic_l3 = {
            'id': 'l3_unclassified',
            'level': 3,
            'title': 'Unclassified Entities',
            'polygon_coords': [],
            'children': orphan_l2s,
            'entity_count': sum(c['entity_count'] for c in orphan_l2s)
        }
        hierarchy_root.append(synthetic_l3)

    # Sort by entity count
    hierarchy_root.sort(key=lambda x: -x['entity_count'])

    # Statistics
    total_entities = sum(l3['entity_count'] for l3 in hierarchy_root)
    total_l2 = sum(len(l3['children']) for l3 in hierarchy_root)
    total_l1 = sum(len(l2['children']) for l3 in hierarchy_root for l2 in l3['children'])

    print(f"  L3 clusters: {len(hierarchy_root)}")
    print(f"  L2 clusters: {total_l2}")
    print(f"  L1 clusters: {total_l1}")
    print(f"  Total entities: {total_entities:,}")

    return hierarchy_root


def generate_debug_plot(hierarchy: List[dict], output_path: Path):
    """Generate debug visualization."""
    print(f"\nGenerating debug plot to {output_path}...")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        import matplotlib.cm as cm
    except ImportError:
        print("  Matplotlib not available")
        return

    fig, ax = plt.subplots(1, 1, figsize=(24, 20), dpi=100)

    # Color map for L3 clusters
    n_l3 = len(hierarchy)
    colors = cm.tab20(np.linspace(0, 1, min(20, n_l3)))

    for l3_idx, l3_node in enumerate(hierarchy):
        l3_color = colors[l3_idx % len(colors)]

        # Draw L3 boundary
        if l3_node.get('polygon_coords'):
            coords = l3_node['polygon_coords']
            poly = MplPolygon(coords, fill=False, edgecolor='black',
                            linewidth=3, linestyle='-')
            ax.add_patch(poly)

            # L3 label
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.text(np.mean(xs), np.max(ys) + 0.3,
                   l3_node.get('title', l3_node['id'])[:30],
                   fontsize=9, ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Draw L2 tiles
        for l2_node in l3_node.get('children', []):
            if l2_node.get('polygon_coords'):
                coords = l2_node['polygon_coords']
                poly = MplPolygon(coords, fill=True, facecolor=l3_color,
                                edgecolor='gray', linewidth=1, alpha=0.3)
                ax.add_patch(poly)

            # Draw L1 tiles inside L2
            for l1_node in l2_node.get('children', []):
                if l1_node.get('polygon_coords'):
                    coords = l1_node['polygon_coords']
                    poly = MplPolygon(coords, fill=True, facecolor=l3_color,
                                    edgecolor='lightgray', linewidth=0.3, alpha=0.5)
                    ax.add_patch(poly)

                # Plot entities
                entities = l1_node.get('entities', [])
                if entities:
                    xs = [e['x'] for e in entities]
                    ys = [e['y'] for e in entities]
                    ax.scatter(xs, ys, s=1, c=[l3_color], alpha=0.6)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_title(f'Hierarchical Voronoi Layout (4 Levels)\n{len(hierarchy)} L3 → L2 → L1 → Entities',
                fontsize=14)
    ax.set_xlabel('UMAP X')
    ax.set_ylabel('UMAP Y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Saved debug plot")


def main():
    """Main entry point."""

    # Load data
    hierarchy_data, node_positions = load_data()
    clusters = hierarchy_data['clusters']

    # Build hierarchy mappings
    l2_to_l3, l1_to_l2 = build_hierarchy_mappings(clusters)

    # Assign entities to L1 (with unclassified fallback)
    entity_to_l1, l1_to_entities = assign_entities_to_l1(node_positions, clusters)

    # Compute L1 centroids
    l1_centroids = compute_l1_centroids(node_positions, l1_to_entities)

    # Calculate bounds
    xs = [p['x'] for p in node_positions.values()]
    ys = [p['y'] for p in node_positions.values()]
    bounds = (min(xs), max(xs), min(ys), max(ys))
    print(f"\nData bounds: X=[{bounds[0]:.2f}, {bounds[1]:.2f}], Y=[{bounds[2]:.2f}, {bounds[3]:.2f}]")

    # Step 4: Generate L1 Voronoi polygons (base layer)
    l1_polygons = generate_voronoi_polygons(l1_centroids, bounds)

    # Step 5: Union L1 → L2
    l2_polygons = union_polygons_to_parent(l1_polygons, l1_to_l2, "L2")

    # Step 6: Union L2 → L3
    l3_polygons = union_polygons_to_parent(l2_polygons, l2_to_l3, "L3")

    # Step 7: Constrain entities inside their L1 polygon
    constrained_positions = constrain_entities_to_polygons(
        node_positions, entity_to_l1, l1_polygons
    )

    # Step 8: Build 4-level hierarchy
    hierarchy_json = build_4level_hierarchy(
        clusters, constrained_positions,
        entity_to_l1, l1_to_l2, l2_to_l3,
        l1_polygons, l2_polygons, l3_polygons
    )

    # Save
    print(f"\nSaving to {OUTPUT_JSON}...")

    # Count totals
    total_l2 = sum(len(l3['children']) for l3 in hierarchy_json)
    total_l1 = sum(len(l2['children']) for l3 in hierarchy_json for l2 in l3['children'])

    output_data = {
        'hierarchy': hierarchy_json,
        'metadata': {
            'total_entities': len(constrained_positions),
            'l1_clusters': total_l1,
            'l2_clusters': total_l2,
            'l3_clusters': len(hierarchy_json),
            'bounds': bounds,
            'description': '4-level hierarchy: L3 → L2 → L1 → Entities'
        }
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Generate debug plot
    generate_debug_plot(hierarchy_json, DEBUG_PNG)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"  Debug: {DEBUG_PNG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
