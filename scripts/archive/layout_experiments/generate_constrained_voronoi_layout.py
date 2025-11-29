#!/usr/bin/env python3
"""
Constrained Voronoi Layout Generator.

This script ensures 100% containment of nodes within their cluster boundaries by:
1. Loading UMAP coordinates from the supervised semantic layout
2. Computing community centroids
3. Generating Voronoi regions from centroids
4. Moving any "leaked" nodes back inside their assigned polygon

Output:
  - data/graphrag_hierarchy/constrained_voronoi_layout.json
  - data/graphrag_hierarchy/constrained_layout_debug.png
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict

# --------------------------------------------------------------------------------------
# Paths & Parameters
# --------------------------------------------------------------------------------------

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
HIERARCHY_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
UMAP_COORDS_CSV = ROOT / "data/graphrag_hierarchy/node_layout_coordinates.csv"
OUTPUT_JSON = ROOT / "data/graphrag_hierarchy/constrained_voronoi_layout.json"
OUTPUT_PNG = ROOT / "data/graphrag_hierarchy/constrained_layout_debug.png"

# Containment parameters
CONTAINMENT_MARGIN = 0.90  # Move nodes to 90% of distance from center to edge
MAX_ITERATIONS = 100       # Maximum iterations for containment check
STEP_SIZE = 0.1            # Step size for moving nodes toward centroid


def load_umap_coordinates(csv_path: Path) -> pd.DataFrame:
    """Load the UMAP coordinates from CSV."""
    print(f"Loading UMAP coordinates from {csv_path}...")

    if not csv_path.exists():
        print(f"  ERROR: CSV file not found at {csv_path}")
        print("  Run layout_supervised_semantic.py first to generate coordinates.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} nodes")
    print(f"  Columns: {list(df.columns)}")
    print(f"  X range: [{df['x'].min():.3f}, {df['x'].max():.3f}]")
    print(f"  Y range: [{df['y'].min():.3f}, {df['y'].max():.3f}]")

    return df


def load_hierarchy_data(path: Path) -> dict:
    """Load the GraphRAG hierarchy JSON."""
    print(f"\nLoading hierarchy data from {path}...")
    with path.open() as f:
        data = json.load(f)

    clusters = data.get('clusters', {})
    print(f"  Cluster levels: {list(clusters.keys())}")

    return data


def build_entity_to_cluster_map(clusters: dict, level: str = 'level_2') -> Dict[str, str]:
    """Build mapping from entity name -> cluster ID."""
    print(f"\nBuilding entity -> {level} cluster mapping...")

    entity_to_cluster = {}
    level_clusters = clusters.get(level, [])

    if isinstance(level_clusters, dict):
        level_clusters = list(level_clusters.values())

    for cluster in level_clusters:
        cluster_id = cluster.get('id', cluster.get('name', 'unknown'))
        entities = cluster.get('entities', [])
        for entity in entities:
            entity_to_cluster[entity] = cluster_id

    print(f"  Mapped {len(entity_to_cluster):,} entities to {len(level_clusters)} clusters")

    return entity_to_cluster


def compute_cluster_centroids(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Compute the centroid (mean X, Y) for each cluster."""
    print("\nComputing cluster centroids...")

    centroids = {}
    cluster_groups = df.groupby('l2_cluster')

    for cluster_id, group in cluster_groups:
        if cluster_id == 'unknown':
            continue
        centroid_x = group['x'].mean()
        centroid_y = group['y'].mean()
        centroids[cluster_id] = (centroid_x, centroid_y)

    print(f"  Computed {len(centroids)} cluster centroids")

    return centroids


def generate_voronoi_polygons(centroids: Dict[str, Tuple[float, float]],
                               bounds: Tuple[float, float, float, float]) -> Dict[str, 'Polygon']:
    """
    Generate Voronoi regions from cluster centroids.

    Args:
        centroids: Dict mapping cluster_id -> (x, y) centroid
        bounds: (min_x, min_y, max_x, max_y) bounding box

    Returns:
        Dict mapping cluster_id -> shapely Polygon
    """
    print("\nGenerating Voronoi polygons...")

    try:
        from scipy.spatial import Voronoi
        from shapely.geometry import Polygon, Point, box
        from shapely.ops import unary_union
    except ImportError as e:
        print(f"  ERROR: Missing required library: {e}")
        print("  Install with: pip install scipy shapely")
        sys.exit(1)

    # Get cluster IDs and points in order
    cluster_ids = list(centroids.keys())
    points = np.array([centroids[cid] for cid in cluster_ids])

    print(f"  Input points: {len(points)}")

    # Expand bounds slightly to ensure all regions are closed
    min_x, min_y, max_x, max_y = bounds
    padding = max(max_x - min_x, max_y - min_y) * 0.5

    # Add corner points to ensure bounded Voronoi regions
    corner_points = np.array([
        [min_x - padding, min_y - padding],
        [min_x - padding, max_y + padding],
        [max_x + padding, min_y - padding],
        [max_x + padding, max_y + padding],
        [min_x - padding, (min_y + max_y) / 2],
        [max_x + padding, (min_y + max_y) / 2],
        [(min_x + max_x) / 2, min_y - padding],
        [(min_x + max_x) / 2, max_y + padding],
    ])

    all_points = np.vstack([points, corner_points])

    # Generate Voronoi diagram
    vor = Voronoi(all_points)

    # Create bounding box for clipping
    bounding_box = box(min_x - padding/2, min_y - padding/2,
                       max_x + padding/2, max_y + padding/2)

    # Convert Voronoi regions to Shapely polygons
    polygons = {}

    for idx, cluster_id in enumerate(cluster_ids):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            # Unbounded or empty region - create a fallback circle
            cx, cy = centroids[cluster_id]
            # Create a small circular polygon around centroid
            angles = np.linspace(0, 2*np.pi, 32)
            radius = 0.5  # Small default radius
            circle_pts = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
            poly = Polygon(circle_pts)
        else:
            # Get vertices for this region
            vertices = [vor.vertices[i] for i in region]
            if len(vertices) >= 3:
                poly = Polygon(vertices)
            else:
                # Fallback for degenerate cases
                cx, cy = centroids[cluster_id]
                angles = np.linspace(0, 2*np.pi, 32)
                radius = 0.5
                circle_pts = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
                poly = Polygon(circle_pts)

        # Clip to bounding box
        try:
            clipped = poly.intersection(bounding_box)
            if clipped.is_empty or not clipped.is_valid:
                clipped = poly
            polygons[cluster_id] = clipped
        except Exception as e:
            polygons[cluster_id] = poly

    print(f"  Generated {len(polygons)} Voronoi polygons")

    # Debug: Check polygon areas
    areas = [p.area for p in polygons.values() if hasattr(p, 'area')]
    if areas:
        print(f"  Polygon areas: min={min(areas):.2f}, max={max(areas):.2f}, mean={np.mean(areas):.2f}")

    return polygons


def check_and_fix_containment(df: pd.DataFrame,
                               polygons: Dict[str, 'Polygon'],
                               centroids: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Check if nodes are inside their assigned polygon and move them if not.

    For nodes outside their polygon:
    - Calculate vector from node to centroid
    - Move node along that vector until strictly inside
    """
    print("\nChecking and fixing node containment...")

    from shapely.geometry import Point

    # Create working copies
    df = df.copy()
    df['x_original'] = df['x']
    df['y_original'] = df['y']
    df['was_moved'] = False
    df['move_distance'] = 0.0

    nodes_outside = 0
    nodes_fixed = 0
    nodes_no_polygon = 0

    for idx, row in df.iterrows():
        cluster_id = row['l2_cluster']

        if cluster_id == 'unknown' or cluster_id not in polygons:
            nodes_no_polygon += 1
            continue

        polygon = polygons[cluster_id]
        centroid = centroids.get(cluster_id)

        if centroid is None:
            continue

        point = Point(row['x'], row['y'])

        # Check if point is inside polygon
        if not polygon.contains(point):
            nodes_outside += 1

            # Get centroid coordinates
            cx, cy = centroid
            px, py = row['x'], row['y']

            # Move point toward centroid
            # Start from current position and step toward centroid
            new_x, new_y = px, py

            for step in range(MAX_ITERATIONS):
                # Calculate direction vector to centroid
                dx = cx - new_x
                dy = cy - new_y
                dist = np.sqrt(dx**2 + dy**2)

                if dist < 0.001:
                    # Already at centroid
                    break

                # Normalize direction
                dx /= dist
                dy /= dist

                # Move a step toward centroid
                step_dist = dist * STEP_SIZE
                new_x += dx * step_dist
                new_y += dy * step_dist

                # Check if now inside
                new_point = Point(new_x, new_y)
                if polygon.contains(new_point):
                    # Move slightly further in (CONTAINMENT_MARGIN)
                    # to ensure we're not on the edge
                    remaining_dist = np.sqrt((cx - new_x)**2 + (cy - new_y)**2)
                    move_in = remaining_dist * (1 - CONTAINMENT_MARGIN)
                    new_x += dx * move_in
                    new_y += dy * move_in

                    nodes_fixed += 1
                    break

            # Update coordinates
            move_dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
            df.at[idx, 'x'] = new_x
            df.at[idx, 'y'] = new_y
            df.at[idx, 'was_moved'] = True
            df.at[idx, 'move_distance'] = move_dist

    print(f"  Nodes outside their polygon: {nodes_outside}")
    print(f"  Nodes successfully moved inside: {nodes_fixed}")
    print(f"  Nodes without polygon assignment: {nodes_no_polygon}")

    if nodes_outside > 0:
        moved_df = df[df['was_moved']]
        if len(moved_df) > 0:
            print(f"  Average move distance: {moved_df['move_distance'].mean():.4f}")
            print(f"  Max move distance: {moved_df['move_distance'].max():.4f}")

    return df


def verify_containment(df: pd.DataFrame, polygons: Dict[str, 'Polygon']) -> Tuple[int, int]:
    """Verify that all nodes are now inside their polygons."""
    print("\nVerifying final containment...")

    from shapely.geometry import Point

    inside_count = 0
    outside_count = 0

    for idx, row in df.iterrows():
        cluster_id = row['l2_cluster']

        if cluster_id == 'unknown' or cluster_id not in polygons:
            continue

        polygon = polygons[cluster_id]
        point = Point(row['x'], row['y'])

        if polygon.contains(point):
            inside_count += 1
        else:
            outside_count += 1

    total = inside_count + outside_count
    if total > 0:
        pct = 100 * inside_count / total
        print(f"  Inside: {inside_count:,} ({pct:.1f}%)")
        print(f"  Outside: {outside_count:,} ({100-pct:.1f}%)")

    return inside_count, outside_count


def save_layout_json(df: pd.DataFrame, centroids: Dict[str, Tuple[float, float]],
                     polygons: Dict[str, 'Polygon'], output_path: Path):
    """Save the constrained layout to JSON."""
    print(f"\nSaving layout to {output_path}...")

    # Build node positions dict
    node_positions = {}
    for _, row in df.iterrows():
        node_positions[row['node_id']] = {
            'x': float(row['x']),
            'y': float(row['y']),
            'cluster': row['l2_cluster'],
            'was_moved': bool(row.get('was_moved', False))
        }

    # Build centroid positions
    centroid_positions = {
        cid: {'x': float(x), 'y': float(y)}
        for cid, (x, y) in centroids.items()
    }

    # Convert Shapely polygons to coordinate lists for JSON
    voronoi_polygons = {}
    for cid, poly in polygons.items():
        try:
            if hasattr(poly, 'exterior'):
                # Get exterior coordinates as list of [x, y] pairs
                coords = [[float(x), float(y)] for x, y in poly.exterior.coords]
                voronoi_polygons[cid] = coords
        except Exception as e:
            print(f"  Warning: Could not export polygon for {cid}: {e}")

    output_data = {
        'node_positions': node_positions,
        'cluster_centroids': centroid_positions,
        'voronoi_polygons': voronoi_polygons,
        'metadata': {
            'total_nodes': len(node_positions),
            'total_clusters': len(centroid_positions),
            'total_polygons': len(voronoi_polygons),
            'containment_margin': CONTAINMENT_MARGIN,
            'layout_type': 'constrained_voronoi'
        }
    }

    with output_path.open('w') as f:
        json.dump(output_data, f)

    print(f"  Saved {len(node_positions):,} nodes")
    print(f"  Saved {len(voronoi_polygons):,} polygons")


def generate_debug_plot(df: pd.DataFrame,
                        polygons: Dict[str, 'Polygon'],
                        centroids: Dict[str, Tuple[float, float]],
                        output_path: Path):
    """Generate a visualization of the constrained layout."""
    print(f"\nGenerating debug plot...")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        import matplotlib.colors as mcolors
    except ImportError:
        print("  ERROR: matplotlib not installed!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=150)

    # Get unique clusters and assign colors
    unique_clusters = [c for c in df['l2_cluster'].unique() if c != 'unknown']
    n_colors = len(unique_clusters)

    # Generate distinct colors using golden angle
    colors = {}
    for i, cluster_id in enumerate(unique_clusters):
        hue = (i * 137.5) % 360 / 360
        saturation = 0.7 + (i % 3) * 0.1
        lightness = 0.45 + (i % 4) * 0.05
        # Convert HSL to RGB
        import colorsys
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors[cluster_id] = (r, g, b)

    # ----- Left plot: Before (Original UMAP) -----
    ax1 = axes[0]
    ax1.set_title('Before: Original UMAP Layout\n(nodes may leak across boundaries)', fontsize=14)

    # Plot polygons
    for cluster_id, polygon in polygons.items():
        if cluster_id in colors:
            try:
                if hasattr(polygon, 'exterior'):
                    coords = list(polygon.exterior.coords)
                    patch = MplPolygon(coords, closed=True,
                                       facecolor=(*colors[cluster_id], 0.1),
                                       edgecolor=(*colors[cluster_id], 0.8),
                                       linewidth=1)
                    ax1.add_patch(patch)
            except Exception as e:
                pass

    # Plot original node positions
    for cluster_id in unique_clusters:
        cluster_df = df[df['l2_cluster'] == cluster_id]
        if len(cluster_df) > 0:
            ax1.scatter(cluster_df['x_original'], cluster_df['y_original'],
                       c=[colors.get(cluster_id, (0.5, 0.5, 0.5))],
                       s=3, alpha=0.6, label=None)

    # Plot centroids
    for cluster_id, (cx, cy) in centroids.items():
        if cluster_id in colors:
            ax1.scatter(cx, cy, c=[colors[cluster_id]], s=100, marker='*',
                       edgecolors='black', linewidths=0.5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)

    # ----- Right plot: After (Constrained) -----
    ax2 = axes[1]
    ax2.set_title('After: Constrained Voronoi Layout\n(100% containment within boundaries)', fontsize=14)

    # Plot polygons
    for cluster_id, polygon in polygons.items():
        if cluster_id in colors:
            try:
                if hasattr(polygon, 'exterior'):
                    coords = list(polygon.exterior.coords)
                    patch = MplPolygon(coords, closed=True,
                                       facecolor=(*colors[cluster_id], 0.1),
                                       edgecolor=(*colors[cluster_id], 0.8),
                                       linewidth=1)
                    ax2.add_patch(patch)
            except Exception as e:
                pass

    # Plot constrained node positions
    for cluster_id in unique_clusters:
        cluster_df = df[df['l2_cluster'] == cluster_id]
        if len(cluster_df) > 0:
            ax2.scatter(cluster_df['x'], cluster_df['y'],
                       c=[colors.get(cluster_id, (0.5, 0.5, 0.5))],
                       s=3, alpha=0.6, label=None)

    # Plot centroids
    for cluster_id, (cx, cy) in centroids.items():
        if cluster_id in colors:
            ax2.scatter(cx, cy, c=[colors[cluster_id]], s=100, marker='*',
                       edgecolors='black', linewidths=0.5)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)

    # Match axis limits
    all_x = list(df['x']) + list(df['x_original'])
    all_y = list(df['y']) + list(df['y_original'])
    margin = 0.5
    xlim = (min(all_x) - margin, max(all_x) + margin)
    ylim = (min(all_y) - margin, max(all_y) + margin)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    # Add statistics
    moved_count = df['was_moved'].sum()
    total_count = len(df[df['l2_cluster'] != 'unknown'])
    stats_text = (
        f"Total nodes: {len(df):,}\n"
        f"Nodes moved: {moved_count:,} ({100*moved_count/total_count:.1f}%)\n"
        f"Clusters: {len(unique_clusters)}\n"
        f"Containment margin: {CONTAINMENT_MARGIN:.0%}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("CONSTRAINED VORONOI LAYOUT GENERATOR")
    print("Ensuring 100% Node Containment Within Cluster Boundaries")
    print("=" * 70)

    # Step 1: Load data
    df = load_umap_coordinates(UMAP_COORDS_CSV)
    hierarchy_data = load_hierarchy_data(HIERARCHY_PATH)

    # Step 2: Build cluster mapping (for reference)
    entity_to_cluster = build_entity_to_cluster_map(hierarchy_data.get('clusters', {}))

    # Step 3: Compute cluster centroids
    centroids = compute_cluster_centroids(df)

    # Step 4: Generate Voronoi polygons
    bounds = (df['x'].min(), df['y'].min(), df['x'].max(), df['y'].max())
    polygons = generate_voronoi_polygons(centroids, bounds)

    # Step 5: Check and fix containment
    df = check_and_fix_containment(df, polygons, centroids)

    # Step 6: Verify containment
    inside, outside = verify_containment(df, polygons)

    # Step 7: Save outputs
    save_layout_json(df, centroids, polygons, OUTPUT_JSON)
    generate_debug_plot(df, polygons, centroids, OUTPUT_PNG)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  Layout JSON: {OUTPUT_JSON}")
    print(f"  Debug PNG: {OUTPUT_PNG}")
    print("=" * 70)

    # Summary
    print("\nSUMMARY:")
    print(f"  Total nodes: {len(df):,}")
    print(f"  Nodes moved: {df['was_moved'].sum():,}")
    print(f"  Final containment: {inside:,} inside, {outside:,} outside")
    if inside + outside > 0:
        print(f"  Containment rate: {100 * inside / (inside + outside):.2f}%")


if __name__ == "__main__":
    main()
