#!/usr/bin/env python3
"""
Post-processing script to fix hierarchy children assignments.
Uses position-based (UMAP coordinates) assignment to ensure each child cluster
belongs to exactly ONE parent, fixing the duplicate children bug.

The bug: Hovering over any top-level cluster shows the SAME nested shapes
because multiple L3 clusters share the same children (only 5 unique sets
among 57 clusters).

The fix: Assign each mid-level cluster (L2) to the nearest coarse cluster (L3)
using Euclidean distance in UMAP 3D space.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy")
HIERARCHY_FILE = DATA_DIR / "graphrag_hierarchy.json"
OUTPUT_FILE = DATA_DIR / "graphrag_hierarchy.json"  # Overwrite in place
BACKUP_FILE = DATA_DIR / "graphrag_hierarchy_backup.json"


def analyze_children_uniqueness(clusters: dict, level_key: str) -> tuple:
    """Analyze how many unique children sets exist at a level."""
    level_clusters = clusters.get(level_key, {})
    children_sets = []
    for cid, cdata in level_clusters.items():
        children = cdata.get("children", [])
        children_sets.append(tuple(sorted(children)))
    unique_sets = set(children_sets)
    return len(unique_sets), len(level_clusters)


def get_cluster_position(cluster_data: dict) -> np.ndarray:
    """Extract 3D position from cluster data."""
    # Position is stored as an array [x, y, z], not individual keys
    pos = cluster_data.get("position") or cluster_data.get("umap_position")
    if pos and len(pos) >= 3:
        return np.array(pos[:3])
    return np.array([0, 0, 0])


def fix_hierarchy_with_positions(hierarchy: dict) -> dict:
    """
    Fix children assignments using position-based nearest-neighbor assignment.

    Strategy:
    1. For each mid-level (L2) cluster, find the nearest coarse (L3) cluster
    2. Assign the L2 cluster as a child of that L3 cluster
    3. Repeat for L1 → L2 assignments
    """
    clusters = hierarchy["clusters"]

    # Get cluster data at each level
    l3_clusters = clusters.get("level_3", {})  # Coarse (57 clusters)
    l2_clusters = clusters.get("level_2", {})  # Mid (681 clusters)
    l1_clusters = clusters.get("level_1", {})  # Fine (2073 clusters)

    print(f"Level 3 (coarse): {len(l3_clusters)} clusters")
    print(f"Level 2 (mid): {len(l2_clusters)} clusters")
    print(f"Level 1 (fine): {len(l1_clusters)} clusters")

    # Check initial state
    unique_before, total = analyze_children_uniqueness(clusters, "level_3")
    print(f"\nBefore fix: {unique_before}/{total} unique L3 children sets")

    # === Fix L2 → L3 assignments ===
    print("\n=== Fixing L2 → L3 (mid to coarse) assignments ===")

    # Build L3 positions array
    l3_ids = list(l3_clusters.keys())
    l3_positions = np.array([get_cluster_position(l3_clusters[cid]) for cid in l3_ids])

    # Clear existing L3 children
    for cid in l3_clusters:
        l3_clusters[cid]["children"] = []

    # Assign each L2 cluster to nearest L3 cluster
    assigned_count = 0
    for l2_id, l2_data in l2_clusters.items():
        l2_pos = get_cluster_position(l2_data)

        # Find nearest L3 cluster
        distances = np.linalg.norm(l3_positions - l2_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_l3_id = l3_ids[nearest_idx]

        # Add L2 as child of nearest L3
        l3_clusters[nearest_l3_id]["children"].append(l2_id)
        assigned_count += 1

    print(f"Assigned {assigned_count} L2 clusters to L3 parents")

    # === Fix L1 → L2 assignments ===
    print("\n=== Fixing L1 → L2 (fine to mid) assignments ===")

    # Build L2 positions array
    l2_ids = list(l2_clusters.keys())
    l2_positions = np.array([get_cluster_position(l2_clusters[cid]) for cid in l2_ids])

    # Clear existing L2 children
    for cid in l2_clusters:
        l2_clusters[cid]["children"] = []

    # Assign each L1 cluster to nearest L2 cluster
    assigned_count = 0
    for l1_id, l1_data in l1_clusters.items():
        l1_pos = get_cluster_position(l1_data)

        # Find nearest L2 cluster
        distances = np.linalg.norm(l2_positions - l1_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_l2_id = l2_ids[nearest_idx]

        # Add L1 as child of nearest L2
        l2_clusters[nearest_l2_id]["children"].append(l1_id)
        assigned_count += 1

    print(f"Assigned {assigned_count} L1 clusters to L2 parents")

    # Verify fix
    unique_l3, total_l3 = analyze_children_uniqueness(clusters, "level_3")
    unique_l2, total_l2 = analyze_children_uniqueness(clusters, "level_2")

    print(f"\n=== Results ===")
    print(f"L3 unique children sets: {unique_l3}/{total_l3} ({100*unique_l3/total_l3:.1f}%)")
    print(f"L2 unique children sets: {unique_l2}/{total_l2} ({100*unique_l2/total_l2:.1f}%)")

    # Show distribution of children per L3 cluster
    print("\n=== L3 Children Distribution ===")
    child_counts = [(cid, len(l3_clusters[cid]["children"]), l3_clusters[cid].get("title", "Untitled"))
                    for cid in l3_ids]
    child_counts.sort(key=lambda x: -x[1])

    for cid, count, title in child_counts[:10]:
        print(f"  {cid}: {count} children - {title[:50]}")
    print(f"  ... and {len(child_counts) - 10} more")

    # Check for any L3 with 0 children
    empty_l3 = [cid for cid in l3_ids if len(l3_clusters[cid]["children"]) == 0]
    if empty_l3:
        print(f"\nWARNING: {len(empty_l3)} L3 clusters have 0 children!")
        for cid in empty_l3[:5]:
            print(f"  {cid}: position {get_cluster_position(l3_clusters[cid])}")

    return hierarchy


def main():
    print("=" * 70)
    print("GraphRAG Hierarchy Children Fix (Position-Based Assignment)")
    print("=" * 70)

    # Load hierarchy
    print(f"\nLoading hierarchy from {HIERARCHY_FILE}")
    with open(HIERARCHY_FILE) as f:
        hierarchy = json.load(f)

    # Create backup
    print(f"Creating backup at {BACKUP_FILE}")
    with open(BACKUP_FILE, 'w') as f:
        json.dump(hierarchy, f)

    # Apply fix
    fixed_hierarchy = fix_hierarchy_with_positions(hierarchy)

    # Check success threshold (accept >= 90% unique)
    clusters = fixed_hierarchy["clusters"]
    unique_l3, total_l3 = analyze_children_uniqueness(clusters, "level_3")
    success_rate = unique_l3 / total_l3

    if success_rate >= 0.90:
        print(f"\n✓ SUCCESS: {unique_l3}/{total_l3} unique children sets ({100*success_rate:.1f}%)")
        print(f"Saving fixed hierarchy to {OUTPUT_FILE}")

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(fixed_hierarchy, f, indent=2)

        print("Done! Hierarchy has been updated.")
        return True
    else:
        print(f"\n✗ FAILED: Only {unique_l3}/{total_l3} unique children sets ({100*success_rate:.1f}%)")
        print("Not saving - threshold not met.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
