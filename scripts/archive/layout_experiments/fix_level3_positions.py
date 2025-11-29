#!/usr/bin/env python3
"""
Fix Level 3 cluster positions in graphrag_hierarchy.json

The Level 3 clusters currently have position=[0,0,0] because we couldn't find
their L2 children's positions. This script calculates proper 3D positions for
L3 clusters by averaging the force layout positions of all their entities.
"""

import json
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
HIERARCHY_PATH = DATA_DIR / "graphrag_hierarchy" / "graphrag_hierarchy.json"
FORCE_LAYOUT_PATH = DATA_DIR / "graphrag_hierarchy" / "force_layout.json"

def main():
    print("=" * 80)
    print("Fixing Level 3 Cluster Positions")
    print("=" * 80)

    # Load data
    print(f"\nLoading hierarchy from {HIERARCHY_PATH}")
    with HIERARCHY_PATH.open("r") as f:
        hierarchy = json.load(f)

    print(f"Loading force layout from {FORCE_LAYOUT_PATH}")
    with FORCE_LAYOUT_PATH.open("r") as f:
        force_layout = json.load(f)

    print(f"Force layout has {len(force_layout)} entity positions")

    # Fix Level 3 positions
    level_3 = hierarchy["clusters"].get("level_3", {})
    print(f"\nFound {len(level_3)} Level 3 clusters")

    for cluster_id, cluster_data in level_3.items():
        entity_ids = cluster_data.get("entity_ids", [])
        print(f"\nProcessing {cluster_id}: {cluster_data.get('name', cluster_id)}")
        print(f"  Has {len(entity_ids)} entities")

        # Get positions for all entities in this cluster
        positions = []
        for entity_id in entity_ids:
            if entity_id in force_layout:
                pos = force_layout[entity_id]
                if len(pos) == 3:
                    positions.append(pos)

        if positions:
            # Calculate centroid
            centroid = np.mean(positions, axis=0).tolist()
            print(f"  Found {len(positions)} entity positions")
            print(f"  Calculated centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")

            # Update cluster
            cluster_data["position"] = centroid
            cluster_data["umap_position"] = centroid
        else:
            print(f"  WARNING: No positions found, keeping [0, 0, 0]")

    # Save updated hierarchy
    print(f"\nSaving updated hierarchy to {HIERARCHY_PATH}")
    with HIERARCHY_PATH.open("w") as f:
        json.dump(hierarchy, f)

    print("\n" + "=" * 80)
    print("✓ Level 3 positions fixed!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Deploy to production:")
    print(f"     sudo cp {HIERARCHY_PATH} /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/")
    print(f"     sudo systemctl reload nginx")


if __name__ == "__main__":
    main()
