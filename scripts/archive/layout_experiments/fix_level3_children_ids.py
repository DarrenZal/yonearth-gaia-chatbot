#!/usr/bin/env python3
"""
Fix Level 3 children IDs to match actual hierarchy structure.

Problem: Level 3 clusters have children like ['level_2_2760', ...]
but the hierarchy stores these as level_1_XXXX.

The original Leiden script created:
- level_0: individual entities (2028)
- level_1: fine communities (676)
- level_2: mid-level topics (57)

Our cluster_registry calls these:
- level_0, level_1, level_2, level_3

But we need to map level_2_XXXX IDs to level_1_XXXX IDs in the hierarchy.
"""

import json
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
HIERARCHY_PATH = DATA_DIR / "graphrag_hierarchy" / "graphrag_hierarchy.json"
REGISTRY_PATH = DATA_DIR / "graphrag_hierarchy" / "cluster_registry.json"

def main():
    print("=" * 80)
    print("Fixing Level 3 Children IDs")
    print("=" * 80)

    # Load files
    print(f"\nLoading hierarchy from {HIERARCHY_PATH}")
    with HIERARCHY_PATH.open("r") as f:
        hierarchy = json.load(f)

    print(f"Loading registry from {REGISTRY_PATH}")
    with REGISTRY_PATH.open("r") as f:
        registry = json.load(f)

    # Build mapping from registry level_2 IDs to hierarchy level_1 IDs
    # Registry level 2 clusters have 57 items
    # Hierarchy level_1 has 676 items (but only 57 are the mid-level ones from Leiden level 2)
    # Hierarchy level_2 has 57 items - wait, that's the right one!

    print("\nAnalyzing cluster structure...")
    print(f"  Hierarchy levels: {list(hierarchy['clusters'].keys())}")
    print(f"  Hierarchy level_1: {len(hierarchy['clusters']['level_1'])} clusters")
    print(f"  Hierarchy level_2: {len(hierarchy['clusters']['level_2'])} clusters")

    registry_l2_ids = [c['id'] for c in registry.values() if c['level'] == 2]
    print(f"  Registry level 2: {len(registry_l2_ids)} clusters")
    print(f"  Sample registry L2 IDs: {registry_l2_ids[:3]}")

    hierarchy_l2_ids = list(hierarchy['clusters']['level_2'].keys())
    print(f"  Sample hierarchy level_2 IDs: {hierarchy_l2_ids[:3]}")

    # The issue: hierarchy stores Leiden level 2 (57 clusters) as "level_2"
    # So level_3 children should reference level_2, not level_1!
    # Let me check what IDs the level_2 clusters have

    print("\nChecking ID format mismatch...")
    # Registry L2 IDs look like: level_2_2760
    # Hierarchy L2 IDs look like: level_1_2079

    # The real issue is the naming convention mismatch
    # We need to map registry level_2_XXXX to whatever exists in hierarchy level_2

    # Simple solution: Map by position/index
    # Or better: check entity overlap

    # Let's try a different approach: update Level 3 children to use hierarchy level_2 IDs
    # by matching entity contents

    print("\nBuilding entity-based mapping...")
    registry_l2_by_entities = {}
    for cid, cdata in registry.items():
        if cdata['level'] == 2:
            # Use frozenset of entities as key
            entity_set = frozenset(cdata['entities'])
            registry_l2_by_entities[entity_set] = cid

    hierarchy_l2_to_registry_l2 = {}
    for h_l2_id, h_l2_data in hierarchy['clusters']['level_2'].items():
        h_entities = frozenset(h_l2_data.get('entities', []))
        if h_entities in registry_l2_by_entities:
            reg_id = registry_l2_by_entities[h_entities]
            hierarchy_l2_to_registry_l2[h_l2_id] = reg_id
            print(f"  Mapped {h_l2_id} -> {reg_id}")

    print(f"\nMapped {len(hierarchy_l2_to_registry_l2)} clusters")

    # Now create reverse mapping
    registry_l2_to_hierarchy_l2 = {v: k for k, v in hierarchy_l2_to_registry_l2.items()}

    # Update Level 3 children in hierarchy
    print("\nUpdating Level 3 children IDs in hierarchy...")
    level_3 = hierarchy['clusters']['level_3']

    for l3_id, l3_data in level_3.items():
        old_children = l3_data.get('children', [])
        new_children = []

        for old_child_id in old_children:
            if old_child_id in registry_l2_to_hierarchy_l2:
                new_child_id = registry_l2_to_hierarchy_l2[old_child_id]
                new_children.append(new_child_id)
            else:
                print(f"  WARNING: Could not map {old_child_id}")
                new_children.append(old_child_id)  # Keep as-is

        l3_data['children'] = new_children
        print(f"  {l3_id}: {len(old_children)} children -> {len(new_children)} mapped")

    # Save updated hierarchy
    print(f"\nSaving updated hierarchy to {HIERARCHY_PATH}")
    with HIERARCHY_PATH.open("w") as f:
        json.dump(hierarchy, f)

    print("\n" + "=" * 80)
    print("✓ Level 3 children IDs fixed!")
    print("=" * 80)

    print("\nNext steps:")
    print("  1. Deploy to production:")
    print(f"     sudo cp {HIERARCHY_PATH} /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/")
    print(f"     sudo systemctl reload nginx")


if __name__ == "__main__":
    main()
