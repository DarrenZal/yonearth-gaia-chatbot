#!/usr/bin/env python3
"""
GraphRAG Hierarchy Audit Script
Investigates the actual community structure, traces leaf-to-root paths,
and creates robust ID mapping from community_reports.
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

# Paths
BASE_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy")
LEIDEN_COMMUNITIES = BASE_DIR / "checkpoints_microsoft/leiden_communities.json"
COMMUNITY_SUMMARIES = BASE_DIR / "checkpoints_microsoft/community_summaries.json"
LEIDEN_HIERARCHIES = BASE_DIR / "checkpoints/leiden_hierarchies.json"

def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def analyze_microsoft_format():
    """
    STEP 1: Map the Real Hierarchy (Follow the Edges)
    Trace from leaf nodes up to root to determine actual max depth and root count.
    """
    print("=" * 80)
    print("STEP 1: MAP THE REAL HIERARCHY (Follow the Edges)")
    print("=" * 80)

    # Load Microsoft format: [level, community_id, parent_id, nodes]
    data = load_json(LEIDEN_COMMUNITIES)
    communities = data["communities"]

    print(f"\nTotal communities: {len(communities)}")

    # Build parent-child mapping
    # NOTE: Levels are INVERTED - Level 0 = ROOT, Level 3 = LEAF
    community_map = {}  # key: (level, community_id) -> value: community data
    parent_map = {}     # key: (level, community_id) -> value: (parent_level, parent_id)
    children_map = defaultdict(list)  # key: (level, community_id) -> value: list of children

    for comm in communities:
        level, comm_id, parent_id, nodes = comm
        key = (level, comm_id)
        community_map[key] = {
            'level': level,
            'id': comm_id,
            'parent_id': parent_id,
            'nodes': nodes,
            'node_count': len(nodes)
        }

        # Track parent relationship
        # Parent is at level-1 (INVERTED: L3→L2→L1→L0)
        if parent_id != -1:
            parent_key = (level - 1, parent_id)  # Parent is at level-1 (CORRECTED)
            parent_map[key] = parent_key
            children_map[parent_key].append(key)

    # Count nodes by level
    level_counts = Counter(c['level'] for c in community_map.values())
    print(f"\nCommunities by level:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} communities")

    # Find root nodes (parent_id = -1)
    roots = [key for key, comm in community_map.items() if comm['parent_id'] == -1]
    print(f"\n⚠️  ROOT NODES (parent_id = -1): {len(roots)}")
    print(f"    Expected: ~30-50 top-level categories")
    print(f"    Actual: {len(roots)} root communities")

    if len(roots) > 100:
        print(f"    ❌ WARNING: Too many roots! Likely graph fragmentation.")

    # Trace depth from several leaf nodes
    # NOTE: Leaves are at the HIGHEST level number (Level 3), roots at Level 0
    print(f"\n📍 TRACING PATHS FROM LEAF TO ROOT (L3 → L0):")
    max_level = max(v['level'] for v in community_map.values())
    leaf_communities = [k for k, v in community_map.items() if v['level'] == max_level]
    sample_leaves = leaf_communities[:5]  # Sample 5 leaves

    print(f"   Max level found: L{max_level} (leaf level)")
    print(f"   Total leaves at L{max_level}: {len(leaf_communities)}")

    max_depth = 0
    for leaf_key in sample_leaves:
        path = []
        current = leaf_key
        while current in parent_map:
            path.append(current)
            current = parent_map[current]
        path.append(current)  # Add root

        depth = len(path) - 1  # Depth is edges, not nodes
        max_depth = max(max_depth, depth)

        leaf_info = community_map[leaf_key]
        print(f"\n  Leaf: {leaf_key} ({len(leaf_info['nodes'])} nodes)")
        print(f"  Path ({depth} levels):")
        for i, node in enumerate(path):
            info = community_map.get(node, {'level': '?', 'id': '?', 'node_count': 0})
            indent = "    " * i
            print(f"    {indent}L{info['level']}_c{info['id']} ({info['node_count']} nodes)")

    print(f"\n✅ MAX DEPTH: {max_depth} levels (L0 → L{max_depth})")

    return {
        'community_map': community_map,
        'parent_map': parent_map,
        'children_map': children_map,
        'roots': roots,
        'max_depth': max_depth,
        'level_counts': level_counts
    }

def create_robust_mapping():
    """
    STEP 2: Switch to Robust ID Mapping (The Rosetta Stone)
    Create direct dictionary mapping {community_id: title} from community_summaries.json
    """
    print("\n" + "=" * 80)
    print("STEP 2: CREATE ROBUST ID MAPPING (The Rosetta Stone)")
    print("=" * 80)

    summaries = load_json(COMMUNITY_SUMMARIES)

    print(f"\nTotal summaries: {len(summaries)}")

    # Create mapping
    id_to_title = {}
    for comm_id, title in summaries.items():
        id_to_title[int(comm_id)] = title

    print(f"\nSample mappings (first 10):")
    for comm_id in sorted(id_to_title.keys())[:10]:
        print(f"  Community {comm_id}: \"{id_to_title[comm_id]}\"")

    print(f"\n✅ Created robust mapping: {len(id_to_title)} community IDs → titles")
    print(f"   No more index-based guessing!")
    print(f"   No more 'Regenerative Crypto' labeling errors!")

    return id_to_title

def check_fragmentation(hierarchy_data):
    """
    STEP 3: Check for Graph Fragmentation
    Why are there so many roots? Check for disconnected components.
    """
    print("\n" + "=" * 80)
    print("STEP 3: CHECK FOR GRAPH FRAGMENTATION")
    print("=" * 80)

    roots = hierarchy_data['roots']
    community_map = hierarchy_data['community_map']
    children_map = hierarchy_data['children_map']

    print(f"\n🔍 Analyzing {len(roots)} root communities...")

    # Analyze root sizes
    root_sizes = []
    for root_key in roots:
        # Count total descendants
        descendants = count_descendants(root_key, children_map, community_map)
        root_info = community_map[root_key]
        root_sizes.append({
            'key': root_key,
            'level': root_info['level'],
            'id': root_info['id'],
            'direct_nodes': root_info['node_count'],
            'total_descendants': descendants
        })

    # Sort by size
    root_sizes.sort(key=lambda x: x['total_descendants'], reverse=True)

    print(f"\n📊 TOP 10 LARGEST ROOT COMMUNITIES:")
    for i, root in enumerate(root_sizes[:10], 1):
        print(f"  {i}. L{root['level']}_c{root['id']}: "
              f"{root['direct_nodes']} direct nodes, "
              f"{root['total_descendants']} total descendants")

    print(f"\n📊 BOTTOM 10 SMALLEST ROOT COMMUNITIES:")
    for i, root in enumerate(root_sizes[-10:], 1):
        print(f"  {i}. L{root['level']}_c{root['id']}: "
              f"{root['direct_nodes']} direct nodes, "
              f"{root['total_descendants']} total descendants")

    # Check for fragmentation
    large_roots = [r for r in root_sizes if r['total_descendants'] > 100]
    small_roots = [r for r in root_sizes if r['total_descendants'] < 10]

    print(f"\n🌳 FRAGMENTATION ANALYSIS:")
    print(f"  Large components (>100 descendants): {len(large_roots)}")
    print(f"  Small components (<10 descendants): {len(small_roots)}")
    print(f"  Total root components: {len(roots)}")

    if len(small_roots) > 50:
        print(f"\n  ❌ SEVERE FRAGMENTATION DETECTED!")
        print(f"     {len(small_roots)} tiny disconnected components (islands)")
        print(f"     These are likely isolated entities that didn't cluster well")
    elif len(small_roots) > 20:
        print(f"\n  ⚠️  MODERATE FRAGMENTATION")
        print(f"     {len(small_roots)} small disconnected components")
    else:
        print(f"\n  ✅ HEALTHY HIERARCHY")
        print(f"     Most entities are connected in large components")

    return root_sizes

def count_descendants(node_key, children_map, community_map):
    """Recursively count all descendants of a node."""
    total = community_map[node_key]['node_count']
    for child_key in children_map.get(node_key, []):
        total += count_descendants(child_key, children_map, community_map)
    return total

def generate_summary_report(hierarchy_data, id_to_title, root_sizes):
    """Generate final summary report."""
    print("\n" + "=" * 80)
    print("📝 FINAL SUMMARY REPORT")
    print("=" * 80)

    print(f"\n1️⃣  HIERARCHY STRUCTURE:")
    print(f"    Max Depth: {hierarchy_data['max_depth']} levels")
    print(f"    Root Nodes: {len(hierarchy_data['roots'])}")
    print(f"    Total Communities: {sum(hierarchy_data['level_counts'].values())}")

    print(f"\n2️⃣  ID MAPPING:")
    print(f"    Robust mapping created: {len(id_to_title)} community IDs → titles")
    print(f"    Source: community_summaries.json (direct, no guessing)")

    print(f"\n3️⃣  FRAGMENTATION:")
    large = len([r for r in root_sizes if r['total_descendants'] > 100])
    small = len([r for r in root_sizes if r['total_descendants'] < 10])
    print(f"    Large components (>100 nodes): {large}")
    print(f"    Small components (<10 nodes): {small}")

    if small > 50:
        print(f"\n    ❌ RECOMMENDATION: High fragmentation detected")
        print(f"       Consider re-running Leiden with lower resolution parameter")
    elif len(hierarchy_data['roots']) > 200:
        print(f"\n    ⚠️  RECOMMENDATION: Many root nodes")
        print(f"       Expected ~30-50, got {len(hierarchy_data['roots'])}")
        print(f"       May need to add another hierarchical level")
    else:
        print(f"\n    ✅ STRUCTURE LOOKS REASONABLE")

    # Save mapping to file
    output_file = BASE_DIR / "community_id_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(id_to_title, f, indent=2)
    print(f"\n💾 SAVED: Robust ID mapping to {output_file}")

def main():
    """Run the complete audit."""
    print("\n🔬 GraphRAG Hierarchy Audit")
    print(f"   Data Directory: {BASE_DIR}")
    print(f"   Source: {LEIDEN_COMMUNITIES.name}")

    try:
        # Step 1: Map hierarchy
        hierarchy_data = analyze_microsoft_format()

        # Step 2: Create robust mapping
        id_to_title = create_robust_mapping()

        # Step 3: Check fragmentation
        root_sizes = check_fragmentation(hierarchy_data)

        # Final summary
        generate_summary_report(hierarchy_data, id_to_title, root_sizes)

        print("\n✅ Audit complete!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
