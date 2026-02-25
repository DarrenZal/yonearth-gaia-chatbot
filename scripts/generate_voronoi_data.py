#!/usr/bin/env python3
"""
Generate compact Voronoi visualization data from the full graphrag_hierarchy.json.

Creates two output files:
- voronoi_hierarchy.json: Recursive hierarchy for drill-down visualization
- voronoi_entity_index.json: Flat lookup for entity search

The hierarchy from Hierarchical Leiden has:
- level_1 (574 clusters): 36 large clusters with level_2 children, 538 tiny standalone clusters
- level_2 (621 clusters): mid-level clusters with level_3 children or entity children
- level_3 (24 clusters): finest clusters with entity children

Strategy:
1. Use the 36 level_1 clusters that have children as top-level Voronoi groups
2. Merge 538 standalone clusters into nearest large group by position
3. Walk level_2 → level_3 → entities for drill-down
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# Color palette for top-level groups
GROUP_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#e6beff',
    '#aa6e28', '#fffac8', '#d2f53c', '#fabebe', '#008080',
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3',
]


def load_hierarchy(path: Path) -> dict:
    print(f"Loading {path} ...")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"  Entities: {len(data['entities'])}")
    print(f"  Relationships: {len(data['relationships'])}")
    for level_key in ['level_1', 'level_2', 'level_3']:
        count = len(data['clusters'].get(level_key, {}))
        print(f"  {level_key}: {count} clusters")
    return data


def get_top_entities(entity_names, entities, level_0, max_count=20):
    """Get the top entities by mention count."""
    scored = []
    for name in entity_names:
        ent = entities.get(name, {})
        pos_data = level_0.get(name, {})
        pos = pos_data.get('position', pos_data.get('umap_position', [0, 0, 0]))
        scored.append({
            'name': name,
            'type': ent.get('type', 'UNKNOWN'),
            'description': (ent.get('description', '') or '')[:200],
            'mentions': ent.get('mention_count', 0),
            'x': pos[0],
            'y': pos[1],
            'connections': pos_data.get('betweenness', 0),
        })
    scored.sort(key=lambda e: e['mentions'], reverse=True)
    return scored[:max_count]


def compute_centroid(entity_names, level_0):
    """Compute 2D centroid from entity positions."""
    xs, ys = [], []
    for name in entity_names:
        pos_data = level_0.get(name, {})
        pos = pos_data.get('position', pos_data.get('umap_position', None))
        if pos:
            xs.append(pos[0])
            ys.append(pos[1])
    if xs:
        return [sum(xs) / len(xs), sum(ys) / len(ys)]
    return [0, 0]


def nearest_group(pos, group_centroids):
    """Find the nearest group index by 2D distance."""
    best_idx = 0
    best_dist = float('inf')
    for i, gc in enumerate(group_centroids):
        d = (pos[0] - gc[0])**2 + (pos[1] - gc[1])**2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def build_child_node(cluster_id, cluster, entities, level_0, all_clusters):
    """
    Build a node for a level_2 or level_3 cluster.
    Level_2 clusters may have level_3 children (cluster IDs) or entity children.
    Level_3 clusters always have entity children.
    """
    children_ids = cluster.get('children', [])
    entity_names = cluster.get('entities', [])

    pos = cluster.get('position', cluster.get('umap_position', None))
    if pos:
        centroid = [pos[0], pos[1]]
    else:
        centroid = compute_centroid(entity_names, level_0)

    node = {
        'id': cluster_id,
        'name': cluster.get('name', cluster.get('title', cluster_id)),
        'summary': (cluster.get('summary_text', '') or '')[:300],
        'entityCount': len(entity_names),
        'x': centroid[0],
        'y': centroid[1],
    }

    # Check if children are cluster IDs (level_3) or entity names
    child_clusters = []
    for child_id in children_ids:
        if child_id in all_clusters.get('level_3', {}):
            child_clusters.append((child_id, all_clusters['level_3'][child_id]))

    if child_clusters:
        # Has sub-clusters — one more level of drill-down
        node['children'] = []
        for child_id, child_cluster in child_clusters:
            child_node = build_leaf_node(child_id, child_cluster, entities, level_0)
            node['children'].append(child_node)
        node['children'].sort(key=lambda c: c['entityCount'], reverse=True)
    else:
        # Leaf — show entities
        node['entities'] = get_top_entities(entity_names, entities, level_0, 20)

    return node


def build_leaf_node(cluster_id, cluster, entities, level_0):
    """Build a leaf node (level_3 or any cluster with only entity children)."""
    entity_names = cluster.get('entities', [])
    pos = cluster.get('position', cluster.get('umap_position', None))
    if pos:
        centroid = [pos[0], pos[1]]
    else:
        centroid = compute_centroid(entity_names, level_0)

    return {
        'id': cluster_id,
        'name': cluster.get('name', cluster.get('title', cluster_id)),
        'summary': (cluster.get('summary_text', '') or '')[:300],
        'entityCount': len(entity_names),
        'x': centroid[0],
        'y': centroid[1],
        'entities': get_top_entities(entity_names, entities, level_0, 20),
    }


def build_voronoi_data(data: dict) -> tuple:
    """Build the voronoi hierarchy and entity index from Leiden hierarchy."""
    entities = data['entities']
    clusters = data['clusters']
    level_0 = clusters.get('level_0', {})
    level_1 = clusters.get('level_1', {})
    level_2 = clusters.get('level_2', {})

    # Separate level_1 into groups with children (top-level) and standalones
    top_groups = {}
    standalones = {}
    for cid, cluster in level_1.items():
        children = cluster.get('children', [])
        # Children that are level_2 cluster IDs (not entity names)
        has_cluster_children = any(c in level_2 for c in children)
        if has_cluster_children:
            top_groups[cid] = cluster
        else:
            standalones[cid] = cluster

    print(f"\n  Top-level groups (have sub-clusters): {len(top_groups)}")
    print(f"  Standalone clusters: {len(standalones)}")

    # Build top-level group nodes with their level_2 children
    groups = []
    group_centroids = []
    for i, (cid, cluster) in enumerate(top_groups.items()):
        entity_names = cluster.get('entities', [])
        centroid = compute_centroid(entity_names, level_0)
        group_centroids.append(centroid)

        # Build children from level_2 clusters
        children_ids = cluster.get('children', [])
        children = []
        for child_id in children_ids:
            if child_id in level_2:
                child_node = build_child_node(
                    child_id, level_2[child_id], entities, level_0, clusters
                )
                children.append(child_node)

        # Sort children by entity count
        children.sort(key=lambda c: c['entityCount'], reverse=True)

        groups.append({
            'id': cid,
            'name': cluster.get('name', cluster.get('title', cid)),
            'summary': (cluster.get('summary_text', '') or '')[:300],
            'entityCount': len(entity_names),
            'color': GROUP_COLORS[i % len(GROUP_COLORS)],
            'x': centroid[0],
            'y': centroid[1],
            'children': children,
        })

    # Merge standalone clusters into nearest top-level group
    merged_count = 0
    merged_entities = 0
    for cid, cluster in standalones.items():
        entity_names = cluster.get('entities', [])
        if not entity_names:
            continue

        centroid = compute_centroid(entity_names, level_0)
        nearest_idx = nearest_group(centroid, group_centroids)

        # Add as a leaf child to the nearest group
        leaf = build_leaf_node(cid, cluster, entities, level_0)
        groups[nearest_idx]['children'].append(leaf)
        groups[nearest_idx]['entityCount'] += len(entity_names)
        merged_count += 1
        merged_entities += len(entity_names)

    print(f"  Merged {merged_count} standalone clusters ({merged_entities} entities) into nearest groups")

    # Sort groups by entity count descending
    groups.sort(key=lambda g: g['entityCount'], reverse=True)

    # Build entity index for search
    entity_index = {}

    def index_entities(node, group_info, parent_path=""):
        """Recursively index entities from all levels."""
        path = f"{parent_path} > {node['name']}" if parent_path else node['name']

        if 'entities' in node:
            for ent in node['entities']:
                entity_index[ent['name']] = {
                    'group': group_info['id'],
                    'groupName': group_info['name'],
                    'cluster': node['id'],
                    'clusterName': node['name'],
                    'path': path,
                    'type': ent['type'],
                    'x': ent['x'],
                    'y': ent['y'],
                }
        if 'children' in node:
            for child in node['children']:
                index_entities(child, group_info, path)

    for group in groups:
        index_entities(group, group)

    # Compute max depth
    def max_depth(node, d=0):
        if 'children' in node:
            child_depths = [max_depth(c, d + 1) for c in node['children']]
            return max(child_depths) if child_depths else d
        return d

    depths = [max_depth(g) for g in groups]
    max_d = max(depths) if depths else 0

    hierarchy = {
        'groups': groups,
        'metadata': {
            'totalEntities': len(entities),
            'totalGroups': len(groups),
            'maxDepth': max_d + 1,
            'hierarchyType': 'natural_leiden',
            'standalonesMerged': merged_count,
        }
    }

    return hierarchy, entity_index


def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / 'data' / 'graphrag_hierarchy.json'

    if not input_path.exists():
        alt_path = base_dir / 'data' / 'graphrag_hierarchy' / 'graphrag_hierarchy.json'
        if alt_path.exists():
            input_path = alt_path

    output_dir = base_dir / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    hierarchy_path = output_dir / 'voronoi_hierarchy.json'
    index_path = output_dir / 'voronoi_entity_index.json'

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    data = load_hierarchy(input_path)
    hierarchy, entity_index = build_voronoi_data(data)

    print(f"\nWriting {hierarchy_path} ...")
    with open(hierarchy_path, 'w') as f:
        json.dump(hierarchy, f, separators=(',', ':'))
    h_size = hierarchy_path.stat().st_size
    print(f"  Size: {h_size / 1024:.1f} KB")

    print(f"Writing {index_path} ...")
    with open(index_path, 'w') as f:
        json.dump(entity_index, f, separators=(',', ':'))
    i_size = index_path.stat().st_size
    print(f"  Size: {i_size / 1024:.1f} KB")

    # Print summary
    print(f"\nDone!")
    print(f"  Groups: {hierarchy['metadata']['totalGroups']}")
    print(f"  Max depth: {hierarchy['metadata']['maxDepth']}")
    print(f"  Indexed entities: {len(entity_index)}")
    print(f"\n  Top 10 groups:")
    for g in hierarchy['groups'][:10]:
        n_children = len(g.get('children', []))
        print(f"    {g['name']}: {g['entityCount']} entities, {n_children} sub-clusters")


if __name__ == '__main__':
    main()
