#!/usr/bin/env python3
"""
Generate a subset of the knowledge graph with well-connected entities.
This creates a smaller initial load file for lazy-loading the full graph.

Uses connectivity threshold (not arbitrary count) to ensure meaningful graph structure.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Set
from collections import Counter


def generate_subset(
    input_path: Path,
    output_path: Path,
    min_connections: int = 20,
    include_links: bool = True
) -> Dict[str, Any]:
    """
    Generate a subset of the knowledge graph with well-connected entities.

    Args:
        input_path: Path to full graphrag_hierarchy.json
        output_path: Path for output subset file
        min_connections: Minimum connections to include (default: 20)
        include_links: Whether to include links between selected entities

    Returns:
        Stats about the generated subset
    """
    print(f"Loading full graph from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    entities = data.get('entities', {})
    relationships = data.get('relationships', [])
    clusters = data.get('clusters', {})
    metadata = data.get('metadata', {})

    print(f"Full graph: {len(entities)} entities, {len(relationships)} relationships")

    # Calculate connectivity for each entity
    source_counts = Counter(r.get('source', '') for r in relationships)
    target_counts = Counter(r.get('target', '') for r in relationships)
    connectivity = {}
    for e in entities.keys():
        connectivity[e] = source_counts.get(e, 0) + target_counts.get(e, 0)

    # Filter by minimum connections
    top_entities = {
        name: {**info, 'connections': connectivity[name]}
        for name, info in entities.items()
        if connectivity[name] >= min_connections
    }
    top_entity_names = set(top_entities.keys())

    # Sort by connectivity for reporting
    sorted_by_conn = sorted(top_entities.items(), key=lambda x: x[1]['connections'], reverse=True)

    print(f"Selected {len(top_entities)} entities with >= {min_connections} connections")
    if sorted_by_conn:
        print(f"Connection range: {sorted_by_conn[0][1]['connections']} - {sorted_by_conn[-1][1]['connections']}")

    # Filter relationships to only include those between top entities
    subset_relationships = []
    if include_links:
        for rel in relationships:
            source = rel.get('source', '')
            target = rel.get('target', '')
            if source in top_entity_names and target in top_entity_names:
                subset_relationships.append(rel)
        print(f"Included {len(subset_relationships)} relationships between top entities")

    # Include cluster hierarchy for entities in subset
    # The frontend needs clusters.level_0 with position data
    subset_clusters = {}

    for level_key in ['level_0', 'level_1', 'level_2', 'level_3', 'L0', 'L1', 'L2', 'L3']:
        if level_key in clusters:
            level_data = clusters[level_key]
            if isinstance(level_data, dict):
                if level_key in ['level_0', 'L0']:
                    # For level_0, filter to only include top entities
                    subset_level = {
                        k: v for k, v in level_data.items()
                        if k in top_entity_names
                    }
                else:
                    # For higher levels, include all (they're cluster summaries)
                    subset_level = level_data
                subset_clusters[level_key] = subset_level
                print(f"  {level_key}: {len(subset_level)} entries")

    # Build subset
    subset = {
        'entities': top_entities,
        'relationships': subset_relationships,
        'clusters': subset_clusters,
        'metadata': {
            **metadata,
            'subset': True,
            'min_connections': min_connections,
            'full_entity_count': len(entities),
            'full_relationship_count': len(relationships),
            'subset_entity_count': len(top_entities),
            'subset_relationship_count': len(subset_relationships)
        }
    }

    # Write output
    print(f"Writing subset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(subset, f)

    # Calculate file sizes
    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)

    stats = {
        'input_entities': len(entities),
        'input_relationships': len(relationships),
        'output_entities': len(top_entities),
        'output_relationships': len(subset_relationships),
        'output_clusters': len(subset_clusters),
        'input_size_mb': round(input_size, 2),
        'output_size_mb': round(output_size, 2),
        'size_reduction': f"{round((1 - output_size/input_size) * 100, 1)}%"
    }

    print(f"\nStats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate KG subset for lazy loading')
    parser.add_argument('--input', '-i', required=True, help='Input graphrag_hierarchy.json path')
    parser.add_argument('--output', '-o', required=True, help='Output subset file path')
    parser.add_argument('--min-connections', '-c', type=int, default=20,
                        help='Minimum connections to include (default: 20, gives ~400 entities)')
    parser.add_argument('--no-links', action='store_true', help='Exclude relationships')

    args = parser.parse_args()

    generate_subset(
        Path(args.input),
        Path(args.output),
        min_connections=args.min_connections,
        include_links=not args.no_links
    )


if __name__ == '__main__':
    main()
