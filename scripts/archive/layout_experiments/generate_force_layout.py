#!/usr/bin/env python3
"""
Generate pre-computed force-directed layout positions for the knowledge graph.

This script runs a force-directed simulation offline and saves the resulting
positions, so the 3D viewer can display the structural view without running
real-time physics simulation.
"""

import json
import numpy as np
from pathlib import Path
import networkx as nx

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "graphrag_hierarchy"
HIERARCHY_FILE = DATA_DIR / "graphrag_hierarchy.json"
OUTPUT_FILE = DATA_DIR / "force_layout.json"


def load_graph_data():
    """Load entities and relationships from the hierarchy file."""
    print(f"Loading graph data from {HIERARCHY_FILE}...")

    with open(HIERARCHY_FILE, 'r') as f:
        data = json.load(f)

    entities = data.get('entities', [])
    relationships = data.get('relationships', [])

    print(f"Loaded {len(entities)} entities and {len(relationships)} relationships")
    return entities, relationships


def build_networkx_graph(entities, relationships):
    """Build a NetworkX graph from entities and relationships."""
    print("Building NetworkX graph...")

    G = nx.Graph()

    # Entities is a dict where keys are entity names/IDs
    entity_ids = set()
    if isinstance(entities, dict):
        for entity_id, entity_data in entities.items():
            entity_ids.add(entity_id)
            G.add_node(entity_id, **(entity_data if isinstance(entity_data, dict) else {}))
    else:
        # Handle list format
        for entity in entities:
            entity_id = entity.get('id') or entity.get('name')
            if entity_id:
                entity_ids.add(entity_id)
                G.add_node(entity_id, **entity)

    # Add edges
    edge_count = 0
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        if source in entity_ids and target in entity_ids:
            weight = rel.get('weight', 1.0)
            G.add_edge(source, target, weight=weight)
            edge_count += 1

    print(f"Built graph with {G.number_of_nodes()} nodes and {edge_count} edges")
    return G


def compute_force_layout(G, iterations=100, k=None, seed=42):
    """
    Compute 3D force-directed layout using spring layout algorithm.

    Args:
        G: NetworkX graph
        iterations: Number of iterations for the spring layout
        k: Optimal distance between nodes (None for automatic)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node IDs to [x, y, z] positions
    """
    print(f"Computing 3D force-directed layout ({iterations} iterations)...")

    if G.number_of_nodes() == 0:
        return {}

    # Use spring_layout with 3 dimensions
    # k controls the optimal distance between nodes
    if k is None:
        k = 1.0 / np.sqrt(G.number_of_nodes()) * 100  # Scale for visualization

    # Compute layout
    pos_3d = nx.spring_layout(
        G,
        dim=3,
        k=k,
        iterations=iterations,
        seed=seed,
        weight='weight'
    )

    # Scale positions for better visualization (similar to other layouts)
    positions = np.array(list(pos_3d.values()))

    # Center and scale
    center = positions.mean(axis=0)
    positions = positions - center

    # Scale to roughly match other layout ranges (around 100 units)
    max_extent = np.abs(positions).max()
    if max_extent > 0:
        scale_factor = 150.0 / max_extent
        positions = positions * scale_factor

    # Convert back to dictionary with lists (JSON serializable)
    layout = {}
    for i, node_id in enumerate(pos_3d.keys()):
        layout[node_id] = positions[i].tolist()

    print(f"Layout computed for {len(layout)} nodes")
    return layout


def compute_layout_with_fruchterman_reingold(G, iterations=500, seed=42):
    """
    Alternative: Use Fruchterman-Reingold algorithm which can give better results
    for larger graphs.
    """
    print(f"Computing 3D Fruchterman-Reingold layout ({iterations} iterations)...")

    if G.number_of_nodes() == 0:
        return {}

    # For very large graphs, we might want to use a subset or different approach
    n_nodes = G.number_of_nodes()

    # Adjust k based on graph size
    k = 2.0 / np.sqrt(n_nodes) * 100

    pos_3d = nx.spring_layout(
        G,
        dim=3,
        k=k,
        iterations=iterations,
        seed=seed,
        weight='weight',
        scale=100.0  # Scale factor
    )

    # Convert to list format
    layout = {node_id: list(pos) for node_id, pos in pos_3d.items()}

    print(f"Layout computed for {len(layout)} nodes")
    return layout


def save_layout(layout, output_path):
    """Save the layout to a JSON file."""
    print(f"Saving layout to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(layout, f)

    # Calculate file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(layout)} positions ({size_mb:.2f} MB)")


def main():
    """Main function to generate force-directed layout."""
    print("=" * 60)
    print("Force-Directed Layout Generator")
    print("=" * 60)

    # Load data
    entities, relationships = load_graph_data()

    # Build graph
    G = build_networkx_graph(entities, relationships)

    # Compute layout - using more iterations for better convergence
    # For ~39k nodes, this will take a few minutes
    layout = compute_force_layout(G, iterations=150, seed=42)

    # Save layout
    save_layout(layout, OUTPUT_FILE)

    print("=" * 60)
    print("Done! Force layout saved to:", OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
