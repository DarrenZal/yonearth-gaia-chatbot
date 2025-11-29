#!/usr/bin/env python3
"""
Diagnostic Scatter Plot for Layout Algorithm Sanity Check.

This script generates Node2Vec + UMAP coordinates for all ~17,000 entities
and visualizes them as a scatter plot colored by Level 2 community.

Purpose: Verify that the layout algorithm produces distinct clusters/islands
rather than a single dense hairball before we add complex Voronoi polygons.

Pipeline:
1. Load GraphRAG entities and relationships
2. Build NetworkX graph
3. Generate Node2Vec structural embeddings (64-dim)
4. Reduce to 2D using UMAP with relaxed parameters
5. Save coordinates to CSV
6. Generate visualization as PNG

Output:
  - data/graphrag_hierarchy/node_layout_coordinates.csv
  - data/graphrag_hierarchy/layout_debug.png
"""

import json
import os
import sys
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx

# --------------------------------------------------------------------------------------
# Paths & Parameters
# --------------------------------------------------------------------------------------

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
HIERARCHY_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
OUTPUT_CSV = ROOT / "data/graphrag_hierarchy/node_layout_coordinates.csv"
OUTPUT_PNG = ROOT / "data/graphrag_hierarchy/layout_debug.png"

# Node2Vec parameters
NODE2VEC_DIMENSIONS = 64
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_NUM_WALKS = 10
NODE2VEC_P = 1.0  # Return parameter (1 = balanced)
NODE2VEC_Q = 1.0  # In-out parameter (1 = balanced)

# UMAP parameters - RELAXED for better spread (avoid tight blob)
UMAP_N_NEIGHBORS = 50    # More global structure (default is 15)
UMAP_MIN_DIST = 0.5      # Force nodes apart (default is 0.1)
UMAP_SPREAD = 2.0        # Additional spread parameter
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42

# Number of CPU cores to use
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)


def load_hierarchy_data(path: Path) -> dict:
    """Load the GraphRAG hierarchy JSON."""
    print(f"Loading hierarchy data from {path}...")
    with path.open() as f:
        data = json.load(f)

    entities = data.get('entities', {})
    relationships = data.get('relationships', [])
    clusters = data.get('clusters', {})

    print(f"  Entities: {len(entities):,}")
    print(f"  Relationships: {len(relationships):,}")
    print(f"  Cluster levels: {list(clusters.keys())}")

    return data


def build_entity_to_l2_cluster_map(clusters: dict) -> Dict[str, str]:
    """
    Build a mapping from entity name -> Level 2 cluster ID.
    This is used for coloring the scatter plot.
    """
    print("\nBuilding entity -> L2 cluster mapping...")

    entity_to_l2 = {}

    # Level 2 clusters contain entity lists
    l2_clusters = clusters.get('level_2', {})

    for cluster_id, cluster_data in l2_clusters.items():
        entities = cluster_data.get('entities', [])
        for entity in entities:
            entity_to_l2[entity] = cluster_id

    print(f"  Mapped {len(entity_to_l2):,} entities to {len(l2_clusters)} L2 clusters")

    return entity_to_l2


def build_graph(entities: dict, relationships: list) -> nx.Graph:
    """
    Build a NetworkX graph from entities and relationships.
    """
    print("\nBuilding NetworkX graph...")
    G = nx.Graph()

    # Add all entities as nodes
    for entity_name in entities.keys():
        G.add_node(entity_name)

    # Add relationships as edges
    edge_count = 0
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')

        if source and target and source in entities and target in entities:
            weight = rel.get('strength', 1.0)
            if G.has_edge(source, target):
                G[source][target]['weight'] += weight
            else:
                G.add_edge(source, target, weight=weight)
                edge_count += 1

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # Handle disconnected components
    components = list(nx.connected_components(G))
    print(f"  Connected components: {len(components)}")

    if len(components) > 1:
        # Connect small components to largest
        components = sorted(components, key=len, reverse=True)
        main_component = components[0]
        main_node = list(main_component)[0]

        for component in components[1:]:
            comp_node = list(component)[0]
            G.add_edge(main_node, comp_node, weight=0.1)

        print(f"  Connected {len(components)-1} isolated components to main graph")

    return G


def generate_simple_walk_embeddings(G: nx.Graph, node_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Fallback: Generate simple random walk embeddings when Node2Vec not available.
    Uses adjacency-based features + random projections.
    """
    print("  Using simple random walk fallback (Node2Vec not installed)")

    n_nodes = len(node_list)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Use adjacency matrix + degree features
    adj_matrix = nx.adjacency_matrix(G, nodelist=node_list).toarray()

    # Degree features
    degrees = np.array([G.degree(node) for node in node_list]).reshape(-1, 1)

    # Random projection to reduce dimensionality
    np.random.seed(UMAP_RANDOM_STATE)
    proj_matrix = np.random.randn(n_nodes, NODE2VEC_DIMENSIONS)

    # Combine: adjacency @ random_projection + degree
    embeddings = np.tanh(adj_matrix @ proj_matrix / np.sqrt(n_nodes))

    # Add normalized degree as first dimension
    embeddings[:, 0] = degrees.flatten() / (degrees.max() + 1e-6)

    return embeddings, node_list


def generate_node2vec_embeddings(G: nx.Graph) -> Tuple[np.ndarray, List[str]]:
    """
    Generate Node2Vec structural embeddings for all nodes.
    """
    print(f"\nGenerating Node2Vec embeddings (dim={NODE2VEC_DIMENSIONS})...")
    print(f"  Walk length: {NODE2VEC_WALK_LENGTH}, Num walks: {NODE2VEC_NUM_WALKS}")
    print(f"  Using {NUM_WORKERS} CPU cores")

    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Try PecanPy first, then node2vec, then fallback
    try:
        from pecanpy import pecanpy
        print("  Using PecanPy (fast C implementation)")

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.edg', delete=False) as f:
            edgelist_path = f.name
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1.0)
                f.write(f"{node_to_idx[u]}\t{node_to_idx[v]}\t{weight}\n")

        try:
            g = pecanpy.SparseOTF(
                p=NODE2VEC_P,
                q=NODE2VEC_Q,
                workers=NUM_WORKERS,
                verbose=True
            )
            g.read_edg(edgelist_path, weighted=True, directed=False)

            print("  Running random walks...")
            embeddings = g.embed(
                dim=NODE2VEC_DIMENSIONS,
                num_walks=NODE2VEC_NUM_WALKS,
                walk_length=NODE2VEC_WALK_LENGTH
            )

            # PecanPy 2.x returns a numpy array directly, already ordered by node index
            if isinstance(embeddings, np.ndarray):
                print(f"  Got embeddings array: {embeddings.shape}")
                os.unlink(edgelist_path)
                return embeddings, node_list
            else:
                # Older API might return dict
                ordered_embeddings = np.zeros((len(node_list), NODE2VEC_DIMENSIONS))
                for node_id, embedding in embeddings.items():
                    if isinstance(node_id, int) and node_id < len(node_list):
                        ordered_embeddings[node_id] = embedding
                os.unlink(edgelist_path)
                return ordered_embeddings, node_list

        except Exception as e:
            print(f"  PecanPy error: {e}")
            os.unlink(edgelist_path)
            raise

    except ImportError:
        pass

    try:
        from node2vec import Node2Vec
        print("  Using node2vec library")

        node2vec = Node2Vec(
            G,
            dimensions=NODE2VEC_DIMENSIONS,
            walk_length=NODE2VEC_WALK_LENGTH,
            num_walks=NODE2VEC_NUM_WALKS,
            p=NODE2VEC_P,
            q=NODE2VEC_Q,
            workers=NUM_WORKERS,
            quiet=False
        )

        print("  Training Word2Vec model...")
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Extract embeddings in order
        embeddings = np.zeros((len(node_list), NODE2VEC_DIMENSIONS))
        for idx, node in enumerate(node_list):
            if node in model.wv:
                embeddings[idx] = model.wv[node]
            else:
                embeddings[idx] = np.random.randn(NODE2VEC_DIMENSIONS) * 0.01

        return embeddings, node_list

    except ImportError:
        pass

    # Fallback
    print("  WARNING: Neither pecanpy nor node2vec installed!")
    print("  Install with: pip install pecanpy  OR  pip install node2vec")
    return generate_simple_walk_embeddings(G, node_list)


def reduce_to_2d_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D using UMAP.
    Uses RELAXED parameters to avoid tight clustering.
    """
    print(f"\nReducing to 2D with UMAP...")
    print(f"  n_neighbors={UMAP_N_NEIGHBORS} (global structure)")
    print(f"  min_dist={UMAP_MIN_DIST} (spread out)")
    print(f"  spread={UMAP_SPREAD}")
    print(f"  metric={UMAP_METRIC}")

    try:
        import umap
    except ImportError:
        print("  ERROR: umap-learn not installed!")
        print("  Install with: pip install umap-learn")
        sys.exit(1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True
    )

    coords_2d = reducer.fit_transform(embeddings)

    print(f"  Output shape: {coords_2d.shape}")
    print(f"  X range: [{coords_2d[:, 0].min():.3f}, {coords_2d[:, 0].max():.3f}]")
    print(f"  Y range: [{coords_2d[:, 1].min():.3f}, {coords_2d[:, 1].max():.3f}]")

    return coords_2d


def save_coordinates_csv(node_list: List[str], coords: np.ndarray,
                          entity_to_l2: Dict[str, str], output_path: Path):
    """
    Save node IDs and coordinates to CSV for later use.
    """
    print(f"\nSaving coordinates to {output_path}...")

    df = pd.DataFrame({
        'node_id': node_list,
        'x': coords[:, 0],
        'y': coords[:, 1],
        'l2_cluster': [entity_to_l2.get(node, 'unknown') for node in node_list]
    })

    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} rows")


def generate_scatter_plot(node_list: List[str], coords: np.ndarray,
                          entity_to_l2: Dict[str, str], output_path: Path):
    """
    Generate a high-resolution scatter plot colored by L2 cluster.
    """
    print(f"\nGenerating scatter plot...")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  ERROR: matplotlib not installed!")
        print("  Install with: pip install matplotlib")
        sys.exit(1)

    # Get unique L2 clusters and assign colors
    l2_clusters = [entity_to_l2.get(node, 'unknown') for node in node_list]
    unique_clusters = sorted(set(l2_clusters))

    print(f"  {len(unique_clusters)} unique L2 clusters")

    # Create color map - use a colormap with enough distinct colors
    n_colors = len(unique_clusters)
    if n_colors <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        # Use a continuous colormap for many clusters
        cmap = plt.cm.get_cmap('nipy_spectral')

    cluster_to_color = {c: cmap(i / n_colors) for i, c in enumerate(unique_clusters)}
    colors = [cluster_to_color[c] for c in l2_clusters]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 16), dpi=150)

    # Plot scatter
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=5,  # Small dots for 17k points
        alpha=0.6,
        edgecolors='none'
    )

    # Add title and labels
    ax.set_title(
        f'Node2Vec + UMAP Layout Sanity Check\n'
        f'{len(node_list):,} nodes, {len(unique_clusters)} L2 clusters\n'
        f'UMAP params: n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, spread={UMAP_SPREAD}',
        fontsize=14
    )
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Calculate and display spread statistics
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()

    stats_text = (
        f'X range: {x_range:.2f}\n'
        f'Y range: {y_range:.2f}\n'
        f'Aspect: {x_range/y_range:.2f}'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def main():
    """Main entry point."""
    print("=" * 70)
    print("LAYOUT ALGORITHM SANITY CHECK")
    print("Node2Vec + UMAP Diagnostic Scatter Plot")
    print("=" * 70)

    # Load data
    data = load_hierarchy_data(HIERARCHY_PATH)
    entities = data['entities']
    relationships = data['relationships']
    clusters = data['clusters']

    # Build entity -> L2 cluster mapping
    entity_to_l2 = build_entity_to_l2_cluster_map(clusters)

    # Build graph
    G = build_graph(entities, relationships)

    # Generate Node2Vec embeddings
    embeddings, node_list = generate_node2vec_embeddings(G)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Reduce to 2D with UMAP
    coords_2d = reduce_to_2d_umap(embeddings)

    # Save coordinates to CSV
    save_coordinates_csv(node_list, coords_2d, entity_to_l2, OUTPUT_CSV)

    # Generate scatter plot
    generate_scatter_plot(node_list, coords_2d, entity_to_l2, OUTPUT_PNG)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"  PNG: {OUTPUT_PNG}")
    print("=" * 70)

    # Summary statistics
    x_range = coords_2d[:, 0].max() - coords_2d[:, 0].min()
    y_range = coords_2d[:, 1].max() - coords_2d[:, 1].min()

    print("\nSANITY CHECK RESULTS:")
    print(f"  Total nodes: {len(node_list):,}")
    print(f"  X spread: {x_range:.2f}")
    print(f"  Y spread: {y_range:.2f}")
    print(f"  L2 clusters represented: {len(set(entity_to_l2.values()))}")

    if x_range < 5 or y_range < 5:
        print("\n  WARNING: Layout appears tightly clustered!")
        print("  Consider increasing UMAP_MIN_DIST or UMAP_SPREAD")
    else:
        print("\n  Layout spread looks reasonable.")
        print("  Check the PNG to verify distinct clusters exist.")


if __name__ == "__main__":
    main()
