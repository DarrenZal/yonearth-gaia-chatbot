#!/usr/bin/env python3
"""
Supervised Semantic Layout using Text Embeddings + Community Labels.

This script creates a layout where:
1. Node positions are determined by SEMANTIC similarity (text embeddings)
2. Community labels (Level 2 clusters) SUPERVISE the UMAP to keep clusters distinct

This fixes the sparse graph problem from Node2Vec by using text meaning instead of
graph structure - isolated nodes will naturally land near semantically similar topics.

Pipeline:
1. Load GraphRAG entities and Level 2 cluster assignments
2. Generate text embeddings using sentence-transformers (all-MiniLM-L6-v2)
3. Run SUPERVISED UMAP with community IDs as target labels
4. Save coordinates to CSV
5. Generate diagnostic scatter plot

Output:
  - data/graphrag_hierarchy/node_layout_coordinates.csv
  - data/graphrag_hierarchy/layout_debug.png
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths & Parameters
# --------------------------------------------------------------------------------------

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
HIERARCHY_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
OUTPUT_CSV = ROOT / "data/graphrag_hierarchy/node_layout_coordinates.csv"
OUTPUT_PNG = ROOT / "data/graphrag_hierarchy/layout_debug.png"
EMBEDDINGS_CACHE = ROOT / "data/graphrag_hierarchy/entity_embeddings_cache.npy"

# Sentence-Transformers model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384-dim embeddings

# UMAP parameters for supervised semantic layout
UMAP_N_NEIGHBORS = 30    # Balance local/global
UMAP_MIN_DIST = 0.1      # Tighter clustering (semantic is cleaner)
UMAP_SPREAD = 1.0        # Standard spread
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42
UMAP_TARGET_WEIGHT = 0.5  # Balance between semantic (0) and supervised (1)


def load_hierarchy_data(path: Path) -> dict:
    """Load the GraphRAG hierarchy JSON."""
    print(f"Loading hierarchy data from {path}...")
    with path.open() as f:
        data = json.load(f)

    entities = data.get('entities', {})
    clusters = data.get('clusters', {})

    print(f"  Entities: {len(entities):,}")
    print(f"  Cluster levels: {list(clusters.keys())}")

    return data


def build_entity_to_l2_cluster_map(clusters: dict) -> Dict[str, str]:
    """
    Build mapping from entity name -> cluster ID with fallback strategy.

    Priority:
    1. Level 2 cluster (coarse communities - 681 clusters)
    2. Level 1 cluster (fine clusters - 2073 clusters)
    3. Level 0 cluster (individual entities - creates unique cluster per entity)

    This ensures 100% coverage - no entity is left as 'unknown'.
    """
    print("\nBuilding entity -> cluster mapping with fallback strategy...")

    # Build Level 2 map (primary)
    entity_to_l2 = {}
    l2_clusters = clusters.get('level_2', {})
    for cluster_id, cluster_data in l2_clusters.items():
        for entity in cluster_data.get('entities', []):
            entity_to_l2[entity] = cluster_id
    print(f"  Level 2: {len(entity_to_l2):,} entities -> {len(l2_clusters)} clusters")

    # Build Level 1 map (fallback 1)
    entity_to_l1 = {}
    l1_clusters = clusters.get('level_1', {})
    for cluster_id, cluster_data in l1_clusters.items():
        for entity in cluster_data.get('entities', []):
            entity_to_l1[entity] = cluster_id
    print(f"  Level 1: {len(entity_to_l1):,} entities -> {len(l1_clusters)} clusters")

    # Level 0: entity name IS the cluster key (fallback 2)
    entity_to_l0 = {name: f"level_0_{name}" for name in clusters.get('level_0', {}).keys()}
    print(f"  Level 0: {len(entity_to_l0):,} entities (individual clusters)")

    # Build final map with fallback
    final_map = {}
    stats = {'l2': 0, 'l1': 0, 'l0': 0}

    all_entities = set(entity_to_l2.keys()) | set(entity_to_l1.keys()) | set(entity_to_l0.keys())

    for entity in all_entities:
        if entity in entity_to_l2:
            final_map[entity] = entity_to_l2[entity]
            stats['l2'] += 1
        elif entity in entity_to_l1:
            final_map[entity] = entity_to_l1[entity]
            stats['l1'] += 1
        elif entity in entity_to_l0:
            final_map[entity] = entity_to_l0[entity]
            stats['l0'] += 1

    print(f"\n  Final coverage with fallback:")
    print(f"    Direct L2:      {stats['l2']:,} ({100*stats['l2']/len(final_map):.1f}%)")
    print(f"    Fallback to L1: {stats['l1']:,} ({100*stats['l1']/len(final_map):.1f}%)")
    print(f"    Fallback to L0: {stats['l0']:,} ({100*stats['l0']/len(final_map):.1f}%)")
    print(f"    TOTAL:          {len(final_map):,} (100%)")

    return final_map


def prepare_text_for_embedding(name: str, entity_data: dict) -> str:
    """
    Prepare text for embedding by combining entity name and description.
    For entities without descriptions, use just the name.
    """
    description = entity_data.get('description', '')
    entity_type = entity_data.get('type', '')

    if description:
        text = f"{name}: {description}"
    elif entity_type:
        text = f"{name} ({entity_type})"
    else:
        text = name

    return text


def generate_text_embeddings(entities: dict, cache_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Generate text embeddings for all entities using sentence-transformers.
    Uses caching to avoid re-computing on subsequent runs.
    """
    node_list = list(entities.keys())

    # Check cache
    if cache_path.exists():
        print(f"\nLoading cached embeddings from {cache_path}...")
        cache_data = np.load(cache_path, allow_pickle=True).item()
        cached_nodes = cache_data.get('nodes', [])
        cached_embeddings = cache_data.get('embeddings', None)

        if list(cached_nodes) == node_list and cached_embeddings is not None:
            print(f"  Loaded {len(cached_nodes):,} cached embeddings")
            return cached_embeddings, node_list
        else:
            print("  Cache mismatch, regenerating embeddings...")

    print(f"\nGenerating text embeddings for {len(entities):,} entities...")
    print(f"  Model: {EMBEDDING_MODEL}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: sentence-transformers not installed!")
        print("  Install with: pip install sentence-transformers")
        sys.exit(1)

    # Load model
    print("  Loading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Prepare texts
    print("  Preparing texts...")
    texts = [prepare_text_for_embedding(name, entities[name]) for name in node_list]

    # Show some examples
    print("  Sample texts:")
    for i in range(min(3, len(texts))):
        print(f"    {i}: {texts[i][:80]}...")

    # Count entities with/without descriptions
    with_desc = sum(1 for name in node_list if entities[name].get('description'))
    print(f"  Entities with descriptions: {with_desc:,} ({100*with_desc/len(node_list):.1f}%)")
    print(f"  Entities without descriptions: {len(node_list) - with_desc:,}")

    # Generate embeddings
    print("  Encoding texts (this may take a minute)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True
    )

    print(f"  Embeddings shape: {embeddings.shape}")

    # Cache for next time
    print(f"  Caching embeddings to {cache_path}...")
    np.save(cache_path, {'nodes': node_list, 'embeddings': embeddings})

    return embeddings, node_list


def run_supervised_umap(embeddings: np.ndarray, node_list: List[str],
                        entity_to_l2: Dict[str, str]) -> np.ndarray:
    """
    Run SUPERVISED UMAP with community labels as targets.

    This balances:
    - Semantic similarity (from text embeddings)
    - Community membership (from L2 cluster labels)
    """
    print(f"\nRunning Supervised UMAP...")
    print(f"  n_neighbors={UMAP_N_NEIGHBORS}")
    print(f"  min_dist={UMAP_MIN_DIST}")
    print(f"  target_weight={UMAP_TARGET_WEIGHT}")
    print(f"  metric={UMAP_METRIC}")

    try:
        import umap
    except ImportError:
        print("  ERROR: umap-learn not installed!")
        print("  Install with: pip install umap-learn")
        sys.exit(1)

    # Create numeric labels for supervision
    # Entities without L2 assignment get label -1 (will be treated as unlabeled)
    unique_clusters = sorted(set(entity_to_l2.values()))
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

    labels = np.array([
        cluster_to_idx.get(entity_to_l2.get(node), -1)
        for node in node_list
    ])

    # Count labeled vs unlabeled
    labeled_count = np.sum(labels >= 0)
    unlabeled_count = np.sum(labels < 0)
    print(f"  Labeled entities: {labeled_count:,} ({100*labeled_count/len(labels):.1f}%)")
    print(f"  Unlabeled entities: {unlabeled_count:,}")
    print(f"  Unique clusters: {len(unique_clusters)}")

    # Run supervised UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        target_weight=UMAP_TARGET_WEIGHT,  # 0=unsupervised, 1=fully supervised
        verbose=True
    )

    # Fit with supervision (y=labels)
    # UMAP treats -1 as "unlabeled" and won't use those for supervision
    print("  Fitting UMAP with supervision...")
    coords_2d = reducer.fit_transform(embeddings, y=labels)

    print(f"  Output shape: {coords_2d.shape}")
    print(f"  X range: [{coords_2d[:, 0].min():.3f}, {coords_2d[:, 0].max():.3f}]")
    print(f"  Y range: [{coords_2d[:, 1].min():.3f}, {coords_2d[:, 1].max():.3f}]")

    return coords_2d


def save_coordinates_csv(node_list: List[str], coords: np.ndarray,
                          entity_to_l2: Dict[str, str], output_path: Path):
    """Save node IDs and coordinates to CSV."""
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
    """Generate a scatter plot colored by L2 cluster."""
    print(f"\nGenerating scatter plot...")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ERROR: matplotlib not installed!")
        sys.exit(1)

    # Get L2 clusters and assign colors
    l2_clusters = [entity_to_l2.get(node, 'unknown') for node in node_list]
    unique_clusters = sorted(set(l2_clusters))

    print(f"  {len(unique_clusters)} unique L2 clusters")

    # Create color map
    n_colors = len(unique_clusters)
    if n_colors <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        cmap = plt.cm.get_cmap('nipy_spectral')

    cluster_to_color = {c: cmap(i / n_colors) for i, c in enumerate(unique_clusters)}
    colors = [cluster_to_color[c] for c in l2_clusters]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 16), dpi=150)

    # Plot scatter
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=5,
        alpha=0.6,
        edgecolors='none'
    )

    # Title
    ax.set_title(
        f'Supervised Semantic UMAP Layout\n'
        f'{len(node_list):,} nodes, {len(unique_clusters)} L2 clusters\n'
        f'UMAP params: n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, '
        f'target_weight={UMAP_TARGET_WEIGHT}',
        fontsize=14
    )
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3)

    # Stats
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()

    stats_text = (
        f'X range: {x_range:.2f}\n'
        f'Y range: {y_range:.2f}\n'
        f'Method: Supervised Semantic UMAP\n'
        f'Embeddings: {EMBEDDING_MODEL}'
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
    print("SUPERVISED SEMANTIC LAYOUT")
    print("Text Embeddings + L2 Community Supervision")
    print("=" * 70)

    # Load data
    data = load_hierarchy_data(HIERARCHY_PATH)
    entities = data['entities']
    clusters = data['clusters']

    # Build entity -> L2 cluster mapping
    entity_to_l2 = build_entity_to_l2_cluster_map(clusters)

    # Generate text embeddings
    embeddings, node_list = generate_text_embeddings(entities, EMBEDDINGS_CACHE)

    # Run supervised UMAP
    coords_2d = run_supervised_umap(embeddings, node_list, entity_to_l2)

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

    print("\nLAYOUT STATISTICS:")
    print(f"  Total nodes: {len(node_list):,}")
    print(f"  X spread: {x_range:.2f}")
    print(f"  Y spread: {y_range:.2f}")
    print(f"  L2 clusters represented: {len(set(entity_to_l2.values()))}")
    print(f"  Method: Supervised Semantic UMAP")
    print(f"  Text embeddings: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
