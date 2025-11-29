#!/usr/bin/env python3
"""
TEST VERSION: Compute UMAP 3D embeddings for 100 sample entities.

This is a lightweight test to validate the pipeline before running the full computation.
"""

import json
import os
import sys
import numpy as np
from openai import OpenAI
import networkx as nx
import time
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import random

# Check for umap-learn
try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
    sys.exit(1)

# Configuration
UNIFIED_KG_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_unified/unified.json"
GRAPHRAG_HIERARCHY_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"
OUTPUT_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy_test_sample.json"

# TEST MODE: Process only N entities
TEST_SAMPLE_SIZE = 100

# UMAP parameters (from design doc)
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42

# Embedding parameters
BATCH_SIZE = 50
MAX_RELATIONSHIPS_IN_EMBEDDING = 2


def load_data():
    """Load unified knowledge graph and existing graphrag hierarchy."""
    print("Loading data...")

    # Load unified KG
    print(f"  Loading unified KG from {UNIFIED_KG_PATH}")
    with open(UNIFIED_KG_PATH) as f:
        kg_data = json.load(f)
    entities = kg_data.get('entities', {})
    relationships = kg_data.get('relationships', [])
    print(f"  Loaded {len(entities)} entities, {len(relationships)} relationships")

    # Load existing graphrag hierarchy
    print(f"  Loading graphrag hierarchy from {GRAPHRAG_HIERARCHY_PATH}")
    with open(GRAPHRAG_HIERARCHY_PATH) as f:
        graphrag_data = json.load(f)
    print(f"  Loaded hierarchy with {len(graphrag_data.get('entities', {}))} entities")

    return entities, relationships, graphrag_data


def sample_entities(entities: Dict, n: int = 100) -> Dict:
    """Sample N entities, ensuring diversity of entity types."""
    print(f"\nSampling {n} diverse entities...")

    # Group entities by type
    by_type = defaultdict(list)
    for entity_id, entity in entities.items():
        entity_type = entity.get('type', 'unknown')
        by_type[entity_type].append(entity_id)

    # Sample proportionally from each type
    sampled = {}
    per_type = n // len(by_type)

    for entity_type, entity_ids in by_type.items():
        sample_size = min(per_type, len(entity_ids))
        sampled_ids = random.sample(entity_ids, sample_size)
        for entity_id in sampled_ids:
            sampled[entity_id] = entities[entity_id]

    # Fill up to N if needed
    remaining = n - len(sampled)
    if remaining > 0:
        all_unsampled = [eid for eid in entities.keys() if eid not in sampled]
        additional = random.sample(all_unsampled, min(remaining, len(all_unsampled)))
        for entity_id in additional:
            sampled[entity_id] = entities[entity_id]

    # Print distribution
    sample_types = {}
    for entity in sampled.values():
        etype = entity.get('type', 'unknown')
        sample_types[etype] = sample_types.get(etype, 0) + 1

    print(f"  Sampled {len(sampled)} entities:")
    for etype, count in sorted(sample_types.items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count}")

    return sampled


def build_relationship_index(relationships: List[dict], entity_ids: set) -> Dict[str, List[dict]]:
    """Build index of relationships for sampled entities only."""
    print("\nBuilding relationship index for sample...")
    index = defaultdict(list)

    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        rel_type = rel.get('type', 'RELATED_TO')

        # Only include relationships involving sampled entities
        if source in entity_ids and target in entity_ids:
            index[source].append({
                'target': target,
                'type': rel_type,
                'weight': rel.get('weight', 1.0)
            })
            index[target].append({
                'target': source,
                'type': rel_type,
                'weight': rel.get('weight', 1.0)
            })

    print(f"  Indexed relationships for {len(index)} entities")
    return dict(index)


def get_top_relationships(entity_id: str, rel_index: Dict, k: int = 2) -> List[str]:
    """Get top-k most important relationships for an entity."""
    if entity_id not in rel_index:
        return []

    # Get relationships sorted by weight
    rels = rel_index[entity_id]
    sorted_rels = sorted(rels, key=lambda x: x.get('weight', 1.0), reverse=True)

    # Format as text
    top_rels = []
    for rel in sorted_rels[:k]:
        rel_text = f"{rel['type']} {rel['target']}"
        top_rels.append(rel_text)

    return top_rels


def create_graph_enriched_embeddings(entities: Dict, rel_index: Dict, client: OpenAI):
    """Create graph-enriched embeddings: entity + description + top-2 relationships."""
    print("\nCreating graph-enriched embeddings...")

    entity_ids = list(entities.keys())
    n_entities = len(entity_ids)

    # Store embeddings in array
    embeddings = np.zeros((n_entities, 1536), dtype='float32')

    for i in range(0, n_entities, BATCH_SIZE):
        batch_ids = entity_ids[i:i+BATCH_SIZE]
        texts = []

        for entity_id in batch_ids:
            entity = entities[entity_id]
            entity_type = entity.get('type', 'unknown')
            description = entity.get('description', '')

            # Build graph-enriched text
            text_parts = [
                entity_id,
                entity_type
            ]

            if description:
                text_parts.append(description[:200])

            # Add top-2 relationships
            top_rels = get_top_relationships(entity_id, rel_index, k=MAX_RELATIONSHIPS_IN_EMBEDDING)
            text_parts.extend(top_rels)

            text = " | ".join(text_parts)
            texts.append(text)

        # Get embeddings
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            for j, embedding in enumerate(response.data):
                embeddings[i+j] = embedding.embedding

            print(f"  Processed {min(i+BATCH_SIZE, n_entities)}/{n_entities} entities")
            time.sleep(0.05)

        except Exception as e:
            print(f"  Error at batch {i}: {e}")
            for j in range(len(batch_ids)):
                embeddings[i+j] = np.random.randn(1536) * 0.1

    print(f"  Completed all {n_entities} embeddings")
    return embeddings, entity_ids


def compute_umap_positions(embeddings: np.ndarray) -> np.ndarray:
    """Compute UMAP 3D positions from embeddings."""
    print("\nComputing UMAP 3D positions...")
    print(f"  Input: {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"  Parameters: n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}")

    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=min(UMAP_N_NEIGHBORS, embeddings.shape[0] - 1),  # Adjust for small sample
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True
    )

    print("  Running UMAP...")
    positions = reducer.fit_transform(embeddings)

    print(f"  Computed {positions.shape[0]} 3D positions")
    print(f"  Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
          f"z=[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

    return positions


def compute_betweenness_centrality(relationships: List[dict], entities: Dict) -> Dict[str, float]:
    """Compute betweenness centrality for sample entities."""
    print("\nComputing betweenness centrality...")

    G = nx.Graph()
    entity_ids = set(entities.keys())

    # Add entities as nodes
    for entity_id in entity_ids:
        G.add_node(entity_id)

    # Add relationships as edges (only within sample)
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        if source in entity_ids and target in entity_ids:
            G.add_edge(source, target)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("  Computing betweenness centrality...")
    betweenness = nx.betweenness_centrality(G, normalized=True)

    # Print top bridge nodes
    top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n  Top 5 bridge nodes in sample:")
    for entity_id, score in top_bridges:
        entity_type = entities[entity_id].get('type', 'unknown')
        print(f"    {entity_id} ({entity_type}): {score:.4f}")

    return betweenness


def save_test_results(entity_ids: List[str], umap_positions: np.ndarray,
                      betweenness: Dict[str, float], entities: Dict):
    """Save test results to JSON."""
    print(f"\nSaving test results to {OUTPUT_PATH}...")

    results = {
        'test_mode': True,
        'sample_size': len(entity_ids),
        'entities': {}
    }

    for i, entity_id in enumerate(entity_ids):
        results['entities'][entity_id] = {
            'type': entities[entity_id].get('type'),
            'description': entities[entity_id].get('description', '')[:100],
            'umap_position': umap_positions[i].tolist(),
            'betweenness': betweenness.get(entity_id, 0.0),
            'original_pca_available': True
        }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"  Saved test results: {file_size_kb:.2f} KB")


def main():
    """Main execution pipeline for test."""
    print("=" * 80)
    print("GraphRAG UMAP Embedding Computation - TEST MODE")
    print(f"Processing {TEST_SAMPLE_SIZE} sample entities")
    print("=" * 80)

    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load data
    entities, relationships, graphrag_data = load_data()

    # Sample entities
    random.seed(42)  # Reproducible sampling
    sample_entities_dict = sample_entities(entities, n=TEST_SAMPLE_SIZE)
    sample_entity_ids = set(sample_entities_dict.keys())

    # Build relationship index for sample
    rel_index = build_relationship_index(relationships, sample_entity_ids)

    # Create graph-enriched embeddings
    embeddings, entity_ids = create_graph_enriched_embeddings(
        sample_entities_dict, rel_index, client
    )

    # Compute UMAP positions
    umap_positions = compute_umap_positions(embeddings)

    # Compute betweenness centrality
    betweenness = compute_betweenness_centrality(relationships, sample_entities_dict)

    # Save test results
    save_test_results(entity_ids, umap_positions, betweenness, sample_entities_dict)

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE!")
    print("=" * 80)
    print(f"\nTest results saved to: {OUTPUT_PATH}")
    print(f"\nSample entities processed: {len(entity_ids)}")
    print(f"UMAP positions computed: {umap_positions.shape[0]}")
    print(f"Betweenness scores computed: {len(betweenness)}")

    print("\nNext steps:")
    print("  1. Review test results in graphrag_hierarchy_test_sample.json")
    print("  2. Verify UMAP positions look reasonable")
    print("  3. If satisfied, run full version: python3 scripts/compute_graphrag_umap_embeddings.py")


if __name__ == '__main__':
    main()
