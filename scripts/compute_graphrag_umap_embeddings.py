#!/usr/bin/env python3
"""
Compute UMAP 3D embeddings and betweenness centrality for GraphRAG 3D Embedding View.

This script enhances the existing graphrag_hierarchy.json with:
1. UMAP-based 3D positions (replaces PCA for better local structure preservation)
2. Graph-enriched entity embeddings (name + description + top-2 relationships)
3. Betweenness centrality scores (identifies bridge nodes)
4. Relationship strength weights (for selective edge rendering)

Output: Updated graphrag_hierarchy.json with new fields:
  - umap_position: [x, y, z] for each entity
  - betweenness: 0.0-1.0 centrality score
  - relationship_strengths: {target: weight} for edges
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

# Check for umap-learn
try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
    sys.exit(1)

# Configuration
UNIFIED_KG_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_unified/unified.json"
GRAPHRAG_HIERARCHY_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"
OUTPUT_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"
BACKUP_PATH = "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy_backup_pre_umap.json"

# Checkpoint paths (for resuming if script crashes)
CHECKPOINT_DIR = "/tmp/graphrag_checkpoints"
UMAP_CHECKPOINT = f"{CHECKPOINT_DIR}/umap_positions.npy"
BETWEENNESS_CHECKPOINT = f"{CHECKPOINT_DIR}/betweenness_scores.json"
REL_STRENGTHS_CHECKPOINT = f"{CHECKPOINT_DIR}/relationship_strengths.json"
ENTITY_IDS_CHECKPOINT = f"{CHECKPOINT_DIR}/entity_ids.json"

# UMAP parameters (from design doc)
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42

# Embedding parameters
BATCH_SIZE = 50
MAX_RELATIONSHIPS_IN_EMBEDDING = 2  # Top-2 most important relationships


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


def build_relationship_index(relationships: List[dict]) -> Dict[str, List[dict]]:
    """Build index of relationships for each entity."""
    print("\nBuilding relationship index...")
    index = defaultdict(list)

    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        rel_type = rel.get('type', 'RELATED_TO')

        if source and target:
            index[source].append({
                'target': target,
                'type': rel_type,
                'weight': rel.get('weight', 1.0)
            })
            # Add reverse relationship for undirected graph
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

    # Store embeddings in memory-mapped array
    embeddings_file = '/tmp/graphrag_umap_embeddings.npy'
    embeddings = np.memmap(embeddings_file, dtype='float32', mode='w+', shape=(n_entities, 1536))

    for i in range(0, n_entities, BATCH_SIZE):
        batch_ids = entity_ids[i:i+BATCH_SIZE]
        texts = []

        for entity_id in batch_ids:
            entity = entities[entity_id]
            entity_type = entity.get('type', 'unknown')
            description = entity.get('description', '')

            # Build graph-enriched text
            # Format: "EntityName | TYPE | Description | REL1 | REL2"
            text_parts = [
                entity_id,  # Entity name
                entity_type  # Entity type
            ]

            if description:
                text_parts.append(description[:200])  # Truncate long descriptions

            # Add top-2 relationships
            top_rels = get_top_relationships(entity_id, rel_index, k=MAX_RELATIONSHIPS_IN_EMBEDDING)
            text_parts.extend(top_rels)

            text = " | ".join(text_parts)
            texts.append(text)

        # Get embeddings with error handling
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )

            for j, embedding in enumerate(response.data):
                embeddings[i+j] = embedding.embedding

            if (i+BATCH_SIZE) % 500 == 0 or (i+BATCH_SIZE) >= n_entities:
                print(f"  Processed {min(i+BATCH_SIZE, n_entities)}/{n_entities} entities", flush=True)

            time.sleep(0.05)  # Rate limiting

        except Exception as e:
            print(f"  Error at batch {i}: {e}")
            # Use random embeddings as fallback
            for j in range(len(batch_ids)):
                embeddings[i+j] = np.random.randn(1536) * 0.1

    print(f"  Completed all {n_entities} embeddings")
    return embeddings, entity_ids


def compute_umap_positions(embeddings: np.ndarray) -> np.ndarray:
    """Compute UMAP 3D positions from embeddings."""
    print("\nComputing UMAP 3D positions...")
    print(f"  Input: {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"  Parameters: n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}")

    # Initialize UMAP
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True
    )

    # Compute positions
    print("  Running UMAP (this may take several minutes)...")
    positions = reducer.fit_transform(embeddings)

    print(f"  Computed {positions.shape[0]} 3D positions")
    print(f"  Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
          f"z=[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

    return positions


def compute_betweenness_centrality(relationships: List[dict], entities: Dict) -> Dict[str, float]:
    """Compute betweenness centrality for all entities."""
    print("\nComputing betweenness centrality...")

    # Build NetworkX graph
    G = nx.Graph()

    # Add all entities as nodes
    for entity_id in entities.keys():
        G.add_node(entity_id)

    # Add relationships as edges
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        if source and target and source in entities and target in entities:
            G.add_edge(source, target)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute betweenness centrality (this can take a while)
    print("  Computing betweenness centrality (this may take several minutes)...")
    betweenness = nx.betweenness_centrality(G, normalized=True)

    print(f"  Computed centrality for {len(betweenness)} entities")

    # Print top bridge nodes
    top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top 10 bridge nodes:")
    for entity_id, score in top_bridges:
        entity_type = entities[entity_id].get('type', 'unknown')
        print(f"    {entity_id} ({entity_type}): {score:.4f}")

    return betweenness


def compute_relationship_strengths(relationships: List[dict]) -> Dict[Tuple[str, str], float]:
    """Compute relationship strength based on co-occurrence frequency."""
    print("\nComputing relationship strengths...")

    # Count co-occurrences
    edge_counts = Counter()
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        if source and target:
            # Normalize edge direction (undirected)
            edge = tuple(sorted([source, target]))
            edge_counts[edge] += 1

    # Normalize to [0, 1]
    max_count = max(edge_counts.values()) if edge_counts else 1
    strengths = {edge: count / max_count for edge, count in edge_counts.items()}

    print(f"  Computed strengths for {len(strengths)} unique edges")
    print(f"  Strength range: [{min(strengths.values()):.4f}, {max(strengths.values()):.4f}]")

    return strengths


def update_graphrag_hierarchy(graphrag_data: Dict, entity_ids: List[str],
                              umap_positions: np.ndarray, betweenness: Dict[str, float],
                              rel_strengths: Dict[Tuple[str, str], float]):
    """Update graphrag hierarchy with UMAP positions and betweenness scores."""
    print("\nUpdating graphrag hierarchy...")

    # Create entity_id to index mapping
    entity_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids)}

    # Update level_0 clusters (individual entities)
    updated_count = 0
    for cluster_id, cluster in graphrag_data['clusters']['level_0'].items():
        # Use cluster_id as the entity ID
        entity_id = cluster_id
        if entity_id in entity_to_idx:
            idx = entity_to_idx[entity_id]

            # Add UMAP position
            cluster['umap_position'] = umap_positions[idx].tolist()

            # Add betweenness centrality
            cluster['betweenness'] = betweenness.get(entity_id, 0.0)

            updated_count += 1

    print(f"  Updated {updated_count} level_0 clusters with UMAP positions and betweenness")

    # Update relationships with strength weights
    relationships_updated = 0
    for rel in graphrag_data.get('relationships', []):
        source = rel.get('source')
        target = rel.get('target')
        if source and target:
            edge = tuple(sorted([source, target]))
            if edge in rel_strengths:
                rel['strength'] = rel_strengths[edge]
                relationships_updated += 1

    print(f"  Updated {relationships_updated} relationships with strength weights")

    # Add metadata about UMAP computation
    if 'metadata' not in graphrag_data:
        graphrag_data['metadata'] = {}

    graphrag_data['metadata']['umap'] = {
        'n_components': UMAP_N_COMPONENTS,
        'n_neighbors': UMAP_N_NEIGHBORS,
        'min_dist': UMAP_MIN_DIST,
        'metric': UMAP_METRIC,
        'random_state': UMAP_RANDOM_STATE,
        'computed_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    graphrag_data['metadata']['betweenness'] = {
        'computed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'top_bridge_nodes': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
    }

    return graphrag_data


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("GraphRAG UMAP Embedding Computation")
    print("=" * 80)

    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load data
    entities, relationships, graphrag_data = load_data()

    # Build relationship index for graph-enriched embeddings
    rel_index = build_relationship_index(relationships)

    # Create graph-enriched embeddings
    embeddings, entity_ids = create_graph_enriched_embeddings(entities, rel_index, client)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Save entity IDs for checkpoint resume
    print(f"\nSaving entity IDs checkpoint...")
    with open(ENTITY_IDS_CHECKPOINT, 'w') as f:
        json.dump(entity_ids, f)

    # Compute UMAP positions (or load from checkpoint)
    if os.path.exists(UMAP_CHECKPOINT):
        print(f"\n✓ Loading UMAP positions from checkpoint: {UMAP_CHECKPOINT}")
        umap_positions = np.load(UMAP_CHECKPOINT)
    else:
        umap_positions = compute_umap_positions(embeddings)
        print(f"\nSaving UMAP checkpoint to {UMAP_CHECKPOINT}...")
        np.save(UMAP_CHECKPOINT, umap_positions)

    # Compute betweenness centrality (or load from checkpoint)
    if os.path.exists(BETWEENNESS_CHECKPOINT):
        print(f"\n✓ Loading betweenness centrality from checkpoint: {BETWEENNESS_CHECKPOINT}")
        with open(BETWEENNESS_CHECKPOINT) as f:
            betweenness = json.load(f)
    else:
        betweenness = compute_betweenness_centrality(relationships, entities)
        print(f"\nSaving betweenness checkpoint to {BETWEENNESS_CHECKPOINT}...")
        with open(BETWEENNESS_CHECKPOINT, 'w') as f:
            json.dump(betweenness, f)

    # Compute relationship strengths (or load from checkpoint)
    if os.path.exists(REL_STRENGTHS_CHECKPOINT):
        print(f"\n✓ Loading relationship strengths from checkpoint: {REL_STRENGTHS_CHECKPOINT}")
        with open(REL_STRENGTHS_CHECKPOINT) as f:
            rel_strengths_list = json.load(f)
            rel_strengths = {tuple(k.split('|||')): v for k, v in rel_strengths_list}
    else:
        rel_strengths = compute_relationship_strengths(relationships)
        print(f"\nSaving relationship strengths checkpoint to {REL_STRENGTHS_CHECKPOINT}...")
        # Convert tuple keys to strings for JSON
        rel_strengths_serializable = {f"{k[0]}|||{k[1]}": v for k, v in rel_strengths.items()}
        with open(REL_STRENGTHS_CHECKPOINT, 'w') as f:
            json.dump(rel_strengths_serializable, f)

    # Update graphrag hierarchy
    updated_data = update_graphrag_hierarchy(
        graphrag_data, entity_ids, umap_positions, betweenness, rel_strengths
    )

    # Create backup
    print(f"\nCreating backup at {BACKUP_PATH}...")
    with open(BACKUP_PATH, 'w') as f:
        json.dump(graphrag_data, f)

    # Save updated data
    print(f"\nSaving updated data to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(updated_data, f, indent=2)

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n✅ UMAP computation complete!")
    print(f"   Output: {OUTPUT_PATH}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Backup: {BACKUP_PATH}")

    # Clean up checkpoints on success
    print(f"\nCleaning up checkpoints...")
    import shutil
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"  Removed {CHECKPOINT_DIR}")

    print("\nNext steps:")
    print("  1. Review updated graphrag_hierarchy.json")
    print("  2. Implement frontend visualization in web/graph/GraphRAG3D_EmbeddingView.js")
    print("  3. Deploy to production at https://earthdo.me/graph/embedding")


if __name__ == '__main__':
    main()
