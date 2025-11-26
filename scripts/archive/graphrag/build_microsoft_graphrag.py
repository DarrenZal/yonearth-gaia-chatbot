#!/usr/bin/env python3
"""
Build GraphRAG hierarchy using Microsoft's methodology.

Hybrid approach:
- Input: Our existing discourse_graph_hybrid.json (17,296 entities + 20,508 relationships)
- Clustering: Microsoft's graspologic.hierarchical_leiden
- Summarization: OpenAI gpt-4.1-mini (not gpt-4o-mini)
- Positioning: UMAP 3D coordinates
- Output: Format compatible with kg-enhanced.js

This combines:
✅ Our high-quality, post-processed knowledge graph
✅ Microsoft's proven clustering methodology
✅ Latest GPT model for summaries
✅ 3D visualization positioning
"""

import json
import os
import sys
import time
import numpy as np
import networkx as nx
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars already set

try:
    from graspologic.partition import hierarchical_leiden
except ImportError:
    print("ERROR: graspologic not installed. Install with:")
    print("  pip install graspologic")
    sys.exit(1)

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Install with:")
    print("  pip install umap-learn")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
INPUT_GRAPH = ROOT / "data/knowledge_graph_unified/discourse_graph_hybrid.json"
OUTPUT_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy_microsoft.json"
CHECKPOINT_DIR = ROOT / "data/graphrag_hierarchy/checkpoints_microsoft"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Microsoft GraphRAG parameters
MAX_CLUSTER_SIZE = int(os.getenv("MAX_CLUSTER_SIZE", "25"))  # ← Reduced from 100 to 25 for tighter clusters
USE_LCC = os.getenv("USE_LCC", "true").lower() == "true"  # Use largest connected component
RANDOM_SEED = 42

# UMAP parameters for 3D positioning
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# OpenAI parameters
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"  # ← Using gpt-4.1-mini as requested
EMBED_DIM = 1536
EMBED_BATCH_SIZE = 100
EMBED_RATE_DELAY = 0.05  # seconds between embedding calls
LLM_RATE_DELAY = 0.05  # seconds between LLM calls

# Summary parameters
SUMMARY_MAX_ENTITIES = 20  # Max entities to include in summary prompt
SUMMARY_MAX_RELATIONSHIPS = 15  # Max relationships to include
SUMMARY_MAX_TOKENS = 300  # Max tokens for summary output

# Cost tracking (gpt-4.1-mini pricing - update when official pricing available)
# Using gpt-4o-mini as placeholder until gpt-4.1-mini pricing is released
COST_PER_1K_INPUT = 0.00015
COST_PER_1K_OUTPUT = 0.0006


# ============================================================================
# Data Loading
# ============================================================================

def load_discourse_graph(path: Path) -> Tuple[Dict, List]:
    """Load the discourse graph."""
    print(f"📂 Loading discourse graph from {path.name}...")
    with path.open() as f:
        data = json.load(f)

    entities = data.get('entities', {})
    relationships = data.get('relationships', [])

    print(f"   ✓ {len(entities):,} entities")
    print(f"   ✓ {len(relationships):,} relationships")

    return entities, relationships


def build_networkx_graph(entities: Dict, relationships: List) -> nx.Graph:
    """Build NetworkX graph for clustering."""
    print(f"\n🕸️  Building NetworkX graph...")
    G = nx.Graph()

    # Add nodes
    for entity_id, entity_data in entities.items():
        G.add_node(entity_id, **entity_data)

    # Add edges with weights (count of relationships between entities)
    edge_counts = defaultdict(int)
    for rel in relationships:
        source = rel.get('source') or rel.get('source_entity')
        target = rel.get('target') or rel.get('target_entity')
        if source in entities and target in entities and source != target:
            edge_key = tuple(sorted([source, target]))
            edge_counts[edge_key] += 1

    # Normalize weights
    max_count = max(edge_counts.values()) if edge_counts else 1
    for (source, target), count in edge_counts.items():
        weight = count / max_count
        G.add_edge(source, target, weight=weight)

    print(f"   ✓ {G.number_of_nodes():,} nodes")
    print(f"   ✓ {G.number_of_edges():,} edges")

    return G


# ============================================================================
# Microsoft GraphRAG Clustering
# ============================================================================

def cluster_with_hierarchical_leiden(G: nx.Graph) -> Tuple[List, Dict]:
    """
    Apply Microsoft's hierarchical Leiden clustering.

    Returns:
        communities: List of (level, cluster_id, parent_id, node_list) tuples
        hierarchy: Dict mapping cluster_id -> parent_id
    """
    print(f"\n🔬 Applying Microsoft GraphRAG clustering...")
    print(f"   Method: hierarchical_leiden")
    print(f"   max_cluster_size: {MAX_CLUSTER_SIZE}")
    print(f"   use_lcc: {USE_LCC}")
    print(f"   random_seed: {RANDOM_SEED}")

    # Check checkpoint
    checkpoint_path = CHECKPOINT_DIR / "leiden_communities.json"
    if checkpoint_path.exists():
        print(f"   ✓ Loading from checkpoint: {checkpoint_path.name}")
        with checkpoint_path.open() as f:
            checkpoint_data = json.load(f)
        return checkpoint_data['communities'], checkpoint_data['hierarchy']

    # Use largest connected component if requested
    if USE_LCC:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"   ✓ Using largest connected component: {len(G):,} nodes")

    # Run hierarchical Leiden
    print(f"   ⏳ Running hierarchical_leiden (this may take a few minutes)...")
    start_time = time.time()

    community_mapping = hierarchical_leiden(
        G,
        max_cluster_size=MAX_CLUSTER_SIZE,
        random_seed=RANDOM_SEED
    )

    elapsed = time.time() - start_time
    print(f"   ✓ Clustering complete in {elapsed:.1f}s")

    # Parse results
    node_to_community = {}  # {level: {node: cluster_id}}
    hierarchy = {}  # {cluster_id: parent_id}

    for partition in community_mapping:
        level = partition.level
        node = partition.node
        cluster_id = partition.cluster
        parent_cluster = partition.parent_cluster

        if level not in node_to_community:
            node_to_community[level] = {}
        node_to_community[level][node] = cluster_id

        hierarchy[cluster_id] = parent_cluster if parent_cluster is not None else -1

    # Group nodes by cluster
    clusters_by_level = {}
    for level, node_map in node_to_community.items():
        clusters_by_level[level] = defaultdict(list)
        for node, cluster_id in node_map.items():
            clusters_by_level[level][cluster_id].append(node)

    # Convert to list format
    communities = []
    for level in sorted(clusters_by_level.keys()):
        for cluster_id, nodes in clusters_by_level[level].items():
            parent_id = hierarchy[cluster_id]
            communities.append((level, cluster_id, parent_id, nodes))

    # Print statistics
    level_counts = defaultdict(int)
    for level, _, _, nodes in communities:
        level_counts[level] += 1

    print(f"\n   📊 Cluster Statistics:")
    for level in sorted(level_counts.keys()):
        print(f"      Level {level}: {level_counts[level]:,} communities")
    print(f"      Total: {len(communities):,} communities")

    # Save checkpoint
    checkpoint_data = {
        'communities': communities,
        'hierarchy': hierarchy,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'max_cluster_size': MAX_CLUSTER_SIZE,
            'use_lcc': USE_LCC,
            'random_seed': RANDOM_SEED
        }
    }
    with checkpoint_path.open('w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"   ✓ Checkpoint saved: {checkpoint_path.name}")

    return communities, hierarchy


# ============================================================================
# Embeddings & UMAP
# ============================================================================

def create_entity_embeddings(entities: Dict, relationships: List, client: OpenAI) -> Tuple[np.ndarray, List[str]]:
    """Generate embeddings for entities."""
    print(f"\n🧮 Generating entity embeddings...")

    # Check checkpoint
    checkpoint_path = CHECKPOINT_DIR / "embeddings.npy"
    entity_ids_path = CHECKPOINT_DIR / "entity_ids.json"

    if checkpoint_path.exists() and entity_ids_path.exists():
        print(f"   ✓ Loading from checkpoint")
        embeddings = np.load(checkpoint_path)
        with entity_ids_path.open() as f:
            entity_ids = json.load(f)
        print(f"   ✓ {len(entity_ids):,} embeddings loaded")
        return embeddings, entity_ids

    # Build relationship index for context
    rel_index = defaultdict(list)
    for rel in relationships:
        src = rel.get('source') or rel.get('source_entity')
        tgt = rel.get('target') or rel.get('target_entity')
        if src and tgt:
            rel_index[src].append({'target': tgt, 'predicate': rel.get('predicate', 'related_to')})
            rel_index[tgt].append({'target': src, 'predicate': rel.get('predicate', 'related_to')})

    # Build embedding texts
    entity_ids = list(entities.keys())
    texts = []

    for entity_id in entity_ids:
        entity_data = entities[entity_id]
        name = entity_data.get('name') or entity_id
        entity_type = entity_data.get('type', 'UNKNOWN')
        description = (entity_data.get('description') or '')[:300]

        # Add relationship context (top 2)
        rels = rel_index.get(entity_id, [])[:2]
        rel_context = ' | '.join([f"{r['predicate']} {r['target']}" for r in rels])

        text = f"{name} | {entity_type} | {description}"
        if rel_context:
            text += f" | {rel_context}"
        texts.append(text)

    # Generate embeddings in batches
    print(f"   ⏳ Generating {len(texts):,} embeddings...")
    embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

        if (i + EMBED_BATCH_SIZE) % 1000 == 0:
            print(f"      Progress: {i + EMBED_BATCH_SIZE:,}/{len(texts):,}")
        time.sleep(EMBED_RATE_DELAY)

    embeddings = np.array(embeddings)

    # Save checkpoint
    np.save(checkpoint_path, embeddings)
    with entity_ids_path.open('w') as f:
        json.dump(entity_ids, f)
    print(f"   ✓ Embeddings saved: {embeddings.shape}")

    return embeddings, entity_ids


def compute_umap_positions(embeddings: np.ndarray) -> np.ndarray:
    """Compute 3D UMAP positions."""
    print(f"\n🗺️  Computing UMAP 3D positions...")

    checkpoint_path = CHECKPOINT_DIR / "umap_positions.npy"
    if checkpoint_path.exists():
        print(f"   ✓ Loading from checkpoint")
        positions = np.load(checkpoint_path)
        return positions

    print(f"   ⏳ Running UMAP (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST})...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED
    )

    positions = reducer.fit_transform(embeddings)

    np.save(checkpoint_path, positions)
    print(f"   ✓ UMAP positions computed: {positions.shape}")

    return positions


# ============================================================================
# LLM Community Summarization
# ============================================================================

def generate_community_summary(
    community_entities: List[str],
    entities: Dict,
    relationships: List,
    client: OpenAI,
    level: int
) -> str:
    """Generate LLM summary for a community using gpt-4.1-mini."""

    # Sample entities if too many
    sampled_entities = community_entities[:SUMMARY_MAX_ENTITIES]

    # Build context
    entity_descriptions = []
    for entity_id in sampled_entities:
        entity_data = entities.get(entity_id, {})
        name = entity_data.get('name') or entity_id
        entity_type = entity_data.get('type', 'UNKNOWN')
        desc = entity_data.get('description', '')[:200]
        entity_descriptions.append(f"- {name} ({entity_type}): {desc}")

    # Get relevant relationships
    entity_set = set(sampled_entities)
    relevant_rels = []
    for rel in relationships:
        src = rel.get('source') or rel.get('source_entity')
        tgt = rel.get('target') or rel.get('target_entity')
        if src in entity_set and tgt in entity_set:
            predicate = rel.get('predicate', 'related_to')
            relevant_rels.append(f"{src} --{predicate}-> {tgt}")
            if len(relevant_rels) >= SUMMARY_MAX_RELATIONSHIPS:
                break

    # Build prompt
    prompt = f"""You are analyzing a community of related entities in a knowledge graph.

Community contains {len(community_entities)} entities. Here are the key entities:

{chr(10).join(entity_descriptions)}

Key relationships:
{chr(10).join(relevant_rels[:10])}

Generate a concise 2-4 word title that captures the main theme of this community. The title should be:
- Specific and descriptive
- Professional and clear
- Focused on the common theme

Just return the title, nothing else."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()

        # Remove quotes if present
        summary = summary.strip('"\'')

        time.sleep(LLM_RATE_DELAY)

        return summary

    except Exception as e:
        print(f"      ⚠️  LLM error: {e}")
        # Fallback to simple naming
        return f"Community {level}"


def summarize_all_communities(
    communities: List[Tuple],
    entities: Dict,
    relationships: List,
    client: OpenAI
) -> Dict[int, str]:
    """Generate summaries for all communities."""
    print(f"\n📝 Generating community summaries with {LLM_MODEL}...")

    checkpoint_path = CHECKPOINT_DIR / "community_summaries.json"
    if checkpoint_path.exists():
        print(f"   ✓ Loading from checkpoint")
        with checkpoint_path.open() as f:
            return json.load(f)

    summaries = {}
    total = len(communities)

    for idx, (level, cluster_id, parent_id, nodes) in enumerate(communities, 1):
        if idx % 50 == 0:
            print(f"   Progress: {idx}/{total} ({idx/total*100:.1f}%)")

        summary = generate_community_summary(nodes, entities, relationships, client, level)
        summaries[cluster_id] = summary

    # Save checkpoint
    with checkpoint_path.open('w') as f:
        json.dump(summaries, f, indent=2)

    print(f"   ✓ {len(summaries):,} summaries generated")

    return summaries


# ============================================================================
# Export
# ============================================================================

def export_graphrag_hierarchy(
    entities: Dict,
    relationships: List,
    communities: List[Tuple],
    hierarchy: Dict,
    summaries: Dict[int, str],
    entity_ids: List[str],
    positions: np.ndarray,
    output_path: Path
):
    """Export in kg-enhanced.js compatible format."""
    print(f"\n💾 Exporting GraphRAG hierarchy...")

    # Create entity position map
    entity_to_position = {entity_id: positions[idx].tolist() for idx, entity_id in enumerate(entity_ids)}

    # Build output structure
    output = {
        'entities': entities,
        'relationships': relationships,
        'clusters': {},
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'method': 'microsoft_graphrag_hierarchical_leiden',
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'total_communities': len(communities),
            'llm_model': LLM_MODEL,
            'max_cluster_size': MAX_CLUSTER_SIZE
        }
    }

    # Map Leiden levels to kg-enhanced.js levels (reversed)
    # Leiden: 0 (coarse) → 1 (medium) → 2 (fine)
    # kg-enhanced: level_3 (coarse) → level_2 (medium) → level_1 (fine)
    level_mapping = {0: 3, 1: 2, 2: 1}

    # Build parent-child relationships
    cluster_children = {}
    for level, cluster_id, parent_id, nodes in communities:
        if parent_id != -1:
            parent_key = f"c{parent_id}"
            if parent_key not in cluster_children:
                cluster_children[parent_key] = []
            cluster_children[parent_key].append(f"c{cluster_id}")

    # Organize clusters by level
    for level, cluster_id, parent_id, nodes in communities:
        # Map to kg-enhanced.js level numbering
        mapped_level = level_mapping.get(level, level)
        level_key = f"level_{mapped_level}"

        if level_key not in output['clusters']:
            output['clusters'][level_key] = {}

        # Add entity positions and metadata
        cluster_entities = {}
        entity_positions = []

        for entity_id in nodes:
            if entity_id in entity_to_position:
                entity_data = entities.get(entity_id, {})
                entity_pos = entity_to_position[entity_id]
                entity_positions.append(entity_pos)

                cluster_entities[entity_id] = {
                    'id': entity_id,
                    'type': 'entity',
                    'entity': entity_data,
                    'position': entity_pos,
                    'umap_position': entity_pos
                }

        # Calculate cluster centroid position
        if entity_positions:
            cluster_position = [
                sum(p[0] for p in entity_positions) / len(entity_positions),
                sum(p[1] for p in entity_positions) / len(entity_positions),
                sum(p[2] for p in entity_positions) / len(entity_positions)
            ]
        else:
            cluster_position = [0, 0, 0]

        # Scale positions based on hierarchy level to spread out higher levels
        # Level 3 (coarsest) needs most spacing, Level 1 (finest) needs least
        position_scale = {3: 20.0, 2: 8.0, 1: 3.0, 0: 1.0}
        scale_factor = position_scale.get(mapped_level, 1.0)
        cluster_position = [p * scale_factor for p in cluster_position]

        cluster_name = summaries.get(str(cluster_id), f"Community {cluster_id}")
        cluster_key = f"c{cluster_id}"

        output['clusters'][level_key][cluster_key] = {
            'id': cluster_key,
            'name': cluster_name,
            'level': mapped_level,  # Use mapped level
            'parent': f"c{parent_id}" if parent_id != -1 else None,
            'children': cluster_children.get(cluster_key, []),
            'entities': list(cluster_entities.keys()),
            'entity_data': cluster_entities,
            'position': cluster_position,  # Add cluster position
            'size': len(nodes)
        }

    # Write output
    with output_path.open('w') as f:
        json.dump(output, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Exported to: {output_path.name}")
    print(f"   ✓ File size: {file_size_mb:.1f} MB")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("=" * 70)
    print("🚀 Microsoft GraphRAG Pipeline")
    print("=" * 70)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load data
    entities, relationships = load_discourse_graph(INPUT_GRAPH)

    # Build graph
    G = build_networkx_graph(entities, relationships)

    # Cluster with hierarchical Leiden
    communities, hierarchy = cluster_with_hierarchical_leiden(G)

    # Generate embeddings
    embeddings, entity_ids = create_entity_embeddings(entities, relationships, client)

    # Compute 3D positions
    positions = compute_umap_positions(embeddings)

    # Generate LLM summaries
    summaries = summarize_all_communities(communities, entities, relationships, client)

    # Export
    export_graphrag_hierarchy(
        entities,
        relationships,
        communities,
        hierarchy,
        summaries,
        entity_ids,
        positions,
        OUTPUT_PATH
    )

    print("\n" + "=" * 70)
    print("✅ Pipeline complete!")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Copy to production: sudo cp {OUTPUT_PATH} /opt/yonearth-chatbot/web/data/graphrag_hierarchy/")
    print(f"  2. Update kg-enhanced.js to load new file")
    print(f"  3. Test visualization at https://gaiaai.xyz/YonEarth/graph/")


if __name__ == "__main__":
    main()
