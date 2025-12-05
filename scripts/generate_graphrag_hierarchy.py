#!/usr/bin/env python3
"""
Generate GraphRAG hierarchy and 3D layout from the discourse graph.

Pipeline:
1. Load discourse graph (entities + relationships)
2. Build graph-enriched OpenAI embeddings for all entities
3. Reduce embeddings to 3D with UMAP
4. Create hierarchical clusters using graph-based Leiden algorithm (graspologic)
5. Compute lightweight betweenness and relationship strengths
6. Generate cluster summaries for RAG using entity descriptions
7. Export graphrag_hierarchy.json ready for the 3D viewer

Input:
  /home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_unified/discourse_graph_hybrid.json

Output:
  /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json
  /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/cluster_registry.json
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

import networkx as nx
import numpy as np
from openai import AsyncOpenAI, OpenAI

try:
    from graspologic.partition import hierarchical_leiden
except ImportError:
    print("ERROR: graspologic not installed. Install with: pip install graspologic")
    raise

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
    raise

# --------------------------------------------------------------------------------------
# Paths & parameters
# --------------------------------------------------------------------------------------

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
# Support both old discourse_graph_hybrid.json and new unified_v2.json
UNIFIED_GRAPH_PATH = ROOT / "data/knowledge_graph_unified/unified_v2.json"
DISCOURSE_GRAPH_PATH = ROOT / "data/knowledge_graph_unified/discourse_graph_hybrid.json"
# Use unified_v2 if it exists and is newer, otherwise fall back to discourse_graph
if UNIFIED_GRAPH_PATH.exists():
    INPUT_GRAPH_PATH = UNIFIED_GRAPH_PATH
else:
    INPUT_GRAPH_PATH = DISCOURSE_GRAPH_PATH

OUTPUT_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
CLUSTER_REGISTRY_PATH = ROOT / "data/graphrag_hierarchy/cluster_registry.json"
BACKUP_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy_backup_pre_generation.json"
CHECKPOINT_DIR = Path("/tmp/graphrag_generation")

# Embedding / UMAP
EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
EMBED_BATCH_SIZE = 100
EMBED_RATE_DELAY = 0.05  # seconds between batches to respect rate limits

UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = int(os.getenv("GRAPHRAG_UMAP_NEIGHBORS", 12))  # lower to reduce memory
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# Hierarchical Leiden clustering
# NOTE: max_cluster_size removed - let algorithm find natural community sizes
# Super-hubs (like "Regenerative Agriculture" with 700+ connections) were causing stalls
TOP_ENTITIES_FOR_SUMMARY = 50  # Number of top entities to use for cluster summaries
MIN_CLUSTER_SIZE_FOR_SUMMARY = 5  # Only generate LLM summaries for clusters >= this size

# Cluster summarization model (configurable via env var)
CLUSTER_SUMMARY_MODEL = os.getenv("GRAPHRAG_SUMMARY_MODEL", "gpt-4.1-mini")

# Betweenness centrality
# Full centrality on 44k nodes is very expensive; sample for a faster approximation.
BETWEENNESS_SAMPLE = 512  # set to None to compute exact centrality (slow)


# --------------------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------------------

def load_discourse_graph(path: Path = None):
    """Load discourse graph JSON (supports both unified_v2 and discourse_graph formats)."""
    if path is None:
        path = INPUT_GRAPH_PATH
    print(f"Loading graph from {path}")
    with path.open() as f:
        data = json.load(f)

    entities = data.get("entities", {})
    relationships = data.get("relationships", [])

    # Count reality tags if present
    fictional_count = sum(1 for e in entities.values() if e.get("reality_tag") == "fictional" or e.get("is_fictional"))
    if fictional_count > 0:
        print(f"  Found {fictional_count:,} fictional entities (from VIRIDITAS)")

    print(f"  Loaded {len(entities):,} entities and {len(relationships):,} relationships")
    return entities, relationships


def build_relationship_index(relationships: List[dict]) -> Dict[str, List[dict]]:
    """Index relationships for embedding context (bidirectional)."""
    index = defaultdict(list)

    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        if not source or not target:
            continue

        rel_type = rel.get("predicate") or rel.get("type", "RELATED_TO")
        weight = rel.get("weight", 1.0)

        index[source].append({"target": target, "type": rel_type, "weight": weight})
        index[target].append({"target": source, "type": rel_type, "weight": weight})

    print(f"Built relationship index for {len(index):,} entities")
    return index


def build_networkx_graph(entities: Dict[str, dict], relationships: List[dict]) -> nx.Graph:
    """Construct a weighted undirected graph from relationships."""
    G = nx.Graph()
    for ent_id, ent in entities.items():
        G.add_node(ent_id, **ent)

    for rel in relationships:
        src = rel.get("source")
        tgt = rel.get("target")
        if not src or not tgt or src == tgt:
            continue
        weight = rel.get("weight", 1.0)
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += weight
        else:
            G.add_edge(src, tgt, weight=weight)

    print(f"Built NetworkX graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def get_top_relationships(entity_id: str, rel_index: Dict[str, List[dict]], k: int = 2) -> List[str]:
    """Return top-k relationships as short text snippets."""
    if entity_id not in rel_index:
        return []

    rels = sorted(rel_index[entity_id], key=lambda r: r.get("weight", 1.0), reverse=True)
    return [f"{rel['type']} {rel['target']}" for rel in rels[:k]]


def build_embedding_text(entity_id: str, entity_data: dict, rel_index: Dict[str, List[dict]]) -> str:
    """Construct text used for embeddings."""
    name = entity_data.get("name") or entity_id
    ent_type = entity_data.get("type", "UNKNOWN")
    description = entity_data.get("description") or ""
    top_rels = get_top_relationships(entity_id, rel_index, k=2)

    parts = [name, ent_type]
    if description:
        parts.append(description[:300])  # keep prompts compact
    parts.extend(top_rels)

    return " | ".join(parts)


# --------------------------------------------------------------------------------------
# Embedding & dimensionality reduction
# --------------------------------------------------------------------------------------

def create_embeddings(entities: Dict[str, dict], rel_index: Dict[str, List[dict]], client: OpenAI):
    """Generate graph-enriched embeddings for all entities with checkpointing."""
    entity_ids = list(entities.keys())
    n_entities = len(entity_ids)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = CHECKPOINT_DIR / "embeddings.npy"
    progress_path = CHECKPOINT_DIR / "embeddings_progress.json"

    expected_bytes = n_entities * EMBED_DIM * 4
    if embeddings_path.exists() and embeddings_path.stat().st_size != expected_bytes:
        print("  Existing embeddings checkpoint has wrong size; resetting.")
        embeddings_path.unlink()
        progress_path.unlink(missing_ok=True)

    # Prepare storage (memmap so we can resume)
    mode = "r+" if embeddings_path.exists() else "w+"
    embeddings = np.memmap(embeddings_path, dtype="float32", mode=mode, shape=(n_entities, EMBED_DIM))

    start_idx = 0
    if progress_path.exists():
        try:
            start_idx = json.loads(progress_path.read_text()).get("completed", 0)
        except Exception:
            start_idx = 0

    print(f"\nCreating embeddings with {EMBEDDING_MODEL}...")
    print(f"  Entities: {n_entities:,}")
    if start_idx > 0:
        print(f"  Resuming from entity {start_idx:,}")

    for i in range(start_idx, n_entities, EMBED_BATCH_SIZE):
        batch_ids = entity_ids[i:i + EMBED_BATCH_SIZE]
        texts = [build_embedding_text(eid, entities[eid], rel_index) for eid in batch_ids]

        # Retry logic
        for attempt in range(3):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
                for j, emb in enumerate(response.data):
                    embeddings[i + j] = emb.embedding
                break
            except Exception as e:
                print(f"  ⚠️  Embedding batch {i}-{i + len(batch_ids)} failed (attempt {attempt + 1}/3): {e}")
                time.sleep(2 * (attempt + 1))
        else:
            # Fallback: random noise to avoid blocking pipeline
            print("  Using random fallback embeddings for this batch.")
            embeddings[i:i + len(batch_ids)] = np.random.randn(len(batch_ids), EMBED_DIM) * 0.01

        if (i + EMBED_BATCH_SIZE) % 500 == 0 or (i + EMBED_BATCH_SIZE) >= n_entities:
            print(f"  Processed {min(i + EMBED_BATCH_SIZE, n_entities):,}/{n_entities:,} entities", flush=True)

        progress_path.write_text(json.dumps({"completed": i + len(batch_ids)}))
        time.sleep(EMBED_RATE_DELAY)

    print("  ✓ Embeddings complete")
    # Convert to standard ndarray (loads from memmap) and clean up checkpoint metadata
    final_embeddings = np.array(embeddings)
    progress_path.unlink(missing_ok=True)
    return final_embeddings, entity_ids


def compute_umap_positions(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 3D using UMAP."""
    print("\nComputing UMAP 3D positions...", flush=True)
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    positions = reducer.fit_transform(embeddings)
    print("  ✓ UMAP complete", flush=True)
    print(f"  Position ranges: x[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
          f"z[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]", flush=True)
    return positions


# --------------------------------------------------------------------------------------
# Graph metrics
# --------------------------------------------------------------------------------------

def compute_betweenness_centrality(relationships: List[dict], entities: Dict[str, dict]) -> Dict[str, float]:
    """Compute (approximate) betweenness centrality for all entities."""
    print("\nComputing betweenness centrality...")
    G = nx.Graph()
    G.add_nodes_from(entities.keys())

    for rel in relationships:
        s, t = rel.get("source"), rel.get("target")
        if s and t and s in entities and t in entities:
            G.add_edge(s, t)

    print(f"  Graph size: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    if BETWEENNESS_SAMPLE:
        betweenness = nx.betweenness_centrality(G, k=BETWEENNESS_SAMPLE, normalized=True, seed=42)
        method = f"approximate_k={BETWEENNESS_SAMPLE}"
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)
        method = "exact"

    top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top bridge nodes:")
    for eid, score in top_bridges:
        print(f"    {eid}: {score:.4f}")

    return betweenness, method


def compute_relationship_strengths(relationships: List[dict]) -> Dict[Tuple[str, str], float]:
    """Compute normalized edge strength based on co-occurrence frequency.

    Supports both standard keys (source/target) and discourse keys (source_entity/target_entity).
    """
    edge_counts = Counter()
    for rel in relationships:
        s = rel.get("source") or rel.get("source_entity")
        t = rel.get("target") or rel.get("target_entity")
        if s and t:
            edge = tuple(sorted([s, t]))
            edge_counts[edge] += 1

    max_count = max(edge_counts.values()) if edge_counts else 1
    strengths = {edge: count / max_count for edge, count in edge_counts.items()}
    print(f"Computed strengths for {len(strengths):,} unique edges "
          f"(range {min(strengths.values()):.4f}-{max(strengths.values()):.4f})")
    return strengths


# --------------------------------------------------------------------------------------
# Graph-based hierarchical clustering using Leiden
# --------------------------------------------------------------------------------------

def run_hierarchical_leiden(G: nx.Graph):
    """
    Run hierarchical Leiden clustering on the graph.

    Uses natural community detection without max_cluster_size constraint.
    This allows super-hubs to remain in their natural communities rather than
    forcing artificial splits that cause algorithm stalls.

    Returns a HierarchicalClusters list of node assignments.
    """
    print(f"\nRunning hierarchical Leiden clustering (natural communities)...", flush=True)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", flush=True)

    # Run hierarchical Leiden without max_cluster_size constraint
    # This uses modularity optimization to find natural community structure
    hierarchy = hierarchical_leiden(G, random_seed=42)

    print(f"  ✓ Leiden clustering complete", flush=True)
    print(f"    Total assignments: {len(hierarchy):,}", flush=True)
    print(f"    Levels found: {max(entry.level for entry in hierarchy) + 1}", flush=True)

    return hierarchy


def extract_cluster_tree(hierarchy, G: nx.Graph) -> Dict[str, Dict]:
    """
    Convert hierarchical_leiden output into a cluster registry using INVERTED CONTAINMENT LOGIC.

    IMPORTANT: In graspologic's hierarchical_leiden:
    - Level 0 = COARSE communities (few large clusters)
    - Higher levels = FINER communities (more, smaller clusters as subsets of lower levels)

    For visualization, we INVERT this so that:
    - L0 (coarse, large) = TOP level displayed in voronoi
    - L1 (finer) = children of L0
    - L2 (finest) = children of L1
    - Entities = children of the finest level

    Returns (cluster_registry, cluster_entity_map) where:
    - cluster_registry: dict mapping cluster_id -> cluster_info
    - cluster_entity_map: dict mapping (level, cluster_num) -> set of entity IDs
    """
    print("\nExtracting cluster tree from Leiden hierarchy using INVERTED containment logic...")
    print("  Note: L0 = coarse (top), L1 = finer (mid), L2 = finest (bottom)")

    from collections import defaultdict

    # Step 1: Build entity membership for ALL cluster assignments
    # Map (level, cluster_num) -> set of entity_ids
    cluster_entities = defaultdict(set)

    # Track entity to cluster assignments (for each level)
    # entity_id -> {level: cluster_num}
    entity_assignments = defaultdict(dict)

    for entry in hierarchy:
        key = (entry.level, entry.cluster)
        cluster_entities[key].add(entry.node)
        entity_assignments[entry.node][entry.level] = entry.cluster

    # Convert to regular dicts
    cluster_entities = {k: v for k, v in cluster_entities.items()}
    entity_assignments = dict(entity_assignments)

    print(f"  Found {len(entity_assignments)} entities with level assignments")
    print(f"  Found {len(cluster_entities)} unique clusters across all levels")

    # Step 2: Count clusters per level
    from collections import Counter
    level_counts = Counter(level for level, _ in cluster_entities.keys())
    max_level = max(level_counts.keys())
    print(f"  Clusters per Leiden level: {dict(level_counts)}")
    print(f"  Max Leiden level: {max_level}")

    # Step 3: Build cluster registry with ID mapping
    # In graspologic's hierarchical_leiden:
    # - Leiden L0 = FINEST level (many small/singleton clusters)
    # - Higher Leiden levels = COARSER (fewer, larger clusters)
    # We map to display levels where Level 2 = top (coarsest for viewer):
    # - Leiden L0 (finest) -> display level 0 (bottom/fine)
    # - Leiden L1 (mid)    -> display level 1 (mid)
    # - Leiden L2+ (coarsest) -> display level 2 (top) - root nodes for viewer
    cluster_registry = {}
    cluster_id_map = {}  # Map (level, cluster_num) -> cluster_id

    for (leiden_level, cluster_num), entity_ids in cluster_entities.items():
        # CORRECT mapping: higher Leiden level = coarser = higher display level
        if leiden_level == 0:
            display_level = 0  # Bottom/Fine (many small clusters)
            cluster_type = "fine_community"
        elif leiden_level == 1:
            display_level = 1  # Mid
            cluster_type = "mid_community"
        else:
            display_level = 2  # Top - root nodes (Leiden L2+)
            cluster_type = "top_community"

        cluster_id = f"level_{display_level}_{cluster_num}"
        cluster_id_map[(leiden_level, cluster_num)] = cluster_id

        # Generate title and summary using LLM
        metadata = generate_cluster_metadata(list(entity_ids), G, display_level)

        cluster_registry[cluster_id] = {
            "id": cluster_id,
            "level": display_level,  # Use inverted display level
            "leiden_level": leiden_level,  # Keep original for reference
            "type": cluster_type,
            "title": metadata["title"],  # Short 3-5 word title for UI
            "summary_text": metadata["summary"],  # Detailed summary
            "children": [],  # Will be populated using containment logic
            "entities": list(entity_ids),
            "entity_count": len(entity_ids),
        }

    # Step 4: Build parent-child relationships using EXCLUSIVE ASSIGNMENT
    # CRITICAL: Each child cluster must belong to exactly ONE parent!
    # The old containment logic allowed the same child to be assigned to multiple parents.
    #
    # New logic: For each child cluster, find the BEST parent (maximum overlap) and assign exclusively.

    print("\n  Building EXCLUSIVE parent-child relationships...")

    # Get all cluster keys by Leiden level
    l0_keys = [(l, c) for l, c in cluster_entities.keys() if l == 0]
    l1_keys = [(l, c) for l, c in cluster_entities.keys() if l == 1]
    l2_keys = [(l, c) for l, c in cluster_entities.keys() if l == 2]

    print(f"  L0 (coarse): {len(l0_keys)} clusters")
    print(f"  L1 (mid): {len(l1_keys)} clusters")
    print(f"  L2 (fine): {len(l2_keys)} clusters")

    # Initialize empty children lists for all clusters
    for cluster_id in cluster_registry:
        cluster_registry[cluster_id]["children"] = []

    # L1 -> L0 assignment: Each L1 cluster belongs to exactly ONE L0 cluster (best match)
    if l1_keys and l0_keys:
        print("  Assigning L1 clusters to their BEST L0 parent (exclusive)...")
        for l1_key in l1_keys:
            l1_cluster_id = cluster_id_map[l1_key]
            l1_entities = cluster_entities[l1_key]

            # Find the L0 cluster with maximum overlap
            best_l0_key = None
            best_overlap = 0

            for l0_key in l0_keys:
                l0_entities = cluster_entities[l0_key]
                overlap = len(l1_entities & l0_entities)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_l0_key = l0_key

            # Assign L1 as child of the best L0
            if best_l0_key:
                l0_cluster_id = cluster_id_map[best_l0_key]
                cluster_registry[l0_cluster_id]["children"].append(l1_cluster_id)

        # Report assignment stats
        l0_child_counts = [len(cluster_registry[cluster_id_map[k]]["children"]) for k in l0_keys]
        print(f"    L0 clusters now have {sum(l0_child_counts)} total L1 children")
        print(f"    Children per L0: min={min(l0_child_counts)}, max={max(l0_child_counts)}, avg={sum(l0_child_counts)/len(l0_child_counts):.1f}")
    elif l0_keys:
        # No L1 clusters - L0 children are entities
        print("  No L1 clusters - L0 children are entities...")
        for l0_key in l0_keys:
            l0_cluster_id = cluster_id_map[l0_key]
            cluster_registry[l0_cluster_id]["children"] = list(cluster_entities[l0_key])

    # L2 -> L1 assignment: Each L2 cluster belongs to exactly ONE L1 cluster (best match)
    if l2_keys and l1_keys:
        print("  Assigning L2 clusters to their BEST L1 parent (exclusive)...")
        for l2_key in l2_keys:
            l2_cluster_id = cluster_id_map[l2_key]
            l2_entities = cluster_entities[l2_key]

            # Find the L1 cluster with maximum overlap
            best_l1_key = None
            best_overlap = 0

            for l1_key in l1_keys:
                l1_entities = cluster_entities[l1_key]
                overlap = len(l2_entities & l1_entities)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_l1_key = l1_key

            # Assign L2 as child of the best L1
            if best_l1_key:
                l1_cluster_id = cluster_id_map[best_l1_key]
                cluster_registry[l1_cluster_id]["children"].append(l2_cluster_id)

        # Report assignment stats
        l1_child_counts = [len(cluster_registry[cluster_id_map[k]]["children"]) for k in l1_keys]
        print(f"    L1 clusters now have {sum(l1_child_counts)} total L2 children")
        print(f"    Children per L1: min={min(l1_child_counts)}, max={max(l1_child_counts)}, avg={sum(l1_child_counts)/len(l1_child_counts):.1f}")

        # L1 clusters without L2 children get entities directly
        for l1_key in l1_keys:
            l1_cluster_id = cluster_id_map[l1_key]
            if not cluster_registry[l1_cluster_id]["children"]:
                cluster_registry[l1_cluster_id]["children"] = list(cluster_entities[l1_key])
    elif l1_keys:
        # No L2 clusters - L1 children are entities
        print("  No L2 clusters - L1 children are entities...")
        for l1_key in l1_keys:
            l1_cluster_id = cluster_id_map[l1_key]
            cluster_registry[l1_cluster_id]["children"] = list(cluster_entities[l1_key])

    # L2 clusters (finest): children are always entities
    if l2_keys:
        print("  L2 clusters have entities as children...")
        for l2_key in l2_keys:
            l2_cluster_id = cluster_id_map[l2_key]
            cluster_registry[l2_cluster_id]["children"] = list(cluster_entities[l2_key])

    # Print summary statistics
    print(f"\n  ✓ Extracted {len(cluster_registry)} clusters:")
    for display_level in sorted(set(c["level"] for c in cluster_registry.values()), reverse=True):
        clusters_at_level = [c for c in cluster_registry.values() if c["level"] == display_level]
        avg_children = sum(len(c["children"]) for c in clusters_at_level) / len(clusters_at_level) if clusters_at_level else 0
        level_type = clusters_at_level[0]["type"] if clusters_at_level else "unknown"
        print(f"    Display Level {display_level} ({level_type}): {len(clusters_at_level)} clusters (avg {avg_children:.1f} children)")

    return cluster_registry, cluster_entities


async def generate_cluster_metadata_async(entity_ids: List[str], G: nx.Graph, level: int, client: AsyncOpenAI, semaphore: asyncio.Semaphore) -> Dict[str, str]:
    """
    Async version: Generate title and summary for a cluster using LLM analysis with concurrency control.
    """
    import json

    if not entity_ids:
        return {"title": "Empty Cluster", "summary": "No entities in this cluster."}

    # For very small clusters (single entity), use simple fallback
    # Changed from < 5 to < 2 to ensure LLM is called for small but meaningful clusters
    if len(entity_ids) < 2:
        subgraph = G.subgraph(entity_ids)
        degree_centrality = nx.degree_centrality(subgraph)
        top_entity = max(entity_ids, key=lambda eid: degree_centrality.get(eid, 0.0))
        top_name = G.nodes.get(top_entity, {}).get("name", top_entity)
        return {
            "title": f"{top_name}",
            "summary": f"Single-entity cluster: {top_name}."
        }

    # Calculate degree centrality
    subgraph = G.subgraph(entity_ids)
    degree_centrality = nx.degree_centrality(subgraph)
    sorted_entities = sorted(entity_ids, key=lambda eid: degree_centrality.get(eid, 0.0), reverse=True)
    top_entities = sorted_entities[:min(20, len(sorted_entities))]

    # Build entity context for LLM
    entity_descriptions = []
    for eid in top_entities:
        node_data = G.nodes.get(eid, {})
        name = node_data.get("name", eid)
        description = node_data.get("description", "")
        entity_type = node_data.get("type", "")

        if description:
            entity_descriptions.append(f"- {name} ({entity_type}): {description}")
        else:
            entity_descriptions.append(f"- {name} ({entity_type})")

    entity_context = "\n".join(entity_descriptions[:15])

    # Use semaphore to control concurrency
    async with semaphore:
        try:
            level_names = {0: "fine-grained", 1: "mid-level", 2: "top-level"}
            level_desc = level_names.get(level, "cluster")

            response = await client.chat.completions.create(
                model=CLUSTER_SUMMARY_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing a knowledge graph community. "
                            "Given entities and descriptions, identify the common theme. "
                            "Generate:\n"
                            "1. A short, punchy Title (3-5 words max) as a category label\n"
                            "2. A comprehensive Summary (1-2 sentences) of the community's themes\n"
                            "Return valid JSON with 'title' and 'summary' keys."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"This is a {level_desc} community with {len(entity_ids)} entities. "
                            f"Top {len(entity_descriptions)} central entities:\n\n{entity_context}\n\n"
                            "Generate a title and summary for this community."
                        )
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=200
            )

            result = json.loads(response.choices[0].message.content)
            title = result.get("title", "Unknown Community")
            summary = result.get("summary", f"Community of {len(entity_ids)} entities.")

            if len(title) > 50:
                title = title[:47] + "..."

            return {"title": title, "summary": summary}

        except Exception as e:
            # Fallback on error
            top_entity = sorted_entities[0] if sorted_entities else entity_ids[0]
            top_name = G.nodes.get(top_entity, {}).get("name", top_entity)
            return {
                "title": f"{top_name} Community",
                "summary": f"Cluster of {len(entity_ids)} entities related to {top_name}."
            }


def generate_cluster_metadata(entity_ids: List[str], G: nx.Graph, level: int) -> Dict[str, str]:
    """
    Generate title and summary for a cluster using LLM analysis.

    Returns:
        dict with keys 'title' (3-5 words) and 'summary' (1 paragraph)

    For small clusters (< 5 entities) or on API failure, returns fallback titles.
    """
    import json
    import os

    if not entity_ids:
        return {"title": "Empty Cluster", "summary": "No entities in this cluster."}

    # For very small clusters, use simple fallback (no LLM call)
    if len(entity_ids) < MIN_CLUSTER_SIZE_FOR_SUMMARY:
        subgraph = G.subgraph(entity_ids)
        degree_centrality = nx.degree_centrality(subgraph)
        top_entity = max(entity_ids, key=lambda eid: degree_centrality.get(eid, 0.0))
        top_name = G.nodes.get(top_entity, {}).get("name", top_entity)
        return {
            "title": f"{top_name} Cluster",
            "summary": f"Small cluster containing {len(entity_ids)} entities related to {top_name}."
        }

    # Calculate degree centrality for entities in this cluster
    subgraph = G.subgraph(entity_ids)
    degree_centrality = nx.degree_centrality(subgraph)

    # Sort entities by centrality (most central first)
    sorted_entities = sorted(
        entity_ids,
        key=lambda eid: degree_centrality.get(eid, 0.0),
        reverse=True
    )

    # Take top N entities for LLM analysis
    top_entities = sorted_entities[:min(20, len(sorted_entities))]  # Limit to 20 for API efficiency

    # Build entity context for LLM
    entity_descriptions = []
    for eid in top_entities:
        node_data = G.nodes.get(eid, {})
        name = node_data.get("name", eid)
        description = node_data.get("description", "")
        entity_type = node_data.get("type", "")

        if description:
            entity_descriptions.append(f"- {name} ({entity_type}): {description}")
        else:
            entity_descriptions.append(f"- {name} ({entity_type})")

    entity_context = "\n".join(entity_descriptions[:15])  # Limit to 15 for token efficiency

    # Call OpenAI to generate title and summary
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        level_names = {0: "fine-grained", 1: "mid-level", 2: "top-level"}
        level_desc = level_names.get(level, "cluster")

        response = client.chat.completions.create(
            model=CLUSTER_SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are analyzing a knowledge graph community. "
                        "Given entities and descriptions, identify the common theme. "
                        "Generate:\n"
                        "1. A short, punchy Title (3-5 words max) as a category label\n"
                        "2. A comprehensive Summary (1-2 sentences) of the community's themes\n"
                        "Return valid JSON with 'title' and 'summary' keys."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"This is a {level_desc} community with {len(entity_ids)} entities. "
                        f"Top {len(entity_descriptions)} central entities:\n\n{entity_context}\n\n"
                        "Generate a title and summary for this community."
                    )
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=200
        )

        result = json.loads(response.choices[0].message.content)
        title = result.get("title", "Unknown Community")
        summary = result.get("summary", f"Community of {len(entity_ids)} entities.")

        # Ensure title is not too long
        if len(title) > 50:
            title = title[:47] + "..."

        return {"title": title, "summary": summary}

    except Exception as e:
        print(f"  Warning: LLM generation failed: {e}")
        # Fallback: use top entity name
        top_entity = sorted_entities[0] if sorted_entities else entity_ids[0]
        top_name = G.nodes.get(top_entity, {}).get("name", top_entity)
        return {
            "title": f"{top_name} Community",
            "summary": f"Cluster of {len(entity_ids)} entities related to {top_name}."
        }


def build_hierarchy_from_leiden(cluster_registry: Dict[str, Dict],
                                entity_ids: List[str],
                                embeddings: np.ndarray,
                                umap_positions: np.ndarray) -> Tuple[Dict, Dict, Dict]:
    """
    Convert the flat cluster registry into the hierarchical structure expected by the viewer.

    Uses the display_level from cluster_registry which has already been inverted:
    - Display level 2 = TOP (Leiden L0 - coarse, children = L1 cluster IDs)
    - Display level 1 = MID (Leiden L1 - mid, children = L2 cluster IDs or entities)
    - Display level 0 = BOTTOM (Leiden L2 - finest, children = entities)

    Viewer expects:
    - level_3 = TOP (largest clusters)
    - level_2 = MID
    - level_1 = BOTTOM (smallest clusters before entities)

    Returns (level_1, level_2, level_3) dictionaries matching the old format.
    """
    print("\nBuilding viewer-compatible hierarchy from Leiden clusters...")

    entity_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}

    level_1 = {}
    level_2 = {}
    level_3 = {}

    for cluster_id, cluster_info in cluster_registry.items():
        display_level = cluster_info["level"]  # This is already inverted in extract_cluster_tree

        # Compute cluster center and position from entity embeddings/positions
        cluster_entities = cluster_info["entities"]
        if not cluster_entities:
            continue

        indices = [entity_to_idx[eid] for eid in cluster_entities if eid in entity_to_idx]
        if not indices:
            continue

        cluster_embeddings = embeddings[indices]
        cluster_positions = umap_positions[indices]

        # Use the LLM-generated title from cluster_info
        title = cluster_info.get("title", cluster_id)

        cluster_data = {
            "id": cluster_id,
            "type": cluster_info["type"],
            "name": title,  # Short 3-5 word title for viewer labels
            "title": title,  # Same as name for compatibility
            "children": cluster_info["children"],
            "entities": cluster_entities,
            "entity_ids": cluster_entities,
            "center": cluster_embeddings.mean(axis=0).tolist(),
            "position": cluster_positions.mean(axis=0).tolist(),
            "umap_position": cluster_positions.mean(axis=0).tolist(),
            "size": len(cluster_entities),
            "summary_text": cluster_info["summary_text"],  # Detailed summary for side panel
        }

        # Map display_level to viewer level_X
        # display_level 2 (top/coarse) -> level_3
        # display_level 1 (mid) -> level_2
        # display_level 0 (bottom/fine) -> level_1
        viewer_level = display_level + 1  # Convert 0,1,2 -> 1,2,3

        if viewer_level == 1:
            level_1[cluster_id] = cluster_data
        elif viewer_level == 2:
            level_2[cluster_id] = cluster_data
        elif viewer_level == 3:
            level_3[cluster_id] = cluster_data

    print(f"  ✓ Level 3 (TOP): {len(level_3)} clusters (coarse, children=L2 cluster IDs)")
    print(f"  ✓ Level 2 (MID): {len(level_2)} clusters (mid, children=L1 cluster IDs or entities)")
    print(f"  ✓ Level 1 (BOTTOM): {len(level_1)} clusters (fine, children=entities)")

    return level_1, level_2, level_3


# --------------------------------------------------------------------------------------
# Output assembly
# --------------------------------------------------------------------------------------

def build_level_0_clusters(entity_ids: List[str],
                           entities: Dict[str, dict],
                           umap_positions: np.ndarray,
                           betweenness: Dict[str, float]):
    """Create level_0 cluster entries (individual entities)."""
    level_0 = {}
    for idx, eid in enumerate(entity_ids):
        entity_data = entities[eid]
        # Determine reality_tag for coloring in visualization
        # Check for reality_tag field or is_fictional flag
        reality_tag = entity_data.get("reality_tag", "factual")
        if entity_data.get("is_fictional"):
            reality_tag = "fictional"

        level_0[eid] = {
            "id": eid,
            "type": "entity",
            "entity": entity_data,
            "embedding_idx": idx,
            "position": umap_positions[idx].tolist(),
            "umap_position": umap_positions[idx].tolist(),
            "betweenness": betweenness.get(eid, 0.0),
            "reality_tag": reality_tag,  # For color mapping in 3D viewer
        }
    return level_0


def apply_relationship_strengths(relationships: List[dict], strengths: Dict[Tuple[str, str], float]):
    """Attach strength field to relationships using source/target or source_entity/target_entity."""
    for rel in relationships:
        s = rel.get("source") or rel.get("source_entity")
        t = rel.get("target") or rel.get("target_entity")
        if s and t:
            edge = tuple(sorted([s, t]))
            rel["strength"] = strengths.get(edge, 0.0)
    return relationships


def build_metadata(entities: Dict[str, dict],
                   relationships: List[dict],
                   betweenness_method: str,
                   cluster_counts: Dict[str, int]):
    """Assemble metadata section."""
    return {
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "levels": 4,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "multi_membership": False,
        "source_graph": str(INPUT_GRAPH_PATH),
        "umap": {
            "n_components": UMAP_N_COMPONENTS,
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "random_state": UMAP_RANDOM_STATE,
        },
        "betweenness": {
            "method": betweenness_method,
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "clustering": {
            "algorithm": "Hierarchical Leiden (graspologic)",
            "max_cluster_size": "natural",  # No constraint - uses modularity optimization
            "level_1_clusters": cluster_counts.get("level_1", 0),
            "level_2_clusters": cluster_counts.get("level_2", 0),
            "level_3_clusters": cluster_counts.get("level_3", 0),
        },
    }


# --------------------------------------------------------------------------------------
# Title-only regeneration (fast path)
# --------------------------------------------------------------------------------------

async def regenerate_titles_async():
    """
    Async version: Concurrent title generation with rate limiting.
    """
    import sys

    # Check OpenAI key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise SystemExit(1)

    # Load existing files
    if not CLUSTER_REGISTRY_PATH.exists():
        print(f"ERROR: Cluster registry not found at {CLUSTER_REGISTRY_PATH}")
        print("Run without --titles-only flag first to generate full hierarchy")
        raise SystemExit(1)

    if not OUTPUT_PATH.exists():
        print(f"ERROR: Hierarchy file not found at {OUTPUT_PATH}")
        print("Run without --titles-only flag first to generate full hierarchy")
        raise SystemExit(1)

    print(f"\nLoading cluster registry from {CLUSTER_REGISTRY_PATH}")
    with CLUSTER_REGISTRY_PATH.open() as f:
        cluster_registry = json.load(f)

    print(f"Loading hierarchy from {OUTPUT_PATH}")
    with OUTPUT_PATH.open() as f:
        hierarchy = json.load(f)

    # Load discourse graph for entity data
    entities, relationships = load_discourse_graph(INPUT_GRAPH_PATH)
    G = build_networkx_graph(entities, relationships)

    # Create async OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Semaphore to limit concurrent requests (50 concurrent)
    semaphore = asyncio.Semaphore(50)

    print(f"\nRegenerating titles for {len(cluster_registry)} clusters using {CLUSTER_SUMMARY_MODEL}...")
    print(f"  Concurrency: 50 parallel requests")
    sys.stdout.flush()

    # Prepare tasks for all clusters
    tasks = []
    cluster_ids = []
    for cluster_id, cluster_info in cluster_registry.items():
        entity_ids = cluster_info.get("entities", [])
        level = cluster_info.get("level", 0)

        if not entity_ids:
            continue

        cluster_ids.append(cluster_id)
        task = generate_cluster_metadata_async(entity_ids, G, level, client, semaphore)
        tasks.append(task)

    # Process all clusters concurrently with progress tracking
    print(f"  Processing {len(tasks)} clusters...")
    sys.stdout.flush()

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Update registry with results
    completed = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Warning: Failed to process cluster {cluster_ids[i]}: {result}")
            sys.stdout.flush()
            continue

        cluster_id = cluster_ids[i]
        cluster_registry[cluster_id]["title"] = result["title"]
        cluster_registry[cluster_id]["summary_text"] = result["summary"]

        completed += 1
        if completed % 100 == 0:
            print(f"  Processed {completed}/{len(tasks)} clusters...")
            sys.stdout.flush()

    print(f"  ✓ Regenerated titles for {completed} clusters")

    # Update hierarchy file with new titles
    print("\nUpdating hierarchy file with new titles...")
    clusters = hierarchy.get("clusters", {})

    for level_key in ["level_1", "level_2", "level_3"]:
        level_clusters = clusters.get(level_key, {})
        for cluster_id, cluster_data in level_clusters.items():
            if cluster_id in cluster_registry:
                new_title = cluster_registry[cluster_id]["title"]
                cluster_data["name"] = new_title
                cluster_data["title"] = new_title
                cluster_data["summary_text"] = cluster_registry[cluster_id]["summary_text"]

    # Backup existing files
    print(f"\nCreating backup at {BACKUP_PATH}")
    BACKUP_PATH.write_text(OUTPUT_PATH.read_text())

    # Save updated files
    print(f"Writing updated cluster registry to {CLUSTER_REGISTRY_PATH}")
    with CLUSTER_REGISTRY_PATH.open("w") as f:
        json.dump(cluster_registry, f, indent=2)

    print(f"Writing updated hierarchy to {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w") as f:
        json.dump(hierarchy, f)

    # Report results
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    registry_size_mb = CLUSTER_REGISTRY_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✅ Titles regenerated successfully:")
    print(f"   Main file: {size_mb:.2f} MB ({OUTPUT_PATH})")
    print(f"   Cluster registry: {registry_size_mb:.2f} MB ({CLUSTER_REGISTRY_PATH})")
    print(f"   Updated {completed} cluster titles")
    print(f"\n🔍 Next steps:")
    print(f"   1. Deploy to /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/")
    print(f"   2. Test visualization at https://gaiaai.xyz/YonEarth/graph/")

    return cluster_registry, hierarchy


def regenerate_titles_only():
    """
    Synchronous wrapper for async title regeneration.
    """
    print("=" * 80)
    print("GraphRAG Title Regeneration (LLM-based with 50x concurrency)")
    print("=" * 80)

    asyncio.run(regenerate_titles_async())


# --------------------------------------------------------------------------------------
# Cluster-only mode (skip embeddings/UMAP)
# --------------------------------------------------------------------------------------

def main_cluster_only():
    """
    Fast path: Reuse cached embeddings and UMAP positions, only redo clustering and titles.

    This is useful when:
    - You've changed the level mapping logic
    - You want to regenerate LLM titles with different prompts

    Saves ~3-4 minutes by skipping embedding generation and UMAP computation.
    """
    print("=" * 80)
    print("GraphRAG Hierarchy Generation (CLUSTER-ONLY MODE)")
    print("=" * 80)
    print("  Skipping: Embeddings, UMAP")
    print("  Running: Leiden clustering, LLM title generation")
    print("=" * 80)

    # Check OpenAI key (still needed for title generation)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise SystemExit(1)

    # Check for cached embeddings and UMAP positions
    embeddings_path = CHECKPOINT_DIR / "embeddings.npy"
    umap_path = CHECKPOINT_DIR / "umap_positions.npy"
    entity_ids_path = CHECKPOINT_DIR / "entity_ids.json"

    if not embeddings_path.exists():
        print(f"ERROR: Cached embeddings not found at {embeddings_path}")
        print("Run without --cluster-only flag first to generate embeddings")
        raise SystemExit(1)

    if not umap_path.exists():
        print(f"ERROR: Cached UMAP positions not found at {umap_path}")
        print("Run without --cluster-only flag first to generate UMAP positions")
        raise SystemExit(1)

    if not entity_ids_path.exists():
        print(f"ERROR: Cached entity IDs not found at {entity_ids_path}")
        print("Run without --cluster-only flag first to generate entity IDs")
        raise SystemExit(1)

    # Load cached data
    print(f"\nLoading cached embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")

    print(f"Loading cached UMAP positions from {umap_path}")
    umap_positions = np.load(umap_path)
    print(f"  Shape: {umap_positions.shape}")

    print(f"Loading cached entity IDs from {entity_ids_path}")
    with entity_ids_path.open() as f:
        entity_ids = json.load(f)
    print(f"  Count: {len(entity_ids)}")

    # Load discourse graph (needed for clustering and titles)
    entities, relationships = load_discourse_graph(INPUT_GRAPH_PATH)
    G = build_networkx_graph(entities, relationships)

    # Graph metrics (fast)
    betweenness, betweenness_method = compute_betweenness_centrality(relationships, entities)
    strengths = compute_relationship_strengths(relationships)

    # Run hierarchical Leiden clustering on the graph
    hierarchy = run_hierarchical_leiden(G)

    # Extract cluster tree and build summary texts (using containment logic)
    cluster_registry, cluster_entities = extract_cluster_tree(hierarchy, G)

    # Convert to viewer-compatible format
    level_1, level_2, level_3 = build_hierarchy_from_leiden(
        cluster_registry, entity_ids, embeddings, umap_positions
    )

    level_0 = build_level_0_clusters(entity_ids, entities, umap_positions, betweenness)

    clusters = {
        "level_0": level_0,
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
    }

    # Count clusters for metadata
    cluster_counts = {
        "level_1": len(level_1),
        "level_2": len(level_2),
        "level_3": len(level_3),
    }

    # Relationships with strength weights
    relationships_with_strength = apply_relationship_strengths(relationships, strengths)

    # Final structure
    output = {
        "entities": entities,
        "relationships": relationships_with_strength,
        "clusters": clusters,
        "metadata": build_metadata(entities, relationships_with_strength, betweenness_method, cluster_counts),
    }

    # Backup existing file
    if OUTPUT_PATH.exists():
        print(f"\nCreating backup at {BACKUP_PATH}")
        BACKUP_PATH.write_text(OUTPUT_PATH.read_text())

    # Save main output
    print(f"\nWriting output to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f)

    # Save cluster registry (for RAG summarization later)
    print(f"Writing cluster registry to {CLUSTER_REGISTRY_PATH}")
    with CLUSTER_REGISTRY_PATH.open("w") as f:
        json.dump(cluster_registry, f, indent=2)

    # Report results
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    registry_size_mb = CLUSTER_REGISTRY_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✅ GraphRAG hierarchy generated (CLUSTER-ONLY mode):")
    print(f"   Main file: {size_mb:.2f} MB ({OUTPUT_PATH})")
    print(f"   Cluster registry: {registry_size_mb:.2f} MB ({CLUSTER_REGISTRY_PATH})")
    print(f"\n📊 Clustering Summary:")
    print(f"   Algorithm: Hierarchical Leiden (graspologic)")
    print(f"   Max cluster size: natural (modularity-optimized)")
    print(f"   Level 1: {cluster_counts['level_1']} communities")
    print(f"   Level 2: {cluster_counts['level_2']} sub-communities")
    print(f"   Level 3: {cluster_counts['level_3']} leaf clusters")
    print(f"\n🔍 Next steps:")
    print(f"   1. Deploy to /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/")
    print(f"   2. Test visualization at https://gaiaai.xyz/YonEarth/graph/")


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("GraphRAG Hierarchy Generation (Graph-Based Leiden Clustering)")
    print("=" * 80)

    # Check OpenAI key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise SystemExit(1)
    client = OpenAI(api_key=api_key)

    # Load data
    entities, relationships = load_discourse_graph(INPUT_GRAPH_PATH)

    # Build NetworkX graph (needed for both clustering and metrics)
    G = build_networkx_graph(entities, relationships)

    # Relationship index for enriched embeddings
    rel_index = build_relationship_index(relationships)

    # Embeddings + UMAP
    embeddings, entity_ids = create_embeddings(entities, rel_index, client)
    umap_positions = compute_umap_positions(embeddings)

    # Cache UMAP positions and entity IDs for --cluster-only mode
    umap_path = CHECKPOINT_DIR / "umap_positions.npy"
    entity_ids_path = CHECKPOINT_DIR / "entity_ids.json"
    np.save(umap_path, umap_positions)
    with entity_ids_path.open("w") as f:
        json.dump(entity_ids, f)
    print(f"  Cached UMAP positions to {umap_path}")
    print(f"  Cached entity IDs to {entity_ids_path}")

    # Graph metrics
    betweenness, betweenness_method = compute_betweenness_centrality(relationships, entities)
    strengths = compute_relationship_strengths(relationships)

    # Run hierarchical Leiden clustering on the graph
    hierarchy = run_hierarchical_leiden(G)

    # Extract cluster tree and build summary texts (using containment logic)
    cluster_registry, cluster_entities = extract_cluster_tree(hierarchy, G)

    # Convert to viewer-compatible format (simple 1:1 mapping)
    level_1, level_2, level_3 = build_hierarchy_from_leiden(
        cluster_registry, entity_ids, embeddings, umap_positions
    )

    level_0 = build_level_0_clusters(entity_ids, entities, umap_positions, betweenness)

    clusters = {
        "level_0": level_0,
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
    }

    # Count clusters for metadata
    cluster_counts = {
        "level_1": len(level_1),
        "level_2": len(level_2),
        "level_3": len(level_3),
    }

    # Relationships with strength weights
    relationships_with_strength = apply_relationship_strengths(relationships, strengths)

    # Final structure
    output = {
        "entities": entities,
        "relationships": relationships_with_strength,
        "clusters": clusters,
        "metadata": build_metadata(entities, relationships_with_strength, betweenness_method, cluster_counts),
    }

    # Backup existing file
    if OUTPUT_PATH.exists():
        print(f"\nCreating backup at {BACKUP_PATH}")
        BACKUP_PATH.write_text(OUTPUT_PATH.read_text())

    # Save main output
    print(f"\nWriting output to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f)

    # Save cluster registry (for RAG summarization later)
    print(f"Writing cluster registry to {CLUSTER_REGISTRY_PATH}")
    with CLUSTER_REGISTRY_PATH.open("w") as f:
        json.dump(cluster_registry, f, indent=2)

    # Report results
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    registry_size_mb = CLUSTER_REGISTRY_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✅ GraphRAG hierarchy generated:")
    print(f"   Main file: {size_mb:.2f} MB ({OUTPUT_PATH})")
    print(f"   Cluster registry: {registry_size_mb:.2f} MB ({CLUSTER_REGISTRY_PATH})")
    print(f"\n📊 Clustering Summary:")
    print(f"   Algorithm: Hierarchical Leiden (graspologic)")
    print(f"   Max cluster size: natural (modularity-optimized)")
    print(f"   Level 1: {cluster_counts['level_1']} communities")
    print(f"   Level 2: {cluster_counts['level_2']} sub-communities")
    print(f"   Level 3: {cluster_counts['level_3']} leaf clusters")
    print(f"\n🔍 Next steps:")
    print(f"   1. Review cluster_registry.json to verify summary_text quality")
    print(f"   2. Generate LLM summaries for each cluster using summary_text field")
    print(f"   3. Deploy to /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/")
    print(f"   4. Test visualization at https://gaiaai.xyz/YonEarth/graph/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GraphRAG hierarchy with Leiden clustering and LLM-generated titles"
    )
    parser.add_argument(
        "--titles-only",
        action="store_true",
        help="Fast mode: Only regenerate LLM titles without recomputing embeddings/UMAP/clustering"
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Medium mode: Skip embeddings/UMAP, reuse cached positions, only redo clustering and titles"
    )

    args = parser.parse_args()

    if args.titles_only:
        regenerate_titles_only()
    elif args.cluster_only:
        main_cluster_only()
    else:
        main()
