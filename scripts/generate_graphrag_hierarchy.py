#!/usr/bin/env python3
"""
Generate GraphRAG hierarchy and 3D layout from the discourse graph.

Pipeline:
1. Load discourse graph (entities + relationships)
2. Build graph-enriched OpenAI embeddings for all entities
3. Reduce embeddings to 3D with UMAP
4. Create hierarchical clusters (L1=300, L2=30, L3=7) with MiniBatchKMeans
5. Compute lightweight betweenness and relationship strengths
6. Export graphrag_hierarchy.json ready for the 3D viewer

Input:
  /home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_unified/discourse_graph_hybrid.json

Output:
  /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json
"""

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from openai import OpenAI
from sklearn.cluster import MiniBatchKMeans

try:
    import community as community_louvain  # python-louvain
except ImportError:
    community_louvain = None
    print("WARNING: python-louvain not installed. Install with: pip install python-louvain")

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Install with: pip install umap-learn")
    raise

# --------------------------------------------------------------------------------------
# Paths & parameters
# --------------------------------------------------------------------------------------

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
DISCOURSE_GRAPH_PATH = ROOT / "data/knowledge_graph_unified/discourse_graph_hybrid.json"
OUTPUT_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
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

# Clustering targets (used to tune Leiden/Louvain resolution)
LEVEL_1_CLUSTERS = 300  # fine clusters (entities)
LEVEL_2_CLUSTERS = 30   # medium clusters (groups of L1)
LEVEL_3_CLUSTERS = 7    # coarse clusters (groups of L2)

# Betweenness centrality
# Full centrality on 44k nodes is very expensive; sample for a faster approximation.
BETWEENNESS_SAMPLE = 512  # set to None to compute exact centrality (slow)


# --------------------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------------------

def load_discourse_graph(path: Path):
    """Load discourse graph JSON."""
    print(f"Loading discourse graph from {path}")
    with path.open() as f:
        data = json.load(f)

    entities = data.get("entities", {})
    relationships = data.get("relationships", [])
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
    print("\nComputing UMAP 3D positions...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )
    positions = reducer.fit_transform(embeddings)
    print("  ✓ UMAP complete")
    print(f"  Position ranges: x[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
          f"z[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
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
# Clustering helpers
# --------------------------------------------------------------------------------------

def cluster_level_1(entity_ids: List[str], embeddings: np.ndarray, umap_positions: np.ndarray):
    """Cluster entities into fine clusters (level 1)."""
    print(f"\nClustering level 1 (fine clusters: {LEVEL_1_CLUSTERS})...")
    model = MiniBatchKMeans(
        n_clusters=LEVEL_1_CLUSTERS,
        random_state=42,
        batch_size=2048,
        n_init="auto",
    )
    labels = model.fit_predict(embeddings)

    clusters = {}
    entity_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}
    for idx, label in enumerate(labels):
        cid = f"l1_{label:03d}"
        clusters.setdefault(cid, {"id": cid, "type": "fine_cluster", "children": [], "entities": []})
        eid = entity_ids[idx]
        clusters[cid]["children"].append(eid)
        clusters[cid]["entities"].append(eid)

    # Compute centers/positions
    for cid, cluster in clusters.items():
        indices = [entity_to_idx[eid] for eid in cluster["entities"]]
        cluster_embeddings = embeddings[indices]
        cluster_positions = umap_positions[indices]
        cluster["center"] = cluster_embeddings.mean(axis=0).tolist()
        position_mean = cluster_positions.mean(axis=0)
        cluster["position"] = position_mean.tolist()
        cluster["umap_position"] = position_mean.tolist()
        cluster["size"] = len(indices)

    print(f"  ✓ Created {len(clusters)} level_1 clusters")
    return clusters, labels


def cluster_parents(child_clusters: Dict[str, dict],
                    child_centers: np.ndarray,
                    child_positions: np.ndarray,
                    n_clusters: int,
                    level_prefix: str,
                    cluster_type: str):
    """Cluster higher levels (L2, L3) from child cluster centers."""
    print(f"Clustering {level_prefix} ({cluster_type}: {n_clusters})...")
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=512,
        n_init="auto",
    )
    labels = model.fit_predict(child_centers)

    clusters = {}
    child_ids = list(child_clusters.keys())
    for idx, label in enumerate(labels):
        cid = f"{level_prefix}_{label:02d}" if level_prefix == "l2" else f"{level_prefix}_{label}"
        clusters.setdefault(cid, {"id": cid, "type": cluster_type, "children": [], "entities": []})
        child_id = child_ids[idx]
        clusters[cid]["children"].append(child_id)
        clusters[cid]["entities"].extend(child_clusters[child_id].get("entities", []))

    for cid, cluster in clusters.items():
        member_indices = [i for i, lbl in enumerate(labels) if lbl == int(cid.split("_")[1])]
        center_mean = child_centers[member_indices].mean(axis=0)
        pos_mean = child_positions[member_indices].mean(axis=0)
        cluster["center"] = center_mean.tolist()
        cluster["position"] = pos_mean.tolist()
        cluster["umap_position"] = pos_mean.tolist()
        cluster["size"] = len(cluster["entities"])

    print(f"  ✓ Created {len(clusters)} {level_prefix} clusters")
    return clusters, labels


def build_hierarchy(entity_ids: List[str], embeddings: np.ndarray, umap_positions: np.ndarray):
    """Build hierarchical clusters (L1, L2, L3)."""
    # Level 1 on entities
    level_1_clusters, l1_labels = cluster_level_1(entity_ids, embeddings, umap_positions)

    # Prepare arrays for parent clustering
    l1_ids = list(level_1_clusters.keys())
    l1_centers = np.array([level_1_clusters[cid]["center"] for cid in l1_ids])
    l1_positions = np.array([level_1_clusters[cid]["position"] for cid in l1_ids])

    # Level 2 on L1 centers
    level_2_clusters, l2_labels = cluster_parents(
        level_1_clusters, l1_centers, l1_positions,
        LEVEL_2_CLUSTERS, "l2", "medium_cluster"
    )

    # Prepare for level 3
    l2_ids = list(level_2_clusters.keys())
    l2_centers = np.array([level_2_clusters[cid]["center"] for cid in l2_ids])
    l2_positions = np.array([level_2_clusters[cid]["position"] for cid in l2_ids])

    level_3_clusters, _ = cluster_parents(
        level_2_clusters, l2_centers, l2_positions,
        LEVEL_3_CLUSTERS, "l3", "coarse_cluster"
    )

    return level_1_clusters, level_2_clusters, level_3_clusters


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
        level_0[eid] = {
            "id": eid,
            "type": "entity",
            "entity": entities[eid],
            "embedding_idx": idx,
            "position": umap_positions[idx].tolist(),
            "umap_position": umap_positions[idx].tolist(),
            "betweenness": betweenness.get(eid, 0.0),
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
                   betweenness_method: str):
    """Assemble metadata section."""
    return {
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "levels": 4,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "multi_membership": False,
        "source_graph": str(DISCOURSE_GRAPH_PATH),
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
            "level_1_clusters": LEVEL_1_CLUSTERS,
            "level_2_clusters": LEVEL_2_CLUSTERS,
            "level_3_clusters": LEVEL_3_CLUSTERS,
            "algorithm": "MiniBatchKMeans",
        },
    }


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("GraphRAG Hierarchy Generation (Discourse Graph)")
    print("=" * 80)

    # Check OpenAI key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise SystemExit(1)
    client = OpenAI(api_key=api_key)

    # Load data
    entities, relationships = load_discourse_graph(DISCOURSE_GRAPH_PATH)

    # Relationship index for enriched embeddings
    rel_index = build_relationship_index(relationships)

    # Embeddings + UMAP
    embeddings, entity_ids = create_embeddings(entities, rel_index, client)
    umap_positions = compute_umap_positions(embeddings)

    # Graph metrics
    betweenness, betweenness_method = compute_betweenness_centrality(relationships, entities)
    strengths = compute_relationship_strengths(relationships)

    # Clusters
    level_1, level_2, level_3 = build_hierarchy(entity_ids, embeddings, umap_positions)
    level_0 = build_level_0_clusters(entity_ids, entities, umap_positions, betweenness)

    clusters = {
        "level_0": level_0,
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
    }

    # Relationships with strength weights
    relationships_with_strength = apply_relationship_strengths(relationships, strengths)

    # Final structure
    output = {
        "entities": entities,
        "relationships": relationships_with_strength,
        "clusters": clusters,
        "metadata": build_metadata(entities, relationships_with_strength, betweenness_method),
    }

    # Backup existing file
    if OUTPUT_PATH.exists():
        print(f"\nCreating backup at {BACKUP_PATH}")
        BACKUP_PATH.write_text(OUTPUT_PATH.read_text())

    # Save output
    print(f"Writing output to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✅ GraphRAG hierarchy generated. File size: {size_mb:.2f} MB")
    print("Next steps:")
    print("  - Deploy to /opt/yonearth-chatbot/web/data/graphrag_hierarchy/")
    print("  - Test visualization at https://gaiaai.xyz/YonEarth/graph/")


if __name__ == "__main__":
    main()
