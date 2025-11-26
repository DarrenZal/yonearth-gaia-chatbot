#!/usr/bin/env python3
"""
Proper GraphRAG implementation using Leiden community detection,
UMAP positioning, and LLM-generated hierarchical summaries.

This follows Microsoft's GraphRAG approach with improvements:
- Leiden algorithm for natural community detection
- UMAP for meaningful 3D positioning
- Pre-indexed relationships for fast lookup
- Checkpointing to resume after failures
- Cost estimation before LLM calls
- Progress tracking with tqdm
"""

import json
import os
import sys
import numpy as np
import networkx as nx
from openai import OpenAI
import time
import psutil
import threading
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from pathlib import Path
from functools import wraps
from datetime import datetime

try:
    import leidenalg
    import igraph as ig
except ImportError:
    print("ERROR: leidenalg and igraph required. Install with:")
    print("  pip install leidenalg python-igraph")
    raise

try:
    import umap
except ImportError:
    print("ERROR: umap-learn required. Install with: pip install umap-learn")
    raise

try:
    from tqdm import tqdm
except ImportError:
    print("WARNING: tqdm not installed. Install with: pip install tqdm")
    tqdm = None

# Configuration
ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
DISCOURSE_GRAPH_PATH = ROOT / "data/knowledge_graph_unified/discourse_graph_hybrid.json"
OUTPUT_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy.json"
BACKUP_PATH = ROOT / "data/graphrag_hierarchy/graphrag_hierarchy_backup_leiden.json"
CHECKPOINT_DIR = ROOT / "data/graphrag_hierarchy/checkpoints"

# Leiden parameters (tunable via env vars)
LEIDEN_RESOLUTION_L0 = float(os.getenv("LEIDEN_RES_L0", "0.8"))  # Root level
LEIDEN_RESOLUTION_L1 = float(os.getenv("LEIDEN_RES_L1", "1.2"))  # Finer partitions
LEIDEN_RESOLUTION_L2 = float(os.getenv("LEIDEN_RES_L2", "1.5"))  # Finest partitions
MAX_RECURSION_DEPTH = 3
MIN_COMMUNITY_SIZE = 15  # Increased from 10 to reduce noise

# UMAP parameters
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 12
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# Embedding parameters
EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
EMBED_BATCH_SIZE = 100

# Summary parameters
SUMMARY_SAMPLE_LIMIT = 20  # Max entities to sample for summary
RELATIONSHIP_SAMPLE_LIMIT = 15  # Max relationships to include
LLM_MODEL = "gpt-4o-mini"
LLM_RATE_DELAY = 0.05  # seconds between requests
SUMMARY_CHECKPOINT_INTERVAL = 50  # Save summaries every N communities
MAX_LLM_RETRIES = 5  # Max retries for LLM calls before skipping community

# Production safeguards
LOCKFILE_PATH = ROOT / "data/graphrag_hierarchy/graphrag.lock"
HEARTBEAT_INTERVAL = 300  # Progress log every 5 minutes
MEMORY_CHECK_INTERVAL = 100  # Log memory every N communities
MEMORY_LIMIT_GB = 20  # Bail if memory exceeds this

# Cost estimation (gpt-4o-mini pricing)
COST_PER_1K_INPUT = 0.00015
COST_PER_1K_OUTPUT = 0.0006

# Global state for heartbeat thread
heartbeat_state = {
    'running': False,
    'current_level': None,
    'current_community': None,
    'total_processed': 0,
    'start_time': None
}


def acquire_lockfile():
    """Create lockfile to prevent concurrent runs."""
    if LOCKFILE_PATH.exists():
        # Check if process is still running
        try:
            with LOCKFILE_PATH.open() as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                print(f"❌ ERROR: Another instance is running (PID {pid})")
                print(f"   Lockfile: {LOCKFILE_PATH}")
                print(f"   If this is stale, remove: rm {LOCKFILE_PATH}")
                sys.exit(1)
            else:
                print(f"⚠️  Removing stale lockfile (PID {pid} no longer exists)")
                LOCKFILE_PATH.unlink()
        except Exception as e:
            print(f"⚠️  Error checking lockfile: {e}")
            print(f"   Proceeding anyway...")

    # Create lockfile
    LOCKFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCKFILE_PATH.open('w') as f:
        f.write(str(os.getpid()))
    print(f"🔒 Acquired lockfile: {LOCKFILE_PATH} (PID {os.getpid()})")


def release_lockfile():
    """Remove lockfile on exit."""
    if LOCKFILE_PATH.exists():
        LOCKFILE_PATH.unlink()
        print(f"🔓 Released lockfile")


def validate_checkpoints():
    """Verify required checkpoints exist before starting."""
    print("\n" + "="*80)
    print("VALIDATING CHECKPOINTS")
    print("="*80)

    embeddings_path = CHECKPOINT_DIR / "embeddings.npy"
    leiden_path = CHECKPOINT_DIR / "leiden_hierarchies.json"

    issues = []
    if not embeddings_path.exists():
        issues.append(f"Missing embeddings checkpoint: {embeddings_path}")
    else:
        size_mb = embeddings_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Embeddings checkpoint: {size_mb:.1f} MB")

    if not leiden_path.exists():
        issues.append(f"Missing Leiden checkpoint: {leiden_path}")
    else:
        size_kb = leiden_path.stat().st_size / 1024
        print(f"  ✓ Leiden checkpoint: {size_kb:.1f} KB")

    if issues:
        print("\n❌ CHECKPOINT VALIDATION FAILED:")
        for issue in issues:
            print(f"   {issue}")
        print("\nRun the full pipeline first to generate checkpoints.")
        sys.exit(1)

    print("✓ All required checkpoints present\n")


def heartbeat_logger():
    """Background thread that logs progress every N seconds."""
    while heartbeat_state['running']:
        time.sleep(HEARTBEAT_INTERVAL)
        if not heartbeat_state['running']:
            break

        elapsed = time.time() - heartbeat_state['start_time']
        level = heartbeat_state['current_level']
        comm_id = heartbeat_state['current_community']
        total = heartbeat_state['total_processed']

        mem = psutil.Process().memory_info().rss / (1024 ** 3)  # GB

        print(f"\n⏱️  HEARTBEAT [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"   Elapsed: {elapsed/3600:.1f}h | Level: {level} | Community: {comm_id}")
        print(f"   Total processed: {total} | Memory: {mem:.2f} GB\n")


def start_heartbeat():
    """Start heartbeat logging thread."""
    heartbeat_state['running'] = True
    heartbeat_state['start_time'] = time.time()
    thread = threading.Thread(target=heartbeat_logger, daemon=True)
    thread.start()
    print("💓 Started heartbeat logger (every 5 minutes)")


def stop_heartbeat():
    """Stop heartbeat logging thread."""
    heartbeat_state['running'] = False


def check_memory():
    """Check memory usage and bail if exceeds limit."""
    mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    if mem_gb > MEMORY_LIMIT_GB:
        print(f"\n❌ MEMORY LIMIT EXCEEDED: {mem_gb:.2f} GB > {MEMORY_LIMIT_GB} GB")
        print("   Bailing to prevent OOM. Restart to resume from checkpoint.")
        stop_heartbeat()
        release_lockfile()
        sys.exit(1)
    return mem_gb


def retry_with_backoff(max_retries=MAX_LLM_RETRIES, initial_delay=1.0):
    """
    Decorator for retrying OpenAI API calls with exponential backoff.

    Handles rate limits (429) and server errors (500, 503).
    Enhanced with configurable max retries and detailed logging.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e)

                    # Check if retryable error
                    is_rate_limit = '429' in error_str or 'rate limit' in error_str.lower()
                    is_server_error = any(code in error_str for code in ['500', '503'])

                    if not (is_rate_limit or is_server_error):
                        # Non-retryable error
                        print(f"  ❌ Non-retryable error: {e}")
                        raise

                    if attempt == max_retries:
                        # Max retries exhausted
                        print(f"  ❌ Max retries ({max_retries}) exhausted: {e}")
                        raise

                    # Exponential backoff
                    print(f"  ⚠️  API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"  ⏳  Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

            # Should never reach here, but fail safe
            raise last_error
        return wrapper
    return decorator


def load_knowledge_graph(kg_path: Path) -> Tuple[Dict, List]:
    """Load the unified knowledge graph."""
    print(f"Loading knowledge graph from {kg_path}...")
    with open(kg_path) as f:
        data = json.load(f)

    entities = data.get('entities', {})
    relationships = data.get('relationships', [])

    print(f"  ✓ Loaded {len(entities):,} entities and {len(relationships):,} relationships")
    return entities, relationships


def build_relationship_index(relationships: List[dict]) -> Dict[str, List[dict]]:
    """Pre-index relationships for O(1) lookup by entity."""
    print("\nIndexing relationships...")
    rel_index = defaultdict(list)

    for rel in relationships:
        src = rel.get('source') or rel.get('source_entity')
        tgt = rel.get('target') or rel.get('target_entity')

        if src and tgt and src != tgt:
            rel_data = {
                'source': src,
                'target': tgt,
                'predicate': rel.get('predicate', 'related_to'),
                'weight': rel.get('weight', 1.0)
            }
            rel_index[src].append(rel_data)
            rel_index[tgt].append(rel_data)

    print(f"  ✓ Indexed {len(relationships):,} relationships for {len(rel_index):,} entities")
    return rel_index


def build_networkx_graph(entities: Dict, relationships: List) -> nx.Graph:
    """Build a NetworkX graph from entities and relationships."""
    print("\nBuilding NetworkX graph...")
    G = nx.Graph()

    # Add nodes
    for entity_id, entity_data in entities.items():
        G.add_node(entity_id, **entity_data)

    # Add edges with weights (normalized counts)
    edge_counts = defaultdict(int)
    for rel in relationships:
        source = rel.get('source') or rel.get('source_entity')
        target = rel.get('target') or rel.get('target_entity')
        if source in entities and target in entities and source != target:
            edge_key = tuple(sorted([source, target]))
            edge_counts[edge_key] += 1

    # Add weighted edges
    max_count = max(edge_counts.values()) if edge_counts else 1
    for (source, target), count in edge_counts.items():
        weight = count / max_count
        G.add_edge(source, target, weight=weight)

    print(f"  ✓ Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def build_embedding_text(entity_id: str, entity_data: dict, rel_index: Dict[str, List[dict]]) -> str:
    """Build text for embedding from entity and its relationships."""
    name = entity_data.get('name') or entity_id
    ent_type = entity_data.get('type', 'UNKNOWN')
    description = entity_data.get('description', '').strip()[:300]

    # Fallback for empty descriptions
    if not description:
        description = f"A {ent_type.lower()} entity in the knowledge graph"

    # Get top relationships
    rels = rel_index.get(entity_id, [])
    top_rels = sorted(rels, key=lambda r: r.get('weight', 1.0), reverse=True)[:2]
    rel_text = ' | '.join([f"{r['predicate']} {r['target']}" for r in top_rels])

    parts = [name, ent_type, description]
    if rel_text:
        parts.append(rel_text)

    return ' | '.join(parts)


def create_embeddings(entities: Dict, rel_index: Dict, client: OpenAI) -> Tuple[np.ndarray, List[str]]:
    """Generate embeddings for entities with checkpointing."""
    entity_ids = list(entities.keys())
    n_entities = len(entity_ids)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = CHECKPOINT_DIR / "embeddings.npy"
    progress_path = CHECKPOINT_DIR / "embeddings_progress.json"

    # Check for existing checkpoint
    if embeddings_path.exists():
        try:
            print("\n✓ Loading embeddings from checkpoint...")
            embeddings = np.load(embeddings_path)
            if embeddings.shape == (n_entities, EMBED_DIM):
                print(f"  Loaded {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]}")
                return embeddings, entity_ids
            else:
                print(f"  ⚠️  Checkpoint shape mismatch: {embeddings.shape} != ({n_entities}, {EMBED_DIM})")
                print(f"  Regenerating embeddings...")
        except Exception as e:
            print(f"  ⚠️  Failed to load checkpoint: {e}")
            print(f"  Regenerating embeddings...")

    print(f"\nGenerating embeddings with {EMBEDDING_MODEL}...")
    embeddings = np.zeros((n_entities, EMBED_DIM), dtype='float32')

    start_idx = 0
    if progress_path.exists():
        try:
            start_idx = json.loads(progress_path.read_text()).get("completed", 0)
            if start_idx > 0:
                print(f"  Resuming from entity {start_idx:,}")
        except:
            start_idx = 0

    iterator = range(start_idx, n_entities, EMBED_BATCH_SIZE)
    if tqdm:
        iterator = tqdm(iterator, desc="  Embedding batches", initial=start_idx//EMBED_BATCH_SIZE,
                       total=(n_entities + EMBED_BATCH_SIZE - 1)//EMBED_BATCH_SIZE)

    for i in iterator:
        batch_ids = entity_ids[i:i + EMBED_BATCH_SIZE]
        texts = [build_embedding_text(eid, entities[eid], rel_index) for eid in batch_ids]

        # Retry wrapper for embeddings API call
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def get_embeddings():
            return client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

        try:
            response = get_embeddings()
            for j, emb in enumerate(response.data):
                embeddings[i + j] = emb.embedding
        except Exception as e:
            print(f"\n  ⚠️  Batch {i} failed after retries: {e}")
            embeddings[i:i + len(batch_ids)] = np.random.randn(len(batch_ids), EMBED_DIM) * 0.01

        progress_path.write_text(json.dumps({"completed": i + len(batch_ids)}))
        time.sleep(0.05)

    # Save checkpoint
    np.save(embeddings_path, embeddings)
    progress_path.unlink(missing_ok=True)
    print("  ✓ Embeddings complete")

    return embeddings, entity_ids


def compute_umap_positions(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 3D using UMAP."""
    print("\nComputing UMAP 3D positions...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
        verbose=False
    )
    positions = reducer.fit_transform(embeddings)
    print(f"  ✓ UMAP complete")
    print(f"    Position ranges: x[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
          f"z[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
    return positions


def compute_betweenness(G: nx.Graph) -> Dict[str, float]:
    """
    Compute betweenness centrality scores for all nodes.

    Uses approximate algorithm for speed on large graphs.
    Returns normalized scores [0, 1] for 3D viewer color mapping.
    """
    print("\nComputing betweenness centrality scores...")

    # Use approximate betweenness for graphs >5000 nodes
    if G.number_of_nodes() > 5000:
        print(f"  Using approximate algorithm (k=1000 samples) for {G.number_of_nodes():,} nodes")
        betweenness = nx.betweenness_centrality(G, k=1000, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)

    # Ensure all scores are in [0, 1]
    max_score = max(betweenness.values()) if betweenness else 1.0
    if max_score > 0:
        betweenness = {node: score / max_score for node, score in betweenness.items()}

    # Report stats
    scores = list(betweenness.values())
    print(f"  ✓ Betweenness computed")
    print(f"    Min: {min(scores):.6f}, Max: {max(scores):.6f}, Mean: {np.mean(scores):.6f}")

    return betweenness


def compute_relationship_strengths(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """
    Compute relationship strength for each entity's connections.

    Returns dict: {entity_id: {neighbor_id: normalized_weight, ...}}
    Uses edge weights from graph (normalized co-occurrence counts).
    """
    print("\nComputing relationship strengths...")

    rel_strengths = {}
    for node in G.nodes():
        strengths = {}
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            weight = edge_data.get('weight', 1.0) if edge_data else 1.0
            strengths[neighbor] = float(weight)
        rel_strengths[node] = strengths

    print(f"  ✓ Relationship strengths computed for {len(rel_strengths):,} entities")

    return rel_strengths


def hierarchical_leiden_clustering(
    G: nx.Graph,
    entity_id_to_idx: Dict[str, int],
    resolution_levels: List[float]
) -> Dict[int, List[Dict]]:
    """
    Apply hierarchical Leiden clustering with proper subgraph handling.

    Fixed version that correctly handles igraph's local indexing.
    """
    print("\nApplying hierarchical Leiden community detection...")
    print(f"  Resolution schedule: {resolution_levels}")

    # Convert to igraph
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    edges = []
    weights = []
    for source, target, data in G.edges(data=True):
        edges.append((node_to_idx[source], node_to_idx[target]))
        weights.append(data.get('weight', 1.0))

    ig_graph = ig.Graph(edges=edges, directed=False)
    ig_graph.es['weight'] = weights

    # Store all hierarchies
    hierarchies = {}

    def detect_communities_recursive(
        graph: ig.Graph,
        node_mapping: List[str],  # Maps local indices to entity IDs
        level: int,
        parent_id: str = None,
        resolution: float = 1.0
    ):
        """Recursively detect communities with proper local indexing."""

        if level >= len(resolution_levels):
            return

        # Apply Leiden with resolution parameter
        # Use RBConfigurationVertexPartition which supports resolution_parameter
        weights_attr = graph.es['weight'] if 'weight' in graph.es.attributes() else None
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights_attr,
            resolution_parameter=resolution
        )

        # Initialize level
        if level not in hierarchies:
            hierarchies[level] = []

        # Process each community
        for comm_idx, community in enumerate(partition):
            if len(community) == 0:
                continue

            # Map local indices back to entity IDs
            entity_ids = [node_mapping[local_idx] for local_idx in community]

            comm_data = {
                'id': f"l{level}_{len(hierarchies[level])}",
                'level': level,
                'nodes': entity_ids,
                'parent': parent_id,
                'modularity': partition.quality()
            }
            hierarchies[level].append(comm_data)

            # Recurse if community is large enough
            if len(entity_ids) > MIN_COMMUNITY_SIZE and level + 1 < len(resolution_levels):
                # Create subgraph using local indices (community already contains local indices)
                try:
                    subgraph = graph.subgraph(community)

                    # Build mapping for subgraph (subgraph has its own 0-N indexing)
                    subgraph_mapping = entity_ids.copy()

                    # Recurse with next resolution level
                    detect_communities_recursive(
                        subgraph,
                        subgraph_mapping,
                        level + 1,
                        comm_data['id'],
                        resolution_levels[level + 1] if level + 1 < len(resolution_levels) else resolution
                    )
                except Exception as e:
                    print(f"  ⚠️  Subgraph creation failed for {comm_data['id']}: {e}")

    # Start recursion
    detect_communities_recursive(
        ig_graph,
        node_list,
        level=0,
        resolution=resolution_levels[0]
    )

    # Print statistics
    print("\n  Community Statistics:")
    for level in sorted(hierarchies.keys()):
        comms = hierarchies[level]
        total_nodes = sum(len(c['nodes']) for c in comms)
        avg_size = total_nodes / len(comms) if comms else 0
        min_size = min(len(c['nodes']) for c in comms) if comms else 0
        max_size = max(len(c['nodes']) for c in comms) if comms else 0
        print(f"    Level {level}: {len(comms):3d} communities | "
              f"size: {avg_size:6.1f} avg, [{min_size:4d}-{max_size:5d}] range")

    return hierarchies


def estimate_llm_cost(hierarchies: Dict) -> Tuple[int, float]:
    """Estimate token count and cost for LLM summarization."""
    total_communities = sum(len(comms) for comms in hierarchies.values())
    avg_input_tokens = 800  # Estimated prompt + context
    avg_output_tokens = 150  # Estimated summary

    total_input_tokens = total_communities * avg_input_tokens
    total_output_tokens = total_communities * avg_output_tokens

    cost = (total_input_tokens / 1000 * COST_PER_1K_INPUT +
            total_output_tokens / 1000 * COST_PER_1K_OUTPUT)

    return total_communities, cost


def generate_community_summary(
    community: Dict,
    entities: Dict,
    rel_index: Dict[str, List[dict]],
    client: OpenAI
) -> Dict[str, str]:
    """Generate LLM summary for a community."""

    # Sample entities
    sampled_nodes = community['nodes'][:SUMMARY_SAMPLE_LIMIT]
    community_entities = []

    for node_id in sampled_nodes:
        if node_id in entities:
            entity = entities[node_id]
            desc = entity.get('description', '')[:100]
            if not desc:
                desc = f"A {entity.get('type', 'entity')} in the sustainability knowledge graph"

            community_entities.append({
                'name': entity.get('name', node_id),
                'type': entity.get('type', 'UNKNOWN'),
                'description': desc
            })

    # Get relationships within community
    node_set = set(community['nodes'])
    community_rels = []

    for node_id in sampled_nodes[:10]:  # Sample nodes for relationships
        for rel in rel_index.get(node_id, []):
            if rel['source'] in node_set and rel['target'] in node_set:
                community_rels.append(rel)
                if len(community_rels) >= RELATIONSHIP_SAMPLE_LIMIT:
                    break
        if len(community_rels) >= RELATIONSHIP_SAMPLE_LIMIT:
            break

    # Build prompt
    entity_text = '\n'.join([
        f"- {e['name']} ({e['type']}): {e['description']}"
        for e in community_entities[:15]
    ])

    rel_text = '\n'.join([
        f"- {entities.get(r['source'], {}).get('name', r['source'])} "
        f"{r['predicate']} "
        f"{entities.get(r['target'], {}).get('name', r['target'])}"
        for r in community_rels[:10]
    ])

    prompt = f"""Analyze this community from a sustainability/regenerative practices knowledge graph.

Level: {community['level']} | Size: {len(community['nodes'])} entities

Key Entities:
{entity_text}

Key Relationships:
{rel_text}

Generate:
1. Title (2-4 words): Capture the main theme
2. Summary (2-3 sentences): Describe entity types, relationships, and overarching theme

Format as JSON:
{{"title": "...", "summary": "..."}}
"""

    # Retry wrapper for LLM API call
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def get_llm_summary():
        return client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing knowledge graphs about sustainability and ecology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"}
        )

    try:
        response = get_llm_summary()
        result = json.loads(response.choices[0].message.content)
        return {
            'title': result.get('title', f"Community {community['id']}"),
            'summary': result.get('summary', f"A collection of {len(community['nodes'])} entities.")
        }

    except Exception as e:
        return {
            'title': f"Community {community['id']}",
            'summary': f"A collection of {len(community['nodes'])} entities. (Error: {str(e)[:50]})"
        }


def build_graphrag_hierarchy(
    entities: Dict,
    relationships: List,
    rel_index: Dict,
    embeddings: np.ndarray,
    entity_ids: List[str],
    client: OpenAI
) -> Tuple[Dict, np.ndarray, Dict[str, float], Dict[str, Dict]]:
    """Build complete GraphRAG hierarchy with Leiden + UMAP + LLM."""

    # Build graph
    G = build_networkx_graph(entities, relationships)
    entity_id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}

    # Compute UMAP positions
    umap_positions = compute_umap_positions(embeddings)

    # Compute betweenness centrality and relationship strengths
    betweenness_scores = compute_betweenness(G)
    relationship_strengths = compute_relationship_strengths(G)

    # Check for cached Leiden results
    leiden_checkpoint = CHECKPOINT_DIR / "leiden_hierarchies.json"

    if leiden_checkpoint.exists():
        print("\n✓ Loading Leiden communities from checkpoint...")
        with leiden_checkpoint.open() as f:
            hierarchies = json.load(f)
        # Convert lists back to dicts
        hierarchies = {int(k): v for k, v in hierarchies.items()}
    else:
        # Apply hierarchical Leiden
        resolution_levels = [LEIDEN_RESOLUTION_L0, LEIDEN_RESOLUTION_L1, LEIDEN_RESOLUTION_L2]
        hierarchies = hierarchical_leiden_clustering(G, entity_id_to_idx, resolution_levels)

        # Save checkpoint
        print("\n  Saving Leiden checkpoint...")
        with leiden_checkpoint.open('w') as f:
            json.dump(hierarchies, f, indent=2)

    # Estimate LLM cost
    num_communities, estimated_cost = estimate_llm_cost(hierarchies)
    print(f"\n💰 LLM Summarization Estimate:")
    print(f"    Communities: {num_communities}")
    print(f"    Estimated cost: ${estimated_cost:.4f}")

    # Allow auto-confirmation via environment variable for background runs
    auto_confirm = os.getenv("AUTO_CONFIRM_LLM", "").lower() in ['y', 'yes', '1', 'true']

    if auto_confirm:
        response = 'y'
        print("\n  AUTO_CONFIRM_LLM=true, proceeding with LLM summarization...")
    else:
        try:
            response = input("\n  Proceed with LLM summarization? [y/N]: ").strip().lower()
        except EOFError:
            print("\n  Running in background (no stdin), skipping LLM summarization.")
            print("  Set AUTO_CONFIRM_LLM=y to enable auto-confirmation.")
            response = 'n'

    if response != 'y':
        print("  Skipping LLM summarization. Using placeholder titles.")
        for level, comms in hierarchies.items():
            for comm in comms:
                comm['title'] = f"Cluster {comm['id']}"
                comm['summary'] = f"A community of {len(comm['nodes'])} entities."
        return hierarchies, umap_positions, betweenness_scores, relationship_strengths

    # Generate summaries
    print("\n" + "="*80)
    print("GENERATING COMMUNITY SUMMARIES")
    print("="*80)

    # Load summary checkpoint if exists
    summary_checkpoint = CHECKPOINT_DIR / "summaries_progress.json"
    if summary_checkpoint.exists():
        print("\n✓ Loading summary checkpoint...")
        with summary_checkpoint.open() as f:
            saved_summaries = json.load(f)

        # Restore summaries to communities
        restored_count = 0
        for level_key, level_summaries in saved_summaries.items():
            level = int(level_key)
            if level in hierarchies:
                for comm_id, summary_data in level_summaries.items():
                    for community in hierarchies[level]:
                        if community['id'] == comm_id:
                            community['title'] = summary_data['title']
                            community['summary'] = summary_data['summary']
                            restored_count += 1
                            break
        print(f"  Restored {restored_count} summaries from checkpoint")

    # Generate summaries for each level
    all_summaries = {}  # Track all summaries for checkpointing
    total_processed = 0
    failed_communities = []  # Track failed communities for retry

    # Start heartbeat logger
    start_heartbeat()

    try:
        for level in sorted(hierarchies.keys(), reverse=True):
            communities = hierarchies[level]
            print(f"\nLevel {level}: {len(communities)} communities")

            # Update heartbeat state
            heartbeat_state['current_level'] = level

            # Count how many already have summaries
            completed = sum(1 for c in communities if 'title' in c and 'summary' in c)
            if completed > 0:
                print(f"  {completed} already completed, continuing from checkpoint...")

            iterator = communities
            if tqdm:
                iterator = tqdm(communities, desc=f"  Summarizing L{level}", initial=completed, total=len(communities))

            level_summaries = {}

            for i, community in enumerate(iterator):
                comm_id = community['id']
                heartbeat_state['current_community'] = comm_id

                # Skip if already summarized
                if 'title' in community and 'summary' in community:
                    level_summaries[comm_id] = {
                        'title': community['title'],
                        'summary': community['summary']
                    }
                    continue

                # Try to generate summary with retry logic
                try:
                    summary_data = generate_community_summary(
                        community,
                        entities,
                        rel_index,
                        client
                    )

                    community['title'] = summary_data['title']
                    community['summary'] = summary_data['summary']
                    level_summaries[comm_id] = summary_data
                    total_processed += 1
                    heartbeat_state['total_processed'] = total_processed

                except Exception as e:
                    # Log failed community for later retry
                    failed_communities.append({
                        'level': level,
                        'id': comm_id,
                        'error': str(e)
                    })
                    print(f"\n  ⚠️  Failed to summarize {comm_id}: {e}")
                    print(f"     Will use fallback summary")

                    # Use fallback summary
                    community['title'] = f"Cluster {comm_id}"
                    community['summary'] = f"A community of {len(community['nodes'])} entities."
                    level_summaries[comm_id] = {
                        'title': community['title'],
                        'summary': community['summary']
                    }

                time.sleep(LLM_RATE_DELAY)

                # Memory check every N communities
                if total_processed % MEMORY_CHECK_INTERVAL == 0:
                    mem_gb = check_memory()
                    if total_processed % (MEMORY_CHECK_INTERVAL * 5) == 0:
                        print(f"  💾 Memory: {mem_gb:.2f} GB")

                # Checkpoint every N communities with fsync
                if total_processed % SUMMARY_CHECKPOINT_INTERVAL == 0:
                    all_summaries[str(level)] = level_summaries
                    with summary_checkpoint.open('w') as f:
                        json.dump(all_summaries, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    print(f"  💾 Checkpoint saved ({total_processed} communities)")

            # Save level summaries
            all_summaries[str(level)] = level_summaries

        # Final checkpoint save with fsync
        print("\n  Saving final summary checkpoint...")
        with summary_checkpoint.open('w') as f:
            json.dump(all_summaries, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Log failed communities
        if failed_communities:
            failed_log = CHECKPOINT_DIR / "failed_communities.json"
            with failed_log.open('w') as f:
                json.dump(failed_communities, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            print(f"\n⚠️  {len(failed_communities)} communities failed (logged to {failed_log})")
            print("   These used fallback summaries. You can retry them later.")

    finally:
        stop_heartbeat()

    # Print samples
    print("\n✓ Sample Community Summaries:")
    for level in sorted(hierarchies.keys())[:2]:
        for comm in hierarchies[level][:2]:
            print(f"    {comm['id']}: {comm.get('title', 'Untitled')}")

    return hierarchies, umap_positions, betweenness_scores, relationship_strengths


def convert_to_graphrag_format(
    hierarchies: Dict,
    entities: Dict,
    relationships: List,
    entity_ids: List[str],
    umap_positions: np.ndarray,
    betweenness_scores: Dict[str, float],
    relationship_strengths: Dict[str, Dict]
) -> Dict:
    """Convert to GraphRAG visualization format with UMAP positions."""
    print("\n" + "="*80)
    print("CONVERTING TO GRAPHRAG FORMAT")
    print("="*80)

    clusters = {
        'level_0': {},
        'level_1': {},
        'level_2': {},
        'level_3': {}
    }

    # Level 0: Individual entities with UMAP positions
    entity_id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}

    for entity_id, entity_data in entities.items():
        idx = entity_id_to_idx.get(entity_id, 0)
        position = umap_positions[idx].tolist() if idx < len(umap_positions) else [0, 0, 0]

        # Get computed betweenness and relationship strengths
        betweenness = betweenness_scores.get(entity_id, 0.0)
        rel_strengths = relationship_strengths.get(entity_id, {})

        clusters['level_0'][entity_id] = {
            'id': entity_id,
            'type': 'entity',
            'entity': entity_data,
            'position': position,
            'umap_position': position,
            'betweenness': betweenness,
            'relationship_strengths': rel_strengths
        }

    # Add Leiden communities (shift levels by 1)
    for leiden_level, communities in hierarchies.items():
        viz_level = leiden_level + 1
        level_key = f'level_{viz_level}'

        if level_key not in clusters:
            continue

        for comm in communities:
            # Determine children
            if leiden_level == 0:
                children = comm['nodes'][:100]
            else:
                lower_level = hierarchies.get(leiden_level - 1, [])
                children = [lc['id'] for lc in lower_level if lc.get('parent') == comm['id']]

            # Compute position as mean of member positions
            member_positions = []
            for node_id in comm['nodes'][:50]:
                if node_id in clusters['level_0']:
                    member_positions.append(clusters['level_0'][node_id]['position'])

            position = np.mean(member_positions, axis=0).tolist() if member_positions else [0, 0, 0]

            cluster_type = ['fine_cluster', 'medium_cluster', 'coarse_cluster'][min(leiden_level, 2)]

            clusters[level_key][comm['id']] = {
                'id': comm['id'],
                'name': comm.get('title', f"Community {comm['id']}"),
                'description': comm.get('summary', ''),
                'type': cluster_type,
                'children': children,
                'entities': comm['nodes'] if leiden_level == 0 else [],
                'position': position,
                'umap_position': position,
                'size': len(comm['nodes']),
                'modularity': comm.get('modularity', 0.0)
            }

    # Print statistics
    for level_key in sorted(clusters.keys()):
        print(f"  {level_key}: {len(clusters[level_key]):,} clusters")

    return {
        'entities': entities,
        'relationships': relationships,
        'clusters': clusters,
        'metadata': {
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'clustering_method': 'hierarchical_leiden',
            'leiden_resolutions': [LEIDEN_RESOLUTION_L0, LEIDEN_RESOLUTION_L1, LEIDEN_RESOLUTION_L2],
            'umap_params': {
                'n_components': UMAP_N_COMPONENTS,
                'n_neighbors': UMAP_N_NEIGHBORS,
                'min_dist': UMAP_MIN_DIST,
                'metric': UMAP_METRIC
            },
            'levels': 4,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_graph': str(DISCOURSE_GRAPH_PATH)
        }
    }


def main():
    print("=" * 80)
    print("PROPER GRAPHRAG GENERATION")
    print("Leiden Community Detection + UMAP Positioning + LLM Summaries")
    print("=" * 80)

    # Production safeguards
    acquire_lockfile()
    validate_checkpoints()

    try:
        # Load API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        client = OpenAI(api_key=api_key)

        # Load data
        entities, relationships = load_knowledge_graph(DISCOURSE_GRAPH_PATH)
        rel_index = build_relationship_index(relationships)

        # Generate embeddings and UMAP
        embeddings, entity_ids = create_embeddings(entities, rel_index, client)

        # Build hierarchy
        hierarchies, umap_positions, betweenness_scores, relationship_strengths = build_graphrag_hierarchy(
            entities,
            relationships,
            rel_index,
            embeddings,
            entity_ids,
            client
        )

        # Convert to visualization format
        graphrag_data = convert_to_graphrag_format(
            hierarchies,
            entities,
            relationships,
            entity_ids,
            umap_positions,
            betweenness_scores,
            relationship_strengths
        )

        # Backup existing file
        if OUTPUT_PATH.exists():
            print(f"\n📦 Creating backup at {BACKUP_PATH}")
            BACKUP_PATH.write_text(OUTPUT_PATH.read_text())

        # Save results with fsync
        print(f"\n💾 Saving GraphRAG data to {OUTPUT_PATH}...")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open('w') as f:
            json.dump(graphrag_data, f)
            f.flush()
            os.fsync(f.fileno())

        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"\n✅ GraphRAG hierarchy complete! ({size_mb:.2f} MB)")
        print(f"\nNext steps:")
        print(f"  1. Deploy: sudo cp {OUTPUT_PATH} /opt/yonearth-chatbot/web/graph/data/graphrag_hierarchy/")
        print(f"  2. Test: https://gaiaai.xyz/graph/GraphRAG3D_EmbeddingView.html")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        release_lockfile()


if __name__ == '__main__':
    main()
