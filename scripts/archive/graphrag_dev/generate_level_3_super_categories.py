#!/usr/bin/env python3
"""
Generate Level 4 "Super Categories" for GraphRAG hierarchy.

Uses a hybrid structural + semantic approach to group the 57 Level 3 clusters
into 5-12 top-level navigation categories.

Approach:
1. Load L3 clusters from cluster_registry.json
2. Build structural affinity matrix (edge counts between L3 clusters)
3. Build semantic affinity matrix (cosine similarity of L3 summary embeddings)
4. Combine affinities (50% structural + 50% semantic)
5. Use AgglomerativeClustering with distance_threshold to find natural groupings
6. Generate LLM-based labels for each super-category
7. Update cluster_registry.json and graphrag_hierarchy.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CLUSTER_REGISTRY_PATH = DATA_DIR / "graphrag_hierarchy" / "cluster_registry.json"
GRAPHRAG_HIERARCHY_PATH = DATA_DIR / "graphrag_hierarchy" / "graphrag_hierarchy.json"
DISCOURSE_GRAPH_PATH = DATA_DIR / "knowledge_graph_unified" / "discourse_graph_hybrid.json"

# Parameters
EMBEDDING_MODEL = "text-embedding-3-small"
TARGET_MIN_CLUSTERS = 5
TARGET_MAX_CLUSTERS = 12
INITIAL_DISTANCE_THRESHOLD = 0.85
STRUCTURAL_WEIGHT = 0.2  # Less weight on structure (L2 clusters are already disconnected)
SEMANTIC_WEIGHT = 0.8   # More weight on semantic similarity


def load_cluster_registry() -> Dict:
    """Load cluster registry JSON."""
    print(f"Loading cluster registry from {CLUSTER_REGISTRY_PATH}")
    with CLUSTER_REGISTRY_PATH.open("r") as f:
        return json.load(f)


def load_graphrag_hierarchy() -> Dict:
    """Load graphrag hierarchy JSON."""
    print(f"Loading graphrag hierarchy from {GRAPHRAG_HIERARCHY_PATH}")
    with GRAPHRAG_HIERARCHY_PATH.open("r") as f:
        return json.load(f)


def load_discourse_graph() -> Tuple[Dict, List]:
    """Load discourse graph (entities and relationships)."""
    print(f"Loading discourse graph from {DISCOURSE_GRAPH_PATH}")
    with DISCOURSE_GRAPH_PATH.open("r") as f:
        data = json.load(f)
    return data["entities"], data["relationships"]


def build_networkx_graph(entities: Dict, relationships: List) -> nx.Graph:
    """Build NetworkX graph from entities and relationships.

    Supports both standard keys (source/target) and discourse keys (source_entity/target_entity).
    """
    print("\nBuilding NetworkX graph...")
    G = nx.Graph()

    # Add nodes
    for entity_id, entity_data in entities.items():
        G.add_node(entity_id, **entity_data)

    # Add edges
    for rel in relationships:
        # Support both naming conventions
        source = rel.get("source") or rel.get("source_entity")
        target = rel.get("target") or rel.get("target_entity")

        if source and target and source in G and target in G:
            weight = rel.get("weight", 1.0)
            G.add_edge(source, target, weight=weight)

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def get_level_2_clusters(cluster_registry: Dict) -> List[Dict]:
    """Extract all Level 2 clusters from registry (the current top level)."""
    l2_clusters = []
    for cluster_id, cluster_data in cluster_registry.items():
        if cluster_data["level"] == 2:
            l2_clusters.append({
                "id": cluster_id,
                "entities": cluster_data["entities"],
                "summary_text": cluster_data["summary_text"],
                "entity_count": cluster_data["entity_count"],
            })

    # Sort by cluster ID for consistency
    l2_clusters.sort(key=lambda x: x["id"])
    print(f"\nFound {len(l2_clusters)} Level 2 clusters")
    return l2_clusters


def build_structural_affinity_matrix(l2_clusters: List[Dict], G: nx.Graph) -> np.ndarray:
    """
    Build structural affinity matrix based on edge counts between L2 clusters.

    Returns a 57x57 matrix where entry [i][j] is the normalized number of edges
    between entities in cluster i and entities in cluster j.
    """
    print("\nBuilding structural affinity matrix...")
    n = len(l2_clusters)
    affinity = np.zeros((n, n))

    # For each pair of clusters, count edges between their entities
    for i in range(n):
        entities_i = set(l2_clusters[i]["entities"])
        for j in range(i, n):
            entities_j = set(l2_clusters[j]["entities"])

            # Count edges between these two clusters
            edge_count = 0
            for entity_i in entities_i:
                if entity_i not in G:
                    continue
                for entity_j in entities_j:
                    if entity_j not in G:
                        continue
                    if G.has_edge(entity_i, entity_j):
                        edge_count += 1

            affinity[i][j] = edge_count
            affinity[j][i] = edge_count  # Symmetric

    # Normalize to 0-1 range
    max_edges = affinity.max()
    if max_edges > 0:
        affinity = affinity / max_edges

    print(f"  Max edge count between L2 clusters: {max_edges:.0f}")
    print(f"  Structural affinity matrix shape: {affinity.shape}")
    return affinity


def get_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    print(f"\nGenerating embeddings for {len(texts)} texts...")

    embeddings = []
    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(texts)}")

        # Truncate long texts (OpenAI has 8191 token limit)
        text = text[:8000]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embeddings.append(response.data[0].embedding)

    print(f"  ✓ Generated {len(embeddings)} embeddings")
    return np.array(embeddings)


def build_semantic_affinity_matrix(l2_clusters: List[Dict], client: OpenAI) -> np.ndarray:
    """
    Build semantic affinity matrix based on cosine similarity of L2 summary embeddings.

    Returns a 57x57 matrix where entry [i][j] is the cosine similarity between
    the summary embeddings of cluster i and cluster j.
    """
    print("\nBuilding semantic affinity matrix...")

    # Extract summary texts
    summaries = [cluster["summary_text"] for cluster in l2_clusters]

    # Get embeddings
    embeddings = get_embeddings(summaries, client)

    # Calculate cosine similarity matrix
    affinity = cosine_similarity(embeddings)

    print(f"  Semantic affinity matrix shape: {affinity.shape}")
    print(f"  Affinity range: [{affinity.min():.3f}, {affinity.max():.3f}]")
    return affinity


def find_optimal_clustering(
    distance_matrix: np.ndarray,
    initial_threshold: float = INITIAL_DISTANCE_THRESHOLD,
    target_min: int = TARGET_MIN_CLUSTERS,
    target_max: int = TARGET_MAX_CLUSTERS,
) -> Tuple[np.ndarray, float]:
    """
    Find optimal AgglomerativeClustering threshold to get 5-12 clusters.

    Returns (cluster_labels, threshold_used).
    """
    print(f"\nFinding optimal clustering threshold...")
    print(f"  Target range: {target_min}-{target_max} clusters")

    threshold = initial_threshold
    best_labels = None
    best_threshold = threshold
    best_n_clusters = None

    # Try different thresholds
    for attempt in range(20):
        # Ensure threshold is non-negative
        threshold = max(0.0, threshold)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average"
        )
        labels = clustering.fit_predict(distance_matrix)
        n_clusters = len(np.unique(labels))

        print(f"  Attempt {attempt + 1}: threshold={threshold:.3f} → {n_clusters} clusters")

        if target_min <= n_clusters <= target_max:
            best_labels = labels
            best_threshold = threshold
            best_n_clusters = n_clusters
            print(f"  ✓ Found optimal clustering: {n_clusters} clusters at threshold={threshold:.3f}")
            break

        # Store best attempt so far
        if best_labels is None or (best_n_clusters and abs(n_clusters - target_max) < abs(best_n_clusters - target_max)):
            best_labels = labels
            best_threshold = threshold
            best_n_clusters = n_clusters

        # Adjust threshold
        if n_clusters < target_min:
            # Too few clusters, increase threshold (more merging)
            threshold += 0.05
        else:
            # Too many clusters, decrease threshold (less merging)
            threshold -= 0.05

    n_clusters = len(np.unique(best_labels))
    print(f"\n  Final result: {n_clusters} clusters at threshold={best_threshold:.3f}")

    return best_labels, best_threshold


def generate_super_category_labels(
    l3_clusters: List[List[Dict]],
    client: OpenAI
) -> List[Dict]:
    """
    Generate human-readable labels for Level 3 super-categories using LLM.

    Returns a list of dicts with 'name' and 'description' for each L3 cluster.
    """
    print("\nGenerating LLM-based labels for Level 3 super-categories...")

    labels = []
    for i, l2_members in enumerate(l3_clusters):
        print(f"\n  Processing L3 cluster {i + 1}/{len(l3_clusters)} ({len(l2_members)} L2 clusters)...")

        # Aggregate summaries from L2 clusters
        aggregated_summary = "\n\n".join([
            f"- {cluster['summary_text'][:500]}"
            for cluster in l2_members
        ])

        # Truncate if too long
        aggregated_summary = aggregated_summary[:4000]

        # LLM prompt
        prompt = f"""You are analyzing a knowledge graph community structure. Below are descriptions of several related sub-communities that have been grouped together.

Your task: Create a concise top-level category label for this group.

Sub-community descriptions:
{aggregated_summary}

Please provide:
1. A short menu label (maximum 4 words, title case)
2. A one-sentence description (1-2 lines maximum)

Format your response as:
LABEL: [your label]
DESCRIPTION: [your description]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a knowledge graph analyst creating concise category labels."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            result = response.choices[0].message.content.strip()

            # Parse result
            name = ""
            description = ""
            for line in result.split("\n"):
                if line.startswith("LABEL:"):
                    name = line.replace("LABEL:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()

            # Fallback if parsing failed
            if not name:
                name = f"Super Category {i + 1}"
            if not description:
                description = "A collection of related knowledge communities."

            print(f"    Label: {name}")
            print(f"    Description: {description[:80]}...")

            labels.append({
                "name": name,
                "description": description,
                "l2_members": [cluster["id"] for cluster in l2_members]
            })

        except Exception as e:
            print(f"    ERROR generating label: {e}")
            labels.append({
                "name": f"Super Category {i + 1}",
                "description": "A collection of related knowledge communities.",
                "l2_members": [cluster["id"] for cluster in l2_members]
            })

    return labels


def update_cluster_registry(
    cluster_registry: Dict,
    l3_labels: List[Dict],
    l2_clusters: List[Dict]
) -> Dict:
    """
    Update cluster_registry.json to include Level 3 super-category clusters.
    """
    print("\nUpdating cluster registry with Level 3 super-categories...")

    for i, l3_data in enumerate(l3_labels):
        cluster_id = f"level_3_{i}"

        # Collect all entities from L2 children
        all_entities = []
        for l2_id in l3_data["l2_members"]:
            l2_cluster = cluster_registry[l2_id]
            all_entities.extend(l2_cluster["entities"])

        # Build summary text
        summary_text = f"{l3_data['name']}: {l3_data['description']}"

        # Add to registry
        cluster_registry[cluster_id] = {
            "id": cluster_id,
            "level": 3,
            "type": "super_category",
            "parent": None,  # Top level
            "children": l3_data["l2_members"],
            "entities": all_entities,
            "entity_count": len(all_entities),
            "summary_text": summary_text,
            "name": l3_data["name"],
            "description": l3_data["description"]
        }

        # Update parent references in L2 clusters
        for l2_id in l3_data["l2_members"]:
            cluster_registry[l2_id]["parent"] = cluster_id

        print(f"  Created {cluster_id}: {l3_data['name']} ({len(all_entities)} entities)")

    return cluster_registry


def update_graphrag_hierarchy(
    hierarchy: Dict,
    cluster_registry: Dict
) -> Dict:
    """
    Update graphrag_hierarchy.json to include Level 3 super-category clusters.

    IMPORTANT: Level 3 clusters need 'position' and 'umap_position' fields (3D coords),
    not just the 1536-dim 'center' embedding field. We calculate these as centroids of
    L2 children's 3D positions.
    """
    print("\nUpdating graphrag hierarchy with Level 3 super-categories...")

    level_3 = {}

    for cluster_id, cluster_data in cluster_registry.items():
        if cluster_data["level"] == 3:
            # Get 3D positions from L2 children (not 1536-dim embeddings!)
            children_ids = cluster_data["children"]
            children_positions = []
            children_embeddings = []

            for child_id in children_ids:
                if child_id in hierarchy["clusters"]["level_2"]:
                    l2_cluster = hierarchy["clusters"]["level_2"][child_id]

                    # Get 3D position (for visualization)
                    position_3d = l2_cluster.get("position") or l2_cluster.get("umap_position")
                    if position_3d and len(position_3d) == 3:
                        children_positions.append(position_3d)

                    # Get 1536-dim embedding (for semantic calculations)
                    embedding = l2_cluster.get("center")
                    if embedding and len(embedding) == 1536:
                        children_embeddings.append(embedding)

            # Calculate 3D position centroid
            if children_positions:
                position_3d = np.mean(children_positions, axis=0).tolist()
            else:
                position_3d = [0.0, 0.0, 0.0]

            # Calculate 1536-dim embedding centroid
            if children_embeddings:
                embedding_center = np.mean(children_embeddings, axis=0).tolist()
            else:
                embedding_center = [0.0] * 1536

            level_3[cluster_id] = {
                "id": cluster_id,
                "entity_ids": cluster_data["entities"],
                "center": embedding_center,  # 1536-dim for semantic calculations
                "position": position_3d,  # 3D for visualization
                "umap_position": position_3d,  # 3D for visualization (alias)
                "children": cluster_data["children"],
                "entity_count": cluster_data["entity_count"],
                "name": cluster_data.get("name", ""),
                "description": cluster_data.get("description", ""),
                "type": "super_category",
                "size": cluster_data["entity_count"]
            }

    hierarchy["clusters"]["level_3"] = level_3

    print(f"  Added {len(level_3)} Level 3 super-categories to hierarchy")

    # Verify positions
    for cid, cluster in level_3.items():
        print(f"    {cid}: position={cluster['position'][:2]}... ({cluster['entity_count']} entities)")

    return hierarchy


def main():
    print("=" * 80)
    print("Level 3 Super-Category Generation")
    print("Hybrid Structural + Semantic Clustering")
    print("=" * 80)

    # Check OpenAI key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise SystemExit(1)
    client = OpenAI(api_key=api_key)

    # Load data
    cluster_registry = load_cluster_registry()
    hierarchy = load_graphrag_hierarchy()
    entities, relationships = load_discourse_graph()
    G = build_networkx_graph(entities, relationships)

    # Get Level 2 clusters (current top level)
    l2_clusters = get_level_2_clusters(cluster_registry)

    # Build affinity matrices
    structural_affinity = build_structural_affinity_matrix(l2_clusters, G)
    semantic_affinity = build_semantic_affinity_matrix(l2_clusters, client)

    # Combine affinities
    print(f"\nCombining affinities ({STRUCTURAL_WEIGHT:.0%} structural + {SEMANTIC_WEIGHT:.0%} semantic)...")
    combined_affinity = (STRUCTURAL_WEIGHT * structural_affinity) + (SEMANTIC_WEIGHT * semantic_affinity)

    # Convert to distance matrix
    distance_matrix = 1 - combined_affinity

    # Cluster
    cluster_labels, threshold = find_optimal_clustering(distance_matrix)

    # Group L2 clusters by L3 label
    l3_clusters = []
    for l3_label in np.unique(cluster_labels):
        l2_members = [l2_clusters[i] for i in range(len(l2_clusters)) if cluster_labels[i] == l3_label]
        l3_clusters.append(l2_members)

    print(f"\nResulting L3 structure:")
    for i, members in enumerate(l3_clusters):
        print(f"  L3 Cluster {i}: {len(members)} L2 clusters, {sum(m['entity_count'] for m in members)} total entities")

    # Generate labels
    l3_labels = generate_super_category_labels(l3_clusters, client)

    # Update registry and hierarchy
    cluster_registry = update_cluster_registry(cluster_registry, l3_labels, l2_clusters)
    hierarchy = update_graphrag_hierarchy(hierarchy, cluster_registry)

    # Save updated files
    print(f"\nSaving updated cluster registry to {CLUSTER_REGISTRY_PATH}")
    with CLUSTER_REGISTRY_PATH.open("w") as f:
        json.dump(cluster_registry, f, indent=2)

    print(f"Saving updated hierarchy to {GRAPHRAG_HIERARCHY_PATH}")
    with GRAPHRAG_HIERARCHY_PATH.open("w") as f:
        json.dump(hierarchy, f)

    print("\n" + "=" * 80)
    print("✓ Level 3 super-categories generated successfully!")
    print("=" * 80)

    print("\nSummary:")
    print(f"  L3 Super-Categories: {len(l3_labels)}")
    for label in l3_labels:
        print(f"    - {label['name']}: {len(label['l2_members'])} L2 clusters")

    print(f"\nNext steps:")
    print(f"  1. Review cluster_registry.json for Level 3 entries")
    print(f"  2. Deploy to production:")
    print(f"     sudo cp {CLUSTER_REGISTRY_PATH} /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/")
    print(f"     sudo cp {GRAPHRAG_HIERARCHY_PATH} /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/")
    print(f"     sudo systemctl reload nginx")


if __name__ == "__main__":
    main()
