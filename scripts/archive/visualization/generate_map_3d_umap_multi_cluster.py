"""
Generate 3D podcast map with UMAP dimensionality reduction
Uses 3D UMAP once + multiple K-means cluster levels (9, 18) + GPT-4 for semantic labels

This generates ONE set of 3D coordinates with MULTIPLE cluster assignments,
similar to the 2D hierarchical map approach.
"""

import os
import json
import re
import numpy as np
from pinecone import Pinecone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import umap
from pathlib import Path

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "yonearth-episodes")
OUTPUT_FILE = "/root/yonearth-gaia-chatbot/data/processed/podcast_map_3d_umap_multi_cluster.json"

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("ERROR: API keys not set!")
    exit(1)

print(f"✓ Environment loaded")

# Parameters
MAX_VECTORS = int(os.getenv("MAX_VECTORS", "6000"))
CLUSTER_LEVELS = [9, 18]  # Default cluster levels (may be overridden by optimal K analysis)
SAMPLE_SIZE = 15
UMAP_MIN_DIST = float(os.getenv("UMAP_MIN_DIST", "0.1"))
UMAP_N_NEIGHBORS = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
USE_OPTIMAL_K = os.getenv("USE_OPTIMAL_K", "true").lower() == "true"  # Enable silhouette analysis

# Transcript path for enriching episode titles
TRANSCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "transcripts"


def load_transcript_title(episode_number: str) -> dict:
    """Load title and guest info from a transcript file."""
    transcript_path = TRANSCRIPTS_DIR / f"episode_{episode_number}.json"

    if not transcript_path.exists():
        return {}

    try:
        with open(transcript_path, 'r') as f:
            data = json.load(f)
        return {
            'title': data.get('title'),
            'guest_name': data.get('guest_name')
        }
    except (json.JSONDecodeError, IOError):
        return {}


def format_dropdown_title(episode_number: str, title: str, guest_name: str = None) -> str:
    """Format episode title for dropdown display.

    Goal: "Ep 1: Nancy Tuchman - Loyola U..." (concise, scannable)
    """
    if title and title != f"Episode {episode_number}":
        # Extract guest and topic from full title
        match = re.match(r'Episode\s*\d+\s*[–\-—]\s*(.+)', title)
        if match:
            rest = match.group(1)

            # Try to split on " on " or " - " to separate guest from topic
            if ' on ' in rest.lower():
                parts = rest.split(' on ', 1)
                if len(parts) == 2:
                    guest, topic = parts[0].strip(), parts[1].strip()
                    if len(topic) > 40:
                        topic = topic[:37] + "..."
                    return f"Ep {episode_number}: {guest} - {topic}"

            for sep in [' – ', ' - ', ' — ']:
                if sep in rest:
                    parts = rest.split(sep, 1)
                    if len(parts) == 2:
                        guest, topic = parts[0].strip(), parts[1].strip()
                        if len(topic) > 40:
                            topic = topic[:37] + "..."
                        return f"Ep {episode_number}: {guest} - {topic}"

            # If no separator found, use the whole thing
            if len(rest) > 50:
                rest = rest[:47] + "..."
            return f"Ep {episode_number}: {rest}"
    elif guest_name:
        return f"Ep {episode_number}: {guest_name}"

    return f"Episode {episode_number}"


def fetch_vectors():
    """Fetch vectors from Pinecone - prioritize timestamped chunks"""
    print("\n" + "="*70)
    print("FETCHING VECTORS FROM PINECONE")
    print("="*70)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    all_vectors = []
    seen_ids = set()
    episode_coverage = set()
    timestamped_count = 0

    # First try to fetch timestamped chunks (from Whisper transcription)
    for seed in range(0, 100, 5):
        if len(all_vectors) >= MAX_VECTORS:
            break

        np.random.seed(seed)
        random_vector = np.random.randn(1536).tolist()

        # Prioritize timestamped episode chunks
        results = index.query(
            vector=random_vector,
            filter={
                'content_type': {'$eq': 'episode'},
                'chunk_type': {'$eq': 'timestamped_segment'}
            },
            top_k=3000,
            include_metadata=True,
            include_values=True
        )

        for match in results['matches']:
            if len(all_vectors) >= MAX_VECTORS:
                break
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                all_vectors.append(match)

                if match.get('metadata', {}).get('timestamp') is not None:
                    timestamped_count += 1

                ep_id = str(match.get('metadata', {}).get('episode_number', ''))
                if ep_id:
                    episode_coverage.add(ep_id)

        print(f"  Seed {seed}: {len(episode_coverage)} episodes, {len(all_vectors)} vectors ({timestamped_count} timestamped)")

        if len(episode_coverage) >= 172 or len(all_vectors) >= MAX_VECTORS:
            break

    # If we don't have enough vectors, fetch non-timestamped episode chunks as fallback
    if len(all_vectors) < MAX_VECTORS:
        print(f"\nFetching additional non-timestamped chunks...")
        for seed in range(0, 100, 5):
            if len(all_vectors) >= MAX_VECTORS:
                break

            np.random.seed(seed + 1000)  # Different seed
            random_vector = np.random.randn(1536).tolist()

            results = index.query(
                vector=random_vector,
                filter={'content_type': {'$eq': 'episode'}},
                top_k=3000,
                include_metadata=True,
                include_values=True
            )

            for match in results['matches']:
                if len(all_vectors) >= MAX_VECTORS:
                    break
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    all_vectors.append(match)

                    if match.get('metadata', {}).get('timestamp') is not None:
                        timestamped_count += 1

                    ep_id = str(match.get('metadata', {}).get('episode_number', ''))
                    if ep_id:
                        episode_coverage.add(ep_id)

    print(f"\n✓ Fetched {len(all_vectors)} vectors from {len(episode_coverage)} episodes")
    print(f"✓ {timestamped_count} chunks have real timestamps ({timestamped_count/len(all_vectors)*100:.1f}%)")
    return all_vectors


def reduce_to_3d(vectors):
    """Reduce to 3D with UMAP ONCE"""
    print("\n" + "="*70)
    print("3D DIMENSIONALITY REDUCTION (UMAP)")
    print("="*70)

    embeddings = np.array([v['values'] for v in vectors])

    print(f"\nReducing {len(embeddings)} embeddings to 3D with UMAP...")
    print("This creates ONE set of 3D coordinates used for all cluster levels")

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    embeddings_3d = reducer.fit_transform(embeddings)
    print("✓ 3D UMAP reduction complete")

    return embeddings_3d


def find_optimal_k(embeddings_3d, k_range=range(5, 16)):
    """
    Find optimal K using silhouette score analysis.

    Silhouette score measures how similar objects are to their own cluster vs other clusters.
    Score ranges from -1 to 1:
      - Score > 0.5: Good separation
      - Score 0.25-0.5: Reasonable structure
      - Score < 0.25: Poor clustering

    Returns: (best_k, scores_dict) - optimal K value and all scores
    """
    print("\n" + "="*70)
    print("SILHOUETTE ANALYSIS FOR OPTIMAL K")
    print("="*70)
    print(f"Testing K values from {k_range.start} to {k_range.stop - 1}...")

    scores = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_3d)
        score = silhouette_score(embeddings_3d, labels)
        scores[k] = score

        # Visual indicator of quality
        if score >= 0.5:
            quality = "★★★ Excellent"
        elif score >= 0.35:
            quality = "★★☆ Good"
        elif score >= 0.25:
            quality = "★☆☆ Fair"
        else:
            quality = "☆☆☆ Poor"

        print(f"  K={k:2d}: silhouette={score:.4f} {quality}")

    # Find best K (highest silhouette score)
    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    print(f"\n✓ Optimal K: {best_k} (silhouette score: {best_score:.4f})")

    if best_score < 0.25:
        print("⚠️  Warning: Even the best K has a low silhouette score.")
        print("   This may indicate that the data doesn't cluster well naturally.")

    return best_k, scores


def cluster_at_level(embeddings_3d, n_clusters):
    """Cluster at specific level"""
    print(f"\nClustering into {n_clusters} semantic topics...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_3d)

    # Calculate and print silhouette score for this clustering
    score = silhouette_score(embeddings_3d, cluster_labels)
    print(f"  Silhouette score: {score:.4f}")

    # Print cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster distribution for {n_clusters} clusters:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} chunks")

    return cluster_labels


def generate_topic_label(vectors, cluster_indices, cluster_id, n_clusters):
    """Generate semantic label for one cluster using GPT-4"""
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Sample random chunks
    sample_indices = np.random.choice(cluster_indices, min(SAMPLE_SIZE, len(cluster_indices)), replace=False)
    sample_texts = [vectors[i].get('metadata', {}).get('text', '')[:200] for i in sample_indices]

    # Create prompt for GPT-4
    prompt = f"""Analyze these {len(sample_texts)} podcast transcript excerpts and identify the main theme or topic they discuss.

Excerpts:
{chr(10).join([f'{i+1}. "{text}..."' for i, text in enumerate(sample_texts)])}

Based on these excerpts, provide:
1. A concise topic label (3-6 words, like "Soil Health & Regenerative Agriculture")
2. 3-5 key themes or concepts

Format your response as JSON:
{{"label": "Topic Label", "themes": ["theme1", "theme2", "theme3"]}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    print(f"  Cluster {cluster_id}: {result['label']}")
    return result


def generate_all_cluster_labels(vectors, all_cluster_labels):
    """Generate labels for all cluster levels"""
    print("\n" + "="*70)
    print("DISCOVERING SEMANTIC TOPICS WITH GPT-4")
    print("="*70)

    clusters_metadata = {}

    for n_clusters, cluster_labels in all_cluster_labels.items():
        print(f"\n--- Labeling {n_clusters} clusters ---")
        clusters_metadata[n_clusters] = {}

        for cluster_id in range(n_clusters):
            # Get chunks in this cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

            # Generate label
            result = generate_topic_label(vectors, cluster_indices, cluster_id, n_clusters)

            clusters_metadata[n_clusters][cluster_id] = {
                'id': cluster_id,
                'label': result['label'],
                'themes': result.get('themes', []),
                'count': len(cluster_indices)
            }

    return clusters_metadata


def assign_colors(n_clusters):
    """Assign colors to clusters using a color palette"""
    # Extended color palette for up to 18 clusters
    colors = [
        '#4CAF50', '#9C27B0', '#FF9800', '#2196F3', '#F44336',
        '#00BCD4', '#8BC34A', '#FF5722', '#3F51B5', '#FFEB3B',
        '#E91E63', '#009688', '#FFC107', '#673AB7', '#CDDC39',
        '#795548', '#607D8B', '#FF4081'
    ]
    return colors[:n_clusters]


def build_output(vectors, embeddings_3d, all_cluster_labels, clusters_metadata):
    """Build final output JSON"""
    print("\n" + "="*70)
    print("BUILDING OUTPUT JSON")
    print("="*70)

    # Get episode list with enriched titles from transcripts
    episodes_dict = {}
    for v in vectors:
        meta = v.get('metadata', {})
        ep_id = str(meta.get('episode_number', ''))
        if ep_id and ep_id not in episodes_dict:
            # Try to get richer title from transcript file
            transcript_meta = load_transcript_title(ep_id)
            title = format_dropdown_title(
                ep_id,
                transcript_meta.get('title'),
                transcript_meta.get('guest_name')
            )

            episodes_dict[ep_id] = {
                'id': ep_id,
                'title': title,
                'audio_url': meta.get('audio_url', '')
            }

            # Also store full title if available
            if transcript_meta.get('title'):
                episodes_dict[ep_id]['full_title'] = transcript_meta['title']
            if transcript_meta.get('guest_name'):
                episodes_dict[ep_id]['guest_name'] = transcript_meta['guest_name']

    # Scale coordinates to reasonable range
    x_coords = embeddings_3d[:, 0]
    y_coords = embeddings_3d[:, 1]
    z_coords = embeddings_3d[:, 2]

    scale_factor = 500
    x_coords = (x_coords - x_coords.mean()) / x_coords.std() * scale_factor
    y_coords = (y_coords - y_coords.mean()) / y_coords.std() * scale_factor
    z_coords = (z_coords - z_coords.mean()) / z_coords.std() * scale_factor

    # Build points with multiple cluster assignments
    # Use REAL timestamps from Whisper transcription (if available)
    # Fallback to proportional estimation for chunks without timestamps
    AVERAGE_EPISODE_DURATION = 3000  # 50 minutes in seconds (fallback)

    points = []
    for i, v in enumerate(vectors):
        meta = v.get('metadata', {})

        episode_id = str(meta.get('episode_number', ''))

        # Use real timestamp if available (from Whisper transcription)
        timestamp = meta.get('timestamp')

        if timestamp is None:
            # Fallback: estimate timestamp based on chunk position (old method)
            chunk_index = meta.get('chunk_index', 0)
            chunk_total = meta.get('chunk_total', 1)
            timestamp = (chunk_index / max(chunk_total, 1)) * AVERAGE_EPISODE_DURATION if chunk_total else 0

        point = {
            'id': v['id'],
            'text': meta.get('text', ''),
            'x': float(x_coords[i]),
            'y': float(y_coords[i]),
            'z': float(z_coords[i]),
            'episode_id': episode_id,
            'episode_title': meta.get('title', ''),
            'timestamp': timestamp  # Real timestamp from Whisper or fallback estimation
        }

        # Add cluster assignments for each level
        for n_clusters, labels in all_cluster_labels.items():
            cluster_id = int(labels[i])
            point[f'cluster_{n_clusters}'] = cluster_id
            point[f'cluster_{n_clusters}_name'] = clusters_metadata[n_clusters][cluster_id]['label']

        points.append(point)

    # Build clusters data for each level
    clusters_by_level = {}
    for n_clusters in CLUSTER_LEVELS:
        colors = assign_colors(n_clusters)
        clusters_by_level[n_clusters] = [
            {
                'id': cluster_id,
                'label': clusters_metadata[n_clusters][cluster_id]['label'],
                'color': colors[cluster_id],
                'count': clusters_metadata[n_clusters][cluster_id]['count'],
                'themes': clusters_metadata[n_clusters][cluster_id]['themes']
            }
            for cluster_id in range(n_clusters)
        ]

    # Build links (edges)
    links = []
    episode_nodes = {}
    for point in points:
        ep_id = point['episode_id']
        if ep_id not in episode_nodes:
            episode_nodes[ep_id] = []
        episode_nodes[ep_id].append(point['id'])

    for ep_id, node_ids in episode_nodes.items():
        for i in range(len(node_ids) - 1):
            links.append({
                'source': node_ids[i],
                'target': node_ids[i + 1],
                'episode_id': ep_id
            })

    output = {
        'points': points,
        'episodes': list(episodes_dict.values()),
        'clusters_by_level': clusters_by_level,
        'links': links,
        'total_points': len(points),
        'cluster_levels': CLUSTER_LEVELS,
        'default_level': 9,
        'umap_params': {
            'n_neighbors': UMAP_N_NEIGHBORS,
            'min_dist': UMAP_MIN_DIST
        }
    }

    print(f"✓ Built output with {len(points)} points and {len(links)} links")
    print(f"✓ Cluster levels: {CLUSTER_LEVELS}")
    return output


def main():
    global CLUSTER_LEVELS

    print("="*70)
    print("3D PODCAST MAP GENERATION (MULTI-CLUSTER)")
    print("="*70)

    # Step 1: Fetch vectors
    vectors = fetch_vectors()

    # Step 2: Reduce to 3D ONCE
    embeddings_3d = reduce_to_3d(vectors)

    # Step 3: Determine cluster levels
    if USE_OPTIMAL_K:
        # Run silhouette analysis to find optimal K
        optimal_k, scores = find_optimal_k(embeddings_3d, k_range=range(5, 16))

        # Use optimal K as primary level, and double it for secondary level
        # But cap secondary at 18 to avoid too many clusters
        secondary_k = min(optimal_k * 2, 18)

        # If optimal K is already high, just use it and a smaller level
        if optimal_k >= 12:
            CLUSTER_LEVELS = [optimal_k // 2, optimal_k]
        else:
            CLUSTER_LEVELS = [optimal_k, secondary_k]

        print(f"\n✓ Using cluster levels: {CLUSTER_LEVELS} (based on silhouette analysis)")
    else:
        print(f"\n✓ Using default cluster levels: {CLUSTER_LEVELS}")

    # Step 4: Cluster at multiple levels
    print("\n" + "="*70)
    print("CLUSTERING AT MULTIPLE LEVELS")
    print("="*70)

    all_cluster_labels = {}
    for n_clusters in CLUSTER_LEVELS:
        all_cluster_labels[n_clusters] = cluster_at_level(embeddings_3d, n_clusters)

    # Step 4: Generate labels for all clusters
    clusters_metadata = generate_all_cluster_labels(vectors, all_cluster_labels)

    # Step 5: Build output
    output = build_output(vectors, embeddings_3d, all_cluster_labels, clusters_metadata)

    # Step 6: Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*70)
    print("✓ 3D MAP GENERATION COMPLETE!")
    print("="*70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Points: {output['total_points']}")
    print(f"Episodes: {len(output['episodes'])}")
    print(f"Cluster levels: {output['cluster_levels']}")


if __name__ == "__main__":
    main()
