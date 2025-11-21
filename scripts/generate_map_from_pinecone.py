"""
Generate podcast map visualization data from existing Pinecone embeddings
This uses your existing data - no Nomic required!
"""

import os
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pinecone import Pinecone
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables - specify path explicitly to avoid frame issues
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "yonearth-episodes")
OUTPUT_FILE = "/root/yonearth-gaia-chatbot/data/processed/podcast_map_data.json"

# Debug: Check if environment variables are loaded
if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not set!")
    print("Please ensure .env file exists or set environment variables manually")
    exit(1)

print(f"Environment loaded successfully")
print(f"Index name: {PINECONE_INDEX_NAME}")

# Cluster names for YonEarth podcasts - using the 5 pillars framework
CLUSTER_NAMES = [
    "Community",
    "Culture",
    "Economy",
    "Ecology",
    "Health"
]

CLUSTER_COLORS = [
    "#4CAF50",  # Green - Community
    "#9C27B0",  # Purple - Culture
    "#FF9800",  # Orange - Economy
    "#2196F3",  # Blue - Ecology
    "#F44336"   # Red - Health
]

# Number of clusters
N_CLUSTERS = 5


def fetch_all_vectors_from_pinecone():
    """Fetch vectors from Pinecone using stratified sampling to ensure ALL episodes"""
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    print("Fetching index stats...")
    stats = index.describe_index_stats()
    total_vectors = stats['total_vector_count']
    print(f"Total vectors in index: {total_vectors}")

    print("\nFetching vectors with stratified sampling by episode...")

    max_vectors = 10000  # Memory-safe limit for t-SNE
    target_episodes = 172

    # Strategy: Use multiple diverse queries with smaller batches
    all_vectors = []
    seen_ids = set()
    episode_coverage = set()

    # Use many different random seeds with smaller batch sizes
    for seed in range(0, 100, 5):  # 20 queries of 3000 vectors each
        if len(episode_coverage) >= target_episodes:
            break

        np.random.seed(seed)
        random_vector = np.random.randn(1536).tolist()

        # Query Pinecone with smaller batch
        results = index.query(
            vector=random_vector,
            filter={'content_type': 'episode'},
            top_k=3000,  # Smaller batches to avoid memory issues
            include_metadata=True,
            include_values=True
        )

        # Add new vectors, prioritizing new episodes
        for match in results['matches']:
            if len(all_vectors) >= max_vectors:
                break

            if match['id'] not in seen_ids:
                metadata = match.get('metadata', {})
                ep_id = str(metadata.get('episode_number') or metadata.get('episode_id') or metadata.get('episode') or 'unknown')

                # Always add if it's a new episode, or if we haven't hit vector limit
                if ep_id not in episode_coverage or len(all_vectors) < max_vectors:
                    seen_ids.add(match['id'])
                    all_vectors.append(match)
                    episode_coverage.add(ep_id)

        print(f"Seed {seed}: {len(episode_coverage)}/{target_episodes} episodes, {len(all_vectors)} total vectors")

        if len(episode_coverage) >= target_episodes or len(all_vectors) >= max_vectors:
            if len(episode_coverage) >= target_episodes:
                print(f"✓ All {target_episodes} episodes covered!")
            break

    print(f"\n✓ Total vectors fetched: {len(all_vectors)}")
    print(f"✓ Episodes covered: {len(episode_coverage)}/{target_episodes}")

    # Show which episodes are missing
    all_episode_nums = set(str(i) for i in range(173))  # 0-172
    missing = sorted([int(e) for e in all_episode_nums - episode_coverage if e.isdigit()])
    if missing:
        print(f"ℹ Missing episodes: {missing}")

    # If we have too many vectors, sample down while preserving ALL episodes
    if len(all_vectors) > max_vectors:
        print(f"\nSampling down from {len(all_vectors)} to {max_vectors} vectors...")

        # Group by episode
        episodes_dict = {}
        for v in all_vectors:
            metadata = v.get('metadata', {})
            ep_id = str(metadata.get('episode_number') or metadata.get('episode_id') or metadata.get('episode') or 'unknown')
            if ep_id not in episodes_dict:
                episodes_dict[ep_id] = []
            episodes_dict[ep_id].append(v)

        # Sample evenly from each episode
        chunks_per_episode = max_vectors // len(episodes_dict)
        sampled_vectors = []

        for ep_id, vectors in episodes_dict.items():
            # Take up to chunks_per_episode from each episode
            sampled = vectors[:chunks_per_episode]
            sampled_vectors.extend(sampled)

        all_vectors = sampled_vectors
        print(f"✓ Sampled to {len(all_vectors)} vectors (~{chunks_per_episode} per episode)")

    return all_vectors


def reduce_dimensions_with_tsne(embeddings: np.ndarray, n_components=2):
    """Reduce high-dimensional embeddings to 2D using t-SNE"""
    print(f"\nReducing {len(embeddings)} embeddings to 2D with t-SNE...")
    print("This may take several minutes...")

    tsne = TSNE(
        n_components=n_components,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42,
        verbose=1
    )

    embeddings_2d = tsne.fit_transform(embeddings)
    print("✓ Dimension reduction complete")

    return embeddings_2d


def cluster_embeddings(embeddings: np.ndarray, n_clusters=N_CLUSTERS):
    """Cluster embeddings using K-means"""
    print(f"\nClustering into {n_clusters} clusters ({', '.join(CLUSTER_NAMES)})...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(embeddings)
    print("✓ Clustering complete")

    return cluster_labels


def extract_episode_info(metadata: Dict) -> tuple:
    """Extract episode ID and title from metadata"""
    # Try different metadata field names
    episode_id = (
        metadata.get('episode_number') or
        metadata.get('episode_id') or
        metadata.get('episode') or
        'unknown'
    )

    episode_title = (
        metadata.get('title') or
        metadata.get('episode_title') or
        f"Episode {episode_id}"
    )

    # Extract timestamp based on chunk index (assuming ~2 minutes per chunk)
    chunk_index = metadata.get('chunk_index', 0)
    timestamp = chunk_index * 120  # 2 minutes = 120 seconds per chunk

    # Get audio URL
    audio_url = metadata.get('audio_url', '')

    return str(episode_id), episode_title, timestamp, audio_url


def generate_map_data():
    """Main function to generate map data from Pinecone"""
    print("\n" + "="*70)
    print("GENERATING PODCAST MAP FROM PINECONE DATA")
    print("="*70)
    print("DEBUG: About to fetch vectors...")

    # Step 1: Fetch vectors from Pinecone
    vectors = fetch_all_vectors_from_pinecone()
    print(f"\n✓ Fetched {len(vectors)} vectors from Pinecone")
    print("DEBUG: Vectors fetched successfully")

    # Step 2: Extract embeddings and metadata
    print("\nExtracting embeddings and metadata...")
    embeddings = np.array([v['values'] for v in vectors])

    # Step 3: Reduce to 2D
    embeddings_2d = reduce_dimensions_with_tsne(embeddings)

    # Step 4: Cluster
    cluster_labels = cluster_embeddings(embeddings_2d)

    # Step 5: Build data structure
    print("\nBuilding data structure...")
    points = []
    episodes_map = {}
    clusters_map = {i: {"count": 0} for i in range(N_CLUSTERS)}

    for idx, vector in enumerate(vectors):
        metadata = vector.get('metadata', {})
        episode_id, episode_title, timestamp, audio_url = extract_episode_info(metadata)
        cluster_id = int(cluster_labels[idx])

        # Get the text content
        text = metadata.get('text', metadata.get('content', ''))

        point = {
            "id": vector['id'],
            "text": text,
            "x": float(embeddings_2d[idx][0]),
            "y": float(embeddings_2d[idx][1]),
            "episode_id": episode_id,
            "episode_title": episode_title,
            "timestamp": float(timestamp) if timestamp else None,
            "cluster": cluster_id,
            "cluster_name": CLUSTER_NAMES[cluster_id]
        }
        points.append(point)

        # Track episodes
        if episode_id not in episodes_map:
            episodes_map[episode_id] = {
                "id": episode_id,
                "title": episode_title,
                "chunk_count": 0,
                "audio_url": audio_url or f"https://media.blubrry.com/y_on_earth/yonearth.org/podcast-player/episode-{episode_id}.mp3"
            }
        episodes_map[episode_id]["chunk_count"] += 1

        # Track clusters
        clusters_map[cluster_id]["count"] += 1

    # Build clusters list
    clusters = [
        {
            "id": i,
            "name": CLUSTER_NAMES[i],
            "color": CLUSTER_COLORS[i],
            "count": clusters_map[i]["count"]
        }
        for i in range(N_CLUSTERS)
    ]

    # Build final structure
    export_data = {
        "points": points,
        "episodes": sorted(episodes_map.values(), key=lambda x: x["id"]),
        "clusters": clusters,
        "total_points": len(points)
    }

    # Step 6: Save to file
    print(f"\nSaving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(export_data, f, indent=2)

    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nGenerated visualization data:")
    print(f"  - {len(points)} data points")
    print(f"  - {len(episodes_map)} episodes")
    print(f"  - {len(clusters)} clusters")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("\nYou can now use the visualization without Nomic!")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_map_data()