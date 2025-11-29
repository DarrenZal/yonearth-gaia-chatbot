"""
Generate podcast map visualization using BERTopic for semantic topic discovery
This discovers emergent topics from the data rather than forcing predefined clusters
"""

import os
import json
import numpy as np
from pinecone import Pinecone
from typing import List, Dict, Any
from dotenv import load_dotenv

# BERTopic imports
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Load environment variables - specify path explicitly
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "yonearth-episodes")
OUTPUT_FILE = "/root/yonearth-gaia-chatbot/data/processed/podcast_map_data.json"

# Validation
if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not set!")
    exit(1)

print(f"✓ Environment loaded")
print(f"✓ Index: {PINECONE_INDEX_NAME}")

# Topic modeling parameters
MAX_VECTORS = 10000  # Memory-safe limit
TARGET_TOPICS = 10  # Target number of topics (will auto-adjust)
MIN_TOPIC_SIZE = 50  # Minimum chunks per topic


def fetch_vectors_from_pinecone():
    """Fetch episode vectors from Pinecone for topic modeling"""
    print("\n" + "="*70)
    print("FETCHING VECTORS FROM PINECONE")
    print("="*70)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats['total_vector_count']}")

    all_vectors = []
    seen_ids = set()
    episode_coverage = set()

    # Fetch vectors using multiple diverse queries
    print(f"\nFetching up to {MAX_VECTORS} episode vectors...")

    for seed in range(0, 100, 5):
        if len(all_vectors) >= MAX_VECTORS:
            break

        np.random.seed(seed)
        random_vector = np.random.randn(1536).tolist()

        results = index.query(
            vector=random_vector,
            filter={'content_type': 'episode'},
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

                metadata = match.get('metadata', {})
                ep_id = str(metadata.get('episode_number', ''))
                if ep_id:
                    episode_coverage.add(ep_id)

        print(f"  Seed {seed}: {len(episode_coverage)} episodes, {len(all_vectors)} vectors")

        if len(episode_coverage) >= 172:
            break

    print(f"\n✓ Fetched {len(all_vectors)} vectors covering {len(episode_coverage)} episodes")
    return all_vectors


def run_bertopic_modeling(vectors: List[Dict]) -> tuple:
    """Run BERTopic to discover semantic topics"""
    print("\n" + "="*70)
    print("DISCOVERING SEMANTIC TOPICS WITH BERTOPIC")
    print("="*70)

    # Extract text and embeddings
    texts = [v.get('metadata', {}).get('text', '') for v in vectors]
    embeddings = np.array([v['values'] for v in vectors])

    print(f"\n✓ Prepared {len(texts)} texts with embeddings")

    # Configure UMAP for dimensionality reduction
    print("\nConfiguring UMAP for 2D projection...")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        verbose=False
    )

    # Configure HDBSCAN for clustering
    print("Configuring HDBSCAN for topic clustering...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eaf',
        prediction_data=True
    )

    # Configure vectorizer for keyword extraction
    print("Configuring keyword extraction...")
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        stop_words="english",
        min_df=5
    )

    # Create BERTopic model
    print(f"\nCreating BERTopic model (target: {TARGET_TOPICS} topics)...")
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=TARGET_TOPICS,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=False  # Faster
    )

    # Fit model
    print("\nRunning topic modeling (this may take 2-5 minutes)...")
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    print(f"\n✓ Discovered {len(topic_info) - 1} topics")  # -1 for outliers

    # Get 2D coordinates from UMAP
    print("Extracting 2D coordinates from UMAP...")
    embeddings_2d = topic_model.umap_model.embedding_

    print(f"✓ Generated 2D coordinates ({embeddings_2d.shape})")

    return topics, topic_model, embeddings_2d, topic_info


def format_topic_label(topic_words: List[tuple]) -> str:
    """Format BERTopic keywords into readable label"""
    # Take top 3-4 keywords
    top_words = [word for word, _ in topic_words[:4]]

    # Capitalize and join with &
    formatted = " & ".join([w.replace("_", " ").title() for w in top_words])

    return formatted


def generate_map_data():
    """Main function to generate map with BERTopic"""
    print("\n" + "="*70)
    print("YONEARTH PODCAST MAP - BERTOPIC SEMANTIC DISCOVERY")
    print("="*70)

    # Step 1: Fetch vectors
    vectors = fetch_vectors_from_pinecone()

    # Step 2: Run BERTopic
    topics, topic_model, embeddings_2d, topic_info = run_bertopic_modeling(vectors)

    # Step 3: Build map data structure
    print("\n" + "="*70)
    print("BUILDING MAP DATA STRUCTURE")
    print("="*70)

    points = []
    episodes_map = {}
    topics_map = {}

    for idx, vector in enumerate(vectors):
        metadata = vector.get('metadata', {})

        # Extract episode info
        episode_id = str(metadata.get('episode_number', 'unknown'))
        episode_title = metadata.get('title', f'Episode {episode_id}')
        chunk_index = metadata.get('chunk_index', 0)
        timestamp = float(chunk_index * 120) if chunk_index else 0
        audio_url = metadata.get('audio_url', '')

        # Get topic assignment
        topic_id = int(topics[idx])

        # Get topic label
        if topic_id == -1:
            topic_label = "General Discussion"
            topic_keywords = []
        else:
            topic_words = topic_model.get_topic(topic_id)
            topic_label = format_topic_label(topic_words)
            topic_keywords = [word for word, _ in topic_words[:10]]

        # Build point
        point = {
            "id": vector['id'],
            "text": metadata.get('text', ''),
            "x": float(embeddings_2d[idx][0]),
            "y": float(embeddings_2d[idx][1]),
            "episode_id": episode_id,
            "episode_title": episode_title,
            "timestamp": timestamp,
            "topic": topic_id,
            "topic_name": topic_label,
            "topic_keywords": topic_keywords
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

        # Track topics
        if topic_id not in topics_map:
            topics_map[topic_id] = {
                "id": topic_id,
                "name": topic_label,
                "keywords": topic_keywords,
                "count": 0
            }
        topics_map[topic_id]["count"] += 1

    # Assign colors to topics (using color palette)
    colors = [
        "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336",
        "#00BCD4", "#FFEB3B", "#795548", "#607D8B", "#E91E63",
        "#3F51B5", "#FFC107", "#009688", "#FF5722", "#8BC34A"
    ]

    topics_list = []
    for topic_id, topic_data in sorted(topics_map.items()):
        color_idx = (topic_id + 1) % len(colors)  # +1 to avoid outlier getting first color
        topics_list.append({
            "id": topic_id,
            "name": topic_data["name"],
            "keywords": topic_data["keywords"],
            "color": colors[color_idx],
            "count": topic_data["count"]
        })

    # Build final export
    export_data = {
        "points": points,
        "episodes": sorted(episodes_map.values(), key=lambda x: x["id"]),
        "topics": topics_list,
        "total_points": len(points),
        "total_topics": len(topics_list),
        "generated_with": "BERTopic",
        "dimensionality_reduction": "UMAP"
    }

    # Save to file
    print(f"\nSaving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(export_data, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nGenerated visualization data:")
    print(f"  - {len(points)} data points")
    print(f"  - {len(episodes_map)} episodes")
    print(f"  - {len(topics_list)} semantic topics")
    print(f"\nDiscovered Topics:")
    for topic in topics_list[:10]:  # Show first 10
        print(f"  • {topic['name']} ({topic['count']} chunks)")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_map_data()
