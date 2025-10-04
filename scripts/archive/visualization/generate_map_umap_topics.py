"""
Generate podcast map with UMAP dimensionality reduction
Uses UMAP (better global structure) + K-means clustering + GPT-4 for labels
"""

import os
import json
import numpy as np
from pinecone import Pinecone
from sklearn.cluster import KMeans
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import umap

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "yonearth-episodes")
OUTPUT_FILE = "/root/yonearth-gaia-chatbot/data/processed/podcast_map_umap_data.json"

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("ERROR: API keys not set!")
    exit(1)

print(f"✓ Environment loaded")

# Parameters
MAX_VECTORS = 3000  # Reduced for memory safety with UMAP
N_CLUSTERS = 8  # Discover 8 semantic topics
SAMPLE_SIZE = 15  # Chunks to sample per cluster for labeling


def fetch_vectors():
    """Fetch vectors from Pinecone"""
    print("\n" + "="*70)
    print("FETCHING VECTORS FROM PINECONE")
    print("="*70)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    all_vectors = []
    seen_ids = set()
    episode_coverage = set()

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

                ep_id = str(match.get('metadata', {}).get('episode_number', ''))
                if ep_id:
                    episode_coverage.add(ep_id)

        print(f"  Seed {seed}: {len(episode_coverage)} episodes, {len(all_vectors)} vectors")

        if len(episode_coverage) >= 172:
            break

    print(f"\n✓ Fetched {len(all_vectors)} vectors from {len(episode_coverage)} episodes")
    return all_vectors


def reduce_and_cluster(vectors):
    """Reduce to 2D with UMAP and cluster"""
    print("\n" + "="*70)
    print("DIMENSIONALITY REDUCTION (UMAP) & CLUSTERING")
    print("="*70)

    embeddings = np.array([v['values'] for v in vectors])

    print(f"\nReducing {len(embeddings)} embeddings to 2D with UMAP...")
    print("UMAP preserves global structure better than t-SNE")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    print("✓ UMAP reduction complete")

    print(f"\nClustering into {N_CLUSTERS} semantic topics...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    print("✓ Clustering complete")

    return embeddings_2d, cluster_labels


def generate_topic_labels(vectors, cluster_labels):
    """Generate semantic labels using GPT-4"""
    print("\n" + "="*70)
    print("GENERATING SEMANTIC TOPIC LABELS WITH GPT-4")
    print("="*70)

    client = OpenAI(api_key=OPENAI_API_KEY)
    topic_labels = {}

    for cluster_id in range(N_CLUSTERS):
        # Get chunks in this cluster
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

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
{{
  "label": "Topic Label Here",
  "themes": ["theme1", "theme2", "theme3"]
}}"""

        print(f"\n  Analyzing cluster {cluster_id} ({len(cluster_indices)} chunks)...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)
            topic_labels[cluster_id] = {
                "label": result.get("label", f"Topic {cluster_id}"),
                "themes": result.get("themes", [])
            }
            print(f"  ✓ {topic_labels[cluster_id]['label']}")

        except Exception as e:
            print(f"  ⚠ Error generating label: {e}")
            topic_labels[cluster_id] = {
                "label": f"Topic {cluster_id}",
                "themes": []
            }

    return topic_labels


def build_map_data(vectors, embeddings_2d, cluster_labels, topic_labels):
    """Build final map data structure"""
    print("\n" + "="*70)
    print("BUILDING MAP DATA")
    print("="*70)

    # Color palette
    colors = [
        "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336",
        "#00BCD4", "#FFEB3B", "#795548", "#607D8B", "#E91E63"
    ]

    points = []
    episodes_map = {}
    topics_list = []

    # Build points
    for idx, vector in enumerate(vectors):
        metadata = vector.get('metadata', {})
        episode_id = str(metadata.get('episode_number', 'unknown'))
        episode_title = metadata.get('title', f'Episode {episode_id}')
        chunk_index = metadata.get('chunk_index', 0)
        timestamp = float(chunk_index * 120) if chunk_index else 0
        audio_url = metadata.get('audio_url', '')

        cluster_id = int(cluster_labels[idx])
        topic_info = topic_labels.get(cluster_id, {"label": f"Topic {cluster_id}", "themes": []})

        point = {
            "id": vector['id'],
            "text": metadata.get('text', ''),
            "x": float(embeddings_2d[idx][0]),
            "y": float(embeddings_2d[idx][1]),
            "episode_id": episode_id,
            "episode_title": episode_title,
            "timestamp": timestamp,
            "cluster": cluster_id,
            "cluster_name": topic_info["label"],
            "topic": cluster_id,
            "topic_name": topic_info["label"]
        }
        points.append(point)

        if episode_id not in episodes_map:
            episodes_map[episode_id] = {
                "id": episode_id,
                "title": episode_title,
                "chunk_count": 0,
                "audio_url": audio_url or f"https://media.blubrry.com/y_on_earth/yonearth.org/podcast-player/episode-{episode_id}.mp3"
            }
        episodes_map[episode_id]["chunk_count"] += 1

    # Build topics list
    for cluster_id in range(N_CLUSTERS):
        topic_info = topic_labels.get(cluster_id, {"label": f"Topic {cluster_id}", "themes": []})
        count = sum(1 for label in cluster_labels if label == cluster_id)

        topics_list.append({
            "id": cluster_id,
            "name": topic_info["label"],
            "themes": topic_info["themes"],
            "color": colors[cluster_id % len(colors)],
            "count": count
        })

    export_data = {
        "points": points,
        "episodes": sorted(episodes_map.values(), key=lambda x: x["id"]),
        "clusters": topics_list,
        "topics": topics_list,
        "total_points": len(points),
        "total_topics": len(topics_list),
        "generated_with": "UMAP + K-means + GPT-4",
        "reduction_method": "UMAP"
    }

    print(f"\nSaving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(export_data, f, indent=2)

    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nDiscovered Topics (UMAP):")
    for topic in topics_list:
        print(f"  • {topic['name']} ({topic['count']} chunks)")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("="*70 + "\n")


def main():
    vectors = fetch_vectors()
    embeddings_2d, cluster_labels = reduce_and_cluster(vectors)
    topic_labels = generate_topic_labels(vectors, cluster_labels)
    build_map_data(vectors, embeddings_2d, cluster_labels, topic_labels)


if __name__ == "__main__":
    main()
