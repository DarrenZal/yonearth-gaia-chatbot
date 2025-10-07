"""
Generate hierarchical topic map with multi-level clustering
3 Levels: Broad (4) → Medium (10) → Detailed (24)
"""

import os
import json
import numpy as np
import time
from pinecone import Pinecone
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import umap
from collections import defaultdict

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "yonearth-episodes")
OUTPUT_FILE = "/root/yonearth-gaia-chatbot/data/processed/podcast_map_hierarchical_data.json"

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("ERROR: API keys not set!")
    exit(1)

print(f"✓ Environment loaded")

# Hierarchical Parameters
MAX_VECTORS = int(os.getenv("HIERARCHICAL_MAX_VECTORS", "3000"))  # Reduced to 3000 for memory constraints
# Generate 18 levels of clustering for semantic zoom (1-18 clusters)
CLUSTER_LEVELS = list(range(1, 19))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
SAMPLE_SIZE = 15  # Snippets per cluster for labeling

# Model Configuration (environment overrides)
LABELING_MODEL = os.getenv("HIERARCHICAL_MODEL", "gpt-4o-mini")  # Use gpt-4o-mini for all levels

# API Configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Exponential backoff base (seconds)
MAX_TOKENS_LABEL = 150  # Strict token limit for label generation
TEMPERATURE = 0.2  # Low temperature for consistency
TFIDF_TERMS = 10  # Number of TF-IDF terms to include per cluster

# Generate color palette dynamically for up to 18 clusters
def generate_color_palette(n_colors):
    """Generate distinct colors using HSL color space"""
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        # Vary saturation and lightness for visual distinction
        saturation = 0.6 + (i % 3) * 0.15
        lightness = 0.5 + (i % 2) * 0.1
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


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


def reduce_and_cluster_hierarchical(vectors):
    """Reduce to 2D with UMAP and create 18 cluster levels for semantic zoom"""
    print("\n" + "="*70)
    print("HIERARCHICAL CLUSTERING (18 LEVELS FOR SEMANTIC ZOOM)")
    print("="*70)

    embeddings = np.array([v['values'] for v in vectors])

    print(f"\nReducing {len(embeddings)} embeddings to 2D with UMAP...")
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

    # Create 18 levels of clustering (1-18 clusters)
    print(f"\nCreating 18 cluster levels for semantic zoom:")
    labels = {}

    for n_clusters in CLUSTER_LEVELS:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels[f'c{n_clusters}'] = clusterer.fit_predict(embeddings_2d)
        print(f"  ✓ Level c{n_clusters}: {n_clusters} clusters")

    return embeddings_2d, labels


def build_hierarchy_structure(labels):
    """Build cluster metadata structure for all 18 levels"""
    print("\n" + "="*70)
    print("BUILDING CLUSTER STRUCTURES")
    print("="*70)

    hierarchy = {}

    for n_clusters in CLUSTER_LEVELS:
        level_key = f'c{n_clusters}'
        colors = generate_color_palette(n_clusters)

        hierarchy[level_key] = [{
            'id': i,
            'name': f"Cluster {i+1}",  # Placeholder, will be labeled later
            'color': colors[i]
        } for i in range(n_clusters)]

    print(f"✓ Created structures for {len(CLUSTER_LEVELS)} cluster levels")
    return hierarchy


def extract_tfidf_terms(texts, n_terms=10):
    """Extract top TF-IDF terms from cluster texts"""
    try:
        if len(texts) == 0:
            return []
        vectorizer = TfidfVectorizer(max_features=n_terms, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        return feature_names.tolist()
    except Exception as e:
        print(f"    ⚠ TF-IDF extraction failed: {e}")
        return []


def call_openai_with_retry(client, model, prompt, max_retries=MAX_RETRIES):
    """Call OpenAI API with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_LABEL
            )
            result = json.loads(response.choices[0].message.content)

            # Validate strict schema: { "label": string, "themes": string[] }
            if not isinstance(result.get("label"), str):
                raise ValueError("Invalid schema: 'label' must be a string")
            if not isinstance(result.get("themes"), list):
                raise ValueError("Invalid schema: 'themes' must be a list")
            if not all(isinstance(t, str) for t in result.get("themes", [])):
                raise ValueError("Invalid schema: 'themes' must contain only strings")

            return result

        except Exception as e:
            wait_time = RETRY_DELAY_BASE ** attempt
            if attempt < max_retries - 1:
                print(f"    ⚠ API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"    ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    ❌ Failed after {max_retries} attempts: {e}")
                raise

    return {"label": "Unknown Topic", "themes": []}


def generate_labels_for_key_levels(vectors, labels, hierarchy):
    """Generate semantic labels for key levels only (to save API calls)"""
    print("\n" + "="*70)
    print("GENERATING SEMANTIC LABELS (GPT-4o-mini)")
    print("="*70)

    # Label ALL 18 levels for complete semantic zoom experience
    # Total API calls: 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18 = 171 calls
    LABELED_LEVELS = list(range(1, 19))  # All levels from 1 to 18

    print(f"⚠️  Labeling ALL {len(LABELED_LEVELS)} levels for semantic zoom")
    print(f"⚠️  Total clusters to label: {sum(LABELED_LEVELS)} (~3-5 minutes with gpt-4o-mini)")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Create checkpoints directory
    checkpoint_dir = "/root/yonearth-gaia-chatbot/data/processed/hierarchical_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for n_clusters in LABELED_LEVELS:
        level_key = f'c{n_clusters}'
        print(f"\nLabeling level {level_key}: {n_clusters} clusters (using {LABELING_MODEL})...")

        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(labels[level_key]) if label == cluster_id]

            if len(cluster_indices) == 0:
                print(f"  ⚠ {level_key}-{cluster_id}: No points in cluster, skipping")
                continue

            sample_indices = np.random.choice(cluster_indices, min(SAMPLE_SIZE, len(cluster_indices)), replace=False)

            # Get short snippets
            sample_texts = [vectors[i].get('metadata', {}).get('text', '')[:100] for i in sample_indices]

            # Extract TF-IDF terms from cluster
            all_cluster_texts = [vectors[i].get('metadata', {}).get('text', '') for i in cluster_indices]
            tfidf_terms = extract_tfidf_terms(all_cluster_texts, TFIDF_TERMS)

            # Determine granularity descriptor
            if n_clusters <= 3:
                granularity = "VERY BROAD theme"
                label_length = "2-3 words"
            elif n_clusters <= 9:
                granularity = "MEDIUM-LEVEL category"
                label_length = "3-5 words"
            else:
                granularity = "SPECIFIC topic"
                label_length = "4-8 words"

            prompt = f"""Analyze this {granularity} cluster from podcast transcripts.

Sample excerpts (10-15 snippets):
{chr(10).join([f'{i+1}. "{text}..."' for i, text in enumerate(sample_texts[:15])])}

Top keywords (TF-IDF):
{', '.join(tfidf_terms)}

Provide a concise label ({label_length}) and 2-4 key themes.

STRICT JSON FORMAT:
{{
  "label": "Cluster Label Here",
  "themes": ["theme1", "theme2"]
}}"""

            try:
                result = call_openai_with_retry(client, LABELING_MODEL, prompt)
                hierarchy[level_key][cluster_id]['name'] = result.get("label", f"Cluster {cluster_id+1}")
                hierarchy[level_key][cluster_id]['themes'] = result.get("themes", [])
                print(f"  ✓ {level_key}-{cluster_id}: {hierarchy[level_key][cluster_id]['name']}")
            except Exception as e:
                hierarchy[level_key][cluster_id]['name'] = f"Cluster {cluster_id+1}"
                hierarchy[level_key][cluster_id]['themes'] = []
                print(f"  ❌ {level_key}-{cluster_id}: Failed - using fallback")

        # Save checkpoint after each level
        with open(f"{checkpoint_dir}/{level_key}_complete.json", 'w') as f:
            json.dump(hierarchy, f, indent=2)
        print(f"  💾 {level_key} saved to checkpoint")

    print("\n✓ Labeled levels complete!")
    return hierarchy


def build_map_data(vectors, embeddings_2d, labels, hierarchy):
    """Build final hierarchical map data structure with all 18 cluster levels"""
    print("\n" + "="*70)
    print("BUILDING HIERARCHICAL MAP DATA")
    print("="*70)

    points = []
    episodes_map = {}

    for idx, vector in enumerate(vectors):
        metadata = vector.get('metadata', {})
        episode_id = str(metadata.get('episode_number', 'unknown'))
        episode_title = metadata.get('title', f'Episode {episode_id}')
        chunk_index = metadata.get('chunk_index', 0)
        timestamp = float(chunk_index * 120) if chunk_index else 0
        audio_url = metadata.get('audio_url', '')

        # Create point with cluster assignments for ALL 18 levels
        point = {
            "id": vector['id'],
            "text": metadata.get('text', ''),
            "x": float(embeddings_2d[idx][0]),
            "y": float(embeddings_2d[idx][1]),
            "episode_id": episode_id,
            "episode_title": episode_title,
            "timestamp": float(timestamp),
        }

        # Add cluster IDs and names for all 18 levels
        for n_clusters in CLUSTER_LEVELS:
            level_key = f'c{n_clusters}'
            cluster_id = int(labels[level_key][idx])
            point[f'cluster_{level_key}'] = cluster_id
            point[f'cluster_{level_key}_name'] = hierarchy[level_key][cluster_id]['name']

        points.append(point)

        if episode_id not in episodes_map:
            episodes_map[episode_id] = {
                "id": episode_id,
                "title": episode_title,
                "chunk_count": 0,
                "audio_url": audio_url or f"https://media.blubrry.com/y_on_earth/yonearth.org/podcast-player/episode-{episode_id}.mp3"
            }
        episodes_map[episode_id]["chunk_count"] += 1

    # Build levels metadata
    levels_metadata = {f'c{n}': n for n in CLUSTER_LEVELS}

    export_data = {
        "points": points,
        "episodes": sorted(episodes_map.values(), key=lambda x: x["id"]),
        "hierarchy": hierarchy,
        "total_points": len(points),
        "cluster_levels": CLUSTER_LEVELS,
        "levels": levels_metadata,
        "generated_with": "18-Level Hierarchical Clustering + UMAP + GPT-4o-mini (Semantic Zoom)",
        "default_level": "c9"  # Start with 9 clusters (medium detail)
    }

    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    export_data = convert_numpy(export_data)

    print(f"\nSaving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(export_data, f, indent=2)

    print("\n" + "="*70)
    print("✓ SUCCESS! SEMANTIC ZOOM MAP CREATED")
    print("="*70)
    print(f"\n18 Cluster Levels Generated for Semantic Zoom:")
    print(f"  • Levels: {CLUSTER_LEVELS}")
    print(f"  • Default view: c9 (9 clusters)")
    print(f"\nSample clusters from c9 (Medium detail):")
    for i, cluster in enumerate(hierarchy['c9'][:5]):  # Show first 5
        print(f"  • {cluster['name']}")
    if len(hierarchy['c9']) > 5:
        print(f"  ... and {len(hierarchy['c9']) - 5} more")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"\n🎯 Zoom in/out to smoothly transition between 1-18 clusters!")
    print("="*70 + "\n")


def main():
    vectors = fetch_vectors()
    embeddings_2d, labels = reduce_and_cluster_hierarchical(vectors)
    hierarchy = build_hierarchy_structure(labels)
    hierarchy = generate_labels_for_key_levels(vectors, labels, hierarchy)
    build_map_data(vectors, embeddings_2d, labels, hierarchy)


if __name__ == "__main__":
    main()
