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
LEVEL_1_CLUSTERS = 3   # Broad themes
LEVEL_2_CLUSTERS = 9   # Medium categories (standardized)
LEVEL_3_CLUSTERS = 27  # Detailed topics
SAMPLE_SIZE = 15  # Snippets per cluster for labeling

# Model Configuration (environment overrides)
LEVEL_1_MODEL = os.getenv("HIERARCHICAL_L1_MODEL", "gpt-4o")  # GPT-4o for broad themes
LEVEL_2_MODEL = os.getenv("HIERARCHICAL_L2_MODEL", "gpt-4o-mini")  # GPT-4o-mini for medium
LEVEL_3_MODEL = os.getenv("HIERARCHICAL_L3_MODEL", "gpt-4o-mini")  # GPT-4o-mini for detailed

# API Configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # Exponential backoff base (seconds)
MAX_TOKENS_LABEL = 150  # Strict token limit for label generation
TEMPERATURE = 0.2  # Low temperature for consistency
TFIDF_TERMS = 10  # Number of TF-IDF terms to include per cluster

# Color palettes for each level
LEVEL_1_COLORS = ["#E53935", "#43A047", "#1E88E5", "#FB8C00"]  # Red, Green, Blue, Orange
LEVEL_2_COLORS = [
    "#EF5350", "#EC407A", "#AB47BC", "#7E57C2",
    "#5C6BC0", "#42A5F5", "#29B6F6", "#26C6DA",
    "#26A69A", "#66BB6A"
]
LEVEL_3_COLORS = [
    "#FFCDD2", "#F8BBD0", "#E1BEE7", "#D1C4E9",
    "#C5CAE9", "#BBDEFB", "#B3E5FC", "#B2EBF2",
    "#B2DFDB", "#C8E6C9", "#DCEDC8", "#F0F4C3",
    "#FFF9C4", "#FFECB3", "#FFE0B2", "#FFCCBC",
    "#D7CCC8", "#CFD8DC", "#E0E0E0", "#EEEEEE",
    "#F5F5F5", "#FAFAFA", "#ECEFF1", "#B0BEC5"
]


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
    """Reduce to 2D with UMAP and create 3-level hierarchy"""
    print("\n" + "="*70)
    print("HIERARCHICAL CLUSTERING (3 LEVELS)")
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

    # Create 3 levels of clustering
    print(f"\nCreating hierarchical clusters:")
    print(f"  Level 1: {LEVEL_1_CLUSTERS} broad themes")
    print(f"  Level 2: {LEVEL_2_CLUSTERS} medium categories")
    print(f"  Level 3: {LEVEL_3_CLUSTERS} detailed topics")

    labels = {}

    # Level 1 - Broad themes
    clusterer_l1 = AgglomerativeClustering(n_clusters=LEVEL_1_CLUSTERS, linkage='ward')
    labels['l1'] = clusterer_l1.fit_predict(embeddings_2d)
    print(f"  ✓ Level 1 complete")

    # Level 2 - Medium categories
    clusterer_l2 = AgglomerativeClustering(n_clusters=LEVEL_2_CLUSTERS, linkage='ward')
    labels['l2'] = clusterer_l2.fit_predict(embeddings_2d)
    print(f"  ✓ Level 2 complete")

    # Level 3 - Detailed topics
    clusterer_l3 = AgglomerativeClustering(n_clusters=LEVEL_3_CLUSTERS, linkage='ward')
    labels['l3'] = clusterer_l3.fit_predict(embeddings_2d)
    print(f"  ✓ Level 3 complete")

    return embeddings_2d, labels


def build_hierarchy_tree(labels):
    """Build parent-child relationships between levels"""
    print("\n" + "="*70)
    print("BUILDING HIERARCHY TREE")
    print("="*70)

    # For each point, we have l1, l2, l3 labels
    # Determine which l2 clusters map to which l1 parents
    # Determine which l3 clusters map to which l2 parents

    l2_to_l1 = defaultdict(lambda: defaultdict(int))
    l3_to_l2 = defaultdict(lambda: defaultdict(int))

    n_points = len(labels['l1'])

    for i in range(n_points):
        l1_id = labels['l1'][i]
        l2_id = labels['l2'][i]
        l3_id = labels['l3'][i]

        # Count co-occurrences
        l2_to_l1[l2_id][l1_id] += 1
        l3_to_l2[l3_id][l2_id] += 1

    # Assign parents based on majority vote
    hierarchy = {
        'l1': [{
            'id': i,
            'name': f"Level 1 Theme {i}",
            'children_l2': [],
            'color': LEVEL_1_COLORS[i % len(LEVEL_1_COLORS)]
        } for i in range(LEVEL_1_CLUSTERS)],
        'l2': [{
            'id': i,
            'name': f"Level 2 Category {i}",
            'parent_l1': None,
            'children_l3': [],
            'color': LEVEL_2_COLORS[i % len(LEVEL_2_COLORS)]
        } for i in range(LEVEL_2_CLUSTERS)],
        'l3': [{
            'id': i,
            'name': f"Level 3 Topic {i}",
            'parent_l2': None,
            'color': LEVEL_3_COLORS[i % len(LEVEL_3_COLORS)]
        } for i in range(LEVEL_3_CLUSTERS)]
    }

    # Assign l2 → l1 parents
    for l2_id in range(LEVEL_2_CLUSTERS):
        if l2_id in l2_to_l1:
            parent_l1 = int(max(l2_to_l1[l2_id].items(), key=lambda x: x[1])[0])  # Convert to Python int
            hierarchy['l2'][l2_id]['parent_l1'] = parent_l1
            hierarchy['l1'][parent_l1]['children_l2'].append(int(l2_id))

    # Assign l3 → l2 parents
    for l3_id in range(LEVEL_3_CLUSTERS):
        if l3_id in l3_to_l2:
            parent_l2 = int(max(l3_to_l2[l3_id].items(), key=lambda x: x[1])[0])  # Convert to Python int
            hierarchy['l3'][l3_id]['parent_l2'] = parent_l2
            hierarchy['l2'][parent_l2]['children_l3'].append(int(l3_id))

    print("✓ Hierarchy tree built")
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


def generate_hierarchical_labels(vectors, labels, hierarchy):
    """Generate semantic labels for all levels using GPT-4"""
    print("\n" + "="*70)
    print("GENERATING SEMANTIC LABELS (GPT-4)")
    print("="*70)
    print("⚠️  This will make 39 API calls (3 + 9 + 27 clusters)")
    print("⚠️  Saving progress after each level in case of interruption...")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Create checkpoints directory
    checkpoint_dir = "/root/yonearth-gaia-chatbot/data/processed/hierarchical_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Level 1 - Broad themes
    print(f"\nLevel 1: Generating {LEVEL_1_CLUSTERS} broad theme labels (using {LEVEL_1_MODEL})...")
    for cluster_id in range(LEVEL_1_CLUSTERS):
        cluster_indices = [i for i, label in enumerate(labels['l1']) if label == cluster_id]
        sample_indices = np.random.choice(cluster_indices, min(SAMPLE_SIZE, len(cluster_indices)), replace=False)

        # Get short snippets (100 chars each, not 200)
        sample_texts = [vectors[i].get('metadata', {}).get('text', '')[:100] for i in sample_indices]

        # Extract TF-IDF terms from all cluster texts
        all_cluster_texts = [vectors[i].get('metadata', {}).get('text', '') for i in cluster_indices]
        tfidf_terms = extract_tfidf_terms(all_cluster_texts, TFIDF_TERMS)

        prompt = f"""Analyze this BROAD theme cluster from podcast transcripts.

Sample excerpts (10-15 snippets):
{chr(10).join([f'{i+1}. "{text}..."' for i, text in enumerate(sample_texts[:15])])}

Top keywords (TF-IDF):
{', '.join(tfidf_terms)}

Provide a HIGH-LEVEL theme label (2-4 words) and 3-5 broad concepts.

STRICT JSON FORMAT:
{{
  "label": "Theme Label Here",
  "themes": ["concept1", "concept2", "concept3"]
}}"""

        try:
            result = call_openai_with_retry(client, LEVEL_1_MODEL, prompt)
            hierarchy['l1'][cluster_id]['name'] = result.get("label", f"Theme {cluster_id}")
            hierarchy['l1'][cluster_id]['themes'] = result.get("themes", [])
            print(f"  ✓ L1-{cluster_id}: {hierarchy['l1'][cluster_id]['name']}")
        except Exception as e:
            hierarchy['l1'][cluster_id]['name'] = f"Theme {cluster_id}"
            hierarchy['l1'][cluster_id]['themes'] = []
            print(f"  ❌ L1-{cluster_id}: Failed - using fallback")

    # Save Level 1 checkpoint
    with open(f"{checkpoint_dir}/level1_complete.json", 'w') as f:
        json.dump(hierarchy, f, indent=2)
    print("  💾 Level 1 saved to checkpoint")

    # Level 2 - Medium categories
    print(f"\nLevel 2: Generating {LEVEL_2_CLUSTERS} category labels (using {LEVEL_2_MODEL})...")
    for cluster_id in range(LEVEL_2_CLUSTERS):
        cluster_indices = [i for i, label in enumerate(labels['l2']) if label == cluster_id]
        sample_indices = np.random.choice(cluster_indices, min(SAMPLE_SIZE, len(cluster_indices)), replace=False)

        # Get short snippets (100 chars each)
        sample_texts = [vectors[i].get('metadata', {}).get('text', '')[:100] for i in sample_indices]

        # Extract TF-IDF terms from cluster
        all_cluster_texts = [vectors[i].get('metadata', {}).get('text', '') for i in cluster_indices]
        tfidf_terms = extract_tfidf_terms(all_cluster_texts, TFIDF_TERMS)

        parent_l1 = hierarchy['l2'][cluster_id]['parent_l1']
        parent_name = hierarchy['l1'][parent_l1]['name'] if parent_l1 is not None else "Unknown"

        prompt = f"""Analyze this MEDIUM-LEVEL category cluster from podcast transcripts.

Parent Theme: {parent_name}

Sample excerpts (10-15 snippets):
{chr(10).join([f'{i+1}. "{text}..."' for i, text in enumerate(sample_texts[:15])])}

Top keywords (TF-IDF):
{', '.join(tfidf_terms)}

Provide a category label (3-6 words) and 3-5 key aspects within "{parent_name}".

STRICT JSON FORMAT:
{{
  "label": "Category Label Here",
  "themes": ["aspect1", "aspect2", "aspect3"]
}}"""

        try:
            result = call_openai_with_retry(client, LEVEL_2_MODEL, prompt)
            hierarchy['l2'][cluster_id]['name'] = result.get("label", f"Category {cluster_id}")
            hierarchy['l2'][cluster_id]['themes'] = result.get("themes", [])
            print(f"  ✓ L2-{cluster_id}: {hierarchy['l2'][cluster_id]['name']} [parent: {parent_name}]")
        except Exception as e:
            hierarchy['l2'][cluster_id]['name'] = f"Category {cluster_id}"
            hierarchy['l2'][cluster_id]['themes'] = []
            print(f"  ❌ L2-{cluster_id}: Failed - using fallback [parent: {parent_name}]")

    # Save Level 2 checkpoint
    with open(f"{checkpoint_dir}/level2_complete.json", 'w') as f:
        json.dump(hierarchy, f, indent=2)
    print("  💾 Level 2 saved to checkpoint")

    # Level 3 - Detailed topics
    print(f"\nLevel 3: Generating {LEVEL_3_CLUSTERS} detailed topic labels (using {LEVEL_3_MODEL})...")
    print("  (This will take ~2-3 minutes for 27 clusters with retry logic)")
    for cluster_id in range(LEVEL_3_CLUSTERS):
        cluster_indices = [i for i, label in enumerate(labels['l3']) if label == cluster_id]
        sample_indices = np.random.choice(cluster_indices, min(SAMPLE_SIZE, len(cluster_indices)), replace=False)

        # Get short snippets (100 chars each)
        sample_texts = [vectors[i].get('metadata', {}).get('text', '')[:100] for i in sample_indices]

        # Extract TF-IDF terms from cluster
        all_cluster_texts = [vectors[i].get('metadata', {}).get('text', '') for i in cluster_indices]
        tfidf_terms = extract_tfidf_terms(all_cluster_texts, TFIDF_TERMS)

        parent_l2 = hierarchy['l3'][cluster_id]['parent_l2']
        parent_name = hierarchy['l2'][parent_l2]['name'] if parent_l2 is not None else "Unknown"

        prompt = f"""Analyze this SPECIFIC topic cluster from podcast transcripts.

Parent Category: {parent_name}

Sample excerpts (10-15 snippets):
{chr(10).join([f'{i+1}. "{text}..."' for i, text in enumerate(sample_texts[:15])])}

Top keywords (TF-IDF):
{', '.join(tfidf_terms)}

Provide a specific topic label (4-8 words) and 2-4 detailed themes within "{parent_name}".

STRICT JSON FORMAT:
{{
  "label": "Specific Topic Label Here",
  "themes": ["theme1", "theme2"]
}}"""

        try:
            result = call_openai_with_retry(client, LEVEL_3_MODEL, prompt)
            hierarchy['l3'][cluster_id]['name'] = result.get("label", f"Topic {cluster_id}")
            hierarchy['l3'][cluster_id]['themes'] = result.get("themes", [])
            print(f"  ✓ L3-{cluster_id+1}/{LEVEL_3_CLUSTERS}: {hierarchy['l3'][cluster_id]['name']}")
        except Exception as e:
            hierarchy['l3'][cluster_id]['name'] = f"Topic {cluster_id}"
            hierarchy['l3'][cluster_id]['themes'] = []
            print(f"  ❌ L3-{cluster_id+1}/{LEVEL_3_CLUSTERS}: Failed - using fallback")

    # Save Level 3 checkpoint
    with open(f"{checkpoint_dir}/level3_complete.json", 'w') as f:
        json.dump(hierarchy, f, indent=2)
    print("  💾 Level 3 saved to checkpoint")

    return hierarchy


def build_map_data(vectors, embeddings_2d, labels, hierarchy):
    """Build final hierarchical map data structure"""
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

        cluster_l1 = int(labels['l1'][idx])
        cluster_l2 = int(labels['l2'][idx])
        cluster_l3 = int(labels['l3'][idx])

        point = {
            "id": vector['id'],
            "text": metadata.get('text', ''),
            "x": float(embeddings_2d[idx][0]),
            "y": float(embeddings_2d[idx][1]),
            "episode_id": episode_id,
            "episode_title": episode_title,
            "timestamp": float(timestamp),
            "cluster_l1": int(cluster_l1),
            "cluster_l2": int(cluster_l2),
            "cluster_l3": int(cluster_l3),
            "cluster_l1_name": hierarchy['l1'][cluster_l1]['name'],
            "cluster_l2_name": hierarchy['l2'][cluster_l2]['name'],
            "cluster_l3_name": hierarchy['l3'][cluster_l3]['name']
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

    export_data = {
        "points": points,
        "episodes": sorted(episodes_map.values(), key=lambda x: x["id"]),
        "hierarchy": hierarchy,
        "total_points": len(points),
        "levels": {
            "l1": LEVEL_1_CLUSTERS,
            "l2": LEVEL_2_CLUSTERS,
            "l3": LEVEL_3_CLUSTERS
        },
        "generated_with": "Hierarchical Agglomerative Clustering + UMAP + GPT-4",
        "default_level": "l2"
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
    print("✓ SUCCESS! HIERARCHICAL MAP CREATED")
    print("="*70)
    print(f"\nLevel 1 - Broad Themes ({LEVEL_1_CLUSTERS}):")
    for cluster in hierarchy['l1']:
        print(f"  • {cluster['name']}")
    print(f"\nLevel 2 - Categories ({LEVEL_2_CLUSTERS}):")
    for cluster in hierarchy['l2']:
        parent = hierarchy['l1'][cluster['parent_l1']]['name'] if cluster['parent_l1'] is not None else "?"
        print(f"  • {cluster['name']} [→ {parent}]")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("="*70 + "\n")


def main():
    vectors = fetch_vectors()
    embeddings_2d, labels = reduce_and_cluster_hierarchical(vectors)
    hierarchy = build_hierarchy_tree(labels)
    hierarchy = generate_hierarchical_labels(vectors, labels, hierarchy)
    build_map_data(vectors, embeddings_2d, labels, hierarchy)


if __name__ == "__main__":
    main()
