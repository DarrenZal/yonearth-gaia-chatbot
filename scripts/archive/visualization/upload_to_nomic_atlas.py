"""
Upload YonEarth podcast data to Nomic Atlas
Creates hierarchical topic map visualization
"""

import os
import json
import pandas as pd
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "yonearth-episodes")
NOMIC_API_KEY = "nk-rW0rNvHyLPgQLsxkKRd11scAZsVcZWhEGzz9RkzM3-w"

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not set!")
    exit(1)

print(f"✓ Environment loaded")

MAX_VECTORS = 6000  # Standardized across all maps


def fetch_vectors_with_embeddings():
    """Fetch vectors from Pinecone with full embeddings"""
    print("\n" + "="*70)
    print("FETCHING VECTORS FROM PINECONE (WITH EMBEDDINGS)")
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
            include_values=True  # Critical: include full embeddings
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


def prepare_nomic_dataframe(vectors):
    """Convert vectors to Nomic-compatible DataFrame"""
    print("\n" + "="*70)
    print("PREPARING DATA FOR NOMIC ATLAS")
    print("="*70)

    # Separate embeddings and metadata for memory efficiency
    embeddings_list = []
    metadata_list = []

    for vector in vectors:
        metadata = vector.get('metadata', {})
        embedding = vector.get('values', [])

        embeddings_list.append(embedding)
        metadata_list.append({
            'id': vector['id'],
            'text': metadata.get('text', ''),
            'episode_id': str(metadata.get('episode_number', 'unknown')),
            'episode_title': metadata.get('title', ''),
            'timestamp': metadata.get('chunk_index', 0) * 120,
            'chunk_index': metadata.get('chunk_index', 0)
        })

    # Convert to numpy for embeddings (more memory efficient)
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    metadata_df = pd.DataFrame(metadata_list)

    print(f"✓ Created data: {len(metadata_df)} rows")
    print(f"  Embeddings shape: {embeddings_array.shape}")
    print(f"  Metadata columns: {list(metadata_df.columns)}")

    return embeddings_array, metadata_df


def upload_to_nomic(embeddings, metadata_df):
    """Upload data to Nomic Atlas and create hierarchical map"""
    print("\n" + "="*70)
    print("UPLOADING TO NOMIC ATLAS")
    print("="*70)

    try:
        import nomic
        from nomic import AtlasDataset
    except ImportError:
        print("ERROR: nomic package not installed!")
        print("Run: pip install nomic")
        exit(1)

    # Login to Nomic
    print(f"\nLogging in to Nomic...")
    nomic.login(NOMIC_API_KEY)
    print("✓ Logged in")

    print(f"\nCreating Atlas dataset...")
    print(f"  Name: YonEarth Podcast Hierarchical Topics")
    print(f"  Points: {len(metadata_df)}")
    print(f"  Embedding dimensions: {embeddings.shape[1]}")
    print(f"  Topic modeling: Enabled by default (will auto-generate hierarchical topics)")

    # Create dataset with AtlasDataset
    dataset = AtlasDataset(
        identifier='yonearth-podcast-hierarchical-topics',
        description='YonEarth podcast episodes (6000 points) with automatic hierarchical topic modeling',
        unique_id_field='id',
        is_public=True  # Required for free tier
    )

    # Add embeddings and metadata
    with dataset.wait_for_dataset_lock():
        dataset.add_data(
            data=metadata_df,
            embeddings=embeddings
        )

    print(f"\n✓ Data added successfully!")

    # Create index with topic modeling (enabled by default)
    print(f"\nCreating map index with topic modeling...")
    with dataset.wait_for_dataset_lock():
        dataset.create_index(
            name='YonEarth Hierarchical Topics Map',
            indexed_field='text',
            modality='embedding',
            topic_model=True  # Enable hierarchical topic modeling
        )

    print(f"\n✓ SUCCESS! Map created with hierarchical topics")

    # Get map URL
    map_url = dataset.maps[0].map_link if dataset.maps else f"https://atlas.nomic.ai/data/{dataset.identifier}"

    print(f"\nAtlas Map URL: {map_url}")
    print(f"Embed Code: <iframe src='{map_url}' width='100%' height='800px'></iframe>")
    print(f"\nNote: Topics will be automatically generated. Check 'Show Nomic Topic labels' in View Settings.")

    # Save URL for webpage creation
    with open('/root/yonearth-gaia-chatbot/data/processed/nomic_map_url.txt', 'w') as f:
        f.write(map_url)

    print(f"\n✓ Map URL saved to: /root/yonearth-gaia-chatbot/data/processed/nomic_map_url.txt")

    return map_url


def main():
    print("\n" + "="*70)
    print("NOMIC ATLAS HIERARCHICAL MAP CREATION")
    print("="*70)

    vectors = fetch_vectors_with_embeddings()
    embeddings, metadata_df = prepare_nomic_dataframe(vectors)

    # Save metadata as backup (embeddings too large for CSV)
    csv_path = '/root/yonearth-gaia-chatbot/data/processed/nomic_metadata.csv'
    print(f"\nSaving metadata backup to: {csv_path}")
    metadata_df.to_csv(csv_path, index=False)
    print(f"✓ Saved")

    print("\n" + "="*70)
    print("READY TO UPLOAD")
    print("="*70)
    print("\nAbout to upload to Nomic Atlas (free tier, public dataset)")
    print("This will create a hierarchical topic map visualization.")
    print("\nProceed? (y/n): ", end='')

    response = input().strip().lower()

    if response == 'y':
        map_url = upload_to_nomic(embeddings, metadata_df)
        print("\n" + "="*70)
        print("✓ COMPLETE!")
        print("="*70)
        print(f"\nView your map at: {map_url}")
        print("\nNext steps:")
        print("1. Visit the map URL to see hierarchical topics")
        print("2. Get embed code from Nomic Atlas interface")
        print("3. Create PodcastMapHierarchicalNomic.html with the embed code")
    else:
        print("\nUpload cancelled. Metadata backup saved for later use.")


if __name__ == "__main__":
    main()
