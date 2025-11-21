"""
Export 2D projections and topic data from Nomic Atlas
Creates a JSON file for custom visualization with Nomic's algorithms
"""

import os
import json
import nomic
from nomic import AtlasDataset

# Configuration
NOMIC_API_KEY = "nk-rW0rNvHyLPgQLsxkKRd11scAZsVcZWhEGzz9RkzM3-w"
DATASET_ID = 'yonearth-podcast-hierarchical-topics'
OUTPUT_FILE = '/root/yonearth-gaia-chatbot/data/processed/nomic_projections.json'

print("="*70)
print("EXPORTING NOMIC ATLAS PROJECTIONS")
print("="*70)

# Login
print("\nLogging in to Nomic...")
nomic.login(NOMIC_API_KEY)
print("✓ Logged in")

# Load dataset
print(f"\nLoading dataset: {DATASET_ID}")
dataset = AtlasDataset(DATASET_ID)
print(f"✓ Dataset loaded")
print(f"  Total datums: {dataset.total_datums}")

# Get the first map
if len(dataset.maps) == 0:
    print("ERROR: No maps found!")
    exit(1)

map_obj = dataset.maps[0]
print(f"✓ Map found")

# Get data with projections from the map
print("\nFetching data from map...")
try:
    # Try to get projections directly from map
    projections_df = map_obj.embeddings.projected
    print(f"✓ Projections loaded: {len(projections_df)} rows")
    print(f"  Columns: {list(projections_df.columns)}")
    data_df = projections_df
except Exception as e:
    print(f"Error getting projections: {e}")
    print("Trying alternative method...")
    # Alternative: get all data without specifying ids
    try:
        data_df = map_obj.data.df
        print(f"✓ Data loaded: {len(data_df)} rows")
        print(f"  Columns: {list(data_df.columns)}")
    except Exception as e2:
        print(f"Error: {e2}")
        exit(1)

# Check for projection columns
projection_cols = [col for col in data_df.columns if 'x' in col.lower() or 'y' in col.lower() or 'projection' in col.lower()]
print(f"  Projection columns found: {projection_cols}")

# Load original metadata
print("\nLoading original metadata...")
import pandas as pd
metadata_path = '/root/yonearth-gaia-chatbot/data/processed/nomic_metadata.csv'
metadata_df = pd.read_csv(metadata_path)
print(f"✓ Metadata loaded: {len(metadata_df)} rows")

# Merge projections with metadata
print("Merging with projections...")
data_df = data_df.merge(metadata_df, on='id', how='left')
print(f"✓ Merged data shape: {data_df.shape}")

# Get topics if available
print("\nFetching topics...")
try:
    topics_df = map_obj.topics.df
    print(f"✓ Topics loaded: {len(topics_df)} rows")
    print(f"  Topic columns: {list(topics_df.columns)}")

    # Merge topics with data
    if 'id' in data_df.columns and 'id' in topics_df.columns:
        data_df = data_df.merge(topics_df, on='id', how='left', suffixes=('', '_topic'))
        print(f"✓ Merged topics with data")
except Exception as e:
    print(f"⚠️  Could not load topics: {e}")

# Convert to export format
print("\nConverting to JSON...")
export_data = {
    'points': [],
    'metadata': {
        'source': 'nomic_atlas',
        'dataset': DATASET_ID,
        'total_points': len(data_df),
        'algorithm': 'nomic_umap'
    }
}

for idx, row in data_df.iterrows():
    # Find x, y coordinates (try different column names)
    x = row.get('x', row.get('projection_x', row.get('umap_x', 0)))
    y = row.get('y', row.get('projection_y', row.get('umap_y', 0)))

    point = {
        'id': row.get('id', row.get('id_', str(idx))),
        'x': float(x) if x is not None else 0,
        'y': float(y) if y is not None else 0,
        'text': row.get('text', ''),
        'episode_id': str(row.get('episode_id', 'unknown')),
        'episode_title': row.get('episode_title', ''),
    }

    # Add topic data if available
    for col in data_df.columns:
        if 'topic_depth' in col:
            point[col] = row[col]

    export_data['points'].append(point)

    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(data_df)} points...")

# Save
print(f"\nSaving to: {OUTPUT_FILE}")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(export_data, f, indent=2)

file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"✓ Saved! ({file_size:.2f} MB)")

print("\n" + "="*70)
print("✓ EXPORT COMPLETE")
print("="*70)
print(f"\nExported {len(export_data['points'])} points")
print("\nReady to build custom visualization with Nomic's algorithms!")
