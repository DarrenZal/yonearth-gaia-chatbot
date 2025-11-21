"""
Export data from Nomic Atlas to local JSON file
Use this if you're on the free plan and can't access the API programmatically
"""

import os
import json
import nomic
from nomic import AtlasDataset

# Configuration
NOMIC_API_KEY = "nk-w0F9BRRb0EInhoEIKIxqjRAvN5iECGfAV36OtFHWxZQ"
OUTPUT_FILE = "data/processed/podcast_map_data.json"

# Cluster names for YonEarth podcasts (customize these)
CLUSTER_NAMES = [
    "Regenerative Agriculture & Permaculture",
    "Climate Action & Environmental Policy",
    "Sustainable Business & Economics",
    "Community Building & Social Innovation",
    "Ecosystem Restoration & Biodiversity",
    "Renewable Energy & Technology",
    "Holistic Health & Wellbeing"
]

# Cluster colors
CLUSTER_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2"   # Pink
]


def export_nomic_data_manual():
    """
    Manual export instructions if API access is restricted
    """
    print("\n" + "="*70)
    print("MANUAL DATA EXPORT INSTRUCTIONS")
    print("="*70)
    print("\nIf you're on the free Nomic plan, you may need to export data manually:")
    print("\n1. Go to https://atlas.nomic.ai/")
    print("2. Open your project")
    print("3. Click 'Export' or 'Download' button")
    print("4. Choose CSV or JSON format")
    print("5. Make sure to include these fields:")
    print("   - text (the transcript chunk)")
    print("   - x, y (the 2D coordinates)")
    print("   - episode_id")
    print("   - episode_title")
    print("   - timestamp")
    print("   - cluster (topic cluster ID)")
    print("\n6. Save the file to: data/processed/nomic_export.csv")
    print("\n7. Then run: python scripts/convert_nomic_export.py")
    print("\n" + "="*70 + "\n")


def try_programmatic_export(project_name_or_id: str):
    """
    Try to export data programmatically (may fail on free plan)
    """
    try:
        print(f"Attempting to connect to Nomic Atlas...")
        nomic.login(NOMIC_API_KEY)
        print("✓ Authentication successful")

        print(f"\nFetching project: {project_name_or_id}")
        dataset = AtlasDataset(project_name_or_id)
        print(f"✓ Found project: {dataset.name}")

        print("\nFetching map data...")
        map_data = dataset.maps[0]  # First map

        print("Extracting embeddings...")
        embeddings = map_data.embeddings.latent

        print("Extracting metadata...")
        metadata_df = map_data.data.df

        print(f"✓ Loaded {len(metadata_df)} data points")

        # Build the data structure
        points = []
        episodes_map = {}
        clusters_map = {}

        for idx, row in metadata_df.iterrows():
            episode_id = str(row.get("episode_id", f"ep-{idx}"))
            cluster_id = int(row.get("cluster", 0))

            point = {
                "id": str(row.get("id", idx)),
                "text": row.get("text", ""),
                "x": float(embeddings[idx][0]),
                "y": float(embeddings[idx][1]),
                "episode_id": episode_id,
                "episode_title": row.get("episode_title", f"Episode {episode_id}"),
                "timestamp": float(row.get("timestamp", 0)) if row.get("timestamp") else None,
                "cluster": cluster_id,
                "cluster_name": CLUSTER_NAMES[cluster_id % len(CLUSTER_NAMES)]
            }
            points.append(point)

            # Track episodes
            if episode_id not in episodes_map:
                episodes_map[episode_id] = {
                    "id": episode_id,
                    "title": point["episode_title"],
                    "chunk_count": 0,
                    "audio_url": row.get("audio_url", f"/audio/episodes/{episode_id}.mp3")
                }
            episodes_map[episode_id]["chunk_count"] += 1

            # Track clusters
            if cluster_id not in clusters_map:
                clusters_map[cluster_id] = {
                    "id": cluster_id,
                    "name": CLUSTER_NAMES[cluster_id % len(CLUSTER_NAMES)],
                    "color": CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                    "count": 0
                }
            clusters_map[cluster_id]["count"] += 1

        # Build final structure
        export_data = {
            "points": points,
            "episodes": list(episodes_map.values()),
            "clusters": list(clusters_map.values()),
            "total_points": len(points)
        }

        # Save to file
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Successfully exported data to: {OUTPUT_FILE}")
        print(f"  - {len(points)} points")
        print(f"  - {len(episodes_map)} episodes")
        print(f"  - {len(clusters_map)} clusters")

        return True

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nThis might be a free plan limitation.")
        return False


def main():
    print("\n" + "="*70)
    print("NOMIC DATA EXPORT TOOL")
    print("="*70)

    print("\nAttempting programmatic export first...")

    # Ask for project name
    project_input = input("\nEnter your Nomic project name or ID (or press Enter to use 'yonearth-podcast'): ").strip()
    if not project_input:
        project_input = "yonearth-podcast"

    success = try_programmatic_export(project_input)

    if not success:
        print("\n⚠️  Programmatic export failed (likely due to free plan restrictions)")
        export_nomic_data_manual()
        print("\nAlternatively, you can:")
        print("- Upgrade to a paid Nomic plan for API access")
        print("- Use the existing YonEarth episode data with manual clustering")
        print("- Create embeddings locally using OpenAI (you already have embeddings in Pinecone!)")


if __name__ == "__main__":
    main()