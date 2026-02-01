#!/usr/bin/env python3
"""
Enrich podcast map JSON with descriptive episode titles from transcript files.

This script reads the episode titles (which include guest names) from the
transcript files and updates the podcast_map_3d_umap_multi_cluster.json
to show "Ep 1: Nancy Tuchman - Loyola U..." instead of just "Episode 1".
"""

import json
import os
import re
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
MAP_FILE = PROJECT_ROOT / "data" / "processed" / "podcast_map_3d_umap_multi_cluster.json"

def extract_guest_and_topic(title: str) -> tuple[str, str]:
    """
    Extract guest name and topic from a title like:
    "Episode 150 – Hunter Lovins on UN's COP 28 Climate Change Summit"

    Returns (guest_name, topic) or (None, None) if can't parse.
    """
    # Remove "Episode X – " prefix
    # Handle various dash types: –, -, —
    match = re.match(r'Episode\s*\d+\s*[–\-—]\s*(.+)', title)
    if not match:
        return None, None

    rest = match.group(1)

    # Try to split on " on " or " - " to separate guest from topic
    # e.g., "Hunter Lovins on UN's COP 28..." -> "Hunter Lovins", "UN's COP 28..."
    # e.g., "Nancy Tuchman – Loyola U..." -> "Nancy Tuchman", "Loyola U..."

    # First try splitting on " on " (most common pattern)
    if ' on ' in rest.lower():
        parts = rest.split(' on ', 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()

    # Then try splitting on " – " or " - "
    for sep in [' – ', ' - ', ' — ']:
        if sep in rest:
            parts = rest.split(sep, 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()

    # If no separator found, the whole thing is likely just the guest name
    return rest.strip(), None

def format_dropdown_title(episode_number: str, title: str, guest_name: str = None) -> str:
    """
    Format episode title for dropdown display.

    Goal: "Ep 1: Nancy Tuchman - Loyola U..." (concise, scannable)

    If we have a full title, extract the key info.
    If we only have guest_name, use that.
    """
    if title and title != f"Episode {episode_number}":
        # We have a descriptive title - extract guest and topic
        guest, topic = extract_guest_and_topic(title)

        if guest and topic:
            # Truncate topic if too long
            if len(topic) > 40:
                topic = topic[:37] + "..."
            return f"Ep {episode_number}: {guest} - {topic}"
        elif guest:
            return f"Ep {episode_number}: {guest}"
        else:
            # Can't parse, use the original title but truncate
            short_title = title.replace(f"Episode {episode_number}", "").strip(" –-—")
            if len(short_title) > 50:
                short_title = short_title[:47] + "..."
            return f"Ep {episode_number}: {short_title}"
    elif guest_name:
        return f"Ep {episode_number}: {guest_name}"
    else:
        return f"Episode {episode_number}"

def load_transcript_metadata(episode_number: str) -> dict:
    """Load metadata from a transcript file."""
    transcript_path = TRANSCRIPTS_DIR / f"episode_{episode_number}.json"

    if not transcript_path.exists():
        return {}

    try:
        with open(transcript_path, 'r') as f:
            data = json.load(f)
        return {
            'title': data.get('title'),
            'guest_name': data.get('guest_name'),
            'url': data.get('url')
        }
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Could not read {transcript_path}: {e}")
        return {}

def main():
    print("=" * 70)
    print("ENRICHING PODCAST MAP WITH EPISODE TITLES")
    print("=" * 70)

    # Check files exist
    if not MAP_FILE.exists():
        print(f"ERROR: Map file not found: {MAP_FILE}")
        print("Please pull from production first:")
        print("  scp claudeuser@152.53.194.214:/var/www/yonearth/data/processed/podcast_map_3d_umap_multi_cluster.json data/processed/")
        return

    if not TRANSCRIPTS_DIR.exists():
        print(f"ERROR: Transcripts directory not found: {TRANSCRIPTS_DIR}")
        return

    # Load map JSON
    print(f"\nLoading map data from {MAP_FILE}...")
    with open(MAP_FILE, 'r') as f:
        map_data = json.load(f)

    episodes = map_data.get('episodes', [])
    print(f"Found {len(episodes)} episodes in map data")

    # Track changes
    updated_count = 0
    unchanged_count = 0

    # Update each episode with proper title
    print("\nUpdating episode titles...")
    for episode in episodes:
        ep_id = episode.get('id', '')
        current_title = episode.get('title', '')

        # Load metadata from transcript
        transcript_meta = load_transcript_metadata(ep_id)

        if transcript_meta:
            # Format the dropdown title
            new_title = format_dropdown_title(
                ep_id,
                transcript_meta.get('title'),
                transcript_meta.get('guest_name')
            )

            if new_title != current_title:
                episode['title'] = new_title
                # Also store the full title for reference
                if transcript_meta.get('title'):
                    episode['full_title'] = transcript_meta.get('title')
                if transcript_meta.get('guest_name'):
                    episode['guest_name'] = transcript_meta.get('guest_name')
                updated_count += 1
                print(f"  ✓ {ep_id}: {current_title} → {new_title}")
            else:
                unchanged_count += 1
        else:
            print(f"  ? No transcript found for episode {ep_id}")
            unchanged_count += 1

    # Save updated map data
    print(f"\nSaving updated map data...")
    with open(MAP_FILE, 'w') as f:
        json.dump(map_data, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Updated: {updated_count} episodes")
    print(f"Unchanged: {unchanged_count} episodes")
    print(f"\nOutput: {MAP_FILE}")
    print("\nNext steps:")
    print("1. Review the changes locally")
    print("2. Deploy to production:")
    print("   scp data/processed/podcast_map_3d_umap_multi_cluster.json claudeuser@152.53.194.214:/var/www/yonearth/data/processed/")

if __name__ == "__main__":
    main()
