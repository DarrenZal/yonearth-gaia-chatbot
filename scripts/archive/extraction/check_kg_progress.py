#!/usr/bin/env python3
"""
Check Knowledge Graph Extraction Progress

Quick script to monitor the progress of episodes 0-43 extraction.
"""

import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
entities_dir = project_root / "data" / "knowledge_graph" / "entities"

def check_progress(start=0, end=43):
    """Check extraction progress"""

    print("=" * 70)
    print("KNOWLEDGE GRAPH EXTRACTION PROGRESS")
    print("=" * 70)

    processed = []
    missing = []
    total_entities = 0
    total_relationships = 0
    total_chunks = 0

    for ep_num in range(start, end + 1):
        file_path = entities_dir / f"episode_{ep_num}_extraction.json"

        if file_path.exists():
            try:
                with open(file_path) as f:
                    data = json.load(f)

                processed.append(ep_num)
                chunks = data.get('total_chunks', 0)
                entities = data.get('summary_stats', {}).get('total_entities', 0)
                rels = data.get('summary_stats', {}).get('total_relationships', 0)

                total_chunks += chunks
                total_entities += entities
                total_relationships += rels

                print(f"✓ Episode {ep_num:3d}: {chunks:2d} chunks, {entities:3d} entities, {rels:3d} relationships")
            except Exception as e:
                print(f"✗ Episode {ep_num:3d}: Error reading file - {e}")
                missing.append(ep_num)
        else:
            missing.append(ep_num)
            print(f"⧖ Episode {ep_num:3d}: Not yet processed")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Episodes Processed: {len(processed)}/{end - start + 1}")
    print(f"Episodes Remaining: {len(missing)}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Total Entities: {total_entities}")
    print(f"Total Relationships: {total_relationships}")

    if missing:
        print(f"\nStill to process: {', '.join(map(str, missing[:10]))}" +
              (" ..." if len(missing) > 10 else ""))

    print("=" * 70)

    return {
        "processed": len(processed),
        "total": end - start + 1,
        "missing": len(missing),
        "total_entities": total_entities,
        "total_relationships": total_relationships
    }

if __name__ == "__main__":
    check_progress(0, 43)
