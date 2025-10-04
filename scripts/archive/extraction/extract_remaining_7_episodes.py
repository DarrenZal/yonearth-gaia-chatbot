#!/usr/bin/env python3
"""
Extract knowledge graph data from the final 7 remaining episodes
Using the new structured outputs implementation
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_graph.extractors.entity_extractor import EntityExtractor
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor
from src.knowledge_graph.extractors.chunking import chunk_transcript

# Episodes to extract
EPISODES_TO_EXTRACT = [9, 18, 22, 23, 30, 35, 75]

# Paths
TRANSCRIPTS_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data/transcripts")
OUTPUT_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph")

def load_transcript(episode_num: int) -> str:
    """Load transcript for an episode"""
    transcript_file = TRANSCRIPTS_DIR / f"episode_{episode_num}.json"

    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_file}")

    with open(transcript_file, 'r') as f:
        data = json.load(f)

    transcript = data.get('full_transcript', '')

    if not transcript or len(transcript) < 100:
        raise ValueError(f"Episode {episode_num} has invalid transcript (too short)")

    return transcript


def extract_episode(episode_num: int, entity_extractor: EntityExtractor,
                    relationship_extractor: RelationshipExtractor):
    """Extract entities and relationships from an episode"""

    print(f"\n{'='*60}")
    print(f"Processing Episode {episode_num}")
    print(f"{'='*60}")

    # Load transcript
    try:
        transcript = load_transcript(episode_num)
        print(f"✅ Loaded transcript ({len(transcript):,} characters)")
    except Exception as e:
        print(f"❌ Error loading transcript: {e}")
        return False

    # Chunk the transcript
    print(f"Chunking transcript...")
    chunks = chunk_transcript(
        transcript=transcript,
        chunk_size=800,
        overlap=100
    )
    print(f"✅ Created {len(chunks)} chunks")

    # Extract entities and relationships from each chunk
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks, 1):
        chunk_id = f"ep{episode_num}_chunk{chunk['chunk_index']}"

        print(f"\n  Chunk {i}/{len(chunks)} (tokens: {chunk['token_count']})")

        # Extract entities
        try:
            entity_result = entity_extractor.extract_entities(
                text=chunk['text'],
                episode_number=episode_num,
                chunk_id=chunk_id
            )

            print(f"    ✅ Entities: {len(entity_result.entities)}")
            all_entities.extend([e.model_dump() for e in entity_result.entities])

        except Exception as e:
            print(f"    ❌ Entity extraction error: {e}")
            continue

        # Extract relationships
        try:
            # Convert entity results to dict format for relationship extractor
            entities_for_rel = [
                {"name": e.name, "type": e.type}
                for e in entity_result.entities
            ]

            relationship_result = relationship_extractor.extract_relationships(
                text=chunk['text'],
                entities=entities_for_rel,
                episode_number=episode_num,
                chunk_id=chunk_id
            )

            print(f"    ✅ Relationships: {len(relationship_result.relationships)}")
            all_relationships.extend([r.model_dump() for r in relationship_result.relationships])

        except Exception as e:
            print(f"    ❌ Relationship extraction error: {e}")
            continue

    # Save results
    entities_file = OUTPUT_DIR / "entities" / f"episode_{episode_num}_extraction.json"
    relationships_file = OUTPUT_DIR / "relationships" / f"episode_{episode_num}_extraction.json"

    # Ensure directories exist
    entities_file.parent.mkdir(parents=True, exist_ok=True)
    relationships_file.parent.mkdir(parents=True, exist_ok=True)

    # Save entities
    with open(entities_file, 'w') as f:
        json.dump({
            "episode_number": episode_num,
            "entities": all_entities,
            "extraction_metadata": {
                "total_chunks": len(chunks),
                "total_entities": len(all_entities),
                "extraction_method": "structured_outputs"
            }
        }, f, indent=2)

    # Save relationships
    with open(relationships_file, 'w') as f:
        json.dump({
            "episode_number": episode_num,
            "relationships": all_relationships,
            "extraction_metadata": {
                "total_chunks": len(chunks),
                "total_relationships": len(all_relationships),
                "extraction_method": "structured_outputs"
            }
        }, f, indent=2)

    print(f"\n✅ Episode {episode_num} complete!")
    print(f"   Entities: {len(all_entities)}")
    print(f"   Relationships: {len(all_relationships)}")
    print(f"   Files saved to: {entities_file.parent}")

    return True


def main():
    """Main extraction process"""

    print("🚀 Starting extraction of final 7 episodes with structured outputs")
    print(f"Episodes to extract: {EPISODES_TO_EXTRACT}")

    # Initialize extractors
    print("\nInitializing extractors...")
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()
    print("✅ Extractors ready")

    # Track results
    successful = []
    failed = []

    # Process each episode
    start_time = time.time()

    for episode_num in EPISODES_TO_EXTRACT:
        try:
            success = extract_episode(episode_num, entity_extractor, relationship_extractor)
            if success:
                successful.append(episode_num)
            else:
                failed.append(episode_num)
        except Exception as e:
            print(f"\n❌ Fatal error processing episode {episode_num}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(episode_num)

    # Summary
    elapsed_time = time.time() - start_time

    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total episodes processed: {len(EPISODES_TO_EXTRACT)}")
    print(f"Successful: {len(successful)} - {successful}")
    print(f"Failed: {len(failed)} - {failed}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")

    if failed:
        print(f"\n⚠️  {len(failed)} episodes failed. Review errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All episodes extracted successfully with structured outputs!")
        sys.exit(0)


if __name__ == "__main__":
    main()
