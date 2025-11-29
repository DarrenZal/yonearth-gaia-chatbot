#!/usr/bin/env python3
"""
Extract knowledge graph from episode transcripts.

Uses EntityExtractor and RelationshipExtractor to process all episodes
and save to data/knowledge_graph/entities/ for later graph building.

Usage:
    python scripts/extract_knowledge_from_episodes.py

    # Or process only specific episodes:
    python scripts/extract_knowledge_from_episodes.py --episodes 120,122,124

    # Or process a range:
    python scripts/extract_knowledge_from_episodes.py --start 0 --end 10
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.extractors.entity_extractor import EntityExtractor
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor
from src.knowledge_graph.extractors.chunking import chunk_transcript

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Token-aware chunking wrapper using chunk_transcript."""
    chunks = chunk_transcript(
        transcript=text,
        chunk_size=chunk_size,
        overlap=chunk_overlap
    )
    return [chunk["text"] for chunk in chunks]


def extract_from_episode(
    episode_file: Path,
    entity_extractor: EntityExtractor,
    relationship_extractor: RelationshipExtractor,
    output_dir: Path
) -> Optional[dict]:
    """
    Extract entities and relationships from a single episode.

    Args:
        episode_file: Path to episode transcript JSON
        entity_extractor: EntityExtractor instance
        relationship_extractor: RelationshipExtractor instance
        output_dir: Directory to save extraction results

    Returns:
        Extraction statistics dict or None if failed
    """
    try:
        # Load episode transcript
        with open(episode_file, 'r', encoding='utf-8') as f:
            episode_data = json.load(f)

        episode_num = episode_data.get('episode_number')
        if episode_num is None:
            # Try to extract from filename
            try:
                episode_num = int(episode_file.stem.split('_')[1])
            except (IndexError, ValueError):
                logger.warning(f"Could not extract episode number from {episode_file.name}")
                return None

        transcript = episode_data.get('full_transcript', '')
        if not transcript or len(transcript) < 100:
            logger.warning(f"Episode {episode_num} has no valid transcript")
            return None

        logger.info(f"Processing Episode {episode_num} ({len(transcript)} chars)")

        # Chunk transcript (800 tokens, 100 overlap - same as book processing)
        chunks = chunk_text(transcript, chunk_size=800, chunk_overlap=100)
        logger.info(f"  Created {len(chunks)} chunks")

        # Extract entities and relationships from each chunk
        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"episode_{episode_num}_chunk_{i}"

            # Extract entities
            try:
                entity_result = entity_extractor.extract_entities(
                    text=chunk,
                    episode_number=episode_num,
                    chunk_id=chunk_id
                )
                all_entities.extend(entity_result.entities)
            except Exception as e:
                logger.error(f"  Error extracting entities from chunk {i}: {e}")
                continue

            # Extract relationships (needs entities from this chunk)
            if entity_result.entities:
                entity_list = [
                    {'name': e.name, 'type': e.type}
                    for e in entity_result.entities
                ]

                try:
                    rel_result = relationship_extractor.extract_relationships(
                        text=chunk,
                        entities=entity_list,
                        episode_number=episode_num,
                        chunk_id=chunk_id
                    )
                    all_relationships.extend(rel_result.relationships)
                except Exception as e:
                    logger.error(f"  Error extracting relationships from chunk {i}: {e}")

        # Aggregate entities and relationships
        aggregated_entities = entity_extractor.aggregate_entities(
            [type('Result', (), {'entities': all_entities})]
        )
        aggregated_relationships = relationship_extractor.aggregate_relationships(
            [type('Result', (), {'relationships': all_relationships})]
        )

        logger.info(
            f"  Extracted {len(aggregated_entities)} unique entities, "
            f"{len(aggregated_relationships)} unique relationships"
        )

        # Save extraction
        extraction_data = {
            'episode_number': episode_num,
            'title': episode_data.get('title', f'Episode {episode_num}'),
            'entities': [
                {
                    'name': e.name,
                    'type': e.type,
                    'description': e.description,
                    'aliases': e.aliases,
                    'metadata': e.metadata
                }
                for e in aggregated_entities
            ],
            'relationships': [
                {
                    'source_entity': r.source_entity,
                    'relationship_type': r.relationship_type,
                    'target_entity': r.target_entity,
                    'description': r.description,
                    'metadata': r.metadata
                }
                for r in aggregated_relationships
            ]
        }

        # Save to file
        output_file = output_dir / f"episode_{episode_num}_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved extraction to {output_file.name}")

        return {
            'episode_number': episode_num,
            'entities_count': len(aggregated_entities),
            'relationships_count': len(aggregated_relationships),
            'chunks_processed': len(chunks)
        }

    except Exception as e:
        logger.error(f"Error processing {episode_file.name}: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract knowledge graph from episodes')
    parser.add_argument('--episodes', type=str, help='Comma-separated episode numbers (e.g., 120,122,124)')
    parser.add_argument('--start', type=int, help='Start episode number (inclusive)')
    parser.add_argument('--end', type=int, help='End episode number (inclusive)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip episodes with existing extractions')
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    transcripts_dir = base_dir / 'data' / 'transcripts'
    output_dir = base_dir / 'data' / 'knowledge_graph' / 'entities'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractors
    logger.info("Initializing extractors...")
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()

    # Get list of episode files to process
    if args.episodes:
        # Specific episodes
        episode_nums = [int(n.strip()) for n in args.episodes.split(',')]
        episode_files = [transcripts_dir / f'episode_{n}.json' for n in episode_nums]
    elif args.start is not None and args.end is not None:
        # Range of episodes
        episode_files = [
            transcripts_dir / f'episode_{n}.json'
            for n in range(args.start, args.end + 1)
        ]
    else:
        # All episodes
        episode_files = sorted(transcripts_dir.glob('episode_*.json'))

    # Filter to existing files
    episode_files = [f for f in episode_files if f.exists()]

    if not episode_files:
        logger.error("No episode files found!")
        return

    # Filter out already processed (if requested)
    if args.skip_existing:
        episode_files = [
            f for f in episode_files
            if not (output_dir / f.name.replace('.json', '_extraction.json')).exists()
        ]

    logger.info(f"Processing {len(episode_files)} episodes...")
    logger.info("=" * 80)

    # Process each episode
    results = []
    for i, episode_file in enumerate(episode_files, 1):
        logger.info(f"\n[{i}/{len(episode_files)}] {episode_file.name}")
        result = extract_from_episode(
            episode_file,
            entity_extractor,
            relationship_extractor,
            output_dir
        )
        if result:
            results.append(result)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Episodes processed: {len(results)}/{len(episode_files)}")

    if results:
        total_entities = sum(r['entities_count'] for r in results)
        total_relationships = sum(r['relationships_count'] for r in results)
        logger.info(f"Total entities: {total_entities}")
        logger.info(f"Total relationships: {total_relationships}")
        logger.info(f"Average entities per episode: {total_entities/len(results):.1f}")
        logger.info(f"Average relationships per episode: {total_relationships/len(results):.1f}")

    logger.info(f"\nExtraction files saved to: {output_dir}")
    logger.info("\nNext step: Run build_unified_graph_v2.py to build the unified graph")


if __name__ == '__main__':
    main()
