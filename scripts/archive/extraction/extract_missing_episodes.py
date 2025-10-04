#!/usr/bin/env python3
"""
Extract entities from missing episodes.

Processes the 28 episodes that have transcripts but no entity extractions yet.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/claudeuser/yonearth-gaia-chatbot/.env')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.extractors.entity_extractor import EntityExtractor
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor
from src.knowledge_graph.extractors.chunking import chunk_transcript

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs/extract_missing_episodes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Episodes that need extraction
MISSING_EPISODES = [3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 18, 19, 21, 22, 23, 24,
                    28, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 42]


def process_episode(episode_num, entity_extractor, relationship_extractor,
                   transcripts_dir, output_dir):
    """Process a single episode."""

    # Load transcript
    transcript_file = transcripts_dir / f'episode_{episode_num}.json'
    if not transcript_file.exists():
        logger.warning(f"Transcript file not found: {transcript_file}")
        return False

    with open(transcript_file) as f:
        episode_data = json.load(f)

    transcript = episode_data.get('full_transcript', '')
    if not transcript or len(transcript) < 100:
        logger.warning(f"Episode {episode_num} has no transcript or too short")
        return False

    title = episode_data.get('title', f'Episode {episode_num}')
    logger.info(f"Processing Episode {episode_num}: {title}")

    # Chunk transcript
    chunks = chunk_transcript(
        transcript=transcript,
        chunk_size=2000,
        overlap=200
    )

    # Add episode metadata to chunks
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = f'ep{episode_num}_chunk{i}'
        chunk['episode_number'] = episode_num

    logger.info(f"  Created {len(chunks)} chunks")

    # Extract entities and relationships
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        logger.info(f"  Processing chunk {i+1}/{len(chunks)}")

        # Extract entities
        entity_result = entity_extractor.extract_entities(
            text=chunk['text'],
            episode_number=episode_num,
            chunk_id=chunk['chunk_id']
        )

        if entity_result and entity_result.entities:
            all_entities.extend([e.model_dump() for e in entity_result.entities])

        # Extract relationships
        # Convert entities to dict format for relationship extraction
        entities_dict = [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description
            }
            for e in (entity_result.entities if entity_result else [])
        ]

        rel_result = relationship_extractor.extract_relationships(
            text=chunk['text'],
            entities=entities_dict,
            episode_number=episode_num,
            chunk_id=chunk['chunk_id']
        )

        if rel_result and rel_result.relationships:
            all_relationships.extend([r.model_dump() for r in rel_result.relationships])

    # Save extraction results
    output_file = output_dir / f'episode_{episode_num}_extraction.json'
    result = {
        'episode_number': episode_num,
        'episode_title': title,
        'guest_name': episode_data.get('guest', ''),
        'total_chunks': len(chunks),
        'entities': all_entities,
        'relationships': all_relationships,
        'extracted_at': datetime.now().isoformat()
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✅ Saved {len(all_entities)} entities, {len(all_relationships)} relationships")
    return True


def main():
    """Main extraction process."""
    logger.info("="*80)
    logger.info("EXTRACTING MISSING EPISODES")
    logger.info("="*80)
    logger.info(f"Episodes to process: {MISSING_EPISODES}")
    logger.info(f"Total: {len(MISSING_EPISODES)} episodes")

    # Setup paths
    transcripts_dir = project_root / 'data/transcripts'
    output_dir = project_root / 'data/knowledge_graph/entities'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractors
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    entity_extractor = EntityExtractor(api_key=api_key, model="gpt-4o-mini")
    relationship_extractor = RelationshipExtractor(api_key=api_key, model="gpt-4o-mini")

    # Process episodes
    successful = 0
    failed = []

    for episode_num in MISSING_EPISODES:
        try:
            if process_episode(episode_num, entity_extractor, relationship_extractor,
                             transcripts_dir, output_dir):
                successful += 1
            else:
                failed.append(episode_num)
        except Exception as e:
            logger.error(f"Error processing episode {episode_num}: {e}", exc_info=True)
            failed.append(episode_num)

    # Summary
    logger.info("="*80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Successful: {successful}/{len(MISSING_EPISODES)}")
    if failed:
        logger.info(f"Failed: {failed}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
