#!/usr/bin/env python3
"""
Extract entities from the final 17 episodes with improved error handling.
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
        logging.FileHandler(project_root / 'logs/extract_final_17.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# The 17 episodes that still need extraction
EPISODES_TO_EXTRACT = [9, 13, 18, 22, 23, 28, 30, 35, 40, 48, 53, 62, 63, 73, 75, 171, 172]


def process_episode(episode_num, entity_extractor, relationship_extractor,
                   transcripts_dir, output_dir):
    """Process a single episode."""

    # Load transcript
    transcript_file = transcripts_dir / f'episode_{episode_num}.json'
    if not transcript_file.exists():
        logger.warning(f"Transcript file not found: {transcript_file}")
        return False

    with open(transcript_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transcript = data.get('full_transcript', '')
    title = data.get('title', f'Episode {episode_num}')

    if not transcript or len(transcript) < 100:
        logger.warning(f"Episode {episode_num} has no/short transcript ({len(transcript)} chars)")
        return False

    logger.info(f"Processing Episode {episode_num}: {title}")

    # Chunk the transcript
    chunks = chunk_transcript(
        transcript=transcript,
        chunk_size=800,
        overlap=100
    )

    logger.info(f"  Created {len(chunks)} chunks")

    # Extract entities and relationships
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        logger.info(f"  Processing chunk {i+1}/{len(chunks)}")

        # Generate chunk_id from episode and chunk_index
        chunk_id = f"ep{episode_num}_chunk{chunk['chunk_index']}"

        # Extract entities
        entity_result = entity_extractor.extract_entities(
            text=chunk['text'],
            episode_number=episode_num,
            chunk_id=chunk_id
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
            chunk_id=chunk_id
        )

        if rel_result and rel_result.relationships:
            all_relationships.extend([r.model_dump() for r in rel_result.relationships])

    # Save extraction results
    output_file = output_dir / f'episode_{episode_num}_extraction.json'
    extraction_data = {
        "episode_number": episode_num,
        "title": title,
        "extraction_date": datetime.now().isoformat(),
        "entities": all_entities,
        "relationships": all_relationships
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extraction_data, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✅ Saved {len(all_entities)} entities, {len(all_relationships)} relationships")
    return True


def main():
    """Main extraction process."""
    logger.info("="*80)
    logger.info("EXTRACTING FINAL 17 EPISODES (WITH ERROR HANDLING)")
    logger.info("="*80)
    logger.info(f"Episodes to extract: {EPISODES_TO_EXTRACT}")
    logger.info(f"Total: {len(EPISODES_TO_EXTRACT)} episodes")

    # Setup paths
    transcripts_dir = project_root / 'data/transcripts'
    output_dir = project_root / 'data/knowledge_graph/entities'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractors
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()

    successful = 0
    failed = []

    for episode_num in EPISODES_TO_EXTRACT:
        logger.info("")
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
    logger.info("")
    logger.info("="*80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Successful: {successful}/{len(EPISODES_TO_EXTRACT)}")
    if failed:
        logger.info(f"Failed: {failed}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
