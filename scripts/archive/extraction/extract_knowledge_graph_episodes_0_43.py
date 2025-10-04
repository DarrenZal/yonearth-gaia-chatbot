#!/usr/bin/env python3
"""
Knowledge Graph Extraction for Episodes 0-43

This script processes podcast episode transcripts and extracts:
1. Entities (people, organizations, concepts, places, etc.)
2. Relationships between entities
3. Domain classifications

Saves results to data/knowledge_graph/entities/episode_{N}_extraction.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

from src.knowledge_graph.extractors.entity_extractor import EntityExtractor
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor
from src.knowledge_graph.extractors.chunking import chunk_transcript

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'data' / 'knowledge_graph' / 'extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EpisodeKnowledgeGraphExtractor:
    """Extracts knowledge graph from episode transcripts"""

    def __init__(
        self,
        start_episode: int = 0,
        end_episode: int = 43,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize the extractor

        Args:
            start_episode: First episode number to process
            end_episode: Last episode number to process (inclusive)
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize extractors
        self.entity_extractor = EntityExtractor(
            api_key=api_key,
            model="gpt-4o-mini"
        )

        self.relationship_extractor = RelationshipExtractor(
            api_key=api_key,
            model="gpt-4o-mini"
        )

        # Setup paths
        self.transcripts_dir = project_root / "data" / "transcripts"
        self.output_dir = project_root / "data" / "knowledge_graph" / "entities"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            "total_episodes_processed": 0,
            "total_episodes_failed": 0,
            "total_chunks_processed": 0,
            "total_entities_extracted": 0,
            "total_relationships_extracted": 0,
            "failed_episodes": [],
            "start_time": None,
            "end_time": None
        }

    def load_episode(self, episode_number: int) -> Dict[str, Any]:
        """Load episode transcript from JSON file

        Args:
            episode_number: Episode number to load

        Returns:
            Episode data dictionary

        Raises:
            FileNotFoundError: If episode file doesn't exist
            ValueError: If episode has no transcript
        """
        episode_file = self.transcripts_dir / f"episode_{episode_number}.json"

        if not episode_file.exists():
            raise FileNotFoundError(f"Episode file not found: {episode_file}")

        with open(episode_file, 'r', encoding='utf-8') as f:
            episode_data = json.load(f)

        # Validate transcript exists
        transcript = episode_data.get("full_transcript", "")
        if not transcript or len(transcript) < 100:
            raise ValueError(f"Episode {episode_number} has no valid transcript")

        return episode_data

    def process_episode(self, episode_number: int) -> Dict[str, Any]:
        """Process a single episode and extract knowledge graph

        Args:
            episode_number: Episode number to process

        Returns:
            Extraction results dictionary
        """
        logger.info(f"Processing Episode {episode_number}")

        try:
            # Load episode
            episode_data = self.load_episode(episode_number)
            transcript = episode_data.get("full_transcript", "")

            # Chunk the transcript
            chunks = chunk_transcript(
                transcript,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            logger.info(f"Created {len(chunks)} chunks for episode {episode_number}")

            # Extract entities and relationships from each chunk
            entity_results = []
            relationship_results = []

            for chunk in chunks:
                chunk_id = f"ep{episode_number}_chunk{chunk['chunk_index']}"

                # Extract entities
                entity_result = self.entity_extractor.extract_entities(
                    text=chunk["text"],
                    episode_number=episode_number,
                    chunk_id=chunk_id
                )
                entity_results.append(entity_result)

                # Extract relationships based on the entities found
                if entity_result.entities:
                    entities_list = [
                        {"name": e.name, "type": e.type}
                        for e in entity_result.entities
                    ]

                    relationship_result = self.relationship_extractor.extract_relationships(
                        text=chunk["text"],
                        entities=entities_list,
                        episode_number=episode_number,
                        chunk_id=chunk_id
                    )
                    relationship_results.append(relationship_result)

            # Aggregate entities and relationships
            aggregated_entities = self.entity_extractor.aggregate_entities(entity_results)
            aggregated_relationships = self.relationship_extractor.aggregate_relationships(relationship_results)

            # Get summary statistics
            summary = {
                "total_chunks": len(chunks),
                "total_entities": len(aggregated_entities),
                "total_relationships": len(aggregated_relationships),
                "entity_types": {},
                "relationship_types": {}
            }

            # Count entity types
            for entity in aggregated_entities:
                entity_type = entity.type
                summary["entity_types"][entity_type] = summary["entity_types"].get(entity_type, 0) + 1

            # Count relationship types
            for rel in aggregated_relationships:
                rel_type = rel.relationship_type
                summary["relationship_types"][rel_type] = summary["relationship_types"].get(rel_type, 0) + 1

            # Create episode extraction result
            episode_result = {
                "episode_number": episode_number,
                "episode_title": episode_data.get("title", ""),
                "guest_name": episode_data.get("guest_name", ""),
                "total_chunks": len(chunks),
                "entities": [
                    {
                        "name": e.name,
                        "type": e.type,
                        "description": e.description,
                        "aliases": e.aliases,
                        "metadata": e.metadata
                    }
                    for e in aggregated_entities
                ],
                "relationships": [
                    {
                        "source_entity": r.source_entity,
                        "relationship_type": r.relationship_type,
                        "target_entity": r.target_entity,
                        "description": r.description,
                        "metadata": r.metadata
                    }
                    for r in aggregated_relationships
                ],
                "summary_stats": summary,
                "extraction_timestamp": datetime.now().isoformat()
            }

            # Update global statistics
            self.stats["total_chunks_processed"] += len(chunks)
            self.stats["total_entities_extracted"] += summary["total_entities"]
            self.stats["total_relationships_extracted"] += summary["total_relationships"]

            return episode_result

        except Exception as e:
            logger.error(f"Failed to process episode {episode_number}: {e}", exc_info=True)
            self.stats["total_episodes_failed"] += 1
            self.stats["failed_episodes"].append({
                "episode_number": episode_number,
                "error": str(e)
            })
            raise

    def save_episode_extraction(self, episode_result: Dict[str, Any]) -> None:
        """Save episode extraction results to JSON file

        Args:
            episode_result: Extraction results dictionary
        """
        episode_number = episode_result["episode_number"]
        output_file = self.output_dir / f"episode_{episode_number}_extraction.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(episode_result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved extraction results to {output_file}")

    def run(self) -> Dict[str, Any]:
        """Run the extraction process for all episodes

        Returns:
            Final statistics dictionary
        """
        self.stats["start_time"] = datetime.now().isoformat()
        logger.info(f"Starting knowledge graph extraction for episodes {self.start_episode}-{self.end_episode}")

        for episode_num in range(self.start_episode, self.end_episode + 1):
            try:
                # Check if already processed (optional: skip if exists)
                output_file = self.output_dir / f"episode_{episode_num}_extraction.json"
                if output_file.exists():
                    logger.info(f"Episode {episode_num} already processed, skipping...")
                    continue

                # Process episode
                episode_result = self.process_episode(episode_num)

                # Save results
                self.save_episode_extraction(episode_result)

                # Update success counter
                self.stats["total_episodes_processed"] += 1

                logger.info(
                    f"Episode {episode_num} complete: "
                    f"{episode_result['summary_stats']['total_entities']} entities, "
                    f"{episode_result['summary_stats']['total_relationships']} relationships"
                )

            except FileNotFoundError as e:
                logger.warning(f"Episode {episode_num} file not found: {e}")
                continue

            except ValueError as e:
                logger.warning(f"Episode {episode_num} has invalid data: {e}")
                continue

            except Exception as e:
                logger.error(f"Unexpected error processing episode {episode_num}: {e}", exc_info=True)
                continue

        self.stats["end_time"] = datetime.now().isoformat()

        # Save final statistics
        stats_file = self.output_dir / "extraction_stats_0_43.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Extraction complete. Statistics saved to {stats_file}")
        return self.stats

    def print_summary(self) -> None:
        """Print summary statistics"""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH EXTRACTION SUMMARY")
        print("="*60)
        print(f"Episodes Processed: {self.stats['total_episodes_processed']}")
        print(f"Episodes Failed: {self.stats['total_episodes_failed']}")
        print(f"Total Chunks: {self.stats['total_chunks_processed']}")
        print(f"Total Entities: {self.stats['total_entities_extracted']}")
        print(f"Total Relationships: {self.stats['total_relationships_extracted']}")

        if self.stats["failed_episodes"]:
            print(f"\nFailed Episodes:")
            for failure in self.stats["failed_episodes"]:
                print(f"  - Episode {failure['episode_number']}: {failure['error']}")

        print(f"\nStart Time: {self.stats['start_time']}")
        print(f"End Time: {self.stats['end_time']}")
        print("="*60)


def main():
    """Main entry point"""
    # Create extractor
    extractor = EpisodeKnowledgeGraphExtractor(
        start_episode=0,
        end_episode=43,
        chunk_size=500,
        chunk_overlap=50
    )

    # Run extraction
    try:
        stats = extractor.run()
        extractor.print_summary()

        return 0 if stats["total_episodes_failed"] == 0 else 1

    except Exception as e:
        logger.error(f"Fatal error during extraction: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
