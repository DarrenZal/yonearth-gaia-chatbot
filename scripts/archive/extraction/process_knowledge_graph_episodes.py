#!/usr/bin/env python3
"""
Process Episodes 88-131 for Knowledge Graph Extraction

This script processes episodes 88-131 from the YonEarth podcast transcripts,
extracting entities and relationships using the EntityExtractor.

Agent 6 Task: Extract entities and relationships from episodes 88-131
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/claudeuser/yonearth-gaia-chatbot/.env')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.extractors.entity_extractor import EntityExtractor, Entity, EntityExtractionResult
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor, Relationship, RelationshipExtractionResult
from src.knowledge_graph.extractors.chunking import chunk_transcript

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/claudeuser/yonearth-gaia-chatbot/logs/kg_extraction_88_131.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EpisodeProcessor:
    """Processes episodes for knowledge graph extraction"""

    def __init__(
        self,
        transcripts_dir: str,
        output_dir: str,
        start_episode: int = 88,
        end_episode: int = 131
    ):
        """
        Initialize the episode processor.

        Args:
            transcripts_dir: Directory containing episode transcript JSON files
            output_dir: Directory to save extraction results
            start_episode: First episode number to process
            end_episode: Last episode number to process
        """
        self.transcripts_dir = Path(transcripts_dir)
        self.output_dir = Path(output_dir)
        self.start_episode = start_episode
        self.end_episode = end_episode

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        self.entity_extractor = EntityExtractor(api_key=api_key, model="gpt-4o-mini")
        self.relationship_extractor = RelationshipExtractor(api_key=api_key, model="gpt-4o-mini")

        # Statistics tracking
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": [],
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "entity_type_counts": {},
            "relationship_type_counts": {},
            "start_time": None,
            "end_time": None
        }

    def load_episode(self, episode_number: int) -> Dict[str, Any]:
        """Load episode transcript from JSON file.

        Args:
            episode_number: Episode number to load

        Returns:
            Episode data dictionary

        Raises:
            FileNotFoundError: If episode file doesn't exist
        """
        filepath = self.transcripts_dir / f"episode_{episode_number}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Episode {episode_number} not found at {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def process_episode(self, episode_number: int) -> Dict[str, Any]:
        """Process a single episode, extracting entities and relationships.

        Args:
            episode_number: Episode number to process

        Returns:
            Dictionary with episode data and extraction results
        """
        logger.info(f"Processing Episode {episode_number}")

        # Load episode data
        episode_data = self.load_episode(episode_number)

        # Get transcript
        transcript = episode_data.get('full_transcript', '')
        if not transcript:
            logger.warning(f"Episode {episode_number} has no transcript")
            return None

        # Extract metadata
        title = episode_data.get('title', '')
        guest = episode_data.get('guest', '')

        logger.info(f"  Title: {title}")
        logger.info(f"  Guest: {guest}")

        # Chunk the transcript
        chunks = chunk_transcript(transcript, chunk_size=500, overlap=50)
        logger.info(f"  Created {len(chunks)} chunks")

        # Process each chunk
        entity_results = []
        relationship_results = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"ep{episode_number}_chunk{chunk['chunk_index']}"

            try:
                logger.info(f"    Processing chunk {i+1}/{len(chunks)}")

                # Extract entities
                entity_result = self.entity_extractor.extract_entities(
                    text=chunk["text"],
                    episode_number=episode_number,
                    chunk_id=chunk_id
                )
                entity_results.append(entity_result)

                # Extract relationships using the entities found
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

                logger.info(f"      Found {len(entity_result.entities)} entities, "
                          f"{len(relationship_result.relationships)} relationships")

            except Exception as e:
                logger.error(f"    Failed to process chunk {i}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # Aggregate results
        episode_result = self._aggregate_episode_results(
            episode_number=episode_number,
            title=title,
            guest=guest,
            entity_results=entity_results,
            relationship_results=relationship_results
        )

        # Update statistics
        self._update_stats(episode_result)

        logger.info(f"  Extracted {episode_result['total_entities']} entities, "
                   f"{episode_result['total_relationships']} relationships")

        return episode_result

    def _aggregate_episode_results(
        self,
        episode_number: int,
        title: str,
        guest: str,
        entity_results: List[EntityExtractionResult],
        relationship_results: List[RelationshipExtractionResult]
    ) -> Dict[str, Any]:
        """Aggregate chunk results for an episode.

        Args:
            episode_number: Episode number
            title: Episode title
            guest: Episode guest
            entity_results: List of entity extraction results from chunks
            relationship_results: List of relationship extraction results from chunks

        Returns:
            Aggregated episode results
        """
        # Use the aggregation methods from the extractors
        unique_entities = self.entity_extractor.aggregate_entities(entity_results)
        unique_relationships = self.relationship_extractor.aggregate_relationships(relationship_results)

        # Count entity types
        entity_type_counts = {}
        for entity in unique_entities:
            entity_type = entity.type
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        # Count relationship types
        relationship_type_counts = {}
        for rel in unique_relationships:
            rel_type = rel.relationship_type
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

        # Convert to dictionaries for JSON serialization
        entities_dicts = [e.model_dump() for e in unique_entities]
        relationships_dicts = [r.model_dump() for r in unique_relationships]

        return {
            "episode_number": episode_number,
            "title": title,
            "guest": guest,
            "total_chunks": len(entity_results),
            "total_entities": len(unique_entities),
            "total_relationships": len(unique_relationships),
            "entity_type_counts": entity_type_counts,
            "relationship_type_counts": relationship_type_counts,
            "entities": entities_dicts,
            "relationships": relationships_dicts
        }

    def _update_stats(self, episode_result: Dict[str, Any]):
        """Update overall statistics with episode results.

        Args:
            episode_result: Results from processing an episode
        """
        self.stats["total_chunks"] += episode_result["total_chunks"]
        self.stats["total_entities"] += episode_result["total_entities"]
        self.stats["total_relationships"] += episode_result["total_relationships"]

        # Update entity type counts
        for entity_type, count in episode_result["entity_type_counts"].items():
            self.stats["entity_type_counts"][entity_type] = \
                self.stats["entity_type_counts"].get(entity_type, 0) + count

        # Update relationship type counts
        for rel_type, count in episode_result.get("relationship_type_counts", {}).items():
            self.stats["relationship_type_counts"][rel_type] = \
                self.stats["relationship_type_counts"].get(rel_type, 0) + count

    def save_episode_result(self, episode_result: Dict[str, Any]):
        """Save episode extraction results to JSON file.

        Args:
            episode_result: Episode extraction results
        """
        episode_number = episode_result["episode_number"]
        filepath = self.output_dir / f"episode_{episode_number}_extraction.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(episode_result, f, indent=2, ensure_ascii=False)

        logger.info(f"  Saved results to {filepath}")

    def process_all_episodes(self):
        """Process all episodes from start_episode to end_episode."""
        logger.info(f"Starting processing of episodes {self.start_episode}-{self.end_episode}")
        self.stats["start_time"] = datetime.now()

        for episode_num in range(self.start_episode, self.end_episode + 1):
            try:
                self.stats["total_episodes"] += 1

                # Process episode
                result = self.process_episode(episode_num)

                if result:
                    # Save results
                    self.save_episode_result(result)
                    self.stats["successful_episodes"] += 1
                else:
                    self.stats["failed_episodes"].append({
                        "episode": episode_num,
                        "reason": "No transcript or processing returned None"
                    })

            except Exception as e:
                logger.error(f"Failed to process episode {episode_num}: {e}")
                self.stats["failed_episodes"].append({
                    "episode": episode_num,
                    "reason": str(e)
                })
                continue

        self.stats["end_time"] = datetime.now()

    def generate_summary_report(self) -> str:
        """Generate a summary report of the processing.

        Returns:
            Formatted summary report string
        """
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "N/A"

        report = f"""
=============================================================================
KNOWLEDGE GRAPH EXTRACTION REPORT - Episodes {self.start_episode}-{self.end_episode}
=============================================================================

PROCESSING SUMMARY
-----------------
Total Episodes Attempted:    {self.stats['total_episodes']}
Successfully Processed:       {self.stats['successful_episodes']}
Failed Episodes:              {len(self.stats['failed_episodes'])}
Processing Time:              {duration_str}

EXTRACTION STATISTICS
--------------------
Total Chunks Processed:       {self.stats['total_chunks']}
Total Entities Extracted:     {self.stats['total_entities']}
Total Relationships Extracted: {self.stats['total_relationships']}

ENTITY TYPE DISTRIBUTION
------------------------
"""
        for entity_type, count in sorted(
            self.stats['entity_type_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"  {entity_type:20} {count:6d}\n"

        report += "\nRELATIONSHIP TYPE DISTRIBUTION\n"
        report += "-------------------------------\n"
        for rel_type, count in sorted(
            self.stats['relationship_type_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"  {rel_type:30} {count:6d}\n"

        if self.stats['failed_episodes']:
            report += "\nFAILED EPISODES\n"
            report += "---------------\n"
            for failure in self.stats['failed_episodes']:
                report += f"  Episode {failure['episode']}: {failure['reason']}\n"

        report += "\n=============================================================================\n"

        return report

    def save_summary_report(self, filepath: str):
        """Save summary report to file.

        Args:
            filepath: Path to save report
        """
        report = self.generate_summary_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Saved summary report to {filepath}")

        # Also save statistics as JSON
        stats_filepath = filepath.replace('.txt', '.json')
        with open(stats_filepath, 'w', encoding='utf-8') as f:
            # Convert datetime objects to strings
            stats_copy = self.stats.copy()
            if stats_copy['start_time']:
                stats_copy['start_time'] = stats_copy['start_time'].isoformat()
            if stats_copy['end_time']:
                stats_copy['end_time'] = stats_copy['end_time'].isoformat()

            json.dump(stats_copy, f, indent=2, ensure_ascii=False)


def main():
    """Main execution function"""
    # Configuration
    transcripts_dir = "/home/claudeuser/yonearth-gaia-chatbot/data/transcripts"
    output_dir = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/entities"
    report_path = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/extraction_report_88_131.txt"

    # Create logs directory if needed
    logs_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/logs")
    logs_dir.mkdir(exist_ok=True)

    # Initialize processor
    processor = EpisodeProcessor(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        start_episode=88,
        end_episode=131
    )

    try:
        # Process all episodes
        processor.process_all_episodes()

        # Generate and save report
        processor.save_summary_report(report_path)

        # Print report to console
        print(processor.generate_summary_report())

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        # Still save what we have
        processor.save_summary_report(report_path)
        print(processor.generate_summary_report())

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
