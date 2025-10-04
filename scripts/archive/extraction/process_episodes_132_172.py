#!/usr/bin/env python3
"""
Process episodes 132-172 for knowledge graph extraction.

This script:
1. Loads episode transcripts from data/transcripts/
2. Chunks each transcript into 500-token segments
3. Extracts entities and relationships from each chunk
4. Aggregates results at the episode level
5. Saves extraction results to data/knowledge_graph/entities/
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.extractors.chunking import chunk_transcript
from src.knowledge_graph.extractors.entity_extractor import EntityExtractor
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor


class EpisodeProcessor:
    """Processes episodes for knowledge graph extraction"""

    def __init__(
        self,
        transcripts_dir: str,
        output_dir: str,
        start_episode: int = 132,
        end_episode: int = 172
    ):
        self.transcripts_dir = Path(transcripts_dir)
        self.output_dir = Path(output_dir)
        self.start_episode = start_episode
        self.end_episode = end_episode

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()

        # Statistics
        self.stats = {
            "episodes_processed": 0,
            "episodes_failed": [],
            "total_entities": 0,
            "total_relationships": 0,
            "total_chunks": 0,
            "processing_times": [],
        }

    def load_transcript(self, episode_number: int) -> Dict[str, Any]:
        """Load transcript from JSON file"""
        transcript_path = self.transcripts_dir / f"episode_{episode_number}.json"

        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_episode(self, episode_number: int) -> Dict[str, Any]:
        """Process a single episode"""
        print(f"\n{'='*60}")
        print(f"Processing Episode {episode_number}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Load transcript
            episode_data = self.load_transcript(episode_number)
            transcript = episode_data.get("full_transcript", "")

            if not transcript or len(transcript) < 100:
                print(f"Warning: Episode {episode_number} has no valid transcript")
                return None

            # Chunk the transcript
            print(f"Chunking transcript...")
            chunks = chunk_transcript(transcript, chunk_size=500, overlap=50)
            print(f"Created {len(chunks)} chunks")

            self.stats["total_chunks"] += len(chunks)

            # Extract entities and relationships from each chunk
            entity_results = []
            relationship_results = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"ep{episode_number}_chunk{i}"
                print(f"  Processing chunk {i+1}/{len(chunks)}...", end=" ")

                try:
                    # Extract entities
                    entity_result = self.entity_extractor.extract_entities(
                        text=chunk["text"],
                        episode_number=episode_number,
                        chunk_id=chunk_id
                    )
                    entity_results.append(entity_result)

                    # Convert entities to dict format for relationship extraction
                    entities_dict = [
                        {
                            "name": e.name,
                            "type": e.type,
                            "description": e.description
                        }
                        for e in entity_result.entities
                    ]

                    # Extract relationships
                    relationship_result = self.relationship_extractor.extract_relationships(
                        text=chunk["text"],
                        entities=entities_dict,
                        episode_number=episode_number,
                        chunk_id=chunk_id
                    )
                    relationship_results.append(relationship_result)

                    print(f"✓ ({len(entity_result.entities)} entities, {len(relationship_result.relationships)} relationships)")

                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

            # Aggregate results
            print("\nAggregating results...")
            aggregated_entities = self.entity_extractor.aggregate_entities(entity_results)
            aggregated_relationships = self.relationship_extractor.aggregate_relationships(relationship_results)

            print(f"  Unique entities: {len(aggregated_entities)}")
            print(f"  Unique relationships: {len(aggregated_relationships)}")

            self.stats["total_entities"] += len(aggregated_entities)
            self.stats["total_relationships"] += len(aggregated_relationships)

            # Create output data
            output_data = {
                "episode_number": episode_number,
                "episode_title": episode_data.get("episode_title", ""),
                "guest": episode_data.get("guest", ""),
                "date": episode_data.get("date", ""),
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunks_processed": len(chunks),
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
                ]
            }

            # Save to file
            output_path = self.output_dir / f"episode_{episode_number}_extraction.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            self.stats["episodes_processed"] += 1

            print(f"\n✓ Episode {episode_number} completed in {processing_time:.1f}s")
            print(f"  Saved to: {output_path}")

            return output_data

        except Exception as e:
            print(f"\n✗ Episode {episode_number} failed: {e}")
            self.stats["episodes_failed"].append({
                "episode": episode_number,
                "error": str(e)
            })
            return None

    def process_all(self):
        """Process all episodes in the range"""
        print(f"\n{'#'*60}")
        print(f"# Knowledge Graph Extraction: Episodes {self.start_episode}-{self.end_episode}")
        print(f"{'#'*60}\n")

        total_episodes = self.end_episode - self.start_episode + 1
        print(f"Total episodes to process: {total_episodes}\n")

        start_time = time.time()

        for episode_num in range(self.start_episode, self.end_episode + 1):
            self.process_episode(episode_num)

            # Progress update
            processed = episode_num - self.start_episode + 1
            print(f"\nProgress: {processed}/{total_episodes} episodes")

        total_time = time.time() - start_time

        # Print final statistics
        self.print_statistics(total_time)

        # Save statistics
        self.save_statistics()

    def print_statistics(self, total_time: float):
        """Print processing statistics"""
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}\n")

        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
        print(f"\nEpisodes:")
        print(f"  Successfully processed: {self.stats['episodes_processed']}")
        print(f"  Failed: {len(self.stats['episodes_failed'])}")

        if self.stats["episodes_failed"]:
            print(f"\n  Failed episodes:")
            for fail in self.stats["episodes_failed"]:
                print(f"    - Episode {fail['episode']}: {fail['error']}")

        print(f"\nChunks:")
        print(f"  Total chunks processed: {self.stats['total_chunks']}")

        print(f"\nExtraction Results:")
        print(f"  Total entities extracted: {self.stats['total_entities']}")
        print(f"  Total relationships extracted: {self.stats['total_relationships']}")

        if self.stats["processing_times"]:
            avg_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            print(f"\nAverage processing time per episode: {avg_time:.1f}s")

        print(f"\nOutput directory: {self.output_dir}")
        print(f"{'='*60}\n")

    def save_statistics(self):
        """Save statistics to JSON file"""
        stats_path = self.output_dir / "extraction_statistics.json"

        stats_output = {
            **self.stats,
            "episode_range": {
                "start": self.start_episode,
                "end": self.end_episode
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_output, f, indent=2, ensure_ascii=False)

        print(f"Statistics saved to: {stats_path}")


def main():
    """Main entry point"""
    # Configuration
    transcripts_dir = "/home/claudeuser/yonearth-gaia-chatbot/data/transcripts"
    output_dir = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/entities"
    start_episode = 132
    end_episode = 172

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Create processor and run
    processor = EpisodeProcessor(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        start_episode=start_episode,
        end_episode=end_episode
    )

    processor.process_all()


if __name__ == "__main__":
    main()
