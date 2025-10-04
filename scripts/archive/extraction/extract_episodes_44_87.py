"""
Extraction orchestrator for processing episodes 44-87.

This script:
1. Loads episode transcripts from data/transcripts/
2. Chunks each transcript into 500-token segments
3. Extracts entities and relationships using OpenAI GPT-4
4. Aggregates results at episode level
5. Saves to data/knowledge_graph/entities/episode_{N}_extraction.json
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env")

from src.knowledge_graph.extractors.chunking import chunk_transcript
from src.knowledge_graph.extractors.entity_extractor import EntityExtractor


class ExtractionOrchestrator:
    """Orchestrates entity extraction across multiple episodes"""

    def __init__(self, start_episode: int = 44, end_episode: int = 87):
        """
        Initialize the orchestrator.

        Args:
            start_episode: First episode to process (inclusive)
            end_episode: Last episode to process (inclusive)
        """
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.data_dir = project_root / "data"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.output_dir = self.data_dir / "knowledge_graph" / "entities"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize entity extractor
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.extractor = EntityExtractor(api_key=api_key)

        # Statistics tracking
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": [],
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "total_tokens_used": 0,
            "start_time": None,
            "end_time": None
        }

    def load_episode_transcript(self, episode_number: int) -> Dict[str, Any]:
        """
        Load transcript JSON for a given episode.

        Args:
            episode_number: Episode number to load

        Returns:
            Episode data dictionary with transcript
        """
        transcript_path = self.transcripts_dir / f"episode_{episode_number}.json"

        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        with open(transcript_path, 'r', encoding='utf-8') as f:
            episode_data = json.load(f)

        return episode_data

    def process_episode(self, episode_number: int) -> Dict[str, Any]:
        """
        Process a single episode: load, chunk, extract entities.

        Args:
            episode_number: Episode number to process

        Returns:
            Extraction results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Processing Episode {episode_number}")
        print(f"{'='*60}")

        try:
            # Load transcript
            episode_data = self.load_episode_transcript(episode_number)
            transcript = episode_data.get("full_transcript", "")

            if not transcript or len(transcript) < 100:
                print(f"Warning: Episode {episode_number} has no/short transcript. Skipping.")
                return None

            print(f"Title: {episode_data.get('title', 'Unknown')}")
            print(f"Transcript length: {len(transcript)} characters")

            # Chunk transcript
            print("\nChunking transcript...")
            chunks = chunk_transcript(transcript, chunk_size=500, overlap=50)
            print(f"Created {len(chunks)} chunks")

            self.stats["total_chunks"] += len(chunks)

            # Add episode metadata to chunks
            for chunk in chunks:
                chunk["episode_number"] = episode_number
                chunk["episode_title"] = episode_data.get("title", "")
                chunk["episode_url"] = episode_data.get("url", "")

            # Extract entities from all chunks
            print("\nExtracting entities and relationships...")
            extraction_results = []

            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)}...", end="\r")
                try:
                    result = self.extractor.extract_entities(
                        text=chunk["text"],
                        episode_number=episode_number,
                        chunk_id=f"episode_{episode_number}_chunk_{i}"
                    )
                    extraction_results.append(result)
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"\n    Warning: Failed to extract from chunk {i}: {e}")
                    continue

            print(f"\n  Extracted from {len(extraction_results)}/{len(chunks)} chunks")

            # Aggregate entities using the extractor's method
            print("\nAggregating entities...")
            unique_entities = self.extractor.aggregate_entities(extraction_results)

            # Convert entities to dict format for JSON serialization
            unique_entities_dict = [
                {
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "aliases": e.aliases,
                    "metadata": e.metadata
                }
                for e in unique_entities
            ]

            # Extract relationships (note: the updated extractor doesn't return relationships separately)
            # We'll need to parse them if they exist in metadata
            unique_relationships = []

            print(f"\nExtraction Summary:")
            print(f"  Chunks processed: {len(extraction_results)}")
            print(f"  Unique entities: {len(unique_entities)}")
            print(f"  Relationships: {len(unique_relationships)}")

            self.stats["total_entities"] += len(unique_entities)
            self.stats["total_relationships"] += len(unique_relationships)

            # Compile episode extraction result
            episode_result = {
                "episode_number": episode_number,
                "episode_title": episode_data.get("title", ""),
                "episode_url": episode_data.get("url", ""),
                "publish_date": episode_data.get("publish_date", "Unknown"),
                "extraction_metadata": {
                    "chunks_processed": len(extraction_results),
                    "total_entities": len(unique_entities),
                    "total_relationships": len(unique_relationships),
                    "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "entities": unique_entities_dict,
                "relationships": unique_relationships
            }

            # Save to file
            output_path = self.output_dir / f"episode_{episode_number}_extraction.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(episode_result, f, indent=2, ensure_ascii=False)

            print(f"\n✓ Saved to: {output_path}")

            return episode_result

        except Exception as e:
            print(f"\n✗ Error processing episode {episode_number}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate entities by name and type.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of unique entities
        """
        entity_map = {}

        for entity in entities:
            name = entity.get("name", "").lower().strip()
            entity_type = entity.get("type", "UNKNOWN")
            key = (name, entity_type)

            if key not in entity_map:
                entity_map[key] = entity
            else:
                # Merge contexts and increase confidence if seen multiple times
                existing = entity_map[key]
                if "contexts" not in existing:
                    existing["contexts"] = []
                existing["contexts"].append(entity.get("context", ""))

                # Increase confidence slightly for repeated mentions
                existing["confidence"] = min(
                    1.0,
                    existing.get("confidence", 0.5) + 0.1
                )

        return list(entity_map.values())

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate relationships by source, target, and type.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            List of unique relationships
        """
        relationship_map = {}

        for rel in relationships:
            source = rel.get("source", "").lower().strip()
            target = rel.get("target", "").lower().strip()
            rel_type = rel.get("type", "UNKNOWN")
            key = (source, target, rel_type)

            if key not in relationship_map:
                relationship_map[key] = rel
            else:
                # Increase confidence for repeated relationships
                existing = relationship_map[key]
                existing["confidence"] = min(
                    1.0,
                    existing.get("confidence", 0.5) + 0.1
                )

        return list(relationship_map.values())

    def process_all_episodes(self):
        """Process all episodes from start_episode to end_episode"""
        print(f"\n{'#'*60}")
        print(f"# Knowledge Graph Extraction")
        print(f"# Episodes {self.start_episode} - {self.end_episode}")
        print(f"{'#'*60}\n")

        self.stats["start_time"] = time.time()
        self.stats["total_episodes"] = self.end_episode - self.start_episode + 1

        for episode_num in range(self.start_episode, self.end_episode + 1):
            result = self.process_episode(episode_num)

            if result:
                self.stats["successful_episodes"] += 1
            else:
                self.stats["failed_episodes"].append(episode_num)

            # Add delay between episodes to avoid rate limits
            if episode_num < self.end_episode:
                print("\nWaiting 5 seconds before next episode...")
                time.sleep(5)

        self.stats["end_time"] = time.time()

        # Generate final report
        self._generate_report()

    def _generate_report(self):
        """Generate and print final summary report"""
        duration = self.stats["end_time"] - self.stats["start_time"]
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE - SUMMARY REPORT")
        print(f"{'='*60}\n")

        print(f"Episodes Range: {self.start_episode} - {self.end_episode}")
        print(f"Total Episodes: {self.stats['total_episodes']}")
        print(f"Successfully Processed: {self.stats['successful_episodes']}")
        print(f"Failed Episodes: {len(self.stats['failed_episodes'])}")

        if self.stats['failed_episodes']:
            print(f"  Failed: {', '.join(map(str, self.stats['failed_episodes']))}")

        print(f"\nExtraction Statistics:")
        print(f"  Total Chunks Processed: {self.stats['total_chunks']:,}")
        print(f"  Total Entities Extracted: {self.stats['total_entities']:,}")
        print(f"  Total Relationships Extracted: {self.stats['total_relationships']:,}")
        print(f"  Total Tokens Used: {self.stats['total_tokens_used']:,}")

        print(f"\nPerformance:")
        print(f"  Duration: {hours}h {minutes}m {seconds}s")
        print(f"  Avg per Episode: {duration / self.stats['total_episodes']:.1f}s")

        # Estimate cost (approximate, based on gpt-4o-mini pricing)
        # $0.150 per 1M input tokens, $0.600 per 1M output tokens
        # Rough estimate: 70% input, 30% output
        input_tokens = self.stats['total_tokens_used'] * 0.7
        output_tokens = self.stats['total_tokens_used'] * 0.3
        estimated_cost = (input_tokens / 1_000_000 * 0.150) + (output_tokens / 1_000_000 * 0.600)

        print(f"  Estimated Cost: ${estimated_cost:.2f}")

        print(f"\nOutput Directory: {self.output_dir}")
        print(f"\n{'='*60}\n")

        # Save report to file
        report_path = self.output_dir / f"extraction_report_{self.start_episode}_{self.end_episode}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

        print(f"Report saved to: {report_path}")


def main():
    """Main entry point"""
    try:
        orchestrator = ExtractionOrchestrator(start_episode=44, end_episode=87)
        orchestrator.process_all_episodes()

    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
