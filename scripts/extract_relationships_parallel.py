#!/usr/bin/env python3
"""
Parallel Relationship Extraction for Knowledge Graph
Same quality as comprehensive extraction, but 3-5x faster using parallel processing
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Pydantic models (same as comprehensive version for quality)
class EntityInRelationship(BaseModel):
    """Entity that can be a person, organization, concept, or literal value"""
    name: str = Field(description="Entity name or literal value (e.g., '95 years old', '2023', '$50,000')")
    type: str = Field(description="Entity type: PERSON, ORGANIZATION, CONCEPT, PLACE, PRACTICE, PRODUCT, EVENT, LITERAL_VALUE, etc.")

class ComprehensiveRelationship(BaseModel):
    """A single comprehensive relationship between entities"""
    source: EntityInRelationship = Field(description="The source entity (discover new ones if needed)")
    relationship_type: str = Field(description="Specific relationship type (e.g., FOUNDED, WORKS_FOR, HAS_AGE, PRACTICES, LOCATED_IN, OCCURRED_IN_YEAR)")
    target: EntityInRelationship = Field(description="The target entity or literal value")
    description: str = Field(description="Natural language description of the relationship")
    confidence: float = Field(default=1.0, description="Confidence score (0-1)")

class ComprehensiveRelationshipList(BaseModel):
    """List of relationships for structured output"""
    relationships: List[ComprehensiveRelationship] = Field(
        default_factory=list,
        description="All relationships found, including with newly discovered entities and literal values"
    )


class ParallelRelationshipExtractor:
    """Parallel extraction with same quality as comprehensive version"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", max_workers: int = 5):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_workers = max_workers  # Number of parallel API calls
        self.rate_limit_delay = 0.2  # Slightly higher delay for parallel calls
        self.api_call_lock = threading.Lock()  # Thread-safe rate limiting
        self.last_api_call = 0

    def extract_relationships_from_chunk(self, chunk_text: str, chunk_id: int, episode_num: int) -> ComprehensiveRelationshipList:
        """Extract relationships from a single chunk (same as comprehensive)"""

        # Rate limiting (thread-safe)
        with self.api_call_lock:
            time_since_last = time.time() - self.last_api_call
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            self.last_api_call = time.time()

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at extracting comprehensive knowledge graph relationships from podcast transcripts.

IMPORTANT:
1. Discover and extract ALL entities mentioned, not just from a predefined list
2. Include literal values as entities (ages, years, amounts, durations)
3. Create specific, descriptive relationship types
4. Extract relationships between any entities you find

Entity types include: PERSON, ORGANIZATION, CONCEPT, PLACE, PRACTICE, PRODUCT, EVENT, LITERAL_VALUE, MATERIAL, METHOD, etc.

For literal values:
- Ages: "95 years old" → LITERAL_VALUE
- Years: "2023" → LITERAL_VALUE
- Amounts: "$50,000" → LITERAL_VALUE
- Percentages: "30%" → LITERAL_VALUE

Example relationships:
- (Aaron Perry, FOUNDED, YonEarth Community)
- (Biochar, SEQUESTERS_CARBON_IN, Soil)
- (John Doe, HAS_AGE, "95 years old")
- (Event, OCCURRED_IN_YEAR, "2023")
- (Solar panels, COSTS, "$15,000")"""
                    },
                    {
                        "role": "user",
                        "content": f"Extract ALL comprehensive relationships from this podcast transcript chunk:\n\n{chunk_text}"
                    }
                ],
                response_format=ComprehensiveRelationshipList,
                temperature=0.1,
                max_tokens=2000
            )

            result = response.choices[0].message.parsed
            return result

        except Exception as e:
            print(f"  Error in chunk {chunk_id}: {str(e)[:100]}")
            return ComprehensiveRelationshipList(relationships=[])

    def process_episode_parallel(self, episode_num: int) -> Dict:
        """Process an entire episode using parallel chunk extraction"""

        # Load transcript
        transcript_file = project_root / f"data/transcripts/episode_{episode_num}.json"
        if not transcript_file.exists():
            return None

        with open(transcript_file) as f:
            transcript_data = json.load(f)

        full_transcript = transcript_data.get("full_transcript", "")
        if not full_transcript or len(full_transcript) < 100:
            return None

        # Chunk the transcript
        chunk_size = 3000
        overlap = 100
        chunks = []

        for i in range(0, len(full_transcript), chunk_size - overlap):
            chunk_text = full_transcript[i:i + chunk_size]
            chunks.append((chunk_text, len(chunks)))

        print(f"  Processing {len(chunks)} chunks in parallel (max {self.max_workers} workers)...")

        # Process chunks in parallel
        all_relationships = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self.extract_relationships_from_chunk,
                    chunk_text,
                    chunk_id,
                    episode_num
                ): chunk_id
                for chunk_text, chunk_id in chunks
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    all_relationships.extend(result.relationships)
                    completed += 1

                    # Progress indicator
                    if completed % 5 == 0:
                        print(f"    Completed {completed}/{len(chunks)} chunks...")

                except Exception as e:
                    print(f"    Chunk {chunk_id} failed: {str(e)[:50]}")

        # Compile results
        extraction_data = {
            "episode": episode_num,
            "title": transcript_data.get("title", ""),
            "relationships": [rel.model_dump() for rel in all_relationships],
            "statistics": {
                "total_relationships": len(all_relationships),
                "total_chunks": len(chunks),
                "chunks_processed": completed,
                "unique_relationship_types": len(set(r.relationship_type for r in all_relationships)),
                "processing_time": time.time()
            }
        }

        return extraction_data


def main():
    """Main function to run parallel extraction"""
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Relationship Extraction")
    parser.add_argument("--episodes", type=str, help="Episode numbers to process (e.g., '1-10' or '5,8,12')")
    parser.add_argument("--max-workers", type=int, default=5, help="Max parallel workers (default: 5)")
    parser.add_argument("--continue", dest="continue_extraction", action="store_true",
                       help="Continue from where comprehensive extraction left off")

    args = parser.parse_args()

    # Initialize extractor
    print("🚀 Initializing Parallel Relationship Extractor...")
    print(f"   Max parallel workers: {args.max_workers}")

    extractor = ParallelRelationshipExtractor(max_workers=args.max_workers)

    # Determine episodes to process
    episodes_to_process = []

    if args.continue_extraction:
        # Find missing episodes
        output_dir = project_root / "data/knowledge_graph/relationships"
        for i in range(172):
            if i == 26:  # Skip episode 26 (doesn't exist)
                continue
            output_file = output_dir / f"episode_{i}_extraction.json"
            if not output_file.exists():
                episodes_to_process.append(i)
        print(f"📊 Found {len(episodes_to_process)} episodes to complete")

    elif args.episodes:
        # Parse episode range/list
        if '-' in args.episodes:
            start, end = map(int, args.episodes.split('-'))
            episodes_to_process = list(range(start, end + 1))
        elif ',' in args.episodes:
            episodes_to_process = list(map(int, args.episodes.split(',')))
        else:
            episodes_to_process = [int(args.episodes)]
    else:
        # Process all episodes
        episodes_to_process = [i for i in range(172) if i != 26]

    print(f"\n📚 Processing {len(episodes_to_process)} episodes with parallel extraction...")

    # Process each episode
    output_dir = project_root / "data/knowledge_graph/relationships"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    processed = 0

    for episode_num in episodes_to_process:
        print(f"\n🎙️ Episode {episode_num}:")

        # Check if already exists
        output_file = output_dir / f"episode_{episode_num}_extraction.json"
        if output_file.exists():
            print(f"  ✓ Already processed, skipping...")
            continue

        # Process episode
        episode_start = time.time()
        extraction_data = extractor.process_episode_parallel(episode_num)

        if extraction_data:
            # Save results
            with open(output_file, 'w') as f:
                json.dump(extraction_data, f, indent=2)

            episode_time = time.time() - episode_start
            processed += 1

            print(f"  ✓ Completed in {episode_time:.1f}s")
            print(f"  📊 Found {extraction_data['statistics']['total_relationships']} relationships")
            print(f"  📝 {extraction_data['statistics']['unique_relationship_types']} unique types")

            # Progress estimate
            if processed > 0:
                avg_time = (time.time() - start_time) / processed
                remaining = len(episodes_to_process) - processed
                est_remaining = (remaining * avg_time) / 3600
                print(f"\n⏱️  Progress: {processed}/{len(episodes_to_process)} episodes")
                print(f"   Average: {avg_time:.1f}s per episode")
                print(f"   Estimated time remaining: {est_remaining:.1f} hours")
        else:
            print(f"  ⚠️ No transcript found or too short")

    # Final summary
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Extraction complete!")
    print(f"   Processed {processed} episodes in {total_time:.1f} minutes")
    if processed > 0:
        print(f"   Average: {total_time/processed*60:.1f} seconds per episode")


if __name__ == "__main__":
    main()