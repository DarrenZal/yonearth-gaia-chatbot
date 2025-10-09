#!/usr/bin/env python3
"""
Comprehensive Relationship Extraction for All Episodes

This script extracts relationships from all podcast episodes using a
comprehensive approach that:
1. Discovers entities during relationship extraction
2. Includes literal/attribute relationships (ages, dates, quantities)
3. Doesn't limit itself to pre-extracted entities
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EntityInRelationship(BaseModel):
    """Entity as it appears in a relationship"""
    name: str
    type: str  # PERSON, ORGANIZATION, CONCEPT, LITERAL_VALUE, DATE, etc.


class ComprehensiveRelationship(BaseModel):
    """A relationship that can include newly discovered entities and literals"""
    source: EntityInRelationship
    relationship_type: str
    target: EntityInRelationship
    description: str
    confidence: float = 1.0  # 0.0-1.0


class ComprehensiveRelationshipList(BaseModel):
    """List of relationships for structured output"""
    relationships: List[ComprehensiveRelationship]


class ComprehensiveRelationshipExtractor:
    """Extracts relationships comprehensively without entity constraints"""

    RELATIONSHIP_TYPES = [
        # Person relationships
        "FOUNDED", "WORKS_FOR", "LEADS", "SERVES_AS", "GRADUATED_FROM",
        "TRAINED_AT", "BORN_IN", "LIVES_IN", "MEMBER_OF", "BOARD_MEMBER_OF",

        # Organization relationships
        "LOCATED_IN", "PART_OF", "OWNS", "PARTNERS_WITH", "FUNDED_BY",
        "COLLABORATES_WITH", "ACQUIRED_BY", "SUBSIDIARY_OF",

        # Practice/Method relationships
        "PRACTICES", "IMPLEMENTS", "TEACHES", "ADVOCATES_FOR", "RESEARCHES",
        "DEVELOPS", "USES", "PRODUCES", "SELLS", "DISTRIBUTES",

        # Conceptual relationships
        "INFLUENCES", "INSPIRED_BY", "BASED_ON", "DERIVED_FROM", "APPLIES",
        "MENTIONS", "DISCUSSES", "EXPLAINS", "DEMONSTRATES",

        # Attribute relationships (literals)
        "HAS_AGE", "HAS_FOUNDING_DATE", "HAS_LOCATION", "HAS_SIZE",
        "HAS_DURATION", "HAS_COST", "HAS_QUANTITY", "HAS_PERCENTAGE",

        # Temporal relationships
        "STARTED_IN", "ENDED_IN", "OCCURRED_IN", "FOUNDED_IN", "ESTABLISHED_IN",

        # General
        "RELATED_TO", "CONNECTED_TO"
    ]

    EXTRACTION_PROMPT = """You are an expert at extracting a comprehensive knowledge graph from podcast transcripts about sustainability and regenerative agriculture.

Extract ALL meaningful relationships from this text, including:
1. Relationships between people, organizations, concepts, practices
2. Attribute relationships (ages, dates, founding years, locations, quantities)
3. Entities that weren't previously identified but are important

For each relationship, identify:
- source: The subject entity (name and type: PERSON, ORGANIZATION, CONCEPT, PLACE, etc.)
- relationship_type: One of {relationship_types}
- target: The object entity (name and type) OR a LITERAL_VALUE (age, date, number, etc.)
- description: Brief description of this specific relationship
- confidence: Your confidence in this relationship (0.0-1.0)

IMPORTANT GUIDELINES:
- Extract entities AS THEY APPEAR in the text (don't require pre-extraction)
- Include literal values: "John is 95 years old" → (John, HAS_AGE, 95)
- Include dates: "Founded in 1985" → (Organization, FOUNDED_IN, 1985)
- Include locations, quantities, percentages, durations
- Be comprehensive - extract as many relationships as you can find

Example:
Text: "Dr. Jane Smith, age 62, founded Green Earth Farm in Boulder, Colorado in 1995. The farm practices regenerative agriculture and sells organic vegetables."

Relationships:
[
  {{
    "source": {{"name": "Dr. Jane Smith", "type": "PERSON"}},
    "relationship_type": "HAS_AGE",
    "target": {{"name": "62", "type": "LITERAL_VALUE"}},
    "description": "Dr. Jane Smith is 62 years old",
    "confidence": 1.0
  }},
  {{
    "source": {{"name": "Dr. Jane Smith", "type": "PERSON"}},
    "relationship_type": "FOUNDED",
    "target": {{"name": "Green Earth Farm", "type": "ORGANIZATION"}},
    "description": "Dr. Jane Smith founded Green Earth Farm",
    "confidence": 1.0
  }},
  {{
    "source": {{"name": "Green Earth Farm", "type": "ORGANIZATION"}},
    "relationship_type": "LOCATED_IN",
    "target": {{"name": "Boulder, Colorado", "type": "PLACE"}},
    "description": "Green Earth Farm is located in Boulder, Colorado",
    "confidence": 1.0
  }},
  {{
    "source": {{"name": "Green Earth Farm", "type": "ORGANIZATION"}},
    "relationship_type": "FOUNDED_IN",
    "target": {{"name": "1995", "type": "LITERAL_VALUE"}},
    "description": "Green Earth Farm was founded in 1995",
    "confidence": 1.0
  }},
  {{
    "source": {{"name": "Green Earth Farm", "type": "ORGANIZATION"}},
    "relationship_type": "PRACTICES",
    "target": {{"name": "regenerative agriculture", "type": "PRACTICE"}},
    "description": "The farm practices regenerative agriculture",
    "confidence": 1.0
  }},
  {{
    "source": {{"name": "Green Earth Farm", "type": "ORGANIZATION"}},
    "relationship_type": "SELLS",
    "target": {{"name": "organic vegetables", "type": "PRODUCT"}},
    "description": "The farm sells organic vegetables",
    "confidence": 1.0
  }}
]

Text to analyze:
{text}

Extract ALL relationships you can find. Return an empty array [] if no relationships are found."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.rate_limit_delay = 0.05  # 50ms between calls

    def extract_from_chunk(self, text: str, chunk_id: str, episode_number: int) -> List[Dict]:
        """Extract relationships from a text chunk"""

        prompt = self.EXTRACTION_PROMPT.format(
            relationship_types=", ".join(self.RELATIONSHIP_TYPES),
            text=text
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert knowledge graph extractor. Extract comprehensive relationships including entities, attributes, and literal values."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=ComprehensiveRelationshipList,
                temperature=0.1,
            )

            parsed = response.choices[0].message.parsed

            if not parsed or not parsed.relationships:
                return []

            # Convert to dictionaries with metadata
            relationships = []
            for rel in parsed.relationships:
                relationships.append({
                    "source_entity": rel.source.name,
                    "source_type": rel.source.type,
                    "relationship_type": rel.relationship_type,
                    "target_entity": rel.target.name,
                    "target_type": rel.target.type,
                    "description": rel.description,
                    "confidence": rel.confidence,
                    "metadata": {
                        "episode_number": episode_number,
                        "chunk_id": chunk_id
                    }
                })

            time.sleep(self.rate_limit_delay)
            return relationships

        except Exception as e:
            print(f"Error extracting from chunk {chunk_id}: {e}")
            return []


def process_episode(episode_number: int, extractor: ComprehensiveRelationshipExtractor,
                    transcripts_dir: Path, output_dir: Path) -> bool:
    """Process a single episode"""

    # Load transcript
    transcript_file = transcripts_dir / f"episode_{episode_number}.json"
    if not transcript_file.exists():
        print(f"  ⚠️  Transcript not found: {transcript_file}")
        return False

    try:
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
    except Exception as e:
        print(f"  ❌ Error loading transcript: {e}")
        return False

    full_transcript = transcript_data.get('full_transcript', '')
    if not full_transcript or len(full_transcript) < 100:
        print(f"  ⚠️  Transcript too short or missing")
        return False

    # Chunk the transcript (800 tokens ~= 600 words ~= 3000 chars)
    chunk_size = 3000
    overlap = 300
    chunks = []

    for i in range(0, len(full_transcript), chunk_size - overlap):
        chunk_text = full_transcript[i:i + chunk_size]
        if len(chunk_text) > 100:  # Skip tiny chunks
            chunks.append({
                "id": f"ep{episode_number}_chunk{len(chunks)}",
                "text": chunk_text
            })

    print(f"  📄 Created {len(chunks)} chunks")

    # Extract relationships from each chunk
    all_relationships = []
    for i, chunk in enumerate(chunks):
        print(f"    Processing chunk {i+1}/{len(chunks)}...", end='\r')

        relationships = extractor.extract_from_chunk(
            text=chunk['text'],
            chunk_id=chunk['id'],
            episode_number=episode_number
        )

        all_relationships.extend(relationships)

    print(f"  ✅ Extracted {len(all_relationships)} relationships")

    # Save results
    output_file = output_dir / f"episode_{episode_number}_extraction.json"
    output_data = {
        "episode_number": episode_number,
        "total_chunks": len(chunks),
        "total_relationships": len(all_relationships),
        "relationships": all_relationships
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  💾 Saved to {output_file}")
    return True


def main():
    """Main extraction process"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract comprehensive relationships from all episodes")
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE RELATIONSHIP EXTRACTION")
    print("=" * 80)

    # Setup paths
    transcripts_dir = project_root / 'data' / 'transcripts'
    output_dir = project_root / 'data' / 'knowledge_graph' / 'relationships'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing extractions
    existing_files = list(output_dir.glob("episode_*_extraction.json"))
    existing_episodes = set()
    for f in existing_files:
        try:
            ep_num = int(f.stem.split('_')[1])
            existing_episodes.add(ep_num)
        except:
            pass

    print(f"📁 Transcripts directory: {transcripts_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"✅ Already extracted: {len(existing_episodes)} episodes")
    print()

    # Find all episodes
    all_episodes = []
    for f in sorted(transcripts_dir.glob("episode_*.json")):
        try:
            ep_num = int(f.stem.split('_')[1])
            all_episodes.append(ep_num)
        except:
            pass

    episodes_to_process = [ep for ep in all_episodes if ep not in existing_episodes]

    print(f"📊 Total episodes: {len(all_episodes)}")
    print(f"⏭️  Already done: {len(existing_episodes)}")
    print(f"🎯 To process: {len(episodes_to_process)}")
    print()

    if not episodes_to_process:
        print("✅ All episodes already processed!")
        return

    # Confirm
    if not args.yes:
        response = input(f"Extract relationships from {len(episodes_to_process)} episodes? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    else:
        print(f"Auto-confirmed: Extracting relationships from {len(episodes_to_process)} episodes")

    # Initialize extractor
    extractor = ComprehensiveRelationshipExtractor()

    # Process episodes
    print()
    print("=" * 80)
    print("STARTING EXTRACTION")
    print("=" * 80)

    success_count = 0
    start_time = time.time()

    for i, episode_num in enumerate(episodes_to_process, 1):
        print(f"\n[{i}/{len(episodes_to_process)}] Episode {episode_num}")

        if process_episode(episode_num, extractor, transcripts_dir, output_dir):
            success_count += 1

        # Progress update
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = len(episodes_to_process) - i
            eta = remaining / rate if rate > 0 else 0
            print(f"\n⏱️  Progress: {i}/{len(episodes_to_process)} | Rate: {rate:.2f} ep/min | ETA: {eta/60:.1f} min")

    # Summary
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count}/{len(episodes_to_process)} episodes")
    print(f"⏱️  Total time: {(time.time() - start_time)/60:.1f} minutes")
    print()
    print("Next step: Rebuild visualization data with:")
    print("  python3 src/knowledge_graph/visualization/export_visualization.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
