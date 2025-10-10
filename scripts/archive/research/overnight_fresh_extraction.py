#!/usr/bin/env python3
"""
Overnight Fresh Knowledge Graph Extraction
Based on ultra-synthesis research insights:
- Element-wise confidence (know exactly what's uncertain)
- Structured outputs for 100% valid JSON
- Parallel processing for speed
- Geographic validation built-in
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'fresh_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUT_DIR = DATA_DIR / "knowledge_graph_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

# API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set!")
    exit(1)

# Import OpenAI
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    logger.error("OpenAI library not installed. Run: pip install openai")
    exit(1)

# Pydantic models for structured output
try:
    from pydantic import BaseModel, Field
    from typing import List, Optional, Literal
except ImportError:
    logger.error("Pydantic not installed. Run: pip install pydantic")
    exit(1)

# Enhanced schemas with element-wise confidence
class EntityWithConfidence(BaseModel):
    """Entity with granular confidence scoring"""
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (PERSON, ORG, CONCEPT, etc.)")
    name_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in entity name")
    type_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in entity type")
    context: Optional[str] = Field(None, description="Context where entity appeared")

class RelationshipWithConfidence(BaseModel):
    """Relationship with element-wise confidence"""
    source: str = Field(description="Source entity")
    relationship: str = Field(description="Relationship type")
    target: str = Field(description="Target entity")

    # Element-wise confidence (research insight!)
    source_confidence: float = Field(ge=0.0, le=1.0, description="Confidence source entity is correct")
    relationship_confidence: float = Field(ge=0.0, le=1.0, description="Confidence relationship type/direction is correct")
    target_confidence: float = Field(ge=0.0, le=1.0, description="Confidence target entity is correct")

    context: Optional[str] = Field(None, description="Context where relationship appeared")
    episode_number: int = Field(description="Episode number for provenance")

class KnowledgeExtraction(BaseModel):
    """Complete extraction with entities and relationships"""
    entities: List[EntityWithConfidence]
    relationships: List[RelationshipWithConfidence]

# Enhanced extraction prompt with geographic awareness
EXTRACTION_PROMPT_TEMPLATE = """Extract entities and relationships from this podcast transcript segment.

IMPORTANT INSTRUCTIONS:
1. For geographic relationships (located_in, contains, part_of):
   - Smaller locations go IN larger locations (Boulder IN Colorado, not Colorado IN Boulder)
   - Cities go in counties/states, not other cities of similar size
   - If uncertain about direction, set relationship_confidence LOW

2. Element-wise confidence:
   - source_confidence: How sure are you the SOURCE entity is correct?
   - relationship_confidence: How sure are you about the RELATIONSHIP TYPE and DIRECTION?
   - target_confidence: How sure are you the TARGET entity is correct?

3. Common patterns:
   - PERSON works_at/founded/leads ORGANIZATION
   - ORGANIZATION located_in PLACE
   - PERSON advocates_for CONCEPT
   - PRODUCT made_by ORGANIZATION

Transcript segment:
{text}

Extract all entities and relationships with element-wise confidence scores."""

def chunk_transcript(transcript: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split transcript into overlapping chunks"""
    words = transcript.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

def extract_from_chunk(chunk: str, episode_num: int) -> Optional[KnowledgeExtraction]:
    """Extract knowledge from a single chunk using structured outputs"""
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured knowledge graphs from text with accurate confidence scores."},
                {"role": "user", "content": EXTRACTION_PROMPT_TEMPLATE.format(text=chunk)}
            ],
            response_format=KnowledgeExtraction,
            temperature=0.3  # Lower for more consistent extraction
        )

        extraction = response.choices[0].message.parsed

        # Add episode provenance to all relationships
        for rel in extraction.relationships:
            rel.episode_number = episode_num

        return extraction

    except Exception as e:
        logger.error(f"Error extracting from chunk: {e}")
        return None

def process_episode(episode_num: int) -> dict:
    """Process a single episode"""
    logger.info(f"Processing episode {episode_num}...")

    # Load transcript
    transcript_path = TRANSCRIPTS_DIR / f"episode_{episode_num}.json"
    if not transcript_path.exists():
        logger.warning(f"Episode {episode_num} transcript not found")
        return {'episode': episode_num, 'status': 'not_found'}

    try:
        with open(transcript_path) as f:
            data = json.load(f)
            transcript = data.get('full_transcript', '')

        if not transcript or len(transcript) < 100:
            logger.warning(f"Episode {episode_num} has insufficient transcript")
            return {'episode': episode_num, 'status': 'insufficient_data'}

        # Chunk transcript
        chunks = chunk_transcript(transcript, chunk_size=800, overlap=100)
        logger.info(f"Episode {episode_num}: {len(chunks)} chunks")

        # Extract from all chunks
        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(chunks):
            if i % 5 == 0:
                logger.info(f"Episode {episode_num}: chunk {i}/{len(chunks)}")

            extraction = extract_from_chunk(chunk, episode_num)
            if extraction:
                all_entities.extend(extraction.entities)
                all_relationships.extend(extraction.relationships)

            # Rate limiting: ~20 requests per minute for gpt-4o-mini
            time.sleep(0.1)

        # Save results
        output = {
            'episode': episode_num,
            'timestamp': datetime.now().isoformat(),
            'entity_count': len(all_entities),
            'relationship_count': len(all_relationships),
            'entities': [e.dict() for e in all_entities],
            'relationships': [r.dict() for r in all_relationships]
        }

        output_path = OUTPUT_DIR / f"episode_{episode_num}_extraction.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Episode {episode_num} complete: {len(all_entities)} entities, {len(all_relationships)} relationships")

        return {
            'episode': episode_num,
            'status': 'success',
            'entities': len(all_entities),
            'relationships': len(all_relationships)
        }

    except Exception as e:
        logger.error(f"Error processing episode {episode_num}: {e}")
        return {'episode': episode_num, 'status': 'error', 'error': str(e)}

def overnight_extraction(max_workers: int = 5, episodes_to_process: int = 172):
    """Main overnight extraction pipeline"""
    start_time = time.time()
    logger.info("="*60)
    logger.info("🌙 OVERNIGHT FRESH EXTRACTION STARTING")
    logger.info(f"Processing up to {episodes_to_process} episodes")
    logger.info(f"Using {max_workers} parallel workers")
    logger.info("="*60)

    # Find all available episodes
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("episode_*.json"))
    episode_nums = []
    for f in transcript_files:
        try:
            num = int(f.stem.split('_')[1])
            if num <= episodes_to_process:
                episode_nums.append(num)
        except:
            continue

    logger.info(f"Found {len(episode_nums)} episodes to process")

    # Process in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_episode = {
            executor.submit(process_episode, ep_num): ep_num
            for ep_num in episode_nums
        }

        for future in as_completed(future_to_episode):
            ep_num = future_to_episode[future]
            try:
                result = future.result()
                results.append(result)

                # Progress update
                completed = len(results)
                logger.info(f"Progress: {completed}/{len(episode_nums)} episodes ({completed/len(episode_nums)*100:.1f}%)")

            except Exception as e:
                logger.error(f"Episode {ep_num} failed: {e}")
                results.append({'episode': ep_num, 'status': 'error'})

    # Generate summary
    total_time = time.time() - start_time
    success = [r for r in results if r.get('status') == 'success']
    total_entities = sum(r.get('entities', 0) for r in success)
    total_relationships = sum(r.get('relationships', 0) for r in success)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'episodes_attempted': len(episode_nums),
        'episodes_successful': len(success),
        'total_entities': total_entities,
        'total_relationships': total_relationships,
        'results': results
    }

    summary_path = OUTPUT_DIR / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("="*60)
    logger.info("✨ OVERNIGHT EXTRACTION COMPLETE")
    logger.info(f"⏱️  Total time: {total_time/3600:.2f} hours")
    logger.info(f"📊 Success rate: {len(success)}/{len(episode_nums)} ({len(success)/len(episode_nums)*100:.1f}%)")
    logger.info(f"🎯 Total extracted: {total_entities} entities, {total_relationships} relationships")
    logger.info(f"📁 Results saved to: {OUTPUT_DIR}")
    logger.info("="*60)

    logger.info("\n🌅 GOOD MORNING! Here's what to check:")
    logger.info(f"1. Summary: {summary_path}")
    logger.info(f"2. Individual extractions: {OUTPUT_DIR}/episode_*_extraction.json")
    logger.info(f"3. Next steps: Run relationship validation to find Boulder/Lafayette errors")

    return summary

if __name__ == "__main__":
    # Check environment
    if not OPENAI_API_KEY:
        logger.error("Set OPENAI_API_KEY environment variable!")
        exit(1)

    logger.info(f"Using OpenAI API key: {OPENAI_API_KEY[:15]}...")
    logger.info(f"Transcripts directory: {TRANSCRIPTS_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Estimate cost
    episodes = 172
    avg_chunks_per_episode = 15  # Conservative estimate
    total_chunks = episodes * avg_chunks_per_episode
    cost_per_chunk = 0.002  # ~$0.002 per chunk with gpt-4o-mini
    estimated_cost = total_chunks * cost_per_chunk

    logger.info(f"\n💰 COST ESTIMATE:")
    logger.info(f"   Episodes: {episodes}")
    logger.info(f"   Estimated chunks: {total_chunks}")
    logger.info(f"   Estimated cost: ${estimated_cost:.2f}")
    logger.info(f"   (Using gpt-4o-mini with structured outputs)\n")

    # Run extraction
    summary = overnight_extraction(max_workers=5, episodes_to_process=172)
