#!/usr/bin/env python3
"""
Retry Knowledge Graph Extraction for Failed Episodes

Retries the 16 episodes that failed during the initial extraction run
due to the candidate_uid mismatch issue.

Failed Episodes: 13, 24, 32, 34, 51, 59, 71, 83, 103, 109, 112, 121, 133, 155, 164, 170

Root Causes (now fixed):
1. TypeError: sequence item 3: expected str instance, NoneType found
   - OpenAI API returned mismatched candidate_uid values
   - Code couldn't join results back to original candidates
   - doc_sha256 was None, causing crash in generate_claim_uid()

2. Token limit exceeded (episodes 32, 133)
   - Some chunks produced 16K+ completion tokens
   - Structured output parsing failed mid-response

Fixes Applied:
1. Skip evaluations where candidate lookup fails (line 615-617 in extract_kg_v3_2_2.py)
2. Handle None doc_sha256 gracefully (line 248-249 in extract_kg_v3_2_2.py)
3. Add max_completion_tokens=10000 limit to prevent overflow (line 421 in extract_kg_v3_2_2.py)
4. Better error handling for token limit errors (line 452-458 in extract_kg_v3_2_2.py)
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import extraction module
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_kg_v3_2_2 import extract_knowledge_graph_v3_2_2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_extraction_retry_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUT_DIR = DATA_DIR / "knowledge_graph_v3_2_2"
OUTPUT_DIR.mkdir(exist_ok=True)

# Failed episodes from initial run
FAILED_EPISODES = [13, 24, 32, 34, 51, 59, 71, 83, 103, 109, 112, 121, 133, 155, 164, 170]


def main():
    """Retry extraction for failed episodes"""
    logger.info("=" * 80)
    logger.info("🔄 RETRY KNOWLEDGE GRAPH EXTRACTION - FAILED EPISODES")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Episodes to retry: {len(FAILED_EPISODES)}")
    logger.info(f"Episode list: {FAILED_EPISODES}")
    logger.info("")
    logger.info("Fixes Applied:")
    logger.info("  ✓ Skip evaluations with mismatched candidate_uid")
    logger.info("  ✓ Handle None doc_sha256 gracefully")
    logger.info("=" * 80)
    logger.info("")

    run_id = f"retry_v3_2_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results_summary = {
        'run_id': run_id,
        'version': 'v3.2.2',
        'timestamp': datetime.now().isoformat(),
        'total_episodes': len(FAILED_EPISODES),
        'successful': [],
        'failed': [],
        'results': []
    }

    start_time = time.time()

    for i, ep_num in enumerate(FAILED_EPISODES, 1):
        logger.info("=" * 80)
        logger.info(f"📍 EPISODE {ep_num} ({i}/{len(FAILED_EPISODES)}) - {i/len(FAILED_EPISODES)*100:.1f}% complete")
        logger.info("=" * 80)

        try:
            # Load transcript
            transcript_path = TRANSCRIPTS_DIR / f"episode_{ep_num}.json"
            if not transcript_path.exists():
                logger.error(f"❌ Transcript not found: {transcript_path}")
                results_summary['failed'].append({
                    'episode': ep_num,
                    'reason': 'Transcript file not found'
                })
                continue

            with open(transcript_path) as f:
                data = json.load(f)
                transcript = data.get('full_transcript', '')

            if not transcript or len(transcript) < 100:
                logger.error(f"❌ Episode {ep_num} has insufficient data")
                results_summary['failed'].append({
                    'episode': ep_num,
                    'reason': 'Insufficient transcript data'
                })
                continue

            # Extract with fixed code
            logger.info(f"🚀 Starting extraction for episode {ep_num}...")
            results = extract_knowledge_graph_v3_2_2(
                episode_num=ep_num,
                transcript=transcript,
                run_id=run_id,
                batch_size=50
            )

            # Save results
            output_path = OUTPUT_DIR / f"episode_{ep_num}_v3_2_2.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"✅ Episode {ep_num} SUCCESSFUL")
            logger.info(f"   📊 Relationships: {len(results.get('relationships', []))}")
            logger.info(f"   💾 Saved to: {output_path}")
            logger.info("")

            results_summary['successful'].append(ep_num)
            results_summary['results'].append({
                'episode': ep_num,
                'relationships_count': len(results.get('relationships', [])),
                'high_confidence': results.get('high_confidence_count', 0),
                'output_file': str(output_path)
            })

        except Exception as e:
            logger.error(f"❌ Error processing episode {ep_num}: {e}")
            logger.error(f"   Traceback:", exc_info=True)
            results_summary['failed'].append({
                'episode': ep_num,
                'reason': str(e)
            })

    total_time = time.time() - start_time

    # Save summary
    results_summary['total_time_seconds'] = total_time
    results_summary['success_count'] = len(results_summary['successful'])
    results_summary['failed_count'] = len(results_summary['failed'])

    summary_path = OUTPUT_DIR / f"summary_retry_{run_id}.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Print final summary
    logger.info("=" * 80)
    logger.info("✨ RETRY EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"✅ Successful: {len(results_summary['successful'])}/{len(FAILED_EPISODES)}")
    logger.info(f"❌ Failed: {len(results_summary['failed'])}/{len(FAILED_EPISODES)}")
    logger.info("")

    if results_summary['successful']:
        logger.info(f"✅ Successfully extracted episodes: {results_summary['successful']}")
        total_relationships = sum(r['relationships_count'] for r in results_summary['results'])
        logger.info(f"   📊 Total relationships: {total_relationships}")

    if results_summary['failed']:
        logger.info(f"❌ Failed episodes: {[f['episode'] for f in results_summary['failed']]}")
        for fail in results_summary['failed']:
            logger.info(f"   Episode {fail['episode']}: {fail['reason']}")

    logger.info("")
    logger.info(f"📄 Summary saved to: {summary_path}")
    logger.info("=" * 80)

    return results_summary


if __name__ == "__main__":
    main()
