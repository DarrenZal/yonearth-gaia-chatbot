#!/usr/bin/env python3
"""
Knowledge Graph Extraction v3.2.2 - FULL 172-Episode Production Run

Automatically skips episodes that already have output files.
This allows resuming from interruptions and avoids re-processing test episodes.

Usage:
    python3 scripts/extract_kg_v3_2_2_full.py

Features:
- Skips episodes with existing output files
- Processes all 172 episodes (0-172, excluding 26)
- Can resume if interrupted
- Saves progress continuously
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main extraction module
from scripts.extract_kg_v3_2_2 import (
    extract_knowledge_graph_v3_2_2,
    TRANSCRIPTS_DIR,
    OUTPUT_DIR,
    logger
)

import json
import time
from pathlib import Path
from datetime import datetime


def get_all_episode_numbers():
    """Get list of all episode numbers (0-172, excluding 26)"""
    episodes = list(range(0, 173))
    episodes.remove(26)  # Episode 26 doesn't exist
    return episodes


def get_processed_episodes():
    """Get list of episodes that already have output files"""
    processed = []
    for file in OUTPUT_DIR.glob("episode_*_v3_2_2.json"):
        try:
            ep_num = int(file.stem.split('_')[1])
            processed.append(ep_num)
        except (ValueError, IndexError):
            continue
    return sorted(processed)


def get_remaining_episodes():
    """Get list of episodes that need to be processed"""
    all_episodes = get_all_episode_numbers()
    processed = get_processed_episodes()
    remaining = [ep for ep in all_episodes if ep not in processed]
    return remaining


def episode_exists(episode_num: int) -> bool:
    """Check if episode transcript file exists"""
    transcript_path = TRANSCRIPTS_DIR / f"episode_{episode_num}.json"
    return transcript_path.exists()


def main():
    """Full 172-episode extraction with auto-skip of processed episodes"""

    logger.info("="*80)
    logger.info("🚀 KNOWLEDGE GRAPH v3.2.2 - FULL PRODUCTION EXTRACTION")
    logger.info("="*80)
    logger.info("")

    # Check status
    all_episodes = get_all_episode_numbers()
    processed = get_processed_episodes()
    remaining = get_remaining_episodes()

    logger.info(f"📊 STATUS:")
    logger.info(f"  Total episodes: {len(all_episodes)}")
    logger.info(f"  Already processed: {len(processed)}")
    logger.info(f"  Remaining to process: {len(remaining)}")
    logger.info("")

    if processed:
        logger.info(f"✅ Already processed: {processed[:10]}{'...' if len(processed) > 10 else ''}")
        logger.info("")

    if not remaining:
        logger.info("🎉 ALL EPISODES ALREADY PROCESSED!")
        logger.info(f"📁 Results in: {OUTPUT_DIR}")
        return

    # Estimate time
    avg_time_per_episode = 12.8  # minutes (from test run)
    estimated_hours = (len(remaining) * avg_time_per_episode) / 60

    logger.info(f"⏱️  ESTIMATED TIME: {estimated_hours:.1f} hours ({len(remaining)} episodes × {avg_time_per_episode} min)")
    logger.info("")
    logger.info(f"🎯 PROCESSING: {len(remaining)} episodes")
    logger.info(f"📁 OUTPUT: {OUTPUT_DIR}")
    logger.info("")
    logger.info("="*80)
    logger.info("")

    # Confirmation
    logger.info("⚠️  This will process all remaining episodes. Press Ctrl+C to cancel.")
    logger.info("")
    time.sleep(3)

    run_id = f"full_v3_2_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    all_results = []
    start_time = time.time()
    successful = 0
    failed = []

    for i, ep_num in enumerate(remaining, 1):
        logger.info(f"{'='*80}")
        logger.info(f"📍 EPISODE {ep_num} ({i}/{len(remaining)}) - {(i/len(remaining)*100):.1f}% complete")
        logger.info(f"{'='*80}")

        # Check if transcript exists
        transcript_path = TRANSCRIPTS_DIR / f"episode_{ep_num}.json"
        if not transcript_path.exists():
            logger.warning(f"⚠️  Episode {ep_num} transcript not found, skipping")
            failed.append(ep_num)
            continue

        # Load transcript
        try:
            with open(transcript_path) as f:
                data = json.load(f)
                transcript = data.get('full_transcript', '')

            if not transcript or len(transcript) < 100:
                logger.warning(f"⚠️  Episode {ep_num} has insufficient data, skipping")
                failed.append(ep_num)
                continue
        except Exception as e:
            logger.error(f"❌ Error loading episode {ep_num}: {e}")
            failed.append(ep_num)
            continue

        # Extract
        try:
            results = extract_knowledge_graph_v3_2_2(
                episode_num=ep_num,
                transcript=transcript,
                run_id=run_id,
                batch_size=50
            )

            all_results.append(results)

            # Save individual results
            output_path = OUTPUT_DIR / f"episode_{ep_num}_v3_2_2.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"✅ Episode {ep_num} complete, saved to {output_path}")
            successful += 1

            # Save progress summary after each episode (for resumability)
            progress_summary = {
                'run_id': run_id,
                'version': 'v3.2.2',
                'timestamp': datetime.now().isoformat(),
                'total_episodes': len(all_episodes),
                'processed_before_run': len(processed),
                'processed_this_run': successful,
                'remaining': len(remaining) - i,
                'failed': failed,
                'elapsed_hours': (time.time() - start_time) / 3600
            }

            progress_path = OUTPUT_DIR / f"progress_{run_id}.json"
            with open(progress_path, 'w') as f:
                json.dump(progress_summary, f, indent=2)

            logger.info("")

        except Exception as e:
            logger.error(f"❌ Error processing episode {ep_num}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed.append(ep_num)
            continue

    total_time = time.time() - start_time

    # Final summary
    logger.info("="*80)
    logger.info("✨ FULL EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/3600:.1f} hours")
    logger.info(f"✅ Successfully processed: {successful}/{len(remaining)}")
    logger.info(f"📊 Total in database: {len(processed) + successful}/{len(all_episodes)}")

    if failed:
        logger.info(f"❌ Failed episodes: {failed}")

    logger.info(f"📁 Results saved to: {OUTPUT_DIR}")
    logger.info("="*80)

    # Generate final summary
    final_summary = {
        'run_id': run_id,
        'version': 'v3.2.2',
        'timestamp': datetime.now().isoformat(),
        'total_episodes': len(all_episodes),
        'processed_before_run': len(processed),
        'processed_this_run': successful,
        'failed_episodes': failed,
        'total_time_hours': total_time / 3600,
        'results': all_results
    }

    summary_path = OUTPUT_DIR / f"summary_{run_id}.json"
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)

    logger.info(f"📄 Summary: {summary_path}")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("="*80)
        logger.info("⚠️  EXTRACTION INTERRUPTED BY USER")
        logger.info("="*80)
        logger.info("ℹ️  Progress has been saved. Re-run this script to resume.")
        logger.info("   Already-processed episodes will be automatically skipped.")
        logger.info("="*80)
