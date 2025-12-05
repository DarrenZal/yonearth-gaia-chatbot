#!/usr/bin/env python3
"""
Master pipeline for Phase 8: Post-Processing & Unified Graph Rebuild.

Runs all steps in sequence:
1. Download batch results (if not already downloaded)
2. Process results through quality filters
3. Deduplicate entities
4. Build unified graph

Usage:
    python scripts/run_phase8_pipeline.py
    python scripts/run_phase8_pipeline.py --skip-download
    python scripts/run_phase8_pipeline.py --dry-run
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def run_command(cmd: str, description: str, dry_run: bool = False) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command to execute
        description: Human-readable description
        dry_run: If True, just print the command without executing

    Returns:
        True if command succeeded, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd}\n")

    if dry_run:
        print("[DRY RUN - Command not executed]")
        return True

    # Run the command and capture output
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=Path(__file__).parent.parent  # Run from project root
    )

    return result.returncode == 0


def check_batch_status() -> dict:
    """Check if batch results are ready for download."""
    state_path = Path("data/batch_jobs/batch_state.json")
    if not state_path.exists():
        return {"ready": False, "error": "No batch state found"}

    import json
    with open(state_path) as f:
        state = json.load(f)

    # Check if results already downloaded
    results_path = Path("data/batch_jobs/results/extraction_results.json")
    if results_path.exists():
        return {"ready": True, "downloaded": True, "state": state}

    return {"ready": True, "downloaded": False, "state": state}


def print_summary(success: bool, start_time: datetime, steps_completed: list):
    """Print pipeline execution summary."""
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    if success:
        print("PHASE 8 PIPELINE COMPLETE")
    else:
        print("PHASE 8 PIPELINE FAILED")
    print("=" * 60)

    print(f"\nDuration: {duration}")
    print(f"Steps completed: {len(steps_completed)}")
    for step in steps_completed:
        print(f"  - {step}")

    if success:
        print("\nOutputs:")
        outputs = [
            "data/knowledge_graph_unified/entities_processed.json",
            "data/knowledge_graph_unified/entities_deduplicated.json",
            "data/knowledge_graph_unified/relationships_processed.json",
            "data/knowledge_graph_unified/unified_v2.json",
            "data/knowledge_graph_unified/graph_stats_v2.json",
        ]
        for output in outputs:
            path = Path(output)
            if path.exists():
                size_kb = path.stat().st_size / 1024
                print(f"  - {output} ({size_kb:.1f} KB)")
            else:
                print(f"  - {output} (not created)")

        print("\nNext: Run Phase 9 (GraphRAG & Visualization Update)")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 8 pipeline: Post-Processing & Unified Graph Rebuild",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (including download)
  python scripts/run_phase8_pipeline.py

  # Skip download step (if results already downloaded)
  python scripts/run_phase8_pipeline.py --skip-download

  # Dry run (show what would be executed)
  python scripts/run_phase8_pipeline.py --dry-run

  # Just check batch status
  python scripts/run_phase8_pipeline.py --status
        """
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip batch download step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--status", action="store_true",
                        help="Just check batch status and exit")

    args = parser.parse_args()

    # Change to project root
    os.chdir(Path(__file__).parent.parent)

    # Check status
    status = check_batch_status()

    if args.status:
        print("=" * 60)
        print("BATCH STATUS")
        print("=" * 60)
        if status.get("error"):
            print(f"\nError: {status['error']}")
            print("\nSubmit a batch job first:")
            print("  python scripts/extract_episodes_batch.py --submit")
        else:
            state = status.get("state", {})
            print(f"\nSubmitted: {state.get('timestamp', 'N/A')}")
            print(f"Episodes: {state.get('episodes_processed', 'N/A')}")
            print(f"Books: {state.get('books_processed', 'N/A')}")
            print(f"Parent chunks: {state.get('parent_count', 'N/A')}")
            print(f"Child chunks: {state.get('child_count', 'N/A')}")

            if status.get("downloaded"):
                print("\nResults: Downloaded and ready")
            else:
                print("\nResults: Not yet downloaded")
                print("  Run: python scripts/extract_episodes_batch.py --poll")
                print("  Then: python scripts/extract_episodes_batch.py --download")
        return

    # Verify batch results are available or will be downloaded
    if not status.get("ready"):
        print(f"Error: {status.get('error')}")
        print("\nSubmit a batch job first:")
        print("  python scripts/extract_episodes_batch.py --submit")
        sys.exit(1)

    start_time = datetime.now()
    steps_completed = []
    steps = []

    # Step 1: Download batch results (if needed)
    if not args.skip_download and not status.get("downloaded"):
        steps.append((
            "python scripts/extract_episodes_batch.py --download",
            "Download batch extraction results"
        ))
    elif args.skip_download:
        print("\nSkipping download step (--skip-download specified)")
        if not status.get("downloaded"):
            print("Warning: Results not yet downloaded. Step 2 may fail.")
    else:
        print("\nSkipping download step (results already downloaded)")

    # Step 2: Process through quality filters
    steps.append((
        "python scripts/process_batch_results.py",
        "Process results through quality filters"
    ))

    # Step 3: Deduplicate entities
    steps.append((
        "python scripts/deduplicate_entities.py",
        "Deduplicate entities"
    ))

    # Step 4: Build unified graph
    steps.append((
        "python scripts/build_unified_graph_v2.py",
        "Build unified knowledge graph"
    ))

    # Run all steps
    all_success = True
    for cmd, description in steps:
        success = run_command(cmd, description, dry_run=args.dry_run)
        if success:
            steps_completed.append(description)
        else:
            all_success = False
            print(f"\n FAILED: {description}")
            print("Pipeline stopped. Fix the error and re-run.")
            break

    print_summary(all_success, start_time, steps_completed)

    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
