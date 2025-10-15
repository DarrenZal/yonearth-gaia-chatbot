#!/usr/bin/env python3
"""
Generic Changeset Applicator - Works on ANY changeset

Finds the latest changeset and applies changes using the intelligent Applicator.
This is the ACE way - autonomous, version-agnostic.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from src.ace_kg.kg_curator import KGCuratorAgent


def find_latest_changeset():
    """Find the most recent changeset."""
    changeset_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/changesets")

    if not changeset_dir.exists():
        print(f"❌ Changeset directory not found: {changeset_dir}")
        return None

    # Find all changeset files and sort by modification time (newest first)
    changeset_files = sorted(
        changeset_dir.glob("changeset_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not changeset_files:
        print("❌ No changesets found")
        return None

    # Get the latest one (first in list since we sorted newest first)
    latest = changeset_files[0]
    print(f"📁 Found latest changeset: {latest.name} (most recent by timestamp)")

    return latest


def main():
    print("="*80)
    print("⚙️  APPLYING CHANGESET (GENERIC)")
    print("="*80)
    print("Finding latest changeset...")
    print()

    # Find latest changeset
    changeset_path = find_latest_changeset()

    if not changeset_path:
        print("❌ Cannot apply changeset without a changeset file")
        print("Run Curator first: python scripts/run_curator_generic.py")
        return

    # Load changeset
    print(f"📂 Loading changeset from {changeset_path.name}...")
    with open(changeset_path) as f:
        changeset = json.load(f)

    # Extract version info
    source_version = changeset['metadata']['source_version']
    target_version = changeset['metadata']['target_version']
    total_changes = len(changeset.get('file_operations', []))

    print(f"✅ Loaded changeset: V{source_version} → V{target_version}")
    print(f"   - Total operations: {total_changes}")
    print(f"   - Estimated impact: {changeset['changeset_metadata'].get('estimated_impact', 'N/A')}")
    print()

    # Ask for confirmation
    print("="*80)
    print("⚠️  APPLY CHANGES?")
    print("="*80)
    print()
    print(f"This will apply {total_changes} changes to the knowledge graph extraction system.")
    print("Operations will be applied using the intelligent Applicator (Claude-powered).")
    print()

    # Group by priority
    critical = [op for op in changeset.get('file_operations', []) if op.get('priority') == 'CRITICAL']
    high = [op for op in changeset.get('file_operations', []) if op.get('priority') == 'HIGH']
    medium = [op for op in changeset.get('file_operations', []) if op.get('priority') == 'MEDIUM']
    low = [op for op in changeset.get('file_operations', []) if op.get('priority') == 'LOW']

    print("CHANGES BY PRIORITY:")
    if critical:
        print(f"  🔴 CRITICAL: {len(critical)} operations")
        for op in critical[:3]:
            print(f"     - {op.get('file_path', 'unknown')}: {op.get('rationale', '')[:60]}...")
    if high:
        print(f"  🟠 HIGH: {len(high)} operations")
        for op in high[:3]:
            print(f"     - {op.get('file_path', 'unknown')}: {op.get('rationale', '')[:60]}...")
    if medium:
        print(f"  🟡 MEDIUM: {len(medium)} operations")
    if low:
        print(f"  🟢 LOW: {len(low)} operations")
    print()

    # Dry run first
    print("Running DRY RUN to preview changes...")
    print()

    curator = KGCuratorAgent()

    # Dry run
    dry_results = curator.apply_changeset(changeset, dry_run=True)

    print()
    print("="*80)
    print("PROCEED WITH ACTUAL APPLICATION?")
    print("="*80)
    print()

    response = input("Type 'yes' to apply changes, 'no' to cancel: ").strip().lower()

    if response != 'yes':
        print("❌ Cancelled by user")
        return

    # Apply changes
    print()
    print("="*80)
    print("⚡ APPLYING CHANGES WITH INTELLIGENT APPLICATOR")
    print("="*80)
    print()
    print("This may take several minutes as Claude reads files and makes strategic changes...")
    print()

    results = curator.apply_changeset(
        changeset,
        dry_run=False,
        auto_apply_low_risk=True  # Auto-apply low-risk changes
    )

    print()
    print("="*80)
    print("✅ CHANGESET APPLICATION COMPLETE")
    print("="*80)
    print()

    # Summary
    print("RESULTS:")
    print(f"  ✅ Applied: {len(results['applied'])}")
    print(f"  ⏸️  Requires Approval: {len(results['requires_approval'])}")
    print(f"  ⏭️  Skipped: {len(results['skipped'])}")
    print(f"  ❌ Failed: {len(results['failed'])}")
    print()

    if results['failed']:
        print("FAILURES:")
        for failure in results['failed']:
            print(f"  ❌ {failure['operation'].get('operation_id', 'unknown')}: {failure['error']}")
        print()

    if results['requires_approval']:
        print("REQUIRES MANUAL APPROVAL:")
        for op in results['requires_approval']:
            print(f"  ⏸️  {op.get('operation_id', 'unknown')}: {op.get('file_path', 'unknown')}")
            print(f"      Risk: {op.get('risk_level', 'unknown')}")
            print(f"      Rationale: {op.get('rationale', 'N/A')[:80]}...")
        print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print()
    print(f"1. Run V{target_version} extraction:")
    print(f"   python3 scripts/extract_kg_v{target_version}_book.py")
    print()
    print("2. Run Reflector on V{target_version} to validate improvements:")
    print("   python3 scripts/run_reflector_generic.py")
    print()
    print("3. Compare results to see if quality improved:")
    print(f"   V{source_version}: {changeset.get('quality_summary', {}).get('issue_rate_percent', 'N/A')}% error rate")
    print(f"   V{target_version}: Expected <{changeset.get('expected_impact', {}).get('estimated_error_rate', 'N/A')}")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
