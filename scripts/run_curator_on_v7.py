#!/usr/bin/env python3
"""
Run KG Curator on V7 Reflector output

Transforms V7 Reflector analysis into executable changeset for V8
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


def main():
    print("="*80)
    print("🎨 RUNNING KG CURATOR ON V7 REFLECTOR OUTPUT")
    print("="*80)
    print()

    # Paths
    v7_reflector_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/reflection_v7_meta_ace_20251012_224756_fixed.json")

    # Load V7 Reflector analysis (FULL version with all detailed examples!)
    print("📂 Loading V7 Reflector analysis (FULL with detailed examples)...")
    with open(v7_reflector_path) as f:
        v7_reflector_data = json.load(f)

    print(f"✅ Loaded V7 Reflector analysis")
    print(f"   - Total issues: {v7_reflector_data['quality_summary']['total_issues']}")
    print(f"   - Critical: {v7_reflector_data['quality_summary']['critical_issues']}")
    print(f"   - High: {v7_reflector_data['quality_summary']['high_priority_issues']}")
    print(f"   - Recommendations: {len(v7_reflector_data.get('improvement_recommendations', []))}")
    print()

    print("="*80)
    print("🎨 RUNNING CURATOR WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 3-5 minutes to analyze and generate changeset...")
    print()
    print("The Curator will:")
    print("  1. Analyze V7 Reflector recommendations")
    print("  2. Review current extraction code structure")
    print("  3. Generate specific code/prompt/config changes")
    print("  4. Prioritize by risk and impact")
    print("  5. Create testing strategy")
    print()

    # Initialize Curator
    curator = KGCuratorAgent()

    # Run curation
    try:
        changeset = curator.curate_improvements(
            reflector_report=v7_reflector_data,
            current_version=7
        )
    except Exception as e:
        print(f"❌ Curator failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("="*80)
    print("✅ CURATOR CHANGESET GENERATED")
    print("="*80)
    print()

    # Display changeset summary
    if "changeset_metadata" in changeset:
        metadata = changeset["changeset_metadata"]
        print("CHANGESET METADATA:")
        print(f"  Source version: V{metadata.get('source_version', 7)}")
        print(f"  Target version: V{metadata.get('target_version', 8)}")
        print(f"  Total changes: {metadata.get('total_changes', 'N/A')}")
        print(f"  Estimated impact: {metadata.get('estimated_impact', 'N/A')}")
        print()

    if "file_operations" in changeset:
        print(f"FILE OPERATIONS PROPOSED: {len(changeset['file_operations'])}")
        print()

        # Group by priority
        critical = [op for op in changeset['file_operations'] if op.get('priority') == 'CRITICAL']
        high = [op for op in changeset['file_operations'] if op.get('priority') == 'HIGH']
        medium = [op for op in changeset['file_operations'] if op.get('priority') == 'MEDIUM']
        low = [op for op in changeset['file_operations'] if op.get('priority') == 'LOW']

        print(f"CRITICAL ({len(critical)}):")
        for op in critical:
            print(f"  - [{op['operation_id']}] {op['operation_type']}: {op['file_path']}")
            print(f"    → {op['rationale'][:80]}...")
            print(f"    → Risk: {op['risk_level']}, Expected: {op.get('expected_improvement', 'N/A')}")
        print()

        print(f"HIGH ({len(high)}):")
        for op in high:
            print(f"  - [{op['operation_id']}] {op['operation_type']}: {op['file_path']}")
            print(f"    → {op['rationale'][:80]}...")
        print()

        print(f"MEDIUM ({len(medium)}):")
        for op in medium[:3]:  # Show first 3
            print(f"  - [{op['operation_id']}] {op['operation_type']}: {op['file_path']}")
        if len(medium) > 3:
            print(f"  ... and {len(medium) - 3} more")
        print()

        print(f"LOW ({len(low)}):")
        print(f"  {len(low)} low-priority optimizations")
        print()

    if "testing_strategy" in changeset:
        print("TESTING STRATEGY:")
        testing = changeset["testing_strategy"]
        if "unit_tests" in testing:
            print(f"  Unit tests: {len(testing['unit_tests'])} tests")
        if "integration_tests" in testing:
            print(f"  Integration tests: {len(testing['integration_tests'])} tests")
        if "success_criteria" in testing:
            print(f"  Success criteria: {len(testing['success_criteria'])} metrics")
        print()

    # Find the saved changeset file
    changesets_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/changesets")
    changeset_files = sorted(changesets_dir.glob("changeset_v7_to_v8_*.json"))
    if changeset_files:
        latest_changeset = changeset_files[-1]
        print(f"📁 Full changeset saved to: {latest_changeset}")
        print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print()
    print("1. Review changeset file to validate Curator's proposed changes")
    print("2. Apply low-risk changes automatically (or manually review all)")
    print("3. Test changes on sample chunk before full extraction")
    print("4. Run V8 extraction")
    print("5. Run Reflector on V8 to validate improvements")
    print()
    print("Commands:")
    print("  # Review changeset")
    print(f"  cat {latest_changeset}")
    print()
    print("  # Apply changeset (dry run first)")
    print("  python scripts/apply_changeset_v7_to_v8.py --dry-run")
    print()
    print("  # Apply changeset (auto-apply low-risk)")
    print("  python scripts/apply_changeset_v7_to_v8.py --auto-apply-low-risk")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
