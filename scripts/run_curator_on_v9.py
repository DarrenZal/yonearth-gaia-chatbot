#!/usr/bin/env python3
"""
Run KG Curator on V9 Reflector output

Transforms V9 Reflector analysis into executable changeset for V10

GOAL: Comprehensive General Knowledge Graph (not just discourse graph)
- Extract ALL factual relationships (aim for 900+ like V8's 1090)
- Maintain 100% attribution + classification from V9
- Reduce issues from 5.8% to <3% (A++ grade)
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
    print("🎨 RUNNING KG CURATOR ON V9 REFLECTOR OUTPUT")
    print("="*80)
    print()

    print("🎯 V10 GOAL: COMPREHENSIVE GENERAL KNOWLEDGE GRAPH")
    print("  Target: 900+ relationships (vs V9's 414, V8's 1,090)")
    print("  Target: <3% issues (A++ grade)")
    print("  Maintain: 100% attribution + classification")
    print("  Focus: ALL factual knowledge, not just discourse elements")
    print()

    # Paths
    v9_reflector_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/reflection_v9_reflector_fixes_20251013_043750.json")

    # Load V9 Reflector analysis
    print("📂 Loading V9 Reflector analysis...")
    with open(v9_reflector_path) as f:
        v9_reflector_data = json.load(f)

    print(f"✅ Loaded V9 Reflector analysis")
    print(f"   - Total relationships: {v9_reflector_data['extraction_metadata']['total_relationships']}")
    print(f"   - Total issues: {v9_reflector_data['quality_summary']['total_issues']}")
    print(f"   - Issue rate: {v9_reflector_data['quality_summary']['issue_rate_percent']}%")
    print(f"   - Critical: {v9_reflector_data['quality_summary']['critical_issues']}")
    print(f"   - High: {v9_reflector_data['quality_summary']['high_priority_issues']}")
    print(f"   - Recommendations: {len(v9_reflector_data.get('improvement_recommendations', []))}")
    print()

    print("⚠️  KEY ISSUE: V9 extracted only 414 relationships")
    print("   Compare to V8: 1,090 relationships")
    print("   Lost: 676 relationships (62% reduction!)")
    print("   V10 must be more comprehensive while maintaining quality")
    print()

    print("="*80)
    print("🎨 RUNNING CURATOR WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 3-5 minutes to analyze and generate changeset...")
    print()
    print("The Curator will:")
    print("  1. Analyze V9 Reflector recommendations")
    print("  2. Identify why V9 extracted so few relationships")
    print("  3. Generate changes for comprehensive extraction")
    print("  4. Fix the 24 true quality issues")
    print("  5. Maintain attribution + classification system")
    print()

    # Initialize Curator
    curator = KGCuratorAgent()

    # Add V10 goal context to reflector report
    v9_reflector_data['v10_goal_context'] = {
        'focus': 'COMPREHENSIVE GENERAL KNOWLEDGE GRAPH (not just discourse)',
        'v9_success': [
            '100% attribution coverage',
            '100% classification coverage',
            '99.3% classification accuracy'
        ],
        'v9_failure': [
            'Only 414 relationships (vs V8: 1,090)',
            'Lost 676 relationships (62% reduction)',
            'Too restrictive - missed factual knowledge'
        ],
        'v10_targets': {
            'relationships': '900+ (comprehensive extraction)',
            'high_confidence': '85%+',
            'issue_rate': '<3% (A++ grade)',
            'attribution': '100% (maintain)',
            'classification': '100% (maintain)'
        },
        'key_changes_needed': [
            'MORE PERMISSIVE Pass 1: Extract ALL factual relationships',
            'LESS HARSH Pass 2: Don\'t over-filter good relationships',
            'FIX ISSUES: Possessive pronouns (8), dedications (6), vague entities (10)',
            'KEEP V9 INNOVATIONS: Attribution, classification, list splitter inheritance'
        ]
    }

    # Run curation
    try:
        changeset = curator.curate_improvements(
            reflector_report=v9_reflector_data,
            current_version=9
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
        print(f"  Source version: V{metadata.get('source_version', 9)}")
        print(f"  Target version: V{metadata.get('target_version', 10)}")
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
    changeset_files = sorted(changesets_dir.glob("changeset_v9_to_v10_*.json"))
    if changeset_files:
        latest_changeset = changeset_files[-1]
        print(f"📁 Full changeset saved to: {latest_changeset}")
        print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print()
    print("1. Review changeset file to validate Curator's proposed changes")
    print("2. Focus on changes that increase extraction coverage")
    print("3. Apply changes to extract_kg_v10_book.py")
    print("4. Run V10 extraction targeting 900+ relationships")
    print("5. Run Reflector on V10 to validate A++ grade")
    print()
    print("TARGET METRICS FOR V10:")
    print("  - Relationships: 900+ (comprehensive)")
    print("  - High confidence: 85%+")
    print("  - Issue rate: <3% (A++ grade)")
    print("  - Attribution: 100%")
    print("  - Classification: 100%")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
