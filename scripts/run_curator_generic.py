#!/usr/bin/env python3
"""
GENERIC Curator Runner - Works on ANY version

Finds the latest Reflector analysis and runs Curator to generate improvements.
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


def find_latest_reflector_analysis():
    """Find the most recent Reflector analysis."""
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")

    if not analysis_dir.exists():
        print(f"❌ Analysis directory not found: {analysis_dir}")
        return None

    # Find all reflection files
    reflection_files = sorted(analysis_dir.glob("reflection_*.json"))

    if not reflection_files:
        print("❌ No Reflector analyses found")
        return None

    # Get the latest one
    latest = reflection_files[-1]
    print(f"📁 Found latest Reflector analysis: {latest.name}")

    return latest


def main():
    print("="*80)
    print("🎨 RUNNING KG CURATOR (GENERIC)")
    print("="*80)
    print("Finding latest Reflector analysis...")
    print()

    # Find latest analysis
    analysis_path = find_latest_reflector_analysis()

    if not analysis_path:
        print("❌ Cannot run Curator without Reflector analysis")
        print("Run Reflector first: python scripts/run_reflector_generic.py")
        return

    # Load analysis
    print(f"📂 Loading Reflector analysis from {analysis_path.name}...")
    with open(analysis_path) as f:
        reflector_analysis = json.load(f)

    # Extract version info
    version_str = reflector_analysis['extraction_metadata']['version']
    total_rels = reflector_analysis['extraction_metadata']['total_relationships']
    quality = reflector_analysis['quality_summary']

    # Parse version number from string (e.g., "v9_reflector_fixes" -> 9)
    import re
    version_match = re.search(r'v(\d+)', version_str)
    if version_match:
        current_version = int(version_match.group(1))
    else:
        print(f"⚠️  Could not parse version from '{version_str}', using 10")
        current_version = 10

    print(f"✅ Loaded analysis for {version_str}")
    print(f"   - Parsed version: V{current_version}")
    print(f"   - Total relationships: {total_rels}")
    print(f"   - Issues: {quality['total_issues']} ({quality['issue_rate_percent']}%)")
    print(f"   - Grade: {quality['grade_confirmed']}")
    print()

    # Initialize Curator
    print("="*80)
    print("🎨 GENERATING IMPROVEMENTS WITH CURATOR...")
    print("="*80)
    print("This will take 2-3 minutes...")
    print()

    curator = KGCuratorAgent()

    # Run Curator
    improvements = curator.curate_improvements(
        reflector_report=reflector_analysis,  # Changed from reflector_analysis
        current_version=current_version  # Now an integer
    )

    print()
    print("="*80)
    print("✅ CURATOR ANALYSIS COMPLETE")
    print("="*80)
    print()

    # Display summary
    if 'file_operations' in improvements:
        print("PROPOSED CHANGES:")
        print(f"  Target version: V{improvements.get('metadata', {}).get('target_version', 'unknown')}")
        print(f"  Total operations: {len(improvements.get('file_operations', []))}")
        print()

        # Group by type
        code_fixes = [op for op in improvements.get('file_operations', []) if op.get('operation_type') == 'CODE_FIX']
        prompt_fixes = [op for op in improvements.get('file_operations', []) if op.get('operation_type') == 'PROMPT_ENHANCEMENT']

        print(f"  Code fixes: {len(code_fixes)}")
        for op in code_fixes[:5]:
            print(f"    [{op.get('priority', '?')}] {op.get('file_path', 'unknown')}")
            print(f"        {op.get('rationale', 'no rationale')[:70]}...")

        print()
        print(f"  Prompt enhancements: {len(prompt_fixes)}")
        for op in prompt_fixes[:5]:
            print(f"    [{op.get('priority', '?')}] {op.get('file_path', 'unknown')}")
            print(f"        {op.get('rationale', 'no rationale')[:70]}...")

        print()

    if 'expected_impact' in improvements:
        print("EXPECTED IMPACT:")
        impact = improvements['expected_impact']
        print(f"  Issues fixed: {impact.get('issues_fixed', 0)}")
        print(f"  Estimated new error rate: {impact.get('estimated_error_rate', 'unknown')}")
        print(f"  Target grade: {impact.get('target_grade', 'unknown')}")
        print()

    # Find the saved changeset file
    changeset_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/changesets")
    changeset_files = sorted(changeset_dir.glob("changeset_*.json"))
    if changeset_files:
        latest_changeset = changeset_files[-1]
        print(f"📁 Full changeset saved to: {latest_changeset}")
        print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print()
    print("Option 1: LET CURATOR APPLY CHANGES (Autonomous ACE)")
    print("  - Run: python scripts/apply_changeset_generic.py")
    print("  - Curator will automatically apply all changes")
    print("  - System creates new version and runs extraction")
    print()
    print("Option 2: MANUAL REVIEW AND APPLICATION")
    print("  - Review changeset file above")
    print("  - Manually apply changes you approve")
    print("  - Test the new version")
    print()
    print("🎯 Recommended: Use Option 1 for true ACE autonomy")
    print("="*80)


if __name__ == "__main__":
    main()
