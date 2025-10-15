#!/usr/bin/env python3
"""
Run KG Curator on V11.2.2 Reflector analysis to generate V12 improvements.

Meta-ACE iteration: After evaluating Curator's output, may improve Curator and re-run.
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
    print("🎨 RUNNING KG CURATOR ON V11.2.2 ANALYSIS")
    print("="*80)
    print()

    # Paths
    v11_2_2_analysis_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/reflection_v11.2.2_20251014_011329.json")

    if not v11_2_2_analysis_path.exists():
        print(f"❌ V11.2.2 Reflector analysis not found: {v11_2_2_analysis_path}")
        return

    # Load V11.2.2 Reflector analysis
    print(f"📂 Loading V11.2.2 Reflector analysis...")
    with open(v11_2_2_analysis_path) as f:
        reflector_analysis = json.load(f)

    # Extract metadata
    version_str = reflector_analysis['extraction_metadata']['version']
    total_rels = reflector_analysis['extraction_metadata']['total_relationships']
    quality = reflector_analysis['quality_summary']

    print(f"✅ Loaded analysis for {version_str}")
    print(f"   - Total relationships: {total_rels}")
    print(f"   - Issues: {quality['total_issues']} ({quality['issue_rate_percent']}%)")
    print(f"   - Grade: {quality['grade_confirmed']}")
    print()

    # Parse version number for next version
    # V11.2.2 → V12
    current_version = 11  # Simplified to V11 for naming purposes
    target_version = 12

    print("="*80)
    print("🎨 GENERATING V12 IMPROVEMENTS WITH CURATOR...")
    print("="*80)
    print("This will take 2-3 minutes...")
    print()

    # Initialize Curator
    curator = KGCuratorAgent()

    # Run Curator
    improvements = curator.curate_improvements(
        reflector_report=reflector_analysis,
        current_version=current_version
    )

    print()
    print("="*80)
    print("✅ CURATOR ANALYSIS COMPLETE")
    print("="*80)
    print()

    # Display summary
    if 'file_operations' in improvements:
        print("PROPOSED CHANGES FOR V12:")
        print(f"  Source version: V11.2.2")
        print(f"  Target version: V{improvements.get('metadata', {}).get('target_version', 12)}")
        print(f"  Total operations: {len(improvements.get('file_operations', []))}")
        print()

        # Group by type
        code_fixes = [op for op in improvements.get('file_operations', []) if op.get('operation_type') == 'CODE_FIX']
        prompt_fixes = [op for op in improvements.get('file_operations', []) if op.get('operation_type') == 'PROMPT_ENHANCEMENT']

        print(f"  Code fixes: {len(code_fixes)}")
        for op in code_fixes[:10]:
            print(f"    [{op.get('priority', '?')}] {op.get('file_path', 'unknown')}")
            rationale = op.get('rationale', 'no rationale')
            print(f"        {rationale[:80]}{'...' if len(rationale) > 80 else ''}")

        print()
        print(f"  Prompt enhancements: {len(prompt_fixes)}")
        for op in prompt_fixes[:10]:
            print(f"    [{op.get('priority', '?')}] {op.get('file_path', 'unknown')}")
            rationale = op.get('rationale', 'no rationale')
            print(f"        {rationale[:80]}{'...' if len(rationale) > 80 else ''}")

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
    changeset_files = sorted(changeset_dir.glob("changeset_*.json"), key=lambda p: p.stat().st_mtime)
    if changeset_files:
        latest_changeset = changeset_files[-1]
        print(f"📁 Full changeset saved to: {latest_changeset}")
        print()

    print("="*80)
    print("🎯 NEXT STEPS (META-ACE ITERATION)")
    print("="*80)
    print()
    print("1. EVALUATE CURATOR OUTPUT")
    print("   - Review changeset quality and accuracy")
    print("   - Check if recommendations are actionable")
    print("   - Verify alignment with Reflector insights")
    print()
    print("2. IF CURATOR IS EXCELLENT:")
    print("   - Move to applying changes (next step in todo list)")
    print()
    print("3. IF CURATOR NEEDS IMPROVEMENT:")
    print("   - Restore from V11.2.2 baseline backup")
    print("   - Modify Curator agent (src/ace_kg/kg_curator.py)")
    print("   - Re-run this script")
    print("   - Repeat until Curator produces excellent output")
    print()
    print("🎯 This is meta-ACE: improving the ACE agents themselves!")
    print("="*80)


if __name__ == "__main__":
    main()
