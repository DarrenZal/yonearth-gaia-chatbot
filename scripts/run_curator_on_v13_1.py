#!/usr/bin/env python3
"""
Run Curator on V13.1 Reflector analysis to generate V14 improvements

V13.1 Grade: B/B- (14.5% issue rate, 127 total issues)
Target: A++ (<3% issue rate)

Top Issues to Address:
1. 12 Praise Quotes Misclassified [HIGH]
2. 18 Vague Abstract Entities [HIGH]
3. 133 Predicate Fragmentation [MEDIUM]
4. 8 Unresolved Possessive Pronouns [MEDIUM]
5. 6 Unresolved Generic Pronouns [MEDIUM]
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.ace_kg.kg_curator import KGCuratorAgent

def main():
    print("="*80)
    print("🎨 RUNNING CURATOR ON V13.1 TO GENERATE V14 IMPROVEMENTS")
    print("="*80)
    print()
    print("V13.1 Baseline (No Confidence Penalties):")
    print("  - Grade: A- confirmed / B+ adjusted (8.6% issue rate)")
    print("  - Total issues: 75")
    print("  - Critical: 0 | High: 8 | Medium: 22 | Mild: 45")
    print()
    print("V14 Goal: Reduce issue rate from 8.6% → <5% (A/A+ grade)")
    print()

    # Paths
    v13_1_reflection_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/reflection_v13.1_20251014_095254.json")

    # V13.1 prompts (base prompts to improve)
    v13_pass1_prompt_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts/pass1_extraction_v13.txt")
    v13_pass2_prompt_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts/pass2_evaluation_v13_1.txt")

    # Check if V13 Pass 1 exists, otherwise use V11 (since V13 used V11's Pass 1)
    if not v13_pass1_prompt_path.exists():
        v13_pass1_prompt_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts/pass1_extraction_v11.txt")
        print(f"⚠️  V13 Pass 1 prompt not found, using V11 Pass 1 as base")

    # Load V13.1 Reflector analysis
    print("📂 Loading V13.1 Reflector analysis...")
    with open(v13_1_reflection_path) as f:
        v13_1_reflection = json.load(f)

    print(f"✅ Loaded V13.1 analysis: {v13_1_reflection['quality_summary']['total_issues']} issues")
    print()

    # Load current prompts
    print("📂 Loading V13.1 prompts...")
    with open(v13_pass1_prompt_path) as f:
        current_pass1_prompt = f.read()

    with open(v13_pass2_prompt_path) as f:
        current_pass2_prompt = f.read()

    print(f"✅ Pass 1 prompt: {len(current_pass1_prompt)} characters")
    print(f"✅ Pass 2 prompt: {len(current_pass2_prompt)} characters")
    print()

    # Prepare prompt versions for context
    prompt_versions = {
        'v13.1_pass1': {
            'path': str(v13_pass1_prompt_path),
            'content': current_pass1_prompt,
            'version': 'v13.1',
            'pass_number': 1
        },
        'v13.1_pass2': {
            'path': str(v13_pass2_prompt_path),
            'content': current_pass2_prompt,
            'version': 'v13.1',
            'pass_number': 2
        }
    }

    print("="*80)
    print("🎨 GENERATING V14 IMPROVEMENTS WITH CURATOR...")
    print("="*80)
    print("This will take 2-3 minutes...")
    print()
    print("Curator will analyze:")
    print("  1. All 75 quality issues from V13.1")
    print("  2. Current V13.1 Pass 1 and Pass 2 prompts")
    print("  3. Systematic error patterns")
    print("  4. Improvement recommendations")
    print()
    print("Output will include:")
    print("  - Specific prompt changes (diff format)")
    print("  - Rationale for each change")
    print("  - Expected impact on quality")
    print("  - Priority ranking")
    print()

    # Add V14 goal context to reflector report
    v13_1_reflection['v14_goal_context'] = {
        'focus': 'Reduce quality issues from A- to A/A+ grade',
        'v13_1_success': [
            'Classification without discrimination (no penalties)',
            'Honest confidence distribution',
            'Better than V12 (A- vs B grade)',
            'Simpler logic (no penalty system)',
            '8.6% issue rate (was 14.3% in V12)'
        ],
        'v13_1_issues': [
            '8 Praise Quote Misclassification [HIGH]',
            '12 Vague Abstract Entities [MEDIUM]',
            '10 Predicate Fragmentation [MEDIUM]',
            '8 Philosophical/Abstract Claims [MEDIUM]',
            '3 Unresolved Possessive Pronouns [MILD]',
            '5 Context-Enriched Source Overcorrection [MILD]',
            '6 Normative Statements Classified as Factual [MEDIUM]',
            '2 Dedication Relationships [MILD]'
        ],
        'v14_targets': {
            'relationships': '~873 (maintain same coverage)',
            'issue_rate': '<5% (A/A+ grade)',
            'critical_issues': '0 (maintain)',
            'high_priority_issues': '<4 (reduce from 8)',
            'medium_priority_issues': '<15 (reduce from 22)',
            'mild_issues': '<20 (reduce from 45)'
        },
        'key_changes_needed': [
            'FIX: Praise quote detection (8 HIGH issues)',
            'STRENGTHEN: Vague entity detection (12 MEDIUM issues)',
            'IMPROVE: Predicate normalization (10 MEDIUM fragmentation)',
            'CLARIFY: Philosophical vs factual distinction (8 MEDIUM)',
            'FIX: Normative statements classified as factual (6 MEDIUM)',
            'NEW MODULE: Filter book metadata relationships (2 MILD)',
            'KEEP: No confidence penalties approach'
        ]
    }

    # Initialize Curator
    curator = KGCuratorAgent()

    # Generate improvements using correct method
    try:
        changeset = curator.curate_improvements(
            reflector_report=v13_1_reflection,
            current_version=13.1
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

    # Display summary
    if 'summary' in changeset:
        summary = changeset['summary']
        print("CHANGESET SUMMARY:")
        print(f"  Version: {summary.get('source_version', 'N/A')} → {summary.get('target_version', 'N/A')}")
        print(f"  Total changes: {summary.get('total_changes', 'N/A')}")
        print(f"  Prompts modified: {summary.get('prompts_modified', 'N/A')}")
        print(f"  Expected issue reduction: {summary.get('expected_issue_reduction', 'N/A')}")
        print()

    if 'changes' in changeset:
        print(f"PROMPT CHANGES: {len(changeset['changes'])}")
        for i, change in enumerate(changeset['changes'][:5], 1):  # Show top 5
            print(f"  {i}. [{change['priority']}] {change['prompt_file']}: {change['change_type']}")
            print(f"     {change['description'][:80]}...")
        print()

    if 'expected_improvements' in changeset:
        print(f"EXPECTED IMPROVEMENTS:")
        for improvement in changeset['expected_improvements'][:3]:  # Show top 3
            print(f"  - {improvement['issue_category']}: {improvement['current_count']} → {improvement['target_count']} ({improvement['reduction_percent']}%)")
        print()

    # Save changeset
    changeset_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/changesets")
    changeset_dir.mkdir(exist_ok=True)

    changeset_path = changeset_dir / f"v13_1_to_v14_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(changeset_path, 'w') as f:
        json.dump(changeset, f, indent=2)

    print(f"📁 Full changeset saved to: {changeset_path}")
    print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print()
    print("1. Review changeset:")
    print(f"   cat {changeset_path}")
    print()
    print("2. Apply changes to create V14 prompts:")
    print("   python scripts/apply_changeset.py \\")
    print(f"     --changeset {changeset_path} \\")
    print("     --output-version v14")
    print()
    print("3. Run V14 extraction:")
    print("   python scripts/extract_kg_v14_book.py")
    print()
    print("4. Run Reflector on V14:")
    print("   python scripts/run_reflector_on_v14.py")
    print()
    print("5. Compare V13.1 vs V14:")
    print("   - V13.1: 8.6% issue rate (A-)")
    print("   - V14: Target <5% (A/A+)")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
