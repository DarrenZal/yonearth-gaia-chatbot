#!/usr/bin/env python3
"""
Analyze V14.3.1 issues, excluding philosophical/metaphorical content (which user accepts)
Calculate real issue rate and identify actionable improvements
"""

import json
from pathlib import Path

def main():
    # Load V14.3.1 reflector analysis
    analysis_path = Path("kg_extraction_playbook/analysis_reports/reflection_v14.3.1_20251015_020114.json")

    with open(analysis_path) as f:
        analysis = json.load(f)

    total_rels = analysis['extraction_metadata']['total_relationships']
    total_issues = analysis['quality_summary']['total_issues']

    print("="*80)
    print("V14.3.1 ISSUE ANALYSIS (EXCLUDING ACCEPTABLE PHILOSOPHICAL CONTENT)")
    print("="*80)
    print()

    print(f"Total relationships: {total_rels}")
    print(f"Total issues (Reflector): {total_issues} ({analysis['quality_summary']['issue_rate_percent']}%)")
    print(f"Grade (Reflector): {analysis['quality_summary']['grade_confirmed']}")
    print()

    # Issue categories
    print("ISSUE CATEGORIES:")
    print()

    philosophical_count = 0
    actionable_issues = []

    for cat in analysis['issue_categories']:
        category = cat['category_name']
        count = cat['count']
        severity = cat['severity']
        percentage = cat['percentage']

        # User accepts philosophical/metaphorical content
        is_philosophical = 'Metaphorical' in category or 'Philosophical' in category

        if is_philosophical:
            philosophical_count += count
            status = "✓ ACCEPTED (user wants to keep)"
        else:
            actionable_issues.append(cat)
            status = f"❌ ACTIONABLE [{severity}]"

        print(f"  {category}: {count} ({percentage:.1f}%) - {status}")

    print()
    print("="*80)
    print("RECALCULATED METRICS (EXCLUDING PHILOSOPHICAL)")
    print("="*80)
    print()

    actionable_count = total_issues - philosophical_count
    actionable_rate = (actionable_count / total_rels) * 100

    print(f"Philosophical/Metaphorical issues (ACCEPTED): {philosophical_count}")
    print(f"Real actionable issues: {actionable_count}")
    print(f"Real issue rate: {actionable_rate:.1f}%")
    print()

    # Grade based on actionable issues
    if actionable_rate < 5:
        real_grade = "A+"
    elif actionable_rate < 8:
        real_grade = "A"
    elif actionable_rate < 10:
        real_grade = "A-"
    elif actionable_rate < 12:
        real_grade = "B+"
    else:
        real_grade = "B or lower"

    print(f"Real grade (excluding philosophical): {real_grade}")
    print()

    # Breakdown actionable issues by severity
    print("="*80)
    print("ACTIONABLE ISSUES BY SEVERITY")
    print("="*80)
    print()

    critical = sum(c['count'] for c in actionable_issues if c['severity'] == 'CRITICAL')
    high = sum(c['count'] for c in actionable_issues if c['severity'] == 'HIGH')
    medium = sum(c['count'] for c in actionable_issues if c['severity'] == 'MEDIUM')
    mild = sum(c['count'] for c in actionable_issues if c['severity'] == 'MILD')

    print(f"CRITICAL: {critical}")
    print(f"HIGH: {high}")
    print(f"MEDIUM: {medium}")
    print(f"MILD: {mild}")
    print()

    # Improvement targets
    print("="*80)
    print("PATH TO A- GRADE (TARGET: <10% ACTIONABLE ISSUES)")
    print("="*80)
    print()

    current_actionable = actionable_count
    target_for_a_minus = int(total_rels * 0.10)  # 10%
    target_for_a = int(total_rels * 0.08)  # 8%
    target_for_a_plus = int(total_rels * 0.05)  # 5%

    print(f"Current actionable issues: {current_actionable} ({actionable_rate:.1f}%)")
    print(f"Target for A-: {target_for_a_minus} issues (10%)")
    print(f"Target for A: {target_for_a} issues (8%)")
    print(f"Target for A+: {target_for_a_plus} issues (5%)")
    print()

    issues_to_fix_for_a_minus = current_actionable - target_for_a_minus

    if issues_to_fix_for_a_minus > 0:
        print(f"Need to fix: {issues_to_fix_for_a_minus} issues to reach A-")
        print()
        print("RECOMMENDED FIXES (in priority order):")
        print()

        for i, cat in enumerate(sorted(actionable_issues, key=lambda x: (
            {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'MILD': 3}[x['severity']],
            -x['count']
        )), 1):
            print(f"{i}. {cat['category_name']}: {cat['count']} issues [{cat['severity']}]")

            # Suggest where to fix
            category = cat['category_name']
            if 'Authorship' in category:
                fix_location = "Pass 2.5: BibliographicCitationParser (authorship direction validation)"
            elif 'Dedication' in category:
                fix_location = "Pass 2.5: BibliographicCitationParser (dedication parsing logic)"
            elif 'Vague' in category or 'Abstract' in category:
                fix_location = "Pass 2: Add entity enrichment OR Pass 2.5: Tune VagueEntityBlocker threshold"
            elif 'Praise' in category:
                fix_location = "Pass 2.5: PraiseQuoteDetector (improved front matter detection)"
            else:
                fix_location = "TBD - needs investigation"

            print(f"   → Fix in: {fix_location}")
            print()
    else:
        print(f"🎉 Already at A- grade! ({actionable_rate:.1f}% < 10%)")
        print(f"   Can aim for A grade by fixing {current_actionable - target_for_a} more issues")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"✅ V14.3.1 achieved {real_grade} when excluding philosophical content")
    print(f"📊 Actionable issues: {actionable_count}/{total_rels} ({actionable_rate:.1f}%)")
    print(f"🎯 Next target: Fix top {issues_to_fix_for_a_minus} issues to reach A- (or continue to A)")


if __name__ == "__main__":
    main()
