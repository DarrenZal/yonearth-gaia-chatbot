#!/usr/bin/env python3
"""
Run KG Reflector on V14.1 extraction results

V14.1 FIXES V14.0 REGRESSION:
- ROOT CAUSE: V14.0's Pass 1 prompt was too restrictive (596 candidates vs V13.1's 861)
- V14.1 SOLUTION: Use V12's proven complex prompt (23KB) + 15 postprocessing modules
- NEW MODULE: SemanticDeduplicator removes redundant relationships using embeddings

Expected Results:
- Total relationships: ~708
- Issue count: ~25-30 (down from V14.0's 65)
- Issue rate: ~3-4% (A or A+ grade)
- Quality improvement: V14.0's 10.78% → V14.1's ~3-4%
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from src.ace_kg.kg_reflector import KGReflectorAgent
import pdfplumber

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from PDF"""
    print(f"📖 Extracting text from {pdf_path.name}...")

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    print(f"✅ Extracted {len(full_text.split())} words")
    return full_text


def main():
    print("="*80)
    print("🤔 RUNNING KG REFLECTOR ON V14.1 OUTPUT")
    print("="*80)
    print()

    # Paths
    v14_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v14_1/soil_stewardship_handbook_v14_1.json")
    v14_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v14/soil_stewardship_handbook_v14.json")
    v13_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v13/soil_stewardship_handbook_v13_1_from_v12.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V14.1 results
    print("📂 Loading V14.1 extraction results...")
    with open(v14_1_output_path) as f:
        v14_1_data = json.load(f)

    print(f"✅ Loaded {len(v14_1_data['relationships'])} relationships from V14.1")
    print()

    # Load V14.0 results for comparison
    print("📂 Loading V14.0 extraction results for comparison...")
    with open(v14_output_path) as f:
        v14_data = json.load(f)

    print(f"✅ Loaded {len(v14_data['relationships'])} relationships from V14.0")
    print()

    # Load V13.1 results for additional context
    print("📂 Loading V13.1 extraction results for context...")
    with open(v13_1_output_path) as f:
        v13_1_data = json.load(f)

    print(f"✅ Loaded {len(v13_1_data['relationships'])} relationships from V13.1")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': 'v14.1',
        'book_title': v14_1_data['metadata']['book_title'],
        'date': v14_1_data['metadata']['extraction_date'],
        'total_relationships': len(v14_1_data['relationships']),
        'pass2_evaluated': v14_1_data['extraction_stats']['pass2_evaluated'],
        'pass2_5_stats': v14_1_data['postprocessing_stats']
    }

    # Prepare previous reports for comparison
    print("📊 Loading previous Reflector analyses for comparison...")
    previous_reports = []
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")

    # Load V14.0 analysis
    v14_analysis_files = sorted(analysis_dir.glob("reflection_v14_*.json"))
    v14_analysis = None
    if v14_analysis_files:
        with open(v14_analysis_files[-1]) as f:
            v14_analysis = json.load(f)

        v14_grade = v14_analysis['quality_summary'].get('grade_confirmed') or v14_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V14.0 Reflector Analysis (B+ regression)',
            'summary': f"V14.0 had {v14_analysis['quality_summary']['total_issues']} issues ({v14_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v14_grade}",
            'content_preview': json.dumps({'quality_summary': v14_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V14.0 Reflector analysis for context")

    # Load V13.1 analysis
    v13_1_analysis_files = sorted(analysis_dir.glob("reflection_v13_1_*.json"))
    v13_1_analysis = None
    if v13_1_analysis_files:
        with open(v13_1_analysis_files[-1]) as f:
            v13_1_analysis = json.load(f)

        v13_1_grade = v13_1_analysis['quality_summary'].get('grade_confirmed') or v13_1_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V13.1 Reflector Analysis (A- baseline)',
            'summary': f"V13.1 had {v13_1_analysis['quality_summary']['total_issues']} issues ({v13_1_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v13_1_grade}",
            'content_preview': json.dumps({'quality_summary': v13_1_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V13.1 Reflector analysis for context")
        print()

    print("="*80)
    print("🤔 ANALYZING V14.1 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V14.1 Fixes V14.0 Regression:")
    print("  ✅ Pass 1: V12's proven complex prompt (23KB) - extracted 782 candidates")
    print("  ✅ Pass 2: V14's enhanced dual-signal evaluation")
    print("  ✅ NEW: SemanticDeduplicator module (15 modules total)")
    print("      - Uses sentence-transformers with 0.87 cosine similarity")
    print("      - Removed 60 redundant relationships from 49 groups")
    print("      - Addressed V14.0's 25 redundant 'is-a' relationships")
    print("  ✅ MetadataFilter, ConfidenceFilter, Predicate Normalizer V1.4")
    print("  ✅ Pronoun Resolver with unresolved filtering")
    print(f"  📊 Results: {len(v14_1_data['relationships'])} relationships")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v14_1_data['relationships'],
        source_text=source_text,
        extraction_metadata=extraction_metadata,
        v4_quality_reports=previous_reports
    )

    print()
    print("="*80)
    print("✅ REFLECTOR ANALYSIS COMPLETE")
    print("="*80)
    print()

    # Display summary
    if 'quality_summary' in analysis:
        summary = analysis['quality_summary']
        print("QUALITY SUMMARY:")
        print(f"  Total issues: {summary.get('total_issues', 'N/A')}")
        print(f"  Issue rate: {summary.get('issue_rate_percent', 'N/A')}%")
        print(f"  Critical issues: {summary.get('critical_issues', 'N/A')}")
        print(f"  High priority: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  Medium priority: {summary.get('medium_priority_issues', 'N/A')}")
        print(f"  Mild issues: {summary.get('mild_issues', 'N/A')}")
        print(f"  Grade (confirmed): {summary.get('grade_confirmed', 'N/A')}")
        if 'grade_adjusted' in summary:
            print(f"  Grade (adjusted): {summary.get('grade_adjusted', 'N/A')}")
        print()

    if 'issue_categories' in analysis:
        print(f"ISSUE CATEGORIES FOUND: {len(analysis['issue_categories'])}")
        for cat in analysis['issue_categories'][:5]:  # Show top 5
            print(f"  - {cat['category_name']}: {cat['count']} ({cat['percentage']:.1f}%) [{cat['severity']}]")
        print()

    if 'novel_error_patterns' in analysis:
        print(f"NOVEL ERROR PATTERNS: {len(analysis['novel_error_patterns'])}")
        for pattern in analysis['novel_error_patterns'][:3]:  # Show top 3
            print(f"  - {pattern['pattern_name']}: {pattern['count']} [{pattern['severity']}]")
        print()

    if 'improvement_recommendations' in analysis:
        print(f"IMPROVEMENT RECOMMENDATIONS: {len(analysis['improvement_recommendations'])}")
        for rec in analysis['improvement_recommendations'][:5]:  # Show top 5
            print(f"  [{rec['priority']}] {rec['type']}: {rec['recommendation'][:80]}...")
        print()

    # Find the saved analysis file
    analysis_files = sorted(analysis_dir.glob("reflection_v14_1_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V14.0 and V13.1
    if v14_analysis and v13_1_analysis:
        print("="*80)
        print("📊 V13.1 vs V14.0 vs V14.1 COMPARISON")
        print("="*80)
        print()

        v13_1_grade_display = v13_1_analysis['quality_summary'].get('grade_confirmed') or v13_1_analysis['quality_summary'].get('grade', 'N/A')
        v14_grade_display = v14_analysis['quality_summary'].get('grade_confirmed') or v14_analysis['quality_summary'].get('grade', 'N/A')

        print(f"V13.1 Results (A- baseline):")
        print(f"  - Total relationships: {len(v13_1_data['relationships'])}")
        print(f"  - High confidence: {v13_1_data['extraction_stats']['high_confidence']} ({v13_1_data['extraction_stats']['high_confidence']/len(v13_1_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {v13_1_analysis['quality_summary']['total_issues']} ({v13_1_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v13_1_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v13_1_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v13_1_grade_display}")
        print()

        print(f"V14.0 Results (B+ regression):")
        print(f"  - Total relationships: {len(v14_data['relationships'])}")
        print(f"  - High confidence: {v14_data['extraction_stats']['high_confidence']} ({v14_data['extraction_stats']['high_confidence']/len(v14_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {v14_analysis['quality_summary']['total_issues']} ({v14_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v14_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v14_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v14_grade_display}")
        print()

        print(f"V14.1 Results (regression fix):")
        print(f"  - Total relationships: {len(v14_1_data['relationships'])}")
        print(f"  - Pass 2 evaluated: {v14_1_data['extraction_stats']['pass2_evaluated']}")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
        print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
        print()

        v14_issues = v14_analysis['quality_summary']['total_issues']
        v14_1_issues = summary.get('total_issues', 0)

        if v14_issues > 0:
            issue_improvement = v14_issues - v14_1_issues
            issue_improvement_pct = (issue_improvement / v14_issues * 100)
            print(f"V14.0 → V14.1 Quality change: {issue_improvement:+d} issues ({issue_improvement_pct:+.1f}% change)")

            if issue_improvement > 0:
                print(f"✅ V14.1 has FEWER quality issues (regression FIXED!)")
            elif issue_improvement < 0:
                print(f"⚠️  V14.1 has MORE quality issues (unexpected)")
            else:
                print(f"➡️  V14.1 has SAME quality issues")
        print()

    print("="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)

    if v14_analysis and v13_1_analysis and 'quality_summary' in analysis:
        v13_1_grade = v13_1_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v14_grade = v14_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v14_1_grade = summary.get('grade_confirmed', 'N/A')

        v13_1_issues = v13_1_analysis['quality_summary']['total_issues']
        v14_issues = v14_analysis['quality_summary']['total_issues']
        v14_1_issues = summary.get('total_issues', 0)

        # Check if V14.1 reached A or A+ target
        v14_1_grade_value = v14_1_grade if isinstance(v14_1_grade, str) else 'N/A'
        reached_target = v14_1_grade_value in ['A+', 'A']

        # Check if V14.1 beats V13.1 baseline
        beats_baseline = v14_1_issues < v13_1_issues

        if beats_baseline and reached_target:
            print("🎉 RECOMMENDATION: V14.1 is NEW STABLE RELEASE")
            print(f"   - Reached target grade: {v14_1_grade} (A or A+ achieved!)")
            print(f"   - Beats V13.1 baseline: {v13_1_issues - v14_1_issues} fewer issues than A-")
            print(f"   - Fixed V14.0 regression: {v14_issues - v14_1_issues} issues fixed")
            print("   - V12 prompt + SemanticDeduplicator working as designed")
            print("   - Ready for production use")
        elif v14_1_issues < v14_issues:
            print("✅ RECOMMENDATION: V14.1 fixes V14.0 regression but not at target yet")
            print(f"   - Fixed V14.0 regression: {v14_issues - v14_1_issues} issues fixed")
            print(f"   - Grade improved: {v14_grade} → {v14_1_grade}")
            if not beats_baseline:
                print(f"   - Still behind V13.1 baseline: {v14_1_issues - v13_1_issues} more issues")
            if not reached_target:
                print(f"   - Grade: {v14_1_grade} (target was A or A+)")
            print("   - Consider additional enhancements for V14.2")
        elif v14_1_issues == v14_issues:
            print("➡️  RECOMMENDATION: Review V14.1 changes")
            print("   - No quality change from V14.0")
            print("   - SemanticDeduplicator may need tuning")
            print("   - Consider different approach for V14.2")
        else:
            print("⚠️  RECOMMENDATION: Investigate V14.1 unexpected result")
            print(f"   - MORE issues than V14.0: {v14_1_issues - v14_issues} additional issues")
            print("   - Root cause analysis needed")
            print("   - Keep V13.1 as stable until resolved")

    print("="*80)


if __name__ == "__main__":
    main()
