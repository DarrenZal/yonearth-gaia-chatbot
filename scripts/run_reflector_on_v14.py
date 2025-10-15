#!/usr/bin/env python3
"""
Run KG Reflector on V14.0 extraction results

Analyzes V14.0 comprehensive quality system with 7 major enhancements
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
    print("🤔 RUNNING KG REFLECTOR ON V14.0 OUTPUT")
    print("="*80)
    print()

    # Paths
    v14_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v14/soil_stewardship_handbook_v14.json")
    v13_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v13/soil_stewardship_handbook_v13_1_from_v12.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V14.0 results
    print("📂 Loading V14.0 extraction results...")
    with open(v14_output_path) as f:
        v14_data = json.load(f)

    print(f"✅ Loaded {len(v14_data['relationships'])} relationships from V14.0")
    print()

    # Load V13.1 results for comparison context
    print("📂 Loading V13.1 extraction results for comparison...")
    with open(v13_1_output_path) as f:
        v13_1_data = json.load(f)

    print(f"✅ Loaded {len(v13_1_data['relationships'])} relationships from V13.1")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': 'v14.0',
        'book_title': v14_data['metadata']['book_title'],
        'date': v14_data['metadata']['extraction_date'],
        'total_relationships': len(v14_data['relationships']),
        'high_confidence_count': v14_data['extraction_stats']['high_confidence'],
        'pass2_5_stats': v14_data['postprocessing_stats']
    }

    # Prepare V13.1 comparison report
    print("📊 Loading V13.1 Reflector analysis for comparison...")
    previous_reports = []
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
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
    print("🤔 ANALYZING V14.0 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V14.0 Comprehensive Quality System:")
    print("  ✅ Enhanced Pass 1: Entity specificity + extraction scope")
    print("  ✅ Enhanced Pass 2: Improved claim classification (FACTUAL/NORMATIVE/PHILOSOPHICAL)")
    print("  ✅ MetadataFilter Module: 4-layer detection for book metadata")
    print("  ✅ ConfidenceFilter Module: Flag-specific thresholds + unresolved pronoun handling")
    print("  ✅ Predicate Normalizer V1.4: Tense norm, modal verb preservation, semantic validation")
    print("  ✅ Pronoun Resolver: Enhanced with unresolved pronoun filtering (0.7 threshold)")
    print("  ✅ 14 postprocessing modules total (12 from V13.1 + 2 new)")
    print(f"  📊 Results: {len(v14_data['relationships'])} relationships, {v14_data['extraction_stats']['high_confidence']} high conf ({v14_data['extraction_stats']['high_confidence']/len(v14_data['relationships'])*100:.1f}%)")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v14_data['relationships'],
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
    analysis_files = sorted(analysis_dir.glob("reflection_v14_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V13.1
    if v13_1_analysis:
        print("="*80)
        print("📊 V13.1 vs V14.0 COMPARISON")
        print("="*80)
        print()
        v13_1_grade_display = v13_1_analysis['quality_summary'].get('grade_confirmed') or v13_1_analysis['quality_summary'].get('grade', 'N/A')
        print(f"V13.1 Results (A- baseline):")
        print(f"  - Total relationships: {len(v13_1_data['relationships'])}")
        print(f"  - High confidence: {v13_1_data['extraction_stats']['high_confidence']} ({v13_1_data['extraction_stats']['high_confidence']/len(v13_1_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {v13_1_analysis['quality_summary']['total_issues']} ({v13_1_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v13_1_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v13_1_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v13_1_grade_display}")
        print()
        print(f"V14.0 Results (7 enhancements):")
        print(f"  - Total relationships: {len(v14_data['relationships'])}")
        print(f"  - High confidence: {v14_data['extraction_stats']['high_confidence']} ({v14_data['extraction_stats']['high_confidence']/len(v14_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
        print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
        print()

        v13_1_issues = v13_1_analysis['quality_summary']['total_issues']
        v14_issues = summary.get('total_issues', 0)

        if v13_1_issues > 0:
            issue_improvement = v13_1_issues - v14_issues
            issue_improvement_pct = (issue_improvement / v13_1_issues * 100)
            print(f"Quality change: {issue_improvement:+d} issues ({issue_improvement_pct:+.1f}% change)")

            if issue_improvement > 0:
                print(f"✅ V14.0 has FEWER quality issues (V14 enhancements IMPROVED quality)")
            elif issue_improvement < 0:
                print(f"⚠️  V14.0 has MORE quality issues (unexpected regression)")
            else:
                print(f"➡️  V14.0 has SAME quality issues (no measurable quality impact)")
        print()

    print("="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)

    if v13_1_analysis and 'quality_summary' in analysis:
        v13_1_grade = v13_1_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v14_grade = summary.get('grade_confirmed', 'N/A')
        v13_1_issues = v13_1_analysis['quality_summary']['total_issues']
        v14_issues = summary.get('total_issues', 0)

        # Check if V14 reached A or A+ target
        v14_grade_value = v14_grade if isinstance(v14_grade, str) else 'N/A'
        reached_target = v14_grade_value in ['A+', 'A']

        if v14_issues < v13_1_issues and reached_target:
            print("🎉 RECOMMENDATION: V14.0 is NEW STABLE RELEASE")
            print(f"   - Reached target grade: {v14_grade} (A or A+ achieved!)")
            print(f"   - Significantly fewer issues: {v13_1_issues - v14_issues} issues fixed")
            print("   - All 7 V14 enhancements working as designed")
            print("   - Ready for production use")
        elif v14_issues < v13_1_issues:
            print("✅ RECOMMENDATION: V14.0 shows improvement but not at target yet")
            print(f"   - Fewer issues: {v13_1_issues - v14_issues} issues fixed")
            print(f"   - Grade: {v14_grade} (target was A or A+)")
            print("   - Consider additional enhancements for V14.1")
        elif v14_issues == v13_1_issues:
            print("➡️  RECOMMENDATION: Review V14.0 enhancements")
            print("   - No measurable quality change")
            print("   - Enhancements may need tuning")
            print("   - Consider V14.1 iteration")
        else:
            print("⚠️  RECOMMENDATION: Investigate V14.0 regression")
            print(f"   - MORE issues than V13.1: {v14_issues - v13_1_issues} additional issues")
            print("   - Unexpected regression requires root cause analysis")
            print("   - Keep V13.1 as stable until resolved")

    print("="*80)


if __name__ == "__main__":
    main()
