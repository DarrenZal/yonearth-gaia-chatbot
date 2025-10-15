#!/usr/bin/env python3
"""
Run KG Reflector on V13.1 extraction results

Analyzes V13.1 quality with dual-signal evaluation WITHOUT penalties
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
    print("🤔 RUNNING KG REFLECTOR ON V13.1 OUTPUT (NO PENALTIES)")
    print("="*80)
    print()

    # Paths
    v13_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v13/soil_stewardship_handbook_v13_1_from_v12.json")
    v12_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v12/soil_stewardship_handbook_v12.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V13.1 results
    print("📂 Loading V13.1 extraction results...")
    with open(v13_1_output_path) as f:
        v13_1_data = json.load(f)

    print(f"✅ Loaded {len(v13_1_data['relationships'])} relationships from V13.1")
    print()

    # Load V12 results for comparison context
    print("📂 Loading V12 extraction results for comparison...")
    with open(v12_output_path) as f:
        v12_data = json.load(f)

    print(f"✅ Loaded {len(v12_data['relationships'])} relationships from V12")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': 'v13.1',
        'book_title': v13_1_data['metadata']['book_title'],
        'date': v13_1_data['metadata']['extraction_date'],
        'total_relationships': len(v13_1_data['relationships']),
        'high_confidence_count': v13_1_data['extraction_stats']['high_confidence'],
        'pass2_5_stats': v13_1_data['postprocessing_stats']
    }

    # Prepare V12 comparison report
    print("📊 Loading V12 Reflector analysis for comparison...")
    previous_reports = []
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
    v12_analysis_files = sorted(analysis_dir.glob("reflection_v12_*.json"))
    v12_analysis = None
    if v12_analysis_files:
        with open(v12_analysis_files[-1]) as f:
            v12_analysis = json.load(f)

        v12_grade = v12_analysis['quality_summary'].get('grade_confirmed') or v12_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V12 Reflector Analysis (WITH penalties)',
            'summary': f"V12 had {v12_analysis['quality_summary']['total_issues']} issues ({v12_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v12_grade}",
            'content_preview': json.dumps({'quality_summary': v12_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V12 Reflector analysis for context")
        print()

    print("="*80)
    print("🤔 ANALYZING V13.1 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V13.1 Dual-Signal WITHOUT Penalties:")
    print("  ✅ entity_specificity_score (concreteness)")
    print("  ✅ classification_flags (FACTUAL, TESTABLE, PHILOSOPHICAL, NORMATIVE, etc.)")
    print("  ❌ NO PHILOSOPHICAL penalty (was -0.4)")
    print("  ❌ NO NORMATIVE penalty (was -0.3)")
    print("  ✅ 12 postprocessing modules (same as V12)")
    print(f"  📊 Results: {len(v13_1_data['relationships'])} relationships, {v13_1_data['extraction_stats']['high_confidence']} high conf ({v13_1_data['extraction_stats']['high_confidence']/len(v13_1_data['relationships'])*100:.1f}%)")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v13_1_data['relationships'],
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
    analysis_files = sorted(analysis_dir.glob("reflection_v13_1_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V12
    if v12_analysis:
        print("="*80)
        print("📊 V12 vs V13.1 COMPARISON")
        print("="*80)
        print()
        v12_grade_display = v12_analysis['quality_summary'].get('grade_confirmed') or v12_analysis['quality_summary'].get('grade', 'N/A')
        print(f"V12 Results (WITH PENALTIES):")
        print(f"  - Total relationships: {len(v12_data['relationships'])}")
        print(f"  - High confidence: {v12_data['extraction_stats']['high_confidence']} ({v12_data['extraction_stats']['high_confidence']/len(v12_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {v12_analysis['quality_summary']['total_issues']} ({v12_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v12_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v12_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v12_grade_display}")
        print()
        print(f"V13.1 Results (NO PENALTIES):")
        print(f"  - Total relationships: {len(v13_1_data['relationships'])}")
        print(f"  - High confidence: {v13_1_data['extraction_stats']['high_confidence']} ({v13_1_data['extraction_stats']['high_confidence']/len(v13_1_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
        print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
        print()

        v12_issues = v12_analysis['quality_summary']['total_issues']
        v13_1_issues = summary.get('total_issues', 0)

        if v12_issues > 0:
            issue_improvement = v12_issues - v13_1_issues
            issue_improvement_pct = (issue_improvement / v12_issues * 100)
            print(f"Quality change: {issue_improvement:+d} issues ({issue_improvement_pct:+.1f}% change)")

            if issue_improvement > 0:
                print(f"✅ V13.1 has FEWER quality issues (removing penalties IMPROVED quality)")
            elif issue_improvement < 0:
                print(f"⚠️  V13.1 has MORE quality issues (penalties were helping filter noise)")
            else:
                print(f"➡️  V13.1 has SAME quality issues (penalties had no quality impact)")
        print()

    print("="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)

    if v12_analysis and 'quality_summary' in analysis:
        v12_grade = v12_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v13_1_grade = summary.get('grade_confirmed', 'N/A')
        v12_issues = v12_analysis['quality_summary']['total_issues']
        v13_1_issues = summary.get('total_issues', 0)

        if v13_1_issues < v12_issues:
            print("✅ RECOMMENDATION: Use V13.1 (NO PENALTIES)")
            print("   - Better quality (fewer issues)")
            print("   - More honest confidence distribution")
            print("   - Simpler prompt (no penalty logic)")
        elif v13_1_issues == v12_issues:
            print("✅ RECOMMENDATION: Use V13.1 (NO PENALTIES)")
            print("   - Equal quality")
            print("   - Simpler prompt (no penalty logic)")
            print("   - More philosophically sound (classify but don't discriminate)")
        else:
            print("⚠️  RECOMMENDATION: Consider keeping V12 (WITH PENALTIES)")
            print("   - Better quality (fewer issues)")
            print("   - Penalties effectively filtered noise")
            print("   - BUT: Review if penalties suppressed valid claims")

    print("="*80)


if __name__ == "__main__":
    main()
