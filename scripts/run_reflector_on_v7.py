#!/usr/bin/env python3
"""
Run KG Reflector on V7 extraction results

Analyzes V7 quality and validates Meta-ACE improvements
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
    print("🤔 RUNNING KG REFLECTOR ON V7 OUTPUT (META-ACE ENHANCED)")
    print("="*80)
    print()

    # Paths
    v7_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v7/soil_stewardship_handbook_v7.json")
    v6_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v6/soil_stewardship_handbook_v6.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V7 results
    print("📂 Loading V7 extraction results...")
    with open(v7_output_path) as f:
        v7_data = json.load(f)

    print(f"✅ Loaded {len(v7_data['relationships'])} relationships from V7")
    print()

    # Load V6 results for comparison
    print("📂 Loading V6 extraction results for comparison...")
    with open(v6_output_path) as f:
        v6_data = json.load(f)

    print(f"✅ Loaded {len(v6_data['relationships'])} relationships from V6")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': v7_data['version'],
        'book_title': v7_data['book_title'],
        'date': v7_data['timestamp'],
        'total_relationships': len(v7_data['relationships']),
        'high_confidence_count': sum(1 for r in v7_data['relationships'] if r.get('p_true', 0) >= 0.75),
        'pass2_5_stats': v7_data['pass2_5_stats']
    }

    # Load previous quality reports for context
    print("📊 Loading previous quality reports for context...")
    previous_reports = []

    # V4 report
    v4_report_path = Path("/home/claudeuser/yonearth-gaia-chatbot/docs/knowledge_graph/V4_EXTRACTION_QUALITY_ISSUES_REPORT.md")
    if v4_report_path.exists():
        with open(v4_report_path) as f:
            v4_report_content = f.read()

        previous_reports.append({
            'title': 'V4 Quality Issues Report',
            'summary': 'V4 had 57% quality issues across 7 categories',
            'content_preview': v4_report_content[:2000]
        })
        print(f"✅ Loaded V4 quality report")

    # V6 Reflector analysis
    v6_analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
    v6_analysis_files = sorted(v6_analysis_dir.glob("reflection_v6_*.json"))
    if v6_analysis_files:
        with open(v6_analysis_files[-1]) as f:
            v6_analysis = json.load(f)

        v6_grade = v6_analysis['quality_summary'].get('grade_confirmed') or v6_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V6 Reflector Analysis',
            'summary': f"V6 had {v6_analysis['quality_summary']['total_issues']} issues ({v6_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v6_grade}",
            'content_preview': json.dumps({
                'quality_summary': v6_analysis['quality_summary'],
                'issue_categories': v6_analysis['issue_categories'][:5],
                'novel_error_patterns': v6_analysis.get('novel_error_patterns', [])[:3]
            }, indent=2)
        })
        print(f"✅ Loaded V6 Reflector analysis")

    print()
    print("="*80)
    print("🤔 ANALYZING V7 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V7 Meta-ACE Improvements Implemented:")
    print("  ✅ Enhanced praise quote detector (16 patterns vs 5)")
    print("  ✅ Multi-pass pronoun resolution (3 passes: 100, 500, 1000 chars)")
    print("  ✅ Vague entity blocker (blocks unfixable abstract entities)")
    print("  ✅ Context enricher runs BEFORE vague entity blocker")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v7_data['relationships'],
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
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
    analysis_files = sorted(analysis_dir.glob("reflection_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V6
    if v6_analysis_files:
        print("="*80)
        print("📊 V6 vs V7 REFLECTOR COMPARISON (META-ACE VALIDATION)")
        print("="*80)
        print()
        v6_grade_display = v6_analysis['quality_summary'].get('grade_confirmed') or v6_analysis['quality_summary'].get('grade', 'N/A')
        print(f"V6 Results (ACE Cycle 1):")
        print(f"  - Total issues: {v6_analysis['quality_summary']['total_issues']} ({v6_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v6_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v6_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v6_grade_display}")
        print()
        print(f"V7 Results (Meta-ACE Enhanced):")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
        print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
        print()

        v6_issues = v6_analysis['quality_summary']['total_issues']
        v7_issues = summary.get('total_issues', 0)
        improvement = v6_issues - v7_issues
        improvement_pct = (improvement / v6_issues * 100) if v6_issues > 0 else 0

        print(f"Improvement: {improvement} fewer issues ({improvement_pct:.1f}% reduction)")
        print()

        # Meta-ACE fix validation
        print("META-ACE FIX VALIDATION:")
        v6_critical = v6_analysis['quality_summary']['critical_issues']
        v7_critical = summary.get('critical_issues', 0)
        print(f"  - Critical issues reduced: {v6_critical} → {v7_critical} ({v6_critical - v7_critical} eliminated)")

        v6_high = v6_analysis['quality_summary']['high_priority_issues']
        v7_high = summary.get('high_priority_issues', 0)
        print(f"  - High priority reduced: {v6_high} → {v7_high} ({v6_high - v7_high} eliminated)")
        print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)

    target_met = summary.get('issue_rate_percent', 100) < 5.0

    if target_met:
        print("🎉 TARGET MET! Quality issues < 5%")
        print()
        print("1. ✅ V7 achieves production quality (A- grade)")
        print("2. 📊 Document V7 results in ACE_CYCLE_1_V7_RESULTS.md")
        print("3. 🚀 Apply V7 system to full corpus (172 episodes + 3 books)")
        print("4. 📚 Build unified knowledge graph")
    else:
        print("📈 Continue ACE cycle:")
        print()
        print("1. Review V7 Reflector analysis")
        print("2. Identify remaining systematic issues")
        print("3. Implement targeted fixes in V8 if needed")
        print("4. Repeat until <5% quality issues")

    print("="*80)


if __name__ == "__main__":
    main()
