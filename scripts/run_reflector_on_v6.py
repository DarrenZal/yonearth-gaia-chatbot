#!/usr/bin/env python3
"""
Run KG Reflector on V6 extraction results

Analyzes V6 quality and generates improvement recommendations for V7
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
    print("🤔 RUNNING KG REFLECTOR ON V6 OUTPUT")
    print("="*80)
    print()

    # Paths
    v6_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v6/soil_stewardship_handbook_v6.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V6 results
    print("📂 Loading V6 extraction results...")
    with open(v6_output_path) as f:
        v6_data = json.load(f)

    print(f"✅ Loaded {len(v6_data['relationships'])} relationships from V6")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': v6_data['version'],
        'book_title': v6_data['book_title'],
        'date': v6_data['timestamp'],
        'total_relationships': len(v6_data['relationships']),
        'high_confidence_count': sum(1 for r in v6_data['relationships'] if r.get('p_true', 0) >= 0.75),
        'pass2_5_stats': v6_data['pass2_5_stats']
    }

    # Load V4 and V5 quality reports for context
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

    # V5 Reflector analysis
    v5_analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
    v5_analysis_files = sorted(v5_analysis_dir.glob("reflection_v5_*.json"))
    if v5_analysis_files:
        with open(v5_analysis_files[-1]) as f:
            v5_analysis = json.load(f)

        previous_reports.append({
            'title': 'V5 Reflector Analysis',
            'summary': f"V5 had {v5_analysis['quality_summary']['total_issues']} issues ({v5_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v5_analysis['quality_summary']['grade']}",
            'content_preview': json.dumps({
                'quality_summary': v5_analysis['quality_summary'],
                'issue_categories': v5_analysis['issue_categories'][:5],
                'novel_error_patterns': v5_analysis.get('novel_error_patterns', [])[:3]
            }, indent=2)
        })
        print(f"✅ Loaded V5 Reflector analysis")

    print()
    print("="*80)
    print("🤔 ANALYZING V6 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V6 Improvements Implemented:")
    print("  ✅ POS tagging for intelligent list splitting")
    print("  ✅ Endorsement detection in bibliographic parser")
    print("  ✅ Generic pronoun handler")
    print("  ✅ Expanded vague entity patterns")
    print("  ✅ NEW Predicate normalizer module")
    print("  ✅ Larger pronoun resolution window")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v6_data['relationships'],
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
        print(f"  Grade: {summary.get('grade', 'N/A')}")
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

    # Compare to V5
    if v5_analysis_files:
        print("="*80)
        print("📊 V5 vs V6 REFLECTOR COMPARISON")
        print("="*80)
        print()
        print(f"V5 Results:")
        print(f"  - Total issues: {v5_analysis['quality_summary']['total_issues']} ({v5_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Grade: {v5_analysis['quality_summary']['grade']}")
        print()
        print(f"V6 Results:")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Grade: {summary.get('grade', 'N/A')}")
        print()

        v5_issues = v5_analysis['quality_summary']['total_issues']
        v6_issues = summary.get('total_issues', 0)
        improvement = v5_issues - v6_issues
        improvement_pct = (improvement / v5_issues * 100) if v5_issues > 0 else 0

        print(f"Improvement: {improvement} fewer issues ({improvement_pct:.1f}% reduction)")
        print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)

    target_met = summary.get('issue_rate_percent', 100) < 5.0

    if target_met:
        print("🎉 TARGET MET! Quality issues < 5%")
        print()
        print("1. Review V6 results and validate quality")
        print("2. Apply V6 system to full corpus (172 episodes + 3 books)")
        print("3. Build unified knowledge graph")
    else:
        print("📈 Continue ACE cycle:")
        print()
        print("1. Review V6 Reflector analysis")
        print("2. Implement V7 improvements based on recommendations")
        print("3. Run V7 extraction")
        print("4. Repeat until <5% quality issues")

    print("="*80)


if __name__ == "__main__":
    main()
