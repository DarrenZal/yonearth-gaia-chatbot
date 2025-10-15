#!/usr/bin/env python3
"""
Run KG Reflector on V10 extraction results

Analyzes V10 quality with Comprehensive Knowledge Graph approach
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
    print("🤔 RUNNING KG REFLECTOR ON V10 OUTPUT (COMPREHENSIVE KNOWLEDGE GRAPH)")
    print("="*80)
    print()

    # Paths
    v10_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v10/soil_stewardship_handbook_v8.json")
    v9_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v9/soil_stewardship_handbook_v8.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V10 results
    print("📂 Loading V10 extraction results...")
    with open(v10_output_path) as f:
        v10_data = json.load(f)

    print(f"✅ Loaded {len(v10_data['relationships'])} relationships from V10")
    print()

    # Load V9 results for comparison
    print("📂 Loading V9 extraction results for comparison...")
    with open(v9_output_path) as f:
        v9_data = json.load(f)

    print(f"✅ Loaded {len(v9_data['relationships'])} relationships from V9")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': v10_data['version'],
        'book_title': v10_data['book_title'],
        'date': v10_data['timestamp'],
        'total_relationships': len(v10_data['relationships']),
        'high_confidence_count': sum(1 for r in v10_data['relationships'] if r.get('p_true', 0) >= 0.75),
        'pass2_5_stats': v10_data['pass2_5_stats']
    }

    # Load previous quality reports for context
    print("📊 Loading previous quality reports for context...")
    previous_reports = []

    # V8 Reflector analysis for comparison context
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
    v8_analysis_files = sorted(analysis_dir.glob("reflection_v8_*.json"))
    v8_analysis = None
    if v8_analysis_files:
        with open(v8_analysis_files[-1]) as f:
            v8_analysis = json.load(f)

        previous_reports.append({
            'title': 'V8 Reflector Analysis',
            'summary': f"V8 had {v8_analysis['quality_summary']['total_issues']} issues ({v8_analysis['quality_summary']['issue_rate_percent']}%)",
            'content_preview': json.dumps({'quality_summary': v8_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V8 Reflector analysis")

    # Load V9 Reflector analysis for comparison context
    v9_analysis_files = sorted(analysis_dir.glob("reflection_v9_*.json"))
    v9_analysis = None
    if v9_analysis_files:
        with open(v9_analysis_files[-1]) as f:
            v9_analysis = json.load(f)

        v9_grade = v9_analysis['quality_summary'].get('grade_confirmed') or v9_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V9 Reflector Analysis',
            'summary': f"V9 had {v9_analysis['quality_summary']['total_issues']} issues ({v9_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v9_grade}",
            'content_preview': json.dumps({'quality_summary': v9_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V9 Reflector analysis")

    print()
    print("="*80)
    print("🤔 ANALYZING V10 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V10 Comprehensive Knowledge Graph Approach:")
    print("  ✅ Enhanced Pass 1 with explicit relationship type examples")
    print("  ✅ Bibliographic, categorical, compositional, functional, organizational relationships")
    print("  ✅ Evidence-based targets: 250+ bibliographic, 70+ categorical")
    print("  ✅ NO arbitrary relationship count targets")
    print("  ✅ 100% attribution and classification")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v10_data['relationships'],
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
    analysis_files = sorted(analysis_dir.glob("reflection_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V9
    if v9_analysis:
        print("="*80)
        print("📊 V9 vs V10 COMPARISON")
        print("="*80)
        print()
        v9_grade_display = v9_analysis['quality_summary'].get('grade_confirmed') or v9_analysis['quality_summary'].get('grade', 'N/A')
        print(f"V9 Results (Complete Discourse Graph):")
        print(f"  - Total relationships: {len(v9_data['relationships'])}")
        print(f"  - Total issues: {v9_analysis['quality_summary']['total_issues']} ({v9_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v9_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v9_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v9_grade_display}")
        print()
        print(f"V10 Results (Comprehensive Knowledge Graph):")
        print(f"  - Total relationships: {len(v10_data['relationships'])}")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
        print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
        print()

        v9_issues = v9_analysis['quality_summary']['total_issues']
        v10_issues = summary.get('total_issues', 0)

        v9_rel_count = len(v9_data['relationships'])
        v10_rel_count = len(v10_data['relationships'])
        rel_improvement = v10_rel_count - v9_rel_count
        rel_improvement_pct = (rel_improvement / v9_rel_count * 100) if v9_rel_count > 0 else 0

        print(f"Relationship count change: +{rel_improvement} relationships ({rel_improvement_pct:+.1f}%)")

        if v9_issues > 0:
            issue_improvement = v9_issues - v10_issues
            issue_improvement_pct = (issue_improvement / v9_issues * 100)
            print(f"Quality improvement: {issue_improvement:+d} issues ({issue_improvement_pct:+.1f}% change)")
        print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)

    target_met = summary.get('issue_rate_percent', 100) < 3.0

    if target_met:
        print("🎉 TARGET MET! Quality issues < 3% (A++ grade)")
        print()
        print("1. ✅ V10 achieves production quality (A++ grade)")
        print("2. 📊 Document V10 results in V10_COMPREHENSIVE_KG_RESULTS.md")
        print("3. 🚀 Deploy V10 to KGC production system")
        print("4. 📚 Apply V10 to full corpus (172 episodes + 3 books)")
    else:
        print("📈 Continue ACE cycle:")
        print()
        print("1. Review V10 Reflector analysis")
        print("2. Identify remaining systematic issues")
        print("3. Run Curator to generate V11 improvements")
        print("4. Repeat until <3% quality issues")

    print("="*80)


if __name__ == "__main__":
    main()
