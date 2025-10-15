#!/usr/bin/env python3
"""
Run KG Reflector on V5 extraction results

Analyzes V5 quality and generates improvement recommendations for V6
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
    print("🤔 RUNNING KG REFLECTOR ON V5 OUTPUT")
    print("="*80)
    print()

    # Paths
    v5_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v5/soil_stewardship_handbook_v5.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V5 results
    print("📂 Loading V5 extraction results...")
    with open(v5_output_path) as f:
        v5_data = json.load(f)

    print(f"✅ Loaded {len(v5_data['relationships'])} relationships from V5")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': v5_data['version'],
        'book_title': v5_data['book_title'],
        'date': v5_data['timestamp'],
        'total_relationships': len(v5_data['relationships']),
        'high_confidence_count': v5_data['high_confidence_count'],
        'pass2_5_stats': v5_data['pass2_5_stats']
    }

    # Load V4 quality reports (for training context)
    print("📊 Loading V4 quality reports for context...")
    v4_reports = []

    v4_report_path = Path("/home/claudeuser/yonearth-gaia-chatbot/docs/knowledge_graph/V4_EXTRACTION_QUALITY_ISSUES_REPORT.md")
    if v4_report_path.exists():
        with open(v4_report_path) as f:
            v4_report_content = f.read()

        v4_reports.append({
            'title': 'V4 Quality Issues Report',
            'summary': 'V4 had 57% quality issues across 7 categories',
            'content_preview': v4_report_content[:2000]
        })
        print(f"✅ Loaded V4 quality report")

    print()
    print("="*80)
    print("🤔 ANALYZING V5 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v5_data['relationships'],
        source_text=source_text,
        extraction_metadata=extraction_metadata,
        v4_quality_reports=v4_reports
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

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print("1. Review the full analysis report")
    print("2. Run KG Curator to generate V6 improvement changeset")
    print("3. Human reviews and approves proposed changes")
    print("4. Apply changes and run V6 extraction")
    print("5. Repeat ACE cycle until <5% quality issues")
    print("="*80)


if __name__ == "__main__":
    main()
