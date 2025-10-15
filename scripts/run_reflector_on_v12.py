#!/usr/bin/env python3
"""
Run KG Reflector on V12 extraction results

Analyzes V12 quality with dual-signal evaluation + penalties
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
    print("🤔 RUNNING KG REFLECTOR ON V12 OUTPUT (WITH PENALTIES)")
    print("="*80)
    print()

    # Paths
    v12_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v12/soil_stewardship_handbook_v12.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V12 results
    print("📂 Loading V12 extraction results...")
    with open(v12_output_path) as f:
        v12_data = json.load(f)

    print(f"✅ Loaded {len(v12_data['relationships'])} relationships from V12")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': 'v12',
        'book_title': v12_data['metadata']['book_title'],
        'date': v12_data['metadata']['extraction_date'],
        'total_relationships': len(v12_data['relationships']),
        'high_confidence_count': v12_data['extraction_stats']['high_confidence'],
        'pass2_5_stats': v12_data['postprocessing_stats']
    }

    print("="*80)
    print("🤔 ANALYZING V12 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V12 Dual-Signal with Penalties:")
    print("  ✅ entity_specificity_score (concreteness)")
    print("  ✅ classification_flags (FACTUAL, TESTABLE, PHILOSOPHICAL, NORMATIVE, etc.)")
    print("  ⚠️  PHILOSOPHICAL penalty: -0.4, cap at 0.3")
    print("  ⚠️  NORMATIVE penalty: -0.3, cap at 0.4")
    print("  ✅ 12 postprocessing modules")
    print(f"  📊 Results: {len(v12_data['relationships'])} relationships, {v12_data['extraction_stats']['high_confidence']} high conf ({v12_data['extraction_stats']['high_confidence']/len(v12_data['relationships'])*100:.1f}%)")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v12_data['relationships'],
        source_text=source_text,
        extraction_metadata=extraction_metadata,
        v4_quality_reports=[]
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
    analysis_files = sorted(analysis_dir.glob("reflection_v12_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    print("="*80)
    print("🎯 NEXT STEP")
    print("="*80)
    print("Compare V12 (WITH penalties) vs V13.1 (NO penalties) to measure penalty impact")
    print("="*80)


if __name__ == "__main__":
    main()
