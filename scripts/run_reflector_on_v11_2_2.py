#!/usr/bin/env python3
"""
Run KG Reflector on V11.2.2 extraction results

Analyzes V11.2.2 quality improvements from targeted bug fixes:
1. Dedication Parser: Extract only proper names
2. ListSplitter: POS tagging for intelligent splitting
3. Predicate Normalizer: 173 → 125 unique predicates

Target: B grade or better (<15% error rate)
Baseline: V11.2.1 had C- grade (21.85% error rate)
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
    print("🔍 RUNNING KG REFLECTOR ON V11.2.2 (QUALITY FIXES)")
    print("="*80)
    print()

    # Paths
    v11_2_2_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v11_2_2/soil_stewardship_handbook_v11_2_2.json")
    v11_2_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v11_2_1/soil_stewardship_handbook_v11_2_1.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V11.2.2 results
    print("📂 Loading V11.2.2 extraction results...")
    with open(v11_2_2_output_path) as f:
        v11_2_2_data = json.load(f)

    print(f"✅ Loaded {len(v11_2_2_data['relationships'])} relationships from V11.2.2")
    print()

    # Load V11.2.1 results for comparison
    print("📂 Loading V11.2.1 extraction results for comparison...")
    with open(v11_2_1_output_path) as f:
        v11_2_1_data = json.load(f)

    print(f"✅ Loaded {len(v11_2_1_data['relationships'])} relationships from V11.2.1")
    print()

    # Extract book text
    source_text = extract_text_from_pdf(book_pdf_path)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': v11_2_2_data['metadata']['extraction_version'],
        'book_title': v11_2_2_data['metadata']['book_title'],
        'date': v11_2_2_data['metadata']['extraction_date'],
        'total_relationships': len(v11_2_2_data['relationships']),
        'high_confidence_count': v11_2_2_data['extraction_stats']['high_confidence'],
        'pass2_5_stats': v11_2_2_data['postprocessing_stats']
    }

    # Load previous quality reports for context (optional)
    print("📊 Loading previous quality reports for context...")
    previous_reports = []
    v11_2_1_analysis = None

    # Add manual baseline context about V11.2.1
    previous_reports.append({
        'title': 'V11.2.1 Baseline (BEFORE fixes)',
        'summary': 'V11.2.1 had Grade C- with 21.85% error rate. Key issues: 28 malformed dedications, 18 bad list splits, 173 unique predicates.',
        'content_preview': json.dumps({
            'baseline_issues': {
                'malformed_dedications': 28,
                'bad_list_splits': 18,
                'unique_predicates': 173,
                'grade': 'C-',
                'error_rate': '21.85%'
            }
        }, indent=2)
    })
    print(f"✅ Added V11.2.1 baseline context (C-, 21.85% errors)")

    print()
    print("="*80)
    print("🤔 ANALYZING V11.2.2 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 3-5 minutes for comprehensive analysis...")
    print()
    print("🎯 V11.2.2 TARGETED FIXES:")
    print("  1. ✅ Dedication Parser: Extract only proper names (fix 28 malformed relationships)")
    print("  2. ✅ ListSplitter: POS tagging for intelligent splitting (fix 18 bad splits)")
    print("  3. ✅ Predicate Normalizer: 173 → 125 unique predicates (28% reduction)")
    print()
    print("📊 BASELINE (V11.2.1): C- grade, 21.85% error rate")
    print("🎯 TARGET: B grade or better (<15% error rate)")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v11_2_2_data['relationships'],
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
        print("🏆 V11.2.2 QUALITY SUMMARY:")
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

    # Find the saved analysis file
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_files = sorted(analysis_dir.glob("reflection_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V11.2.1 (baseline)
    print("="*80)
    print("📊 V11.2.1 (BASELINE) vs V11.2.2 (WITH FIXES) COMPARISON")
    print("="*80)
    print()
    print(f"V11.2.1 Results (Baseline - BEFORE fixes):")
    print(f"  - Total relationships: {len(v11_2_1_data['relationships'])}")
    print(f"  - Grade: C- (21.85% error rate)")
    print(f"  - Issues: 28 malformed dedications + 18 bad list splits + 173 predicates")
    print()
    print(f"V11.2.2 Results (WITH 3 targeted fixes):")
    print(f"  - Total relationships: {len(v11_2_2_data['relationships'])}")
    print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
    print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
    print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
    print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
    print()

    v11_2_1_rel_count = len(v11_2_1_data['relationships'])
    v11_2_2_rel_count = len(v11_2_2_data['relationships'])
    rel_change = v11_2_2_rel_count - v11_2_1_rel_count
    rel_change_pct = (rel_change / v11_2_1_rel_count * 100) if v11_2_1_rel_count > 0 else 0

    print(f"Relationship count change: {rel_change:+d} relationships ({rel_change_pct:+.1f}%)")
    print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)

    target_met = summary.get('issue_rate_percent', 100) < 15.0  # B grade target

    if target_met:
        print("🎉 TARGET MET! V11.2.2 achieves B grade or better (<15% error rate)")
        print()
        print("1. ✅ V11.2.2 fixes successfully improved quality")
        print("2. 📊 Document V11.2.2 results")
        print("3. 🔄 V11.2.2 becomes new baseline")
        print("4. 🎯 Continue ACE cycle for further improvements (target: A grade, <5% errors)")
    else:
        print("📈 Target not yet met - Continue ACE cycle:")
        print()
        print("1. Review V11.2.2 Reflector analysis for remaining issues")
        print("2. Run Curator to generate V11.2.3 improvements")
        print("3. ✨ Use NEW Validator to test fixes on problem chunks (5 min vs 52 min)")
        print("4. Repeat until <15% error rate (B grade)")

    print("="*80)


if __name__ == "__main__":
    main()
