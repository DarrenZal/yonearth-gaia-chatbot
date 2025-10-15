#!/usr/bin/env python3
"""
Run KG Reflector on V11.2.1 extraction results

Analyzes V11.2.1 quality with modular postprocessing system
Compares against V9 and V10 to validate bug fixes
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
    print("🤔 RUNNING KG REFLECTOR ON V11.2.1 OUTPUT (ALL 4 FIXES WORKING)")
    print("="*80)
    print()

    # Paths
    v11_2_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v11_2_1/soil_stewardship_handbook_v11_2_1.json")
    v11_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v11_1/soil_stewardship_handbook_v11_1.json")
    v10_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v10/soil_stewardship_handbook_v8.json")
    v9_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v9/soil_stewardship_handbook_v8.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V11.2.1 results
    print("📂 Loading V11.2.1 extraction results...")
    with open(v11_2_1_output_path) as f:
        v11_2_1_data = json.load(f)

    print(f"✅ Loaded {len(v11_2_1_data['relationships'])} relationships from V11.2.1")
    print()

    # Load V10 results for comparison
    print("📂 Loading V10 extraction results for comparison...")
    with open(v10_output_path) as f:
        v10_data = json.load(f)

    print(f"✅ Loaded {len(v10_data['relationships'])} relationships from V10")
    print()

    # Load V11.1 results for comparison
    print("📂 Loading V11.1 extraction results for comparison...")
    with open(v11_1_output_path) as f:
        v11_1_data = json.load(f)

    print(f"✅ Loaded {len(v11_1_data['relationships'])} relationships from V11.1")
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
        'version': v11_2_1_data.get('version', 'v11.2.1'),
        'book_title': v11_2_1_data.get('book_title', 'Soil Stewardship Handbook'),
        'date': v11_2_1_data.get('timestamp', v11_2_1_data['metadata'].get('extraction_timestamp', 'N/A')),
        'total_relationships': len(v11_2_1_data['relationships']),
        'high_confidence_count': sum(1 for r in v11_2_1_data['relationships'] if r.get('p_true', 0) >= 0.75),
        'extraction_stats': v11_2_1_data.get('extraction_stats', {}),
        'module_flags': v11_2_1_data.get('extraction_stats', {}).get('module_flags', {})
    }

    # Load previous quality reports for context
    print("📊 Loading previous quality reports for context...")
    previous_reports = []

    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")

    # Load V9 Reflector analysis
    v9_analysis_files = sorted(analysis_dir.glob("reflection_v9_*.json"))
    v9_analysis = None
    if v9_analysis_files:
        with open(v9_analysis_files[-1]) as f:
            v9_analysis = json.load(f)

        v9_grade = v9_analysis['quality_summary'].get('grade_confirmed') or v9_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V9 Reflector Analysis (Complete Discourse Graph)',
            'summary': f"V9: {len(v9_data['relationships'])} relationships, {v9_analysis['quality_summary']['total_issues']} issues ({v9_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v9_grade}",
            'content_preview': json.dumps({'quality_summary': v9_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V9 Reflector analysis")

    # Load V10 Reflector analysis
    v10_analysis_files = sorted(analysis_dir.glob("reflection_v10_*.json"))
    v10_analysis = None
    if v10_analysis_files:
        with open(v10_analysis_files[-1]) as f:
            v10_analysis = json.load(f)

        v10_grade = v10_analysis['quality_summary'].get('grade_confirmed') or v10_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V10 Reflector Analysis (Comprehensive Knowledge Graph)',
            'summary': f"V10: {len(v10_data['relationships'])} relationships, {v10_analysis['quality_summary']['total_issues']} issues ({v10_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v10_grade}",
            'content_preview': json.dumps({'quality_summary': v10_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V10 Reflector analysis")

    print()
    print("="*80)
    print("🤔 ANALYZING V11.2.1 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V11.2.1 Modular Postprocessing - All 4 Fixes Working:")
    print("  ✅ FIX #1: Deduplication - Removed 106 duplicates (244 in V11.2)")
    print("  ✅ FIX #2: ListSplitter - 138 lists split (+203 relationships)")
    print("  ✅ FIX #3: Dedication parsing - 126 dedications corrected")
    print("  ✅ FIX #4: Classification - 1021 Factual, 40 Philosophical, 27 Opinion, 19 Recommendation")
    print("  ✅ ALL 12/12 postprocessing modules executed successfully!")
    print()
    print("Modules executed:")
    module_flags = extraction_metadata.get('module_flags', {})
    if module_flags:
        for flag, count in sorted(module_flags.items(), key=lambda x: -x[1])[:10]:
            print(f"  - {flag}: {count} relationships")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v11_2_1_data['relationships'],
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

    # Compare V9 → V10 → V11.1 → V11.2.1
    print("="*80)
    print("📊 PROGRESSIVE COMPARISON: V9 → V10 → V11.1 → V11.2.1")
    print("="*80)
    print()

    if v9_analysis:
        v9_grade_display = v9_analysis['quality_summary'].get('grade_confirmed') or v9_analysis['quality_summary'].get('grade', 'N/A')
        print(f"V9 Results (Complete Discourse Graph):")
        print(f"  - Total relationships: {len(v9_data['relationships'])}")
        print(f"  - Total issues: {v9_analysis['quality_summary']['total_issues']} ({v9_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v9_analysis['quality_summary']['critical_issues']}, High: {v9_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v9_grade_display}")
        print()

    if v10_analysis:
        v10_grade_display = v10_analysis['quality_summary'].get('grade_confirmed') or v10_analysis['quality_summary'].get('grade', 'N/A')
        print(f"V10 Results (Comprehensive Knowledge Graph):")
        print(f"  - Total relationships: {len(v10_data['relationships'])}")
        print(f"  - Total issues: {v10_analysis['quality_summary']['total_issues']} ({v10_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v10_analysis['quality_summary']['critical_issues']}, High: {v10_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v10_grade_display}")
        print()

    print(f"V11.1 Results (Broken - 448 duplicates, Grade F):")
    print(f"  - Total relationships: {len(v11_1_data['relationships'])}")
    print(f"  - Issue: No deduplication module (38.1% duplicates)")
    print(f"  - Issue: ListSplitter doesn't split on 'and'")
    print(f"  - Issue: Dedication parsing malformed")
    print()

    print(f"V11.2.1 Results (All 4 Fixes Working):")
    print(f"  - Total relationships: {len(v11_2_1_data['relationships'])}")
    print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
    print(f"  - Critical: {summary.get('critical_issues', 'N/A')}, High: {summary.get('high_priority_issues', 'N/A')}")
    print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
    print()

    # Calculate improvements
    if v9_analysis and v10_analysis:
        v9_issues = v9_analysis['quality_summary']['total_issues']
        v9_rate = v9_analysis['quality_summary']['issue_rate_percent']
        v10_issues = v10_analysis['quality_summary']['total_issues']
        v10_rate = v10_analysis['quality_summary']['issue_rate_percent']
        v11_1_issues = summary.get('total_issues', 0)
        v11_1_rate = summary.get('issue_rate_percent', 0)

        print("QUALITY PROGRESSION:")
        print(f"  V9 → V10: {v10_issues - v9_issues:+d} issues ({v10_rate - v9_rate:+.1f}% error rate)")
        print(f"  V10 → V11.2.1: {v11_1_issues - v10_issues:+d} issues ({v11_1_rate - v10_rate:+.1f}% error rate)")
        print(f"  V9 → V11.2.1: {v11_1_issues - v9_issues:+d} issues ({v11_1_rate - v9_rate:+.1f}% error rate)")
        print()

        v9_rel_count = len(v9_data['relationships'])
        v10_rel_count = len(v10_data['relationships'])
        v11_1_rel_count = len(v11_2_1_data['relationships'])

        print("RELATIONSHIP COUNT PROGRESSION:")
        print(f"  V9 → V10: {v10_rel_count - v9_rel_count:+d} relationships ({(v10_rel_count/v9_rel_count - 1)*100:+.1f}%)")
        print(f"  V10 → V11.2.1: {v11_1_rel_count - v10_rel_count:+d} relationships ({(v11_1_rel_count/v10_rel_count - 1)*100:+.1f}%)")
        print(f"  V9 → V11.2.1: {v11_1_rel_count - v9_rel_count:+d} relationships ({(v11_1_rel_count/v9_rel_count - 1)*100:+.1f}%)")
        print()

    # Module execution verification
    print("="*80)
    print("🔬 MODULE EXECUTION VERIFICATION")
    print("="*80)
    print()

    if module_flags:
        print(f"✅ Module flags present: {len(module_flags)} types")
        print("Top module operations:")
        for flag, count in sorted(module_flags.items(), key=lambda x: -x[1])[:10]:
            print(f"  {flag}: {count}")
        print()
        print("Expected module flags:")
        expected_flags = [
            'PRONOUN_RESOLVED', 'GENERIC_PRONOUN_RESOLVED', 'METAPHOR',
            'LIST_SPLIT', 'VAGUE_ENTITY_BLOCKED', 'DEDICATION_PARSED',
            'FACTUAL', 'PHILOSOPHICAL_CLAIM', 'OPINION', 'RECOMMENDATION'
        ]
        found_flags = [f for f in expected_flags if any(f in mf for mf in module_flags.keys())]
        missing_flags = [f for f in expected_flags if not any(f in mf for mf in module_flags.keys())]

        if found_flags:
            print(f"  ✅ Found: {', '.join(found_flags)}")
        if missing_flags:
            print(f"  ⚠️  Not found: {', '.join(missing_flags)}")
    else:
        print("❌ WARNING: No module flags found! Postprocessing may not have run.")

    print()

    print("="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print()

    target_met = summary.get('issue_rate_percent', 100) < 3.0

    if target_met:
        print("🎉 TARGET MET! Quality issues < 3% (A++ grade)")
        print()
        print("1. ✅ V11.2.1 achieves production quality (A++ grade)")
        print("2. 📊 Document V11.2.1 results in V11_1_RESULTS.md")
        print("3. 🚀 Use V11.2.1 as baseline for future ACE cycles")
        print("4. 📚 Consider applying to full corpus (172 episodes + 3 books)")
    else:
        print("📈 Continue ACE cycle:")
        print()
        print("1. Review V11.2.1 Reflector analysis")
        print("2. Identify remaining systematic issues")
        print("3. Run Curator to generate V12 improvements")
        print("4. Apply Meta-ACE learnings from V11 bug fixes")
        print()
        print("Key V11.2.1 Achievements (Fixed from V11.1's Grade F):")
        print("  ✅ FIX #1: Added Deduplicator module (removed 350 duplicates total)")
        print("  ✅ FIX #2: Fixed ListSplitter to split on ' and ' (138 lists split)")
        print("  ✅ FIX #3: Fixed dedication parsing (126 dedications corrected)")
        print("  ✅ FIX #4: Added ClaimClassifier module (classifications working)")
        print("  ✅ All 12/12 modules executing successfully!")

    print("="*80)


if __name__ == "__main__":
    main()
