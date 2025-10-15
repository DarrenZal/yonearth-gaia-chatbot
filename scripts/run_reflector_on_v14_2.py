#!/usr/bin/env python3
"""
Run KG Reflector on V14.2 extraction results

Analyzes V14.2 conservative rollback approach (V14 Pass 1 + V14 Pass 2 + V13.1 Pass 2.5)
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
    print("🤔 RUNNING KG REFLECTOR ON V14.2 OUTPUT")
    print("="*80)
    print()

    # Paths
    v14_2_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v14_2/soil_stewardship_handbook_v14_2.json")
    v14_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v14/soil_stewardship_handbook_v14.json")
    v13_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v13/soil_stewardship_handbook_v13_1_from_v12.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V14.2 results
    print("📂 Loading V14.2 extraction results...")
    with open(v14_2_output_path) as f:
        v14_2_data = json.load(f)

    print(f"✅ Loaded {len(v14_2_data['relationships'])} relationships from V14.2")
    print()

    # Load V14.0 results for comparison context
    print("📂 Loading V14.0 extraction results for comparison...")
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
        'version': 'v14.2',
        'book_title': v14_2_data['metadata']['book_title'],
        'date': v14_2_data['metadata']['extraction_date'],
        'total_relationships': len(v14_2_data['relationships']),
        'high_confidence_count': v14_2_data['extraction_stats']['high_confidence'],
        'pass2_5_stats': v14_2_data['postprocessing_stats']
    }

    # Prepare previous reports (V13.1 and V14.0)
    print("📊 Loading V13.1 and V14.0 Reflector analyses for comparison...")
    previous_reports = []
    analysis_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports")

    # V13.1 analysis
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

    # V14.0 analysis
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
    print()

    print("="*80)
    print("🤔 ANALYZING V14.2 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V14.2 Conservative Rollback Strategy:")
    print("  ✅ V14 Pass 1 Prompt: Filters poetry/quotes (prevents Rumi poetry, praise quotes)")
    print("  ✅ V14 Pass 2 Prompt: Dual-signal evaluation (IDENTICAL to V13.1's A- baseline)")
    print("  ✅ V13.1 Pass 2.5 Pipeline: 12 modules (proven A- grade configuration)")
    print()
    print("🔍 Root Cause Fix:")
    print("  - V14.0 regression (B+) caused by Pass 2.5 changes (12 → 14 modules)")
    print("  - V14.0 added MetadataFilter and ConfidenceFilter (introduced issues)")
    print("  - V14.2 rolls back to V13.1's 12-module configuration")
    print()
    print(f"📊 V14.2 Results: {len(v14_2_data['relationships'])} relationships, {v14_2_data['extraction_stats']['high_confidence']} high conf ({v14_2_data['extraction_stats']['high_confidence']/len(v14_2_data['relationships'])*100:.1f}%)")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v14_2_data['relationships'],
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
    analysis_files = sorted(analysis_dir.glob("reflection_v14_2_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V13.1 and V14.0
    if v13_1_analysis and v14_analysis:
        print("="*80)
        print("📊 V13.1 vs V14.0 vs V14.2 COMPARISON")
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

        print(f"V14.2 Results (conservative rollback):")
        print(f"  - Total relationships: {len(v14_2_data['relationships'])}")
        print(f"  - High confidence: {v14_2_data['extraction_stats']['high_confidence']} ({v14_2_data['extraction_stats']['high_confidence']/len(v14_2_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
        print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
        print()

        v13_1_issues = v13_1_analysis['quality_summary']['total_issues']
        v14_issues = v14_analysis['quality_summary']['total_issues']
        v14_2_issues = summary.get('total_issues', 0)

        # Compare V14.2 to V14.0 (regression)
        issue_improvement_from_v14 = v14_issues - v14_2_issues
        issue_improvement_from_v14_pct = (issue_improvement_from_v14 / v14_issues * 100) if v14_issues > 0 else 0

        # Compare V14.2 to V13.1 (baseline)
        issue_improvement_from_v13_1 = v13_1_issues - v14_2_issues
        issue_improvement_from_v13_1_pct = (issue_improvement_from_v13_1 / v13_1_issues * 100) if v13_1_issues > 0 else 0

        print(f"V14.2 vs V14.0: {issue_improvement_from_v14:+d} issues ({issue_improvement_from_v14_pct:+.1f}% change)")
        print(f"V14.2 vs V13.1: {issue_improvement_from_v13_1:+d} issues ({issue_improvement_from_v13_1_pct:+.1f}% change)")
        print()

    print("="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)

    if v13_1_analysis and v14_analysis and 'quality_summary' in analysis:
        v13_1_grade = v13_1_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v14_grade = v14_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v14_2_grade = summary.get('grade_confirmed', 'N/A')

        v13_1_issues = v13_1_analysis['quality_summary']['total_issues']
        v14_issues = v14_analysis['quality_summary']['total_issues']
        v14_2_issues = summary.get('total_issues', 0)

        # Check if V14.2 reached A or A+ target
        v14_2_grade_value = v14_2_grade if isinstance(v14_2_grade, str) else 'N/A'
        reached_target = v14_2_grade_value in ['A+', 'A', 'A-']

        if v14_2_issues <= v13_1_issues and reached_target:
            print("🎉 RECOMMENDATION: V14.2 ROLLBACK SUCCESSFUL")
            print(f"   - Reached target grade: {v14_2_grade} (A- or better achieved!)")
            print(f"   - Fixed V14.0 regression: {v14_issues - v14_2_issues} issues fixed")
            print(f"   - Quality matches or exceeds V13.1 baseline")
            print("   - Conservative rollback approach validated")
            print("   - Ready to adopt V14.2 as new stable release")
            print()
            print("🔬 NEXT STEPS:")
            print("   1. ✅ Adopt V14.2 as new baseline")
            print("   2. 🔬 Investigate which V14.0 module caused regression")
            print("   3. 🔧 Create V14.3 with fixed MetadataFilter/ConfidenceFilter")
            print("   4. 🎯 Target V15: A+ grade (<5% issue rate)")
        elif v14_2_issues < v14_issues:
            print("✅ RECOMMENDATION: V14.2 shows improvement over V14.0")
            print(f"   - Fixed {v14_issues - v14_2_issues} issues from V14.0 regression")
            print(f"   - Grade: {v14_2_grade} (target was A or A-)")
            print(f"   - Still {abs(v14_2_issues - v13_1_issues)} issues away from V13.1 baseline")
            print()
            print("🔬 NEXT STEPS:")
            print("   1. 📊 Analyze which issue categories still problematic")
            print("   2. 🔧 Create V14.2.1 with targeted fixes")
            print("   3. 🔬 Run ablation tests on individual modules")
        else:
            print("⚠️  RECOMMENDATION: V14.2 rollback incomplete")
            print(f"   - Issues: {v14_2_issues} (V13.1: {v13_1_issues}, V14.0: {v14_issues})")
            print(f"   - Grade: {v14_2_grade}")
            print("   - Conservative rollback didn't fully restore quality")
            print()
            print("🔬 NEXT STEPS:")
            print("   1. ❌ Abandon partial rollback approach")
            print("   2. 🔬 Deep dive into module differences")
            print("   3. 🔬 Consider full V13.1 configuration (V13 Pass 1 + V13 Pass 2 + V13 Pass 2.5)")

    print("="*80)


if __name__ == "__main__":
    main()
