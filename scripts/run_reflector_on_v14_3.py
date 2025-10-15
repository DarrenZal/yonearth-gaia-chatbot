#!/usr/bin/env python3
"""
Run KG Reflector on V14.3 extraction results

Analyzes V14.3 baseline restoration (V12 Pass 1 + V13.1 Pass 2 + V13.1 Pass 2.5)
Phase 1 of incremental improvement strategy: Establish stable A- baseline
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
    print("🤔 RUNNING KG REFLECTOR ON V14.3 OUTPUT")
    print("="*80)
    print()

    # Paths
    v14_3_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v14_3/soil_stewardship_handbook_v14_3.json")
    v14_2_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v14_2/soil_stewardship_handbook_v14_2.json")
    v13_1_output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v13/soil_stewardship_handbook_v13_1_from_v12.json")
    book_pdf_path = Path("/home/claudeuser/yonearth-gaia-chatbot/data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf")

    # Load V14.3 results
    print("📂 Loading V14.3 extraction results...")
    with open(v14_3_output_path) as f:
        v14_3_data = json.load(f)

    print(f"✅ Loaded {len(v14_3_data['relationships'])} relationships from V14.3")
    print()

    # Load V14.2 results for comparison context
    print("📂 Loading V14.2 extraction results for comparison...")
    with open(v14_2_output_path) as f:
        v14_2_data = json.load(f)

    print(f"✅ Loaded {len(v14_2_data['relationships'])} relationships from V14.2")
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
        'version': 'v14.3',
        'book_title': v14_3_data['metadata']['book_title'],
        'date': v14_3_data['metadata']['extraction_date'],
        'total_relationships': len(v14_3_data['relationships']),
        'high_confidence_count': v14_3_data['extraction_stats']['high_confidence'],
        'pass2_5_stats': v14_3_data['postprocessing_stats']
    }

    # Prepare previous reports (V13.1 and V14.2)
    print("📊 Loading V13.1 and V14.2 Reflector analyses for comparison...")
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

    # V14.2 analysis
    v14_2_analysis_files = sorted(analysis_dir.glob("reflection_v14.2_*.json"))
    v14_2_analysis = None
    if v14_2_analysis_files:
        with open(v14_2_analysis_files[-1]) as f:
            v14_2_analysis = json.load(f)

        v14_2_grade = v14_2_analysis['quality_summary'].get('grade_confirmed') or v14_2_analysis['quality_summary'].get('grade', 'N/A')
        previous_reports.append({
            'title': 'V14.2 Reflector Analysis (C+ catastrophic failure)',
            'summary': f"V14.2 had {v14_2_analysis['quality_summary']['total_issues']} issues ({v14_2_analysis['quality_summary']['issue_rate_percent']}%), Grade: {v14_2_grade}",
            'content_preview': json.dumps({'quality_summary': v14_2_analysis['quality_summary']}, indent=2)
        })
        print(f"✅ Loaded V14.2 Reflector analysis for context")
    print()

    print("="*80)
    print("🤔 ANALYZING V14.3 QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 2-3 minutes for comprehensive analysis...")
    print()
    print("V14.3 Full Baseline Restoration Strategy:")
    print("  ✅ V12 Pass 1 Prompt: Simple predicates (authored, contains, is-a)")
    print("  ✅ V13.1 Pass 2 Prompt: Dual-signal evaluation (proven A- baseline)")
    print("  ✅ V13.1 Pass 2.5 Pipeline: 12 modules (proven A- grade configuration)")
    print()
    print("🔍 Phase 1 Goal:")
    print("  - Establish stable A- baseline (8.6% issue rate)")
    print("  - Verify PredicateNormalizer works with V12 prompts")
    print("  - Foundation for incremental improvements (Phases 2-5)")
    print()
    print(f"📊 V14.3 Results: {len(v14_3_data['relationships'])} relationships, {v14_3_data['extraction_stats']['high_confidence']} high conf ({v14_3_data['extraction_stats']['high_confidence']/len(v14_3_data['relationships'])*100:.1f}%)")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=v14_3_data['relationships'],
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
    analysis_files = sorted(analysis_dir.glob("reflection_v14.3_*.json"))
    if analysis_files:
        latest_analysis = analysis_files[-1]
        print(f"📁 Full analysis saved to: {latest_analysis}")
        print()

    # Compare to V13.1 and V14.2
    if v13_1_analysis and v14_2_analysis:
        print("="*80)
        print("📊 V13.1 vs V14.2 vs V14.3 COMPARISON")
        print("="*80)
        print()

        v13_1_grade_display = v13_1_analysis['quality_summary'].get('grade_confirmed') or v13_1_analysis['quality_summary'].get('grade', 'N/A')
        v14_2_grade_display = v14_2_analysis['quality_summary'].get('grade_confirmed') or v14_2_analysis['quality_summary'].get('grade', 'N/A')

        print(f"V13.1 Results (A- baseline):")
        print(f"  - Total relationships: {len(v13_1_data['relationships'])}")
        print(f"  - High confidence: {v13_1_data['extraction_stats']['high_confidence']} ({v13_1_data['extraction_stats']['high_confidence']/len(v13_1_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {v13_1_analysis['quality_summary']['total_issues']} ({v13_1_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v13_1_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v13_1_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v13_1_grade_display}")
        print()

        print(f"V14.2 Results (C+ catastrophic failure):")
        print(f"  - Total relationships: {len(v14_2_data['relationships'])}")
        print(f"  - High confidence: {v14_2_data['extraction_stats']['high_confidence']} ({v14_2_data['extraction_stats']['high_confidence']/len(v14_2_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {v14_2_analysis['quality_summary']['total_issues']} ({v14_2_analysis['quality_summary']['issue_rate_percent']}%)")
        print(f"  - Critical: {v14_2_analysis['quality_summary']['critical_issues']}")
        print(f"  - High: {v14_2_analysis['quality_summary']['high_priority_issues']}")
        print(f"  - Grade: {v14_2_grade_display}")
        print()

        print(f"V14.3 Results (full baseline restoration):")
        print(f"  - Total relationships: {len(v14_3_data['relationships'])}")
        print(f"  - High confidence: {v14_3_data['extraction_stats']['high_confidence']} ({v14_3_data['extraction_stats']['high_confidence']/len(v14_3_data['relationships'])*100:.1f}%)")
        print(f"  - Total issues: {summary.get('total_issues', 'N/A')} ({summary.get('issue_rate_percent', 'N/A')}%)")
        print(f"  - Critical: {summary.get('critical_issues', 'N/A')}")
        print(f"  - High: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  - Grade: {summary.get('grade_confirmed', 'N/A')}")
        print()

        v13_1_issues = v13_1_analysis['quality_summary']['total_issues']
        v14_2_issues = v14_2_analysis['quality_summary']['total_issues']
        v14_3_issues = summary.get('total_issues', 0)

        # Compare V14.3 to V14.2 (catastrophic failure)
        issue_improvement_from_v14_2 = v14_2_issues - v14_3_issues
        issue_improvement_from_v14_2_pct = (issue_improvement_from_v14_2 / v14_2_issues * 100) if v14_2_issues > 0 else 0

        # Compare V14.3 to V13.1 (baseline)
        issue_difference_from_v13_1 = v13_1_issues - v14_3_issues
        issue_difference_from_v13_1_pct = (issue_difference_from_v13_1 / v13_1_issues * 100) if v13_1_issues > 0 else 0

        print(f"V14.3 vs V14.2: {issue_improvement_from_v14_2:+d} issues ({issue_improvement_from_v14_2_pct:+.1f}% change)")
        print(f"V14.3 vs V13.1: {issue_difference_from_v13_1:+d} issues ({issue_difference_from_v13_1_pct:+.1f}% change)")
        print()

    print("="*80)
    print("🎯 FINAL RECOMMENDATION")
    print("="*80)

    if v13_1_analysis and v14_2_analysis and 'quality_summary' in analysis:
        v13_1_grade = v13_1_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v14_2_grade = v14_2_analysis['quality_summary'].get('grade_confirmed', 'N/A')
        v14_3_grade = summary.get('grade_confirmed', 'N/A')

        v13_1_issues = v13_1_analysis['quality_summary']['total_issues']
        v14_2_issues = v14_2_analysis['quality_summary']['total_issues']
        v14_3_issues = summary.get('total_issues', 0)

        # Check if V14.3 reached A- target (Phase 1 goal)
        v14_3_grade_value = v14_3_grade if isinstance(v14_3_grade, str) else 'N/A'
        reached_a_minus = v14_3_grade_value == 'A-'

        if reached_a_minus and v14_3_issues <= v13_1_issues + 5:  # Allow small variance
            print("🎉 RECOMMENDATION: PHASE 1 COMPLETE - STABLE BASELINE ESTABLISHED")
            print(f"   - Reached Phase 1 goal: {v14_3_grade} (A- baseline restored!)")
            print(f"   - Fixed V14.2 catastrophic failure: {v14_2_issues - v14_3_issues} issues fixed")
            print(f"   - Quality matches V13.1 baseline")
            print(f"   - PredicateNormalizer works correctly with V12 prompts")
            print("   - Ready to proceed to Phase 2")
            print()
            print("🔬 PHASE 2: Enhance PredicateNormalizer (A- → A)")
            print("   GOAL: Reduce predicate fragmentation from 1.1% to <0.5%")
            print("   - Add 20-30 new normalization rules for common patterns")
            print("   - Target: A grade (5-8% issue rate)")
            print("   - Keep V12 Pass 1 + V13.1 Pass 2 prompts unchanged")
            print()
            print("🔬 FUTURE PHASES:")
            print("   Phase 3: Add PhilosophicalClaimFilter (A → A)")
            print("   Phase 4: Add MetadataFilter (A → A+)")
            print("   Phase 5: Add ConfidenceFilter (A+ → <5%)")
        elif v14_3_grade_value in ['A+', 'A', 'B+', 'B']:
            print(f"✅ RECOMMENDATION: V14.3 shows improvement but needs adjustment")
            print(f"   - Fixed V14.2 catastrophic failure: {v14_2_issues - v14_3_issues} issues fixed")
            print(f"   - Grade: {v14_3_grade} (target was A-)")
            print(f"   - Issues: {v14_3_issues} (V13.1: {v13_1_issues})")
            print()
            print("🔬 NEXT STEPS:")
            if v14_3_grade_value == 'B+' or v14_3_grade_value == 'B':
                print("   1. 📊 Analyze which issue categories most problematic")
                print("   2. 🔧 Create V14.3.1 with targeted fixes")
                print("   3. 🎯 Retry Phase 1 until A- achieved")
            else:
                print("   1. 📊 Document which issue categories improved")
                print("   2. 🔬 Proceed to Phase 2 (enhance PredicateNormalizer)")
        else:
            print("⚠️  RECOMMENDATION: V14.3 baseline restoration incomplete")
            print(f"   - Issues: {v14_3_issues} (V13.1: {v13_1_issues}, V14.2: {v14_2_issues})")
            print(f"   - Grade: {v14_3_grade}")
            print("   - Full V13.1 config didn't restore quality")
            print()
            print("🔬 NEXT STEPS:")
            print("   1. 🔬 Investigate unexpected issue patterns")
            print("   2. 📊 Compare V14.3 vs V13.1 issue categories")
            print("   3. 🔧 Debug why identical config produces different results")

    print("="*80)


if __name__ == "__main__":
    main()
