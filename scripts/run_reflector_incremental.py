#!/usr/bin/env python3
"""
Incremental KG Reflector for Section-Specific Analysis

🎯 PURPOSE: Analyze individual chapter/section extractions for quality gates.

**Strategy**:
- Run Reflector on one section at a time
- Generate machine-readable summary for A+ gate checking
- Support automated freeze decisions based on quality criteria
- Save section-specific analysis for iteration tracking

**Usage**:
```bash
# Analyze Front Matter extraction
python3 scripts/run_reflector_incremental.py \\
  --input kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/front_matter_v14_3_3_20251015_143052.json \\
  --book our_biggest_deal \\
  --section front_matter \\
  --pages 1-30

# Analyze Chapter 1
python3 scripts/run_reflector_incremental.py \\
  --input kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/chapter_01_v14_3_3_20251015_150234.json \\
  --book our_biggest_deal \\
  --section chapter_01 \\
  --pages 31-50
```

**Quality Gate Criteria** (A+ Grade):
- 0 CRITICAL issues
- ≤2 HIGH issues
- Issue rate ≤2%

**Outputs**:
- Full reflection report (JSON)
- Machine-readable summary (JSON) for automated quality gates
- Human-readable console output

**Integration**:
- Used by extraction script to check if section achieves A+ grade
- Automated freeze decision based on quality criteria
- Supports iteration tracking (compare across attempts)
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.ace_kg.kg_reflector import KGReflectorAgent
import pdfplumber


def extract_pages_from_pdf(pdf_path: Path, page_range: str) -> str:
    """
    Extract specific page range from PDF.

    Args:
        pdf_path: Path to PDF file
        page_range: Page range string like "1-30" or "51-70"

    Returns:
        Full text from specified pages
    """
    print(f"📖 Extracting pages {page_range} from {pdf_path.name}...")

    # Parse page range
    start, end = map(int, page_range.split('-'))

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start - 1, min(end, len(pdf.pages))):  # 0-indexed
            page = pdf.pages[i]
            text = page.extract_text()
            if text:
                pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    print(f"✅ Extracted {len(full_text.split())} words from pages {start}-{end}")
    return full_text


def generate_quality_summary(
    analysis: dict,
    section: str,
    extraction_file: str,
    total_relationships: int
) -> dict:
    """
    Generate machine-readable quality summary for automated quality gates.

    This enables automated A+ gate checking without parsing natural language.

    Returns dict with:
    - overall_grade: Letter grade (A+, A, A-, B+, etc.)
    - passes_a_plus_gate: Boolean
    - criteria: Dict of specific criteria checks
    - actionable_issues: List of issues that need addressing
    - recommendations: List of improvement recommendations
    """
    quality_summary = analysis.get('quality_summary', {})

    # Extract metrics
    total_issues = quality_summary.get('total_issues', 0)
    critical_issues = quality_summary.get('critical_issues', 0)
    high_issues = quality_summary.get('high_priority_issues', 0)
    medium_issues = quality_summary.get('medium_priority_issues', 0)
    mild_issues = quality_summary.get('mild_issues', 0)
    issue_rate = quality_summary.get('issue_rate_percent', 0.0)
    overall_grade = quality_summary.get('grade_confirmed', quality_summary.get('grade', 'N/A'))

    # Check A+ criteria
    criteria_checks = {
        'critical_issues_zero': critical_issues == 0,
        'high_issues_lte_2': high_issues <= 2,
        'issue_rate_lte_2_percent': issue_rate <= 2.0
    }

    passes_a_plus = all(criteria_checks.values())

    # Extract actionable issues (CRITICAL, HIGH, MEDIUM)
    actionable_issues = []

    if 'issue_categories' in analysis:
        for cat in analysis['issue_categories']:
            if cat.get('severity') in ['CRITICAL', 'HIGH', 'MEDIUM']:
                actionable_issues.append({
                    'severity': cat['severity'],
                    'category': cat['category_name'],
                    'count': cat['count'],
                    'percentage': cat.get('percentage', 0.0),
                    'description': cat.get('description', ''),
                    'examples': cat.get('examples', [])[:2]  # First 2 examples
                })

    # Extract improvement recommendations
    recommendations = []
    if 'improvement_recommendations' in analysis:
        for rec in analysis['improvement_recommendations'][:10]:  # Top 10
            recommendations.append({
                'priority': rec.get('priority', 'UNKNOWN'),
                'type': rec.get('type', 'unknown'),
                'recommendation': rec.get('recommendation', '')
            })

    # Build summary
    summary = {
        'section': section,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'extraction_file': extraction_file,
        'quality_metrics': {
            'overall_grade': overall_grade,
            'issue_rate': issue_rate,
            'total_relationships': total_relationships,
            'total_issues': total_issues
        },
        'issue_breakdown': {
            'CRITICAL': critical_issues,
            'HIGH': high_issues,
            'MEDIUM': medium_issues,
            'MILD': mild_issues
        },
        'quality_gate': {
            'passes_a_plus_gate': passes_a_plus,
            'criteria': criteria_checks
        },
        'actionable_issues': actionable_issues,
        'recommendations': recommendations
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run KG Reflector on incremental section extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Front Matter
  python3 scripts/run_reflector_incremental.py \\
    --input kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/front_matter_v14_3_3_20251015_143052.json \\
    --book our_biggest_deal \\
    --section front_matter \\
    --pages 1-30

  # Analyze Chapter 1
  python3 scripts/run_reflector_incremental.py \\
    --input kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/chapter_01_v14_3_3_20251015_150234.json \\
    --book our_biggest_deal \\
    --section chapter_01 \\
    --pages 31-50
        """
    )

    parser.add_argument('--input', required=True, help='Path to extraction JSON file')
    parser.add_argument('--book', required=True, help='Book identifier (e.g., our_biggest_deal)')
    parser.add_argument('--section', required=True, help='Section identifier (e.g., front_matter, chapter_01)')
    parser.add_argument('--pages', required=True, help='Page range (e.g., 1-30, 31-50)')
    parser.add_argument('--version', default='v14_3_3', help='Version identifier (default: v14_3_3)')

    args = parser.parse_args()

    print("="*80)
    print("🤔 INCREMENTAL KG REFLECTOR - SECTION QUALITY ANALYSIS")
    print("="*80)
    print(f"  Book: {args.book}")
    print(f"  Section: {args.section}")
    print(f"  Pages: {args.pages}")
    print(f"  Input: {Path(args.input).name}")
    print("="*80)
    print()

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    BOOKS_DIR = BASE_DIR / "data" / "books"
    OUTPUT_DIR = BASE_DIR / "kg_extraction_playbook" / "output" / args.book / args.version
    ANALYSIS_DIR = OUTPUT_DIR / "analysis"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    # Find PDF
    book_dir = BOOKS_DIR / args.book
    pdf_files = list(book_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF found in {book_dir}")
        return

    pdf_path = pdf_files[0]

    # Load extraction results
    print(f"📂 Loading extraction results from {input_path.name}...")
    with open(input_path) as f:
        extraction_data = json.load(f)

    relationships = extraction_data.get('relationships', [])
    print(f"✅ Loaded {len(relationships)} relationships")
    print()

    # Extract source text for section
    source_text = extract_pages_from_pdf(pdf_path, args.pages)
    print()

    # Prepare extraction metadata
    extraction_metadata = {
        'version': args.version,
        'section': args.section,
        'pages': args.pages,
        'book': args.book,
        'date': extraction_data.get('metadata', {}).get('extraction_date', ''),
        'total_relationships': len(relationships)
    }

    # Check if there are previous analyses for comparison
    previous_reports = []
    previous_analyses = sorted(ANALYSIS_DIR.glob(f"{args.section}_reflection_*.json"))

    if previous_analyses:
        print(f"📊 Found {len(previous_analyses)} previous analysis/analyses for comparison...")
        # Load most recent previous analysis
        with open(previous_analyses[-1]) as f:
            prev_analysis = json.load(f)

        if 'quality_summary' in prev_analysis:
            prev_summary = prev_analysis['quality_summary']
            prev_grade = prev_summary.get('grade_confirmed', prev_summary.get('grade', 'N/A'))

            previous_reports.append({
                'title': f'{args.section} Previous Analysis',
                'summary': f"Previous: {prev_summary['total_issues']} issues ({prev_summary['issue_rate_percent']}%), Grade: {prev_grade}",
                'content_preview': json.dumps({'quality_summary': prev_summary}, indent=2)
            })

        print(f"✅ Loaded previous analysis for context: {previous_analyses[-1].name}")
        print()

    print("="*80)
    print("🤔 ANALYZING EXTRACTION QUALITY WITH CLAUDE SONNET 4.5...")
    print("="*80)
    print("This will take 1-2 minutes for comprehensive analysis...")
    print()

    # Initialize reflector
    reflector = KGReflectorAgent()

    # Run analysis
    analysis = reflector.analyze_kg_extraction(
        relationships=relationships,
        source_text=source_text,
        extraction_metadata=extraction_metadata,
        v4_quality_reports=previous_reports
    )

    print()
    print("="*80)
    print("✅ REFLECTOR ANALYSIS COMPLETE")
    print("="*80)
    print()

    # Generate machine-readable summary
    quality_summary = generate_quality_summary(
        analysis,
        args.section,
        input_path.name,
        len(relationships)
    )

    # Display console summary
    if 'quality_summary' in analysis:
        summary = analysis['quality_summary']
        print("QUALITY SUMMARY:")
        print(f"  Total issues: {summary.get('total_issues', 'N/A')}")
        print(f"  Issue rate: {summary.get('issue_rate_percent', 'N/A')}%")
        print(f"  Critical issues: {summary.get('critical_issues', 'N/A')}")
        print(f"  High priority: {summary.get('high_priority_issues', 'N/A')}")
        print(f"  Medium priority: {summary.get('medium_priority_issues', 'N/A')}")
        print(f"  Mild issues: {summary.get('mild_issues', 'N/A')}")
        print(f"  Grade: {summary.get('grade_confirmed', summary.get('grade', 'N/A'))}")
        print()

    # Display quality gate results
    print("="*80)
    print("🎯 QUALITY GATE CHECK (A+ CRITERIA)")
    print("="*80)
    gate = quality_summary['quality_gate']

    print(f"  ✓ Critical issues = 0: {gate['criteria']['critical_issues_zero']}")
    print(f"  ✓ High issues ≤ 2: {gate['criteria']['high_issues_lte_2']}")
    print(f"  ✓ Issue rate ≤ 2%: {gate['criteria']['issue_rate_lte_2_percent']}")
    print()

    if gate['passes_a_plus_gate']:
        print("  🎉 PASSES A+ QUALITY GATE!")
        print(f"  → Section '{args.section}' is ready to FREEZE")
        print()
    else:
        print("  ❌ DOES NOT PASS A+ QUALITY GATE")
        print(f"  → Section '{args.section}' needs iteration")
        print()
        print("  Failed criteria:")
        for criterion, passed in gate['criteria'].items():
            if not passed:
                print(f"    - {criterion}")
        print()

    # Display top actionable issues
    if quality_summary['actionable_issues']:
        print("TOP ACTIONABLE ISSUES:")
        for issue in quality_summary['actionable_issues'][:5]:
            print(f"  [{issue['severity']}] {issue['category']}: {issue['count']} ({issue['percentage']:.1f}%)")
        print()

    # Display top recommendations
    if quality_summary['recommendations']:
        print("TOP RECOMMENDATIONS:")
        for rec in quality_summary['recommendations'][:5]:
            print(f"  [{rec['priority']}] {rec['type']}: {rec['recommendation'][:80]}...")
        print()

    # Save full analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_path = ANALYSIS_DIR / f"{args.section}_reflection_{args.version}_{timestamp}.json"

    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"💾 Full analysis saved to: {analysis_path.name}")

    # Save machine-readable summary
    summary_path = ANALYSIS_DIR / f"{args.section}_summary_{timestamp}.json"

    with open(summary_path, 'w') as f:
        json.dump(quality_summary, f, indent=2)

    print(f"💾 Quality summary saved to: {summary_path.name}")
    print()

    # Final recommendation
    print("="*80)
    print("🔬 NEXT STEPS")
    print("="*80)

    if gate['passes_a_plus_gate']:
        print("RECOMMENDATION: FREEZE THIS SECTION ✅")
        print()
        print("Manual freeze command:")
        print(f"  1. Edit status.json")
        print(f"  2. Set sections.{args.section}.status = 'frozen'")
        print(f"  3. Set sections.{args.section}.grade = '{quality_summary['quality_metrics']['overall_grade']}'")
        print(f"  4. Set sections.{args.section}.issue_rate = {quality_summary['quality_metrics']['issue_rate']}")
        print()
        print("Or extraction script will auto-freeze if quality gate passes")
    else:
        print("RECOMMENDATION: ITERATE ON THIS SECTION 🔄")
        print()
        print("To improve quality:")
        print("  1. Review actionable issues above")
        print("  2. Analyze improvement recommendations")
        print("  3. Adjust prompts/modules as needed")
        print("  4. Re-run extraction:")
        print(f"     python3 scripts/extract_kg_v14_3_3_incremental.py \\")
        print(f"       --book {args.book} \\")
        print(f"       --section {args.section} \\")
        print(f"       --pages {args.pages}")
        print("  5. Re-run Reflector to verify improvement")

    print("="*80)


if __name__ == "__main__":
    main()
