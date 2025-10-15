#!/usr/bin/env python3
"""
Enhanced ACE Cycle with Validation Step

This demonstrates the improved ACE workflow:
1. Analyze (Reflector): Identify errors
2. Cure (Curator): Apply fixes
3. Validate (Validator): Test fixes on problem chunks (~2-5 min)
4. Evaluate (Full Extraction): Run full corpus (~52 min) ONLY if validation passes

Benefits:
- Fast feedback on whether fixes work
- Don't waste 52 minutes if fixes failed
- Iterative refinement until fixes validated
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ace_kg.kg_reflector import KGReflectorAgent
from src.ace_kg.kg_validator import KGValidatorAgent

# Import extraction pipeline
from scripts.extract_kg_v11_2_2_book import (
    extract_pass1,
    evaluate_pass2,
    postprocess_pass2_5,
    ModuleRelationship
)
from src.knowledge_graph.postprocessing import ProcessingContext


def run_validation_extraction(
    chunks,
    document_metadata,
    version="v11.2.2_validation"
):
    """
    Run extraction pipeline on a subset of chunks (for validation).

    This is the same pipeline as full extraction, but on fewer chunks.
    """
    print(f"\n🔬 Running validation extraction on {len(chunks)} chunks...")

    # Pass 1: Extract relationships
    all_candidates = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)} (pages {chunk['pages'][0]}-{chunk['pages'][-1]})")
        candidates, _ = extract_pass1(chunk)
        all_candidates.extend(candidates)

    print(f"✅ Pass 1: Extracted {len(all_candidates)} candidates")

    # Pass 2: Evaluation
    evaluated = evaluate_pass2(all_candidates, batch_size=25)
    print(f"✅ Pass 2: Evaluated {len(evaluated)} relationships")

    # Pass 2.5: Postprocessing
    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata,
        pages_with_text=[],
        run_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        extraction_version=version
    )

    final_relationships, _ = postprocess_pass2_5(evaluated, context)
    print(f"✅ Pass 2.5: {len(final_relationships)} final relationships")

    # Convert to dicts for validation
    return [rel.to_dict() for rel in final_relationships]


def enhanced_ace_cycle_with_validation():
    """
    Enhanced ACE cycle that validates fixes before full extraction.
    """
    print("="*80)
    print("🔄 ENHANCED ACE CYCLE WITH VALIDATION")
    print("="*80)
    print()

    BASE_DIR = Path(__file__).parent.parent
    PLAYBOOK_DIR = BASE_DIR / "kg_extraction_playbook"
    BOOKS_DIR = BASE_DIR / "data" / "books"

    book_dir = BOOKS_DIR / "soil-stewardship-handbook"
    pdf_path = book_dir / "Soil-Stewardship-Handbook-eBook.pdf"
    book_title = "Soil Stewardship Handbook"

    document_metadata = {
        'author': 'Aaron Perry',
        'title': 'Soil Stewardship Handbook',
        'publication_year': 2017
    }

    # ====================
    # STEP 1: ANALYZE (Reflector)
    # ====================
    print("📊 STEP 1: ANALYZE (Reflector)")
    print("-" * 80)
    print("Load previous extraction results and analyze quality...")
    print()

    # Load most recent Reflector analysis
    analysis_dir = PLAYBOOK_DIR / "analysis_reports"
    analysis_files = sorted(analysis_dir.glob("reflection_v11.2.1_*.json"))

    if not analysis_files:
        print("❌ No Reflector analysis found. Run Reflector first.")
        return

    analysis_path = analysis_files[-1]
    print(f"📁 Loading analysis: {analysis_path.name}")

    with open(analysis_path) as f:
        reflector_analysis = json.load(f)

    quality_summary = reflector_analysis.get('quality_summary', {})
    print(f"   Error rate: {quality_summary.get('issue_rate_percent', 0)}%")
    print(f"   Grade: {quality_summary.get('grade_confirmed', 'N/A')}")
    print()

    # ====================
    # STEP 2: CURE (Curator) - Manual for now
    # ====================
    print("🔧 STEP 2: CURE (Curator)")
    print("-" * 80)
    print("Curator applies fixes based on Reflector recommendations...")
    print()
    print("ℹ️  In this demo, we assume fixes were already applied (V11.2.2)")
    print("   Real ACE cycle would have automated Curator agent here.")
    print()

    # ====================
    # STEP 3: VALIDATE (NEW!)
    # ====================
    print("🧪 STEP 3: VALIDATE (Validator) - NEW!")
    print("-" * 80)
    print("Test fixes on problem chunks before full extraction...")
    print()

    validator = KGValidatorAgent(playbook_path=str(PLAYBOOK_DIR))

    # Define extraction function for validation
    def validation_extraction_func(chunks):
        return run_validation_extraction(chunks, document_metadata)

    # Run validation
    validation_result = validator.validate_fixes(
        reflector_analysis=reflector_analysis,
        pdf_path=pdf_path,
        extraction_function=validation_extraction_func,
        expected_error_reduction=0.5  # Expect 50% error reduction
    )

    # ====================
    # STEP 4: DECISION POINT
    # ====================
    print()
    print("="*80)
    print("🎯 DECISION POINT")
    print("="*80)

    if validation_result['validation_status'] == 'passed':
        print("✅ VALIDATION PASSED!")
        print(f"   Error reduction: {validation_result['error_reduction']*100:.1f}%")
        print(f"   Recommendation: {validation_result['recommendation']}")
        print()
        print("▶️  PROCEED TO STEP 4: Full corpus extraction (~52 minutes)")
        print()
        print("💡 To run full extraction:")
        print("   python3 scripts/extract_kg_v11_2_2_book.py")

    elif validation_result['validation_status'] == 'failed':
        print("❌ VALIDATION FAILED")
        print(f"   Error reduction: {validation_result['error_reduction']*100:.1f}%")
        print(f"   Target reduction: {validation_result['expected_error_reduction']*100:.1f}%")
        print()
        print("🔄 RETURN TO STEP 2: Refine fixes and validate again")
        print()
        print("💡 Next steps:")
        print("   1. Review validation report in kg_extraction_playbook/validation_reports/")
        print("   2. Refine fixes based on validation errors")
        print("   3. Run validation again until it passes")
        print("   4. Only then run full extraction")

    else:
        print("⚠️  VALIDATION SKIPPED")
        print(f"   Reason: {validation_result.get('reason', 'unknown')}")
        print()
        print("You may proceed with full extraction, but validation recommended.")

    # ====================
    # COMPARISON: Old vs New ACE Cycle
    # ====================
    print()
    print("="*80)
    print("📈 EFFICIENCY GAINS")
    print("="*80)
    print()
    print("OLD ACE CYCLE:")
    print("  Analyze → Cure → Evaluate (full 52 min) → Repeat if failed")
    print("  ❌ Total time per iteration: ~52 minutes")
    print("  ❌ Wasted time if fixes don't work: 52 minutes")
    print()
    print("NEW ACE CYCLE:")
    print("  Analyze → Cure → Validate (2-5 min) → Evaluate (only if passed)")
    print("  ✅ Total time if validation fails: ~5 minutes")
    print("  ✅ Time saved per failed iteration: ~47 minutes")
    print("  ✅ Can iterate 10x faster on fixes")
    print()


def main():
    """Main entry point."""
    try:
        enhanced_ace_cycle_with_validation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
