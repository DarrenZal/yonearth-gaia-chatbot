#!/usr/bin/env python3
"""
Run V13.1 (Classification Without Discrimination) from V12 Pass 1 Checkpoint

This script runs the perfect A/B test:
- Loads V12's Pass 1 checkpoint (861 candidates)
- Runs Pass 2 evaluation with V13.1 non-discriminatory prompt (no penalties)
- Runs Pass 2.5 postprocessing with same modules as V12
- Compares results to measure penalty impact

V12 WITH PENALTIES:
  - PHILOSOPHICAL: subtract 0.4, cap at 0.3
  - NORMATIVE: subtract 0.3, cap at 0.4
  - Result: 873 relationships (81.2% high confidence)

V13.1 WITHOUT PENALTIES:
  - All claim types scored equally based on evidence
  - Classification via flags only (no score manipulation)
  - Result: ? relationships (? high confidence)
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_extraction_v13_1_from_v12_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import V13 script functions
import extract_kg_v13_book
from extract_kg_v13_book import (
    evaluate_pass2,
    postprocess_pass2_5,
    ExtractedRelationship,
    ModuleRelationship,
    ProcessingContext,
    OUTPUT_DIR,
    client,
    EvaluationBatchResult
)

# Load V13.1 prompt (with NORMATIVE flag restored)
PROMPTS_DIR = Path(__file__).parent.parent / "kg_extraction_playbook" / "prompts"
PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v13_1.txt"

if not PASS2_PROMPT_FILE.exists():
    raise FileNotFoundError(f"V13.1 prompt not found: {PASS2_PROMPT_FILE}")

with open(PASS2_PROMPT_FILE, 'r') as f:
    V13_1_PROMPT = f.read()

# Monkey-patch the V13.1 prompt into extract_kg_v13_book module
extract_kg_v13_book.DUAL_SIGNAL_EVALUATION_PROMPT = V13_1_PROMPT

logging.info(f"✅ Loaded V13.1 prompt with NORMATIVE flag from: {PASS2_PROMPT_FILE.name}")

def load_v12_pass1_checkpoint(checkpoint_path: Path):
    """Load V12's Pass 1 checkpoint"""
    logger.info(f"📂 Loading V12 Pass 1 checkpoint: {checkpoint_path.name}")

    with open(checkpoint_path, 'r') as f:
        data = json.load(f)

    # Convert to ExtractedRelationship objects
    candidates = []
    for item in data:
        candidates.append(ExtractedRelationship(
            source=item['source'],
            relationship=item['relationship'],
            target=item['target'],
            source_type=item['source_type'],
            target_type=item['target_type'],
            context=item['context'],
            page=item['page']
        ))

    logger.info(f"✅ Loaded {len(candidates)} candidates from V12 Pass 1")
    return candidates


def main():
    """Run V13.1 evaluation from V12 checkpoint"""
    logger.info("="*80)
    logger.info("🚀 V13.1 FROM V12 CHECKPOINT - CLASSIFICATION WITHOUT DISCRIMINATION")
    logger.info("="*80)
    logger.info("")
    logger.info("🧪 A/B TEST SETUP:")
    logger.info("  V12: Same Pass 1 + WITH penalties → 873 relationships (81.2% high conf)")
    logger.info("  V13.1: Same Pass 1 + NO penalties → ? relationships (? high conf)")
    logger.info("")
    logger.info("📋 V13.1 DIFFERENCES FROM V12:")
    logger.info("  ❌ REMOVED: claim_type penalty system (PHILOSOPHICAL: -0.4, NORMATIVE: -0.3)")
    logger.info("  ✅ KEPT: classification_flags multi-label classifier")
    logger.info("  ✅ KEPT: entity_specificity_score (concreteness)")
    logger.info("  ✅ KEPT: All 12 postprocessing modules")
    logger.info("")

    start_time = time.time()

    # Find V12 checkpoint
    checkpoint_path = OUTPUT_DIR.parent / "v12" / "book_soil_handbook_v12_20251014_044425_pass1_checkpoint.json"

    if not checkpoint_path.exists():
        logger.error(f"❌ V12 checkpoint not found: {checkpoint_path}")
        return

    # Load V12 Pass 1 candidates
    candidates = load_v12_pass1_checkpoint(checkpoint_path)

    # Run V13.1 Pass 2 (no penalties)
    logger.info("")
    logger.info("🔍 PASS 2: Dual-signal evaluation (V13.1 - NO PENALTIES)...")
    logger.info(f"  Evaluating {len(candidates)} candidates in batches of 25")

    evaluated = evaluate_pass2(candidates, batch_size=25)

    logger.info(f"✅ Pass 2 complete: {len(evaluated)} relationships evaluated")

    # Save Pass 2 checkpoint
    run_id = f"book_soil_handbook_v13_1_from_v12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_pass2 = OUTPUT_DIR / f"{run_id}_pass2_checkpoint.json"
    with open(checkpoint_pass2, 'w') as f:
        json.dump([rel.to_dict() for rel in evaluated], f, indent=2)
    logger.info(f"💾 Pass 2 checkpoint saved: {checkpoint_pass2.name}")

    # Run V13.1 Pass 2.5 (same modules as V12)
    logger.info("")
    logger.info("🔧 PASS 2.5: Modular postprocessing (same 12 modules as V12)...")

    document_metadata = {
        'author': 'Aaron Perry',
        'title': 'Soil Stewardship Handbook',
        'publication_year': 2017
    }

    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata,
        pages_with_text=[],  # Not needed for Pass 2.5
        run_id=run_id,
        extraction_version='v13'
    )

    final_relationships, pp_stats = postprocess_pass2_5(evaluated, context)

    logger.info(f"✅ Pass 2.5 complete: {len(final_relationships)} final relationships")

    # Calculate stats
    elapsed = time.time() - start_time

    high_conf = sum(1 for r in final_relationships if r.p_true >= 0.75)
    med_conf = sum(1 for r in final_relationships if 0.5 <= r.p_true < 0.75)
    low_conf = sum(1 for r in final_relationships if r.p_true < 0.5)

    # Count classification flags
    flag_counts = {}
    for rel in final_relationships:
        for flag in rel.classification_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    # Count module flags
    module_flag_counts = {}
    for rel in final_relationships:
        if rel.flags:
            for flag_key in rel.flags.keys():
                module_flag_counts[flag_key] = module_flag_counts.get(flag_key, 0) + 1

    # Prepare results
    results = {
        'metadata': {
            'book_title': 'Soil Stewardship Handbook',
            'extraction_version': 'v13',
            'run_id': run_id,
            'extraction_date': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'document_metadata': document_metadata,
            'source': 'V12 Pass 1 checkpoint (861 candidates)',
            'v12_checkpoint': str(checkpoint_path),
            'differences_from_v12': [
                'REMOVED: claim_type penalty system',
                'REMOVED: PHILOSOPHICAL penalty (-0.4, cap 0.3)',
                'REMOVED: NORMATIVE penalty (-0.3, cap 0.4)',
                'KEPT: classification_flags multi-label classifier',
                'KEPT: entity_specificity_score',
                'KEPT: All 12 postprocessing modules'
            ]
        },
        'extraction_stats': {
            'pass1_candidates': len(candidates),
            'pass1_source': 'V12 checkpoint',
            'pass2_evaluated': len(evaluated),
            'pass2_5_final': len(final_relationships),
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'classification_flags': flag_counts,
            'module_flags': module_flag_counts
        },
        'postprocessing_stats': pp_stats,
        'relationships': [rel.to_dict() for rel in final_relationships]
    }

    # Save results
    output_path = OUTPUT_DIR / f"soil_stewardship_handbook_v13_1_from_v12.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Display results
    logger.info("")
    logger.info("="*80)
    logger.info("📊 V13.1 FINAL RESULTS (NO PENALTIES)")
    logger.info("="*80)
    logger.info(f"  - Pass 1 (from V12): {len(candidates)} candidates")
    logger.info(f"  - Pass 2 evaluated: {len(evaluated)}")
    logger.info(f"  - ✨ V13.1 Pass 2.5 final: {len(final_relationships)}")
    logger.info(f"  - High confidence (p≥0.75): {high_conf} ({100*high_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Medium confidence: {med_conf} ({100*med_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Low confidence: {low_conf} ({100*low_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Total time: {elapsed/60:.1f} minutes")

    if flag_counts:
        logger.info(f"  - Classification flags:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            logger.info(f"      {flag}: {count}")

    logger.info("")
    logger.info("="*80)
    logger.info("🔬 V12 VS V13.1 COMPARISON")
    logger.info("="*80)
    logger.info("  V12 (WITH PENALTIES):")
    logger.info("    - Final: 873 relationships")
    logger.info("    - High confidence: 709 (81.2%)")
    logger.info("    - PHILOSOPHICAL flags: 16")
    logger.info("    - NORMATIVE flags: 16")
    logger.info("")
    logger.info(f"  V13.1 (NO PENALTIES):")
    logger.info(f"    - Final: {len(final_relationships)} relationships")
    logger.info(f"    - High confidence: {high_conf} ({100*high_conf/len(final_relationships) if final_relationships else 0:.1f}%)")

    philosophical = flag_counts.get('PHILOSOPHICAL_CLAIM', 0)
    normative = flag_counts.get('NORMATIVE', 0)
    logger.info(f"    - PHILOSOPHICAL flags: {philosophical}")
    logger.info(f"    - NORMATIVE flags: {normative}")
    logger.info("")

    # Calculate impact
    rel_diff = len(final_relationships) - 873
    conf_diff = high_conf - 709

    logger.info("  📈 PENALTY IMPACT:")
    logger.info(f"    - Relationship difference: {rel_diff:+d} ({rel_diff/873*100:+.1f}%)")
    logger.info(f"    - High confidence difference: {conf_diff:+d}")
    logger.info(f"    - Penalty effect: {'Suppressed valid claims' if rel_diff > 0 else 'Filtered noise'}")
    logger.info("")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Run Reflector on V13.1 to measure quality")
    logger.info("2. Compare V12 vs V13.1 Reflector grades")
    logger.info("3. Analyze which philosophical/normative claims were affected")
    logger.info("4. Decide: Keep penalties (V12) or classification only (V13.1)?")
    logger.info("="*80)


if __name__ == "__main__":
    main()
