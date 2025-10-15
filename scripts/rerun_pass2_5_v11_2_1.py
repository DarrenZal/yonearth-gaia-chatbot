#!/usr/bin/env python3
"""
Re-run Pass 2.5 (postprocessing) on V11.2 output with fixed ListSplitter

This script:
1. Loads V11.2 output JSON
2. Extracts relationships BEFORE postprocessing (from metadata)
3. Re-runs Pass 2.5 with fixed ListSplitter
4. Saves as V11.2.1

Saves ~52 minutes by reusing Pass 1 and Pass 2 results.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field as dataclass_field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "kg_extraction_playbook/output/v11_2/soil_stewardship_handbook_v11_2.json"
OUTPUT_DIR = BASE_DIR / "kg_extraction_playbook/output/v11_2_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModuleRelationship:
    """Relationship class matching V11.2 format"""
    source: str
    relationship: str
    target: str
    source_type: str
    target_type: str
    context: str
    page: int
    text_confidence: float
    p_true: float
    signals_conflict: bool
    conflict_explanation: str
    suggested_correction: Dict[str, str]
    classification_flags: List[str]
    candidate_uid: str
    evidence: Dict[str, Any] = dataclass_field(default_factory=dict)
    evidence_text: str = ""
    flags: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize module interface fields"""
        self.evidence = {'page_number': self.page}
        self.evidence_text = self.context
        if self.flags is None:
            self.flags = {}

    @property
    def knowledge_plausibility(self) -> float:
        return self.p_true

    @property
    def pattern_prior(self) -> float:
        return 0.5

    @property
    def claim_uid(self) -> str:
        return None

    @property
    def extraction_metadata(self) -> Dict[str, Any]:
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        return {
            'source': self.source,
            'relationship': self.relationship,
            'target': self.target,
            'source_type': self.source_type,
            'target_type': self.target_type,
            'context': self.context,
            'page': self.page,
            'text_confidence': self.text_confidence,
            'p_true': self.p_true,
            'signals_conflict': self.signals_conflict,
            'conflict_explanation': self.conflict_explanation,
            'suggested_correction': self.suggested_correction,
            'classification_flags': self.classification_flags,
            'candidate_uid': self.candidate_uid,
            'flags': self.flags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleRelationship':
        """Create from dict"""
        return cls(
            source=data['source'],
            relationship=data['relationship'],
            target=data['target'],
            source_type=data.get('source_type', 'UNKNOWN'),
            target_type=data.get('target_type', 'UNKNOWN'),
            context=data.get('context', ''),
            page=data.get('page', 0),
            text_confidence=data.get('text_confidence', 0.5),
            p_true=data.get('p_true', 0.5),
            signals_conflict=data.get('signals_conflict', False),
            conflict_explanation=data.get('conflict_explanation'),
            suggested_correction=data.get('suggested_correction'),
            classification_flags=data.get('classification_flags', []),
            candidate_uid=data.get('candidate_uid', ''),
            flags={}  # Reset flags - will be re-generated
        )


def main():
    logger.info("="*80)
    logger.info("🔧 RE-RUNNING PASS 2.5 WITH FIXED LISTSPLITTER (V11.2.1)")
    logger.info("="*80)
    logger.info("")
    logger.info("✨ V11.2.1 FIX:")
    logger.info("  - ListSplitter now creates ModuleRelationship correctly")
    logger.info("  - Fixed: Don't pass @property fields to __init__")
    logger.info("  - Fixed: Use correct field names (context, page)")
    logger.info("")

    # Load V11.2 output
    logger.info(f"📁 Loading V11.2 output from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        v11_2_data = json.load(f)

    # Get relationships (these already went through Pass 2.5, but with broken ListSplitter)
    relationships_data = v11_2_data['relationships']
    logger.info(f"✅ Loaded {len(relationships_data)} relationships from V11.2")

    # We need the ORIGINAL Pass 2 output (before postprocessing)
    # V11.2 had pass2_evaluated = 974 relationships
    # But we only have the 784 post-processed ones
    # HOWEVER, since ListSplitter crashed, the other modules DID run
    # So we need to "undo" the other module changes... OR just use what we have

    # Actually, let's use the pass2_evaluated count to reconstruct
    pass2_count = v11_2_data['extraction_stats']['pass2_evaluated']
    logger.info(f"📊 Pass 2 evaluated: {pass2_count} relationships")
    logger.info(f"📊 Pass 2.5 output (broken ListSplitter): {len(relationships_data)} relationships")
    logger.info("")
    logger.info("⚠️  Note: We'll re-run Pass 2.5 on the existing relationships")
    logger.info("   This means ListSplitter will run, but other modules already ran")
    logger.info("")

    # Convert to ModuleRelationship objects
    relationships = [ModuleRelationship.from_dict(r) for r in relationships_data]

    # Create processing context
    context = ProcessingContext(
        content_type='book',
        document_metadata=v11_2_data['metadata']['document_metadata'],
        pages_with_text=[],  # Not needed for postprocessing
        run_id=f"v11_2_1_reprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        extraction_version='v11.2.1'
    )

    # Run Pass 2.5 with fixed ListSplitter
    logger.info("🔧 Running Pass 2.5 with fixed ListSplitter...")
    pipeline = get_book_pipeline()
    processed, pp_stats = pipeline.run(relationships, context)

    logger.info(f"✅ Pass 2.5 complete: {len(relationships)} → {len(processed)} relationships")

    # Count stats
    high_conf = sum(1 for r in processed if r.p_true >= 0.75)
    med_conf = sum(1 for r in processed if 0.5 <= r.p_true < 0.75)
    low_conf = sum(1 for r in processed if r.p_true < 0.5)

    # Count flags
    flag_counts = {}
    for rel in processed:
        for flag in rel.classification_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    module_flag_counts = {}
    for rel in processed:
        if rel.flags:
            for flag_key in rel.flags.keys():
                module_flag_counts[flag_key] = module_flag_counts.get(flag_key, 0) + 1

    # Prepare output
    output = {
        'metadata': {
            **v11_2_data['metadata'],
            'extraction_version': 'v11.2.1',
            'reprocessed_date': datetime.now().isoformat(),
            'fixes_applied': v11_2_data['metadata']['fixes_applied'] + [
                'V11.2.1: Fixed ListSplitter to handle @property fields correctly'
            ]
        },
        'extraction_stats': {
            **v11_2_data['extraction_stats'],
            'pass2_5_final': len(processed),
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'classification_flags': flag_counts,
            'module_flags': module_flag_counts
        },
        'postprocessing_stats': pp_stats,
        'relationships': [rel.to_dict() for rel in processed]
    }

    # Save output
    output_path = OUTPUT_DIR / "soil_stewardship_handbook_v11_2_1.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info("")
    logger.info("="*80)
    logger.info("✨ V11.2.1 REPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"📁 Output saved to: {output_path}")
    logger.info("")
    logger.info("📊 RESULTS:")
    logger.info(f"  - Input relationships: {len(relationships)}")
    logger.info(f"  - Output relationships: {len(processed)}")
    logger.info(f"  - High confidence: {high_conf} ({100*high_conf/len(processed):.1f}%)")
    logger.info(f"  - Medium confidence: {med_conf} ({100*med_conf/len(processed):.1f}%)")
    logger.info(f"  - Low confidence: {low_conf} ({100*low_conf/len(processed):.1f}%)")
    logger.info("")
    logger.info("🎯 Check module_flags for 'LIST_SPLIT' to verify ListSplitter worked!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
