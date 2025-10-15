#!/usr/bin/env python3
"""
Reprocess V11.1 Output with ListSplitter Module

This script loads the existing V11.1 extraction output and runs just the ListSplitter
module to fix the missing `knowledge_plausibility` attribute error.

Usage:
    python3 scripts/reprocess_v11_1_list_splitter.py

Output:
    Updates the V11.1 output file in place (creates backup first)
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from shutil import copy2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.universal.list_splitter import ListSplitter as OriginalListSplitter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "kg_extraction_playbook" / "output" / "v11_1"
INPUT_FILE = OUTPUT_DIR / "soil_stewardship_handbook_v11_1.json"
BACKUP_FILE = OUTPUT_DIR / "soil_stewardship_handbook_v11_1_backup.json"


# ==============================================================================
# FIXED MODULE RELATIONSHIP CLASS
# ==============================================================================

@dataclass
class ModuleRelationship:
    """
    ✨ V11.1.1 FIXED: Relationship format compatible with ALL postprocessing modules.

    Now includes all attributes expected by ListSplitter:
    - knowledge_plausibility (alias for p_true)
    - pattern_prior, claim_uid, extraction_metadata (defaults)
    """
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
    conflict_explanation: str = None
    suggested_correction: Dict[str, str] = None
    classification_flags: List[str] = dataclass_field(default_factory=list)
    candidate_uid: str = ""

    # Module interface fields
    evidence: Dict[str, Any] = dataclass_field(default_factory=dict)
    evidence_text: str = ""
    flags: Dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Initialize module interface fields from existing data"""
        if not self.evidence:
            self.evidence = {'page_number': self.page}
        if not self.evidence_text:
            self.evidence_text = self.context
        if self.flags is None:
            self.flags = {}

    # ✨ V11.1.1 FIX: Add properties for ListSplitter compatibility
    @property
    def knowledge_plausibility(self) -> float:
        """Alias for p_true (ListSplitter expects this name)"""
        return self.p_true

    @property
    def pattern_prior(self) -> float:
        """Default pattern prior (not used but ListSplitter expects it)"""
        return 0.5

    @property
    def claim_uid(self):
        """Default claim_uid (not used but ListSplitter expects it)"""
        return None

    @property
    def extraction_metadata(self) -> Dict[str, Any]:
        """Default extraction metadata (not used but ListSplitter expects it)"""
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
        """
        Create from dict.

        Filters out property-only keys that ListSplitter might add
        (knowledge_plausibility, pattern_prior, claim_uid, extraction_metadata)
        """
        # Filter out property-only keys before creating object
        clean_data = {k: v for k, v in data.items()
                     if k not in ['knowledge_plausibility', 'pattern_prior', 'claim_uid', 'extraction_metadata']}

        return cls(
            source=clean_data['source'],
            relationship=clean_data['relationship'],
            target=clean_data['target'],
            source_type=clean_data.get('source_type', 'UNKNOWN'),
            target_type=clean_data.get('target_type', 'UNKNOWN'),
            context=clean_data.get('context', ''),
            page=clean_data.get('page', 0),
            text_confidence=clean_data.get('text_confidence', 0.5),
            p_true=clean_data.get('p_true', clean_data.get('knowledge_plausibility', 0.5)),
            signals_conflict=clean_data.get('signals_conflict', False),
            conflict_explanation=clean_data.get('conflict_explanation'),
            suggested_correction=clean_data.get('suggested_correction'),
            classification_flags=clean_data.get('classification_flags', []),
            candidate_uid=clean_data.get('candidate_uid', ''),
            evidence=clean_data.get('evidence', {}),
            evidence_text=clean_data.get('evidence_text', clean_data.get('context', '')),
            flags=clean_data.get('flags', {})
        )


# ==============================================================================
# FIXED LIST SPLITTER
# ==============================================================================

class ListSplitter(OriginalListSplitter):
    """
    ✨ V11.1.1 FIX: ListSplitter that filters property-only keys before creating new relationships.

    Overrides split_relationship to filter out knowledge_plausibility, pattern_prior, etc.
    before calling ModuleRelationship.__init__()
    """

    def split_relationship(self, rel):
        """Split a single relationship with list target into multiple (FIXED VERSION)"""
        items = self.split_target_list(rel.target)

        if len(items) <= 1:
            return [rel]

        split_rels = []
        for i, item in enumerate(items):
            # Create new relationship dict
            new_rel_dict = {
                'source': rel.source,
                'relationship': rel.relationship,
                'target': item,
                'source_type': rel.source_type,
                'target_type': rel.target_type,
                'evidence_text': rel.evidence_text,
                'evidence': rel.evidence.copy() if hasattr(rel.evidence, 'copy') else dict(rel.evidence),
                'text_confidence': rel.text_confidence,
                # ✨ V11.1.1 FIX: Use p_true instead of knowledge_plausibility
                'p_true': rel.p_true,
                'signals_conflict': rel.signals_conflict,
                'conflict_explanation': rel.conflict_explanation,
                'candidate_uid': rel.candidate_uid + f"_split_{i}",
                'flags': rel.flags.copy() if hasattr(rel.flags, 'copy') and rel.flags else {},
            }

            # ✨ V11.1.1 FIX: Use from_dict which filters property-only keys
            new_rel = ModuleRelationship.from_dict(new_rel_dict)

            # Update flags
            if new_rel.flags is None:
                new_rel.flags = {}
            new_rel.flags['LIST_SPLIT'] = True
            new_rel.flags['split_index'] = i
            new_rel.flags['split_total'] = len(items)
            new_rel.flags['original_target'] = rel.target

            split_rels.append(new_rel)

        return split_rels


# ==============================================================================
# REPROCESSING LOGIC
# ==============================================================================

def reprocess_with_list_splitter(input_path: Path, backup_path: Path) -> Dict[str, Any]:
    """
    Reprocess V11.1 output with ListSplitter module.

    Args:
        input_path: Path to V11.1 JSON output
        backup_path: Path to save backup before modification

    Returns:
        Updated extraction results dict
    """
    logger.info("="*80)
    logger.info("🔧 REPROCESSING V11.1 WITH LIST SPLITTER")
    logger.info("="*80)
    logger.info("")

    # Step 1: Load existing V11.1 output
    logger.info(f"📂 Loading V11.1 output from: {input_path}")
    with open(input_path, 'r') as f:
        results = json.load(f)

    original_count = len(results['relationships'])
    logger.info(f"✅ Loaded {original_count} relationships")

    # Step 2: Create backup
    logger.info(f"💾 Creating backup at: {backup_path}")
    copy2(input_path, backup_path)
    logger.info("✅ Backup created")

    # Step 3: Convert dicts → ModuleRelationship objects
    logger.info("🔄 Converting relationships to ModuleRelationship objects...")
    relationships = [ModuleRelationship.from_dict(rel) for rel in results['relationships']]
    logger.info(f"✅ Converted {len(relationships)} relationships")

    # Step 4: Run ListSplitter module
    logger.info("")
    logger.info("🔀 Running ListSplitter module...")

    # Create processing context
    context = ProcessingContext(
        content_type='book',
        document_metadata=results['metadata'].get('document_metadata', {}),
        pages_with_text=None,  # Not needed for ListSplitter
        run_id=results['metadata'].get('run_id', 'reprocess'),
        extraction_version='v11.1.1'
    )

    # Initialize and run ListSplitter
    list_splitter = ListSplitter()
    processed_relationships = list_splitter.process_batch(relationships, context)

    final_count = len(processed_relationships)
    split_diff = final_count - original_count

    logger.info("")
    logger.info(f"✅ ListSplitter complete: {original_count} → {final_count} relationships ({split_diff:+d})")

    # Step 5: Update results
    logger.info("📝 Updating results metadata...")

    # Convert back to dicts
    results['relationships'] = [rel.to_dict() for rel in processed_relationships]

    # Update stats
    results['extraction_stats']['pass2_5_final'] = final_count
    results['extraction_stats']['list_splitter_reprocess'] = {
        'original_count': original_count,
        'final_count': final_count,
        'relationships_added': split_diff,
        'reprocessed_at': datetime.now().isoformat()
    }

    # Count module flags
    module_flag_counts = {}
    for rel in processed_relationships:
        if rel.flags:
            for flag_key in rel.flags.keys():
                module_flag_counts[flag_key] = module_flag_counts.get(flag_key, 0) + 1

    results['extraction_stats']['module_flags'] = module_flag_counts

    # Add reprocessing note to metadata
    if 'fixes_applied' not in results['metadata']:
        results['metadata']['fixes_applied'] = []
    results['metadata']['fixes_applied'].append('ListSplitter reprocessing (v11.1.1 fix)')
    results['metadata']['extraction_version'] = 'v11.1.1'

    # Update postprocessing stats
    if 'postprocessing_stats' in results:
        results['postprocessing_stats']['list_splitter'] = list_splitter.get_summary()

    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main reprocessing function"""

    # Check input file exists
    if not INPUT_FILE.exists():
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        logger.error("Please ensure V11.1 extraction has completed successfully.")
        return 1

    # Run reprocessing
    try:
        updated_results = reprocess_with_list_splitter(INPUT_FILE, BACKUP_FILE)

        # Save updated results
        logger.info("")
        logger.info(f"💾 Saving updated results to: {INPUT_FILE}")
        with open(INPUT_FILE, 'w') as f:
            json.dump(updated_results, f, indent=2)

        logger.info("✅ Results saved successfully")

        # Print summary
        logger.info("")
        logger.info("="*80)
        logger.info("✨ REPROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Original file: {INPUT_FILE}")
        logger.info(f"Backup saved: {BACKUP_FILE}")
        logger.info(f"Original relationships: {updated_results['extraction_stats']['list_splitter_reprocess']['original_count']}")
        logger.info(f"Final relationships: {updated_results['extraction_stats']['pass2_5_final']}")
        logger.info(f"Relationships added: {updated_results['extraction_stats']['list_splitter_reprocess']['relationships_added']:+d}")
        logger.info("")

        # Show module flag counts
        if 'module_flags' in updated_results['extraction_stats']:
            logger.info("Module flags (updated):")
            for flag, count in sorted(updated_results['extraction_stats']['module_flags'].items(), key=lambda x: -x[1])[:10]:
                logger.info(f"  {flag}: {count}")

        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"❌ Reprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
