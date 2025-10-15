#!/usr/bin/env python3
"""
Chapter Consolidation Script for Incremental Knowledge Graph Extraction

🎯 PURPOSE: Merge completed chapter extractions and run cross-chapter postprocessing.

**Strategy**:
- Load multiple A+ grade chapter extractions
- Merge all relationships
- Run cross-chapter Deduplicator (remove exact duplicates)
- Run cross-chapter EntityResolver (canonicalize entity names)
- Optional: Run SemanticDeduplicator (expensive, embedding-based)
- Save consolidated output with statistics

**Architecture**:
- Phase 1: Load and merge chapter files
- Phase 2: Run postprocessing modules via PipelineOrchestrator
- Phase 3: Save consolidated results + statistics + alias maps

**Usage**:
```bash
# Consolidate Part I (chapters 1-13)
python3 scripts/consolidate_chapters.py \\
  --book our_biggest_deal \\
  --version v14_3_3 \\
  --part part_1 \\
  --chapters front_matter,chapter_01,chapter_02,...,chapter_13

# Consolidate all parts (final)
python3 scripts/consolidate_chapters.py \\
  --book our_biggest_deal \\
  --version v14_3_3 \\
  --final \\
  --parts part_1,part_2,part_3,part_4,part_5 \\
  --semantic-dedup
```

**Idempotency Guarantee**:
- Running consolidation twice on same inputs MUST produce identical outputs
- Deterministic tie-breaking in all modules
- Sorted outputs for stable results

**Provenance**:
- Saves consolidation statistics (dedup metrics, entity merges)
- Saves entity alias maps for inspection
- Timestamps all outputs for version tracking
"""

import json
import logging
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Postprocessing system
from src.knowledge_graph.postprocessing.base import PipelineOrchestrator, ProcessingContext
from src.knowledge_graph.postprocessing.universal import (
    FieldNormalizer, Deduplicator, EntityResolver, SemanticDeduplicator
)

# Setup logging
def setup_logging(consolidation_name: str):
    """Setup logging with consolidation-specific filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'kg_consolidation_{consolidation_name}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), timestamp


# ==============================================================================
# MODULE-COMPATIBLE RELATIONSHIP CLASS
# ==============================================================================

@dataclass
class ModuleRelationship:
    """
    Simplified relationship class for consolidation.

    Note: For consolidation, we only need core fields since chapters are already
    postprocessed to A+ grade. Full validation was done in chapter extraction.
    """
    source: str
    relationship: str  # Canonical field (FieldNormalizer ensures this)
    target: str
    source_type: str = 'UNKNOWN'
    target_type: str = 'UNKNOWN'
    context: str = ''
    page: int = 0
    text_confidence: float = 0.5
    p_true: float = 0.5
    signals_conflict: bool = False
    conflict_explanation: str = None
    suggested_correction: Dict[str, str] = None
    classification_flags: List[str] = None
    candidate_uid: str = ''
    entity_specificity_score: float = 1.0
    evidence: Dict[str, Any] = None
    evidence_text: str = ''
    flags: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.evidence is None:
            self.evidence = {'page_number': self.page}
        if self.evidence_text == '':
            self.evidence_text = self.context
        if self.flags is None:
            self.flags = {}
        if self.classification_flags is None:
            self.classification_flags = []
        if self.suggested_correction is None:
            self.suggested_correction = {}

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
            'entity_specificity_score': self.entity_specificity_score,
            'flags': self.flags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleRelationship':
        """Create from dict"""
        return cls(
            source=data.get('source', ''),
            relationship=data.get('relationship', data.get('predicate', '')),  # Handle legacy
            target=data.get('target', ''),
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
            entity_specificity_score=data.get('entity_specificity_score', 1.0),
            evidence=data.get('evidence', {}),
            evidence_text=data.get('evidence_text', data.get('context', '')),
            flags=data.get('flags', {})
        )


# ==============================================================================
# CONSOLIDATION LOGIC
# ==============================================================================

def load_chapter_files(
    book: str,
    version: str,
    identifiers: List[str],
    is_parts: bool = False,
    logger = None
) -> Tuple[List[ModuleRelationship], List[str]]:
    """
    Load multiple chapter or part files and merge relationships.

    Args:
        book: Book identifier (e.g., 'our_biggest_deal')
        version: Version identifier (e.g., 'v14_3_3')
        identifiers: List of chapter/part identifiers
        is_parts: If True, load from consolidations/ directory (parts), else chapters/
        logger: Logger instance

    Returns:
        (merged_relationships, loaded_files)
    """
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "kg_extraction_playbook" / "output" / book / version

    if is_parts:
        SOURCE_DIR = OUTPUT_DIR / "consolidations"
    else:
        SOURCE_DIR = OUTPUT_DIR / "chapters"

    if logger:
        logger.info(f"📂 Loading from: {SOURCE_DIR}")
        logger.info(f"   Identifiers: {', '.join(identifiers)}")

    all_relationships = []
    loaded_files = []

    for identifier in identifiers:
        # Find most recent file matching identifier
        pattern = f"{identifier}_{version}_*.json"
        matching_files = sorted(SOURCE_DIR.glob(pattern))

        if not matching_files:
            if logger:
                logger.warning(f"⚠️  No files found for {identifier} (pattern: {pattern})")
            continue

        # Use most recent (last in sorted list)
        file_path = matching_files[-1]

        if logger:
            logger.info(f"   Loading: {file_path.name}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract relationships (handle different JSON structures)
        if 'relationships' in data:
            rels = data['relationships']
        else:
            # Might be a list at top level
            rels = data if isinstance(data, list) else []

        # Convert dicts to ModuleRelationship objects
        for rel_data in rels:
            rel_obj = ModuleRelationship.from_dict(rel_data)
            all_relationships.append(rel_obj)

        loaded_files.append(file_path.name)

        if logger:
            logger.info(f"      → {len(rels)} relationships loaded")

    if logger:
        logger.info(f"✅ Loaded {len(all_relationships)} total relationships from {len(loaded_files)} files")

    return all_relationships, loaded_files


def consolidate_relationships(
    relationships: List[ModuleRelationship],
    document_metadata: Dict[str, Any],
    use_semantic_dedup: bool = False,
    logger = None
) -> Tuple[List[ModuleRelationship], Dict[str, Any], Dict[str, str]]:
    """
    Run cross-chapter consolidation pipeline.

    Pipeline:
    1. FieldNormalizer (priority 5) - Ensure consistent field names
    2. Deduplicator (priority 110) - Remove exact duplicates
    3. EntityResolver (priority 112) - Canonicalize entity names
    4. [Optional] SemanticDeduplicator (priority 115) - Remove semantic duplicates

    Args:
        relationships: List of ModuleRelationship objects from all chapters
        document_metadata: Metadata dict (known_entities, title, etc.)
        use_semantic_dedup: Whether to run expensive SemanticDeduplicator
        logger: Logger instance

    Returns:
        (consolidated_relationships, pipeline_stats, entity_alias_map)
    """
    if logger:
        logger.info("🔧 Running cross-chapter consolidation pipeline...")
        logger.info(f"  Input relationships: {len(relationships)}")
        logger.info(f"  Semantic dedup: {'ENABLED' if use_semantic_dedup else 'DISABLED'}")

    # Create processing context
    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata
    )

    # Build consolidation pipeline
    modules = [
        FieldNormalizer(),       # Priority 5 - normalize field names
        Deduplicator(),          # Priority 110 - remove exact duplicates
        EntityResolver(),        # Priority 112 - resolve entity variations
    ]

    if use_semantic_dedup:
        if logger:
            logger.info("   ✨ Adding SemanticDeduplicator (expensive, uses embeddings)")

        modules.append(
            SemanticDeduplicator(config={
                'similarity_threshold': 0.85,
                'model_name': 'all-MiniLM-L6-v2'
            })
        )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(modules, config={
        'halt_on_error': False  # Continue even if a module fails
    })

    # Run pipeline
    start_time = time.time()
    consolidated, pipeline_stats = orchestrator.run(relationships, context)
    elapsed = time.time() - start_time

    # Extract entity alias map from EntityResolver
    entity_alias_map = {}
    for module in modules:
        if hasattr(module, 'alias_map'):
            entity_alias_map = module.alias_map
            break

    if logger:
        logger.info(f"✅ Consolidation complete: {len(relationships)} → {len(consolidated)} relationships")
        logger.info(f"   Reduction: {len(relationships) - len(consolidated)} duplicates removed ({100*(len(relationships) - len(consolidated))/len(relationships):.1f}%)")
        logger.info(f"   Entity aliases: {len(entity_alias_map)} mappings created")
        logger.info(f"   Time: {elapsed:.1f}s")

    return consolidated, pipeline_stats, entity_alias_map


def save_consolidation_results(
    book: str,
    version: str,
    consolidation_name: str,
    consolidated: List[ModuleRelationship],
    pipeline_stats: Dict[str, Any],
    entity_alias_map: Dict[str, str],
    loaded_files: List[str],
    timestamp: str,
    logger = None
) -> Tuple[Path, Path, Path]:
    """
    Save consolidated results with statistics and alias maps.

    Args:
        book: Book identifier
        version: Version identifier
        consolidation_name: Name of consolidation (e.g., 'part_1', 'final')
        consolidated: Consolidated relationships
        pipeline_stats: Statistics from pipeline
        entity_alias_map: Entity alias mappings
        loaded_files: List of source files
        timestamp: Timestamp string
        logger: Logger instance

    Returns:
        (consolidated_path, stats_path, aliases_path)
    """
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "kg_extraction_playbook" / "output" / book / version
    CONSOLIDATIONS_DIR = OUTPUT_DIR / "consolidations"
    CONSOLIDATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Save consolidated relationships
    consolidated_path = CONSOLIDATIONS_DIR / f"{consolidation_name}_consolidated_{version}_{timestamp}.json"

    consolidated_data = {
        'metadata': {
            'book': book,
            'version': version,
            'consolidation': consolidation_name,
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'source_files': loaded_files
        },
        'relationships': [rel.to_dict() for rel in consolidated]
    }

    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    if logger:
        logger.info(f"💾 Saved consolidated relationships: {consolidated_path.name}")

    # Save statistics
    stats_path = CONSOLIDATIONS_DIR / f"{consolidation_name}_stats_{timestamp}.json"

    # Build detailed stats
    stats_data = {
        'consolidation': consolidation_name,
        'timestamp': timestamp,
        'version': version,
        'input_files': loaded_files,
        'processing_steps': [],
        'summary': {
            'total_input_relationships': 0,
            'total_output_relationships': len(consolidated),
            'reduction_count': 0,
            'reduction_percentage': 0.0,
            'processing_time_seconds': 0
        }
    }

    # Extract module stats from pipeline_stats
    if 'modules' in pipeline_stats:
        for module_stats in pipeline_stats['modules']:
            step_stats = {
                'module': module_stats.get('module', 'Unknown'),
                'priority': module_stats.get('priority', 0),
                'before_count': module_stats.get('input_count', 0),
                'after_count': module_stats.get('output_count', 0),
                'changes': module_stats.get('stats', {})
            }

            # First module gives us input count
            if not stats_data['processing_steps']:
                stats_data['summary']['total_input_relationships'] = step_stats['before_count']

            stats_data['processing_steps'].append(step_stats)

    # Calculate summary
    stats_data['summary']['reduction_count'] = (
        stats_data['summary']['total_input_relationships'] -
        stats_data['summary']['total_output_relationships']
    )

    if stats_data['summary']['total_input_relationships'] > 0:
        stats_data['summary']['reduction_percentage'] = round(
            100 * stats_data['summary']['reduction_count'] /
            stats_data['summary']['total_input_relationships'],
            1
        )

    stats_data['output_file'] = consolidated_path.name

    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)

    if logger:
        logger.info(f"💾 Saved consolidation stats: {stats_path.name}")

    # Save entity alias map
    aliases_path = CONSOLIDATIONS_DIR / f"{consolidation_name}_entity_aliases_{timestamp}.json"

    with open(aliases_path, 'w') as f:
        json.dump(entity_alias_map, f, indent=2, sort_keys=True)  # sort_keys for determinism

    if logger:
        logger.info(f"💾 Saved entity aliases: {aliases_path.name} ({len(entity_alias_map)} mappings)")

    return consolidated_path, stats_path, aliases_path


# ==============================================================================
# MAIN CONSOLIDATION
# ==============================================================================

def consolidate(args: argparse.Namespace):
    """
    Consolidate chapter or part extractions.

    Workflow:
    1. Load source files (chapters or parts)
    2. Merge relationships
    3. Run consolidation pipeline (FieldNormalizer → Deduplicator → EntityResolver)
    4. Save consolidated output + statistics + alias maps
    """
    # Setup logging
    consolidation_name = 'final' if args.final else args.part
    logger, timestamp = setup_logging(consolidation_name)

    logger.info("="*80)
    logger.info("🔄 CHAPTER/PART CONSOLIDATION")
    logger.info("="*80)
    logger.info(f"  Book: {args.book}")
    logger.info(f"  Version: {args.version}")
    logger.info(f"  Consolidation: {consolidation_name}")

    if args.final:
        logger.info(f"  Mode: FINAL (merging parts)")
        logger.info(f"  Parts: {args.parts}")
    else:
        logger.info(f"  Mode: PART (merging chapters)")
        logger.info(f"  Chapters: {args.chapters}")

    logger.info(f"  Semantic dedup: {'ENABLED' if args.semantic_dedup else 'DISABLED'}")
    logger.info("="*80)
    logger.info("")

    # Prepare identifiers
    if args.final:
        identifiers = [p.strip() for p in args.parts.split(',')]
        is_parts = True
    else:
        identifiers = [c.strip() for c in args.chapters.split(',')]
        is_parts = False

    # Load source files
    logger.info("📂 Loading source files...")
    relationships, loaded_files = load_chapter_files(
        args.book, args.version, identifiers, is_parts, logger
    )
    logger.info("")

    if not relationships:
        logger.error("❌ No relationships loaded. Exiting.")
        return

    # Prepare document metadata
    document_metadata = {
        'title': args.book.replace('_', ' ').title(),
        'consolidation': consolidation_name,
        'known_entities': args.known_entities.split(',') if args.known_entities else []
    }

    # Run consolidation
    logger.info("🔧 Running consolidation pipeline...")
    consolidated, pipeline_stats, entity_alias_map = consolidate_relationships(
        relationships,
        document_metadata,
        args.semantic_dedup,
        logger
    )
    logger.info("")

    # Save results
    logger.info("💾 Saving consolidated results...")
    consolidated_path, stats_path, aliases_path = save_consolidation_results(
        args.book,
        args.version,
        consolidation_name,
        consolidated,
        pipeline_stats,
        entity_alias_map,
        loaded_files,
        timestamp,
        logger
    )
    logger.info("")

    # Summary
    logger.info("="*80)
    logger.info("✅ CONSOLIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"  Input: {len(relationships)} relationships from {len(loaded_files)} files")
    logger.info(f"  Output: {len(consolidated)} relationships")
    logger.info(f"  Reduction: {len(relationships) - len(consolidated)} duplicates ({100*(len(relationships) - len(consolidated))/len(relationships):.1f}%)")
    logger.info(f"  Entity aliases: {len(entity_alias_map)} mappings")
    logger.info("")
    logger.info(f"📁 Output files:")
    logger.info(f"  - Consolidated: {consolidated_path.name}")
    logger.info(f"  - Statistics: {stats_path.name}")
    logger.info(f"  - Aliases: {aliases_path.name}")
    logger.info("="*80)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate chapter or part extractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Consolidate Part I (chapters 1-13)
  python3 scripts/consolidate_chapters.py \\
    --book our_biggest_deal \\
    --version v14_3_3 \\
    --part part_1 \\
    --chapters front_matter,chapter_01,chapter_02,...,chapter_13

  # Final consolidation (all parts)
  python3 scripts/consolidate_chapters.py \\
    --book our_biggest_deal \\
    --version v14_3_3 \\
    --final \\
    --parts part_1,part_2,part_3,part_4,part_5 \\
    --semantic-dedup \\
    --known-entities "Aaron William Perry,John Perkins"
        """
    )

    parser.add_argument('--book', required=True, help='Book identifier')
    parser.add_argument('--version', default='v14_3_3', help='Version identifier')

    # Consolidation mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--part', help='Part consolidation name (e.g., part_1)')
    mode_group.add_argument('--final', action='store_true', help='Final consolidation (merge all parts)')

    # Sources
    parser.add_argument('--chapters', help='Comma-separated chapter identifiers (for --part)')
    parser.add_argument('--parts', help='Comma-separated part identifiers (for --final)')

    # Options
    parser.add_argument('--semantic-dedup', action='store_true', help='Run SemanticDeduplicator (expensive)')
    parser.add_argument('--known-entities', default='', help='Comma-separated known entity names (for EntityResolver allowlist)')

    args = parser.parse_args()

    # Validate arguments
    if args.part and not args.chapters:
        parser.error("--part requires --chapters")
    if args.final and not args.parts:
        parser.error("--final requires --parts")

    consolidate(args)


if __name__ == "__main__":
    main()
