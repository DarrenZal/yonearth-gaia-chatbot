#!/usr/bin/env python3
"""
Build unified knowledge graph with strict validation.

Uses GraphBuilder with EntityMergeValidator to create a clean unified.json
with no catastrophic entity merges.

Usage:
    python scripts/build_unified_graph_v2.py

    # With custom parameters:
    python scripts/build_unified_graph_v2.py \
        --similarity-threshold 95 \
        --output data/knowledge_graph_unified/unified_v2.json
"""

import json
import logging
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.graph.graph_builder import GraphBuilder
from src.knowledge_graph.validators.entity_merge_validator import EntityMergeValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Build unified knowledge graph with validation')
    parser.add_argument(
        '--similarity-threshold',
        type=int,
        default=95,
        help='Fuzzy matching threshold (0-100). Default: 95'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/knowledge_graph_unified/unified_v2.json',
        help='Output path for unified JSON'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable entity merge validation (not recommended)'
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("UNIFIED KNOWLEDGE GRAPH BUILDER V2")
    logger.info("=" * 80)
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info(f"Validation enabled: {not args.no_validation}")
    logger.info(f"Output path: {args.output}")
    logger.info("")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    extraction_dir = base_dir / 'data' / 'knowledge_graph' / 'entities'
    output_path = base_dir / args.output

    # Check if extraction directory exists
    if not extraction_dir.exists() or not list(extraction_dir.glob('*_extraction.json')):
        logger.error(f"ERROR: No extraction files found in {extraction_dir}")
        logger.error("Run extract_knowledge_from_episodes.py first!")
        sys.exit(1)

    # Count extraction files
    episode_files = list(extraction_dir.glob('episode_*_extraction.json'))
    book_files = list(extraction_dir.glob('book_*_extraction.json'))
    logger.info(f"Found {len(episode_files)} episode extractions")
    logger.info(f"Found {len(book_files)} book extractions")
    logger.info("")

    # Initialize validator (if enabled)
    validator = None
    if not args.no_validation:
        logger.info("Initializing EntityMergeValidator...")
        validator = EntityMergeValidator(
            similarity_threshold=args.similarity_threshold,
            min_length_ratio=0.6,
            type_strict_matching=True,
            semantic_validation=True
        )
        logger.info("✓ Validator ready")
        logger.info("")

    # Initialize GraphBuilder (without Neo4j client for JSON-only export)
    logger.info("Initializing GraphBuilder...")
    builder = GraphBuilder(
        extraction_dir=str(extraction_dir),
        neo4j_client=None,  # No Neo4j needed for JSON export
        similarity_threshold=args.similarity_threshold,
        validator=validator,
        type_strict_matching=True
    )
    logger.info("✓ GraphBuilder ready")
    logger.info("")

    # Load extractions
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Extraction Files")
    logger.info("=" * 80)
    load_stats = builder.load_extractions()
    logger.info(f"✓ Loaded {load_stats['files_loaded']} files")
    logger.info(f"  - Raw entities: {load_stats['total_entities_raw']}")
    logger.info(f"  - Raw relationships: {load_stats['total_relationships_raw']}")
    logger.info(f"  - Unique entities (before dedup): {load_stats['unique_entities_before_dedup']}")
    logger.info("")

    # Deduplicate entities (with validation)
    logger.info("=" * 80)
    logger.info("STEP 2: Deduplicating Entities")
    logger.info("=" * 80)
    dedup_stats = builder.deduplicate_entities()
    logger.info(f"✓ Deduplication complete")
    logger.info(f"  - Unique entities (after dedup): {dedup_stats['entities_after_dedup']}")
    logger.info(f"  - Entities merged: {dedup_stats['entities_merged']}")
    reduction = (dedup_stats['entities_merged'] / load_stats['total_entities_raw']) * 100
    logger.info(f"  - Reduction: {reduction:.1f}%")
    logger.info("")

    # Deduplicate relationships
    logger.info("=" * 80)
    logger.info("STEP 3: Deduplicating Relationships")
    logger.info("=" * 80)
    rel_stats = builder.deduplicate_relationships()
    logger.info(f"✓ Relationship deduplication complete")
    logger.info(f"  - Unique relationships: {rel_stats['unique_relationships']}")
    logger.info("")

    # Calculate entity importance
    logger.info("=" * 80)
    logger.info("STEP 4: Calculating Entity Importance")
    logger.info("=" * 80)
    builder.calculate_entity_importance()
    logger.info("✓ Importance scores calculated")
    logger.info("")

    # Export to unified JSON
    logger.info("=" * 80)
    logger.info("STEP 5: Exporting to Unified JSON")
    logger.info("=" * 80)
    builder.export_unified_json(str(output_path))
    logger.info(f"✓ Exported to {output_path}")
    logger.info("")

    # Print summary statistics
    stats = builder.get_statistics()
    logger.info("=" * 80)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total entities: {stats['total_entities']}")
    logger.info(f"Total relationships: {stats['total_relationships']}")
    logger.info("")

    logger.info("Entity Type Distribution:")
    for entity_type, count in list(stats['entity_type_distribution'].items())[:10]:
        logger.info(f"  - {entity_type}: {count}")
    logger.info("")

    logger.info("Relationship Type Distribution:")
    for rel_type, count in list(stats['relationship_type_distribution'].items())[:10]:
        logger.info(f"  - {rel_type}: {count}")
    logger.info("")

    logger.info("Top Entities by Importance:")
    for i, entity in enumerate(stats['top_entities_by_importance'], 1):
        logger.info(
            f"  {i}. {entity['name']} ({entity['type']}) - "
            f"Score: {entity['importance_score']:.2f}"
        )
    logger.info("")

    # Save metadata
    metadata_file = output_path.parent / f"{output_path.stem}_metadata.json"
    metadata = {
        'build_date': datetime.now().isoformat(),
        'similarity_threshold': args.similarity_threshold,
        'validation_enabled': not args.no_validation,
        'statistics': {
            'files_processed': load_stats['files_loaded'],
            'total_entities': stats['total_entities'],
            'total_relationships': stats['total_relationships'],
            'entities_merged': dedup_stats['entities_merged'],
            'merge_reduction_pct': reduction
        }
    }

    if validator:
        metadata['validation_statistics'] = validator.get_statistics()

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to: {metadata_file}")
    logger.info("")

    logger.info("=" * 80)
    logger.info("BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nUnified graph: {output_path}")
    logger.info(f"Metadata: {metadata_file}")
    logger.info("\nNext steps:")
    logger.info("1. Run validate_unified_graph.py to test the new graph")
    logger.info("2. Compare with v1 using compare_graph_versions.py")
    logger.info("3. Deploy if tests pass")


if __name__ == '__main__':
    main()
