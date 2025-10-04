#!/usr/bin/env python3
"""
Unified Knowledge Graph Builder Script

Builds both Neo4j graph and Obsidian wiki from synchronized data sources.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.unified_builder import UnifiedBuilder

def setup_logging():
    """Configure logging for the build process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / 'unified_build.log')
        ]
    )

def main():
    """Main build script."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("UNIFIED KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 80)

    # Paths
    extraction_dir = project_root / 'data' / 'knowledge_graph' / 'entities'
    transcripts_dir = project_root / 'data' / 'transcripts'
    wiki_output_dir = project_root / 'web' / 'wiki'

    logger.info(f"Extraction directory: {extraction_dir}")
    logger.info(f"Transcripts directory: {transcripts_dir}")
    logger.info(f"Wiki output directory: {wiki_output_dir}")

    # Check directories exist
    if not extraction_dir.exists():
        logger.error(f"Extraction directory not found: {extraction_dir}")
        sys.exit(1)

    if not transcripts_dir.exists():
        logger.error(f"Transcripts directory not found: {transcripts_dir}")
        sys.exit(1)

    # Neo4j credentials (optional - graph building will be skipped if not provided)
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        logger.warning("Neo4j credentials not provided - graph building will be skipped")
        logger.warning("Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD to enable graph building")

    # Build
    try:
        builder = UnifiedBuilder(
            extraction_dir=extraction_dir,
            transcripts_dir=transcripts_dir,
            wiki_output_dir=wiki_output_dir,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )

        stats = builder.build_all()

        # Save statistics
        import json
        stats_file = project_root / 'data' / 'knowledge_graph' / 'build_statistics.json'
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            return obj

        stats_json = convert_sets(stats)

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_json, f, indent=2)

        logger.info("=" * 80)
        logger.info("BUILD COMPLETE!")
        logger.info(f"Statistics saved to: {stats_file}")
        logger.info("=" * 80)

        # Print summary
        print("\n" + "=" * 80)
        print("BUILD SUMMARY")
        print("=" * 80)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Total Extractions: {stats['total_extractions']}")
        print(f"Synchronized: {stats['synchronized']}")
        print(f"\nWiki Statistics:")
        print(f"  - Entities: {stats['wiki'].get('total_entities', 0)}")
        print(f"  - Episodes: {stats['wiki'].get('total_episodes', 0)}")
        print(f"  - People: {stats['wiki'].get('people_count', 0)}")
        print(f"  - Organizations: {stats['wiki'].get('organizations_count', 0)}")
        print(f"  - Concepts: {stats['wiki'].get('concepts_count', 0)}")
        print(f"  - Practices: {stats['wiki'].get('practices_count', 0)}")
        print(f"  - Technologies: {stats['wiki'].get('technologies_count', 0)}")
        print(f"  - Locations: {stats['wiki'].get('locations_count', 0)}")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Build failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
