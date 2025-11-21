#!/usr/bin/env python3
"""
Generate Obsidian-compatible wiki from knowledge graph extractions.

This script creates a complete wiki with entity pages, episode pages,
indexes, and summary pages.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.wiki.wiki_builder import WikiBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'data' / 'knowledge_graph' / 'wiki_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main wiki generation function."""
    logger.info("=" * 80)
    logger.info("KNOWLEDGE GRAPH WIKI GENERATION")
    logger.info("=" * 80)

    # Paths
    extraction_dir = project_root / 'data' / 'knowledge_graph' / 'entities'
    output_dir = project_root / 'data' / 'knowledge_graph' / 'wiki'

    logger.info(f"Extraction directory: {extraction_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Check extraction directory exists
    if not extraction_dir.exists():
        logger.error(f"Extraction directory not found: {extraction_dir}")
        sys.exit(1)

    # Count extraction files
    extraction_files = list(extraction_dir.glob('episode_*_extraction.json'))
    logger.info(f"Found {len(extraction_files)} extraction files")

    if len(extraction_files) == 0:
        logger.error("No extraction files found!")
        sys.exit(1)

    # Build wiki
    builder = WikiBuilder(output_dir)
    stats = builder.build(extraction_dir)

    # Print results
    logger.info("=" * 80)
    logger.info("WIKI GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Statistics:")
    logger.info(f"  Total Entities: {stats['total_entities']}")
    logger.info(f"  Total Episodes: {stats['total_episodes']}")
    logger.info(f"  People: {stats['people_count']}")
    logger.info(f"  Organizations: {stats['organizations_count']}")
    logger.info(f"  Concepts: {stats['concepts_count']}")
    logger.info(f"  Practices: {stats['practices_count']}")
    logger.info(f"  Technologies: {stats['technologies_count']}")
    logger.info(f"  Locations: {stats['locations_count']}")
    logger.info("")

    # Print directory structure
    logger.info("Directory Structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            file_count = len(list(subdir.glob('*.md')))
            logger.info(f"  {subdir.name}/: {file_count} files")

    logger.info("")
    logger.info("To view the wiki:")
    logger.info(f"  1. Open Obsidian")
    logger.info(f"  2. Open folder as vault: {output_dir}")
    logger.info(f"  3. Start with Index.md")
    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
