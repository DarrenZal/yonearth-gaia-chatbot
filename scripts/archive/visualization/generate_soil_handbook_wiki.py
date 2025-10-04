#!/usr/bin/env python3
"""
Generate Obsidian-compatible wiki for Soil Stewardship Handbook.

This script creates a wiki from the Soil Stewardship Handbook knowledge graph extraction.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.wiki.markdown_generator import MarkdownGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'soil_handbook_wiki_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_book_extraction(book_path: Path) -> dict:
    """Load book extraction JSON."""
    logger.info(f"Loading extraction from: {book_path}")
    with open(book_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data.get('entities', []))} entities, {len(data.get('relationships', []))} relationships")
    return data


def build_wiki_structure(extraction_data: dict, output_dir: Path):
    """Build wiki directory structure and pages."""

    # Create directories
    directories = [
        output_dir,
        output_dir / 'chapters',
        output_dir / 'people',
        output_dir / 'organizations',
        output_dir / 'concepts',
        output_dir / 'practices',
        output_dir / 'locations',
        output_dir / 'technologies',
        output_dir / '_indexes',
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # Process entities
    entities_by_type = defaultdict(list)
    entities_by_name = {}
    chapters = defaultdict(list)

    for entity in extraction_data.get('entities', []):
        entity_type = entity.get('type', 'CONCEPT')
        entity_name = entity.get('name', '')

        # Track by type
        entities_by_type[entity_type].append(entity)
        entities_by_name[entity_name] = entity

        # Track by chapter
        chapter = entity.get('metadata', {}).get('chapter', 'Unknown')
        chapters[chapter].append(entity)

    # Generate markdown
    generator = MarkdownGenerator()

    # Generate entity pages
    logger.info("Generating entity pages...")
    for entity_type, entities in entities_by_type.items():
        type_dir_map = {
            'PERSON': 'people',
            'ORGANIZATION': 'organizations',
            'CONCEPT': 'concepts',
            'PRACTICE': 'practices',
            'PLACE': 'locations',
            'TECHNOLOGY': 'technologies',
        }

        type_dir = type_dir_map.get(entity_type, 'concepts')

        for entity in entities:
            # Prepare entity data
            entity_data = {
                'name': entity.get('name', ''),
                'description': entity.get('description', ''),
                'aliases': entity.get('aliases', []),
                'episode_count': 0,  # Book doesn't have episodes
                'chapters': [],
                'mention_count': 1,
                'importance_score': 1,
                'domains': [],
                'affiliations': [],
                'related_concepts': [],
                'related_practices': [],
            }

            # Add chapter reference
            chapter = entity.get('metadata', {}).get('chapter', 'Unknown')
            if chapter != 'Unknown':
                entity_data['chapters'] = [{'number': chapter, 'title': f'Chapter {chapter}'}]

            # Generate markdown using templates directly
            template_type = entity_type if entity_type in generator.templates else 'CONCEPT'

            if template_type in ['PERSON', 'ORGANIZATION']:
                # Use template directly
                markdown = generator.generate_entity_page(entity_data, template_type)
            elif template_type == 'CONCEPT':
                # Adapt concept template for books
                markdown = f"""---
type: concept
name: {entity_data['name']}
aliases: {', '.join(entity_data['aliases']) if entity_data['aliases'] else 'none'}
source: Soil Stewardship Handbook
---

# {entity_data['name']}

## Definition
{entity_data['description']}

## Chapters
"""
                for chapter in entity_data['chapters']:
                    markdown += f"- [[Chapter {chapter['number']}]]: {chapter['title']}\n"

                markdown += "\n---\n*Source: Soil Stewardship Handbook*\n"
            else:
                # Default template
                markdown = f"""---
type: {entity_type.lower()}
name: {entity_data['name']}
source: Soil Stewardship Handbook
---

# {entity_data['name']}

{entity_data['description']}

## Chapters
"""
                for chapter in entity_data['chapters']:
                    markdown += f"- [[Chapter {chapter['number']}]]: {chapter['title']}\n"

            # Write file
            filename = entity_data['name'].replace('/', '_').replace(' ', '_') + '.md'
            filepath = output_dir / type_dir / filename
            filepath.write_text(markdown, encoding='utf-8')

    # Generate chapter pages
    logger.info("Generating chapter pages...")
    for chapter_num, chapter_entities in sorted(chapters.items()):
        if chapter_num == 'Unknown':
            continue

        markdown = f"""---
type: chapter
number: {chapter_num}
source: Soil Stewardship Handbook
---

# Chapter {chapter_num}

## Entities in This Chapter

### People
"""
        # Group by type
        people = [e for e in chapter_entities if e.get('type') == 'PERSON']
        orgs = [e for e in chapter_entities if e.get('type') == 'ORGANIZATION']
        concepts = [e for e in chapter_entities if e.get('type') == 'CONCEPT']
        practices = [e for e in chapter_entities if e.get('type') == 'PRACTICE']

        for person in people:
            markdown += f"- [[{person['name']}]]\n"

        markdown += "\n### Organizations\n"
        for org in orgs:
            markdown += f"- [[{org['name']}]]\n"

        markdown += "\n### Concepts\n"
        for concept in concepts:
            markdown += f"- [[{concept['name']}]]\n"

        markdown += "\n### Practices\n"
        for practice in practices:
            markdown += f"- [[{practice['name']}]]\n"

        markdown += "\n---\n*Part of Soil Stewardship Handbook Knowledge Graph*\n"

        filepath = output_dir / 'chapters' / f'Chapter_{chapter_num}.md'
        filepath.write_text(markdown, encoding='utf-8')

    # Generate index page
    logger.info("Generating index page...")
    index_md = f"""---
title: Soil Stewardship Handbook Knowledge Graph
date: {datetime.now().strftime('%Y-%m-%d')}
---

# Soil Stewardship Handbook Knowledge Graph

Welcome to the knowledge graph extracted from the Soil Stewardship Handbook.

## Statistics

- **Total Entities**: {len(extraction_data.get('entities', []))}
- **Total Relationships**: {len(extraction_data.get('relationships', []))}
- **Chapters Covered**: {len([c for c in chapters.keys() if c != 'Unknown'])}

## Browse by Type

- [[People Index]]: {len(entities_by_type.get('PERSON', []))} people
- [[Organizations Index]]: {len(entities_by_type.get('ORGANIZATION', []))} organizations
- [[Concepts Index]]: {len(entities_by_type.get('CONCEPT', []))} concepts
- [[Practices Index]]: {len(entities_by_type.get('PRACTICE', []))} practices
- [[Locations Index]]: {len(entities_by_type.get('PLACE', []))} locations

## Browse by Chapter

"""
    for chapter_num in sorted([c for c in chapters.keys() if c != 'Unknown']):
        entity_count = len(chapters[chapter_num])
        index_md += f"- [[Chapter {chapter_num}]]: {entity_count} entities\n"

    index_md += "\n---\n*Generated from Soil Stewardship Handbook extraction*\n"

    (output_dir / 'Index.md').write_text(index_md, encoding='utf-8')

    # Generate type indexes
    logger.info("Generating type index pages...")
    type_names = {
        'PERSON': 'People',
        'ORGANIZATION': 'Organizations',
        'CONCEPT': 'Concepts',
        'PRACTICE': 'Practices',
        'PLACE': 'Locations',
    }

    for entity_type, type_name in type_names.items():
        entities = entities_by_type.get(entity_type, [])
        index_content = f"""# {type_name} Index

Total {type_name}: {len(entities)}

"""
        for entity in sorted(entities, key=lambda e: e.get('name', '')):
            index_content += f"- [[{entity['name']}]]\n"

        (output_dir / '_indexes' / f'{type_name}_Index.md').write_text(index_content, encoding='utf-8')

    return {
        'total_entities': len(extraction_data.get('entities', [])),
        'total_relationships': len(extraction_data.get('relationships', [])),
        'total_chapters': len([c for c in chapters.keys() if c != 'Unknown']),
        'entities_by_type': {k: len(v) for k, v in entities_by_type.items()},
    }


def main():
    """Main wiki generation function."""
    logger.info("=" * 80)
    logger.info("SOIL STEWARDSHIP HANDBOOK WIKI GENERATION")
    logger.info("=" * 80)

    # Paths
    book_extraction_file = project_root / 'data' / 'knowledge_graph' / 'books' / 'Soil_Stewardship_Handbook_extraction.json'
    output_dir = project_root / 'web' / 'soil-handbook-wiki'

    logger.info(f"Book extraction: {book_extraction_file}")
    logger.info(f"Output directory: {output_dir}")

    # Check extraction file exists
    if not book_extraction_file.exists():
        logger.error(f"Extraction file not found: {book_extraction_file}")
        sys.exit(1)

    # Load extraction
    extraction_data = load_book_extraction(book_extraction_file)

    # Build wiki
    stats = build_wiki_structure(extraction_data, output_dir)

    # Print results
    logger.info("=" * 80)
    logger.info("WIKI GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Statistics:")
    logger.info(f"  Total Entities: {stats['total_entities']}")
    logger.info(f"  Total Relationships: {stats['total_relationships']}")
    logger.info(f"  Total Chapters: {stats['total_chapters']}")
    logger.info("")
    logger.info("Entities by Type:")
    for entity_type, count in stats['entities_by_type'].items():
        logger.info(f"  {entity_type}: {count}")
    logger.info("")
    logger.info("To view the wiki:")
    logger.info(f"  1. Open Obsidian")
    logger.info(f"  2. Open folder as vault: {output_dir}")
    logger.info(f"  3. Start with Index.md")
    logger.info("")
    logger.info("Or deploy to Quartz:")
    logger.info(f"  cd wiki-quartz")
    logger.info(f"  npx quartz build --serve")
    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
