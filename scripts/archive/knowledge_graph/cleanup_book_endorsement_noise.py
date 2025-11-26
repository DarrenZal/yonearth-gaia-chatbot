#!/usr/bin/env python3
"""
One-time cleanup script for current book extractions (VIRIDITAS, Soil Stewardship, Y on Earth).

Removes "X MENTIONS Book_Title" noise that should have been filtered as endorsements.
This is a temporary fix until we implement page provenance tracking for proper endorsement detection.

NOT a permanent module - just a cleanup script for this extraction run.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_book_endorsement_noise(rel: Dict[str, Any], book_title: str, author_name: str) -> bool:
    """
    Detect if relationship is likely endorsement noise (should be filtered).

    Heuristics:
    1. Person name + MENTIONS/ENDORSES + exact book title (praise quotes)
    2. Author name + any predicate + exact book title (author-book metadata)
    3. Organization + PUBLISHES + exact book title (publication metadata)

    Uses EXACT book title matching to avoid false positives.
    """
    source = rel.get("source_entity", "")
    target = rel.get("target_entity", "")
    predicate = rel.get("relationship_type", "").upper()

    # Extract main book name (e.g., "VIRIDITAS" from "VIRIDITAS: THE GREAT HEALING")
    # Use first part before colon or subtitle
    book_main_name = book_title.split(':')[0].strip()
    book_main_name_lower = book_main_name.lower()

    # Check if target is exact book title or main book name
    target_lower = target.lower().strip()
    target_is_book = (
        target_lower == book_main_name_lower or
        target_lower == book_title.lower().strip() or
        target_lower == f"{book_main_name_lower} society"  # Handle variations
    )

    # Check if source is exact book title
    source_lower = source.lower().strip()
    source_is_book = (
        source_lower == book_main_name_lower or
        source_lower == book_title.lower().strip()
    )

    # Check if source is author (handle variations)
    author_parts = author_name.lower().split()
    source_is_author = any(part in source_lower for part in author_parts if len(part) > 3)

    # Praise quote predicates (endorsements from people - FILTER THESE)
    praise_predicates = {'MENTIONS', 'ENDORSES', 'REFERENCES', 'DISCUSSES', 'RECOMMENDS'}

    # Bibliographic predicates (authorship/publication - KEEP THESE as valuable metadata)
    # These tell us WHO wrote and WHO published - factual information worth preserving
    bibliographic_predicates = {
        'PRODUCES', 'WRITES', 'CREATES', 'FOUNDED', 'AUTHORED',
        'PUBLISHES', 'PUBLISHED_BY', 'DISTRIBUTED_BY'
    }

    # Historical/conceptual figures (likely discussing concept, not book)
    # Keep these relationships as domain knowledge
    historical_figures = {
        'hildegard von bingen', 'st. francis', 'francis of assisi',
        'dalai lama', 'fritjof capra', 'william irwin thompson',
        'mama-gaia', 'gaia', 'sophia', 'leo'  # Also story characters discussing concepts
    }

    # Check if source is a historical figure or story character
    source_is_historical = any(fig in source_lower for fig in historical_figures)

    # ONLY filter praise quotes - NOT bibliographic metadata!
    # Pattern: Person MENTIONS/ENDORSES Viriditas (praise quote from front matter)
    if target_is_book and predicate in praise_predicates and not source_is_historical:
        # Extra checks to avoid filtering domain knowledge:
        # - Historical figures discussing concept (already checked)
        # - Story locations/organizations (likely in narrative)
        if 'springs' in source_lower or 'community' in source_lower:
            return False
        # - Author relationships (bibliographic, not praise)
        if source_is_author:
            return False
        # - Publishers (bibliographic, not praise)
        if 'press' in source_lower or 'publishing' in source_lower:
            return False

        # Everything else with MENTIONS is likely a praise quote - filter it
        return True

    return False


def cleanup_book_endorsements(input_file: Path, output_file: Path, book_title: str, author_name: str):
    """
    Remove endorsement noise from a book extraction JSON file.

    Args:
        input_file: Path to input JSON (e.g., veriditas_ace_v14_3_8_improved.json)
        output_file: Path to output JSON (e.g., veriditas_ace_v14_3_8_cleaned.json)
        book_title: Full book title for detection
        author_name: Author name for detection
    """
    logger.info(f"Loading: {input_file}")
    with open(input_file) as f:
        data = json.load(f)

    original_count = len(data['relationships'])
    logger.info(f"Original: {original_count} relationships")

    # Filter relationships
    kept = []
    filtered = []

    for rel in data['relationships']:
        if is_book_endorsement_noise(rel, book_title, author_name):
            filtered.append(rel)
        else:
            kept.append(rel)

    # Update data
    data['relationships'] = kept
    data['pipeline_version'] = data.get('pipeline_version', '') + '_endorsement_cleaned'

    # Add cleanup stats
    if 'postprocessing_stats' not in data:
        data['postprocessing_stats'] = {}

    data['postprocessing_stats']['endorsement_cleanup'] = {
        'filtered_count': len(filtered),
        'kept_count': len(kept),
        'filter_percentage': f"{(len(filtered) / original_count * 100):.1f}%"
    }

    # Save cleaned version
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"✅ Cleaned version saved: {output_file}")
    logger.info(f"📊 Filtered: {len(filtered)} endorsement relationships")
    logger.info(f"📊 Kept: {len(kept)} domain relationships")
    logger.info(f"📊 Noise reduction: {(len(filtered) / original_count * 100):.1f}%")

    # Show examples of what was filtered
    if filtered:
        logger.info("\n📋 Sample filtered relationships:")
        for i, rel in enumerate(filtered[:5], 1):
            logger.info(f"  {i}. ({rel['source_entity']}) --[{rel['relationship_type']}]--> ({rel['target_entity']})")


def main():
    project_root = Path("/home/claudeuser/yonearth-gaia-chatbot")
    kg_dir = project_root / "data/knowledge_graph/books"

    # Book configurations
    books = [
        {
            "slug": "veriditas",
            "title": "VIRIDITAS: THE GREAT HEALING",
            "author": "Aaron William Perry",
            "input_file": kg_dir / "veriditas_ace_v14_3_8_improved.json",
            "output_file": kg_dir / "veriditas_ace_v14_3_8_cleaned.json"
        },
        {
            "slug": "soil_stewardship",
            "title": "Soil Stewardship Handbook",
            "author": "Aaron William Perry",
            "input_file": kg_dir / "soil-stewardship-handbook_ace_v14_3_8.json",
            "output_file": kg_dir / "soil-stewardship-handbook_ace_v14_3_8_cleaned.json"
        },
        {
            "slug": "y_on_earth",
            "title": "Y on Earth: Get Smarter, Feel Better, Heal the Planet",
            "author": "Aaron William Perry",
            "input_file": kg_dir / "y-on-earth_ace_v14_3_8.json",
            "output_file": kg_dir / "y-on-earth_ace_v14_3_8_cleaned.json"
        },
        {
            "slug": "our_biggest_deal",
            "title": "Our Biggest Deal",
            "author": "Aaron William Perry",
            "input_file": kg_dir / "OurBiggestDeal_ace_v14_3_8.json",
            "output_file": kg_dir / "OurBiggestDeal_ace_v14_3_8_cleaned.json"
        }
    ]

    for book_config in books:
        if not book_config["input_file"].exists():
            logger.warning(f"⚠️  Input file not found: {book_config['input_file']}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {book_config['title']}")
        logger.info(f"{'='*60}")

        cleanup_book_endorsements(
            input_file=book_config["input_file"],
            output_file=book_config["output_file"],
            book_title=book_config["title"],
            author_name=book_config["author"]
        )


if __name__ == "__main__":
    main()
