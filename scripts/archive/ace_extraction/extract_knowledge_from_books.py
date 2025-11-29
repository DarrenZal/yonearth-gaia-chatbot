#!/usr/bin/env python3
"""
Extract knowledge graph entities/relationships from books.

This mirrors the episode extractor but reads book PDFs, chunks the text,
and writes book_*_extraction.json files under data/knowledge_graph/entities.

Usage:
    python scripts/extract_knowledge_from_books.py

    # Limit to specific books (comma separated slugs or titles)
    python scripts/extract_knowledge_from_books.py --books veriditas,y-on-earth
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.extractors.chunking import chunk_transcript
from src.knowledge_graph.extractors.entity_extractor import EntityExtractor
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default book slugs (match directory names) and friendly names
DEFAULT_BOOKS = {
    "veriditas": "VIRIDITAS: THE GREAT HEALING",
    "OurBiggestDeal": "Our Biggest Deal",
    "soil-stewardship-handbook": "Soil Stewardship Handbook",
    "y-on-earth": "Y on Earth",
}


def load_book_metadata(book_dir: Path) -> Optional[Dict]:
    """Load metadata.json from a book directory."""
    meta_path = book_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning(f"Missing metadata.json in {book_dir}")
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def chunk_book_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Chunk book text using token-aware chunking."""
    return chunk_transcript(
        transcript=text,
        chunk_size=chunk_size,
        overlap=chunk_overlap
    )


def extract_book(
    book_slug: str,
    book_dir: Path,
    entity_extractor: EntityExtractor,
    relationship_extractor: RelationshipExtractor,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int
) -> Optional[Dict]:
    """Process a single book directory."""
    meta = load_book_metadata(book_dir)
    if not meta:
        return None

    book_title = meta.get("title", book_slug)
    pdf_rel_path = meta.get("file_path")
    pdf_path = book_dir / pdf_rel_path if pdf_rel_path else None

    if not pdf_path or not pdf_path.exists():
        logger.warning(f"Book '{book_title}' ({book_slug}) missing PDF at {pdf_path}")
        return None

    logger.info(f"Processing book: {book_title} ({book_slug})")
    logger.info(f"PDF: {pdf_path}")

    full_text = extract_pdf_text(pdf_path)
    if not full_text:
        logger.warning(f"No text extracted from {pdf_path}")
        return None

    chunks = chunk_book_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"  Created {len(chunks)} chunks")

    all_entities = []
    all_relationships = []

    for chunk in chunks:
        chunk_id = f"book_{book_slug}_chunk_{chunk['chunk_index']}"
        text = chunk["text"]

        # Extract entities
        try:
            entity_result = entity_extractor.extract_entities(
                text=text,
                episode_number=0,  # Placeholder; not used downstream
                chunk_id=chunk_id
            )
            # Annotate book context
            for ent in entity_result.entities:
                ent.metadata["book_slug"] = book_slug
                ent.metadata["book_title"] = book_title
            all_entities.extend(entity_result.entities)
        except Exception as e:
            logger.error(f"  Error extracting entities for chunk {chunk_id}: {e}")
            continue

        # Extract relationships
        if entity_result.entities:
            entity_list = [
                {"name": e.name, "type": e.type}
                for e in entity_result.entities
            ]
            try:
                rel_result = relationship_extractor.extract_relationships(
                    text=text,
                    entities=entity_list,
                    episode_number=0,
                    chunk_id=chunk_id
                )
                for rel in rel_result.relationships:
                    rel.metadata["book_slug"] = book_slug
                    rel.metadata["book_title"] = book_title
                all_relationships.extend(rel_result.relationships)
            except Exception as e:
                logger.error(f"  Error extracting relationships for chunk {chunk_id}: {e}")

    # Aggregate
    aggregated_entities = entity_extractor.aggregate_entities(
        [type("Result", (), {"entities": all_entities})]
    )
    aggregated_relationships = relationship_extractor.aggregate_relationships(
        [type("Result", (), {"relationships": all_relationships})]
    )

    logger.info(
        f"  Extracted {len(aggregated_entities)} unique entities, "
        f"{len(aggregated_relationships)} unique relationships"
    )

    # Save extraction file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"book_{book_slug}_extraction.json"
    extraction_payload = {
        "book_slug": book_slug,
        "book_title": book_title,
        "entities": [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "aliases": e.aliases,
                "metadata": e.metadata
            }
            for e in aggregated_entities
        ],
        "relationships": [
            {
                "source_entity": r.source_entity,
                "relationship_type": r.relationship_type,
                "target_entity": r.target_entity,
                "description": r.description,
                "metadata": r.metadata
            }
            for r in aggregated_relationships
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extraction_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved extraction to {output_file}")

    return {
        "book_slug": book_slug,
        "book_title": book_title,
        "entities_count": len(aggregated_entities),
        "relationships_count": len(aggregated_relationships),
        "chunks_processed": len(chunks),
    }


def parse_book_list(raw: Optional[str]) -> Dict[str, str]:
    """Parse CLI list into slug->title map, falling back to defaults."""
    if not raw:
        return DEFAULT_BOOKS

    result = {}
    canonical_lookup = {slug.lower(): slug for slug in DEFAULT_BOOKS.keys()}
    title_lookup = {title.lower(): slug for slug, title in DEFAULT_BOOKS.items()}
    for item in raw.split(","):
        slug = item.strip()
        if not slug:
            continue
        slug_lower = slug.lower()
        # Map known titles to their slug
        if slug_lower in title_lookup:
            canonical_slug = title_lookup[slug_lower]
            result[canonical_slug] = DEFAULT_BOOKS[canonical_slug]
            continue
        # Map slug variants (case-insensitive or hyphenated) to canonical directories
        canonical_slug = canonical_lookup.get(slug_lower) or canonical_lookup.get(slug_lower.replace("-", ""))
        if canonical_slug:
            result[canonical_slug] = DEFAULT_BOOKS[canonical_slug]
            continue
        # Fallback: use as-is
        result[slug] = slug
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge graph from books")
    parser.add_argument(
        "--books",
        type=str,
        help="Comma-separated book slugs or titles (default: all supported)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Token chunk size (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Token overlap between chunks (default: 100)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip books with existing extraction files",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    books_dir = base_dir / "data" / "books"
    output_dir = base_dir / "data" / "knowledge_graph" / "entities"

    book_map = parse_book_list(args.books)

    # Initialize extractors
    logger.info("Initializing extractors...")
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()

    results = []
    for slug, title in book_map.items():
        book_path = books_dir / slug
        if not book_path.exists():
            logger.warning(f"Book directory missing: {book_path} (slug '{slug}', title '{title}')")
            continue

        output_file = output_dir / f"book_{slug}_extraction.json"
        if args.skip_existing and output_file.exists():
            logger.info(f"Skipping existing extraction: {output_file}")
            continue

        result = extract_book(
            book_slug=slug,
            book_dir=book_path,
            entity_extractor=entity_extractor,
            relationship_extractor=relationship_extractor,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if result:
            results.append(result)

    logger.info("\n" + "=" * 80)
    logger.info("BOOK EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Books processed: {len(results)}/{len(book_map)}")
    if results:
        total_entities = sum(r["entities_count"] for r in results)
        total_relationships = sum(r["relationships_count"] for r in results)
        logger.info(f"Total entities: {total_entities}")
        logger.info(f"Total relationships: {total_relationships}")
        logger.info(f"Average entities per book: {total_entities / len(results):.1f}")
        logger.info(f"Average relationships per book: {total_relationships / len(results):.1f}")
    logger.info(f"\nExtraction files saved to: {output_dir}")
    logger.info("Next step: run build_unified_graph_v2.py to build the unified graph")


if __name__ == "__main__":
    main()
