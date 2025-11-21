#!/usr/bin/env python3
"""
Extract full books with ACE V14.3.8 postprocessing quality.

This script extracts knowledge graphs from complete books (not chapter-by-chapter)
and applies the V14.3.8 ACE postprocessing pipeline for A+ quality.

Usage:
    python scripts/extract_books_ace_full.py

    # Specific books
    python scripts/extract_books_ace_full.py --books veriditas,soil-stewardship-handbook
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv(project_root / ".env")

from src.knowledge_graph.extractors.chunking import chunk_transcript
from src.knowledge_graph.extractors.entity_extractor import EntityExtractor
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor
from src.knowledge_graph.postprocessing.pipelines.book_pipeline import get_book_pipeline_v1438

import pdfplumber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Books to process
BOOKS = {
    "veriditas": {
        "title": "VIRIDITAS: THE GREAT HEALING",
        "author": "Aaron William Perry",
        "file": "VIRIDITAS by AARON WILLIAM PERRY.pdf"
    },
    "soil-stewardship-handbook": {
        "title": "Soil Stewardship Handbook",
        "author": "Y on Earth Community",
        "file": "Soil-Stewardship-Handbook-eBook.pdf"
    },
    "y-on-earth": {
        "title": "Y on Earth",
        "author": "Aaron William Perry",
        "file": "Y ON EARTH by AARON WILLIAM PERRY.pdf"
    },
    "OurBiggestDeal": {
        "title": "Our Biggest Deal",
        "author": "Aaron William Perry",
        "file": "our_biggest_deal.pdf"
    }
}


def extract_book_with_ace(book_slug: str, books_dir: Path, output_dir: Path):
    """Extract a complete book with ACE V14.3.8 postprocessing."""

    book_info = BOOKS[book_slug]
    book_dir = books_dir / book_slug
    pdf_path = book_dir / book_info["file"]

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    logger.info(f"Processing: {book_info['title']}")
    logger.info(f"PDF: {pdf_path}")

    # Extract text from PDF
    logger.info("Extracting text from PDF...")
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        total_pages = len(pdf.pages)

    logger.info(f"  Total pages: {total_pages}")
    logger.info(f"  Total characters: {len(full_text):,}")

    # Chunk the text
    logger.info("Chunking text...")
    chunk_dicts = chunk_transcript(
        full_text,
        chunk_size=800,
        overlap=100
    )
    chunks = [c["text"] for c in chunk_dicts]
    logger.info(f"  Created {len(chunks)} chunks")

    # Initialize extractors
    logger.info("Initializing extractors...")
    entity_extractor = EntityExtractor()
    relationship_extractor = RelationshipExtractor()

    # Extract entities and relationships
    logger.info("Extracting entities and relationships...")
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processing chunk {i+1}/{len(chunks)}")

        # Extract entities
        entity_result = entity_extractor.extract_entities(
            text=chunk,
            episode_number=0,  # Not episode-based
            chunk_id=f"book_{book_slug}_chunk_{i}"
        )
        all_entities.extend(entity_result.entities)

        # Convert entities to dict format for relationship extractor
        entities_for_rel_extraction = [
            {
                "name": e.name,
                "type": e.type
            }
            for e in entity_result.entities
        ]

        # Extract relationships (requires entities parameter)
        rel_result = relationship_extractor.extract_relationships(
            text=chunk,
            entities=entities_for_rel_extraction,
            episode_number=0,
            chunk_id=f"book_{book_slug}_chunk_{i}"
        )
        all_relationships.extend(rel_result.relationships)

    logger.info(f"Extracted {len(all_entities)} entities")
    logger.info(f"Extracted {len(all_relationships)} relationships")

    # Apply ACE V14.3.8 postprocessing pipeline
    logger.info("Applying ACE V14.3.8 postprocessing...")
    pipeline = get_book_pipeline_v1438()

    # Convert to pipeline format
    pipeline_rels = []
    for rel in all_relationships:
        pipeline_rels.append({
            "source": rel.source_entity,
            "relationship": rel.relationship_type,
            "target": rel.target_entity,
            "context": rel.description,
            "metadata": rel.metadata
        })

    # Create context object
    class Context:
        def __init__(self, book_title, author):
            self.document_metadata = {
                "title": book_title,
                "author": author,
                "content_type": "book"
            }

    context = Context(book_info["title"], book_info["author"])

    # Run pipeline
    processed_rels = pipeline.process(pipeline_rels, context)

    logger.info(f"After postprocessing: {len(processed_rels)} relationships")

    # Save output
    output_file = output_dir / f"{book_slug}_ace_v14_3_8.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "book": book_slug,
        "title": book_info["title"],
        "author": book_info["author"],
        "pipeline_version": "v14_3_8",
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "entities": [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "metadata": e.metadata
            }
            for e in all_entities
        ],
        "relationships": processed_rels,
        "postprocessing_stats": pipeline.get_stats()
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved: {output_file}")
    logger.info(f"✅ {book_info['title']} complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--books",
        help="Comma-separated book slugs (default: all except OurBiggestDeal)"
    )
    args = parser.parse_args()

    # Default: Extract 3 books (skip OurBiggestDeal - we have ACE chapters)
    books_to_process = ["veriditas", "soil-stewardship-handbook", "y-on-earth"]

    if args.books:
        books_to_process = [b.strip() for b in args.books.split(",")]

    books_dir = project_root / "data" / "books"
    output_dir = project_root / "data" / "knowledge_graph" / "books"

    logger.info(f"Starting ACE extraction for {len(books_to_process)} books")
    logger.info(f"Books: {', '.join(books_to_process)}")

    for book_slug in books_to_process:
        if book_slug not in BOOKS:
            logger.error(f"Unknown book: {book_slug}")
            continue

        try:
            extract_book_with_ace(book_slug, books_dir, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {book_slug}: {e}", exc_info=True)

    logger.info("✅ All books processed!")


if __name__ == "__main__":
    main()
