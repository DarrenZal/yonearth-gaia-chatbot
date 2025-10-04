#!/usr/bin/env python3
"""
Extract Knowledge Graphs from Books

This script processes book PDFs to extract entities and relationships,
creating a comprehensive knowledge graph for the Y on Earth book collection.

Books to process:
- Soil Stewardship Handbook
- Y on Earth: Get Smarter, Feel Better, Heal the Planet
- VIRIDITAS: THE GREAT HEALING (if PDF available)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/claudeuser/yonearth-gaia-chatbot/.env')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_graph.extractors.entity_extractor import EntityExtractor, Entity, EntityExtractionResult
from src.knowledge_graph.extractors.relationship_extractor import RelationshipExtractor, Relationship, RelationshipExtractionResult
from src.knowledge_graph.extractors.chunking import chunk_transcript
from src.ingestion.book_processor import BookProcessor, Book

# Configure logging
log_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'book_kg_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BookKnowledgeGraphExtractor:
    """Extracts knowledge graphs from books"""

    def __init__(
        self,
        books_dir: str,
        output_dir: str,
        chunk_size: int = 1000,  # Larger chunks for books
        chunk_overlap: int = 100   # More overlap for context
    ):
        """
        Initialize the book knowledge graph extractor.

        Args:
            books_dir: Directory containing book metadata and PDFs
            output_dir: Directory to save extraction results
            chunk_size: Size of text chunks in tokens (larger for books)
            chunk_overlap: Overlap between chunks in tokens
        """
        self.books_dir = Path(books_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "books").mkdir(exist_ok=True)

        # Initialize book processor
        self.book_processor = BookProcessor()

        # Initialize extractors
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        self.entity_extractor = EntityExtractor(api_key=api_key, model="gpt-4o-mini")
        self.relationship_extractor = RelationshipExtractor(api_key=api_key, model="gpt-4o-mini")

        # Statistics tracking
        self.stats = {
            "total_books": 0,
            "successful_books": 0,
            "failed_books": [],
            "total_chapters": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "entity_type_counts": {},
            "relationship_type_counts": {},
            "start_time": None,
            "end_time": None
        }

    def process_book(self, book: Book) -> Optional[Dict[str, Any]]:
        """
        Process a single book, extracting entities and relationships.

        Args:
            book: Book object to process

        Returns:
            Dictionary with book data and extraction results
        """
        logger.info(f"Processing Book: {book.title}")
        logger.info(f"  Author: {book.author}")
        logger.info(f"  PDF Path: {book.pdf_path}")

        # Check if PDF exists
        if not book.pdf_path.exists():
            logger.warning(f"  PDF file not found: {book.pdf_path}")
            return None

        try:
            # Extract text from PDF
            book.extract_text_from_pdf()
            logger.info(f"  Extracted {book.word_count} words from {book.pages} pages")

            # Detect chapters
            chapters = book.detect_chapters()
            logger.info(f"  Detected {len(chapters)} chapters")

            if not chapters:
                logger.warning(f"  No chapters detected in {book.title}")
                return None

            # Process each chapter
            all_entity_results = []
            all_relationship_results = []
            chapter_summaries = []

            for chapter in chapters:
                if not chapter.has_content:
                    logger.info(f"    Skipping chapter {chapter.number} (insufficient content)")
                    continue

                logger.info(f"  Processing Chapter {chapter.number}: {chapter.title}")
                logger.info(f"    Word count: {chapter.word_count}")

                # Chunk the chapter content
                chunks = chunk_transcript(
                    chapter.content,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap
                )
                logger.info(f"    Created {len(chunks)} chunks")

                # Process each chunk
                chapter_entities = []
                chapter_relationships = []

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{book.title.replace(' ', '_')}_ch{chapter.number}_chunk{chunk['chunk_index']}"

                    try:
                        logger.info(f"      Processing chunk {i+1}/{len(chunks)}")

                        # Extract entities
                        entity_result = self.entity_extractor.extract_entities(
                            text=chunk["text"],
                            episode_number=0,  # Not an episode
                            chunk_id=chunk_id
                        )

                        # Update metadata to include book/chapter info
                        for entity in entity_result.entities:
                            entity.metadata.update({
                                "book_title": book.title,
                                "chapter_number": chapter.number,
                                "chapter_title": chapter.title,
                                "content_type": "book"
                            })

                        chapter_entities.append(entity_result)
                        all_entity_results.append(entity_result)

                        # Extract relationships using the entities found
                        entities_list = [
                            {"name": e.name, "type": e.type}
                            for e in entity_result.entities
                        ]

                        relationship_result = self.relationship_extractor.extract_relationships(
                            text=chunk["text"],
                            entities=entities_list,
                            episode_number=0,  # Not an episode
                            chunk_id=chunk_id
                        )

                        # Update metadata to include book/chapter info
                        for rel in relationship_result.relationships:
                            rel.metadata.update({
                                "book_title": book.title,
                                "chapter_number": chapter.number,
                                "chapter_title": chapter.title,
                                "content_type": "book"
                            })

                        chapter_relationships.append(relationship_result)
                        all_relationship_results.append(relationship_result)

                        logger.info(f"        Found {len(entity_result.entities)} entities, "
                                  f"{len(relationship_result.relationships)} relationships")

                    except Exception as e:
                        logger.error(f"      Failed to process chunk {i}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue

                # Aggregate chapter results
                chapter_unique_entities = self.entity_extractor.aggregate_entities(chapter_entities)
                chapter_unique_relationships = self.relationship_extractor.aggregate_relationships(chapter_relationships)

                chapter_summary = {
                    "chapter_number": chapter.number,
                    "chapter_title": chapter.title,
                    "word_count": chapter.word_count,
                    "chunks": len(chunks),
                    "entities": len(chapter_unique_entities),
                    "relationships": len(chapter_unique_relationships)
                }
                chapter_summaries.append(chapter_summary)

                logger.info(f"    Chapter {chapter.number} complete: "
                          f"{len(chapter_unique_entities)} unique entities, "
                          f"{len(chapter_unique_relationships)} unique relationships")

            # Aggregate all results for the book
            book_result = self._aggregate_book_results(
                book=book,
                entity_results=all_entity_results,
                relationship_results=all_relationship_results,
                chapter_summaries=chapter_summaries
            )

            # Update statistics
            self._update_stats(book_result)

            logger.info(f"  Book complete: {book_result['total_entities']} total unique entities, "
                       f"{book_result['total_relationships']} total unique relationships")

            return book_result

        except Exception as e:
            logger.error(f"  Failed to process book {book.title}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _aggregate_book_results(
        self,
        book: Book,
        entity_results: List[EntityExtractionResult],
        relationship_results: List[RelationshipExtractionResult],
        chapter_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate all extraction results for a book.

        Args:
            book: Book object
            entity_results: List of entity extraction results from all chunks
            relationship_results: List of relationship extraction results from all chunks
            chapter_summaries: List of chapter summary dictionaries

        Returns:
            Aggregated book results
        """
        # Use the aggregation methods from the extractors
        unique_entities = self.entity_extractor.aggregate_entities(entity_results)
        unique_relationships = self.relationship_extractor.aggregate_relationships(relationship_results)

        # Count entity types
        entity_type_counts = {}
        for entity in unique_entities:
            entity_type = entity.type
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        # Count relationship types
        relationship_type_counts = {}
        for rel in unique_relationships:
            rel_type = rel.relationship_type
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

        # Convert to dictionaries for JSON serialization
        entities_dicts = [e.model_dump() for e in unique_entities]
        relationships_dicts = [r.model_dump() for r in unique_relationships]

        return {
            "book_title": book.title,
            "author": book.author,
            "publication_year": book.publication_year,
            "category": book.category,
            "topics": book.topics,
            "pages": book.pages,
            "word_count": book.word_count,
            "total_chapters": len(chapter_summaries),
            "total_chunks": sum(ch["chunks"] for ch in chapter_summaries),
            "total_entities": len(unique_entities),
            "total_relationships": len(unique_relationships),
            "entity_type_counts": entity_type_counts,
            "relationship_type_counts": relationship_type_counts,
            "chapter_summaries": chapter_summaries,
            "entities": entities_dicts,
            "relationships": relationships_dicts
        }

    def _update_stats(self, book_result: Dict[str, Any]):
        """Update overall statistics with book results."""
        self.stats["total_chapters"] += book_result["total_chapters"]
        self.stats["total_chunks"] += book_result["total_chunks"]
        self.stats["total_entities"] += book_result["total_entities"]
        self.stats["total_relationships"] += book_result["total_relationships"]

        # Update entity type counts
        for entity_type, count in book_result["entity_type_counts"].items():
            self.stats["entity_type_counts"][entity_type] = \
                self.stats["entity_type_counts"].get(entity_type, 0) + count

        # Update relationship type counts
        for rel_type, count in book_result.get("relationship_type_counts", {}).items():
            self.stats["relationship_type_counts"][rel_type] = \
                self.stats["relationship_type_counts"].get(rel_type, 0) + count

    def save_book_result(self, book_result: Dict[str, Any]):
        """
        Save book extraction results to JSON file.

        Args:
            book_result: Book extraction results
        """
        book_title = book_result["book_title"].replace(" ", "_").replace(":", "")
        filepath = self.output_dir / "books" / f"{book_title}_extraction.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(book_result, f, indent=2, ensure_ascii=False)

        logger.info(f"  Saved results to {filepath}")

    def process_all_books(self):
        """Process all books found in the books directory."""
        logger.info("Starting book knowledge graph extraction")
        self.stats["start_time"] = datetime.now()

        # Load books
        books = self.book_processor.load_books()
        logger.info(f"Found {len(books)} books to process")

        for book in books:
            try:
                self.stats["total_books"] += 1

                # Process book
                result = self.process_book(book)

                if result:
                    # Save results
                    self.save_book_result(result)
                    self.stats["successful_books"] += 1
                else:
                    self.stats["failed_books"].append({
                        "book": book.title,
                        "reason": "Processing returned None (likely missing PDF or no chapters)"
                    })

            except Exception as e:
                logger.error(f"Failed to process book {book.title}: {e}")
                self.stats["failed_books"].append({
                    "book": book.title,
                    "reason": str(e)
                })
                continue

        self.stats["end_time"] = datetime.now()

    def generate_summary_report(self) -> str:
        """Generate a summary report of the book processing."""
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = self.stats["end_time"] - self.stats["start_time"]
            duration_str = str(duration).split('.')[0]
        else:
            duration_str = "N/A"

        report = f"""
=============================================================================
BOOK KNOWLEDGE GRAPH EXTRACTION REPORT
=============================================================================

PROCESSING SUMMARY
-----------------
Total Books Attempted:        {self.stats['total_books']}
Successfully Processed:       {self.stats['successful_books']}
Failed Books:                 {len(self.stats['failed_books'])}
Processing Time:              {duration_str}

EXTRACTION STATISTICS
--------------------
Total Chapters Processed:     {self.stats['total_chapters']}
Total Chunks Processed:       {self.stats['total_chunks']}
Total Entities Extracted:     {self.stats['total_entities']}
Total Relationships Extracted: {self.stats['total_relationships']}

ENTITY TYPE DISTRIBUTION
------------------------
"""
        for entity_type, count in sorted(
            self.stats['entity_type_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"  {entity_type:20} {count:6d}\n"

        report += "\nRELATIONSHIP TYPE DISTRIBUTION\n"
        report += "-------------------------------\n"
        for rel_type, count in sorted(
            self.stats['relationship_type_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"  {rel_type:30} {count:6d}\n"

        if self.stats['failed_books']:
            report += "\nFAILED BOOKS\n"
            report += "-------------\n"
            for failure in self.stats['failed_books']:
                report += f"  {failure['book']}: {failure['reason']}\n"

        report += "\n=============================================================================\n"

        return report

    def save_summary_report(self, filepath: str):
        """Save summary report to file."""
        report = self.generate_summary_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Saved summary report to {filepath}")

        # Also save statistics as JSON
        stats_filepath = filepath.replace('.txt', '.json')
        with open(stats_filepath, 'w', encoding='utf-8') as f:
            # Convert datetime objects to strings
            stats_copy = self.stats.copy()
            if stats_copy['start_time']:
                stats_copy['start_time'] = stats_copy['start_time'].isoformat()
            if stats_copy['end_time']:
                stats_copy['end_time'] = stats_copy['end_time'].isoformat()

            json.dump(stats_copy, f, indent=2, ensure_ascii=False)


def main():
    """Main execution function"""
    # Configuration
    books_dir = "/home/claudeuser/yonearth-gaia-chatbot/data/books"
    output_dir = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph"
    report_path = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/book_extraction_report.txt"

    # Initialize extractor
    extractor = BookKnowledgeGraphExtractor(
        books_dir=books_dir,
        output_dir=output_dir,
        chunk_size=1000,  # Larger chunks for books
        chunk_overlap=100   # More overlap for context
    )

    try:
        # Process all books
        extractor.process_all_books()

        # Generate and save report
        extractor.save_summary_report(report_path)

        # Print report to console
        print(extractor.generate_summary_report())

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        # Still save what we have
        extractor.save_summary_report(report_path)
        print(extractor.generate_summary_report())

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
