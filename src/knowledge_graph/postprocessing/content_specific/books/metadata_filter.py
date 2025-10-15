"""
Metadata Filter Module (V14.0)

Filters out book metadata relationships (praise quotes, dedications, publication info)
from knowledge graph output to ensure only domain knowledge is preserved.

Multi-layer detection approach:
1. Flag-based: PRAISE_QUOTE_CORRECTED, DEDICATION_DETECTED
2. Predicate-based: endorsed, dedicated, published by/in, wrote foreword for
3. Page-based: Front/back matter (pages 1-10, last 10 pages)
4. Combined: Book title + person name on early pages

Purpose:
Separates book metadata from domain knowledge, preventing pollution of
knowledge graph with publication details that aren't part of the subject matter.

Version History:
- v1.0.0 (V14.0): Initial implementation with multi-layer detection
"""

import re
import logging
from typing import Optional, List, Dict, Any

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class MetadataFilter(PostProcessingModule):
    """
    Filters book metadata relationships to preserve only domain knowledge.

    Content Types: Books only
    Priority: 11 (runs after PraiseQuoteDetector which flags metadata)
    Dependencies: PraiseQuoteDetector (provides PRAISE_QUOTE_CORRECTED flag)
    """

    name = "MetadataFilter"
    description = "Filters book metadata (praise, dedications, publication info)"
    content_types = ["book"]
    priority = 11  # Run right after PraiseQuoteDetector (priority 10)
    dependencies = ["PraiseQuoteDetector"]  # Depends on praise quote detection
    version = "1.0.0"  # V14.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Configuration
        self.front_matter_threshold = self.config.get('front_matter_threshold', 10)
        self.back_matter_threshold = self.config.get('back_matter_threshold', 10)

        # Metadata predicates (normalized to lowercase for matching)
        self.metadata_predicates = set(self.config.get('metadata_predicates', [
            'endorsed',
            'dedicated',
            'dedicated to',
            'published by',
            'published in',
            'wrote foreword for',
            'wrote introduction for',
            'wrote preface for',
            'foreword by',
            'introduction by',
            'preface by'
        ]))

        # Store filtered metadata for potential later use
        self.filtered_metadata: List[Any] = []

    def is_metadata_by_flag(self, rel: Any) -> bool:
        """Layer 1: Check if relationship is flagged as metadata"""
        if rel.flags is None:
            return False

        return (
            rel.flags.get('PRAISE_QUOTE_CORRECTED', False) or
            rel.flags.get('DEDICATION_DETECTED', False) or
            rel.flags.get('PRAISE_QUOTE_FILTERED', False)
        )

    def is_metadata_by_predicate(self, rel: Any, book_title: str) -> bool:
        """Layer 2: Check if predicate + book title indicates metadata"""
        predicate = rel.relationship.lower().strip()

        # Check if predicate is in metadata list
        if predicate not in self.metadata_predicates:
            return False

        # Check if source or target matches book title (case-insensitive)
        source_matches = book_title.lower() in rel.source_entity.lower()
        target_matches = book_title.lower() in rel.target_entity.lower()

        return source_matches or target_matches

    def is_metadata_by_page(self, rel: Any, book_title: str, total_pages: int) -> bool:
        """Layer 3: Check if page number + book title indicates front/back matter"""
        page = rel.evidence.get('page_number', 0)

        # Check if in front or back matter
        in_front_matter = page > 0 and page <= self.front_matter_threshold
        in_back_matter = total_pages > 0 and page > (total_pages - self.back_matter_threshold)

        if not (in_front_matter or in_back_matter):
            return False

        # Check if relationship involves book title
        book_title_lower = book_title.lower()
        source_has_title = book_title_lower in rel.source_entity.lower()
        target_has_title = book_title_lower in rel.target_entity.lower()

        return source_has_title or target_has_title

    def is_metadata_combined(self, rel: Any, book_title: str) -> bool:
        """Layer 4: Combined heuristic - book title + person name on early page"""
        page = rel.evidence.get('page_number', 0)

        # Only check early pages (likely front matter)
        if page > 20:
            return False

        book_title_lower = book_title.lower()

        # Check if one entity is book title and other is a person name
        source_is_title = book_title_lower in rel.source_entity.lower()
        target_is_title = book_title_lower in rel.target_entity.lower()

        # Person name heuristic: capitalized words (likely proper nouns)
        source_is_person = bool(re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', rel.source_entity))
        target_is_person = bool(re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', rel.target_entity))

        # If book title + person name on early page → likely metadata
        return (
            (source_is_title and target_is_person) or
            (target_is_title and source_is_person)
        )

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Filter metadata relationships, preserving only domain knowledge"""

        # Reset stats and filtered storage
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0
        self.filtered_metadata = []

        # Get book metadata from context
        book_title = context.document_metadata.get('title', '')
        total_pages = context.document_metadata.get('total_pages', 0)

        if not book_title:
            logger.warning(f"   {self.name}: No book title in metadata, skipping filtering")
            return relationships

        # Process relationships
        kept = []
        filtered_count = 0

        for rel in relationships:
            # Check all detection layers
            is_metadata = (
                self.is_metadata_by_flag(rel) or
                self.is_metadata_by_predicate(rel, book_title) or
                self.is_metadata_by_page(rel, book_title, total_pages) or
                self.is_metadata_combined(rel, book_title)
            )

            if is_metadata:
                # Mark as filtered
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['METADATA_FILTERED'] = True
                rel.flags['filter_reason'] = 'Book metadata (not domain knowledge)'

                # Store in metadata collection
                self.filtered_metadata.append(rel)
                filtered_count += 1
                self.stats['modified_count'] += 1

                logger.debug(
                    f"Filtered metadata: ({rel.source_entity}) --[{rel.relationship}]--> "
                    f"({rel.target_entity}) on page {rel.evidence.get('page_number', '?')}"
                )
            else:
                # Keep relationship (domain knowledge)
                kept.append(rel)

        # Update stats
        self.stats['filtered'] = filtered_count
        self.stats['metadata_count'] = len(self.filtered_metadata)

        logger.info(
            f"   {self.name}: {filtered_count} metadata relationships filtered, "
            f"{len(kept)} domain relationships kept"
        )

        return kept

    def get_filtered_metadata(self) -> List[Any]:
        """
        Get filtered metadata relationships for potential later use.

        Returns:
            List of filtered metadata relationships
        """
        return self.filtered_metadata.copy()
