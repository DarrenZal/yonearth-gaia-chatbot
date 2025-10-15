"""
Deduplication Module

Removes duplicate relationships based on normalized (source, relationship, target) tuples.

Features:
- Case-insensitive deduplication
- Whitespace normalization
- Preserves first occurrence of each unique relationship
- Flags removed duplicates for transparency

Version History:
- v1.0.0 (V11.2): Initial implementation
"""

import logging
from typing import List, Set, Tuple, Dict, Any

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class Deduplicator(PostProcessingModule):
    """
    Removes duplicate relationships from the knowledge graph.

    Content Types: All
    Priority: 110 (runs last, after all other transformations)
    """

    name = "Deduplicator"
    description = "Remove duplicate relationships based on normalized tuples"
    content_types = ["book", "podcast", "article"]
    priority = 110  # Run last
    dependencies = []  # No dependencies - should run after everything
    version = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Configuration
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)

    def normalize_string(self, s: str) -> str:
        """Normalize string for comparison"""
        if self.normalize_whitespace:
            s = ' '.join(s.split())  # Normalize whitespace

        if not self.case_sensitive:
            s = s.lower()

        return s.strip()

    def get_relationship_tuple(self, rel: Any) -> Tuple[str, str, str]:
        """
        Extract normalized (source, relationship, target) tuple for deduplication.

        Args:
            rel: Relationship object

        Returns:
            Tuple of (normalized_source, normalized_relationship, normalized_target)
        """
        source = self.normalize_string(rel.source)
        relationship = self.normalize_string(rel.relationship)
        target = self.normalize_string(rel.target)

        return (source, relationship, target)

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """
        Remove duplicate relationships from batch.

        Args:
            relationships: List of relationship objects
            context: Processing context

        Returns:
            List of unique relationships (duplicates removed)
        """

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        seen: Set[Tuple[str, str, str]] = set()
        unique_relationships: List[Any] = []
        duplicates_removed = 0

        for rel in relationships:
            # Get normalized tuple
            rel_tuple = self.get_relationship_tuple(rel)

            if rel_tuple not in seen:
                # First occurrence - keep it
                seen.add(rel_tuple)
                unique_relationships.append(rel)
            else:
                # Duplicate - skip it but log
                duplicates_removed += 1

                # Optionally flag the duplicate (not added to results)
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['DUPLICATE_REMOVED'] = True
                rel.flags['duplicate_tuple'] = rel_tuple

        # Update stats
        self.stats['duplicates_removed'] = duplicates_removed
        self.stats['unique_relationships'] = len(unique_relationships)
        self.stats['modified_count'] = duplicates_removed

        logger.info(
            f"   {self.name}: Removed {duplicates_removed} duplicates "
            f"({len(relationships)} → {len(unique_relationships)})"
        )

        return unique_relationships
