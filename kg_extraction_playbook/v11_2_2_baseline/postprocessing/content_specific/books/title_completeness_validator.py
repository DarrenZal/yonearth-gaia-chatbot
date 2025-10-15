"""
Title Completeness Validator Module

Validates book titles for completeness in authorship relationships.

Features:
- Bad ending detection (ending with prepositions, conjunctions)
- Unmatched quote detection
- Ellipsis ending detection
- Too-short title detection
- Confidence penalty for incomplete titles

Version History:
- v1.0.0 (V6): Initial implementation
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class TitleCompletenessValidator(PostProcessingModule):
    """
    Validates book titles for completeness in authorship relationships.

    Content Types: Books only
    Priority: 90 (runs late, after most processing)
    """

    name = "TitleCompletenessValidator"
    description = "Validates book title completeness"
    content_types = ["book"]
    priority = 90
    dependencies = []
    version = "1.0.0"  # V6 (no V8 changes)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.bad_endings = self.config.get('bad_endings', {
            'and', 'or', 'but', 'to', 'for', 'with', 'by',
            'in', 'on', 'at', 'of', 'the', 'a', 'an'
        })

        self.title_relationships = self.config.get('title_relationships', {
            'authored', 'wrote', 'published', 'edited',
            'compiled', 'created', 'produced', 'endorsed'
        })

        self.confidence_penalty = self.config.get('confidence_penalty', 0.7)

    def is_incomplete_title(self, title: str) -> Tuple[bool, str]:
        """
        Check if title appears incomplete.

        Returns:
            (is_incomplete: bool, reason: str)
        """
        words = title.split()

        # Check for bad endings (prepositions, conjunctions, articles)
        if words:
            last_word = words[-1].lower().rstrip('.,!?')
            if last_word in self.bad_endings:
                return True, f"ends_with_{last_word}"

        # Check for unmatched quotes
        if title.count('"') == 1:
            return True, "unmatched_quotes"

        # Check for too-short titles
        if len(words) <= 2 and ':' not in title:
            return True, "too_short"

        # Check for ellipsis endings
        if title.rstrip().endswith('...'):
            return True, "ellipsis_ending"

        return False, ""

    def validate_relationship(self, rel: Any) -> Any:
        """Validate a single relationship for title completeness"""
        if rel.relationship not in self.title_relationships:
            return rel

        is_incomplete, reason = self.is_incomplete_title(rel.target)

        if is_incomplete:
            if rel.flags is None:
                rel.flags = {}
            rel.flags['INCOMPLETE_TITLE'] = True
            rel.flags['incompleteness_reason'] = reason
            # Reduce confidence
            rel.p_true = rel.p_true * self.confidence_penalty

        return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to validate title completeness"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        incomplete_count = 0

        for rel in relationships:
            rel = self.validate_relationship(rel)

            if rel.flags and rel.flags.get('INCOMPLETE_TITLE'):
                incomplete_count += 1
                self.stats['modified_count'] += 1

            processed.append(rel)

        # Update stats
        self.stats['incomplete_flagged'] = incomplete_count

        logger.info(f"   {self.name}: {incomplete_count} incomplete titles flagged")

        return processed
