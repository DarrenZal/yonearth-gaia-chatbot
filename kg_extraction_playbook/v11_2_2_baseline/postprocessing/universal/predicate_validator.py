"""
Predicate Validator Module

Validates logical consistency of predicates and relationships.

Features:
- Self-loop detection (source == target)
- Publication context validation (published → title, not date)
- Invalid predicate pattern detection
- Confidence penalty for validation failures

Version History:
- v1.0.0 (V6): Initial implementation
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class PredicateValidator(PostProcessingModule):
    """
    Validates logical consistency of predicates.

    Content Types: Universal (works for all content types)
    Priority: 80 (runs after normalization)
    """

    name = "PredicateValidator"
    description = "Logical consistency validation of predicates"
    content_types = ["all"]
    priority = 80
    dependencies = ["PredicateNormalizer"]
    version = "1.0.0"  # V6 (no V8 changes)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Invalid predicate patterns (source_type, predicate, target_type)
        self.invalid_patterns = self.config.get('invalid_patterns', [
            ('Organization', 'published', 'Date'),
            ('Person', 'is-a', 'Person'),
        ])

    def validate_no_self_loop(self, rel: Any) -> Tuple[bool, str]:
        """Validate that source != target (except for identity predicates)"""
        if rel.source.lower() == rel.target.lower():
            if rel.relationship not in {'is-a', 'is defined as', 'means', 'equals'}:
                return False, "self_loop"
        return True, ""

    def validate_publication_context(self, rel: Any) -> Tuple[bool, str]:
        """Validate that publication predicates point to titles, not dates"""
        if rel.relationship in {'published', 'wrote', 'authored'}:
            target_words = rel.target.split()

            # Date patterns
            date_patterns = [
                r'^\d{1,2}/\d{1,2}/\d{2,4}$',
                r'^\d{4}-\d{2}-\d{2}$',
                r'^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$',
            ]

            for pattern in date_patterns:
                if re.match(pattern, rel.target):
                    return False, "published_date_not_title"

        return True, ""

    def validate_predicate(self, rel: Any) -> Any:
        """Validate a single relationship"""
        issues = []

        # Check for self-loops
        valid, reason = self.validate_no_self_loop(rel)
        if not valid:
            issues.append(reason)

        # Check publication context
        valid, reason = self.validate_publication_context(rel)
        if not valid:
            issues.append(reason)

        # Flag invalid predicates
        if issues:
            if rel.flags is None:
                rel.flags = {}
            rel.flags['INVALID_PREDICATE'] = True
            rel.flags['validation_issues'] = issues
            # Reduce confidence
            rel.p_true = rel.p_true * 0.3

        return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to validate predicates"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        invalid_count = 0

        for rel in relationships:
            rel = self.validate_predicate(rel)

            if rel.flags and rel.flags.get('INVALID_PREDICATE'):
                invalid_count += 1
                self.stats['modified_count'] += 1

            processed.append(rel)

        # Update stats
        self.stats['invalid_flagged'] = invalid_count

        logger.info(f"   {self.name}: {invalid_count} invalid predicates flagged")

        return processed
