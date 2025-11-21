"""
DedicationNormalizer Module

Normalizes malformed dedication targets that include book title prefixes.

Common issues in front matter dedications:
- "Our Biggest Deal to Kevin Townley" → "Kevin Townley"
- "Book Title to Person Name" → "Person Name"

This module:
1. Detects dedication relationships
2. Strips book title prefixes from targets
3. Ensures proper direction (Person → dedicated to → Person)
4. Sets correct entity types

Version History:
- 1.0.0 (V14.3.8): Initial implementation to fix malformed dedication targets
"""

import re
import copy
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ...base import PostProcessingModule


class DedicationNormalizer(PostProcessingModule):
    """
    Normalizes malformed dedication targets by removing book title prefixes.

    Handles patterns like:
    - "Our Biggest Deal to Kevin Townley" → "Kevin Townley"
    - "Book Title to Person" → "Person"
    - Ensures Person → dedicated to → Person structure
    """

    name = "DedicationNormalizer"
    description = "Normalizes malformed dedication targets (removes book title prefixes)"
    content_types = ["book"]
    priority = 18  # Before SubtitleJoiner (19) and BibliographicCitationParser (20)
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Patterns for detecting book title prefixes in dedication targets
        # NOTE: We intentionally keep patterns strict to avoid over-matching.
        # A dynamic book-title–aware pattern is constructed at runtime in _normalize_target.
        self.dedication_patterns = [
            # Conservative heuristic with some common title keywords
            r'^([A-Z][A-Za-z\s]+(?:Deal|Book|Work|Manual|Guide|Handbook))\s+(?:to|for)\s+(.+)$',
            # Very conservative fallback for explicit leading 'to '
            r'^to\s+(.+)$',
        ]

        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dedication_patterns]

    def process_batch(self, relationships: List[Any], context: Any) -> List[Any]:
        """Process batch of relationships to normalize dedications

        Always returns a FLAT list of relationship objects (never tuples/nested lists).
        Module statistics are stored in self.stats for the orchestrator to collect.
        """
        processed: List[Any] = []

        # Reset and accumulate stats
        self.stats['targets_normalized'] = 0
        self.stats['patterns_matched'] = {}
        self.stats['dedications_processed'] = 0

        for rel in relationships:
            try:
                # Only process dedication relationships
                if not self._is_dedication(rel):
                    processed.append(rel)
                    continue

                self.stats['dedications_processed'] += 1

                # Normalize target if needed (uses context for book title matching)
                normalized_rel = self._normalize_target(rel, self.stats, context)
                processed.append(normalized_rel)
            except Exception:
                # On any failure, preserve original relationship to avoid unintended drops
                processed.append(rel)

        # modified_count: number of relationships that were changed
        self.stats['modified_count'] = self.stats.get('targets_normalized', 0)
        self.stats['processed_count'] = len(relationships)

        return processed

    def _is_dedication(self, rel: Any) -> bool:
        """Check if relationship is a dedication"""
        relationship = getattr(rel, 'relationship', '').lower()
        return 'dedicat' in relationship

    def _normalize_target(self, rel: Any, stats: Dict[str, Any], context: Any) -> Any:
        """Normalize dedication target by removing book title prefix"""
        target = getattr(rel, 'target', '')

        # Check if already normalized
        if hasattr(rel, 'flags') and rel.flags and rel.flags.get('DEDICATION_NORMALIZED'):
            return rel

        # 0) Try dynamic book-title–aware pattern if title available
        book_title = None
        try:
            book_title = (getattr(context, 'document_metadata', {}) or {}).get('title')
        except Exception:
            book_title = None

        if book_title and isinstance(book_title, str) and len(book_title) >= 4:
            bt_pattern = re.compile(rf"^({re.escape(book_title)})\s+(?:to|for)\s+(.+)$", re.IGNORECASE)
            m = bt_pattern.match(target)
            if m:
                clean_target = m.group(2).strip()
                if self._is_valid_person_name(clean_target):
                    new_rel = copy.deepcopy(rel)
                    new_rel.target = clean_target
                    new_rel.target_type = "Person"
                    if not hasattr(new_rel, 'source_type') or new_rel.source_type != "Person":
                        new_rel.source_type = "Person"
                    if new_rel.relationship.lower() in ['dedicated', 'dedicates']:
                        new_rel.relationship = 'dedicated to'
                    if not hasattr(new_rel, 'flags') or new_rel.flags is None:
                        new_rel.flags = {}
                    new_rel.flags['DEDICATION_NORMALIZED'] = True
                    new_rel.flags['original_target'] = target
                    new_rel.flags['pattern'] = 'book_title_prefix'
                    stats['targets_normalized'] += 1
                    stats['patterns_matched']['book_title_prefix'] = stats['patterns_matched'].get('book_title_prefix', 0) + 1
                    return new_rel

        # 1) Try each conservative compiled pattern
        for i, pattern in enumerate(self.compiled_patterns):
            match = pattern.match(target)

            if match:
                # Extract clean target
                if len(match.groups()) == 2:
                    # Pattern with book title and person name
                    book_title, person_name = match.groups()
                    clean_target = person_name.strip()
                elif len(match.groups()) == 1:
                    # Pattern with just person name
                    clean_target = match.group(1).strip()
                else:
                    continue

                # Validate cleaned target
                if not self._is_valid_person_name(clean_target):
                    continue

                # Create normalized relationship
                new_rel = copy.deepcopy(rel)
                new_rel.target = clean_target
                new_rel.target_type = "Person"

                # Ensure source is person and relationship is correct
                if not hasattr(new_rel, 'source_type') or new_rel.source_type != "Person":
                    new_rel.source_type = "Person"

                # Normalize relationship to "dedicated to"
                if new_rel.relationship.lower() in ['dedicated', 'dedicates']:
                    new_rel.relationship = "dedicated to"

                # Set flag
                if not hasattr(new_rel, 'flags') or new_rel.flags is None:
                    new_rel.flags = {}
                new_rel.flags['DEDICATION_NORMALIZED'] = True
                new_rel.flags['original_target'] = target
                new_rel.flags['pattern_index'] = i

                # Update stats
                stats['targets_normalized'] += 1
                pattern_key = f"pattern_{i}"
                stats['patterns_matched'][pattern_key] = stats['patterns_matched'].get(pattern_key, 0) + 1

                return new_rel

        # No match - return original
        return rel

    def _is_valid_person_name(self, name: str) -> bool:
        """
        Validate that cleaned target looks like a person name.

        Rules:
        - At least 2 characters
        - Contains at least one letter
        - Doesn't start with common non-person words
        """
        if len(name) < 2:
            return False

        if not any(c.isalpha() for c in name):
            return False

        # Check for non-person indicators
        non_person_starts = [
            'the ', 'a ', 'an ', 'this ', 'that ',
            'our biggest deal', 'book', 'work', 'essay'
        ]

        name_lower = name.lower()
        for indicator in non_person_starts:
            if name_lower.startswith(indicator):
                return False

        return True

    def format_stats(self, stats: Dict[str, Any]) -> str:
        """Format statistics for logging"""
        parts = [
            f"{stats['targets_normalized']} targets normalized"
        ]

        if stats['dedications_processed'] > 0:
            parts.append(f"{stats['dedications_processed']} dedications processed")

        if stats['patterns_matched']:
            pattern_summary = ", ".join(
                f"p{k.split('_')[1]}:{v}"
                for k, v in sorted(stats['patterns_matched'].items())
            )
            parts.append(f"patterns: {pattern_summary}")

        return ", ".join(parts)
