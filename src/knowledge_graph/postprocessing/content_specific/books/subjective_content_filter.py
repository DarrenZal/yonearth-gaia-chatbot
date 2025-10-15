"""
Subjective Content Filter Module

Filters out subjective, philosophical, and metaphorical content based on Pass 2 flags.

Features:
- Filters low-confidence philosophical claims (p_true < 0.5)
- Filters figurative/metaphorical language (p_true < 0.5)
- Filters opinion statements
- Filters semantic type mismatches

Version History:
- v1.0.0 (V14.3.1): Initial implementation for factual-only knowledge graph
"""

import logging
from typing import Optional, List, Dict, Any

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class SubjectiveContentFilter(PostProcessingModule):
    """
    Filters subjective content based on Pass 2 classification flags.

    Content Types: Books and episodes
    Priority: 15 (runs after PraiseQuoteDetector but early in pipeline)
    """

    name = "SubjectiveContentFilter"
    description = "Filters subjective/philosophical/metaphorical content"
    content_types = ["book", "episode"]
    priority = 15
    dependencies = []
    version = "1.0.0"  # V14.3.1

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Confidence threshold for subjective content (default: 0.5)
        self.p_true_threshold = self.config.get('p_true_threshold', 0.5)

        # Flags that indicate subjective content
        self.subjective_flags = self.config.get('subjective_flags', [
            'PHILOSOPHICAL_CLAIM',
            'FIGURATIVE_LANGUAGE',
            'METAPHOR'
        ])

        # Flags that always trigger filtering (regardless of confidence)
        self.always_filter_flags = self.config.get('always_filter_flags', [
            'OPINION',
            'SEMANTIC_INCOMPATIBILITY'
        ])

        # Enable/disable filtering (allows turning off via config)
        self.filter_enabled = self.config.get('filter_subjective_content', True)

    def should_filter(self, rel: Any) -> tuple[bool, str]:
        """
        Determine if a relationship should be filtered.

        Returns:
            (should_filter: bool, reason: str)
        """
        # Check if filtering is enabled
        if not self.filter_enabled:
            return False, ""

        # Extract relevant fields
        flags = rel.flags if hasattr(rel, 'flags') and rel.flags is not None else {}
        classification_flags = rel.classification_flags if hasattr(rel, 'classification_flags') and rel.classification_flags is not None else []
        p_true = rel.p_true if hasattr(rel, 'p_true') else 1.0

        # Convert classification_flags to set for faster lookup
        if isinstance(classification_flags, list):
            flag_set = set(classification_flags)
        else:
            flag_set = set()

        # Rule 1: Always filter OPINION and SEMANTIC_INCOMPATIBILITY
        for flag in self.always_filter_flags:
            if flag in flag_set or flag in flags:
                return True, f"Always-filter flag detected: {flag}"

        # Rule 2: Filter low-confidence subjective content
        if p_true < self.p_true_threshold:
            for flag in self.subjective_flags:
                if flag in flag_set or flag in flags:
                    return True, f"Low confidence ({p_true:.2f}) + subjective flag: {flag}"

        return False, ""

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process relationships to filter subjective content"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        filtered_relationships = []
        filter_count = 0
        filter_reasons = {}

        for rel in relationships:
            should_filter, reason = self.should_filter(rel)

            if should_filter:
                # Mark as filtered
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['SUBJECTIVE_CONTENT_FILTERED'] = True
                rel.flags['filter_reason'] = reason

                # Track filter reasons
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

                filter_count += 1
                self.stats['modified_count'] += 1

                # Don't add to output list (filtered out)
                logger.debug(f"Filtered relationship: {rel.source} -> {rel.relationship} -> {rel.target} | Reason: {reason}")
            else:
                # Keep relationship
                filtered_relationships.append(rel)

        # Update stats
        self.stats['filtered_count'] = filter_count
        self.stats['filter_reasons'] = filter_reasons

        logger.info(f"   {self.name}: {filter_count} subjective relationships filtered ({len(filtered_relationships)} remaining)")

        if filter_count > 0 and logger.isEnabledFor(logging.INFO):
            logger.info(f"   Filter breakdown:")
            for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"     - {reason}: {count}")

        return filtered_relationships
