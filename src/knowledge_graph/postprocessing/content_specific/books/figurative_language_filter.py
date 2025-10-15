"""
Figurative Language Filter Module

Flags and normalizes metaphorical language in relationships.

Features:
- Metaphorical term detection (sacred, magical, spiritual, etc.)
- Abstract noun detection (compass, journey, gateway, etc.)
- Metaphor normalization to literal equivalents (V8)
- Confidence penalty for remaining figurative language

Version History:
- v1.0.0 (V6): Basic metaphor flagging
- v1.1.0 (V8): Metaphor normalization to literal equivalents
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class FigurativeLanguageFilter(PostProcessingModule):
    """
    Flags and normalizes metaphorical language in relationships.

    Content Types: Books only (though could work for other content)
    Priority: 100 (runs last, final cleanup)
    """

    name = "FigurativeLanguageFilter"
    description = "Metaphor normalization to literal equivalents"
    content_types = ["book"]
    priority = 100
    dependencies = []
    version = "1.1.0"  # V8 enhanced

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.metaphorical_terms = self.config.get('metaphorical_terms', {
            'sacred', 'magic', 'magical', 'mystical', 'spiritual',
            'alchemy', 'divine', 'blessed', 'holy', 'sanctity',
            'touch of god', "god's touch", 'miracle', 'miraculous',
            'soul', 'spirit', 'essence', 'nexus'
        })

        self.abstract_nouns = self.config.get('abstract_nouns', {
            'compass', 'journey', 'quest', 'adventure', 'path',
            'gateway', 'portal', 'door', 'key', 'bridge'
        })

        # V8 NEW: Metaphorical predicate mappings to literal equivalents
        self.metaphor_normalizations = self.config.get('metaphor_normalizations', {
            'is a road-map': 'provides guidance',
            'is a compass': 'provides direction',
            'is a guide': 'provides guidance',
            'is wedded to': 'depends on',
            'is tied to': 'depends on',
            'is connected to': 'relates to',
            'road-map of sorts': 'guide',
            'compass for': 'guide for'
        })

        self.confidence_penalty = self.config.get('confidence_penalty', 0.6)

    def contains_metaphorical_language(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text contains metaphorical language.

        Returns:
            (contains_metaphor: bool, found_terms: List[str])
        """
        text_lower = text.lower()
        found_terms = []

        # Check metaphorical terms
        for term in self.metaphorical_terms:
            if term in text_lower:
                found_terms.append(term)

        # Check abstract nouns used metaphorically
        for noun in self.abstract_nouns:
            if f"is a {noun}" in text_lower or f"is the {noun}" in text_lower:
                found_terms.append(f"metaphor:{noun}")

        return len(found_terms) > 0, found_terms

    def filter_relationship(self, rel: Any) -> Any:
        """Flag relationship with metaphorical language"""
        is_metaphorical, terms = self.contains_metaphorical_language(rel.evidence_text)

        if is_metaphorical:
            if rel.flags is None:
                rel.flags = {}
            rel.flags['FIGURATIVE_LANGUAGE'] = True
            rel.flags['metaphorical_terms'] = terms
            # Reduce confidence
            rel.p_true = rel.p_true * self.confidence_penalty

        return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """V8 ENHANCEMENT: Normalize metaphors to literal equivalents"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        metaphorical_count = 0
        normalized_count = 0

        for rel in relationships:
            relationship_lower = rel.relationship.lower()
            target_lower = rel.target.lower()

            # V8 NEW: Check for metaphorical predicates and normalize
            normalized = False
            for metaphor, literal in self.metaphor_normalizations.items():
                # Check relationship
                if metaphor in relationship_lower:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['METAPHOR_NORMALIZED'] = True
                    rel.flags['original_relationship'] = rel.relationship
                    rel.relationship = literal
                    normalized = True
                    normalized_count += 1
                    self.stats['modified_count'] += 1
                    break

                # Also check target for metaphorical terms
                if metaphor in target_lower:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['METAPHOR_NORMALIZED_TARGET'] = True
                    rel.flags['original_target'] = rel.target
                    rel.target = target_lower.replace(metaphor, literal)
                    normalized = True
                    normalized_count += 1
                    self.stats['modified_count'] += 1
                    break

            # Flag remaining figurative language (unchanged from V6)
            if not normalized:
                rel = self.filter_relationship(rel)

            if rel.flags and rel.flags.get('FIGURATIVE_LANGUAGE'):
                metaphorical_count += 1

            processed.append(rel)

        # Update stats
        self.stats['normalized'] = normalized_count
        self.stats['remaining_flagged'] = metaphorical_count

        logger.info(
            f"   {self.name} (V8 enhanced): {normalized_count} metaphors normalized, "
            f"{metaphorical_count} remaining flagged"
        )

        return processed
