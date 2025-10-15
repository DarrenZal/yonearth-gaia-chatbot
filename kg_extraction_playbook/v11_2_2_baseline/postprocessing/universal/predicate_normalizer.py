"""
Predicate Normalizer Module

Normalizes verbose/awkward predicates to standard forms.

Features:
- Mapping of verbose predicates to concise standard forms
- Semantic validation against entity types (V8)
- Entity type detection (Books, Persons, etc.)
- Forbidden predicate replacement (e.g., Books can't "heal", but can "guide")
- Comprehensive predicate reduction: 173 → ~80 (V11.2.2)

Version History:
- v1.0.0 (V6): Basic predicate normalization
- v1.1.0 (V8): Semantic validation against entity types
- v1.2.0 (V11.2.2): Comprehensive mappings (173 → ~80 unique predicates)
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class PredicateNormalizer(PostProcessingModule):
    """
    Normalizes verbose predicates and validates semantic compatibility.

    Content Types: Universal (works for all content types)
    Priority: 70 (mid-pipeline, after pronoun resolution)
    """

    name = "PredicateNormalizer"
    description = "Comprehensive predicate normalization (173 → ~80)"
    content_types = ["all"]
    priority = 70
    dependencies = []
    version = "1.2.0"  # V11.2.2 enhanced

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # V11.2.2 ENHANCED: Comprehensive mapping of predicates (173 → ~80)
        self.predicate_mappings = self.config.get('predicate_mappings', {
            # Original V8 mappings
            'flourish with': 'experience',
            'flourishes with': 'experience',
            'have the choice to': 'can',
            'has the choice to': 'can',
            'is wedded to': 'depends on',
            'are wedded to': 'depends on',
            'unlock the door to': 'enables',
            'unlocks the door to': 'enables',
            'make it possible to': 'enables',
            'makes it possible to': 'enables',

            # V11.2.2 NEW: Variations of "is" (normalize to is-a)
            'is': 'is-a',
            'are': 'is-a',
            'is a': 'is-a',

            # V11.2.2 NEW: Variations of "enhance"
            'will enhance': 'enhances',
            'will enhance and restore': 'enhances',
            'are enhanced': 'enhances',
            'are boosted': 'enhances',

            # V11.2.2 NEW: Variations of "help/heal"
            'helps to alleviate': 'helps',
            'helps to heal': 'helps',
            'heals': 'helps',
            'will heal': 'helps',

            # V11.2.2 NEW: Variations of "include"
            'include': 'includes',

            # V11.2.2 NEW: Variations of "increase"
            'increased': 'increases',
            'increased by': 'increases',
            'increases by': 'increases',

            # V11.2.2 NEW: Variations of "reduce"
            'are reduced': 'reduces',

            # V11.2.2 NEW: Variations of "reverse"
            'is reversed by': 'reverses',

            # V11.2.2 NEW: Verbose predicates simplified
            'is a practice that involves': 'involves',
            'is essential reading for': 'is essential for',
            'is essential to': 'is essential for',
            'is key to': 'is key for',
            'is needed for': 'requires',
            'are required for': 'requires',
            'are needed to': 'requires',
            'means to provide': 'provides',
            'should always be deployed with an ethos of': 'should embody',
            'should always ask ourselves': 'prompts',
            'should conclude that': 'implies',
            'is a commitment to': 'embodies',
            'is what it means to be': 'is-a',

            # V11.2.2 NEW: Future tense → present tense
            'will embody': 'embodies',
            'can work together to': 'works with',
            'have the opportunity to': 'can',
            'get to choose to': 'can',
            'can cultivate': 'cultivates',
            'must regenerate': 'regenerates',
            'will collapse and die': 'degrades',

            # V11.2.2 NEW: Publication-related normalization
            'is released by': 'produced by',
            'releases': 'produced by',

            # V11.2.2 NEW: Specific verbose phrases
            'is a spiritual system of': 'embodies',
            'is the foundation of': 'is key for',
            'is at the core of': 'is key for',
            'is dependent on': 'depends on',
            'opens doors to': 'enables',
            'is a powerful way to': 'enables',
            'is a practice': 'involves',
            'is a complex system': 'is-a',
            'is a group that encourages': 'encourages',
            'is to help': 'helps',
            'is doing': 'does',
            'is equivalent to': 'equals',
            'is equal to': 'equals',

            # V11.2.2 NEW: Special actions
            'super-charges': 'enhances',
            'amplified by': 'enhanced by',
            'locks up': 'sequesters',
            'turns into': 'becomes',

            # V11.2.2 NEW: Attribution variations
            'is authored by': 'authored by',
            'were developed by': 'developed by',
            'made by': 'created by',
            'was created by': 'created by',
            'is created from': 'created from',

            # V11.2.2 NEW: Engagement/participation
            'can engage in': 'engages in',
            'are embarking on': 'engages in',
            'have been conducting': 'conducts',

            # V11.2.2 NEW: Questions/purpose
            'is not just': 'is',
            'promise to be a': 'is',

            # V11.2.2 NEW: Location/status
            'are at': 'located in',
            'are from': 'from',
            'is in': 'located in',
            'resides in': 'located in',
            'planted in': 'located in',
            'hails from': 'from',

            # V11.2.2 NEW: Envisioning/planning
            'are envisioned in': 'planned for',
            'are envisioned to': 'planned for',
            'should be established at': 'planned for',
            'are especially needed for': 'needed for',
            'are mobilizing and scaling in': 'active in',
        })

        # V8 NEW: Entity type constraints for semantic validation
        self.entity_type_predicates = self.config.get('entity_type_predicates', {
            'Book': {
                'allowed': ['guides', 'informs', 'describes', 'explains', 'teaches',
                           'provides', 'presents', 'covers', 'discusses', 'authored by'],
                'forbidden': ['heals', 'cures', 'fixes', 'repairs', 'treats'],
                'replacements': {
                    'heals': 'guides readers to heal',
                    'helps heal': 'provides guidance for healing',
                    'cures': 'provides information about curing',
                    'fixes': 'provides solutions for'
                }
            },
            'Person': {
                'allowed': ['wrote', 'authored', 'created', 'founded', 'established',
                           'teaches', 'researches', 'studies', 'works on'],
                'forbidden': [],
                'replacements': {}
            }
        })

    def _detect_entity_type(self, entity: str) -> Optional[str]:
        """V8 NEW: Detect entity type from entity string"""
        entity_lower = entity.lower()

        # Book detection
        if any(keyword in entity_lower for keyword in ['handbook', 'book', 'guide', 'manual']):
            return 'Book'

        # Person detection (capitalized, contains name patterns)
        if entity and entity[0].isupper() and ' ' in entity:
            return 'Person'

        return None

    def normalize_predicate(self, predicate: str) -> Tuple[str, bool]:
        """
        Normalize a predicate to standard form.

        Returns:
            (normalized_predicate, was_normalized: bool)
        """
        predicate_lower = predicate.lower().strip()

        if predicate_lower in self.predicate_mappings:
            return self.predicate_mappings[predicate_lower], True

        return predicate, False

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """V8 ENHANCEMENT: Added semantic validation"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        normalized_count = 0
        semantically_corrected_count = 0

        for rel in relationships:
            predicate = rel.relationship.lower()
            source = rel.source

            # V8 NEW: Detect entity types and check semantic compatibility
            source_type = self._detect_entity_type(source)

            if source_type and source_type in self.entity_type_predicates:
                rules = self.entity_type_predicates[source_type]

                # Check if predicate is forbidden for this entity type
                if any(forbidden in predicate for forbidden in rules['forbidden']):
                    # Try to find replacement
                    for forbidden, replacement in rules['replacements'].items():
                        if forbidden in predicate:
                            if rel.flags is None:
                                rel.flags = {}
                            rel.flags['PREDICATE_SEMANTICALLY_CORRECTED'] = True
                            rel.flags['original_relationship'] = rel.relationship
                            rel.relationship = replacement
                            semantically_corrected_count += 1
                            self.stats['modified_count'] += 1
                            break

            # Apply standard normalization
            normalized_pred, was_normalized = self.normalize_predicate(rel.relationship)

            if was_normalized:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['PREDICATE_NORMALIZED'] = True
                rel.flags['original_predicate'] = rel.relationship
                rel.relationship = normalized_pred
                normalized_count += 1
                self.stats['modified_count'] += 1

            processed.append(rel)

        # Update stats
        self.stats['normalized'] = normalized_count
        self.stats['semantically_corrected'] = semantically_corrected_count

        logger.info(
            f"   {self.name} (V8 enhanced): {normalized_count} predicates normalized, "
            f"{semantically_corrected_count} semantically corrected"
        )

        return processed
