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
- v1.3.0 (V12): "is X for" patterns, absolute predicate moderation, "is made" standardization
- v1.4.0 (V14): Tense normalization, modal verb preservation, enhanced semantic validation
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
    description = "V14: Modal verb preservation, tense normalization, semantic validation"
    content_types = ["all"]
    priority = 70
    dependencies = []
    version = "1.4.0"  # V14 enhanced

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
            # V14 REMOVED: DO NOT strip modal verbs (epistemic uncertainty)
            # 'can work together to': 'works with',  # ❌ REMOVED - strips 'can'
            # 'can cultivate': 'cultivates',  # ❌ REMOVED - strips 'can'
            'have the opportunity to': 'can',
            'get to choose to': 'can',
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

            # V12 NEW: "is X for" pattern normalization
            'is key for': 'is essential for',
            'is required for': 'requires',
            'is critical for': 'is essential for',
            'is important for': 'is essential for',
            'is necessary for': 'requires',

            # V12 NEW: Absolute predicate moderation
            'reverses': 'can help mitigate',
            'eliminates': 'can help reduce',
            'solves': 'can help address',
            'is the answer to': 'can help address',
            'is the solution to': 'can help address',
            'prevents': 'can help prevent',

            # V12 NEW: "is made" pattern standardization
            'is made of': 'is made from',
            'is made by': 'is made from',
            'is made with': 'is made from',

            # V14 NEW: Tense normalization
            'has preserved': 'preserved',
            'has enabled': 'enabled',
            'has enhanced': 'enhanced',
            'has improved': 'improved',
            'has created': 'created',
            'has developed': 'developed',
            'is produced': 'produces',
            'are produced': 'produces',
            'is provided': 'provides',
            'are provided': 'provides',

            # V14 NEW: 'is-X' variant consolidation
            'is about': 'relates to',
            'are about': 'relates to',
            'is characterized by': 'characterized by',  # Keep as-is, semantic meaning preserved
            'are characterized by': 'characterized by',
            'is composed of': 'composed of',
            'are composed of': 'composed of',
            'is comprised of': 'comprises',
            'are comprised of': 'comprises',
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

    def validate_semantic_compatibility(
        self,
        source: str,
        predicate: str,
        target: str,
        source_type: str,
        target_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        V14 NEW: Validate semantic compatibility between source, predicate, and target.

        Checks:
        1. Abstract source + physical predicate → FLAG
        2. 'is-a' with incompatible types → REJECT

        Returns:
            (is_valid, error_reason or None)
        """
        predicate_lower = predicate.lower().strip()

        # Check 1: Abstract source + physical predicate
        abstract_indicators = ['concept', 'idea', 'principle', 'philosophy', 'approach', 'method']
        physical_predicates = ['builds', 'constructs', 'plants', 'digs', 'harvests', 'produces physically']

        source_is_abstract = any(indicator in source.lower() for indicator in abstract_indicators)
        predicate_is_physical = any(phys in predicate_lower for phys in physical_predicates)

        if source_is_abstract and predicate_is_physical:
            return False, f"Abstract source '{source}' with physical predicate '{predicate}'"

        # Check 2: 'is-a' type compatibility
        if predicate_lower in ['is-a', 'is a', 'are', 'is']:
            # PERSON is-a ORGANIZATION → invalid
            # CONCEPT is-a PHYSICAL_OBJECT → invalid
            if source_type == 'Person' and 'organization' in target.lower():
                return False, f"Type mismatch: Person '{source}' cannot be-a Organization '{target}'"

            if source_type == 'Organization' and any(word in target.lower() for word in ['person', 'individual', 'people']):
                return False, f"Type mismatch: Organization '{source}' cannot be-a Person '{target}'"

            # Check for category mismatches
            abstract_types = ['concept', 'principle', 'idea', 'philosophy', 'approach', 'method']
            physical_types = ['tool', 'equipment', 'machine', 'device', 'plant', 'animal']

            source_is_abstract_type = any(atype in source.lower() for atype in abstract_types)
            target_is_physical_type = any(ptype in target.lower() for ptype in physical_types)

            if source_is_abstract_type and target_is_physical_type:
                return False, f"Type mismatch: Abstract '{source}' cannot be-a Physical '{target}'"

            if not source_is_abstract_type and target_is_physical_type:
                # Physical can be physical - that's okay
                pass

        return True, None

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """V14 ENHANCEMENT: Added semantic validation and modal verb preservation"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        normalized_count = 0
        semantically_corrected_count = 0
        semantically_invalid_count = 0

        for rel in relationships:
            predicate = rel.relationship.lower()
            source = rel.source
            target = rel.target

            # V8 NEW: Detect entity types
            source_type = self._detect_entity_type(source)
            target_type = self._detect_entity_type(target)

            # V14 NEW: Semantic compatibility validation
            is_valid, error_reason = self.validate_semantic_compatibility(
                source, rel.relationship, target,
                source_type or '', target_type or ''
            )

            if not is_valid:
                # Flag as semantically invalid
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['SEMANTIC_INCOMPATIBILITY'] = True
                rel.flags['incompatibility_reason'] = error_reason
                semantically_invalid_count += 1
                logger.debug(f"  ⚠️  Semantic incompatibility: {error_reason}")

            # V8 EXISTING: Check entity type predicates
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
        self.stats['semantically_invalid'] = semantically_invalid_count

        logger.info(
            f"   {self.name} (V14 enhanced): {normalized_count} normalized, "
            f"{semantically_corrected_count} corrected, {semantically_invalid_count} semantic issues flagged"
        )

        return processed
