"""
Entity Quality Filter Module

This module provides quality filtering for entities extracted from knowledge graphs.
It blocks low-quality entities that are pronouns, generic nouns, purely numeric values,
tautological (name equals type), invalid lowercase single-word PERSON entities,
generic person patterns, and sentence-like entities.

Part of Phase 1 implementation (Work Chunks 1.1-1.7) from the Knowledge Graph
Extraction Implementation Plan.
"""

import re
from typing import Dict, List, Set, Tuple


class EntityQualityFilter:
    """
    Filter class to validate and reject low-quality entities.

    Implements seven primary filters:
    1. Stop-word entity blocker (pronouns, generic nouns)
    2. Numeric entity filter (pure numbers/years)
    3. Tautological entity filter (name equals type)
    4. Lowercase single-word PERSON filter
    5. Generic person patterns filter (descriptive phrases)
    6. Sentence-like entity filter (sentence fragments, long names)
    7. Unified filter with statistics tracking
    """

    # Work Chunk 1.1: Stop-word entities to block
    STOP_WORD_ENTITIES: Set[str] = {
        # Pronouns
        'we', 'she', 'he', 'they', 'it', 'i', 'you',
        # Generic collective nouns
        'people', 'person', 'individual', 'individuals',
        'everyone', 'someone', 'anyone', 'nobody',
        # Generic familial/social references
        'mom', 'dad', 'mother', 'father', 'friend', 'friends',
        'guy', 'woman', 'man', 'kid', 'kids',
        # Generic occupational (singular lowercase)
        'farmer', 'teacher', 'scientist', 'activist',
    }

    # Pattern for purely numeric entities (Work Chunk 1.2)
    NUMERIC_PATTERN = re.compile(r'^\d+$')

    # Work Chunk 1.5: Generic person patterns to block
    # These patterns identify descriptive phrases masquerading as PERSON entities
    GENERIC_PERSON_PATTERNS = [
        re.compile(r'^(the |a |an |our |their |my |your |his |her )', re.IGNORECASE),  # Determiner start
        re.compile(r'(friends|teachers|officials|people|generations|character|speaker)s?$', re.IGNORECASE),
        re.compile(r'^(who|which|that|those|these|some|many|few|all) ', re.IGNORECASE),
        re.compile(r'^(someone|anyone|everyone|nobody|somebody) ', re.IGNORECASE),
    ]

    # Work Chunk 1.6: Sentence-like patterns to block
    SENTENCE_PATTERNS = [
        re.compile(r'\b(is|are|was|were|has|have|had|will|would|could|should|can|may|might)\b', re.IGNORECASE),
        re.compile(r'\b(the most|in order to|according to|in terms of|as well as)\b', re.IGNORECASE),
        re.compile(r'[.!?;]'),  # Sentence punctuation
        re.compile(r',.*,.*,'),  # Multiple commas (likely a list)
    ]

    # Work Chunk 1.6: Length limits for entity names
    MAX_NAME_LENGTH = 80  # characters
    MAX_WORD_COUNT = 8    # words

    def __init__(self):
        """Initialize the entity quality filter with statistics tracking."""
        self._stats = {
            'total_checked': 0,
            'total_filtered': 0,
            'total_passed': 0,
            'reasons': {}
        }

    def is_stop_word_entity(self, name: str) -> bool:
        """
        Check if entity name is a stop word that should be blocked.

        Work Chunk 1.1: Stop-Word Entity Blocker

        Args:
            name: The entity name to check

        Returns:
            True if the entity name is a stop word (should be blocked),
            False otherwise (should be allowed)

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.is_stop_word_entity("we")
            True
            >>> filter.is_stop_word_entity("Aaron Perry")
            False
        """
        # Normalize to lowercase for comparison
        normalized_name = name.strip().lower()
        return normalized_name in self.STOP_WORD_ENTITIES

    def is_numeric_entity(self, name: str) -> bool:
        """
        Check if entity is purely numeric (years, numbers).

        Work Chunk 1.2: Numeric Entity Filter

        Args:
            name: The entity name to check

        Returns:
            True if the entity is purely numeric (should be blocked),
            False otherwise (should be allowed)

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.is_numeric_entity("2030")
            True
            >>> filter.is_numeric_entity("35")
            True
            >>> filter.is_numeric_entity("Episode 120")
            False
        """
        normalized_name = name.strip()
        return bool(self.NUMERIC_PATTERN.match(normalized_name))

    def is_tautological_entity(self, name: str, entity_type: str) -> bool:
        """
        Check if entity name essentially equals its type.

        Work Chunk 1.3: Tautological Entity Filter

        A tautological entity is one where the name is just the type itself,
        which provides no meaningful information.

        Args:
            name: The entity name to check
            entity_type: The entity type (e.g., "ORGANIZATION", "PLACE")

        Returns:
            True if the entity name is tautological (should be blocked),
            False otherwise (should be allowed)

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.is_tautological_entity("organization", "ORGANIZATION")
            True
            >>> filter.is_tautological_entity("places", "PLACE")
            True
            >>> filter.is_tautological_entity("Y on Earth", "ORGANIZATION")
            False
        """
        # Normalize both to lowercase
        normalized_name = name.strip().lower()
        normalized_type = entity_type.strip().lower()

        # Remove trailing 's' from both (handles plurals)
        if normalized_name.endswith('s'):
            normalized_name_singular = normalized_name[:-1]
        else:
            normalized_name_singular = normalized_name

        if normalized_type.endswith('s'):
            normalized_type_singular = normalized_type[:-1]
        else:
            normalized_type_singular = normalized_type

        # Check if name equals type (with or without plural)
        return (
            normalized_name == normalized_type or
            normalized_name == normalized_type_singular or
            normalized_name_singular == normalized_type or
            normalized_name_singular == normalized_type_singular
        )

    def is_invalid_lowercase_person(self, name: str, entity_type: str) -> bool:
        """
        Check if this is a generic lowercase single-word PERSON entity.

        Work Chunk 1.4: Lowercase Single-Word PERSON Filter

        Single-word lowercase PERSON entities are typically generic references
        like "mom", "friend", etc. rather than actual named individuals.

        Args:
            name: The entity name to check
            entity_type: The entity type

        Returns:
            True if this is an invalid lowercase single-word PERSON (should be blocked),
            False otherwise (should be allowed)

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.is_invalid_lowercase_person("mom", "PERSON")
            True
            >>> filter.is_invalid_lowercase_person("friend", "PERSON")
            True
            >>> filter.is_invalid_lowercase_person("Aaron", "PERSON")
            False  # Capitalized
            >>> filter.is_invalid_lowercase_person("John Smith", "PERSON")
            False  # Multi-word
        """
        # Only applies to PERSON entities
        if entity_type.upper() != 'PERSON':
            return False

        stripped_name = name.strip()

        # Check if it's a single word (no spaces)
        if ' ' in stripped_name:
            return False  # Multi-word names are allowed

        # Check if the first character is lowercase
        # Empty strings should be blocked
        if not stripped_name:
            return True

        # If the name starts with a lowercase letter, it's likely a generic reference
        if stripped_name[0].islower():
            return True

        return False

    def is_generic_person(self, name: str, entity_type: str) -> bool:
        """
        Check if this is a generic person description, not a named individual.

        Work Chunk 1.5: Generic Person Patterns Filter

        Detects descriptive phrases masquerading as PERSON entities by checking
        for determiners, generic group nouns, and indefinite references.

        Args:
            name: The entity name to check
            entity_type: The entity type

        Returns:
            True if this is a generic person description (should be blocked),
            False otherwise (should be allowed)

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.is_generic_person("the character", "PERSON")
            True
            >>> filter.is_generic_person("our friends", "PERSON")
            True
            >>> filter.is_generic_person("People from poorest countries", "PERSON")
            True
            >>> filter.is_generic_person("future generations", "PERSON")
            True
            >>> filter.is_generic_person("Dr. Jane Goodall", "PERSON")
            False
            >>> filter.is_generic_person("Aaron Perry", "PERSON")
            False
        """
        # This filter primarily applies to PERSON entities, but can catch
        # generic descriptions in any type
        stripped_name = name.strip()

        # Check against all generic person patterns
        for pattern in self.GENERIC_PERSON_PATTERNS:
            if pattern.search(stripped_name):
                return True

        return False

    def is_sentence_like(self, name: str) -> bool:
        """
        Check if entity name reads like a sentence or description.

        Work Chunk 1.6: Sentence-Like Entity Filter

        Detects entity names that contain sentence structure indicators
        like verbs, conjunctions, or sentence punctuation.

        Args:
            name: The entity name to check

        Returns:
            True if the entity name reads like a sentence (should be blocked),
            False otherwise (should be allowed)

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.is_sentence_like("the most important thing is to make...")
            True
            >>> filter.is_sentence_like("according to the latest research")
            True
            >>> filter.is_sentence_like("This is what we need to do.")
            True
            >>> filter.is_sentence_like("Aaron Perry")
            False
            >>> filter.is_sentence_like("Regenerative Agriculture")
            False
        """
        stripped_name = name.strip()

        # Check against all sentence-like patterns
        for pattern in self.SENTENCE_PATTERNS:
            if pattern.search(stripped_name):
                return True

        return False

    def exceeds_length_limits(self, name: str) -> bool:
        """
        Check if entity name is too long.

        Work Chunk 1.6: Length Limits Filter

        Entity names exceeding 80 characters or 8 words are likely descriptions
        rather than proper entity names.

        Args:
            name: The entity name to check

        Returns:
            True if the entity name exceeds length limits (should be blocked),
            False otherwise (should be allowed)

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.exceeds_length_limits("A very long description that goes on and on and on")
            True  # More than 8 words
            >>> filter.exceeds_length_limits("x" * 100)
            True  # More than 80 characters
            >>> filter.exceeds_length_limits("Aaron Perry")
            False
            >>> filter.exceeds_length_limits("Y on Earth Community")
            False
        """
        stripped_name = name.strip()

        # Check character length
        if len(stripped_name) > self.MAX_NAME_LENGTH:
            return True

        # Check word count
        word_count = len(stripped_name.split())
        if word_count > self.MAX_WORD_COUNT:
            return True

        return False

    def filter_entity(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check entity against all filters.

        Work Chunk 1.7: Unified Filter Integration

        Applies all quality filters in sequence and returns the first
        failure reason, or passes if all filters pass.

        Args:
            entity: Dict with at least 'name' key, optionally 'type'

        Returns:
            (passes_filter, rejection_reason)
            - (True, "") if entity passes all filters
            - (False, "reason") if entity should be filtered out

        Examples:
            >>> filter = EntityQualityFilter()
            >>> filter.filter_entity({"name": "Aaron Perry", "type": "PERSON"})
            (True, "")
            >>> filter.filter_entity({"name": "we", "type": "PERSON"})
            (False, "stop_word_entity")
            >>> filter.filter_entity({"name": "2030", "type": "TIME"})
            (False, "numeric_entity")
        """
        name = entity.get('name', '')
        entity_type = entity.get('type', '')

        # 1. Stop-word filter
        if self.is_stop_word_entity(name):
            return (False, "stop_word_entity")

        # 2. Numeric entity filter
        if self.is_numeric_entity(name):
            return (False, "numeric_entity")

        # 3. Tautological entity filter
        if entity_type and self.is_tautological_entity(name, entity_type):
            return (False, "tautological_entity")

        # 4. Lowercase single-word PERSON filter
        if entity_type and self.is_invalid_lowercase_person(name, entity_type):
            return (False, "invalid_lowercase_person")

        # 5. Generic person patterns filter
        if self.is_generic_person(name, entity_type):
            return (False, "generic_person_pattern")

        # 6. Sentence-like filter
        if self.is_sentence_like(name):
            return (False, "sentence_like_entity")

        # 7. Length limits filter
        if self.exceeds_length_limits(name):
            return (False, "exceeds_length_limits")

        return (True, "")

    def filter_batch(self, entities: List[Dict]) -> List[Dict]:
        """
        Filter a batch of entities, returning only those that pass.

        Work Chunk 1.7: Batch Filtering with Statistics

        Filters all entities in the batch and updates internal statistics.

        Args:
            entities: List of entity dicts with at least 'name' key

        Returns:
            List of entities that pass all filters

        Examples:
            >>> filter = EntityQualityFilter()
            >>> entities = [
            ...     {"name": "Aaron Perry", "type": "PERSON"},
            ...     {"name": "we", "type": "PERSON"},
            ...     {"name": "2030", "type": "TIME"}
            ... ]
            >>> passed = filter.filter_batch(entities)
            >>> len(passed)
            1
            >>> passed[0]["name"]
            'Aaron Perry'
        """
        passed_entities = []

        for entity in entities:
            self._stats['total_checked'] += 1

            passes, reason = self.filter_entity(entity)

            if passes:
                self._stats['total_passed'] += 1
                passed_entities.append(entity)
            else:
                self._stats['total_filtered'] += 1
                # Track rejection reasons
                if reason not in self._stats['reasons']:
                    self._stats['reasons'][reason] = 0
                self._stats['reasons'][reason] += 1

        return passed_entities

    def get_stats(self) -> Dict:
        """
        Return filtering statistics.

        Work Chunk 1.7: Statistics Reporting

        Returns:
            Dict with:
            - total_checked: int - total entities processed
            - total_filtered: int - entities that were filtered out
            - total_passed: int - entities that passed all filters
            - reasons: Dict[str, int] - count per rejection reason

        Examples:
            >>> filter = EntityQualityFilter()
            >>> _ = filter.filter_batch([{"name": "we"}, {"name": "Aaron Perry"}])
            >>> stats = filter.get_stats()
            >>> stats['total_checked']
            2
            >>> stats['total_passed']
            1
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """
        Reset statistics counters.

        Work Chunk 1.7: Statistics Management

        Clears all accumulated statistics for fresh tracking.

        Examples:
            >>> filter = EntityQualityFilter()
            >>> _ = filter.filter_batch([{"name": "we"}])
            >>> filter.get_stats()['total_checked']
            1
            >>> filter.reset_stats()
            >>> filter.get_stats()['total_checked']
            0
        """
        self._stats = {
            'total_checked': 0,
            'total_filtered': 0,
            'total_passed': 0,
            'reasons': {}
        }
