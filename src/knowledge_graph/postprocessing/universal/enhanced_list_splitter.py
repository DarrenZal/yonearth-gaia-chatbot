"""
Enhanced List Splitter Module for Entity Processing

This module detects and splits entity names that are actually lists of multiple entities.
For example, "United States, China, France, Brazil" should be 4 separate PLACE entities.

This is distinct from the existing list_splitter.py which operates on RELATIONSHIPS
(splitting list targets into individual relationships). This module operates on ENTITIES.

Work Chunks: 2.1, 2.2, 2.3 from KNOWLEDGE_GRAPH_EXTRACTION_REVIEW.md

Features:
- Detect list entities (2+ commas, Oxford comma patterns, "and" patterns)
- Split list entities into individual entities while preserving metadata
- Split compound person names ("X with Y", "John and Jane Smith")
- Provenance tracking for split entities

Version History:
- v1.0.0: Initial implementation (Work Chunks 2.1, 2.2, 2.3)
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)


# Patterns that should NOT be treated as lists (legitimate single entities)
PROTECTED_PATTERNS = [
    # Academic/professional titles with suffixes
    r'^(Dr\.|Prof\.|Rev\.|Mr\.|Mrs\.|Ms\.)\s+.+,\s*(Jr\.|Sr\.|Ph\.?D\.?|M\.?D\.?|M\.?A\.?|Esq\.)$',
    # Names with suffix
    r'^[A-Z][a-z]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+,\s*(Jr\.|Sr\.|II|III|IV)$',
    # City, State/Country patterns (only 1 comma)
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)?,\s*[A-Z][a-z]+(\s+[A-Z][a-z]+)?$',
    # Academic credentials with multiple commas but single person
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*,\s*(Ph\.?D\.?,?\s*)?(M\.?D\.?,?\s*)?(M\.?B\.?A\.?,?\s*)?(J\.?D\.?,?\s*)?$',
]

# Compile patterns for efficiency
PROTECTED_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in PROTECTED_PATTERNS]

# Common compound terms that look like lists but are single concepts
COMPOUND_TERMS = {
    'research and development',
    'trial and error',
    'bread and butter',
    'supply and demand',
    'law and order',
    'give and take',
    'back and forth',
    'peace and quiet',
    'life and death',
    'body and soul',
    'ups and downs',
    'black and white',
    'mom and dad',
    'mother and father',
    'husband and wife',
    'brother and sister',
    'men and women',
    'boys and girls',
    'young and old',
}


def is_list_entity(name: str) -> bool:
    """
    Detect if an entity name is actually a list of multiple entities.

    Detection logic:
    - 2+ commas indicates likely list
    - "X, Y, and Z" pattern (Oxford comma)
    - "X, Y and Z" pattern (no Oxford comma)

    Examples that SHOULD return True:
    - "United States, China, France, Brazil" -> True (4 countries)
    - "Albert Einstein, Richard Nixon, Eisenhower" -> True (3 people)
    - "Glasgow, Paris, Copenhagen" -> True (3 cities)
    - "A, B, and C" -> True (Oxford comma pattern)
    - "A, B and C" -> True (no Oxford comma)

    Examples that SHOULD return False:
    - "Y on Earth Community" -> False (organization name)
    - "San Francisco, California" -> False (city, state - only 1 comma, legitimate format)
    - "Dr. Martin Luther King, Jr." -> False (name with suffix)
    - "Dr. Robert Cloninger, Ph.D., M.D." -> False (academic credentials)

    Args:
        name: Entity name to check

    Returns:
        True if the name appears to be a list of entities, False otherwise
    """
    if not name or not isinstance(name, str):
        return False

    name = name.strip()

    # Check if it matches protected patterns first
    for pattern in PROTECTED_PATTERNS_COMPILED:
        if pattern.match(name):
            return False

    # Check for compound terms
    if name.lower() in COMPOUND_TERMS:
        return False

    comma_count = name.count(',')

    # 2+ commas is almost certainly a list
    if comma_count >= 2:
        # Exception: Academic credentials like "Dr. Robert Cloninger, Ph.D., M.D."
        # These typically have periods near the commas
        academic_pattern = r',\s*(?:Ph\.?D\.?|M\.?D\.?|M\.?A\.?|M\.?B\.?A\.?|J\.?D\.?|Jr\.?|Sr\.?|II|III|IV)'
        if re.search(academic_pattern, name, re.IGNORECASE):
            # Count how many commas are academic credentials vs list separators
            academic_matches = len(re.findall(academic_pattern, name, re.IGNORECASE))
            if academic_matches >= comma_count:
                return False
        return True

    # Single comma cases
    if comma_count == 1:
        # "City, State" pattern should NOT be split
        city_state_pattern = r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)?,\s*[A-Z][a-z]+(\s+[A-Z][a-z]+)?$'
        if re.match(city_state_pattern, name):
            return False

        # "Name, Jr." or "Name, Sr." pattern should NOT be split
        suffix_pattern = r',\s*(Jr\.?|Sr\.?|II|III|IV|Esq\.?)$'
        if re.search(suffix_pattern, name, re.IGNORECASE):
            return False

    # Check for "X, Y and Z" or "X, Y, and Z" patterns (with comma before 'and')
    oxford_comma_pattern = r'.+,\s*.+,?\s+and\s+.+'
    if re.match(oxford_comma_pattern, name, re.IGNORECASE):
        return True

    # Check for "X and Y" pattern without commas - only if both parts are capitalized names
    and_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$'
    if re.match(and_pattern, name):
        # Make sure it's not a compound term
        if name.lower() not in COMPOUND_TERMS:
            return True

    return False


def split_list_entity(entity: Dict) -> List[Dict]:
    """
    Split a list entity into individual entities.

    Args:
        entity: Dict with 'name', 'type', and other metadata

    Returns:
        List of new entity dicts, each with:
        - Cleaned individual name (stripped whitespace)
        - Original entity type preserved
        - Other metadata preserved (episode_ids, sources, etc.)
        - New field: 'split_from_list' containing original entity name

    Split pattern: r',\\s*(?:and\\s+)?|\\s+and\\s+'
    This handles:
    - "A, B, C, D" -> split on commas
    - "A, B, and C" -> split on ", and "
    - "A, B and C" -> split on " and "

    Edge cases handled:
    - Empty strings after split -> filtered out
    - Whitespace-only parts -> filtered out
    - Single item after split -> return as-is (wasn't really a list)
    """
    if not entity or not isinstance(entity, dict):
        return [entity] if entity else []

    name = entity.get('name', '')
    if not name or not isinstance(name, str):
        return [entity]

    # Don't split if not detected as list
    if not is_list_entity(name):
        return [entity]

    # Split pattern: commas (optionally followed by "and") OR standalone " and "
    split_pattern = r',\s*(?:and\s+)?|\s+and\s+'
    parts = re.split(split_pattern, name, flags=re.IGNORECASE)

    # Clean up parts
    cleaned_parts = []
    for part in parts:
        part = part.strip()
        if part:  # Skip empty strings and whitespace-only
            cleaned_parts.append(part)

    # If only one part after split, wasn't really a list
    if len(cleaned_parts) <= 1:
        return [entity]

    # Create new entities from parts
    result = []
    for part in cleaned_parts:
        new_entity = entity.copy()
        new_entity['name'] = part
        new_entity['split_from_list'] = name

        # Deep copy lists/dicts to avoid mutation
        for key in ['provenance', 'aliases', 'evidence', 'sources']:
            if key in new_entity and isinstance(new_entity[key], list):
                new_entity[key] = new_entity[key].copy()
            elif key in new_entity and isinstance(new_entity[key], dict):
                new_entity[key] = new_entity[key].copy()

        result.append(new_entity)

    return result


def split_compound_entity(entity: Dict) -> List[Dict]:
    """
    Split compound person names that were incorrectly merged.

    ONLY applies to PERSON type entities.

    Patterns to detect and split:
    1. "X with Y" -> ["X", "Y"]
       Example: "Joanna Macy with Chris Johnstone" -> ["Joanna Macy", "Chris Johnstone"]

    2. "FirstName and FirstName LastName" (shared last name pattern)
       Example: "John and Jane Smith" -> ["John Smith", "Jane Smith"]

    3. "LastName, FirstName" (inverted name) -> Reorder, don't split
       Example: "Macy, Joanna" -> ["Joanna Macy"]

    Args:
        entity: Dict with 'name', 'type', and other metadata

    Returns:
        List of split entities with 'split_from' provenance field
        If no pattern matches, returns [entity] unchanged
    """
    if not entity or not isinstance(entity, dict):
        return [entity] if entity else []

    name = entity.get('name', '')
    entity_type = entity.get('type', '')

    if not name or not isinstance(name, str):
        return [entity]

    # Only apply to PERSON type entities
    if entity_type.upper() != 'PERSON':
        return [entity]

    name = name.strip()

    # Pattern 1: "X with Y" - split on "with"
    with_pattern = r'^(.+?)\s+with\s+(.+)$'
    match = re.match(with_pattern, name, re.IGNORECASE)
    if match:
        person1 = match.group(1).strip()
        person2 = match.group(2).strip()

        # Ensure both parts look like names (start with capital letter)
        if person1 and person2 and person1[0].isupper() and person2[0].isupper():
            result = []
            for person in [person1, person2]:
                new_entity = entity.copy()
                new_entity['name'] = person
                new_entity['split_from'] = name
                for key in ['provenance', 'aliases', 'evidence', 'sources']:
                    if key in new_entity and isinstance(new_entity[key], list):
                        new_entity[key] = new_entity[key].copy()
                result.append(new_entity)
            return result

    # Pattern 2: "FirstName and FirstName LastName" (shared last name)
    # Example: "John and Jane Smith" -> "John Smith", "Jane Smith"
    shared_lastname_pattern = r'^([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)$'
    match = re.match(shared_lastname_pattern, name)
    if match:
        first1 = match.group(1)
        first2 = match.group(2)
        last = match.group(3)

        result = []
        for first in [first1, first2]:
            new_entity = entity.copy()
            new_entity['name'] = f"{first} {last}"
            new_entity['split_from'] = name
            for key in ['provenance', 'aliases', 'evidence', 'sources']:
                if key in new_entity and isinstance(new_entity[key], list):
                    new_entity[key] = new_entity[key].copy()
            result.append(new_entity)
        return result

    # Pattern 3: "LastName, FirstName" (inverted name) -> reorder to "FirstName LastName"
    # This is NOT a split, just a normalization
    inverted_pattern = r'^([A-Z][a-z]+),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$'
    match = re.match(inverted_pattern, name)
    if match:
        last = match.group(1)
        first = match.group(2)

        new_entity = entity.copy()
        new_entity['name'] = f"{first} {last}"
        new_entity['split_from'] = name
        return [new_entity]

    # No pattern matched, return unchanged
    return [entity]


def process_entities_for_splitting(entities: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Process all entities, splitting lists and compounds as needed.

    Args:
        entities: List of entity dicts with 'name', 'type', and other fields

    Returns:
        (processed_entities, stats)

    Stats dict contains:
        - total_processed: int
        - list_entities_found: int
        - list_entities_split: int
        - new_entities_from_lists: int
        - compound_entities_found: int
        - compound_entities_split: int
    """
    stats = {
        'total_processed': 0,
        'list_entities_found': 0,
        'list_entities_split': 0,
        'new_entities_from_lists': 0,
        'compound_entities_found': 0,
        'compound_entities_split': 0,
    }

    if not entities:
        return [], stats

    processed = []

    for entity in entities:
        stats['total_processed'] += 1
        name = entity.get('name', '')

        # First try list splitting
        if is_list_entity(name):
            stats['list_entities_found'] += 1
            split_results = split_list_entity(entity)
            if len(split_results) > 1:
                stats['list_entities_split'] += 1
                stats['new_entities_from_lists'] += len(split_results)

                # Apply compound splitting to each result
                for split_entity in split_results:
                    compound_results = split_compound_entity(split_entity)
                    if len(compound_results) > 1:
                        stats['compound_entities_found'] += 1
                        stats['compound_entities_split'] += 1
                    processed.extend(compound_results)
                continue

        # Try compound splitting on non-list entities
        compound_results = split_compound_entity(entity)
        if len(compound_results) > 1:
            stats['compound_entities_found'] += 1
            stats['compound_entities_split'] += 1
        processed.extend(compound_results)

    return processed, stats


def analyze_list_entities(entities: Dict[str, Dict]) -> Dict:
    """
    Analyze a dictionary of entities to find list entities.

    Args:
        entities: Dictionary mapping entity names to entity data

    Returns:
        Analysis dictionary with counts and samples
    """
    list_entities = []
    compound_entities = []

    for name, data in entities.items():
        entity = {'name': name, **data}

        if is_list_entity(name):
            list_entities.append(name)

        if entity.get('type', '').upper() == 'PERSON':
            # Check compound patterns
            if ' with ' in name.lower():
                compound_entities.append(name)
            elif re.match(r'^[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+$', name):
                compound_entities.append(name)

    return {
        'total_entities': len(entities),
        'list_entities_count': len(list_entities),
        'compound_entities_count': len(compound_entities),
        'list_entity_samples': list_entities[:20],
        'compound_entity_samples': compound_entities[:20],
    }
