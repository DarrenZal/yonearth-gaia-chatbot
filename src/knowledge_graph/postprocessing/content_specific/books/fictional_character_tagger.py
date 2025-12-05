"""
Fictional Character Tagger

Tags entities as fictional when they originate from narrative sources like
"Our Biggest Deal" (also known as "Viriditas").

This module helps isolate fictional characters from real people/entities
in the knowledge graph, addressing the problem where characters like
Leo, Sophia, and Brigitte dominate hub metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class FictionalCharacterTagger:
    """Tags entities as fictional when they originate from narrative sources."""

    def __init__(self, registry_path: Path = None):
        """
        Initialize the tagger with a fictional character registry.

        Args:
            registry_path: Path to the fictional_characters.json file.
                          If None, uses default location in data/ directory.
        """
        if registry_path is None:
            # Default path relative to project root
            registry_path = Path(__file__).parents[5] / "data" / "fictional_characters.json"

        self.registry_path = registry_path
        self.registry = self._load_registry(registry_path)
        self._build_alias_lookup()

        # Statistics tracking
        self._stats = {
            "total_checked": 0,
            "tagged_fictional": 0,
            "by_type": {},
            "by_character": {}
        }

    def _load_registry(self, path: Path) -> Dict:
        """
        Load the fictional character registry from JSON file.

        Args:
            path: Path to the registry JSON file.

        Returns:
            Dictionary containing the registry data.

        Raises:
            FileNotFoundError: If the registry file doesn't exist.
            json.JSONDecodeError: If the registry file is invalid JSON.
        """
        if not path.exists():
            raise FileNotFoundError(f"Fictional character registry not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_alias_lookup(self) -> None:
        """
        Build lookup dictionaries mapping all aliases to canonical names.

        Creates:
            - _alias_to_canonical: Maps lowercase alias -> canonical character key
            - _fictional_names: Set of all known fictional names (lowercase)
            - _source_identifiers: Set of source IDs indicating narrative fiction
            - _character_info: Maps canonical key -> full character info
        """
        self._alias_to_canonical: Dict[str, str] = {}
        self._fictional_names: Set[str] = set()
        self._character_info: Dict[str, Dict] = {}

        # Build character lookup from main characters
        for char_key, char_data in self.registry.get("characters", {}).items():
            self._character_info[char_key] = char_data

            # Add all aliases (lowercase for case-insensitive matching)
            for alias in char_data.get("aliases", []):
                alias_lower = alias.lower()
                self._alias_to_canonical[alias_lower] = char_key
                self._fictional_names.add(alias_lower)

            # Also add the full name
            full_name = char_data.get("full_name", "")
            if full_name:
                full_name_lower = full_name.lower()
                self._alias_to_canonical[full_name_lower] = char_key
                self._fictional_names.add(full_name_lower)

        # Add narrative locations
        for loc_key, loc_data in self.registry.get("narrative_locations", {}).items():
            loc_lower = loc_key.lower()
            self._alias_to_canonical[loc_lower] = f"location:{loc_key}"
            self._fictional_names.add(loc_lower)
            self._character_info[f"location:{loc_key}"] = loc_data

        # Build source identifiers set
        self._source_identifiers: Set[str] = set()
        for source_data in self.registry.get("sources", {}).values():
            for identifier in source_data.get("source_identifiers", []):
                self._source_identifiers.add(identifier.lower())

    def is_fictional(self, name: str, source: str = None) -> bool:
        """
        Check if an entity name is a known fictional character.

        Args:
            name: Entity name to check.
            source: Optional source identifier (e.g., "our-biggest-deal").
                   If provided, also checks if source is a narrative fiction source.

        Returns:
            True if this is a known fictional character or from a narrative source.
        """
        name_lower = name.lower().strip()

        # Direct name match
        if name_lower in self._fictional_names:
            return True

        # Check if the source is a narrative fiction source
        if source and source.lower() in self._source_identifiers:
            return True

        return False

    def get_canonical_name(self, name: str) -> Optional[str]:
        """
        Get the canonical character key for a name/alias.

        Args:
            name: Entity name or alias.

        Returns:
            Canonical character key, or None if not a known fictional character.
        """
        name_lower = name.lower().strip()
        return self._alias_to_canonical.get(name_lower)

    def get_character_info(self, name: str) -> Optional[Dict]:
        """
        Get full character information for a name/alias.

        Args:
            name: Entity name or alias.

        Returns:
            Character info dictionary, or None if not found.
        """
        canonical = self.get_canonical_name(name)
        if canonical:
            return self._character_info.get(canonical)
        return None

    def tag_entity(self, entity: Dict, strict_mode: bool = True) -> Dict:
        """
        Tag an entity as fictional if it matches the registry.

        Modifications made to entity when fictional:
        - is_fictional: True
        - source_type: "narrative"
        - original_type: preserves original type
        - type: changed to "FICTIONAL_CHARACTER" (or appropriate fictional type)
        - fictional_source: source book/content identifier
        - fictional_canonical_name: canonical name from registry

        Args:
            entity: Entity dictionary with at least 'name' field.
            strict_mode: If True (default), only tags entities that are:
                        a) Known fictional character names, OR
                        b) EXCLUSIVELY from narrative sources (no episode sources)
                        If False, tags any entity that has a narrative source.

        Returns:
            Modified entity dict (or original if not fictional).
            The original dict is modified in place.
        """
        self._stats["total_checked"] += 1

        name = entity.get("name", "")
        sources = entity.get("sources", [])

        # Check if name is a known fictional character
        is_name_fictional = self.is_fictional(name)

        # Check sources
        narrative_sources = [src for src in sources if src.lower() in self._source_identifiers]
        non_narrative_sources = [src for src in sources if src.lower() not in self._source_identifiers]

        # Determine if entity should be tagged as fictional
        if strict_mode:
            # In strict mode, only tag if:
            # 1. It's a known fictional character name, OR
            # 2. It ONLY has narrative sources (no episodes, no other books)
            is_exclusively_narrative = len(narrative_sources) > 0 and len(non_narrative_sources) == 0
            should_tag = is_name_fictional or is_exclusively_narrative
        else:
            # In non-strict mode, tag if name matches OR any source is narrative
            should_tag = is_name_fictional or len(narrative_sources) > 0

        if not should_tag:
            return entity

        # Get character info for additional tagging
        char_info = self.get_character_info(name)
        canonical = self.get_canonical_name(name)

        # Tag the entity
        entity["is_fictional"] = True
        entity["source_type"] = "narrative"

        # Preserve original type
        if "type" in entity and "original_type" not in entity:
            entity["original_type"] = entity["type"]

        # Set fictional type
        if char_info:
            fictional_type = char_info.get("type", "FICTIONAL_CHARACTER")
            entity["type"] = fictional_type
            entity["fictional_canonical_name"] = canonical
            entity["fictional_source"] = char_info.get("source", "OurBiggestDeal")

            # Track stats by character
            if canonical not in self._stats["by_character"]:
                self._stats["by_character"][canonical] = 0
            self._stats["by_character"][canonical] += 1
        else:
            # Entity from fictional source but not a named character
            entity["type"] = "FICTIONAL_ENTITY"
            entity["fictional_source"] = "OurBiggestDeal"

        # Track stats
        self._stats["tagged_fictional"] += 1
        fictional_type = entity.get("type", "FICTIONAL_ENTITY")
        if fictional_type not in self._stats["by_type"]:
            self._stats["by_type"][fictional_type] = 0
        self._stats["by_type"][fictional_type] += 1

        return entity

    def tag_batch(self, entities: List[Dict], strict_mode: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Tag a batch of entities, separating fictional from real.

        Args:
            entities: List of entity dictionaries.
            strict_mode: If True (default), only tags entities that are:
                        a) Known fictional character names, OR
                        b) EXCLUSIVELY from narrative sources (no episode sources)
                        If False, tags any entity that has a narrative source.

        Returns:
            Tuple of (real_entities, fictional_entities).
            Each list contains the tagged entities.
        """
        real_entities = []
        fictional_entities = []

        for entity in entities:
            tagged = self.tag_entity(entity, strict_mode=strict_mode)
            if tagged.get("is_fictional"):
                fictional_entities.append(tagged)
            else:
                real_entities.append(tagged)

        return real_entities, fictional_entities

    def is_narrative_source(self, source: str) -> bool:
        """
        Check if a source identifier indicates narrative fiction content.

        Args:
            source: Source identifier string.

        Returns:
            True if this source is a known narrative fiction source.
        """
        return source.lower() in self._source_identifiers

    def get_stats(self) -> Dict:
        """
        Return tagging statistics.

        Returns:
            Dictionary with:
            - total_checked: Number of entities checked
            - tagged_fictional: Number tagged as fictional
            - by_type: Breakdown by fictional entity type
            - by_character: Breakdown by canonical character name
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset the statistics counters."""
        self._stats = {
            "total_checked": 0,
            "tagged_fictional": 0,
            "by_type": {},
            "by_character": {}
        }

    def get_all_fictional_names(self) -> Set[str]:
        """
        Get all known fictional character names (lowercase).

        Returns:
            Set of all lowercase fictional names/aliases.
        """
        return self._fictional_names.copy()

    def get_narrative_sources(self) -> Set[str]:
        """
        Get all known narrative fiction source identifiers.

        Returns:
            Set of source identifier strings.
        """
        return self._source_identifiers.copy()
