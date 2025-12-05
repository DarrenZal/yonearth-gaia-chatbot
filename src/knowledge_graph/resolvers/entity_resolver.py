"""
Entity Resolver - Resolves entity name variants to canonical forms.

This module provides functionality to normalize entity names by mapping
various name variants (aliases, misspellings, alternate formats) to their
canonical (authoritative) forms using a registry-based approach.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher


class EntityResolver:
    """Resolves entity name variants to canonical forms using a registry."""

    # URL/domain detection patterns
    URL_PATTERNS = [
        r'^https?://',                                    # Full URLs
        r'^www\.',                                        # www. prefix
        r'^[a-zA-Z0-9-]+\.(org|com|net|edu|gov|io|co|info)$',  # bare domains
        r'\.(org|com|net|edu|gov|io|co|info)/',           # domain with path
    ]

    def __init__(self, registry_path: Path = None):
        """Load canonical entity registry.

        Args:
            registry_path: Path to canonical_entities.json
                          Defaults to data/canonical_entities.json
        """
        if registry_path is None:
            # Try multiple paths for flexibility
            possible_paths = [
                Path("data/canonical_entities.json"),
                Path(__file__).parent.parent.parent.parent / "data" / "canonical_entities.json",
            ]
            for path in possible_paths:
                if path.exists():
                    registry_path = path
                    break
            if registry_path is None:
                registry_path = possible_paths[0]

        self.registry_path = registry_path
        self.registry = self._load_registry(registry_path)
        self._build_lookup_indices()
        self.reset_stats()

    def _load_registry(self, path: Path) -> Dict:
        """Load the canonical entity registry."""
        with open(path) as f:
            return json.load(f)

    def _build_lookup_indices(self) -> None:
        """Build efficient lookup structures.

        Creates:
        - self._exact_lookup: Dict[str, str] - lowercase alias -> canonical name
        - self._patterns: List[Tuple[re.Pattern, str]] - compiled patterns -> canonical
        - self._canonical_info: Dict[str, Dict] - canonical name -> full info
        - self._canonical_to_type: Dict[str, str] - canonical name -> entity type
        """
        self._exact_lookup: Dict[str, str] = {}
        self._patterns: List[Tuple[re.Pattern, str]] = []
        self._canonical_info: Dict[str, Dict] = {}
        self._canonical_to_type: Dict[str, str] = {}
        self._all_canonical_names: List[str] = []

        # Process each category (organizations, people, products, concepts)
        for category in ["organizations", "people", "products", "concepts"]:
            if category not in self.registry:
                continue

            for entity_id, entity_data in self.registry[category].items():
                canonical_name = entity_data["canonical_name"]
                entity_type = entity_data.get("type", category.upper().rstrip("S"))

                # Store canonical info
                self._canonical_info[canonical_name] = entity_data
                self._canonical_to_type[canonical_name] = entity_type
                self._all_canonical_names.append(canonical_name)

                # Add canonical name itself as exact match
                self._exact_lookup[canonical_name.lower()] = canonical_name

                # Add all aliases as exact matches
                for alias in entity_data.get("aliases", []):
                    self._exact_lookup[alias.lower()] = canonical_name

                # Compile regex patterns
                for pattern in entity_data.get("merge_patterns", []):
                    try:
                        compiled = re.compile(pattern, re.IGNORECASE)
                        self._patterns.append((compiled, canonical_name))
                    except re.error:
                        # Skip invalid patterns
                        pass

    def is_url_or_domain(self, name: str) -> bool:
        """Detect if a name is a URL or domain name.

        Used to identify entities that should be extracted as URL type
        rather than organization names.

        Examples:
            - "yonearth.org" → True
            - "https://patagonia.com/about" → True
            - "www.example.com" → True
            - "Y on Earth" → False
            - "Dr. Bronner's" → False
            - "Aaron Perry" → False

        Args:
            name: Entity name to check

        Returns:
            True if the name appears to be a URL or domain
        """
        if not name or not isinstance(name, str):
            return False

        name_lower = name.lower().strip()
        for pattern in self.URL_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        return False

    def resolve(self, name: str, entity_type: str = None) -> Tuple[str, float, str]:
        """Resolve an entity name to its canonical form.

        Resolution order:
        1. Check if URL/domain → return as-is with "url" method (don't resolve)
        2. Exact alias match (case-insensitive) -> confidence 1.0
        3. Regex pattern match -> confidence 0.95
        4. Fuzzy match (>=85% similarity) -> confidence = similarity score
        5. No match -> return original name, confidence 0.0

        Args:
            name: Entity name to resolve
            entity_type: Optional type hint (PERSON, ORGANIZATION, etc.)

        Returns:
            Tuple of:
            - resolved_name: The canonical name (or original if unresolved)
            - confidence: Float 0.0-1.0 indicating match confidence
            - method: One of "url", "exact", "pattern", "fuzzy", "unresolved"
        """
        if not name or not isinstance(name, str):
            return name, 0.0, "unresolved"

        # 1. Check if this is a URL - don't try to resolve, flag it
        if self.is_url_or_domain(name):
            self._stats["by_method"]["url"] = self._stats["by_method"].get("url", 0) + 1
            return (name, 1.0, "url")  # Return as-is, flagged as URL

        name_lower = name.lower().strip()

        # 2. Exact alias match (case-insensitive)
        if name_lower in self._exact_lookup:
            canonical = self._exact_lookup[name_lower]
            self._stats["by_method"]["exact"] += 1
            self._stats["total_resolved"] += 1
            self._stats["confidence_sum"] += 1.0
            return canonical, 1.0, "exact"

        # 3. Regex pattern match
        for pattern, canonical in self._patterns:
            if pattern.fullmatch(name_lower):
                self._stats["by_method"]["pattern"] += 1
                self._stats["total_resolved"] += 1
                self._stats["confidence_sum"] += 0.95
                return canonical, 0.95, "pattern"

        # 4. Fuzzy match
        fuzzy_result = self._fuzzy_match(name)
        if fuzzy_result:
            canonical, confidence = fuzzy_result
            self._stats["by_method"]["fuzzy"] += 1
            self._stats["total_resolved"] += 1
            self._stats["confidence_sum"] += confidence
            return canonical, confidence, "fuzzy"

        # 5. No match
        self._stats["by_method"]["unresolved"] += 1
        return name, 0.0, "unresolved"

    def _fuzzy_match(self, name: str, threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Find fuzzy match among canonical names.

        Uses SequenceMatcher for similarity scoring.
        Only returns matches above threshold.

        Args:
            name: Name to match
            threshold: Minimum similarity score (default 0.85)

        Returns:
            Tuple of (canonical_name, similarity) if match found, None otherwise
        """
        name_lower = name.lower()
        best_match = None
        best_score = 0.0

        # Check against all canonical names
        for canonical in self._all_canonical_names:
            score = SequenceMatcher(None, name_lower, canonical.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = canonical

        # Also check against all aliases
        for alias, canonical in self._exact_lookup.items():
            score = SequenceMatcher(None, name_lower, alias).ratio()
            if score > best_score:
                best_score = score
                best_match = canonical

        if best_score >= threshold:
            return best_match, best_score

        return None

    def resolve_batch(self, entities: List[Dict]) -> List[Dict]:
        """Resolve a batch of entities.

        For each entity:
        - Attempt resolution
        - If resolved:
          - Update 'name' to canonical form
          - Add 'original_name' with the original
          - Add 'resolution' dict with method and confidence

        Args:
            entities: List of entity dicts with 'name' field

        Returns:
            List of entities with resolved names
        """
        result = []

        for entity in entities:
            entity_copy = entity.copy()

            if "name" not in entity_copy:
                # Handle dict format where key is entity name
                result.append(entity_copy)
                continue

            original_name = entity_copy["name"]
            entity_type = entity_copy.get("type")

            resolved_name, confidence, method = self.resolve(original_name, entity_type)

            if method != "unresolved":
                entity_copy["original_name"] = original_name
                entity_copy["name"] = resolved_name
                entity_copy["resolution"] = {
                    "method": method,
                    "confidence": confidence
                }

            result.append(entity_copy)

        return result

    def get_stats(self) -> Dict:
        """Return resolution statistics.

        Returns:
            {
                "total_resolved": int,
                "by_method": {
                    "exact": int,
                    "pattern": int,
                    "fuzzy": int,
                    "unresolved": int
                },
                "avg_confidence": float
            }
        """
        total = self._stats["total_resolved"]
        avg_confidence = (
            self._stats["confidence_sum"] / total if total > 0 else 0.0
        )

        return {
            "total_resolved": total,
            "by_method": self._stats["by_method"].copy(),
            "avg_confidence": avg_confidence
        }

    def reset_stats(self) -> None:
        """Reset resolution statistics."""
        self._stats = {
            "total_resolved": 0,
            "confidence_sum": 0.0,
            "by_method": {
                "exact": 0,
                "pattern": 0,
                "fuzzy": 0,
                "unresolved": 0
            }
        }

    def get_canonical_info(self, canonical_name: str) -> Optional[Dict]:
        """Get full information for a canonical entity.

        Args:
            canonical_name: The canonical entity name

        Returns:
            Entity info dict or None if not found
        """
        return self._canonical_info.get(canonical_name)

    def list_canonical_names(self, entity_type: str = None) -> List[str]:
        """List all canonical entity names.

        Args:
            entity_type: Optional filter by type (PERSON, ORGANIZATION, etc.)

        Returns:
            List of canonical names
        """
        if entity_type is None:
            return self._all_canonical_names.copy()

        return [
            name for name in self._all_canonical_names
            if self._canonical_to_type.get(name) == entity_type
        ]

    def add_alias(self, canonical_name: str, alias: str) -> bool:
        """Add a new alias for an existing canonical entity.

        Note: This only updates the in-memory index, not the registry file.

        Args:
            canonical_name: The canonical entity name
            alias: New alias to add

        Returns:
            True if added successfully, False if canonical not found
        """
        if canonical_name not in self._canonical_info:
            return False

        self._exact_lookup[alias.lower()] = canonical_name
        return True
