"""
Entity Resolver Module

Resolves entity name variations across chapters/sections with deterministic canonicalization.

Purpose:
- Consolidates entity variations into canonical forms (e.g., "Aaron Perry" → "Aaron William Perry")
- Applies deterministic tie-breaking rules for reproducibility
- Persists alias map for reuse and inspection
- Essential for cross-chapter/section consolidation

Features:
- 4-level deterministic tie-breaking (longest → most frequent → earliest → allowlist)
- Fuzzy matching for entity variants (normalized comparison)
- Alias map generation and persistence
- Statistics tracking (merges, aliases created, etc.)

Version History:
- v1.0.0 (V14.3.3): Initial implementation for incremental consolidation
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class EntityResolver(PostProcessingModule):
    """
    Resolves entity name variations across chapters/sections.

    This module identifies entity variants (e.g., "Aaron Perry", "Aaron William Perry", "Perry")
    and consolidates them into a canonical form using deterministic tie-breaking rules.

    Content Types: All
    Priority: 112 (after Deduplicator at 110, before SemanticDeduplicator at 115)

    Deterministic Tie-Breaking Rules (in order):
    1. Longest name wins (more specific)
    2. Most frequent occurrence across all input relationships
    3. Earliest occurrence (by relationship index/page number if available)
    4. Explicit allowlist (known authors/entities from metadata)

    Example:
        Input:
        - Chapter 1: "Aaron Perry" (appears 5 times, first page 12)
        - Chapter 10: "Aaron William Perry" (appears 15 times, first page 145)
        - Chapter 15: "Perry" (appears 2 times, first page 278)

        Resolution: All → "Aaron William Perry" (canonical)
        Alias Map: {
            "aaron perry": "Aaron William Perry",
            "perry": "Aaron William Perry"
        }
    """

    name = "EntityResolver"
    description = "Resolves entity name variations with deterministic canonicalization"
    content_types = ["all"]
    priority = 112  # After Deduplicator (110), before SemanticDeduplicator (115)
    dependencies = []
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Configuration
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.min_variant_length = self.config.get('min_variant_length', 2)
        self.case_sensitive = self.config.get('case_sensitive', False)

        # State
        self.alias_map: Dict[str, str] = {}  # variant_normalized -> canonical_form
        self.canonical_entities: Dict[str, str] = {}  # variant_key -> canonical_name
        self.entity_stats: Dict[str, Dict[str, Any]] = {}  # For tracking resolution stats

    def normalize_entity_key(self, entity: str) -> str:
        """
        Normalize entity name for comparison.

        Args:
            entity: Entity name

        Returns:
            Normalized key for comparison
        """
        normalized = entity.strip()

        if not self.case_sensitive:
            normalized = normalized.lower()

        # Additional normalization
        normalized = ' '.join(normalized.split())  # Collapse whitespace

        return normalized

    def is_potential_variant(self, name1: str, name2: str) -> bool:
        """
        Check if two names are potential variants of the same entity.

        Uses simple substring matching for now. More sophisticated approaches
        could use fuzzy matching or embedding similarity.

        Args:
            name1: First entity name
            name2: Second entity name

        Returns:
            True if names are potential variants
        """
        if name1 == name2:
            return True

        # Normalize
        norm1 = self.normalize_entity_key(name1)
        norm2 = self.normalize_entity_key(name2)

        # Check if one is substring of the other
        # (e.g., "Aaron Perry" contains "Perry")
        if norm1 in norm2 or norm2 in norm1:
            return True

        # Check if they share significant words
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for'}
        words1 = {w for w in words1 if w not in common_words}
        words2 = {w for w in words2 if w not in common_words}

        if not words1 or not words2:
            return False

        # If they share all words from the shorter name, consider them variants
        shorter_words = words1 if len(words1) <= len(words2) else words2
        longer_words = words2 if len(words1) <= len(words2) else words1

        if shorter_words.issubset(longer_words):
            return True

        return False

    def find_entity_variants(
        self,
        relationships: List[Any]
    ) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        Find entity variants and track their occurrences.

        Args:
            relationships: List of relationship objects

        Returns:
            Dict mapping variant_key to list of (entity_name, count, first_index) tuples
        """
        # Track all entity mentions
        entity_mentions: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        for idx, rel in enumerate(relationships):
            # Extract source and target entities
            source = getattr(rel, 'source', None) or (rel.get('source') if isinstance(rel, dict) else None)
            target = getattr(rel, 'target', None) or (rel.get('target') if isinstance(rel, dict) else None)

            if source:
                entity_mentions[source].append((source, idx))
            if target:
                entity_mentions[target].append((target, idx))

        # Count occurrences and find first appearance
        entity_counts: Dict[str, Tuple[int, int]] = {}  # entity -> (count, first_index)

        for entity, mentions in entity_mentions.items():
            count = len(mentions)
            first_index = min(idx for _, idx in mentions)
            entity_counts[entity] = (count, first_index)

        # Group variants by normalized form
        variant_groups: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)

        # Track which entities have been grouped
        grouped: Set[str] = set()

        for entity1 in entity_counts.keys():
            if entity1 in grouped:
                continue

            # Find all variants of this entity
            variants = [entity1]

            for entity2 in entity_counts.keys():
                if entity2 != entity1 and entity2 not in grouped:
                    if self.is_potential_variant(entity1, entity2):
                        variants.append(entity2)

            # Group all these variants under a common key
            # Use normalized form of the longest variant as the key
            variant_key = self.normalize_entity_key(max(variants, key=len))

            for variant in variants:
                count, first_index = entity_counts[variant]
                variant_groups[variant_key].append((variant, count, first_index))
                grouped.add(variant)

        return variant_groups

    def select_canonical_forms(
        self,
        variant_groups: Dict[str, List[Tuple[str, int, int]]],
        context: ProcessingContext
    ) -> Dict[str, str]:
        """
        Apply deterministic tie-breaking to select canonical entity names.

        Tie-breaking rules (in order):
        1. Longest name (most specific)
        2. Most frequent occurrence
        3. Earliest occurrence (first appearance)
        4. Allowlist override (known entities from metadata)

        Args:
            variant_groups: Dict of variant_key -> list of (name, count, first_index)
            context: Processing context (for allowlist lookup)

        Returns:
            Dict of variant_key -> canonical_name
        """
        canonical: Dict[str, str] = {}

        # Get allowlist from metadata
        allowlist = context.document_metadata.get('known_entities', [])
        if isinstance(allowlist, str):
            allowlist = [allowlist]

        allowlist_lower = [e.lower() for e in allowlist]

        for variant_key, variants in variant_groups.items():
            if not variants:
                continue

            # Rule 4 (highest priority): Check allowlist first
            for variant_name, count, first_index in variants:
                if variant_name.lower() in allowlist_lower:
                    canonical[variant_key] = variant_name
                    logger.debug(f"   EntityResolver: Canonical from allowlist: {variant_name}")
                    break
            else:
                # Rules 1-3: Apply tie-breaking
                # Sort by: longest → most frequent → earliest
                sorted_variants = sorted(
                    variants,
                    key=lambda v: (
                        -len(v[0]),      # Longest first (negative for descending)
                        -v[1],           # Most frequent first
                        v[2]             # Earliest first (ascending)
                    )
                )

                canonical[variant_key] = sorted_variants[0][0]
                logger.debug(
                    f"   EntityResolver: Canonical selected: {sorted_variants[0][0]} "
                    f"from {len(variants)} variants"
                )

        return canonical

    def build_alias_map(
        self,
        variant_groups: Dict[str, List[Tuple[str, int, int]]],
        canonical_forms: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Build alias map for persistence.

        Args:
            variant_groups: Dict of variant_key -> list of (name, count, first_index)
            canonical_forms: Dict of variant_key -> canonical_name

        Returns:
            Dict mapping variant_normalized -> canonical_form
        """
        alias_map = {}

        for variant_key, variants in variant_groups.items():
            canonical_name = canonical_forms.get(variant_key)

            if not canonical_name:
                continue

            # Map all variants (except canonical) to canonical form
            for variant_name, count, first_index in variants:
                variant_normalized = self.normalize_entity_key(variant_name)

                # Don't map canonical to itself
                if variant_name != canonical_name:
                    alias_map[variant_normalized] = canonical_name

        return alias_map

    def apply_canonical_forms(
        self,
        relationships: List[Any],
        canonical_forms: Dict[str, str]
    ) -> List[Any]:
        """
        Update all relationships with canonical entity names.

        Args:
            relationships: List of relationship objects
            canonical_forms: Dict of variant_key -> canonical_name

        Returns:
            List of relationships with canonical entities
        """
        resolved = []
        resolutions_applied = 0

        for rel in relationships:
            # Get source and target
            is_dict = isinstance(rel, dict)

            if is_dict:
                source = rel.get('source')
                target = rel.get('target')
            else:
                source = getattr(rel, 'source', None)
                target = getattr(rel, 'target', None)

            # Look up canonical forms
            source_key = self.normalize_entity_key(source) if source else None
            target_key = self.normalize_entity_key(target) if target else None

            canonical_source = canonical_forms.get(source_key, source)
            canonical_target = canonical_forms.get(target_key, target)

            # Check if resolution was applied
            if (source and canonical_source != source) or (target and canonical_target != target):
                resolutions_applied += 1

            # Update relationship
            if is_dict:
                rel_copy = rel.copy()
                if source:
                    rel_copy['source'] = canonical_source
                if target:
                    rel_copy['target'] = canonical_target
                resolved.append(rel_copy)
            else:
                # Create copy and update
                import copy
                rel_copy = copy.deepcopy(rel)
                if source:
                    rel_copy.source = canonical_source
                if target:
                    rel_copy.target = canonical_target
                resolved.append(rel_copy)

        return resolved, resolutions_applied

    def save_alias_map(self, output_path: str):
        """
        Persist alias map for reproducibility and inspection.

        Args:
            output_path: Path to save alias map JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(
                self.alias_map,
                f,
                indent=2,
                sort_keys=True,  # REQUIRED for determinism
                ensure_ascii=False
            )

        logger.info(f"   {self.name}: Saved {len(self.alias_map)} aliases to {output_path}")

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """
        Resolve entity name variations in all relationships.

        Args:
            relationships: List of relationship objects
            context: Processing context (for allowlist lookup)

        Returns:
            List of relationships with resolved entity names
        """
        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        # Step 1: Find entity variants
        logger.info(f"   {self.name}: Finding entity variants...")
        variant_groups = self.find_entity_variants(relationships)

        # Count total variants
        total_variants = sum(len(variants) for variants in variant_groups.values())
        variant_groups_count = len([v for v in variant_groups.values() if len(v) > 1])

        logger.info(
            f"   {self.name}: Found {variant_groups_count} variant groups "
            f"covering {total_variants} entity mentions"
        )

        # Step 2: Select canonical forms
        self.canonical_entities = self.select_canonical_forms(variant_groups, context)

        # Step 3: Build alias map
        self.alias_map = self.build_alias_map(variant_groups, self.canonical_entities)

        # Step 4: Apply canonical forms to relationships
        resolved, resolutions_applied = self.apply_canonical_forms(
            relationships,
            self.canonical_entities
        )

        # Update stats
        self.stats['variant_groups'] = variant_groups_count
        self.stats['total_variants'] = total_variants
        self.stats['aliases_created'] = len(self.alias_map)
        self.stats['resolutions_applied'] = resolutions_applied
        self.stats['modified_count'] = resolutions_applied

        logger.info(
            f"   {self.name}: Resolved {resolutions_applied} entity mentions "
            f"({len(self.alias_map)} aliases)"
        )

        return resolved

    def get_summary(self) -> Dict[str, Any]:
        """Get module processing summary"""
        summary = super().get_summary()
        summary['stats']['variant_groups'] = self.stats.get('variant_groups', 0)
        summary['stats']['aliases_created'] = self.stats.get('aliases_created', 0)
        summary['stats']['resolutions_applied'] = self.stats.get('resolutions_applied', 0)
        return summary
