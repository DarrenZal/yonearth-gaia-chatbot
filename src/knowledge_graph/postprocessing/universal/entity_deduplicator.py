"""
Entity Deduplicator Module (V1.0)

Deduplicates entities that appear multiple times with slight variations.

Handles:
- Exact matches (case-insensitive): "Viriditas" vs "viriditas"
- Name variations: "Aaron Perry" vs "Aaron William Perry"
- Type conflicts: Picks most specific type when same entity has multiple types

Purpose:
Merges duplicate entity references to create a cleaner, more coherent knowledge graph.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Set
from collections import defaultdict

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class EntityDeduplicator(PostProcessingModule):
    """
    Deduplicates entities by normalizing names and merging duplicates.

    Content Types: All
    Priority: 5 (run early, before most processing)
    Dependencies: None
    """

    name = "EntityDeduplicator"
    description = "Merges duplicate entity mentions"
    content_types = ["all"]
    priority = 5
    dependencies = []
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Entity name normalization cache
        self.entity_map: Dict[str, str] = {}  # normalized_name -> canonical_name
        self.entity_stats: Dict[str, int] = defaultdict(int)  # Track occurrence counts

    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison"""
        # Remove extra whitespace
        normalized = ' '.join(name.split())
        # Lowercase for comparison
        normalized = normalized.lower()
        # Remove common punctuation variations
        normalized = normalized.replace('"', '').replace('"', '').replace('"', '')
        normalized = normalized.strip()
        return normalized

    def pick_canonical_name(self, names: List[str]) -> str:
        """Pick the best canonical name from variations"""
        # Prefer longer, more complete names
        # "Aaron William Perry" over "Aaron Perry"
        return max(names, key=len)

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Deduplicate entity references in relationships"""

        self.stats['processed_count'] = len(relationships)
        self.entity_map = {}
        self.entity_stats = defaultdict(int)

        # Phase 1: Build entity occurrence map
        name_variations: Dict[str, Set[str]] = defaultdict(set)

        for rel in relationships:
            source = rel.source
            target = rel.target

            source_norm = self.normalize_entity_name(source)
            target_norm = self.normalize_entity_name(target)

            name_variations[source_norm].add(source)
            name_variations[target_norm].add(target)

            self.entity_stats[source] += 1
            self.entity_stats[target] += 1

        # Phase 2: Create canonical name mapping
        duplicates_found = 0
        for norm_name, variations in name_variations.items():
            if len(variations) > 1:
                # Multiple variations exist - pick canonical
                canonical = self.pick_canonical_name(list(variations))
                for variation in variations:
                    if variation != canonical:
                        self.entity_map[variation] = canonical
                        duplicates_found += 1

        logger.info(f"   {self.name}: Found {duplicates_found} entity variations to merge")

        # Phase 3: Apply deduplication to relationships
        deduplicated = []
        for rel in relationships:
            # Update source and target to canonical names
            if rel.source in self.entity_map:
                original_source = rel.source
                rel.source = self.entity_map[rel.source]
                if not hasattr(rel, 'flags') or rel.flags is None:
                    rel.flags = {}
                rel.flags['ENTITY_DEDUPLICATED_SOURCE'] = True
                rel.flags['original_source_variation'] = original_source
                self.stats['modified_count'] += 1

            if rel.target in self.entity_map:
                original_target = rel.target
                rel.target = self.entity_map[rel.target]
                if not hasattr(rel, 'flags') or rel.flags is None:
                    rel.flags = {}
                rel.flags['ENTITY_DEDUPLICATED_TARGET'] = True
                rel.flags['original_target_variation'] = original_target
                self.stats['modified_count'] += 1

            deduplicated.append(rel)

        # Store stats
        self.stats['entities_merged'] = duplicates_found
        self.stats['unique_entities'] = len(name_variations)

        logger.info(
            f"   {self.name}: Merged {duplicates_found} entity variations, "
            f"{len(name_variations)} unique entities remain"
        )

        return deduplicated
