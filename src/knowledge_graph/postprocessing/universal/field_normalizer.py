"""
Field Normalizer Module

Adapter module to normalize field naming inconsistencies across pipeline versions.

Purpose:
- Ensures consolidated data uses consistent field names regardless of source
- Handles historical inconsistencies (predicate vs relationship, source_entity vs source)
- Runs earliest in the pipeline (priority 5) to normalize all incoming data
- Makes 'relationship' canonical (most modules expect this field)

Features:
- Field name normalization (predicate → relationship, source_entity → source)
- Mirrors both 'relationship' and 'predicate' for compatibility
- Handles both dict and object inputs
- Backfills missing canonical fields from legacy fields
- Transparent operation (doesn't modify data, only field names)

Version History:
- v1.0.0 (V14.3.3): Initial implementation for incremental consolidation
"""

import logging
from typing import List, Dict, Any, Optional

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class FieldNormalizer(PostProcessingModule):
    """
    Adapter module to normalize field naming inconsistencies across pipeline versions.

    This module addresses historical field naming inconsistencies where some pipelines
    used 'relationship' while others used 'predicate' for the same field. It ensures
    all relationships use canonical field names for consolidation.

    Content Types: All
    Priority: 5 (runs earliest - before all other modules)

    Normalizations:
    1. predicate → relationship (canonical - most modules expect this)
    2. source_entity → source (canonical)
    3. target_entity → target (canonical)

    Notes:
    - Both 'relationship' and 'predicate' fields are set to the same value for compatibility
    - Most modules (Deduplicator, PredicateNormalizer, PredicateValidator) expect 'relationship'
    - A few modules (SemanticDeduplicator) may read 'predicate', so we mirror both

    Example:
        Input: {source_entity: "Aaron Perry", predicate: "authored", ...}
        Output: {source: "Aaron Perry", relationship: "authored", predicate: "authored", ...}
    """

    name = "FieldNormalizer"
    description = "Normalizes field naming across pipeline versions"
    content_types = ["all"]
    priority = 5  # Run earliest
    dependencies = []  # No dependencies
    version = "1.0.0"

    # Field mappings: legacy_field -> canonical_field
    FIELD_MAPPINGS = {
        'predicate': 'relationship',  # relationship is canonical
        'source_entity': 'source',
        'target_entity': 'target',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Configuration
        self.preserve_legacy_fields = self.config.get('preserve_legacy_fields', False)

    def normalize_relationship(self, rel: Any) -> Any:
        """
        Normalize field names in a single relationship.

        Args:
            rel: Relationship dict or object

        Returns:
            Relationship with normalized field names
        """
        # Convert to dict if it's an object
        if hasattr(rel, '__dict__'):
            rel_dict = rel.__dict__.copy()
            is_object = True
        else:
            rel_dict = rel.copy()
            is_object = False

        # Create normalized dict
        normalized_dict = {}

        # First pass: Copy all fields, applying mappings
        for key, value in rel_dict.items():
            # Check if key needs normalization
            canonical_key = self.FIELD_MAPPINGS.get(key, key)
            normalized_dict[canonical_key] = value

            # Optionally preserve legacy field alongside canonical
            if self.preserve_legacy_fields and key in self.FIELD_MAPPINGS:
                normalized_dict[key] = value

        # Second pass: Backfill canonical fields from legacy if missing
        for legacy, canonical in self.FIELD_MAPPINGS.items():
            if legacy in rel_dict and canonical not in normalized_dict:
                normalized_dict[canonical] = rel_dict[legacy]
                logger.debug(f"   Backfilled {canonical} from {legacy}")

        # Special handling: Mirror relationship/predicate for compatibility
        # Most modules expect 'relationship', but some may read 'predicate'
        if 'relationship' in normalized_dict:
            normalized_dict['predicate'] = normalized_dict['relationship']
        elif 'predicate' in normalized_dict:
            normalized_dict['relationship'] = normalized_dict['predicate']

        # Return in original format
        if is_object:
            # Create object with normalized fields
            class NormalizedRel:
                pass

            normalized_obj = NormalizedRel()
            for key, value in normalized_dict.items():
                setattr(normalized_obj, key, value)

            return normalized_obj
        else:
            return normalized_dict

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """
        Normalize field names in all relationships.

        Args:
            relationships: List of relationship dicts or objects
            context: Processing context (unused for this module)

        Returns:
            List of relationships with normalized field names
        """
        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        normalized = []
        normalizations_applied = 0

        for rel in relationships:
            normalized_rel = self.normalize_relationship(rel)
            normalized.append(normalized_rel)

            # Track if any normalization was applied
            # (Check if any legacy fields existed)
            rel_dict = rel if isinstance(rel, dict) else rel.__dict__
            if any(legacy in rel_dict for legacy in self.FIELD_MAPPINGS.keys()):
                normalizations_applied += 1

        # Update stats
        self.stats['normalizations_applied'] = normalizations_applied
        self.stats['modified_count'] = normalizations_applied

        if normalizations_applied > 0:
            logger.info(
                f"   {self.name}: Normalized fields in {normalizations_applied} relationships"
            )
        else:
            logger.info(
                f"   {self.name}: No field normalizations needed (all fields already canonical)"
            )

        return normalized

    def get_summary(self) -> Dict[str, Any]:
        """Get module processing summary"""
        summary = super().get_summary()
        summary['stats']['normalizations_applied'] = self.stats.get('normalizations_applied', 0)
        summary['field_mappings'] = self.FIELD_MAPPINGS
        return summary
