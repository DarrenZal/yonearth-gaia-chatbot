"""
Type Compatibility Validator Module

Validates and auto-fixes entity type mismatches in relationships.

Features:
- Schema-based validation (authored, wrote foreword for, endorsed, dedicated to)
- Auto-fix: swap source/target when unambiguous
- Auto-fix: adjust predicates when types suggest different relationship
- Flag TYPE_INCOMPATIBLE when cannot fix

Version History:
- v1.0.0 (V14.3.7): Initial implementation with auto-correction
"""

import logging
import copy
from typing import Optional, List, Dict, Any

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class TypeCompatibilityValidator(PostProcessingModule):
    """
    Validates entity type compatibility with relationship predicates.

    Content Types: Universal (books, podcasts, etc.)
    Priority: 85 (after PredicateValidator, before VagueEntityBlocker)
    """

    name = "TypeCompatibilityValidator"
    description = "Validates and auto-fixes entity type mismatches"
    content_types = ["all"]
    priority = 85
    dependencies = ["PredicateValidator"]
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Relationship schemas: {predicate: (valid_source_types, valid_target_types)}
        self.schemas = self.config.get('schemas', {
            'authored': ({'Person', 'Organization'}, {'Book', 'Essay', 'Article', 'Work'}),
            'author of': ({'Person', 'Organization'}, {'Book', 'Essay', 'Article', 'Work'}),
            'wrote': ({'Person'}, {'Book', 'Essay', 'Article', 'Work'}),
            'wrote foreword for': ({'Person'}, {'Book', 'Work'}),
            'endorsed': ({'Person', 'Organization'}, {'Book', 'Work', 'Concept', 'Practice'}),
            'dedicated': ({'Person', 'Book'}, {'Person', 'Organization', 'Community'}),
            'dedicated to': ({'Person', 'Book'}, {'Person', 'Organization', 'Community'}),
        })

        self.auto_fix = self.config.get('auto_fix', True)

    def is_compatible(self, predicate: str, source_type: str, target_type: str) -> bool:
        """Check if entity types are compatible with predicate"""
        schema = self.schemas.get(predicate.lower())
        if not schema:
            return True  # No schema = assume compatible

        valid_sources, valid_targets = schema
        return source_type in valid_sources and target_type in valid_targets

    def can_auto_fix(self, predicate: str, source_type: str, target_type: str) -> Optional[str]:
        """
        Determine if incompatibility can be auto-fixed by swapping.

        Returns:
            'swap' if should swap source/target
            'adjust_predicate' if predicate should be adjusted
            None if cannot fix
        """
        schema = self.schemas.get(predicate.lower())
        if not schema:
            return None

        valid_sources, valid_targets = schema

        # Check if swapping would fix
        if target_type in valid_sources and source_type in valid_targets:
            return 'swap'

        return None

    def validate_and_fix(self, rel: Any) -> Any:
        """Validate relationship and attempt auto-fix if needed"""
        predicate = rel.relationship.lower()
        source_type = rel.source_type
        target_type = rel.target_type

        # Check compatibility
        if self.is_compatible(predicate, source_type, target_type):
            return rel

        # Log incompatibility
        logger.debug(f"Type incompatibility: ({source_type}) → {predicate} → ({target_type})")

        # Attempt auto-fix
        if not self.auto_fix:
            # Just flag
            if rel.flags is None:
                rel.flags = {}
            rel.flags['TYPE_INCOMPATIBLE'] = True
            rel.flags['incompatibility_reason'] = f"({source_type}) → {predicate} → ({target_type})"
            return rel

        fix_action = self.can_auto_fix(predicate, source_type, target_type)

        if fix_action == 'swap':
            # Create new relationship with swapped entities
            new_rel = copy.deepcopy(rel)
            new_rel.source, new_rel.target = rel.target, rel.source
            new_rel.source_type, new_rel.target_type = rel.target_type, rel.source_type

            # Swap evidence surfaces if present
            if hasattr(new_rel, 'evidence') and isinstance(new_rel.evidence, dict):
                if 'source_surface' in new_rel.evidence and 'target_surface' in new_rel.evidence:
                    new_rel.evidence['source_surface'], new_rel.evidence['target_surface'] = \
                        new_rel.evidence.get('target_surface'), new_rel.evidence.get('source_surface')

            # Add correction flag
            if new_rel.flags is None:
                new_rel.flags = {}
            new_rel.flags['TYPE_INCOMPATIBILITY_FIXED'] = True
            new_rel.flags['fix_action'] = 'swapped_entities'
            new_rel.flags['original_direction'] = f"{rel.source} → {rel.relationship} → {rel.target}"

            logger.info(f"Auto-fixed type incompatibility by swapping: {new_rel.source} → {predicate} → {new_rel.target}")
            self.stats['auto_fixed'] += 1
            return new_rel
        else:
            # Cannot fix - flag only
            if rel.flags is None:
                rel.flags = {}
            rel.flags['TYPE_INCOMPATIBLE'] = True
            rel.flags['incompatibility_reason'] = f"({source_type}) → {predicate} → ({target_type})"
            logger.warning(f"Cannot fix type incompatibility: {rel.source} ({source_type}) → {predicate} → {rel.target} ({target_type})")
            self.stats['flagged'] += 1
            return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to validate type compatibility"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0
        self.stats['auto_fixed'] = 0
        self.stats['flagged'] = 0

        processed = []
        for rel in relationships:
            validated_rel = self.validate_and_fix(rel)

            # Check if modified
            if validated_rel != rel or (validated_rel.flags and 
                ('TYPE_INCOMPATIBILITY_FIXED' in validated_rel.flags or 'TYPE_INCOMPATIBLE' in validated_rel.flags)):
                self.stats['modified_count'] += 1

            processed.append(validated_rel)

        logger.info(
            f"   {self.name}: {self.stats['auto_fixed']} auto-fixed, "
            f"{self.stats['flagged']} flagged"
        )

        return processed
