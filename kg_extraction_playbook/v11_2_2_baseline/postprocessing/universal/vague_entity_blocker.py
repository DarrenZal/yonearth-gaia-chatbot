"""
Vague Entity Blocker Module

Blocks overly vague/abstract entities upfront.

Features:
- Pattern-based vague entity detection
- Filters out relationships with entities too abstract to be useful
- Unlike ContextEnricher (which tries to fix), this BLOCKS entirely
- Provides detailed blocking statistics

Version History:
- v1.0.0 (V7): Initial implementation
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class VagueEntityBlocker(PostProcessingModule):
    """
    Blocks relationships with overly vague/abstract entities.

    Content Types: Universal (works for all content types)
    Priority: 30 (runs early, before processing)
    """

    name = "VagueEntityBlocker"
    description = "Filters overly vague/abstract entities"
    content_types = ["all"]
    priority = 30
    dependencies = []
    version = "1.0.0"  # V7 (no V8 changes)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Patterns for entities that are too vague to be useful
        self.vague_abstract_patterns = self.config.get('vague_abstract_patterns', [
            r'^the (way|answer|solution|problem|challenge|issue|question|matter)$',
            r'^the (way|path|approach|method) (through|to|from|of)',
            r'^the (reason|cause|result|outcome|consequence) (for|of|why)',
            r'^(something|someone|anything|anyone|everything|everyone)$',
            r'^(things|ways|practices|methods|approaches|solutions)$',
            r'^(this|that)$',
            r'^it$',
        ])

        # Compile patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.vague_abstract_patterns]

    def is_too_vague(self, entity: str) -> Tuple[bool, str]:
        """
        Check if entity is too vague.

        Returns:
            (is_vague: bool, pattern_matched: str)
        """
        entity_lower = entity.lower().strip()

        # Check against all patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.match(entity_lower):
                return True, self.vague_abstract_patterns[i]

        return False, ""

    def should_block_relationship(self, rel: Any) -> Tuple[bool, str]:
        """
        Determine if relationship should be blocked.

        Returns:
            (should_block: bool, reason: str)
        """
        # Check source
        source_vague, source_pattern = self.is_too_vague(rel.source)
        if source_vague:
            return True, f"vague_source: {source_pattern}"

        # Check target
        target_vague, target_pattern = self.is_too_vague(rel.target)
        if target_vague:
            return True, f"vague_target: {target_pattern}"

        return False, ""

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Filter out relationships with overly vague entities"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        kept = []
        blocked_count = 0
        blocked_reasons = {}

        for rel in relationships:
            should_block, reason = self.should_block_relationship(rel)

            if should_block:
                blocked_count += 1
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                self.stats['modified_count'] += 1
            else:
                kept.append(rel)

        # Update stats
        self.stats['blocked'] = blocked_count
        self.stats['blocked_reasons'] = blocked_reasons

        logger.info(f"   {self.name}: {blocked_count} relationships blocked")
        if blocked_reasons:
            for reason, count in sorted(blocked_reasons.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"     - {reason}: {count}")

        return kept
