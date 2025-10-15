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
    description = "Filters overly vague/abstract entities that couldn't be resolved"
    content_types = ["all"]
    priority = 85  # V14.3.2: Moved from 30 to 85 to run AFTER ContextEnricher
    dependencies = ["ContextEnricher"]  # V14.3.2: Requires ContextEnricher to run first
    version = "1.1.0"  # V14.3.2: Priority reordering + flag-based blocking

    SPECIFICITY_THRESHOLD = 0.90
    
    ABSTRACT_PATTERNS = [
        'the answer',
        'the way',
        'the solution',
        'the process',
        'aspects of',
        'things',
        'matters'
    ]
    
    DEMONSTRATIVE_PATTERNS = ['this', 'that', 'these', 'those']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Patterns for entities that are too vague to be useful
        self.vague_abstract_patterns = self.config.get('vague_abstract_patterns', [
            # V14.3.2 NEW: Specific vague terms from V14.3.1 analysis
            r'^unknown$',  # "published by unknown"
            r'^(community|personal) (activities|life-hacks|practices)$',  # "community activities", "personal life-hacks"
            r'^(activities|life-hacks|practices)$',  # Generic activities
            r'^(poisonous chemical inputs|ammunition manufacturers|hazardous materials suit|spraying chemicals)$',  # Too generic

            # Original patterns
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

    def calculate_specificity_score(self, entity: str) -> float:
        """
        Calculate specificity score for an entity with penalty rules.
        
        Returns:
            Score from 0.0 (very vague) to 1.0 (very specific)
        """
        entity_lower = entity.lower().strip()
        score = 1.0
        
        # Check for demonstrative pronouns (as standalone words)
        entity_words = entity_lower.split()
        for word in entity_words:
            if word in self.DEMONSTRATIVE_PATTERNS:
                score -= 0.15
                logger.debug(f"Applied -0.15 penalty for demonstrative pronoun in '{entity}'")
                break
        
        # Check for abstract patterns
        for pattern in self.ABSTRACT_PATTERNS:
            if pattern in entity_lower:
                score -= 0.20
                logger.debug(f"Applied -0.20 penalty for abstract pattern '{pattern}' in '{entity}'")
                break
        
        # Check for possessive pronouns
        possessive_patterns = [r'\b(my|your|his|her|its|our|their)\b']
        for pattern in possessive_patterns:
            if re.search(pattern, entity_lower):
                score -= 0.10
                logger.debug(f"Applied -0.10 penalty for possessive pronoun in '{entity}'")
                break
        
        return max(0.0, score)

    def is_too_vague(self, entity: str) -> Tuple[bool, str]:
        """
        Check if entity is too vague.

        Returns:
            (is_vague: bool, pattern_matched: str)
        """
        entity_lower = entity.lower().strip()

        # Check specificity score first
        specificity_score = self.calculate_specificity_score(entity)
        if specificity_score < self.SPECIFICITY_THRESHOLD:
            return True, f"low_specificity_score: {specificity_score:.2f}"

        # Check against all patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.match(entity_lower):
                return True, self.vague_abstract_patterns[i]

        return False, ""

    def should_block_relationship(self, rel: Any) -> Tuple[bool, str]:
        """
        V14.3.2 ENHANCED: Determine if relationship should be blocked.

        Priority logic:
        1. Check if ContextEnricher flagged entities as vague (couldn't resolve)
        2. Fall back to specificity score and pattern matching

        This ensures we only block entities that:
        - Were identified as vague by ContextEnricher AND couldn't be resolved
        - OR fail our specificity/pattern checks

        Returns:
            (should_block: bool, reason: str)
        """
        # V14.3.2 NEW: Check if ContextEnricher flagged as unresolvable
        # If ContextEnricher set VAGUE_SOURCE/VAGUE_TARGET flag, that means it
        # tried to resolve the entity but FAILED. We should block these.
        if rel.flags:
            if rel.flags.get('VAGUE_SOURCE'):
                return True, f"vague_source_unresolved: {rel.source}"
            if rel.flags.get('VAGUE_TARGET'):
                return True, f"vague_target_unresolved: {rel.target}"

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