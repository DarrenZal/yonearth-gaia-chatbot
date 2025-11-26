"""
Narrative Filter Module (V1.0)

Filters out story narrative relationships from fiction books.

Filters:
- Character actions: "Leo GREETS Sophia", "Sophia EATS corn mush"
- Plot events: "X SAYS Y", "X THINKS Y", "X FEELS Y"
- Dialogue: "X EMAILS Y", "X CALLS Y"

Preserves:
- Domain knowledge: "Leo USES permaculture", "Sophia ADVOCATES_FOR sustainability"
- Real-world facts: "MIT LOCATED_IN Cambridge"

Purpose:
Separates fictional narrative from embedded domain knowledge in novels.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Set

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class NarrativeFilter(PostProcessingModule):
    """
    Filters narrative/plot relationships from fiction books.

    Content Types: Books only
    Priority: 12 (run after MetadataFilter)
    Dependencies: MetadataFilter
    """

    name = "NarrativeFilter"
    description = "Filters story plot relationships from fiction"
    content_types = ["book"]
    priority = 12
    dependencies = ["MetadataFilter"]
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Narrative action predicates (story plot, not knowledge)
        self.narrative_predicates = set(self.config.get('narrative_predicates', [
            'GREETS', 'SAYS', 'THINKS', 'FEELS', 'ASKS', 'REPLIES',
            'EATS', 'DRINKS', 'SLEEPS', 'WAKES', 'RUNS', 'WALKS',
            'EMAILS', 'CALLS', 'TEXTS', 'WRITES_TO',
            'SEES', 'HEARS', 'SMELLS', 'TOUCHES',
            'LOVES', 'HATES', 'FEARS', 'HOPES',
            'ARRIVES', 'DEPARTS', 'ENTERS', 'LEAVES',
            'GIVES', 'RECEIVES', 'TAKES', 'HOLDS'
        ]))

        # Character name patterns (common first names in fiction)
        self.common_character_names = set(self.config.get('character_names', [
            'leo', 'sophia', 'brigitte', 'charlotte', 'thompson',
            'dustin', 'sam', 'david', 'beau', 'roger', 'steve',
            'mike', 'john', 'jane', 'mary', 'robert', 'emily'
        ]))

        # Filter relationships between likely characters
        self.filter_character_relationships = self.config.get('filter_character_relationships', True)

    def is_narrative_relationship(self, rel: Any) -> bool:
        """Check if relationship is a narrative/plot action"""
        predicate = rel.relationship.upper().strip()

        # Check if predicate is a narrative action
        if predicate in self.narrative_predicates:
            return True

        return False

    def is_character_relationship(self, rel: Any) -> bool:
        """Check if relationship is between fictional characters"""
        if not self.filter_character_relationships:
            return False

        source_lower = rel.source.lower()
        target_lower = rel.target.lower()

        # Check if both are likely character names (short, common first names)
        source_is_character = (
            source_lower in self.common_character_names or
            (len(rel.source.split()) == 1 and len(rel.source) < 12)  # Single short name
        )

        target_is_character = (
            target_lower in self.common_character_names or
            (len(rel.target.split()) == 1 and len(rel.target) < 12)
        )

        # If both are character names AND it's an interpersonal relationship
        interpersonal_predicates = {'GREETS', 'MEETS', 'TALKS_TO', 'COLLABORATES_WITH'}
        if source_is_character and target_is_character:
            if rel.relationship.upper() in interpersonal_predicates:
                return True

        return False

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Filter narrative relationships, preserving domain knowledge"""

        self.stats['processed_count'] = len(relationships)

        kept = []
        filtered_count = 0

        for rel in relationships:
            # Check if it's a narrative relationship
            is_narrative = self.is_narrative_relationship(rel)
            is_character = self.is_character_relationship(rel)

            if is_narrative or is_character:
                # Mark as filtered
                if not hasattr(rel, 'flags') or rel.flags is None:
                    rel.flags = {}
                rel.flags['NARRATIVE_FILTERED'] = True
                if is_narrative:
                    rel.flags['filter_reason'] = 'Story narrative action'
                else:
                    rel.flags['filter_reason'] = 'Character interaction'

                filtered_count += 1
                self.stats['modified_count'] += 1

                logger.debug(
                    f"Filtered narrative: ({rel.source}) --[{rel.relationship}]--> ({rel.target})"
                )
            else:
                # Keep relationship (domain knowledge)
                kept.append(rel)

        # Update stats
        self.stats['filtered'] = filtered_count

        logger.info(
            f"   {self.name}: {filtered_count} narrative relationships filtered, "
            f"{len(kept)} domain relationships kept"
        )

        return kept
