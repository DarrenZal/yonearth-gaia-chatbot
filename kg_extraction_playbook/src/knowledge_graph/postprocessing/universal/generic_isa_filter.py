"""
Post-processing module to filter uninformative 'is-a' relationships with generic targets.
"""

import logging
from typing import List, Dict, Any
from knowledge_graph.postprocessing.base import PostProcessingModule

logger = logging.getLogger(__name__)


class GenericIsAFilter(PostProcessingModule):
    """
    Filters 'is-a' relationships where the target is too generic to add taxonomic value.
    
    Examples of filtered relationships:
    - "Permaculture Designer's Manual is-a handbook" (too generic)
    - "Gaia Education is-a compass" (metaphorical, not taxonomic)
    
    Examples of kept relationships:
    - "Permaculture Designer's Manual is-a permaculture handbook" (specific subtype)
    """
    
    GENERIC_TARGETS = [
        "tool",
        "resource",
        "handbook",
        "guide",
        "manual",
        "book",
        "framework",
        "approach",
        "method",
        "way",
        "process",
        "mission",
        "quest",
        "journey",
        "path",
    ]
    
    METAPHORICAL_TARGETS = [
        "compass",
        "road-map",
        "roadmap",
        "beacon",
        "light",
        "bridge",
    ]
    
    def __init__(self):
        """Initialize the GenericIsAFilter."""
        super().__init__()
        self.filtered_count = 0
        self.all_generic_targets = set(
            [t.lower() for t in self.GENERIC_TARGETS] +
            [t.lower() for t in self.METAPHORICAL_TARGETS]
        )
    
    def process_batch(
        self, 
        relationships: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter out 'is-a' relationships with generic or metaphorical targets.
        
        Args:
            relationships: List of relationship dictionaries
            metadata: Batch metadata
            
        Returns:
            Filtered list of relationships
        """
        filtered_relationships = []
        
        for rel in relationships:
            if self._should_filter(rel):
                self.filtered_count += 1
                logger.debug(
                    f"Filtered generic is-a: '{rel.get('source')}' is-a '{rel.get('target')}' "
                    f"(reason: generic target)"
                )
            else:
                filtered_relationships.append(rel)
        
        if self.filtered_count > 0:
            logger.info(
                f"GenericIsAFilter: Filtered {self.filtered_count} generic 'is-a' relationships"
            )
        
        return filtered_relationships
    
    def _should_filter(self, relationship: Dict[str, Any]) -> bool:
        """
        Determine if a relationship should be filtered.
        
        Args:
            relationship: Relationship dictionary
            
        Returns:
            True if relationship should be filtered, False otherwise
        """
        predicate = relationship.get("predicate", "").lower()
        
        # Only filter 'is-a' relationships
        if predicate != "is-a":
            return False
        
        target = relationship.get("target", "").strip()
        if not target:
            return False
        
        target_lower = target.lower()
        
        # Check if target is exactly a generic term
        if target_lower in self.all_generic_targets:
            return True
        
        # Check for multi-word targets with specificity
        target_words = target_lower.split()
        
        if len(target_words) >= 2:
            # Check if last word is generic (e.g., "permaculture handbook")
            last_word = target_words[-1]
            if last_word in self.all_generic_targets:
                # Check if there's a specific modifier
                # If first word(s) add specificity, keep it
                modifier = " ".join(target_words[:-1])
                if self._is_specific_modifier(modifier):
                    return False
                else:
                    # Generic modifier or no real specificity
                    return True
            
            # Check if first word is generic (e.g., "handbook for permaculture")
            first_word = target_words[0]
            if first_word in self.all_generic_targets:
                # This pattern is less common but still generic
                return True
        
        return False
    
    def _is_specific_modifier(self, modifier: str) -> bool:
        """
        Check if a modifier adds meaningful specificity.
        
        Args:
            modifier: The modifier string (e.g., "permaculture" in "permaculture handbook")
            
        Returns:
            True if modifier is specific, False if generic
        """
        if not modifier:
            return False
        
        # List of generic modifiers that don't add real specificity
        generic_modifiers = {
            "comprehensive",
            "complete",
            "essential",
            "practical",
            "useful",
            "helpful",
            "important",
            "key",
            "main",
            "primary",
            "basic",
            "advanced",
            "ultimate",
            "definitive",
            "authoritative",
        }
        
        modifier_lower = modifier.lower().strip()
        
        # If modifier is in generic list, it's not specific
        if modifier_lower in generic_modifiers:
            return False
        
        # If modifier has substance (not just articles/prepositions), consider it specific
        # This catches domain-specific terms like "permaculture", "regenerative", etc.
        return len(modifier_lower) > 2
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about filtering operations.
        
        Returns:
            Dictionary with filtering statistics
        """
        return {
            "filtered_count": self.filtered_count,
            "generic_targets": len(self.GENERIC_TARGETS),
            "metaphorical_targets": len(self.METAPHORICAL_TARGETS),
        }