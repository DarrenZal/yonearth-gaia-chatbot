import logging
from typing import List, Dict, Any
from ..base import PostProcessingModule

logger = logging.getLogger(__name__)


class GenericIsAFilter(PostProcessingModule):
    """
    Filter 'is-a' relationships where the target is too generic to add taxonomic value.
    
    Removes relationships like:
    - "X is-a tool"
    - "X is-a guide"
    - "X is-a compass" (metaphorical)
    
    But keeps specific relationships like:
    - "X is-a permaculture handbook"
    - "X is-a design framework"
    """
    
    GENERIC_TARGETS = {
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
        "path"
    }
    
    METAPHORICAL_TARGETS = {
        "compass",
        "road-map",
        "roadmap",
        "beacon",
        "light",
        "bridge"
    }
    
    SPECIFICITY_KEYWORDS = {
        "permaculture",
        "design",
        "ecological",
        "regenerative",
        "sustainable",
        "holistic",
        "systems",
        "community",
        "agricultural",
        "environmental",
        "social",
        "economic",
        "cultural",
        "educational",
        "practical",
        "technical",
        "scientific",
        "philosophical"
    }
    
    def __init__(self):
        super().__init__()
        self.filtered_count = 0
        self.filtered_relationships = []
    
    def process_batch(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out 'is-a' relationships with generic or metaphorical targets.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Filtered list of relationships
        """
        filtered = []
        
        for rel in relationships:
            if self._should_filter(rel):
                self.filtered_count += 1
                self.filtered_relationships.append({
                    "source": rel.get("source"),
                    "predicate": rel.get("predicate"),
                    "target": rel.get("target"),
                    "reason": self._get_filter_reason(rel)
                })
                logger.debug(
                    f"Filtered generic is-a: '{rel.get('source')}' {rel.get('predicate')} '{rel.get('target')}' "
                    f"(Reason: {self._get_filter_reason(rel)})"
                )
            else:
                filtered.append(rel)
        
        if self.filtered_count > 0:
            logger.info(f"GenericIsAFilter: Filtered {self.filtered_count} generic 'is-a' relationships")
        
        return filtered
    
    def _should_filter(self, rel: Dict[str, Any]) -> bool:
        """
        Determine if a relationship should be filtered.
        
        Args:
            rel: Relationship dictionary
            
        Returns:
            True if relationship should be filtered, False otherwise
        """
        predicate = rel.get("predicate", "").lower()
        
        # Only filter 'is-a' relationships
        if predicate != "is-a":
            return False
        
        target = rel.get("target", "").lower().strip()
        
        if not target:
            return False
        
        # Check if target is a single generic word
        if self._is_generic_single_word(target):
            return True
        
        # Check if target is a single metaphorical word
        if self._is_metaphorical_single_word(target):
            return True
        
        # Check multi-word targets
        if self._is_generic_multi_word(target):
            return True
        
        return False
    
    def _is_generic_single_word(self, target: str) -> bool:
        """Check if target is a single generic word."""
        words = target.split()
        if len(words) == 1:
            return target in self.GENERIC_TARGETS
        return False
    
    def _is_metaphorical_single_word(self, target: str) -> bool:
        """Check if target is a single metaphorical word."""
        words = target.split()
        if len(words) == 1:
            return target in self.METAPHORICAL_TARGETS
        return False
    
    def _is_generic_multi_word(self, target: str) -> bool:
        """
        Check if multi-word target is generic.
        
        A multi-word target is considered generic if:
        - It starts with a generic word AND
        - The remaining words don't add specificity
        """
        words = target.split()
        
        if len(words) < 2:
            return False
        
        first_word = words[0]
        remaining_words = words[1:]
        
        # If first word is not generic, it's not a generic multi-word target
        if first_word not in self.GENERIC_TARGETS and first_word not in self.METAPHORICAL_TARGETS:
            return False
        
        # Check if any remaining word adds specificity
        for word in remaining_words:
            if word in self.SPECIFICITY_KEYWORDS:
                return False
            # If word is longer than 4 chars and not a common article/preposition, consider it specific
            if len(word) > 4 and word not in {"about", "through", "within", "around"}:
                return False
        
        # All remaining words are non-specific
        return True
    
    def _get_filter_reason(self, rel: Dict[str, Any]) -> str:
        """Get the reason why a relationship was filtered."""
        target = rel.get("target", "").lower().strip()
        
        if target in self.GENERIC_TARGETS:
            return "generic_target"
        elif target in self.METAPHORICAL_TARGETS:
            return "metaphorical_target"
        else:
            words = target.split()
            if len(words) > 1:
                first_word = words[0]
                if first_word in self.GENERIC_TARGETS:
                    return "generic_multi_word_no_specificity"
                elif first_word in self.METAPHORICAL_TARGETS:
                    return "metaphorical_multi_word_no_specificity"
        
        return "generic"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about filtered relationships.
        
        Returns:
            Dictionary with filtering statistics
        """
        return {
            "filtered_count": self.filtered_count,
            "filtered_relationships": self.filtered_relationships[:100]  # Limit to first 100 for readability
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.filtered_count = 0
        self.filtered_relationships = []