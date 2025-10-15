import logging
from typing import List, Dict, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class PredicateNormalizer:
    """Normalizes predicate variations to canonical forms using mapping dictionary."""
    
    def __init__(self, config_path: str = "config/predicate_mappings.yaml"):
        """
        Initialize the predicate normalizer.
        
        Args:
            config_path: Path to the YAML file containing predicate mappings
        """
        self.config_path = Path(config_path)
        self.mappings = self._load_mappings()
        self.normalization_count = 0
        self.normalization_stats = {}
        
    def _load_mappings(self) -> Dict[str, str]:
        """
        Load predicate mappings from YAML configuration file.
        
        Returns:
            Dictionary mapping variant predicates to canonical forms
        """
        if not self.config_path.exists():
            logger.warning(f"Predicate mappings file not found: {self.config_path}")
            return self._get_default_mappings()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                mappings = config.get('predicate_mappings', {})
                logger.info(f"Loaded {len(mappings)} predicate mappings from {self.config_path}")
                return mappings
        except Exception as e:
            logger.error(f"Error loading predicate mappings: {e}")
            return self._get_default_mappings()
    
    def _get_default_mappings(self) -> Dict[str, str]:
        """
        Get default predicate mappings if config file is not available.
        
        Returns:
            Dictionary of default predicate mappings
        """
        return {
            "is-a": "is",
            "is a": "is",
            "are": "is",
            "were": "was",
            "is about": "discusses",
            "is related to": "relates to",
            "is in": "located in",
            "is within": "located in",
            "is key to": "enables",
            "is essential to": "enables",
            "is critical to": "enables",
            "is part of": "part of",
            "is a part of": "part of",
            "belongs to": "part of",
            "has": "contains",
            "includes": "contains",
            "comprises": "contains",
            "consists of": "contains",
            "is composed of": "contains",
            "is made up of": "contains",
            "leads to": "causes",
            "results in": "causes",
            "brings about": "causes",
            "is caused by": "caused by",
            "stems from": "caused by",
            "originates from": "caused by",
            "is associated with": "associated with",
            "is linked to": "associated with",
            "is connected to": "associated with",
            "is similar to": "similar to",
            "resembles": "similar to",
            "is like": "similar to",
        }
    
    def normalize_predicate(self, predicate: str) -> tuple[str, bool]:
        """
        Normalize a single predicate to its canonical form.
        
        Args:
            predicate: The predicate to normalize
            
        Returns:
            Tuple of (normalized_predicate, was_normalized)
        """
        if not predicate:
            return predicate, False
        
        # Case-insensitive matching
        predicate_lower = predicate.lower().strip()
        
        if predicate_lower in self.mappings:
            canonical = self.mappings[predicate_lower]
            return canonical, True
        
        return predicate, False
    
    def process_batch(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of relationships and normalize their predicates.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            List of relationships with normalized predicates
        """
        if not relationships:
            return relationships
        
        normalized_relationships = []
        batch_normalization_count = 0
        
        for rel in relationships:
            if not isinstance(rel, dict):
                normalized_relationships.append(rel)
                continue
            
            predicate = rel.get('predicate', '')
            
            if not predicate:
                normalized_relationships.append(rel)
                continue
            
            normalized_predicate, was_normalized = self.normalize_predicate(predicate)
            
            if was_normalized:
                # Create a copy to avoid modifying original
                normalized_rel = rel.copy()
                
                # Preserve original predicate in metadata
                if 'metadata' not in normalized_rel:
                    normalized_rel['metadata'] = {}
                
                normalized_rel['metadata']['original_predicate'] = predicate
                normalized_rel['predicate'] = normalized_predicate
                
                # Update statistics
                batch_normalization_count += 1
                self.normalization_count += 1
                
                # Track which predicates were normalized to what
                if predicate not in self.normalization_stats:
                    self.normalization_stats[predicate] = {
                        'canonical': normalized_predicate,
                        'count': 0
                    }
                self.normalization_stats[predicate]['count'] += 1
                
                logger.debug(f"Normalized predicate: '{predicate}' -> '{normalized_predicate}'")
                normalized_relationships.append(normalized_rel)
            else:
                normalized_relationships.append(rel)
        
        if batch_normalization_count > 0:
            logger.info(f"Normalized {batch_normalization_count} predicates in batch of {len(relationships)}")
        
        return normalized_relationships
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about predicate normalizations performed.
        
        Returns:
            Dictionary containing normalization statistics
        """
        return {
            'total_normalizations': self.normalization_count,
            'unique_predicates_normalized': len(self.normalization_stats),
            'normalization_details': self.normalization_stats
        }
    
    def log_statistics(self):
        """Log summary statistics about normalizations performed."""
        stats = self.get_statistics()
        logger.info(f"Predicate Normalization Statistics:")
        logger.info(f"  Total normalizations: {stats['total_normalizations']}")
        logger.info(f"  Unique predicates normalized: {stats['unique_predicates_normalized']}")
        
        if stats['normalization_details']:
            logger.info("  Top normalized predicates:")
            sorted_predicates = sorted(
                stats['normalization_details'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            for predicate, info in sorted_predicates[:10]:
                logger.info(f"    '{predicate}' -> '{info['canonical']}': {info['count']} times")
    
    def reset_statistics(self):
        """Reset normalization statistics."""
        self.normalization_count = 0
        self.normalization_stats = {}