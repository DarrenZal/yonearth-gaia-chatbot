"""
Semantic Category Matcher - Enables intelligent category matching using embeddings
"""
import logging
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class CategoryMatch:
    """Represents a semantic category match"""
    category: str
    similarity: float
    matched_terms: List[str]


class SemanticCategoryMatcher:
    """
    Matches queries to categories using semantic similarity, not just keywords
    This solves the "soil" → BIOCHAR matching problem
    """
    
    # Category descriptions for richer semantic matching
    CATEGORY_DESCRIPTIONS = {
        'BIOCHAR': 'biochar carbon sequestration soil amendment pyrolysis charcoal agriculture climate carbon capture terra preta black carbon',
        'SOIL': 'soil health regeneration fertility microbiome earth ground dirt land topsoil humus organic matter',
        'HERBAL MEDICINE': 'herbs herbal medicine natural healing plants botanical remedies wellness medicinal plants traditional medicine',
        'CLIMATE & SCIENCE': 'climate change science research global warming carbon emissions greenhouse gas atmosphere temperature',
        'PERMACULTURE': 'permaculture design systems thinking sustainable agriculture food forest polyculture guilds zones',
        'REGENERATIVE': 'regenerative agriculture farming restoration healing land stewardship renewal rebuilding',
        'COMPOSTING': 'compost composting organic matter decomposition soil building waste recycling nutrients rot decay',
        'SUSTAINABILITY': 'sustainable sustainability environment ecology conservation future renewable resilient',
        'FARMING & FOOD': 'farming agriculture food production crops harvest growing cultivation planting seeds vegetables',
        'HEALTH & WELLNESS': 'health wellness wellbeing nutrition holistic healing vitality medicine prevention cure',
        'ECOLOGY & NATURE': 'ecology nature ecosystems biodiversity wildlife habitat environment natural systems',
        'COMMUNITY': 'community cooperation collaboration social connection people together collective mutual aid',
        'EDUCATION': 'education learning teaching knowledge wisdom understanding growth study school training',
        'BUSINESS': 'business enterprise economics commerce trade market entrepreneurship company profit',
        'TECHNOLOGY & MATERIALS': 'technology innovation materials engineering solutions tools equipment devices',
        'INDIGENOUS WISDOM': 'indigenous native traditional wisdom ancestral knowledge culture first nations tribal',
        'GREEN BUILDING': 'green building sustainable construction architecture eco-friendly materials efficient',
        'POLICY & GOVERNMT': 'policy government regulation legislation politics governance law rules ordinance',
        'ESOTERICA': 'spiritual esoteric mystical consciousness metaphysical sacred divine cosmic energy',
        'IMPACT INVESTING': 'impact investing finance social environmental returns conscious capital ethical money',
        'BIO-DYNAMICS': 'biodynamic agriculture holistic farming cosmic rhythms steiner preparations moon cycles',
        'GREEN FAITH': 'faith spirituality religion ecology environmental sacred earth creation care stewardship',
        'WATER': 'water hydrology watershed rain harvesting conservation flow rivers streams aquifer drought',
        'ENERGY': 'energy renewable solar wind power electricity sustainable clean fossil fuel alternative'
    }
    
    # Related category mappings (semantic relationships)
    CATEGORY_RELATIONSHIPS = {
        'SOIL': ['BIOCHAR', 'COMPOSTING', 'REGENERATIVE', 'FARMING & FOOD', 'PERMACULTURE'],
        'BIOCHAR': ['SOIL', 'CLIMATE & SCIENCE', 'REGENERATIVE', 'TECHNOLOGY & MATERIALS'],
        'HERBAL MEDICINE': ['HEALTH & WELLNESS', 'INDIGENOUS WISDOM', 'ECOLOGY & NATURE'],
        'CLIMATE & SCIENCE': ['BIOCHAR', 'SUSTAINABILITY', 'ENERGY', 'TECHNOLOGY & MATERIALS'],
        'PERMACULTURE': ['SOIL', 'REGENERATIVE', 'FARMING & FOOD', 'WATER', 'ECOLOGY & NATURE'],
        'HEALTH & WELLNESS': ['HERBAL MEDICINE', 'GREEN FAITH', 'ESOTERICA'],
    }
    
    def __init__(self, categorizer=None):
        self.categorizer = categorizer
        self.category_embeddings = {}
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """
        Initialize category embeddings (in production, would use OpenAI embeddings)
        For now, using simple keyword overlap as a proxy
        """
        logger.info("Initializing semantic category matcher...")
        # In production: self.category_embeddings = self._create_openai_embeddings()
        pass
    
    def get_semantic_category_matches(
        self, 
        query: str, 
        threshold: float = 0.7,
        max_matches: int = 5
    ) -> List[CategoryMatch]:
        """
        Find categories semantically related to the query
        
        Args:
            query: User's search query
            threshold: Minimum similarity score (0.0 to 1.0)
            max_matches: Maximum number of category matches to return
            
        Returns:
            List of CategoryMatch objects sorted by similarity
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        matches = []
        
        # Score each category based on description overlap (simplified semantic matching)
        for category, description in self.CATEGORY_DESCRIPTIONS.items():
            desc_terms = set(description.lower().split())
            
            # Calculate similarity (in production, use cosine similarity of embeddings)
            common_terms = query_terms.intersection(desc_terms)
            if common_terms:
                # Simple Jaccard similarity as proxy for semantic similarity
                similarity = len(common_terms) / len(query_terms.union(desc_terms))
                
                # Boost score if category name is mentioned
                if category.lower() in query_lower:
                    similarity = min(1.0, similarity + 0.3)
                
                # Check for related terms
                matched_terms = list(common_terms)
                
                # Add related categories with reduced score
                if category in self.CATEGORY_RELATIONSHIPS:
                    for related_cat in self.CATEGORY_RELATIONSHIPS[category]:
                        related_desc = self.CATEGORY_DESCRIPTIONS.get(related_cat, '').lower()
                        related_terms = set(related_desc.split())
                        related_common = query_terms.intersection(related_terms)
                        if related_common:
                            similarity = min(1.0, similarity + 0.1 * len(related_common))
                            matched_terms.extend(related_common)
                
                if similarity >= threshold:
                    matches.append(CategoryMatch(
                        category=category,
                        similarity=similarity,
                        matched_terms=list(set(matched_terms))
                    ))
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x.similarity, reverse=True)
        
        # Apply special rules for specific queries
        matches = self._apply_special_rules(query_lower, matches)
        
        return matches[:max_matches]
    
    def _apply_special_rules(self, query: str, matches: List[CategoryMatch]) -> List[CategoryMatch]:
        """Apply domain-specific rules for better matching"""
        
        # Special case: "soil" should always match BIOCHAR
        if 'soil' in query and not any(m.category == 'BIOCHAR' for m in matches):
            # Check if BIOCHAR would be close to threshold
            for category, description in self.CATEGORY_DESCRIPTIONS.items():
                if category == 'BIOCHAR':
                    matches.append(CategoryMatch(
                        category='BIOCHAR',
                        similarity=0.75,  # Give it a good score
                        matched_terms=['soil', 'carbon', 'agriculture']
                    ))
                    break
        
        # Special case: "healing" should match HERBAL MEDICINE
        if 'healing' in query or 'heal' in query:
            if not any(m.category == 'HERBAL MEDICINE' for m in matches):
                matches.append(CategoryMatch(
                    category='HERBAL MEDICINE',
                    similarity=0.8,
                    matched_terms=['healing', 'natural', 'medicine']
                ))
        
        # Re-sort after additions
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches
    
    def get_episodes_for_semantic_matches(
        self, 
        matches: List[CategoryMatch]
    ) -> Set[int]:
        """
        Get all episode IDs for the matched categories
        
        Args:
            matches: List of CategoryMatch objects
            
        Returns:
            Set of episode IDs
        """
        if not self.categorizer:
            logger.warning("No categorizer available for episode lookup")
            return set()
        
        episode_ids = set()
        for match in matches:
            category_episodes = self.categorizer.get_episodes_by_category(match.category)
            episode_ids.update(category_episodes)
            logger.info(f"Category {match.category} has {len(category_episodes)} episodes")
        
        return episode_ids
    
    def explain_matches(self, matches: List[CategoryMatch]) -> str:
        """
        Generate human-readable explanation of category matches
        
        Args:
            matches: List of CategoryMatch objects
            
        Returns:
            Explanation string
        """
        if not matches:
            return "No category matches found"
        
        explanations = []
        for match in matches[:3]:  # Top 3 matches
            terms = ', '.join(match.matched_terms[:3]) if match.matched_terms else 'semantic similarity'
            explanations.append(
                f"{match.category} ({match.similarity:.0%} match via {terms})"
            )
        
        return "Matched categories: " + "; ".join(explanations)


def test_semantic_matcher():
    """Test the semantic category matcher"""
    matcher = SemanticCategoryMatcher()
    
    test_queries = [
        "what makes healthy dirt",
        "natural ways to heal",
        "growing food without chemicals",
        "how to capture carbon naturally",
        "regenerative farming practices",
        "indigenous knowledge about plants",
        "community resilience building"
    ]
    
    print("Testing Semantic Category Matcher")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        matches = matcher.get_semantic_category_matches(query, threshold=0.1)
        
        if matches:
            print(matcher.explain_matches(matches))
            for match in matches:
                print(f"  - {match.category}: {match.similarity:.2%}")
        else:
            print("  No matches found")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_semantic_matcher()