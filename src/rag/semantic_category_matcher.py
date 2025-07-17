"""
Semantic Category Matcher - Enables intelligent category matching using embeddings
"""
import logging
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json
import os
from pathlib import Path
import openai

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
        self.embedding_cache_path = Path("/root/yonearth-gaia-chatbot/data/processed/category_embeddings.json")
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """
        Initialize category embeddings using OpenAI embeddings API
        Loads from cache if available, otherwise creates new embeddings
        """
        logger.info("Initializing semantic category matcher...")
        
        # Try to load from cache first
        if self.embedding_cache_path.exists():
            try:
                with open(self.embedding_cache_path, 'r') as f:
                    cache_data = json.load(f)
                    self.category_embeddings = cache_data.get('embeddings', {})
                    logger.info(f"Loaded {len(self.category_embeddings)} category embeddings from cache")
                    return
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
        
        # Create new embeddings
        self.category_embeddings = self._create_openai_embeddings()
        
        # Save to cache
        self._save_embeddings_cache()
    
    def _create_openai_embeddings(self) -> Dict[str, List[float]]:
        """
        Create OpenAI embeddings for each category
        """
        logger.info("Creating OpenAI embeddings for categories...")
        embeddings = {}
        
        for category, description in self.CATEGORY_DESCRIPTIONS.items():
            # Combine category name and description for richer embedding
            text = f"{category}: {description}"
            
            try:
                response = self.openai_client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=text
                )
                embeddings[category] = response.data[0].embedding
                logger.debug(f"Created embedding for {category}")
            except Exception as e:
                logger.error(f"Failed to create embedding for {category}: {e}")
                # Fallback to empty embedding
                embeddings[category] = [0.0] * 1536  # Default embedding size
        
        logger.info(f"Created embeddings for {len(embeddings)} categories")
        return embeddings
    
    def _save_embeddings_cache(self):
        """
        Save embeddings to cache file
        """
        try:
            # Ensure directory exists
            self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'embeddings': self.category_embeddings,
                'descriptions': self.CATEGORY_DESCRIPTIONS
            }
            
            with open(self.embedding_cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"Saved embeddings cache to {self.embedding_cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
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
        matches = []
        
        # Get query embedding
        try:
            query_response = self.openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=query
            )
            query_embedding = query_response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}")
            # Fallback to keyword matching
            return self._fallback_keyword_matching(query, threshold, max_matches)
        
        # Calculate cosine similarity with each category
        for category, category_embedding in self.category_embeddings.items():
            if not category_embedding or len(category_embedding) == 0:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, category_embedding)
            
            # Boost score if category name is directly mentioned
            if category.lower() in query_lower:
                similarity = min(1.0, similarity + 0.2)
            
            # Extract matched terms from description for explanation
            desc_terms = self.CATEGORY_DESCRIPTIONS.get(category, '').lower().split()
            query_terms = set(query_lower.split())
            matched_terms = list(query_terms.intersection(set(desc_terms)))
            
            if similarity >= threshold:
                matches.append(CategoryMatch(
                    category=category,
                    similarity=similarity,
                    matched_terms=matched_terms if matched_terms else ['semantic similarity']
                ))
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x.similarity, reverse=True)
        
        # Apply special rules for specific queries
        matches = self._apply_special_rules(query_lower, matches)
        
        return matches[:max_matches]
    
    def _fallback_keyword_matching(self, query: str, threshold: float, max_matches: int) -> List[CategoryMatch]:
        """
        Fallback to keyword matching if embedding fails
        """
        logger.warning("Using fallback keyword matching")
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        matches = []
        
        for category, description in self.CATEGORY_DESCRIPTIONS.items():
            desc_terms = set(description.lower().split())
            common_terms = query_terms.intersection(desc_terms)
            
            if common_terms:
                # Simple Jaccard similarity
                similarity = len(common_terms) / len(query_terms.union(desc_terms))
                
                if category.lower() in query_lower:
                    similarity = min(1.0, similarity + 0.3)
                
                if similarity >= threshold:
                    matches.append(CategoryMatch(
                        category=category,
                        similarity=similarity,
                        matched_terms=list(common_terms)
                    ))
        
        matches.sort(key=lambda x: x.similarity, reverse=True)
        
        # Apply special rules for fallback matching too
        matches = self._apply_special_rules(query.lower(), matches)
        
        return matches[:max_matches]
    
    def _apply_special_rules(self, query: str, matches: List[CategoryMatch]) -> List[CategoryMatch]:
        """Apply domain-specific rules for better matching"""
        
        # Special case: "soil" or "dirt" should always match BIOCHAR
        has_soil_dirt = 'soil' in query or 'dirt' in query
        has_biochar = any(m.category == 'BIOCHAR' for m in matches)
        
        if has_soil_dirt and not has_biochar:
            # BIOCHAR is highly relevant to soil health
            matches.append(CategoryMatch(
                category='BIOCHAR',
                similarity=0.72,  # Reasonable semantic score
                matched_terms=['soil', 'carbon', 'sequestration']
            ))
            logger.debug(f"Applied special rule: added BIOCHAR for soil/dirt query: '{query}'")
        
        # Special case: "carbon" related queries should match BIOCHAR
        if any(term in query for term in ['carbon', 'sequester', 'capture carbon']) and not any(m.category == 'BIOCHAR' for m in matches):
            matches.append(CategoryMatch(
                category='BIOCHAR',
                similarity=0.85,
                matched_terms=['carbon', 'sequestration', 'climate']
            ))
        
        # Special case: "healing" should match HERBAL MEDICINE
        if ('healing' in query or 'heal' in query or 'medicine' in query) and not any(m.category == 'HERBAL MEDICINE' for m in matches):
            matches.append(CategoryMatch(
                category='HERBAL MEDICINE',
                similarity=0.78,
                matched_terms=['healing', 'natural', 'medicine']
            ))
        
        # Special case: "farm" or "grow" should match FARMING & FOOD
        if any(term in query for term in ['farm', 'grow', 'crop', 'harvest']) and not any(m.category == 'FARMING & FOOD' for m in matches):
            matches.append(CategoryMatch(
                category='FARMING & FOOD',
                similarity=0.76,
                matched_terms=['farming', 'growing', 'food']
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