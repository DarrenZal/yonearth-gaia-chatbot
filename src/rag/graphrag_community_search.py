"""
GraphRAG Community Search - Global search using pre-computed community summaries.

Implements Microsoft GraphRAG's "Global Search" approach:
- Load hierarchical community summaries (Level 1: 573 clusters, Level 2: 73 clusters)
- Generate/cache embeddings for community summaries
- Semantic search to find relevant communities for a query
- Token-overlap validation to prevent false matches
- Return community context for response generation
"""
import json
import logging
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np

from openai import OpenAI

from ..config import settings

logger = logging.getLogger(__name__)

# Common stopwords to exclude from token overlap calculation
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what',
    'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
}


@dataclass
class CommunityContext:
    """Context from a matched community cluster"""
    id: str
    name: str
    title: str
    summary: str
    level: int
    entity_count: int
    relevance_score: float
    entities: List[str]  # Sample entity names


class GraphRAGCommunitySearch:
    """
    Global search using community summaries from the GraphRAG hierarchy.

    The hierarchy contains:
    - Level 0: 26,219 individual entities
    - Level 1: 573 fine-grained community clusters with summaries
    - Level 2: 73 broader theme clusters with summaries
    - Level 3: Reserved for super-clusters (currently empty)
    """

    def __init__(
        self,
        hierarchy_path: Optional[str] = None,
        embeddings_cache_path: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.hierarchy_path = hierarchy_path or "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"
        self.embeddings_cache_path = embeddings_cache_path or "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/community_embeddings.json"
        self.embedding_model = embedding_model

        self.hierarchy = None
        self.community_embeddings = {}  # {cluster_id: embedding_vector}
        self.communities_by_level = {1: [], 2: [], 3: []}  # Cached community metadata

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.is_initialized = False

        self.stats = {
            'total_searches': 0,
            'level_1_searches': 0,
            'level_2_searches': 0,
            'cache_hits': 0,
            'embeddings_generated': 0
        }

    def initialize(self, force_rebuild_embeddings: bool = False):
        """Load hierarchy and community embeddings"""
        logger.info("Initializing GraphRAG Community Search...")

        # Load hierarchy JSON
        logger.info(f"Loading hierarchy from {self.hierarchy_path}")
        start_time = time.time()

        with open(self.hierarchy_path, 'r', encoding='utf-8') as f:
            self.hierarchy = json.load(f)

        load_time = time.time() - start_time
        logger.info(f"Hierarchy loaded in {load_time:.2f}s")

        # Extract community metadata
        self._extract_communities()

        # Load or generate embeddings for community summaries
        if not force_rebuild_embeddings and os.path.exists(self.embeddings_cache_path):
            self._load_embeddings_cache()
        else:
            self._generate_community_embeddings()

        self.is_initialized = True
        logger.info(f"GraphRAG Community Search initialized with {len(self.communities_by_level[1])} L1 and {len(self.communities_by_level[2])} L2 communities")

    def _extract_communities(self):
        """Extract community metadata from hierarchy"""
        clusters = self.hierarchy.get('clusters', {})

        for level in [1, 2, 3]:
            level_key = f"level_{level}"
            level_clusters = clusters.get(level_key, {})

            if isinstance(level_clusters, dict):
                cluster_list = list(level_clusters.values())
            else:
                cluster_list = level_clusters

            for cluster in cluster_list:
                community_meta = {
                    'id': cluster.get('id', ''),
                    'name': cluster.get('name', ''),
                    'title': cluster.get('title', ''),
                    'summary_text': cluster.get('summary_text', ''),
                    'entity_count': len(cluster.get('entities', [])),
                    'entities': cluster.get('entities', [])[:10],  # Sample entities
                    'level': level
                }

                # Only include communities with summaries
                if community_meta['summary_text']:
                    self.communities_by_level[level].append(community_meta)

        logger.info(f"Extracted communities - L1: {len(self.communities_by_level[1])}, L2: {len(self.communities_by_level[2])}, L3: {len(self.communities_by_level[3])}")

    def _load_embeddings_cache(self):
        """Load cached community embeddings"""
        logger.info(f"Loading embeddings cache from {self.embeddings_cache_path}")

        try:
            with open(self.embeddings_cache_path, 'r') as f:
                cache_data = json.load(f)

            self.community_embeddings = {
                k: np.array(v) for k, v in cache_data.get('embeddings', {}).items()
            }

            self.stats['cache_hits'] = len(self.community_embeddings)
            logger.info(f"Loaded {len(self.community_embeddings)} cached embeddings")

        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
            self._generate_community_embeddings()

    def _generate_community_embeddings(self):
        """Generate embeddings for all community summaries"""
        logger.info("Generating community embeddings...")

        all_communities = []
        for level in [1, 2]:
            all_communities.extend(self.communities_by_level[level])

        if not all_communities:
            logger.warning("No communities with summaries found")
            return

        # Generate embeddings in batches
        batch_size = 100
        total = len(all_communities)

        for i in range(0, total, batch_size):
            batch = all_communities[i:i+batch_size]
            texts = [c['summary_text'] for c in batch]

            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )

                for j, embedding_data in enumerate(response.data):
                    cluster_id = batch[j]['id']
                    self.community_embeddings[cluster_id] = np.array(embedding_data.embedding)

                self.stats['embeddings_generated'] += len(batch)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i}: {e}")

        # Cache embeddings
        self._save_embeddings_cache()

    def _save_embeddings_cache(self):
        """Save community embeddings to cache file"""
        cache_data = {
            'model': self.embedding_model,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'embeddings': {
                k: v.tolist() for k, v in self.community_embeddings.items()
            }
        }

        try:
            with open(self.embeddings_cache_path, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Saved {len(self.community_embeddings)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text into meaningful words.

        Removes stopwords and short tokens for overlap calculation.
        """
        # Lowercase and extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # Filter stopwords and short words
        return {w for w in words if w not in STOPWORDS and len(w) >= 3}

    def _calculate_token_overlap(self, query: str, community_text: str) -> float:
        """
        Calculate token overlap score between query and community text.

        Returns a score between 0 and 1 based on what fraction of
        query tokens appear in the community text.
        """
        query_tokens = self._tokenize(query)
        community_tokens = self._tokenize(community_text)

        if not query_tokens:
            return 0.0

        # Count how many query tokens appear in community text
        overlap = query_tokens & community_tokens
        overlap_score = len(overlap) / len(query_tokens)

        return overlap_score

    def _is_person_or_org_query(self, query: str) -> bool:
        """
        Detect if query is asking about a specific person or organization.

        These queries need stricter matching to avoid name collisions.
        """
        query_lower = query.lower()

        # Common patterns for person/org queries
        person_patterns = [
            r'\bwho is\b',
            r'\btell me about\b',
            r'\bwhat is (\w+\s+){1,3}(institute|foundation|organization|company|group)\b',
            r'\bwhat does (\w+\s+){1,3}do\b',
        ]

        for pattern in person_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check for capitalized proper nouns (likely names)
        # If query has multiple capitalized words, likely asking about person/org
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        if len(capitalized_words) >= 2:
            return True

        return False

    def search(
        self,
        query: str,
        level: int = 1,
        k: int = 5,
        min_score: float = 0.3,
        require_token_overlap: bool = True,
        min_overlap_score: float = 0.0
    ) -> List[CommunityContext]:
        """
        Search for relevant communities using semantic similarity with token overlap validation.

        Args:
            query: User's question
            level: Community level to search (1=fine-grained, 2=themes)
            k: Number of communities to return
            min_score: Minimum semantic similarity score threshold
            require_token_overlap: Whether to require token overlap for validation
            min_overlap_score: Minimum token overlap score (0.0-1.0)

        Returns:
            List of CommunityContext objects with matched communities
        """
        if not self.is_initialized:
            raise RuntimeError("GraphRAG Community Search not initialized")

        self.stats['total_searches'] += 1
        if level == 1:
            self.stats['level_1_searches'] += 1
        else:
            self.stats['level_2_searches'] += 1

        # Detect query type for adaptive thresholds
        is_person_query = self._is_person_or_org_query(query)

        # Raise min_score for person/org queries to avoid name collisions
        effective_min_score = min_score
        effective_min_overlap = min_overlap_score

        if is_person_query and level == 1:
            # Stricter thresholds for person/org queries at L1
            effective_min_score = max(min_score, 0.4)
            effective_min_overlap = max(min_overlap_score, 0.2)
            logger.debug(f"Person/org query detected - raising thresholds to score={effective_min_score}, overlap={effective_min_overlap}")

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Score all communities at the specified level
        scored_communities = []

        for community in self.communities_by_level[level]:
            cluster_id = community['id']

            if cluster_id not in self.community_embeddings:
                continue

            community_embedding = self.community_embeddings[cluster_id]
            embedding_score = self._cosine_similarity(query_embedding, community_embedding)

            # Skip if below minimum embedding score
            if embedding_score < effective_min_score:
                continue

            # Calculate token overlap if required
            overlap_score = 0.0
            if require_token_overlap:
                # Combine title and summary for overlap calculation
                community_text = f"{community.get('title', '')} {community.get('summary_text', '')}"
                overlap_score = self._calculate_token_overlap(query, community_text)

                # For person/org queries, require meaningful overlap to prevent name collisions
                if is_person_query and overlap_score < effective_min_overlap:
                    logger.debug(f"Skipping community '{community.get('title', '')}' - low overlap ({overlap_score:.2f}) for person query")
                    continue

            # Combine scores: embedding score is primary, overlap is a boost/filter
            # Combined score = embedding_score * (1 + overlap_bonus)
            # This way, overlap improves ranking but embedding is primary
            overlap_bonus = overlap_score * 0.3  # Up to 30% boost for perfect overlap
            combined_score = embedding_score * (1 + overlap_bonus)

            scored_communities.append((community, combined_score, embedding_score, overlap_score))

        # Sort by combined score and take top k
        scored_communities.sort(key=lambda x: x[1], reverse=True)
        top_communities = scored_communities[:k]

        # Convert to CommunityContext objects
        results = []
        for community, combined_score, embedding_score, overlap_score in top_communities:
            ctx = CommunityContext(
                id=community['id'],
                name=community['name'],
                title=community['title'],
                summary=community['summary_text'],
                level=community['level'],
                entity_count=community['entity_count'],
                relevance_score=combined_score,  # Use combined score
                entities=community['entities']
            )
            results.append(ctx)
            logger.debug(f"Community '{community.get('title', '')}': embedding={embedding_score:.3f}, overlap={overlap_score:.3f}, combined={combined_score:.3f}")

        logger.info(f"Community search returned {len(results)} results for query: {query[:50]}...")
        return results

    def search_multi_level(
        self,
        query: str,
        k_level1: int = 3,
        k_level2: int = 2,
        min_score: float = 0.3
    ) -> Dict[str, List[CommunityContext]]:
        """
        Search across multiple community levels for comprehensive coverage.

        Returns results from both fine-grained (L1) and thematic (L2) clusters.
        """
        results = {
            'level_1': self.search(query, level=1, k=k_level1, min_score=min_score),
            'level_2': self.search(query, level=2, k=k_level2, min_score=min_score)
        }

        return results

    def get_community_by_id(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Get full community details by ID"""
        if not self.is_initialized:
            return None

        clusters = self.hierarchy.get('clusters', {})

        # Determine level from ID pattern
        for level in [1, 2, 3]:
            level_key = f"level_{level}"
            level_clusters = clusters.get(level_key, {})

            if isinstance(level_clusters, dict):
                if cluster_id in level_clusters:
                    return level_clusters[cluster_id]
            else:
                for cluster in level_clusters:
                    if cluster.get('id') == cluster_id:
                        return cluster

        return None

    def format_community_context(
        self,
        communities: List[CommunityContext],
        max_summary_length: int = 500
    ) -> str:
        """
        Format community contexts into a string for LLM consumption.

        Args:
            communities: List of matched communities
            max_summary_length: Maximum characters per summary

        Returns:
            Formatted string with community summaries
        """
        if not communities:
            return ""

        context_parts = []

        for i, comm in enumerate(communities, 1):
            summary = comm.summary[:max_summary_length]
            if len(comm.summary) > max_summary_length:
                summary += "..."

            part = f"""
Community {i}: {comm.title or comm.name}
Level: {comm.level} ({"Fine-grained topic" if comm.level == 1 else "Broader theme"})
Entities: {comm.entity_count} related concepts
Relevance: {comm.relevance_score:.2%}
Summary: {summary}
"""
            context_parts.append(part)

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            **self.stats,
            'communities_loaded': {
                'level_1': len(self.communities_by_level[1]),
                'level_2': len(self.communities_by_level[2]),
                'level_3': len(self.communities_by_level[3])
            },
            'embeddings_cached': len(self.community_embeddings)
        }


# Singleton instance for reuse
_community_search_instance: Optional[GraphRAGCommunitySearch] = None


def get_community_search() -> GraphRAGCommunitySearch:
    """Get or create the community search singleton"""
    global _community_search_instance

    if _community_search_instance is None:
        _community_search_instance = GraphRAGCommunitySearch()
        _community_search_instance.initialize()

    return _community_search_instance
