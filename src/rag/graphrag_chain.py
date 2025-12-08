"""
GraphRAG Chain - DRIFT-style orchestrator combining Global and Local search.

Implements Microsoft GraphRAG's "DRIFT Search" approach:
1. Global Search: Find relevant community summaries for thematic context
2. Local Search: Extract entities and relationships for specifics
3. Combine both into rich context for response generation
4. KG-guided retrieval: Use entity/community source IDs to filter/boost chunk retrieval
5. Generate response using Gaia character with full GraphRAG context
"""
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set

from dataclasses import dataclass, field

from langchain_core.documents import Document

from ..config import settings
from ..character.gaia import GaiaCharacter
from .graphrag_community_search import GraphRAGCommunitySearch, CommunityContext, get_community_search
from .graphrag_local_search import GraphRAGLocalSearch, EntityContext, RelationshipContext, LocalSearchResult, get_local_search
from .vectorstore import YonEarthVectorStore, create_vectorstore

logger = logging.getLogger(__name__)

# KG boost factor is now configurable via settings.graphrag_kg_boost_factor
# Default 1.3x means a KG match needs ~77% of the similarity of a non-KG match to rank equal.
# Increase if KG matches aren't surfacing enough; decrease if irrelevant KG sources dominate.


@dataclass
class GraphRAGContext:
    """Combined context from GraphRAG search"""
    communities: List[CommunityContext] = field(default_factory=list)
    entities: List[EntityContext] = field(default_factory=list)
    relationships: List[RelationshipContext] = field(default_factory=list)
    source_episodes: List[str] = field(default_factory=list)
    source_books: List[str] = field(default_factory=list)
    chunks: List[Document] = field(default_factory=list)


class GraphRAGChain:
    """
    DRIFT-style GraphRAG chain combining global community search
    with local entity-centric retrieval.

    Search Modes:
    - "global": Community summaries only (best for broad thematic questions)
    - "local": Entity extraction only (best for specific queries)
    - "drift": Combines both (default, best overall)
    """

    def __init__(self, initialize_data: bool = False):
        self.community_search: Optional[GraphRAGCommunitySearch] = None
        self.local_search: Optional[GraphRAGLocalSearch] = None
        self.vectorstore: Optional[YonEarthVectorStore] = None
        self.gaia: Optional[GaiaCharacter] = None

        self.is_initialized = False

        self.stats = {
            'total_queries': 0,
            'global_searches': 0,
            'local_searches': 0,
            'drift_searches': 0,
            'total_response_time': 0.0
        }

        if initialize_data:
            self.initialize()

    def initialize(self):
        """Initialize all GraphRAG components"""
        logger.info("Initializing GraphRAG Chain...")

        try:
            # Initialize community search (global)
            logger.info("Initializing community search...")
            self.community_search = get_community_search()

            # Initialize local search
            logger.info("Initializing local search...")
            self.local_search = get_local_search()

            # Initialize vectorstore for chunk retrieval
            logger.info("Connecting to vectorstore...")
            self.vectorstore = create_vectorstore()

            # Initialize Gaia character
            logger.info("Initializing Gaia character...")
            self.gaia = GaiaCharacter()

            self.is_initialized = True
            logger.info("GraphRAG Chain initialized successfully!")

        except Exception as e:
            logger.error(f"Error initializing GraphRAG Chain: {e}")
            raise

    def _classify_query(self, query: str) -> str:
        """
        Classify query to determine best search strategy.

        Returns:
            "global" for broad/thematic questions
            "local" for specific entity questions
            "grounded" for questions requiring specific citations
            "drift" for mixed questions (default)
        """
        query_lower = query.lower()

        # Keywords suggesting broad/thematic questions (global search)
        global_indicators = [
            'what are the main', 'what themes', 'summarize', 'overview',
            'general', 'broadly', 'overall', 'key topics', 'main ideas',
            'what is discussed', 'explain the concept', 'what role does',
            'how do', 'why is', 'what is the significance'
        ]

        # Keywords suggesting specific questions (local search)
        local_indicators = [
            'who is', 'who was', 'where is', 'when did',
            'specific', 'exactly', 'particular',
            'person', 'organization', 'company', 'tell me about'
        ]

        # Keywords suggesting need for specific citations (grounded search)
        grounded_indicators = [
            'which episode', 'what episode', 'where can i find',
            'episode number', 'recommend', 'listen to',
            'which podcast', 'book chapter', 'where is this discussed'
        ]

        global_score = sum(1 for ind in global_indicators if ind in query_lower)
        local_score = sum(1 for ind in local_indicators if ind in query_lower)
        grounded_score = sum(1 for ind in grounded_indicators if ind in query_lower)

        if grounded_score >= 1:
            return "grounded"
        elif global_score > local_score + 1:
            return "global"
        elif local_score > global_score + 1:
            return "local"
        else:
            return "drift"

    def _get_search_weights(self, query_type: str) -> Dict[str, float]:
        """
        Get search weights based on query classification.

        Returns weights for:
        - community_weight: How much to rely on community summaries
        - entity_weight: How much to rely on entity extraction
        - chunk_weight: How much to rely on raw chunk retrieval
        """
        weights = {
            'global': {
                'community_weight': 0.7,
                'entity_weight': 0.2,
                'chunk_weight': 0.1,
                'k_communities': 5,
                'k_entities': 5,
                'k_chunks': 2
            },
            'local': {
                'community_weight': 0.2,
                'entity_weight': 0.6,
                'chunk_weight': 0.2,
                'k_communities': 2,
                'k_entities': 12,
                'k_chunks': 3
            },
            'grounded': {
                'community_weight': 0.2,
                'entity_weight': 0.3,
                'chunk_weight': 0.5,
                'k_communities': 2,
                'k_entities': 5,
                'k_chunks': 6
            },
            'drift': {
                'community_weight': 0.4,
                'entity_weight': 0.3,
                'chunk_weight': 0.3,
                'k_communities': 3,
                'k_entities': 8,
                'k_chunks': 4
            }
        }

        return weights.get(query_type, weights['drift'])

    def _global_search(
        self,
        query: str,
        community_level: int = 1,
        k_communities: int = 5
    ) -> List[CommunityContext]:
        """Perform global search using community summaries"""
        self.stats['global_searches'] += 1

        communities = self.community_search.search(
            query=query,
            level=community_level,
            k=k_communities,
            min_score=0.3
        )

        return communities

    def _local_search(
        self,
        query: str,
        k_entities: int = 10,
        k_relationships: int = 15
    ) -> LocalSearchResult:
        """Perform local search using entity extraction"""
        self.stats['local_searches'] += 1

        result = self.local_search.search(
            query=query,
            k_entities=k_entities,
            k_relationships=k_relationships,
            expand_neighbors=True
        )

        return result

    def _extract_source_ids(
        self,
        entities: List[EntityContext],
        communities: List[CommunityContext]
    ) -> Tuple[Set[int], Set[str]]:
        """
        Extract source episode IDs and book IDs from KG entities and communities.

        Returns:
            Tuple of (episode_ids: Set[int], book_ids: Set[str])
        """
        episode_ids = set()
        book_ids = set()

        # Extract from entities
        for entity in entities:
            for source in entity.sources:
                if source.startswith('episode_'):
                    try:
                        # Extract episode number from "episode_123" format
                        ep_num = int(source.replace('episode_', ''))
                        episode_ids.add(ep_num)
                    except ValueError:
                        pass
                elif source.startswith('book_'):
                    book_ids.add(source)

        # Extract from communities (if they have source info)
        for community in communities:
            for entity_name in community.entities[:20]:  # Sample entities
                # Look up entity in local search to get sources
                if self.local_search and entity_name in self.local_search.entities:
                    entity_data = self.local_search.entities[entity_name]
                    for source in entity_data.get('sources', []):
                        if source.startswith('episode_'):
                            try:
                                ep_num = int(source.replace('episode_', ''))
                                episode_ids.add(ep_num)
                            except ValueError:
                                pass
                        elif source.startswith('book_'):
                            book_ids.add(source)

        logger.debug(f"Extracted {len(episode_ids)} episode IDs and {len(book_ids)} book IDs from KG")
        return episode_ids, book_ids

    def _retrieve_chunks(
        self,
        query: str,
        k: int = 5,
        kg_episode_ids: Optional[Set[int]] = None,
        kg_book_ids: Optional[Set[str]] = None
    ) -> List[Document]:
        """
        Retrieve relevant chunks from vectorstore, optionally filtered/boosted by KG sources.

        Args:
            query: User's question
            k: Number of chunks to retrieve
            kg_episode_ids: Episode IDs from KG entities/communities to prioritize
            kg_book_ids: Book IDs from KG entities/communities to prioritize
        """
        if not self.vectorstore:
            return []

        try:
            # Get more results than needed so we can filter/rerank
            search_k = k * 3 if kg_episode_ids or kg_book_ids else k

            results = self.vectorstore.similarity_search_with_score(query, k=search_k)

            if not results:
                return []

            # If we have KG source IDs, boost matching chunks
            # IMPORTANT: This assumes Pinecone index uses cosine metric (higher = more similar).
            # If index metric changes to Euclidean/dot-product, revisit boost math and sort order.
            # Current index: yonearth-episodes with cosine metric, 1536 dimensions.
            if kg_episode_ids or kg_book_ids:
                scored_docs = []

                for doc, score in results:
                    metadata = doc.metadata or {}

                    # Check if this chunk is from a KG-matched source
                    kg_boost = 1.0
                    match_reason = None

                    # Check episode match - guard against non-numeric values
                    episode_num = metadata.get('episode_number')
                    if episode_num and kg_episode_ids:
                        try:
                            if int(episode_num) in kg_episode_ids:
                                kg_boost = settings.graphrag_kg_boost_factor
                                match_reason = f"Episode {episode_num} (KG match)"
                        except (ValueError, TypeError):
                            pass  # Skip non-numeric episode numbers

                    # Check book match
                    content_type = metadata.get('content_type')
                    book_id = metadata.get('book_id', '')
                    if content_type == 'book' and kg_book_ids and book_id in kg_book_ids:
                        kg_boost = settings.graphrag_kg_boost_factor
                        match_reason = f"Book (KG match)"

                    # Apply boost to score (higher is better for cosine similarity, so multiply)
                    boosted_score = score * kg_boost

                    scored_docs.append((doc, boosted_score, match_reason))

                # Sort by boosted score descending (higher = more similar)
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # Log KG boost effectiveness for monitoring
                # Track: how many KG matches made it to top-k, and their rank positions
                kg_matches = [(i, m) for i, (d, s, m) in enumerate(scored_docs[:k]) if m]
                total_kg_candidates = sum(1 for d, s, m in scored_docs if m)

                if kg_matches:
                    positions = [pos for pos, _ in kg_matches]
                    logger.info(f"KG boost: {len(kg_matches)}/{total_kg_candidates} KG-matched chunks in top-{k} "
                               f"(positions: {positions[:5]})")
                elif total_kg_candidates > 0:
                    logger.debug(f"KG boost: 0/{total_kg_candidates} KG-matched chunks reached top-{k} - "
                                f"consider increasing boost factor if this is frequent")

                return [doc for doc, score, reason in scored_docs[:k]]
            else:
                return [doc for doc, score in results[:k]]

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    def _drift_search(
        self,
        query: str,
        community_level: int = 1,
        k_communities: int = 3,
        k_entities: int = 8,
        k_relationships: int = 12,
        k_chunks: int = 3
    ) -> GraphRAGContext:
        """
        DRIFT-style search combining global and local approaches.

        1. Start with community-level context (global)
        2. Drill down to entity-level specifics (local)
        3. Extract source IDs from KG results
        4. Retrieve chunks prioritizing KG-matched sources for grounding
        """
        self.stats['drift_searches'] += 1

        # Global: Get community summaries
        communities = self._global_search(
            query=query,
            community_level=community_level,
            k_communities=k_communities
        )

        # Local: Get entity details and relationships
        local_result = self._local_search(
            query=query,
            k_entities=k_entities,
            k_relationships=k_relationships
        )

        # Extract source IDs from KG results for targeted chunk retrieval
        kg_episode_ids, kg_book_ids = self._extract_source_ids(
            entities=local_result.entities,
            communities=communities
        )

        # Combine with direct source IDs from local search
        # Convert source_episodes (strings like "episode_123") to ints
        local_episode_ids = set()
        for source in local_result.source_episodes:
            if source.startswith('episode_'):
                try:
                    local_episode_ids.add(int(source.replace('episode_', '')))
                except ValueError:
                    pass

        all_episode_ids = kg_episode_ids | local_episode_ids
        all_book_ids = kg_book_ids | local_result.source_books

        # Retrieve chunks, prioritizing KG-matched sources
        chunks = self._retrieve_chunks(
            query=query,
            k=k_chunks,
            kg_episode_ids=all_episode_ids if all_episode_ids else None,
            kg_book_ids=all_book_ids if all_book_ids else None
        )

        # Extract episode numbers from chunks for citation
        citation_episodes = set()
        citation_books = set()
        for chunk in chunks:
            metadata = chunk.metadata or {}
            ep_num = metadata.get('episode_number')
            if ep_num:
                citation_episodes.add(f"episode_{ep_num}")
            if metadata.get('content_type') == 'book':
                citation_books.add(metadata.get('book_id', 'unknown'))

        # Combine all source references
        final_episodes = list(all_episode_ids | {int(e.replace('episode_', '')) for e in citation_episodes if e.startswith('episode_')})
        final_books = list(all_book_ids | citation_books)

        return GraphRAGContext(
            communities=communities,
            entities=local_result.entities,
            relationships=local_result.relationships,
            source_episodes=[f"episode_{ep}" if isinstance(ep, int) else ep for ep in final_episodes],
            source_books=final_books,
            chunks=chunks
        )

    def _format_graphrag_context(self, context: GraphRAGContext) -> str:
        """Format GraphRAG context for LLM consumption"""
        parts = []

        # Community summaries (global context)
        if context.communities:
            parts.append("=== THEMATIC CONTEXT (from Knowledge Graph Communities) ===")
            community_text = self.community_search.format_community_context(
                context.communities,
                max_summary_length=400
            )
            parts.append(community_text)

        # Entity details (local context)
        if context.entities:
            parts.append("\n=== ENTITY DETAILS (from Knowledge Graph) ===")
            entity_text = self.local_search.format_entity_context(
                context.entities,
                max_description_length=250
            )
            parts.append(entity_text)

        # Relationships
        if context.relationships:
            parts.append("\n=== RELATIONSHIPS ===")
            rel_text = self.local_search.format_relationship_context(
                context.relationships
            )
            parts.append(rel_text)

        # Chunk content (grounding)
        if context.chunks:
            parts.append("\n=== RELEVANT CONTENT EXCERPTS ===")
            for i, chunk in enumerate(context.chunks[:3], 1):
                content_preview = chunk.page_content[:300]
                if len(chunk.page_content) > 300:
                    content_preview += "..."

                metadata = chunk.metadata or {}

                # Build source label from available metadata
                # Episodes use episode_number and title, books use book_title
                content_type = metadata.get('content_type', 'episode')
                if content_type == 'book':
                    book_title = metadata.get('book_title', metadata.get('title', 'Unknown Book'))
                    chapter = metadata.get('chapter_title', '')
                    source = f"{book_title}" + (f", {chapter}" if chapter else "")
                else:
                    # Episode content
                    episode_num = metadata.get('episode_number', '')
                    episode_title = metadata.get('title', metadata.get('episode_title', ''))
                    if episode_num:
                        source = f"Episode {episode_num}"
                        if episode_title:
                            source += f": {episode_title[:50]}"
                    else:
                        source = metadata.get('episode_id', 'Unknown Episode')

                parts.append(f"\nExcerpt {i} (from {source}):\n{content_preview}")

        return "\n".join(parts)

    def chat(
        self,
        message: str,
        search_mode: str = "drift",  # "global", "local", "drift", "auto"
        community_level: int = 1,
        k_communities: int = 5,
        k_entities: int = 10,
        k_chunks: int = 3,
        personality: str = "warm_mother",
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat message using GraphRAG.

        Args:
            message: User's question
            search_mode: "global", "local", "drift", or "auto"
            community_level: 1 (fine-grained) or 2 (thematic)
            k_communities: Number of communities to retrieve
            k_entities: Number of entities to retrieve
            k_chunks: Number of chunks to retrieve
            personality: Gaia personality variant
            custom_prompt: Custom system prompt

        Returns:
            Dict with response, metadata, and GraphRAG context details
        """
        if not self.is_initialized:
            raise RuntimeError("GraphRAG Chain not initialized. Call initialize() first.")

        start_time = time.time()
        self.stats['total_queries'] += 1

        logger.info(f"Processing GraphRAG query: {message[:50]}...")

        try:
            # Auto-detect search mode if requested and get dynamic weights
            query_type = None
            if search_mode == "auto":
                query_type = self._classify_query(message)
                weights = self._get_search_weights(query_type)

                # Use query-type-specific parameters
                k_communities = weights['k_communities']
                k_entities = weights['k_entities']
                k_chunks = weights['k_chunks']

                # Map query types to search modes
                search_mode = 'drift' if query_type in ['drift', 'grounded'] else query_type
                logger.info(f"Auto-detected query type: {query_type} -> search mode: {search_mode}")

            # Perform search based on mode
            if search_mode == "global":
                communities = self._global_search(
                    message,
                    community_level=community_level,
                    k_communities=k_communities
                )
                context = GraphRAGContext(communities=communities)

            elif search_mode == "local":
                local_result = self._local_search(
                    message,
                    k_entities=k_entities
                )
                context = GraphRAGContext(
                    entities=local_result.entities,
                    relationships=local_result.relationships,
                    source_episodes=list(local_result.source_episodes),
                    source_books=list(local_result.source_books)
                )

            else:  # drift (default) or grounded
                context = self._drift_search(
                    message,
                    community_level=community_level,
                    k_communities=k_communities,
                    k_entities=k_entities,
                    k_chunks=k_chunks
                )

            # Format context for Gaia
            formatted_context = self._format_graphrag_context(context)

            # Create a pseudo-document list for Gaia's generate_response
            context_docs = []
            if formatted_context:
                context_docs.append(Document(
                    page_content=formatted_context,
                    metadata={'source': 'graphrag', 'search_mode': search_mode}
                ))

            # Add any retrieved chunks
            context_docs.extend(context.chunks)

            # Handle personality
            if personality and personality != self.gaia.personality_variant:
                if personality != 'custom' or custom_prompt is None:
                    self.gaia.switch_personality(personality)

            # Generate response using Gaia
            response_data = self.gaia.generate_response(
                user_input=message,
                retrieved_docs=context_docs,
                custom_prompt=custom_prompt
            )

            processing_time = time.time() - start_time
            self.stats['total_response_time'] += processing_time

            # Build response with GraphRAG metadata
            result = {
                'response': response_data.get('response', ''),
                'search_mode': search_mode,
                'query_type': query_type,  # Include detected query type for debugging
                'communities_used': [
                    {
                        'id': c.id,
                        'name': c.name,
                        'title': c.title,
                        'summary': c.summary[:200] + '...' if len(c.summary) > 200 else c.summary,
                        'level': c.level,
                        'entity_count': c.entity_count,
                        'relevance_score': c.relevance_score
                    }
                    for c in context.communities
                ],
                'entities_matched': [
                    {
                        'name': e.name,
                        'type': e.type,
                        'description': e.description[:150] + '...' if len(e.description) > 150 else e.description,
                        'sources': e.sources[:5],
                        'mention_count': e.mention_count,
                        'relevance_score': e.relevance_score
                    }
                    for e in context.entities
                ],
                'relationships': [
                    {
                        'source': r.source,
                        'predicate': r.predicate,
                        'target': r.target,
                        'weight': r.weight
                    }
                    for r in context.relationships
                ],
                'source_episodes': context.source_episodes,
                'source_books': context.source_books,
                'processing_time': processing_time,
                'success': True
            }

            logger.info(f"GraphRAG response generated in {processing_time:.2f}s using {search_mode} mode")
            return result

        except Exception as e:
            logger.error(f"Error in GraphRAG chat: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your question using the knowledge graph. Please try again.",
                'error': str(e),
                'search_mode': search_mode,
                'success': False
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics"""
        stats = self.stats.copy()

        if stats['total_queries'] > 0:
            stats['avg_response_time'] = stats['total_response_time'] / stats['total_queries']

        # Add component stats
        if self.community_search:
            stats['community_search'] = self.community_search.get_stats()
        if self.local_search:
            stats['local_search'] = self.local_search.get_stats()

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        return {
            'initialized': self.is_initialized,
            'community_search_ready': self.community_search is not None and self.community_search.is_initialized,
            'local_search_ready': self.local_search is not None and self.local_search.is_initialized,
            'vectorstore_ready': self.vectorstore is not None,
            'gaia_ready': self.gaia is not None,
            'stats': self.get_stats()
        }


# Singleton instance
_graphrag_chain_instance: Optional[GraphRAGChain] = None


def get_graphrag_chain() -> GraphRAGChain:
    """Get or create the GraphRAG chain singleton"""
    global _graphrag_chain_instance

    if _graphrag_chain_instance is None:
        _graphrag_chain_instance = GraphRAGChain()
        _graphrag_chain_instance.initialize()

    return _graphrag_chain_instance
