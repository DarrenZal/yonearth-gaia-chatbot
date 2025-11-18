"""
Advanced Hybrid RAG Retriever with BM25, as outlined in ImplimentationPlan.md
Combines BM25 keyword search + semantic search with cross-encoder reranking
"""
import logging
import pickle
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from ..utils.lc_compat import Document
from sentence_transformers import CrossEncoder

from .vectorstore import YonEarthVectorStore
from .episode_categorizer import EpisodeCategorizer
from .semantic_category_matcher import SemanticCategoryMatcher, CategoryMatch
from .graph_retriever import GraphRetriever, GraphRetrievalResult
from ..config import settings

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Some NLTK versions require 'punkt_tab' (3.8+)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class BM25HybridRetriever:
    """
    Advanced Hybrid RAG Retriever implementing the system outlined in ImplimentationPlan.md
    Features:
    - BM25 keyword search using rank-bm25
    - Semantic search with vector embeddings
    - Reciprocal Rank Fusion (RRF) for combining results
    - Cross-encoder reranking for final relevance scoring
    - Query analysis for adaptive search strategy
    """
    
    def __init__(
        self,
        vectorstore: YonEarthVectorStore,
        keyword_weight: float = 0.15,
        semantic_weight: float = 0.25,
        category_weight: float = 0.6,
        use_reranker: bool = True,
        reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        category_first_mode: bool = True
    ):
        self.vectorstore = vectorstore
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.category_weight = category_weight
        self.use_reranker = use_reranker
        self.category_first_mode = category_first_mode
        
        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.tokenized_docs: List[List[str]] = []
        
        # Reranking model
        self.reranker: Optional[CrossEncoder] = None
        if use_reranker:
            try:
                self.reranker = CrossEncoder(reranker_model)
                logger.info(f"Loaded reranker model: {reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}")
                self.reranker = None
        
        # Technical terms for query analysis
        self.technical_terms = {
            "biochar", "permaculture", "regenerative", "compost", 
            "mycorrhizal", "carbon", "agroforestry", "biodynamic",
            "sustainable", "organic", "ecosystem", "biodiversity",
            "climate", "soil", "farming", "agriculture", "garden"
        }

        # Heuristics for detecting book-oriented queries where book chunks should
        # not be aggressively deprioritized in favour of episodes.
        self._book_query_hints: Set[str] = {
            "y on earth",
            "viriditas",
            "soil stewardship",
            "soil stewardship handbook",
            "our biggest deal",
        }
        self._last_query_book_oriented: bool = False
        
        # Episode categorizer for enhanced search
        self.categorizer: Optional[EpisodeCategorizer] = None
        self.semantic_matcher: Optional[SemanticCategoryMatcher] = None
        self._load_categorizer()

        # Graph retriever for entity-aware search
        self.graph_retriever: Optional[GraphRetriever] = None
        self._last_graph_result: Optional[GraphRetrievalResult] = None
        self._load_graph_retriever()
        
        # Load or build BM25 index
        self._load_or_build_bm25_index()
    
    def _load_categorizer(self):
        """Load episode categorizer and semantic matcher for category-based search"""
        try:
            self.categorizer = EpisodeCategorizer()
            self.semantic_matcher = SemanticCategoryMatcher(categorizer=self.categorizer)
            logger.info(f"Loaded episode categorizer with {len(self.categorizer.episodes)} episodes and semantic matcher")
        except Exception as e:
            logger.warning(f"Failed to load episode categorizer: {e}")
            self.categorizer = None
            self.semantic_matcher = None

    def _load_graph_retriever(self):
        """Load GraphRetriever for entity-aware retrieval"""
        # Respect feature flag
        if not settings.enable_graph_retrieval:
            logger.info("Graph retrieval disabled via settings.enable_graph_retrieval=false")
            self.graph_retriever = None
            return

        try:
            self.graph_retriever = GraphRetriever(max_edges=settings.graph_max_edges)
            entity_count = len(self.graph_retriever.graph.entities_lexicon.get("alias_index", {}))
            logger.info(f"Loaded GraphRetriever with {entity_count} entity aliases")
        except Exception as e:
            logger.warning(f"Failed to load GraphRetriever: {e}. Graph-enhanced search will be disabled.")
            self.graph_retriever = None

    def _get_bm25_cache_path(self) -> Path:
        """Get path for BM25 index cache"""
        cache_dir = settings.data_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "bm25_index.pkl"
    
    def _load_or_build_bm25_index(self):
        """Load existing BM25 index or build new one"""
        cache_path = self._get_bm25_cache_path()
        
        try:
            if cache_path.exists():
                logger.info("Loading BM25 index from cache...")
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.bm25 = cache_data['bm25']
                    self.documents = cache_data['documents']
                    self.tokenized_docs = cache_data['tokenized_docs']
                logger.info(f"Loaded BM25 index with {len(self.documents)} documents")
            else:
                logger.info("Building new BM25 index...")
                self._build_bm25_index()
        except Exception as e:
            logger.error(f"Error loading BM25 cache: {e}")
            logger.info("Building new BM25 index...")
            self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from vectorstore documents"""
        try:
            # Get all documents from vectorstore
            # Note: This is a simplified approach. In production, you might want to
            # load documents more efficiently or from a dedicated document store
            logger.info("Fetching documents from vectorstore...")
            
            # Get ALL documents from vectorstore by using a very generic query
            # We use multiple generic queries to ensure we capture all content types
            all_documents = []
            
            # Query 1: Get episode-related content
            episode_query = "episode podcast guest interview"
            episode_results = self.vectorstore.similarity_search(episode_query, k=5000)
            all_documents.extend(episode_results)
            
            # Query 2: Get book-related content
            book_query = "chapter book viriditas healing nature"
            book_results = self.vectorstore.similarity_search(book_query, k=5000)
            all_documents.extend(book_results)
            
            # Query 3: Get any remaining content with generic terms
            generic_query = "the and of to in is that with for on"
            generic_results = self.vectorstore.similarity_search(generic_query, k=5000)
            all_documents.extend(generic_results)
            
            # Remove duplicates based on page content
            seen_content = set()
            search_results = []
            for doc in all_documents:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    search_results.append(doc)
            
            if not search_results:
                logger.warning("No documents found in vectorstore for BM25 indexing")
                return
            
            self.documents = search_results
            logger.info(f"Found {len(self.documents)} documents for BM25 indexing")
            
            # Tokenize documents for BM25
            self.tokenized_docs = []
            for doc in self.documents:
                tokenized = self._tokenize_document(doc.page_content)
                self.tokenized_docs.append(tokenized)
            
            # Create BM25 index
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info("BM25 index built successfully")
            
            # Cache the index
            self._cache_bm25_index()
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25 = None
    
    def _tokenize_document(self, text: str) -> List[str]:
        """Tokenize document text for BM25"""
        try:
            # Convert to lowercase and tokenize
            tokens = nltk.word_tokenize(text.lower())
            
            # Filter out punctuation and very short tokens
            tokens = [token for token in tokens if token.isalnum() and len(token) > 2]
            
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing document: {e}")
            return text.lower().split()
    
    def _cache_bm25_index(self):
        """Cache BM25 index to disk"""
        try:
            cache_path = self._get_bm25_cache_path()
            cache_data = {
                'bm25': self.bm25,
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"BM25 index cached to {cache_path}")
        except Exception as e:
            logger.error(f"Error caching BM25 index: {e}")
    
    def analyze_query(self, query: str, category_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Analyze query to determine best search strategy
        Based on ImplimentationPlan.md query analysis approach
        """
        query_lower = query.lower()
        
        # Use semantic category matching if available
        category_matches = {}
        if self.semantic_matcher:
            semantic_matches = self.semantic_matcher.get_semantic_category_matches(
                query, 
                threshold=category_threshold,  # Use configurable threshold
                max_matches=5
            )
            category_matches = {match.category: match.similarity for match in semantic_matches}
        elif self.categorizer:
            # Fallback to keyword-based matching
            category_matches = self.categorizer.analyze_query_categories(query)
        
        analysis = {
            'has_episode_ref': bool(re.search(r'episode\s*\d+', query_lower)),
            'has_technical_terms': any(term in query_lower for term in self.technical_terms),
            'query_length': len(query.split()),
            'is_question': query.strip().endswith('?'),
            'suggested_method': 'hybrid',  # default
            'category_matches': category_matches
        }
        
        # Simple heuristics for search method
        if analysis['has_episode_ref'] or analysis['has_technical_terms']:
            analysis['suggested_method'] = 'keyword_heavy'  # More weight on BM25
        elif analysis['query_length'] > 15:
            analysis['suggested_method'] = 'semantic_heavy'  # More weight on semantic
        elif analysis['category_matches']:
            analysis['suggested_method'] = 'category_heavy'  # More weight on categories
            
        return analysis
    
    def bm25_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search"""
        if not self.bm25:
            logger.warning("BM25 index not available")
            return []
        
        try:
            # Tokenize query
            tokenized_query = self._tokenize_document(query)
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k documents
            top_indices = np.argsort(scores)[-k:][::-1]
            
            results = []
            for i in top_indices:
                if scores[i] > 0:  # Only include documents with positive scores
                    results.append((self.documents[i], float(scores[i])))
            
            logger.info(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def semantic_search(
        self, 
        query: str, 
        k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform semantic search using vector similarity"""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Semantic search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def graph_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """
        Perform entity-aware search using knowledge graph

        This retrieves chunks based on entities mentioned in the query,
        expanding to multi-hop neighbors and scoring based on entity and cluster relevance.
        """
        mode = getattr(settings, "graph_mode", "full")
        if mode == "off":
            logger.debug("Graph mode=off; skipping graph search")
            self._last_graph_result = None
            return []

        if not self.graph_retriever:
            logger.debug("Graph retriever not available, skipping graph search")
            self._last_graph_result = None
            return []

        try:
            # Use GraphRetriever to find entity-related chunks
            graph_result = self.graph_retriever.retrieve(query, k=k)
            self._last_graph_result = graph_result

            if not graph_result.matched_entities and not graph_result.matched_clusters:
                logger.debug(f"No entities or clusters matched in query: {query}")
                return []

            logger.info(f"Graph search matched entities: {graph_result.matched_entities[:5]}")
            logger.info(f"Found {len(graph_result.chunks)} chunks and {len(graph_result.triples)} triples")
            if graph_result.matched_clusters:
                cluster_ids = [c.get("cluster_id") for c in graph_result.matched_clusters[:3]]
                logger.info(f"Graph search matched clusters: {cluster_ids}")

            # Convert Graph chunks to scored tuples
            # Graph-retrieved chunks get high base score since they're entity-relevant
            scored_docs = []
            base_score = 0.9  # High relevance for entity-matched chunks

            for i, doc in enumerate(graph_result.chunks):
                # Decay score slightly for lower-ranked chunks
                score = base_score * (1.0 - (i * 0.05))
                scored_docs.append((doc, max(score, 0.5)))  # Minimum score of 0.5

            return scored_docs

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            self._last_graph_result = None
            return []

    def _compute_graph_weight(self, base_weight: float) -> float:
        """Compute an adaptive graph weight based on entity/cluster signal.

        When adaptive weighting is disabled, the provided base_weight is used.
        """
        if not settings.graph_enable_adaptive_weight:
            return base_weight

        result = self._last_graph_result
        if not result:
            return 0.0

        entity_count = len(result.matched_entities or [])
        cluster_count = len(result.matched_clusters or [])
        if entity_count == 0 and cluster_count == 0:
            return 0.0

        # Treat cluster-only matches as a weak signal
        effective_entities = entity_count
        if entity_count == 0 and cluster_count > 0:
            effective_entities = 1

        cap = max(settings.graph_weight_entity_cap, 1)
        strength = min(effective_entities, cap) / cap

        # Interpolate between configured min/max weights, biased towards base_weight
        min_w = settings.graph_min_weight
        max_w = settings.graph_max_weight
        base_clamped = max(min_w, min(max_w, base_weight))
        raw = min_w + (max_w - min_w) * strength
        weight = 0.5 * base_clamped + 0.5 * raw

        return max(min_w, min(max_w, weight))

    def category_search(self, query: str, k: int = 20, category_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """Perform category-based search using semantic category matching"""
        if not self.semantic_matcher and not self.categorizer:
            logger.warning("No category matcher available")
            return []
        
        try:
            # Get matching categories using semantic matcher
            if self.semantic_matcher:
                # Use semantic category matching
                category_matches = self.semantic_matcher.get_semantic_category_matches(
                    query,
                    threshold=category_threshold,  # Use configurable threshold
                    max_matches=10
                )
                
                if not category_matches:
                    logger.info("No semantic category matches found")
                    return []
                
                # Get all episodes for matched categories
                episode_ids = self.semantic_matcher.get_episodes_for_semantic_matches(category_matches)
                
                # Log what categories matched
                logger.info(f"Semantic category matches: {self.semantic_matcher.explain_matches(category_matches)}")
                logger.info(f"Found {len(episode_ids)} episodes from matched categories")
                
                # Use diverse episode search to ensure all episodes are represented
                return self.diverse_episode_search(query, episode_ids, k)
                
            else:
                # Fallback to old keyword-based approach
                top_episodes = self.categorizer.get_top_episodes_for_query(query, k=k)
                
                if not top_episodes:
                    logger.info("No episodes found matching query categories")
                    return []
                
                # Convert episode IDs to documents
                results = []
                for episode_id, score in top_episodes:
                    episode_docs = self._find_documents_for_episode(episode_id)
                    for doc in episode_docs:
                        results.append((doc, score))
                
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]
            
        except Exception as e:
            logger.error(f"Error in category search: {e}")
            return []
    
    def _find_documents_for_episode(self, episode_id: int) -> List[Document]:
        """Find all documents that belong to a specific episode"""
        episode_docs = []
        
        for doc in self.documents:
            metadata = getattr(doc, 'metadata', {})
            doc_episode_id = metadata.get('episode_id')
            
            # Try to extract episode ID from metadata
            if doc_episode_id:
                try:
                    if int(doc_episode_id) == episode_id:
                        episode_docs.append(doc)
                except (ValueError, TypeError):
                    continue
            
            # Also check episode_number field
            doc_episode_number = metadata.get('episode_number')
            if doc_episode_number:
                try:
                    if int(doc_episode_number) == episode_id:
                        episode_docs.append(doc)
                except (ValueError, TypeError):
                    continue
            
            # Fallback: check if episode mentioned in content
            if f"episode {episode_id}" in doc.page_content.lower():
                episode_docs.append(doc)
        
        # If no documents found, create a placeholder document for category scoring
        if not episode_docs:
            logger.debug(f"No documents found for episode {episode_id}, creating placeholder")
            # Get episode info from categorizer
            if self.categorizer:
                episode_info = self.categorizer.get_episode_info(episode_id)
                if episode_info:
                    placeholder_content = f"Episode {episode_id}: {episode_info.guest_name} - {episode_info.guest_title}"
                    placeholder_doc = Document(
                        page_content=placeholder_content,
                        metadata={
                            'episode_id': str(episode_id),
                            'episode_number': str(episode_id),
                            'guest': episode_info.guest_name,
                            'title': episode_info.guest_title,
                            'content_type': 'episode',
                            'placeholder': True
                        }
                    )
                    episode_docs.append(placeholder_doc)
        
        return episode_docs
    
    def diverse_episode_search(self, query: str, episode_ids: Set[int], k: int) -> List[Tuple[Document, float]]:
        """
        Ensure all matching episodes are represented in results
        This solves the problem where Episode 120's 31 chunks fill all k=20 slots
        """
        if not episode_ids:
            return []
        
        logger.info(f"Performing diverse episode search across {len(episode_ids)} episodes")
        
        # Step 1: Get best chunks from each episode
        episode_chunks = {}
        max_chunks_per_episode = max(3, k // len(episode_ids))  # At least 3 chunks per episode
        
        for ep_id in episode_ids:
            # Get all chunks for this episode
            ep_docs = self._find_documents_for_episode(ep_id)
            
            if not ep_docs:
                continue
            
            # Score each chunk against the query
            scored_chunks = []
            for doc in ep_docs:
                # Calculate BM25 score
                bm25_score = 0.0
                if self.bm25:
                    tokenized_query = self._tokenize_document(query)
                    doc_idx = self.documents.index(doc) if doc in self.documents else -1
                    if doc_idx >= 0:
                        bm25_score = float(self.bm25.get_scores(tokenized_query)[doc_idx])
                
                # Calculate semantic score (would need to call vectorstore)
                semantic_score = 0.5  # Placeholder - in production, would calculate actual semantic similarity
                
                # Combined score
                combined_score = (self.keyword_weight * bm25_score + 
                                self.semantic_weight * semantic_score + 
                                self.category_weight * 1.0)  # Full category weight since it matched
                
                scored_chunks.append((doc, combined_score))
            
            # Keep top chunks for this episode
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            episode_chunks[ep_id] = scored_chunks[:max_chunks_per_episode]
            
            logger.debug(f"Episode {ep_id}: selected {len(episode_chunks[ep_id])} chunks")
        
        # Step 2: Combine all chunks and sort by score
        all_chunks = []
        for chunks in episode_chunks.values():
            all_chunks.extend(chunks)
        
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Ensure diversity - if we have too many chunks from one episode, limit them
        final_results = []
        episode_counts = {}
        max_per_episode_final = max(3, k // max(len(episode_ids) // 2, 1))
        
        for doc, score in all_chunks:
            ep_id = doc.metadata.get('episode_number', -1)
            
            # Check if we've already added too many from this episode
            if episode_counts.get(ep_id, 0) >= max_per_episode_final:
                continue
            
            final_results.append((doc, score))
            episode_counts[ep_id] = episode_counts.get(ep_id, 0) + 1
            
            if len(final_results) >= k:
                break
        
        # Log diversity stats
        unique_episodes = len(set(doc.metadata.get('episode_number', -1) for doc, _ in final_results))
        logger.info(f"Diverse search returned {len(final_results)} chunks from {unique_episodes} unique episodes")
        
        return final_results
    
    def category_first_fusion(
        self,
        keyword_results: List[Tuple[Document, float]],
        semantic_results: List[Tuple[Document, float]],
        category_results: List[Tuple[Document, float]],
        graph_results: Optional[List[Tuple[Document, float]]] = None,
        k: int = 60,
        graph_weight: float = 0.2
    ) -> List[Document]:
        """
        Category-first fusion: Prioritize category matches, then use semantic/BM25/graph for ranking
        This ensures ALL category matches appear in results
        """
        logger.info(f"Using category-first fusion with {len(category_results)} category matches and {len(graph_results or [])} graph matches")

        # Step 1: Get all category-matched documents (these get priority)
        category_docs = {self._get_document_id(doc): doc for doc, score in category_results}

        # Step 2: Create combined scoring for category matches
        category_scores = {}

        # Score category matches with heavy category weighting
        for rank, (doc, score) in enumerate(category_results):
            doc_id = self._get_document_id(doc)
            category_scores[doc_id] = self.category_weight * (1.0 / (k + rank + 1))

        # Add semantic scores for category matches
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = self._get_document_id(doc)
            if doc_id in category_docs:  # Only for category matches
                category_scores[doc_id] = category_scores.get(doc_id, 0) + self.semantic_weight * (1.0 / (k + rank + 1))

        # Add BM25 scores for category matches
        for rank, (doc, score) in enumerate(keyword_results):
            doc_id = self._get_document_id(doc)
            if doc_id in category_docs:  # Only for category matches
                category_scores[doc_id] = category_scores.get(doc_id, 0) + self.keyword_weight * (1.0 / (k + rank + 1))

        # Add graph scores for category matches (modest boost)
        if graph_results and graph_weight > 0:
            for rank, (doc, score) in enumerate(graph_results):
                doc_id = self._get_document_id(doc)
                if doc_id in category_docs:  # Only for category matches
                    category_scores[doc_id] = category_scores.get(doc_id, 0) + graph_weight * (1.0 / (k + rank + 1))
        
        # Step 3: Sort category matches by combined score
        sorted_category_ids = sorted(category_scores.keys(), key=lambda x: category_scores[x], reverse=True)
        
        # Step 4: Add non-category matches if we need more results
        non_category_scores = {}

        # Score non-category documents with traditional weighting
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = self._get_document_id(doc)
            if doc_id not in category_docs:
                non_category_scores[doc_id] = self.semantic_weight * (1.0 / (k + rank + 1))

        for rank, (doc, score) in enumerate(keyword_results):
            doc_id = self._get_document_id(doc)
            if doc_id not in category_docs:
                non_category_scores[doc_id] = non_category_scores.get(doc_id, 0) + self.keyword_weight * (1.0 / (k + rank + 1))

        # Add graph scores for non-category documents too
        if graph_results and graph_weight > 0:
            for rank, (doc, score) in enumerate(graph_results):
                doc_id = self._get_document_id(doc)
                if doc_id not in category_docs:
                    non_category_scores[doc_id] = non_category_scores.get(doc_id, 0) + graph_weight * (1.0 / (k + rank + 1))

        # Sort non-category matches
        sorted_non_category_ids = sorted(non_category_scores.keys(), key=lambda x: non_category_scores[x], reverse=True)

        # Step 5: Combine results with category matches first
        final_order = sorted_category_ids + sorted_non_category_ids

        # Step 6: Convert to documents, avoiding duplicates
        seen_content = set()
        results = []

        all_results = [keyword_results, semantic_results, category_results]
        if graph_results:
            all_results.append(graph_results)

        for doc_id in final_order:
            # Find the document
            doc = self._find_document_by_id(doc_id, *all_results)
            if doc and doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                results.append(doc)

        logger.info(f"Category-first fusion: {len(sorted_category_ids)} category matches + {len(sorted_non_category_ids)} other matches")
        # Prefer episode documents over book-only chunks in the final ordering
        try:
            results = self._prefer_episodes(results)
        except Exception:
            pass
        return results
    
    def reciprocal_rank_fusion(
        self,
        keyword_results: List[Tuple[Document, float]],
        semantic_results: List[Tuple[Document, float]],
        graph_results: Optional[List[Tuple[Document, float]]] = None,
        k: int = 60,
        graph_weight: float = 1.2
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion (RRF) algorithm
        As outlined in ImplimentationPlan.md, now with graph-aware retrieval
        """
        scores = {}

        # Process keyword results
        for rank, (doc, score) in enumerate(keyword_results):
            doc_id = self._get_document_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Process semantic results
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = self._get_document_id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Process graph results with a configurable boost (default 1.2x)
        if graph_results and graph_weight > 0:
            for rank, (doc, score) in enumerate(graph_results):
                doc_id = self._get_document_id(doc)
                scores[doc_id] = scores.get(doc_id, 0) + graph_weight / (k + rank + 1)

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return documents (avoiding duplicates)
        seen_content = set()
        results = []

        all_results = [keyword_results, semantic_results]
        if graph_results:
            all_results.append(graph_results)

        for doc_id in sorted_ids:
            # Find the document
            doc = self._find_document_by_id(doc_id, *all_results)
            if doc and doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                results.append(doc)

        # Prefer episode documents over book-only chunks in the final ordering
        try:
            results = self._prefer_episodes(results)
        except Exception:
            pass
        return results
    
    def _get_document_id(self, doc: Document) -> str:
        """Get unique ID for document"""
        metadata = getattr(doc, 'metadata', {})
        doc_id = metadata.get('chunk_id') or metadata.get('episode_id')
        if not doc_id:
            # Fallback to content hash
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
        return str(doc_id)

    def _ensure_backbone_coverage(
        self,
        backbone_docs: List[Document],
        candidate_docs: List[Document],
        k: int
    ) -> List[Document]:
        """
        Ensure that all top-k backbone (BM25+semantic+category) documents remain
        present in the final top-k after adding graph-based signals.

        This provides a "do no harm" guarantee: graph-enhanced fusion may add
        additional documents, but will not drop backbone hits from the top-k.
        """
        if not backbone_docs or not candidate_docs:
            return candidate_docs[:k]

        # Only enforce coverage for the top-k backbone docs
        backbone_top = backbone_docs[:k]
        backbone_ids = {self._get_document_id(d) for d in backbone_top}

        final = list(candidate_docs[:k])
        final_ids = {self._get_document_id(d) for d in final}

        # Find backbone docs missing from the current final set
        missing = [d for d in backbone_top if self._get_document_id(d) not in final_ids]
        if not missing:
            return final

        for doc in missing:
            doc_id = self._get_document_id(doc)
            if doc_id in final_ids:
                continue

            if len(final) < k:
                final.append(doc)
                final_ids.add(doc_id)
                continue

            # Replace the lowest-ranked non-backbone doc, if any
            replaced = False
            for idx in range(len(final) - 1, -1, -1):
                cand_id = self._get_document_id(final[idx])
                if cand_id not in backbone_ids:
                    final[idx] = doc
                    final_ids.add(doc_id)
                    replaced = True
                    break
            if not replaced:
                # All docs are backbone docs already; nothing to replace safely
                break

        return final[:k]

    def _ensure_book_episode_mix(self, final_docs: List[Document], candidate_docs: List[Document], k: int) -> List[Document]:
        """Ensure at least one book and one episode are present in final top-k when available.

        If missing a type, pull the highest-ranked candidate of the missing type
        from candidate_docs (that isn't already in final_docs), replace the last
        item in final_docs if length == k, otherwise append.
        """
        if not final_docs:
            return final_docs

        has_episode = any(self._is_episode_doc(d) for d in final_docs)
        has_book = any(self._is_book_doc(d) for d in final_docs)
        if has_episode and has_book:
            return final_docs

        # Build a set of seen IDs to avoid duplicates
        seen_ids = set()
        for d in final_docs:
            seen_ids.add(self._get_document_id(d))

        need_episode = not has_episode
        need_book = not has_book

        def first_candidate(is_ok) -> Optional[Document]:
            for d in candidate_docs:
                if self._get_document_id(d) in seen_ids:
                    continue
                if is_ok(d):
                    return d
            return None

        ep_cand = first_candidate(self._is_episode_doc) if need_episode else None
        bk_cand = first_candidate(self._is_book_doc) if need_book else None

        updated = list(final_docs)

        def insert_candidate(cand: Optional[Document]):
            nonlocal updated
            if not cand:
                return
            if len(updated) < k:
                updated.append(cand)
            else:
                # Replace the last item (lowest-ranked) to guarantee presence
                updated[-1] = cand

        # Prefer adding missing episode first, then book
        if need_episode:
            insert_candidate(ep_cand)
        if need_book:
            insert_candidate(bk_cand)

        # If we added items beyond k, trim
        return updated[:k]
    
    def _find_document_by_id(self, doc_id: str, *result_sets: List[Tuple[Document, float]]) -> Optional[Document]:
        """Find a document by ID across any number of (doc, score) result sets."""
        for result_set in result_sets:
            if not result_set:
                continue
            for doc, _ in result_set:
                if self._get_document_id(doc) == doc_id:
                    return doc
        return None

    def _is_book_oriented_query(self, query: str) -> bool:
        """Detect whether the query is primarily about books/chapters instead of episodes."""
        q = (query or "").lower()
        if "book" in q or "chapter" in q:
            return True
        for hint in self._book_query_hints:
            if hint in q:
                return True
        return False
    
    def _is_episode_doc(self, doc: Document) -> bool:
        """Heuristic to detect episode-derived documents vs. book chunks."""
        md = getattr(doc, 'metadata', {}) or {}
        epn = md.get('episode_number')
        return epn not in (None, '', 'unknown')

    def _is_book_doc(self, doc: Document) -> bool:
        """Heuristic to detect book-derived documents."""
        if self._is_episode_doc(doc):
            return False
        md = getattr(doc, 'metadata', {}) or {}
        if md.get('book_title') or md.get('book_slug') or md.get('chapter_title'):
            return True
        cid = md.get('chunk_id') or ''
        return isinstance(cid, str) and not cid.startswith('ep')

    def _prefer_episodes(self, docs: List[Document]) -> List[Document]:
        """Stable-partition documents to prefer episode chunks over book-only chunks.

        For book-oriented queries, keep the original ordering so that book chunks
        are not deprioritized relative to episode chunks.
        """
        if self._last_query_book_oriented:
            return docs

        eps: List[Document] = []
        books: List[Document] = []
        for d in docs:
            if self._is_episode_doc(d):
                eps.append(d)
            else:
                books.append(d)
        return eps + books
    
    def rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank using cross-encoder for better relevance
        As outlined in ImplimentationPlan.md
        """
        if not documents or not self.reranker:
            return documents
            
        try:
            # Create query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Sort by reranker scores
            ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents")
            return [doc for doc, _ in ranked_docs]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return documents
    
    def hybrid_search(self, query: str, k: int = 10, category_threshold: float = 0.7) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search with reranking
        Implements the complete pipeline from ImplimentationPlan.md
        """
        logger.info(f"Performing hybrid search for: {query[:50]}...")

        # Track whether this query appears book-oriented so that downstream
        # fusion can avoid over-prioritizing episodes.
        self._last_query_book_oriented = self._is_book_oriented_query(query)
        
        # 1. Analyze query
        query_analysis = self.analyze_query(query, category_threshold)
        logger.info(f"Query analysis: {query_analysis}")
        
        # 2. Adjust weights based on query analysis - Category is PRIMARY
        if query_analysis['suggested_method'] == 'keyword_heavy':
            self.keyword_weight = 0.25
            self.semantic_weight = 0.15
            self.category_weight = 0.6
        elif query_analysis['suggested_method'] == 'semantic_heavy':
            self.keyword_weight = 0.1
            self.semantic_weight = 0.3
            self.category_weight = 0.6
        elif query_analysis['suggested_method'] == 'category_heavy' or query_analysis['category_matches']:
            self.keyword_weight = 0.05
            self.semantic_weight = 0.15
            self.category_weight = 0.8
        else:
            # Default: Category is still PRIMARY
            self.keyword_weight = 0.15
            self.semantic_weight = 0.25
            self.category_weight = 0.6
        
        # 3. BM25 keyword search
        keyword_results = self.bm25_search(query, k=20)

        # 4. Semantic search
        semantic_results = self.semantic_search(query, k=20)

        # 5. Category search
        category_results = self.category_search(query, k=20, category_threshold=category_threshold)

        # 5b. Build a backbone fusion (BM25 + semantic + categories) WITHOUT graph
        if self.category_first_mode and category_results:
            backbone_docs = self.category_first_fusion(
                keyword_results,
                semantic_results,
                category_results,
                graph_results=None,
                k=60,
                graph_weight=0.0
            )
        else:
            backbone_docs = self.reciprocal_rank_fusion(
                keyword_results,
                semantic_results,
                graph_results=None,
                k=60,
                graph_weight=0.0
            )

        # 6. Graph-based entity/cluster search
        graph_results = self.graph_search(query, k=20)

        # Determine graph weight for fusion based on entity/cluster signal and mode
        mode = getattr(settings, "graph_mode", "full")
        graph_weight = 0.0
        if graph_results and mode == "full":
            # Use a lower base weight when running in category-first mode to
            # keep categories primary; otherwise use the default graph base.
            base_weight = (
                settings.graph_min_weight
                if self.category_first_mode and category_results
                else settings.graph_default_weight
            )
            graph_weight = self._compute_graph_weight(base_weight)
            logger.info(f"Adaptive graph weight for fusion (mode=full): {graph_weight:.3f}")
        elif graph_results and mode == "evidence_only":
            logger.info("Graph mode=evidence_only; graph results will not affect ranking (weight=0).")

        # 7. Combine results using Category-First or Reciprocal Rank Fusion (with graph)
        # If graph results are available and mode=full, blend them in with a boost;
        # in evidence_only/off modes, graph_weight will be 0 and graph_results will
        # not influence ranking.
        if self.category_first_mode and category_results:
            fused_results = self.category_first_fusion(
                keyword_results,
                semantic_results,
                category_results,
                graph_results,
                k=60,
                graph_weight=graph_weight,
            )
        else:
            fused_results = self.reciprocal_rank_fusion(
                keyword_results,
                semantic_results,
                graph_results,
                k=60,
                graph_weight=graph_weight,
            )
        
        # 7. Rerank top candidates with cross-encoder
        if len(fused_results) > k and self.use_reranker:
            reranked = self.rerank_results(query, fused_results[:k*2])
            final_results = reranked[:k]
        else:
            final_results = fused_results[:k]
        
        # 7b. Ensure at least one episode and one book are present when available.
        # In graph_mode='off' or 'evidence_only', we avoid pulling in additional
        # graph chunks to influence recall and rely purely on the backbone +
        # existing fusion; in 'full' mode we keep the existing diversity logic.
        try:
            mode = getattr(settings, "graph_mode", "full")
            if mode == "full":
                # Build candidate pool preserving order: fused_results first, then others
                candidate_pool = list(fused_results)
                # Flatten other result sets and append
                for src in (graph_results or []):
                    candidate_pool.append(src[0])
                for src in semantic_results:
                    candidate_pool.append(src[0])
                for src in keyword_results:
                    candidate_pool.append(src[0])
                # Remove duplicates while preserving order
                seen = set()
                uniq_pool = []
                for d in candidate_pool:
                    did = self._get_document_id(d)
                    if did in seen:
                        continue
                    seen.add(did)
                    uniq_pool.append(d)

                # If book or episode missing from final_results, expand pool using GraphRetriever at higher k
                need_episode = not any(self._is_episode_doc(d) for d in final_results)
                need_book = not any(self._is_book_doc(d) for d in final_results)
                if (need_episode or need_book) and self.graph_retriever:
                    try:
                        # Expand graph candidates aggressively for diversity
                        k_extra = max(300, settings.graph_retrieval_k * 10)
                        extra = self.graph_retriever.retrieve(query, k=k_extra)
                        # Append in rank order, skipping dups
                        for d in extra.chunks:
                            did = self._get_document_id(d)
                            if did in seen:
                                continue
                            seen.add(did)
                            uniq_pool.append(d)
                    except Exception:
                        pass
                    # Absolute fallback: pick any book/episode doc from the graph index if still missing
                    try:
                        # Refresh flags based on pool + final
                        has_ep = any(self._is_episode_doc(d) for d in final_results)
                        has_bk = any(self._is_book_doc(d) for d in final_results)
                        if self.graph_retriever and hasattr(self.graph_retriever, 'graph'):
                            # helper to add first matching chunk from hierarchy
                            def add_first(predicate):
                                nonlocal uniq_pool, seen
                                for cid in self.graph_retriever.graph.hierarchy.keys():
                                    is_ep = isinstance(cid, str) and cid.startswith('ep')
                                    if predicate(is_ep):
                                        doc = self.graph_retriever._chunk_to_document(cid)  # type: ignore
                                        if doc:
                                            did = self._get_document_id(doc)
                                            if did not in seen:
                                                seen.add(did)
                                                uniq_pool.append(doc)
                                                return True
                                return False
                            if not has_bk:
                                add_first(lambda is_ep: not is_ep)
                            if not has_ep:
                                add_first(lambda is_ep: is_ep)
                    except Exception:
                        pass
                final_results = self._ensure_book_episode_mix(final_results, uniq_pool, k)
        except Exception:
            pass

        # 7c. Guarantee backbone coverage: all backbone top-k docs remain in final top-k
        try:
            final_results = self._ensure_backbone_coverage(backbone_docs, final_results, k)
        except Exception:
            logger.debug("Backbone coverage enforcement failed; returning original hybrid results")
        
        logger.info(f"Hybrid search completed: {len(final_results)} final results")
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'bm25_available': self.bm25 is not None,
            'total_documents': len(self.documents),
            'reranker_available': self.reranker is not None,
            'categorizer_available': self.categorizer is not None,
            'current_keyword_weight': self.keyword_weight,
            'current_semantic_weight': self.semantic_weight,
            'current_category_weight': self.category_weight,
            'technical_terms_count': len(self.technical_terms),
            'total_episodes': len(self.categorizer.episodes) if self.categorizer else 0,
            'category_first_mode': self.category_first_mode
        }


def main():
    """Test BM25 hybrid retriever functionality"""
    import logging
    from .vectorstore import create_vectorstore
    
    logging.basicConfig(level=logging.INFO)
    
    # Create components
    logger.info("Setting up BM25 hybrid retriever...")
    vectorstore = create_vectorstore()
    retriever = BM25HybridRetriever(vectorstore)
    
    # Test queries from ImplimentationPlan.md
    test_queries = [
        "what is biochar",
        "regenerative agriculture techniques",
        "episode 147 permaculture",
        "how to start composting at home"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: '{query}'")
        print(f"{'='*60}")
        
        results = retriever.hybrid_search(query, k=5)
        
        print(f"Found {len(results)} results:\n")
        
        for i, doc in enumerate(results, 1):
            metadata = getattr(doc, 'metadata', {})
            episode_id = metadata.get('episode_id', 'Unknown')
            title = metadata.get('title', 'Unknown Title')
            
            print(f"{i}. Episode {episode_id}: {title}")
            print(f"   Content Preview: {doc.page_content[:150]}...")
            print()
    
    # Show stats
    stats = retriever.get_stats()
    print(f"\nBM25 Hybrid Retriever Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
