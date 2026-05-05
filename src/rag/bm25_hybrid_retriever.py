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
from langchain.schema import Document
from sentence_transformers import CrossEncoder

from .vectorstore import YonEarthVectorStore
from .episode_categorizer import EpisodeCategorizer
from .semantic_category_matcher import SemanticCategoryMatcher, CategoryMatch
from ..config import settings

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
        
        # Episode categorizer for enhanced search
        self.categorizer: Optional[EpisodeCategorizer] = None
        self.semantic_matcher: Optional[SemanticCategoryMatcher] = None
        self._load_categorizer()
        
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

        # Build secondary indexes for O(1) lookups during theme-saturation
        # (was O(N²) doc.index() per chunk; cost the system 30-90s on long
        # multi-theme queries). Indexes by content_type + key.
        self._episode_chunk_indices: Dict[int, List[int]] = {}
        self._book_chunk_indices: Dict[str, List[int]] = {}
        for i, doc in enumerate(self.documents or []):
            md = getattr(doc, 'metadata', {}) or {}
            ct = md.get('content_type', 'episode')
            if ct == 'book':
                bt = md.get('book_title')
                if bt:
                    self._book_chunk_indices.setdefault(bt, []).append(i)
                continue
            # Episode-like (default): episode_number can be int or str
            ep_num = md.get('episode_number')
            if ep_num is None:
                ep_num = md.get('episode_id')
            try:
                ep_id = int(str(ep_num))
            except (TypeError, ValueError):
                continue
            self._episode_chunk_indices.setdefault(ep_id, []).append(i)
        logger.info(
            f"Built chunk indexes: {len(self._episode_chunk_indices)} episodes, "
            f"{len(self._book_chunk_indices)} books "
            f"({sum(len(v) for v in self._episode_chunk_indices.values())} ep chunks, "
            f"{sum(len(v) for v in self._book_chunk_indices.values())} book chunks)"
        )
    
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
    
    def analyze_query(self, query: str, category_threshold: float = 0.55) -> Dict[str, Any]:
        """
        Analyze query to determine best search strategy
        Based on ImplimentationPlan.md query analysis approach.

        Threshold default lowered from 0.7 → 0.55 (Aaron 2026-05-02): short
        theme queries like "finance" need to land on IMPACT INVESTING even when
        the semantic similarity isn't strong. Below the semantic threshold we
        also union in the keyword-categorizer fallback, so "finance" reliably
        triggers theme-saturation.
        """
        query_lower = query.lower()

        # Try semantic category matching first
        category_matches: Dict[str, float] = {}
        if self.semantic_matcher:
            semantic_matches = self.semantic_matcher.get_semantic_category_matches(
                query,
                threshold=category_threshold,
                max_matches=5
            )
            category_matches = {match.category: match.similarity for match in semantic_matches}

        # Always also union in keyword-categorizer matches. Aaron's "finance"
        # case fails semantic similarity but matches the IMPACT INVESTING
        # synonym set in episode_categorizer. Union, don't replace.
        if self.categorizer:
            keyword_matches = self.categorizer.analyze_query_categories(query)
            for cat, score in keyword_matches.items():
                if score > 0 and cat not in category_matches:
                    # Normalize keyword score to similar scale (clamp to [threshold, 0.9])
                    category_matches[cat] = max(category_threshold, min(0.9, score / 2.0))
        
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
        This solves the problem where Episode 120's 31 chunks fill all k=20 slots.

        Optimized 2026-05-04: was O(N²) due to `self.documents.index(doc)` per
        chunk plus per-chunk `self.bm25.get_scores()` recomputation; for queries
        matching 5+ categories this took 3+ minutes. Now uses the precomputed
        episode→chunk-indices map and computes BM25 scores once.
        """
        if not episode_ids:
            return []

        logger.info(f"Performing diverse episode search across {len(episode_ids)} episodes")

        # Compute BM25 scores ONCE for the whole corpus
        bm25_scores = None
        if self.bm25:
            tokenized_query = self._tokenize_document(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)

        ep_index_map = getattr(self, '_episode_chunk_indices', {}) or {}

        # Step 1: Get best chunks from each episode
        episode_chunks = {}
        max_chunks_per_episode = max(3, k // len(episode_ids))  # At least 3 chunks per episode

        for ep_id in episode_ids:
            indices = ep_index_map.get(ep_id, [])
            if not indices:
                # Fallback: rare case where episode isn't in the prebuilt map
                ep_docs = self._find_documents_for_episode(ep_id)
                if not ep_docs:
                    continue
                indices = list(range(len(ep_docs)))
                ep_docs_iter = ep_docs
                index_to_doc = {i: ep_docs[i] for i in indices}
                idx_to_score = {i: 0.0 for i in indices}
            else:
                index_to_doc = {i: self.documents[i] for i in indices}
                idx_to_score = {
                    i: float(bm25_scores[i]) if bm25_scores is not None else 0.0
                    for i in indices
                }

            scored_chunks = [
                (
                    index_to_doc[i],
                    (self.keyword_weight * idx_to_score[i]
                     + self.semantic_weight * 0.5
                     + self.category_weight * 1.0),
                )
                for i in indices
            ]
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            episode_chunks[ep_id] = scored_chunks[:max_chunks_per_episode]
        
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
        k: int = 60
    ) -> List[Document]:
        """
        Category-first fusion: Prioritize category matches, then use semantic/BM25 for ranking
        This ensures ALL category matches appear in results
        """
        logger.info(f"Using category-first fusion with {len(category_results)} category matches")
        
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
        
        # Sort non-category matches
        sorted_non_category_ids = sorted(non_category_scores.keys(), key=lambda x: non_category_scores[x], reverse=True)
        
        # Step 5: Combine results with category matches first
        final_order = sorted_category_ids + sorted_non_category_ids
        
        # Step 6: Convert to documents, avoiding duplicates
        seen_content = set()
        results = []
        
        for doc_id in final_order:
            # Find the document
            doc = self._find_document_by_id(doc_id, keyword_results, semantic_results, category_results)
            if doc and doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                results.append(doc)
        
        logger.info(f"Category-first fusion: {len(sorted_category_ids)} category matches + {len(sorted_non_category_ids)} other matches")
        return results
    
    def reciprocal_rank_fusion(
        self, 
        keyword_results: List[Tuple[Document, float]], 
        semantic_results: List[Tuple[Document, float]], 
        k: int = 60
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion (RRF) algorithm
        As outlined in ImplimentationPlan.md
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
            
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return documents (avoiding duplicates)
        seen_content = set()
        results = []
        
        for doc_id in sorted_ids:
            # Find the document
            doc = self._find_document_by_id(doc_id, keyword_results, semantic_results)
            if doc and doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                results.append(doc)
                
        return results
    
    def _get_document_id(self, doc: Document) -> str:
        """Get unique ID for document"""
        metadata = getattr(doc, 'metadata', {})
        doc_id = metadata.get('chunk_id') or metadata.get('episode_id')
        if not doc_id:
            # Fallback to content hash
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
        return str(doc_id)
    
    def _find_document_by_id(
        self, 
        doc_id: str, 
        keyword_results: List[Tuple[Document, float]], 
        semantic_results: List[Tuple[Document, float]],
        category_results: List[Tuple[Document, float]] = None
    ) -> Optional[Document]:
        """Helper to find document by ID from results"""
        for doc, _ in keyword_results:
            if self._get_document_id(doc) == doc_id:
                return doc
        for doc, _ in semantic_results:
            if self._get_document_id(doc) == doc_id:
                return doc
        return None
    
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
    
    def theme_saturation_search(
        self,
        query: str,
        category_threshold: float = 0.55,
        max_chunks: int = 30,
    ) -> List[Document]:
        """
        Aaron's #2 issue (2026-05-02): when a query hits a YOE theme, EVERY
        episode (and book chapter) tagged with that theme should be reachable.

        Composition rule (refined 2026-05-04 after the 25-prompt suite showed
        large categories crowding out small ones):

          1. Resolve matched categories with similarity scores.
          2. Iterate categories in DESCENDING similarity order. Each category
             gets a quota of `max(2, max_chunks // num_categories)` chunks,
             but never more chunks than the category has tagged episodes.
             Within a category, pick the highest BM25-scoring chunk per
             episode (one chunk per episode).
          3. Pool books across all matched categories, score by BM25, take
             the top `min(5, max_chunks // 3)`.
          4. Stop early when max_chunks is filled.

        This means GREEN BUILDING (8 eps) and SUSTAIN-ABILITY (115 eps) both
        get fair representation when both match — the smaller category isn't
        flooded out.

        Empty list if no category matches at the given threshold.
        Pure BM25 — no extra OpenAI calls.
        """
        if not self.categorizer:
            return []

        analysis = self.analyze_query(query, category_threshold=category_threshold)
        cat_matches: Dict[str, float] = analysis.get('category_matches') or {}
        if not cat_matches:
            return []

        # DESCENDING by similarity so the strongest-matching category gets first dibs
        ordered_cats = sorted(cat_matches.items(), key=lambda kv: kv[1], reverse=True)
        logger.info(
            f"Theme-saturation: query {query!r} matched categories "
            f"{[(c, round(s, 2)) for c, s in ordered_cats]}"
        )

        per_cat_quota = max(2, max_chunks // max(1, len(ordered_cats)))
        logger.info(f"Theme-saturation: per-category quota={per_cat_quota}, max_chunks={max_chunks}")

        # Pre-tokenize query once for BM25 scoring
        tokenized_query = self._tokenize_document(query) if self.bm25 else None
        bm25_scores = self.bm25.get_scores(tokenized_query) if (self.bm25 and tokenized_query) else None

        ep_index_map: Dict[int, List[int]] = getattr(self, '_episode_chunk_indices', {}) or {}
        book_index_map: Dict[str, List[int]] = getattr(self, '_book_chunk_indices', {}) or {}

        def best_chunk_for_episode(ep_id: int) -> Optional[Tuple[Document, float]]:
            indices = ep_index_map.get(ep_id, [])
            if not indices:
                return None
            if bm25_scores is None:
                return (self.documents[indices[0]], 0.0)
            best_idx = max(indices, key=lambda i: float(bm25_scores[i]))
            return (self.documents[best_idx], float(bm25_scores[best_idx]))

        def best_chunk_for_book(book_key: str) -> Optional[Tuple[Document, float]]:
            indices = book_index_map.get(book_key, [])
            if not indices:
                return None
            if bm25_scores is None:
                return (self.documents[indices[0]], 0.0)
            best_idx = max(indices, key=lambda i: float(bm25_scores[i]))
            return (self.documents[best_idx], float(bm25_scores[best_idx]))

        # Per-category episode picks with quotas (prevents large-cat flood).
        # We collect per-category lists first, then round-robin across them so
        # the strongest-matching category's picks lead but every matched
        # category contributes at least one chunk before any category gets a
        # second turn. Then we append within-category fallback after first round.
        per_cat_lists: List[List[Tuple[Document, float, int]]] = []
        seen_episodes: Set[int] = set()
        for cat, sim in ordered_cats:
            cat_eps = [
                ep for ep in self.categorizer.get_episodes_by_category(cat)
                if ep not in seen_episodes
            ]
            scored: List[Tuple[Document, float, int]] = []
            for ep_id in cat_eps:
                picked = best_chunk_for_episode(ep_id)
                if picked is not None:
                    scored.append((picked[0], picked[1], ep_id))
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:per_cat_quota]
            for doc, score, ep_id in scored:
                seen_episodes.add(ep_id)
            per_cat_lists.append(scored)

        # Reserve a few slots for books so they aren't truncated by the
        # max_chunks cap when many episodes are eligible.
        book_reserved = max(1, min(5, max_chunks // 3))
        episode_budget = max(1, max_chunks - book_reserved)

        # Round-robin: take the i-th element from each category in turn so a
        # rare-but-strongly-matched category isn't buried by a large category's
        # higher BM25 scores. Within each round, categories are visited in
        # similarity-descending order (preserved from `ordered_cats`).
        episode_picks: List[Tuple[Document, float]] = []
        rr_round = 0
        while len(episode_picks) < episode_budget:
            progressed = False
            for cat_list in per_cat_lists:
                if rr_round < len(cat_list):
                    doc, score, _ep_id = cat_list[rr_round]
                    episode_picks.append((doc, score))
                    progressed = True
                    if len(episode_picks) >= episode_budget:
                        break
            if not progressed:
                break
            rr_round += 1

        logger.info(
            f"Theme-saturation: collected {len(episode_picks)} episode chunks "
            f"across {len(ordered_cats)} categories ({len(seen_episodes)} unique episodes; "
            f"per_cat_quota={per_cat_quota})"
        )

        # Books pooled across all matched categories
        book_keys: Set[str] = set()
        get_books = getattr(self.categorizer, 'get_books_by_category', None)
        if callable(get_books):
            for cat, _ in ordered_cats:
                book_keys.update(get_books(cat))
        book_picks: List[Tuple[Document, float]] = []
        for book_key in book_keys:
            picked = best_chunk_for_book(book_key)
            if picked is not None:
                book_picks.append(picked)
        book_picks.sort(key=lambda x: x[1], reverse=True)
        book_picks = book_picks[:book_reserved]
        logger.info(
            f"Theme-saturation: collected {len(book_picks)} book chunks "
            f"(reserved {book_reserved}; pool {len(book_keys)} books)"
        )

        # Final order: episode round-robin first (preserves category fairness),
        # then book picks. NOT re-sorted by BM25 — that would un-do the
        # quota-based composition.
        merged: List[Tuple[Document, float]] = []
        merged.extend(episode_picks)
        merged.extend(book_picks)

        return [doc for doc, _ in merged[:max_chunks]]

    def hybrid_search(self, query: str, k: int = 10, category_threshold: float = 0.55) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search with reranking
        Implements the complete pipeline from ImplimentationPlan.md
        """
        logger.info(f"Performing hybrid search for: {query[:50]}...")
        
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
        
        # 6. Combine results using Category-First or Reciprocal Rank Fusion
        if self.category_first_mode and category_results:
            fused_results = self.category_first_fusion(keyword_results, semantic_results, category_results)
        else:
            fused_results = self.reciprocal_rank_fusion(keyword_results, semantic_results)
        
        # 7. Rerank top candidates with cross-encoder
        if len(fused_results) > k and self.use_reranker:
            reranked = self.rerank_results(query, fused_results[:k*2])
            final_results = reranked[:k]
        else:
            final_results = fused_results[:k]
        
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