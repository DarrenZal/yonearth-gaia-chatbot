"""
Advanced Hybrid RAG Retriever with BM25, as outlined in ImplimentationPlan.md
Combines BM25 keyword search + semantic search with cross-encoder reranking
"""
import logging
import pickle
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from langchain.schema import Document
from sentence_transformers import CrossEncoder

from .vectorstore import YonEarthVectorStore
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
        keyword_weight: float = 0.5,
        semantic_weight: float = 0.5,
        use_reranker: bool = True,
        reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ):
        self.vectorstore = vectorstore
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.use_reranker = use_reranker
        
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
        
        # Load or build BM25 index
        self._load_or_build_bm25_index()
    
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
            
            # Perform a broad search to get documents
            broad_query = "regenerative sustainable permaculture agriculture farming"
            search_results = self.vectorstore.similarity_search(broad_query, k=1000)
            
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
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine best search strategy
        Based on ImplimentationPlan.md query analysis approach
        """
        query_lower = query.lower()
        
        analysis = {
            'has_episode_ref': bool(re.search(r'episode\s*\d+', query_lower)),
            'has_technical_terms': any(term in query_lower for term in self.technical_terms),
            'query_length': len(query.split()),
            'is_question': query.strip().endswith('?'),
            'suggested_method': 'hybrid'  # default
        }
        
        # Simple heuristics for search method
        if analysis['has_episode_ref'] or analysis['has_technical_terms']:
            analysis['suggested_method'] = 'keyword_heavy'  # More weight on BM25
        elif analysis['query_length'] > 15:
            analysis['suggested_method'] = 'semantic_heavy'  # More weight on semantic
            
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
        semantic_results: List[Tuple[Document, float]]
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
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search with reranking
        Implements the complete pipeline from ImplimentationPlan.md
        """
        logger.info(f"Performing hybrid search for: {query[:50]}...")
        
        # 1. Analyze query
        query_analysis = self.analyze_query(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # 2. Adjust weights based on query analysis
        if query_analysis['suggested_method'] == 'keyword_heavy':
            self.keyword_weight = 0.7
            self.semantic_weight = 0.3
        elif query_analysis['suggested_method'] == 'semantic_heavy':
            self.keyword_weight = 0.3
            self.semantic_weight = 0.7
        else:
            self.keyword_weight = 0.5
            self.semantic_weight = 0.5
        
        # 3. BM25 keyword search
        keyword_results = self.bm25_search(query, k=20)
        
        # 4. Semantic search
        semantic_results = self.semantic_search(query, k=20)
        
        # 5. Combine results using Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(keyword_results, semantic_results)
        
        # 6. Rerank top candidates with cross-encoder
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
            'current_keyword_weight': self.keyword_weight,
            'current_semantic_weight': self.semantic_weight,
            'technical_terms_count': len(self.technical_terms)
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