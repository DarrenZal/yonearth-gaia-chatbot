"""
Advanced BM25 Hybrid RAG Chain - New pipeline with BM25 integration
Keeps existing RAG pipeline intact while adding state-of-the-art BM25 hybrid search
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document

from ..config import settings
from ..character.gaia import GaiaCharacter
from .vectorstore import YonEarthVectorStore, create_vectorstore
from .bm25_hybrid_retriever import BM25HybridRetriever
from ..ingestion.process_episodes import process_episodes_for_ingestion

logger = logging.getLogger(__name__)


class BM25RAGChain:
    """
    Advanced RAG chain with BM25 hybrid search
    
    Features:
    - BM25 keyword search + semantic vector search
    - Reciprocal Rank Fusion for result combination
    - Cross-encoder reranking for improved relevance
    - Query-adaptive search strategy
    - Comparison metrics with original RAG chain
    """
    
    def __init__(self, initialize_data: bool = False):
        self.vectorstore = None
        self.bm25_retriever = None
        self.gaia = None
        self.is_initialized = False
        self.search_stats = {
            'total_queries': 0,
            'bm25_queries': 0,
            'semantic_queries': 0,
            'hybrid_queries': 0,
            'reranked_queries': 0
        }
        
        if initialize_data:
            self.initialize()
    
    def initialize(self, recreate_index: bool = False):
        """Initialize the BM25 RAG chain with data and components"""
        logger.info("Initializing BM25 Hybrid RAG chain...")
        
        try:
            # Step 1: Process episodes if vectorstore is empty or recreating
            documents = None
            if recreate_index:
                logger.info("Processing episodes for vector database...")
                documents = process_episodes_for_ingestion()
            
            # Step 2: Create/connect to vectorstore
            logger.info("Setting up vector database...")
            self.vectorstore = create_vectorstore(
                documents=documents,
                recreate_index=recreate_index
            )
            
            # Step 3: Initialize BM25 hybrid retriever
            logger.info("Setting up BM25 hybrid retriever...")
            self.bm25_retriever = BM25HybridRetriever(self.vectorstore)
            
            # Step 4: Initialize Gaia character
            logger.info("Setting up Gaia character...")
            self.gaia = GaiaCharacter()
            
            self.is_initialized = True
            logger.info("BM25 RAG chain initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing BM25 RAG chain: {e}")
            raise
    
    def chat(
        self,
        message: str,
        search_method: str = "auto",  # "auto", "bm25", "semantic", "hybrid"
        k: int = 5,
        include_sources: bool = True,
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat message using BM25 hybrid RAG
        
        Args:
            message: User's question/message
            search_method: Search strategy ("auto", "bm25", "semantic", "hybrid")
            k: Number of documents to retrieve
            include_sources: Whether to include source citations
            **kwargs: Additional parameters for Gaia
        
        Returns:
            Dict containing response, sources, metadata, and performance metrics
        """
        if not self.is_initialized:
            raise RuntimeError("BM25 RAG chain not initialized. Call initialize() first.")
        
        logger.info(f"Processing chat message: {message[:50]}...")
        
        # Update stats
        self.search_stats['total_queries'] += 1
        
        try:
            # Step 1: Retrieve relevant documents
            if search_method == "auto":
                # Use query analysis to determine best method
                query_analysis = self.bm25_retriever.analyze_query(message)
                search_method = query_analysis['suggested_method']
            
            documents = self._retrieve_documents(message, search_method, k)
            
            # Step 2: Switch personality if requested (but not for custom when custom_prompt is provided)
            personality_variant = kwargs.get('personality_variant')
            if personality_variant and personality_variant != self.gaia.personality_variant:
                if personality_variant != 'custom' or custom_prompt is None:
                    self.gaia.switch_personality(personality_variant)
            
            # Step 3: Generate response using Gaia
            response_data = self.gaia.generate_response(
                user_input=message,
                retrieved_docs=documents,
                session_id=kwargs.get('session_id'),
                custom_prompt=custom_prompt
            )
            
            # Step 4: Add BM25-specific metadata
            response_data.update({
                'search_method_used': search_method,
                'documents_retrieved': len(documents),
                'bm25_stats': self.bm25_retriever.get_stats(),
                'performance_stats': self.search_stats.copy()
            })
            
            # Step 5: Add sources if requested
            if include_sources:
                response_data['sources'] = self._format_sources(documents)
                response_data['episode_references'] = self._extract_episode_references(documents)
            
            logger.info(f"BM25 RAG response generated successfully using {search_method} search")
            return response_data
            
        except Exception as e:
            logger.error(f"Error in BM25 chat processing: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your question. Please try again.",
                'error': str(e),
                'search_method_used': search_method,
                'success': False
            }
    
    def _retrieve_documents(
        self, 
        query: str, 
        search_method: str, 
        k: int
    ) -> List[Document]:
        """Retrieve documents using specified search method"""
        
        if search_method == "bm25":
            # Pure BM25 search
            results = self.bm25_retriever.bm25_search(query, k=k)
            documents = [doc for doc, score in results]
            self.search_stats['bm25_queries'] += 1
            
        elif search_method == "semantic":
            # Pure semantic search
            results = self.bm25_retriever.semantic_search(query, k=k)
            documents = [doc for doc, score in results]
            self.search_stats['semantic_queries'] += 1
            
        elif search_method in ["hybrid", "keyword_heavy", "semantic_heavy"]:
            # Full hybrid search with RRF and reranking
            documents = self.bm25_retriever.hybrid_search(query, k=k)
            self.search_stats['hybrid_queries'] += 1
            if self.bm25_retriever.use_reranker:
                self.search_stats['reranked_queries'] += 1
        
        else:
            # Fallback to hybrid
            documents = self.bm25_retriever.hybrid_search(query, k=k)
            self.search_stats['hybrid_queries'] += 1
        
        logger.info(f"Retrieved {len(documents)} documents using {search_method} method")
        return documents
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source citations from retrieved documents"""
        sources = []
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            source = {
                'episode_id': metadata.get('episode_id', 'Unknown'),
                'episode_number': metadata.get('episode_number', 'Unknown'),
                'title': metadata.get('title', 'Unknown Title'),
                'guest_name': metadata.get('guest_name', 'Unknown Guest'),
                'url': metadata.get('url', ''),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            
            # Add BM25-specific scores if available
            if 'keyword_component' in metadata:
                source['keyword_score'] = metadata['keyword_component']
            if 'semantic_component' in metadata:
                source['semantic_score'] = metadata['semantic_component']
            if 'final_score' in metadata:
                source['final_score'] = metadata['final_score']
            
            sources.append(source)
        
        return sources
    
    def _extract_episode_references(self, documents: List[Document]) -> List[str]:
        """Extract unique episode references from documents"""
        episodes = set()
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            episode_id = metadata.get('episode_id')
            episode_number = metadata.get('episode_number')
            
            if episode_id:
                episodes.add(str(episode_id))
            elif episode_number:
                episodes.add(str(episode_number))
        
        return sorted(list(episodes))
    
    def search_episodes(
        self, 
        query: str, 
        k: int = 10,
        search_method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search for episodes using BM25 hybrid search
        
        Args:
            query: Search query
            k: Number of results to return
            search_method: Search method to use
        
        Returns:
            List of episode information with relevance scores
        """
        if not self.is_initialized:
            raise RuntimeError("BM25 RAG chain not initialized")
        
        documents = self._retrieve_documents(query, search_method, k)
        
        # Group results by episode
        episode_groups = {}
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            episode_id = metadata.get('episode_id', 'unknown')
            
            if episode_id not in episode_groups:
                episode_groups[episode_id] = {
                    'episode_id': episode_id,
                    'episode_number': metadata.get('episode_number', 'Unknown'),
                    'title': metadata.get('title', 'Unknown Title'),
                    'url': metadata.get('url', ''),
                    'chunks': [],
                    'max_score': 0
                }
            
            chunk_info = {
                'content': doc.page_content,
                'score': metadata.get('final_score', 0)
            }
            
            episode_groups[episode_id]['chunks'].append(chunk_info)
            episode_groups[episode_id]['max_score'] = max(
                episode_groups[episode_id]['max_score'], 
                chunk_info['score']
            )
        
        # Sort by relevance score
        episodes = sorted(
            episode_groups.values(), 
            key=lambda x: x['max_score'], 
            reverse=True
        )
        
        return episodes
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance statistics for comparison with original RAG"""
        stats = self.search_stats.copy()
        
        if stats['total_queries'] > 0:
            stats['bm25_percentage'] = (stats['bm25_queries'] / stats['total_queries']) * 100
            stats['semantic_percentage'] = (stats['semantic_queries'] / stats['total_queries']) * 100
            stats['hybrid_percentage'] = (stats['hybrid_queries'] / stats['total_queries']) * 100
            stats['reranking_percentage'] = (stats['reranked_queries'] / stats['total_queries']) * 100
        
        stats.update(self.bm25_retriever.get_stats())
        
        return stats
    
    def compare_search_methods(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Compare different search methods for the same query
        Useful for A/B testing and performance analysis
        """
        if not self.is_initialized:
            raise RuntimeError("BM25 RAG chain not initialized")
        
        comparison = {
            'query': query,
            'methods': {}
        }
        
        methods = ["bm25", "semantic", "hybrid"]
        
        for method in methods:
            try:
                documents = self._retrieve_documents(query, method, k)
                
                comparison['methods'][method] = {
                    'documents_count': len(documents),
                    'episodes_referenced': self._extract_episode_references(documents),
                    'top_episodes': [
                        {
                            'episode_id': getattr(doc, 'metadata', {}).get('episode_id', 'Unknown'),
                            'title': getattr(doc, 'metadata', {}).get('title', 'Unknown'),
                            'preview': doc.page_content[:100] + "..."
                        }
                        for doc in documents[:3]
                    ]
                }
            except Exception as e:
                comparison['methods'][method] = {
                    'error': str(e)
                }
        
        return comparison
    
    def health_check(self) -> Dict[str, Any]:
        """Check health status of BM25 RAG chain"""
        return {
            'initialized': self.is_initialized,
            'vectorstore_available': self.vectorstore is not None,
            'bm25_retriever_available': self.bm25_retriever is not None,
            'gaia_available': self.gaia is not None,
            'bm25_index_ready': self.bm25_retriever.bm25 is not None if self.bm25_retriever else False,
            'reranker_available': self.bm25_retriever.reranker is not None if self.bm25_retriever else False,
            'performance_stats': self.search_stats,
            'component_stats': self.bm25_retriever.get_stats() if self.bm25_retriever else {}
        }


def main():
    """Test BM25 RAG chain functionality"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create and initialize BM25 RAG chain
    logger.info("Creating BM25 RAG chain...")
    bm25_chain = BM25RAGChain(initialize_data=True)
    
    # Test chat functionality
    test_queries = [
        "what is biochar and how is it used?",
        "tell me about regenerative agriculture practices",
        "episode 147 permaculture techniques",
        "how can I start composting at home?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: '{query}'")
        print(f"{'='*60}")
        
        # Test different search methods
        for method in ["bm25", "semantic", "hybrid"]:
            print(f"\n--- {method.upper()} METHOD ---")
            
            try:
                response = bm25_chain.chat(
                    message=query,
                    search_method=method,
                    k=3,
                    include_sources=True
                )
                
                print(f"Response: {response.get('response', 'No response')[:200]}...")
                print(f"Episodes referenced: {response.get('episode_references', [])}")
                print(f"Search method used: {response.get('search_method_used')}")
                
            except Exception as e:
                print(f"Error with {method}: {e}")
    
    # Show performance statistics
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    stats = bm25_chain.get_performance_comparison()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Health check
    print(f"\n{'='*60}")
    print("HEALTH CHECK")
    print(f"{'='*60}")
    
    health = bm25_chain.health_check()
    for key, value in health.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()