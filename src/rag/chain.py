"""
RAG chain implementation connecting retrieval, character, and generation
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from langchain_core.documents import Document

from ..config import settings
from ..character.gaia import GaiaCharacter
from .vectorstore import YonEarthVectorStore, create_vectorstore
from .hybrid_retriever import HybridRetriever
from ..ingestion.process_episodes import process_episodes_for_ingestion
from .graph_retriever import GraphRetriever, GraphRetrievalResult

logger = logging.getLogger(__name__)


class YonEarthRAGChain:
    """Complete RAG chain for YonEarth chatbot"""
    
    def __init__(self, initialize_data: bool = False):
        self.vectorstore = None
        self.hybrid_retriever = None
        self.graph_retriever = None
        self.gaia = None
        self.is_initialized = False

        if initialize_data:
            self.initialize()
    
    def initialize(self, recreate_index: bool = False):
        """Initialize the RAG chain with data and components"""
        logger.info("Initializing YonEarth RAG chain...")
        
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
            
            # Step 3: Initialize hybrid retriever
            logger.info("Setting up hybrid retriever...")
            self.hybrid_retriever = HybridRetriever(self.vectorstore)

            # Step 4: Initialize graph retriever if enabled
            if settings.enable_graph_retrieval:
                logger.info("Setting up graph retriever...")
                try:
                    self.graph_retriever = GraphRetriever(max_edges=settings.graph_max_edges)
                    logger.info("Graph retriever initialized successfully")
                except Exception as e:
                    logger.warning(f"Could not initialize graph retriever: {e}. Continuing with hybrid retrieval only.")
                    self.graph_retriever = None

            # Step 5: Initialize Gaia character
            logger.info("Initializing Gaia character...")
            self.gaia = GaiaCharacter()
            
            # Step 4: Verify setup
            stats = self.vectorstore.get_stats()
            logger.info(f"RAG chain initialized successfully. Vector count: {stats.get('total_vector_count', 0)}")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            raise
    
    def ensure_initialized(self):
        """Ensure the RAG chain is initialized"""
        if not self.is_initialized:
            self.initialize()
    
    def query(
        self, 
        user_input: str,
        k: int = 5,
        session_id: Optional[str] = None,
        personality_variant: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user query through the complete RAG pipeline"""
        self.ensure_initialized()
        
        try:
            # Step 1: Retrieve relevant documents using hybrid search
            logger.info(f"Processing query: {user_input[:50]}...")
            retrieved_docs = self.hybrid_retriever.retrieve(
                query=user_input,
                k=k
            )

            # Extract documents and scores
            docs = [doc for doc, score in retrieved_docs]
            scores = [score for doc, score in retrieved_docs]

            # Step 1b: Optionally retrieve using graph if enabled
            graph_evidence = []
            graph_retrieval_ms = 0
            if settings.enable_graph_retrieval and self.graph_retriever:
                try:
                    graph_start = time.time()
                    graph_result = self.graph_retriever.retrieve(
                        query=user_input,
                        k=settings.graph_retrieval_k
                    )

                    # Merge graph chunks with hybrid results using RRF
                    docs, scores = self._merge_retrieval_results(
                        hybrid_docs=list(zip(docs, scores)),
                        graph_docs=graph_result.chunks,
                        k=k
                    )

                    # Store graph evidence (triples) for context with capping
                    graph_evidence = self._cap_evidence(graph_result.triples)
                    graph_retrieval_ms = int((time.time() - graph_start) * 1000)

                    logger.info(f"Graph retrieval: {len(graph_result.chunks)} chunks, {len(graph_evidence)} triples (capped from {len(graph_result.triples)}) in {graph_retrieval_ms}ms")

                except Exception as e:
                    logger.warning(f"Graph retrieval failed: {e}. Using hybrid retrieval only.")
            
            # Step 2: Create Gaia instance with the specified model if different
            gaia_instance = self.gaia
            if model_name and model_name != self.gaia.model_name:
                # Create a new Gaia instance with different model
                gaia_instance = GaiaCharacter(
                    personality_variant=personality_variant or self.gaia.personality_variant,
                    model_name=model_name
                )
            elif personality_variant and personality_variant != self.gaia.personality_variant:
                if personality_variant != 'custom' or custom_prompt is None:
                    self.gaia.switch_personality(personality_variant)
            
            # Step 3: Generate response with Gaia
            response = gaia_instance.generate_response(
                user_input=user_input,
                retrieved_docs=docs,
                session_id=session_id,
                custom_prompt=custom_prompt
            )
            
            # Step 4: Add retrieval metadata
            response.update({
                "retrieval_scores": scores,
                "query": user_input,
                "retrieval_count": len(docs),
                "graph_evidence": graph_evidence if graph_evidence else [],
                "graph_evidence_count": len(graph_evidence),
                "graph_retrieval_ms": graph_retrieval_ms
            })
            
            logger.info(f"Successfully processed query with {len(docs)} retrieved documents")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, dear one, but I'm experiencing difficulties accessing the Earth's wisdom right now. Please try again in a moment.",
                "error": str(e),
                "query": user_input,
                "retrieval_count": 0,
                "citations": []
            }
    
    def _merge_retrieval_results(
        self,
        hybrid_docs: List[Tuple[Document, float]],
        graph_docs: List[Document],
        k: int
    ) -> Tuple[List[Document], List[float]]:
        """Merge hybrid and graph retrieval results using Reciprocal Rank Fusion"""

        # Create rank maps
        hybrid_ranks = {}
        for rank, (doc, score) in enumerate(hybrid_docs):
            doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
            hybrid_ranks[doc_id] = rank + 1

        graph_ranks = {}
        for rank, doc in enumerate(graph_docs):
            doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
            graph_ranks[doc_id] = rank + 1

        # Combine all documents
        all_docs = {}
        for doc, score in hybrid_docs:
            doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
            all_docs[doc_id] = (doc, score)

        for doc in graph_docs:
            doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
            if doc_id not in all_docs:
                all_docs[doc_id] = (doc, 0.0)

        # Calculate RRF scores
        rrf_scores = {}
        k_rrf = 60  # Standard RRF constant

        for doc_id in all_docs:
            hybrid_rank = hybrid_ranks.get(doc_id, 1000)
            graph_rank = graph_ranks.get(doc_id, 1000)

            # Weight hybrid slightly more than graph
            rrf_score = (0.6 / (k_rrf + hybrid_rank)) + (0.4 / (k_rrf + graph_rank))
            rrf_scores[doc_id] = rrf_score

        # Sort by RRF score and return top k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        result_docs = []
        result_scores = []
        for doc_id, rrf_score in sorted_docs:
            doc, original_score = all_docs[doc_id]
            result_docs.append(doc)
            result_scores.append(rrf_score)

        return result_docs, result_scores

    def _cap_evidence(self, triples: List[Dict[str, Any]],
                     min_confidence: float = None,
                     max_per_predicate: int = None,
                     max_total: int = None) -> List[Dict[str, Any]]:
        """Cap evidence to high-confidence assertions and limit per predicate"""

        # Use settings if not provided
        min_confidence = min_confidence or settings.graph_evidence_min_confidence
        max_per_predicate = max_per_predicate or settings.graph_evidence_per_predicate
        max_total = max_total or settings.graph_evidence_max_total

        # Filter by confidence
        high_confidence = [t for t in triples if t.get('p_true', 0) >= min_confidence]

        # Group by predicate
        by_predicate = defaultdict(list)
        for triple in high_confidence:
            by_predicate[triple.get('predicate', 'unknown')].append(triple)

        # Cap per predicate and collect results
        capped = []
        for predicate, predicate_triples in by_predicate.items():
            # Sort by confidence and take top N per predicate
            sorted_triples = sorted(predicate_triples,
                                   key=lambda x: x.get('p_true', 0),
                                   reverse=True)
            capped.extend(sorted_triples[:max_per_predicate])

        # Final cap on total
        capped = sorted(capped, key=lambda x: x.get('p_true', 0), reverse=True)[:max_total]

        # Log for analysis
        logger.debug(f"Evidence capping: {len(triples)} -> {len(high_confidence)} (confidence) -> {len(capped)} (capped)")

        return capped

    def get_episode_recommendations(
        self, 
        user_input: str, 
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get episode recommendations based on user query"""
        self.ensure_initialized()
        
        try:
            # Use hybrid retrieval to find relevant episodes
            retrieved_docs = self.hybrid_retriever.retrieve(
                query=user_input,
                k=k * 2  # Get more to deduplicate by episode
            )
            
            # Group by episode and get unique episodes
            episodes_seen = set()
            recommendations = []
            
            for doc, score in retrieved_docs:
                metadata = getattr(doc, 'metadata', {})
                episode_num = metadata.get('episode_number')
                
                if episode_num and episode_num not in episodes_seen:
                    episodes_seen.add(episode_num)
                    
                    recommendation = {
                        "episode_number": episode_num,
                        "title": metadata.get('title', 'Unknown Episode'),
                        "guest_name": metadata.get('guest_name', 'Guest'),
                        "url": metadata.get('url', ''),
                        "relevance_score": float(score),
                        "reason": f"Highly relevant to your interest in: {user_input[:50]}..."
                    }
                    recommendations.append(recommendation)
                    
                    if len(recommendations) >= k:
                        break
            
            logger.info(f"Generated {len(recommendations)} episode recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting episode recommendations: {e}")
            return []
    
    def search_episodes(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for episodes with optional filters"""
        self.ensure_initialized()
        
        try:
            retrieved_docs = self.hybrid_retriever.retrieve(
                query=query,
                k=k
            )
            
            results = []
            for doc, score in retrieved_docs:
                metadata = getattr(doc, 'metadata', {})
                content = getattr(doc, 'page_content', '')
                
                result = {
                    "episode_number": metadata.get('episode_number'),
                    "title": metadata.get('title'),
                    "guest_name": metadata.get('guest_name'),
                    "url": metadata.get('url'),
                    "relevance_score": float(score),
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "metadata": metadata
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching episodes: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG chain statistics"""
        stats = {
            "initialized": self.is_initialized,
            "gaia_personality": None,
            "vectorstore_stats": {},
            "keyword_index_stats": {}
        }
        
        if self.gaia:
            stats["gaia_personality"] = self.gaia.personality_variant
            
        if self.vectorstore:
            stats["vectorstore_stats"] = self.vectorstore.get_stats()
            
        if self.hybrid_retriever:
            stats["keyword_index_stats"] = self.hybrid_retriever.get_keyword_stats()
            
        return stats
    
    def reset_conversation(self, session_id: Optional[str] = None):
        """Reset conversation memory"""
        if self.gaia:
            self.gaia.clear_memory()
            logger.info(f"Reset conversation memory for session: {session_id}")


# Global instance for the application
rag_chain = YonEarthRAGChain()


def get_rag_chain() -> YonEarthRAGChain:
    """Get the global RAG chain instance"""
    return rag_chain


def main():
    """Test RAG chain functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG chain (will process episodes if needed)
    chain = YonEarthRAGChain(initialize_data=True)
    
    # Test query
    response = chain.query("Tell me about regenerative agriculture practices")
    
    print(f"\nQuery: Tell me about regenerative agriculture practices")
    print(f"Response: {response['response'][:200]}...")
    print(f"Citations: {len(response.get('citations', []))}")
    print(f"Retrieved docs: {response.get('retrieval_count', 0)}")
    
    # Test episode recommendations
    recommendations = chain.get_episode_recommendations("soil health", k=3)
    print(f"\nEpisode Recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"- Episode {rec['episode_number']}: {rec['title']}")


if __name__ == "__main__":
    main()