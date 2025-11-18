"""
Vector store implementation using Pinecone and LangChain
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from ..utils.lc_compat import Document
import tiktoken

from ..config import settings
from .pinecone_setup import PineconeManager

logger = logging.getLogger(__name__)


class YonEarthVectorStore:
    """Vector store for YonEarth podcast episodes"""
    
    def __init__(self, pinecone_manager: Optional[PineconeManager] = None):
        # Initialize Pinecone manager
        self.pinecone_manager = pinecone_manager or PineconeManager()
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore(
            index=self.pinecone_manager.get_index(),
            embedding=self.embeddings,
            text_key="text"
        )
        
        # Token counter for cost estimation (fallback to cl100k_base if model not mapped)
        try:
            self.encoding = tiktoken.encoding_for_model(settings.openai_embedding_model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def estimate_embedding_cost(self, documents: List[Document]) -> Dict[str, Any]:
        """Estimate the cost of embedding documents"""
        total_tokens = 0
        for doc in documents:
            tokens = len(self.encoding.encode(doc.page_content))
            total_tokens += tokens
        
        # OpenAI text-embedding-3-small pricing: $0.00002 per 1K tokens
        cost_per_1k_tokens = 0.00002
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            "total_documents": len(documents),
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "avg_tokens_per_doc": total_tokens // len(documents) if documents else 0
        }
    
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[str]:
        """Add documents to vector store in batches"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Estimate cost
        cost_estimate = self.estimate_embedding_cost(documents)
        logger.info(f"Embedding cost estimate: ${cost_estimate['estimated_cost_usd']:.4f}")
        
        # Add documents in batches
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            if show_progress:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            try:
                ids = self.vectorstore.add_documents(batch)
                all_ids.extend(ids)
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                
        logger.info(f"Successfully added {len(all_ids)} documents to vector store")
        return all_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents"""
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with relevance scores"""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Found {len(results)} results with scores for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def delete_all(self) -> None:
        """Delete all vectors from the store"""
        self.pinecone_manager.delete_all_vectors()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.pinecone_manager.get_index_stats()


def create_vectorstore(
    documents: Optional[List[Document]] = None,
    recreate_index: bool = False
) -> YonEarthVectorStore:
    """Create and optionally populate a vector store"""
    from .pinecone_setup import setup_pinecone
    
    # Setup Pinecone
    pinecone_manager = setup_pinecone(recreate_index=recreate_index)
    
    # Create vector store
    vectorstore = YonEarthVectorStore(pinecone_manager)
    
    # Add documents if provided
    if documents:
        vectorstore.add_documents(documents)
    
    return vectorstore


def main():
    """Test vector store functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test documents
    test_docs = [
        Document(
            page_content="This is a test about regenerative agriculture and soil health.",
            metadata={"episode_number": "1", "title": "Test Episode 1"}
        ),
        Document(
            page_content="Climate change solutions through community action.",
            metadata={"episode_number": "2", "title": "Test Episode 2"}
        )
    ]
    
    # Create vector store
    vectorstore = create_vectorstore(documents=test_docs, recreate_index=False)
    
    # Test search
    results = vectorstore.similarity_search_with_score("soil health", k=2)
    
    print("\nSearch Results:")
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print()


if __name__ == "__main__":
    main()
