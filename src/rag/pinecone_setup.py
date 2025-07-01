"""
Pinecone vector database setup and configuration
"""
import os
import logging
from typing import Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec

from ..config import settings

logger = logging.getLogger(__name__)


class PineconeManager:
    """Manage Pinecone vector database operations"""
    
    def __init__(self):
        self.api_key = settings.pinecone_api_key
        self.environment = settings.pinecone_environment
        self.index_name = settings.pinecone_index_name
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
    def create_index(self, recreate: bool = False) -> None:
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check existing indexes
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                if recreate:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.pc.delete_index(self.index_name)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    self.index = self.pc.Index(self.index_name)
                    return
            
            # Create new index
            logger.info(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Free tier region
                )
            )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            import time
            time.sleep(10)  # Give it time to initialize
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Index {self.index_name} created successfully")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def get_index(self):
        """Get Pinecone index instance"""
        if not self.index:
            self.index = self.pc.Index(self.index_name)
        return self.index
    
    def get_index_stats(self) -> dict:
        """Get statistics about the index"""
        try:
            index = self.get_index()
            stats = index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def delete_all_vectors(self) -> None:
        """Delete all vectors from the index"""
        try:
            index = self.get_index()
            index.delete(delete_all=True)
            logger.info("All vectors deleted from index")
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def verify_connection(self) -> bool:
        """Verify Pinecone connection and index"""
        try:
            stats = self.get_index_stats()
            logger.info(f"Pinecone connection verified. Index stats: {stats}")
            return True
        except Exception as e:
            logger.error(f"Pinecone connection failed: {e}")
            return False


def setup_pinecone(recreate_index: bool = False) -> PineconeManager:
    """Initialize Pinecone for the application"""
    manager = PineconeManager()
    manager.create_index(recreate=recreate_index)
    
    if manager.verify_connection():
        logger.info("Pinecone setup completed successfully")
    else:
        raise Exception("Failed to setup Pinecone")
    
    return manager


def main():
    """Test Pinecone setup"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test connection and index creation
        manager = setup_pinecone(recreate_index=False)
        
        # Get index stats
        stats = manager.get_index_stats()
        print(f"\nIndex Statistics:")
        print(f"Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"Index dimension: {stats.get('dimension', 0)}")
        
    except Exception as e:
        logger.error(f"Pinecone setup failed: {e}")
        raise


if __name__ == "__main__":
    main()