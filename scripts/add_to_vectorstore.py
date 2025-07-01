#!/usr/bin/env python3
"""
Script to add processed episodes to the vector database
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vectorstore import create_vectorstore
from src.ingestion.process_episodes import process_episodes_for_ingestion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Add processed episodes to vector database"""
    print("🌍 Adding Episodes to Vector Database")
    print("=" * 40)
    
    try:
        # Process episodes to get documents
        print("📊 Processing episodes...")
        documents = process_episodes_for_ingestion()
        
        print(f"✅ Created {len(documents)} document chunks")
        
        # Create vectorstore and add documents
        print("🔄 Adding to vector database...")
        vectorstore = create_vectorstore(documents=documents, recreate_index=False)
        
        # Get final stats
        stats = vectorstore.get_stats()
        print(f"\n🎉 Vector Database Updated!")
        print(f"📈 Final Stats:")
        print(f"   - Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   - Index dimension: {stats.get('dimension', 0)}")
        
        print(f"\n✅ Ready for chatbot use!")
        
    except Exception as e:
        logger.error(f"Failed to add to vectorstore: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())