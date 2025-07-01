#!/usr/bin/env python3
"""
Setup script to process episodes and initialize the vector database
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chain import YonEarthRAGChain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the RAG system with episode data"""
    print("🌍 YonEarth Gaia Chatbot - Data Setup")
    print("=" * 40)
    
    try:
        # Initialize RAG chain with data processing
        print("📊 Processing episodes and setting up vector database...")
        rag_chain = YonEarthRAGChain(initialize_data=True)
        
        # Get stats
        stats = rag_chain.get_stats()
        print(f"\n✅ Setup Complete!")
        print(f"📈 Vector Database Stats:")
        print(f"   - Total vectors: {stats['vectorstore_stats'].get('total_vector_count', 0)}")
        print(f"   - Index dimension: {stats['vectorstore_stats'].get('dimension', 0)}")
        print(f"   - Gaia personality: {stats['gaia_personality']}")
        
        # Test query
        print(f"\n🧪 Testing with sample query...")
        response = rag_chain.query("Tell me about regenerative agriculture")
        print(f"   - Response length: {len(response['response'])} characters")
        print(f"   - Citations: {len(response.get('citations', []))}")
        print(f"   - Retrieved docs: {response.get('retrieval_count', 0)}")
        
        print(f"\n🎉 RAG system ready! You can now start the chatbot.")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        print(f"\n❌ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())