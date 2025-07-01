"""
Main script to process episodes and prepare them for vector database
"""
import logging
import json
from pathlib import Path
from datetime import datetime

from .episode_processor import EpisodeProcessor
from .chunker import TranscriptChunker
from ..config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_episodes_for_ingestion():
    """Process episodes and prepare chunks for vector database"""
    logger.info("Starting episode processing for vector database ingestion")
    
    # Initialize processors
    episode_processor = EpisodeProcessor()
    chunker = TranscriptChunker()
    
    # Load episodes
    logger.info(f"Loading episodes (limit: {settings.episodes_to_process})")
    all_episodes = episode_processor.load_episodes()
    
    # Select diverse episodes for MVP
    episodes = episode_processor.get_diverse_episodes(
        all_episodes, 
        count=settings.episodes_to_process
    )
    
    # Save episode metadata
    episode_processor.save_processed_episodes(episodes)
    
    # Chunk episodes
    logger.info("Chunking episodes for vector storage")
    documents = chunker.chunk_episodes(episodes)
    
    # Save chunks for reference
    chunks_data = []
    for doc in documents:
        chunks_data.append({
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
            "content_length": len(doc.page_content)
        })
    
    chunks_file = settings.processed_dir / "chunks_preview.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(documents),
            "total_episodes": len(episodes),
            "chunk_config": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            },
            "processed_at": datetime.now().isoformat(),
            "chunks_preview": chunks_data[:10]  # Save first 10 chunks as preview
        }, f, indent=2)
    
    logger.info(f"Saved chunks preview to {chunks_file}")
    
    # Summary statistics
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chunk_size = total_chars / len(documents) if documents else 0
    
    logger.info(f"""
Episode Processing Complete:
- Episodes processed: {len(episodes)}
- Total chunks created: {len(documents)}
- Average chunk size: {avg_chunk_size:.0f} characters
- Total content: {total_chars:,} characters
""")
    
    return documents


def main():
    """Run episode processing"""
    try:
        documents = process_episodes_for_ingestion()
        logger.info("Episode processing completed successfully")
        return documents
    except Exception as e:
        logger.error(f"Error processing episodes: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()