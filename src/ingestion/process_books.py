"""
Main script to process books and prepare them for vector database
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from .book_processor import BookProcessor
from .chunker import DocumentChunker
from ..config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_books_for_ingestion():
    """Process books and prepare chunks for vector database"""
    logger.info("Starting book processing for vector database ingestion")
    
    # Initialize processors
    book_processor = BookProcessor()
    chunker = DocumentChunker()
    
    # Load books
    logger.info("Loading books from metadata files")
    books = book_processor.load_books()
    
    if not books:
        logger.warning("No books found to process")
        return []
    
    # Process each book
    all_documents = []
    
    for book in books:
        logger.info(f"Processing book: {book.title}")
        
        try:
            documents = book_processor.process_book(book)
            
            # Chunk book documents
            chunked_docs = chunker.chunk_documents(documents)
            all_documents.extend(chunked_docs)
            
            logger.info(f"Added {len(chunked_docs)} chunks from {book.title}")
            
        except Exception as e:
            logger.error(f"Error processing book {book.title}: {e}")
            continue
    
    # Save book metadata
    book_processor.save_processed_books(books)
    
    # Save chunks for reference
    chunks_data = []
    for doc in all_documents:
        chunks_data.append({
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
            "content_length": len(doc.page_content)
        })
    
    chunks_file = settings.processed_dir / "book_chunks_preview.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(all_documents),
            "total_books": len(books),
            "chunk_config": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            },
            "processed_at": datetime.now().isoformat(),
            "chunks_preview": chunks_data[:10]  # Save first 10 chunks as preview
        }, f, indent=2)
    
    logger.info(f"Saved book chunks preview to {chunks_file}")
    
    # Summary statistics
    total_chars = sum(len(doc.page_content) for doc in all_documents)
    avg_chunk_size = total_chars / len(all_documents) if all_documents else 0
    
    logger.info(f"""
Book Processing Complete:
- Books processed: {len(books)}
- Total chunks created: {len(all_documents)}
- Average chunk size: {avg_chunk_size:.0f} characters
- Total content: {total_chars:,} characters
""")
    
    return all_documents


def add_books_to_vectorstore():
    """Process books and add them to the vector database"""
    from ..rag.vectorstore import YonEarthVectorStore
    from ..rag.pinecone_setup import setup_pinecone
    
    logger.info("Adding books to vector database")
    
    # Setup Pinecone
    setup_pinecone()
    
    # Process books
    documents = process_books_for_ingestion()
    
    if not documents:
        logger.warning("No book documents to add to vector database")
        return
    
    # Add to vector database
    vectorstore = YonEarthVectorStore()
    
    # Add documents in batches to avoid memory issues
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        try:
            # Documents are already LangChain Document objects from DocumentChunker
            vectorstore.add_documents(batch)
            logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} to vector database")
            
        except Exception as e:
            logger.error(f"Error adding batch to vector database: {e}")
            continue
    
    logger.info(f"Successfully added {len(documents)} book documents to vector database")


def main():
    """Run book processing"""
    try:
        documents = process_books_for_ingestion()
        logger.info("Book processing completed successfully")
        return documents
    except Exception as e:
        logger.error(f"Error processing books: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()