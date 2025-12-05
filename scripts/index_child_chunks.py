#!/usr/bin/env python3
"""
Index Child Chunks for Vector Database

This script loads child chunks from the batch extraction process and
indexes them into Pinecone for RAG retrieval.

Child chunks are ~600 token chunks that:
- Strictly nest within parent chunks (used for extraction)
- Include parent_id in metadata for tracing back to extractions
- Are sized appropriately for RAG retrieval

Usage:
    python scripts/index_child_chunks.py                    # Index all chunks
    python scripts/index_child_chunks.py --dry-run          # Preview without indexing
    python scripts/index_child_chunks.py --source episode   # Only episodes
    python scripts/index_child_chunks.py --source book      # Only books
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document


def load_child_chunks(child_chunks_path: Path) -> List[Dict[str, Any]]:
    """
    Load child chunks from JSON file.

    Args:
        child_chunks_path: Path to child_chunks.json from batch extraction

    Returns:
        List of child chunk dictionaries
    """
    if not child_chunks_path.exists():
        raise FileNotFoundError(f"Child chunks file not found: {child_chunks_path}")

    with open(child_chunks_path) as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} child chunks from {child_chunks_path}")
    return chunks


def convert_to_documents(
    chunks: List[Dict[str, Any]],
    source_filter: Optional[str] = None
) -> List[Document]:
    """
    Convert child chunks to LangChain Documents for indexing.

    Args:
        chunks: List of child chunk dictionaries
        source_filter: Optional filter for "episode" or "book"

    Returns:
        List of LangChain Document objects
    """
    documents = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source_type = metadata.get("source_type", "unknown")

        # Apply source filter if specified
        if source_filter and source_type != source_filter:
            continue

        # Build document metadata
        doc_metadata = {
            "chunk_id": chunk["id"],
            "parent_id": chunk["parent_id"],
            "content_type": source_type,
            "source_id": metadata.get("source_id", ""),
            "parent_index": metadata.get("parent_index", 0),
            "child_index": metadata.get("child_index", 0),
            "start_offset": chunk.get("start_offset", 0),
            "end_offset": chunk.get("end_offset", 0),
        }

        # Add episode-specific metadata
        if source_type == "episode":
            episode_num = metadata.get("source_id", "0")
            doc_metadata["episode_number"] = str(episode_num)
            # Load episode metadata for additional context
            doc_metadata["title"] = f"Episode {episode_num}"

        # Add book-specific metadata
        elif source_type == "book":
            doc_metadata["book_slug"] = metadata.get("source_id", "")
            doc_metadata["title"] = metadata.get("source_id", "Unknown Book")

        doc = Document(
            page_content=chunk["content"],
            metadata=doc_metadata
        )
        documents.append(doc)

    return documents


def enrich_documents_with_metadata(
    documents: List[Document],
    episodes_metadata_path: Optional[Path] = None,
    books_metadata_path: Optional[Path] = None
) -> List[Document]:
    """
    Enrich documents with additional metadata from source files.

    Args:
        documents: List of Document objects
        episodes_metadata_path: Path to episode_metadata.json
        books_metadata_path: Path to books metadata directory

    Returns:
        Enriched documents with titles, dates, etc.
    """
    # Load episode metadata if available
    episode_metadata = {}
    if episodes_metadata_path and episodes_metadata_path.exists():
        with open(episodes_metadata_path) as f:
            data = json.load(f)
            for ep in data.get("episodes", []):
                episode_metadata[str(ep["number"])] = ep

    # Enrich documents
    for doc in documents:
        if doc.metadata.get("content_type") == "episode":
            ep_num = doc.metadata.get("episode_number", "")
            if ep_num in episode_metadata:
                ep_data = episode_metadata[ep_num]
                doc.metadata["title"] = ep_data.get("title", doc.metadata.get("title", ""))
                doc.metadata["guest_name"] = ep_data.get("guest", "")
                doc.metadata["publish_date"] = ep_data.get("date", "")

    return documents


def index_documents_to_pinecone(
    documents: List[Document],
    namespace: str = "",
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Index documents to Pinecone vector database.

    Args:
        documents: List of Document objects to index
        namespace: Pinecone namespace (optional)
        batch_size: Number of documents per batch

    Returns:
        Statistics about the indexing operation
    """
    from src.rag.vectorstore import get_vectorstore

    print(f"Initializing Pinecone vectorstore...")
    vectorstore = get_vectorstore()

    print(f"Indexing {len(documents)} documents in batches of {batch_size}...")
    start_time = datetime.now()

    indexed_count = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        try:
            vectorstore.add_texts(texts=texts, metadatas=metadatas, namespace=namespace)
            indexed_count += len(batch)

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(documents):
                print(f"  Indexed {indexed_count}/{len(documents)} documents...")

        except Exception as e:
            print(f"Error indexing batch {i//batch_size}: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()

    return {
        "total_documents": len(documents),
        "indexed_successfully": indexed_count,
        "elapsed_seconds": elapsed,
        "documents_per_second": indexed_count / elapsed if elapsed > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Index child chunks for RAG retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("data/batch_jobs/child_chunks.json"),
        help="Path to child_chunks.json (default: data/batch_jobs/child_chunks.json)"
    )
    parser.add_argument(
        "--source",
        choices=["episode", "book"],
        help="Only index specific source type"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview documents without indexing"
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="Pinecone namespace for isolation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Documents per indexing batch (default: 100)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Child Chunk Indexer")
    print("=" * 60)
    print()

    # Load chunks
    try:
        chunks = load_child_chunks(args.chunks_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you've run the batch extraction first:")
        print("  python scripts/extract_episodes_batch.py --submit")
        sys.exit(1)

    # Convert to documents
    documents = convert_to_documents(chunks, source_filter=args.source)
    print(f"Converted {len(documents)} chunks to documents")

    if args.source:
        print(f"  (filtered to {args.source} only)")

    # Enrich with metadata
    episodes_meta_path = Path("data/processed/episode_metadata.json")
    documents = enrich_documents_with_metadata(
        documents,
        episodes_metadata_path=episodes_meta_path
    )

    # Statistics
    episode_docs = [d for d in documents if d.metadata.get("content_type") == "episode"]
    book_docs = [d for d in documents if d.metadata.get("content_type") == "book"]
    print(f"\nDocument breakdown:")
    print(f"  - Episode chunks: {len(episode_docs)}")
    print(f"  - Book chunks: {len(book_docs)}")

    if args.dry_run:
        print("\n[DRY RUN] Preview of documents:")
        for doc in documents[:5]:
            print(f"\n  ID: {doc.metadata.get('chunk_id')}")
            print(f"  Parent: {doc.metadata.get('parent_id')}")
            print(f"  Type: {doc.metadata.get('content_type')}")
            print(f"  Content: {doc.page_content[:100]}...")
        print(f"\n  ... and {len(documents) - 5} more documents")
        print("\n[DRY RUN] No documents were indexed")
        return

    # Index to Pinecone
    print()
    stats = index_documents_to_pinecone(
        documents,
        namespace=args.namespace,
        batch_size=args.batch_size
    )

    print(f"\n{'=' * 40}")
    print("Indexing Complete")
    print(f"{'=' * 40}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Successfully indexed: {stats['indexed_successfully']}")
    print(f"Time elapsed: {stats['elapsed_seconds']:.1f} seconds")
    print(f"Rate: {stats['documents_per_second']:.1f} docs/second")

    if stats['indexed_successfully'] < stats['total_documents']:
        failed = stats['total_documents'] - stats['indexed_successfully']
        print(f"\n⚠️ {failed} documents failed to index")


if __name__ == "__main__":
    main()
