"""
MemoRAG API endpoints for Our Biggest Deal book Q&A.

Uses direct RAG with OpenAI embeddings for retrieval from the book chunks.
This bypasses MemoRAG's memory.bin to ensure full book coverage.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memorag", tags=["memorag"])

# Global state - loaded on first request
_chunks: List[str] = []
_chunk_embeddings: np.ndarray = None
_openai_client = None

# Configuration
MEMORY_DIR = Path(__file__).parent.parent.parent / "experiments" / "memorag" / "indices"
EMBEDDINGS_CACHE_PATH = MEMORY_DIR / "chunk_embeddings.npy"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_CHUNKS = 8  # Number of relevant chunks to retrieve


class MemoRAGChatRequest(BaseModel):
    """Request model for MemoRAG chat."""
    message: str = Field(..., description="User's question about the book")
    max_tokens: int = Field(default=512, ge=64, le=2048, description="Max tokens in response")


class MemoRAGChatResponse(BaseModel):
    """Response model for MemoRAG chat."""
    answer: str = Field(..., description="Generated answer")
    query_time_ms: int = Field(..., description="Query processing time in milliseconds")
    source: str = Field(default="Our Biggest Deal", description="Source material")


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def load_chunks() -> List[str]:
    """Load chunks from JSON file."""
    global _chunks
    if _chunks:
        return _chunks

    chunks_path = MEMORY_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found at {chunks_path}")

    with open(chunks_path, 'r') as f:
        _chunks = json.load(f)

    logger.info(f"Loaded {len(_chunks)} chunks from {chunks_path}")
    return _chunks


def get_embedding(text: str) -> np.ndarray:
    """Get OpenAI embedding for text."""
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding)


def load_or_create_embeddings(chunks: List[str]) -> np.ndarray:
    """Load embeddings from cache or create them."""
    global _chunk_embeddings

    if _chunk_embeddings is not None:
        return _chunk_embeddings

    # Check for cached embeddings
    if EMBEDDINGS_CACHE_PATH.exists():
        logger.info(f"Loading cached embeddings from {EMBEDDINGS_CACHE_PATH}")
        _chunk_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
        if len(_chunk_embeddings) == len(chunks):
            logger.info(f"Loaded {len(_chunk_embeddings)} cached embeddings")
            return _chunk_embeddings
        else:
            logger.warning(f"Cached embeddings count ({len(_chunk_embeddings)}) doesn't match chunks ({len(chunks)}), regenerating...")

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    client = get_openai_client()

    # Process in batches of 100
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info(f"Generated embeddings for chunks {i} to {i + len(batch)}")

    _chunk_embeddings = np.array(all_embeddings)

    # Cache embeddings
    np.save(EMBEDDINGS_CACHE_PATH, _chunk_embeddings)
    logger.info(f"Cached {len(_chunk_embeddings)} embeddings to {EMBEDDINGS_CACHE_PATH}")

    return _chunk_embeddings


def retrieve_relevant_chunks(query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = TOP_K_CHUNKS) -> List[str]:
    """Retrieve top-k relevant chunks using cosine similarity."""
    query_embedding = get_embedding(query)

    # Compute cosine similarities
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant_chunks = [chunks[i] for i in top_indices]

    logger.info(f"Retrieved {len(relevant_chunks)} chunks with similarities: {[f'{similarities[i]:.3f}' for i in top_indices]}")

    return relevant_chunks


def generate_answer(query: str, context_chunks: List[str], max_tokens: int = 512) -> str:
    """Generate answer using GPT-4.1-mini with retrieved context."""
    client = get_openai_client()

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = """You are an AI assistant helping to answer questions about the book "Our Biggest Deal: Pathways to Planetary Prosperity" by Aaron William Perry.

Use ONLY the provided context to answer questions. If the context doesn't contain enough information to answer, say so clearly.

Be concise but comprehensive. Reference specific content from the book when possible."""

    user_prompt = f"""Context from the book:

{context}

---

Question: {query}

Answer based on the book context above:"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.3
    )

    return response.choices[0].message.content


@router.post("/chat", response_model=MemoRAGChatResponse)
async def memorag_chat(request: MemoRAGChatRequest):
    """
    Chat endpoint for Our Biggest Deal book Q&A.

    Uses OpenAI embeddings for retrieval and GPT-4.1-mini for generation.
    This ensures the full book content is searchable.
    """
    start_time = time.time()

    try:
        # Load chunks and embeddings
        chunks = load_chunks()
        embeddings = load_or_create_embeddings(chunks)

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            request.message,
            chunks,
            embeddings,
            top_k=TOP_K_CHUNKS
        )

        # Generate answer
        answer = generate_answer(
            request.message,
            relevant_chunks,
            max_tokens=request.max_tokens
        )

        query_time_ms = int((time.time() - start_time) * 1000)

        return MemoRAGChatResponse(
            answer=answer,
            query_time_ms=query_time_ms,
            source="Our Biggest Deal"
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/health")
async def memorag_health():
    """
    Health check for the book Q&A service.
    """
    chunks_exist = (MEMORY_DIR / "chunks.json").exists()
    embeddings_exist = EMBEDDINGS_CACHE_PATH.exists()

    # Count chunks if file exists
    chunk_count = 0
    if chunks_exist:
        try:
            with open(MEMORY_DIR / "chunks.json", 'r') as f:
                chunk_count = len(json.load(f))
        except:
            pass

    return {
        "status": "healthy" if chunks_exist else "degraded",
        "chunks_loaded": len(_chunks) if _chunks else 0,
        "embeddings_cached": embeddings_exist,
        "total_chunks": chunk_count,
        "book": "Our Biggest Deal",
        "retrieval_method": "OpenAI embeddings + GPT-4.1-mini"
    }
