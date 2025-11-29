"""
MemoRAG API endpoints for Our Biggest Deal book Q&A.

Provides a dedicated chat endpoint that uses MemoRAG for context retrieval
from the "Our Biggest Deal" book.
"""
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memorag", tags=["memorag"])

# Global MemoRAG pipeline - loaded on first request
_memorag_pipe = None


class MemoRAGChatRequest(BaseModel):
    """Request model for MemoRAG chat."""
    message: str = Field(..., description="User's question about the book")
    max_tokens: int = Field(default=512, ge=64, le=2048, description="Max tokens in response")


class MemoRAGChatResponse(BaseModel):
    """Response model for MemoRAG chat."""
    answer: str = Field(..., description="Generated answer from MemoRAG")
    query_time_ms: int = Field(..., description="Query processing time in milliseconds")
    source: str = Field(default="Our Biggest Deal", description="Source material")


def get_memorag_pipeline():
    """
    Get or initialize the MemoRAG pipeline.
    Uses lazy loading to avoid startup delays.
    """
    global _memorag_pipe

    if _memorag_pipe is not None:
        return _memorag_pipe

    logger.info("Loading MemoRAG pipeline...")

    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        from memorag import MemoRAG, Agent

        # Memory index location
        memory_dir = Path(__file__).parent.parent.parent / "experiments" / "memorag" / "indices"

        if not memory_dir.exists():
            raise FileNotFoundError(f"Memory directory not found: {memory_dir}")

        # Check for required files
        required_files = ["memory.bin", "index.bin", "chunks.json"]
        for f in required_files:
            if not (memory_dir / f).exists():
                raise FileNotFoundError(f"Missing required file: {memory_dir / f}")

        # Configure OpenAI generation
        openai_key = os.getenv("OPENAI_API_KEY")
        customized_gen_model = None
        if openai_key:
            logger.info("Configuring hybrid mode: Local retrieval + OpenAI generation")
            customized_gen_model = Agent(
                model="gpt-4.1-mini",
                source="openai",
                api_dict={"api_key": openai_key}
            )
        else:
            logger.warning("OPENAI_API_KEY not found - using local generation (slow)")

        # Initialize pipeline
        model_name = "Qwen/Qwen2-1.5B-Instruct"
        cache_path = str(memory_dir.parent / "model_cache")

        _memorag_pipe = MemoRAG(
            mem_model_name_or_path=model_name,
            ret_model_name_or_path=model_name,
            cache_dir=cache_path,
            customized_gen_model=customized_gen_model,
            enable_flash_attn=False,
            load_in_4bit=False
        )

        # Load the memory index
        _memorag_pipe.load(str(memory_dir))

        logger.info("MemoRAG pipeline loaded successfully")
        return _memorag_pipe

    except ImportError as e:
        logger.error(f"MemoRAG not installed: {e}")
        raise HTTPException(
            status_code=503,
            detail="MemoRAG not installed on this server"
        )
    except Exception as e:
        logger.error(f"Failed to load MemoRAG pipeline: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load MemoRAG: {str(e)}"
        )


@router.post("/chat", response_model=MemoRAGChatResponse)
async def memorag_chat(request: MemoRAGChatRequest):
    """
    Chat endpoint using MemoRAG for Our Biggest Deal book Q&A.

    Uses hybrid architecture:
    - Local CPU retrieval with Qwen2-1.5B-Instruct
    - Cloud generation with GPT-4.1-mini (via OpenAI API)
    """
    start_time = time.time()

    try:
        pipe = get_memorag_pipeline()

        # Query the memory using mem_model.answer()
        answer = pipe.mem_model.answer(
            request.message,
            max_new_tokens=request.max_tokens
        )

        query_time_ms = int((time.time() - start_time) * 1000)

        return MemoRAGChatResponse(
            answer=answer,
            query_time_ms=query_time_ms,
            source="Our Biggest Deal"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MemoRAG query error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/health")
async def memorag_health():
    """
    Health check for MemoRAG service.
    Returns status of the memory index.
    """
    memory_dir = Path(__file__).parent.parent.parent / "experiments" / "memorag" / "indices"

    files_exist = {
        "memory.bin": (memory_dir / "memory.bin").exists(),
        "index.bin": (memory_dir / "index.bin").exists(),
        "chunks.json": (memory_dir / "chunks.json").exists()
    }

    all_present = all(files_exist.values())

    return {
        "status": "healthy" if all_present else "degraded",
        "memory_loaded": _memorag_pipe is not None,
        "index_files": files_exist,
        "book": "Our Biggest Deal"
    }
