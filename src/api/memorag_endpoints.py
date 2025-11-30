"""
MemoRAG API endpoints for Our Biggest Deal book Q&A.

Uses SHARDED MemoRAG architecture with Qwen model:
1. BM25 routing to select relevant shards
2. MemoRAG memory model (Qwen) for reasoning/answers
3. GPT-4.1-mini for final generation

This preserves MemoRAG's "clue" reasoning capabilities.
"""
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memorag", tags=["memorag"])

# Configuration
INDICES_DIR = Path(__file__).parent.parent.parent / "experiments" / "memorag" / "indices"
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# Global state - loaded lazily
_router: Optional["BM25ShardRouter"] = None
_pipe = None  # MemoRAG pipeline


class MemoRAGChatRequest(BaseModel):
    """Request model for MemoRAG chat."""
    message: str = Field(..., description="User's question about the book")
    max_tokens: int = Field(default=512, ge=64, le=2048, description="Max tokens in response")
    top_shards: int = Field(default=3, ge=1, le=10, description="Number of shards to query")


class MemoRAGChatResponse(BaseModel):
    """Response model for MemoRAG chat."""
    answer: str = Field(..., description="Generated answer")
    query_time_ms: int = Field(..., description="Query processing time in milliseconds")
    shards_queried: List[int] = Field(default=[], description="Which shards were queried")
    source: str = Field(default="Our Biggest Deal", description="Source material")


class BM25ShardRouter:
    """
    BM25-based router to select most relevant shards for a query.
    Uses shard metadata (top_terms) for fast pre-selection.
    """

    def __init__(self, metadata_path: Path):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.num_shards = self.metadata["num_shards"]
        self.shards = self.metadata["shards"]

        # Build inverted index from shard terms
        self.term_to_shards: Dict[str, List[int]] = {}
        self.shard_term_counts: Dict[int, Counter] = {}

        for shard in self.shards:
            idx = shard["index"]
            terms = shard["top_terms"]
            self.shard_term_counts[idx] = Counter(terms)

            for term in terms:
                if term not in self.term_to_shards:
                    self.term_to_shards[term] = []
                self.term_to_shards[term].append(idx)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        words = text.lower().split()
        return [''.join(c for c in word if c.isalnum()) for word in words if len(word) > 2]

    def _bm25_score(self, query_terms: List[str], shard_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calculate BM25 score for a shard given query terms.
        """
        shard_terms = set(self.shard_term_counts[shard_idx].keys())
        avg_terms = sum(len(s["top_terms"]) for s in self.shards) / len(self.shards)

        score = 0.0
        for term in query_terms:
            if term in shard_terms:
                # IDF component
                n_containing = len(self.term_to_shards.get(term, []))
                idf = math.log((self.num_shards - n_containing + 0.5) / (n_containing + 0.5) + 1)

                # TF component (simplified)
                tf = 1.0
                doc_len = len(shard_terms)
                tf_normalized = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_terms)))

                score += idf * tf_normalized

        return score

    def route(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Route query to most relevant shards using BM25.

        Returns:
            List of (shard_idx, score) tuples, sorted by score descending
        """
        query_terms = self._tokenize(query)

        scores = []
        for shard in self.shards:
            idx = shard["index"]
            score = self._bm25_score(query_terms, idx)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])

        # Return top-k shards (at least 1, even if score is 0)
        selected = scores[:top_k]

        # If all scores are 0, include all shards (fallback)
        if all(s[1] == 0 for s in selected):
            logger.warning("No term matches, querying all shards")
            return [(s["index"], 0.0) for s in self.shards]

        return selected


def get_router() -> BM25ShardRouter:
    """Get or create the BM25 router."""
    global _router
    if _router is None:
        metadata_path = INDICES_DIR / "shard_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Shard metadata not found at {metadata_path}")
        _router = BM25ShardRouter(metadata_path)
        logger.info(f"Loaded BM25 router with {_router.num_shards} shards")
    return _router


def get_memorag_pipeline():
    """Get or create the MemoRAG pipeline."""
    global _pipe
    if _pipe is not None:
        return _pipe

    try:
        from memorag import MemoRAG, Agent
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="MemoRAG not installed on server"
        )

    logger.info("Initializing MemoRAG pipeline with Qwen model...")

    # Configure OpenAI for generation
    customized_gen_model = None
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        customized_gen_model = Agent(
            model="gpt-4.1-mini",
            source="openai",
            api_dict={"api_key": openai_key}
        )
        logger.info("Using GPT-4.1-mini for generation")

    cache_path = str(INDICES_DIR.parent / "model_cache")

    _pipe = MemoRAG(
        mem_model_name_or_path=MODEL_NAME,
        ret_model_name_or_path=MODEL_NAME,
        cache_dir=cache_path,
        customized_gen_model=customized_gen_model,
        enable_flash_attn=False,
        load_in_4bit=False
    )

    logger.info("MemoRAG pipeline initialized")
    return _pipe


@router.post("/chat", response_model=MemoRAGChatResponse)
async def memorag_chat(request: MemoRAGChatRequest):
    """
    Chat endpoint for Our Biggest Deal book Q&A.

    Uses SHARDED MemoRAG architecture:
    1. BM25 routing to select relevant shards
    2. Query each shard with MemoRAG's Qwen memory model
    3. Return best answer based on routing score
    """
    start_time = time.time()

    try:
        # Get router
        router_instance = get_router()

        # Route to relevant shards
        logger.info(f"Routing query: {request.message[:50]}...")
        selected_shards = router_instance.route(request.message, top_k=request.top_shards)
        logger.info(f"Selected shards: {[s[0] for s in selected_shards]}")

        # Get MemoRAG pipeline
        pipe = get_memorag_pipeline()

        # Query each shard
        answers = []
        for shard_idx, route_score in selected_shards:
            shard_path = INDICES_DIR / f"shard_{shard_idx}"

            if not shard_path.exists():
                logger.warning(f"Shard {shard_idx} not found, skipping")
                continue

            logger.info(f"Querying shard {shard_idx}...")

            try:
                # Load shard
                pipe.load(str(shard_path))

                # Query MemoRAG's memory model
                answer = pipe.mem_model.answer(
                    request.message,
                    max_new_tokens=request.max_tokens
                )

                answers.append({
                    "shard_idx": shard_idx,
                    "route_score": route_score,
                    "answer": answer
                })

                logger.info(f"Got answer from shard {shard_idx}")

            except Exception as e:
                logger.error(f"Error querying shard {shard_idx}: {e}")
                continue

        # Select best answer (highest route score with non-empty answer)
        best_answer = None
        for ans in answers:
            if ans["answer"] and not ans["answer"].lower().startswith("the context does not"):
                if best_answer is None or ans["route_score"] > best_answer["route_score"]:
                    best_answer = ans

        # Fallback to first answer if all were "no context"
        if best_answer is None and answers:
            best_answer = answers[0]

        query_time_ms = int((time.time() - start_time) * 1000)

        return MemoRAGChatResponse(
            answer=best_answer["answer"] if best_answer else "No relevant information found in the book.",
            query_time_ms=query_time_ms,
            shards_queried=[s[0] for s in selected_shards],
            source="Our Biggest Deal"
        )

    except FileNotFoundError as e:
        logger.error(f"Index not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="MemoRAG index not built. Please run build_memory.py first."
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
    Health check for the sharded MemoRAG service.
    """
    metadata_path = INDICES_DIR / "shard_metadata.json"
    metadata_exists = metadata_path.exists()

    num_shards = 0
    shard_dirs = []

    if metadata_exists:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            num_shards = metadata.get("num_shards", 0)

            # Check which shards actually exist
            for i in range(num_shards):
                shard_path = INDICES_DIR / f"shard_{i}"
                if shard_path.exists():
                    shard_dirs.append(i)
        except Exception:
            pass

    return {
        "status": "healthy" if len(shard_dirs) > 0 else "degraded",
        "architecture": "Sharded MemoRAG (Qwen + GPT-4.1-mini)",
        "metadata_exists": metadata_exists,
        "expected_shards": num_shards,
        "available_shards": len(shard_dirs),
        "shard_indices": shard_dirs,
        "memory_model": MODEL_NAME,
        "book": "Our Biggest Deal",
        "indices_dir": str(INDICES_DIR)
    }
