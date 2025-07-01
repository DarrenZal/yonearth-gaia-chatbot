"""
FastAPI main application for YonEarth Gaia chatbot
"""
import time
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ..config import settings
from ..rag.chain import get_rag_chain
from .models import (
    ChatRequest, ChatResponse, Citation,
    RecommendationsRequest, RecommendationsResponse, EpisodeRecommendation,
    SearchRequest, SearchResponse, SearchResult,
    HealthResponse, ErrorResponse
)
from .bm25_endpoints import router as bm25_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global variables
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global rag_chain
    
    # Startup
    logger.info("Starting YonEarth Gaia Chatbot API...")
    try:
        rag_chain = get_rag_chain()
        rag_chain.initialize()
        logger.info("RAG chain initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        # Continue anyway - will initialize on first request
    
    yield
    
    # Shutdown
    logger.info("Shutting down YonEarth Gaia Chatbot API...")


# Create FastAPI app
app = FastAPI(
    title="YonEarth Gaia Chatbot API",
    description="Chat with Gaia, the spirit of Earth, using wisdom from YonEarth podcast episodes",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list + ["*"] if settings.debug else settings.allowed_origins_list,
    allow_credentials=False,  # Set to False when allowing all origins
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted host middleware for production
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "152.53.194.214", "yonearth.org", "*.yonearth.org", "*.onrender.com"]
    )

# Include BM25 RAG router (with error handling)
try:
    app.include_router(bm25_router)
    logger.info("✅ BM25 router loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load BM25 router: {e}")
    # Add a simple test endpoint instead
    @app.get("/bm25/status")
    async def bm25_status():
        return {
            "status": "BM25 system implemented but not loaded",
            "error": str(e),
            "solution": "Install dependencies: pip install rank-bm25 sentence-transformers",
            "restart_needed": True
        }


def get_rag_dependency():
    """Dependency to get RAG chain instance"""
    global rag_chain
    if not rag_chain:
        rag_chain = get_rag_chain()
        rag_chain.initialize()
    return rag_chain


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Simple health check without RAG dependency issues
        global rag_chain
        rag_status = rag_chain is not None
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "rag_initialized": rag_status,
            "service": "YonEarth Gaia Chatbot",
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "version": "1.0.0",
            "error": str(e)
        }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat_with_gaia(
    request: Request,
    chat_request: ChatRequest,
    rag: Any = Depends(get_rag_dependency)
):
    """Chat with Gaia using RAG pipeline"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing chat request: {chat_request.message[:50]}...")
        
        # Process query through RAG chain
        response = rag.query(
            user_input=chat_request.message,
            k=chat_request.max_results,
            session_id=chat_request.session_id,
            personality_variant=chat_request.personality,
            custom_prompt=chat_request.custom_prompt
        )
        
        # Convert citations to response model
        citations = [
            Citation(
                episode_number=cite["episode_number"],
                title=cite["title"],
                guest_name=cite["guest_name"],
                url=cite["url"],
                relevance=cite.get("relevance", "High")
            )
            for cite in response.get("citations", [])
        ]
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response["response"],
            personality=response.get("personality", "warm_mother"),
            citations=citations,
            context_used=response.get("context_used", 0),
            session_id=chat_request.session_id,
            retrieval_count=response.get("retrieval_count", 0),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="I apologize, dear one, but I'm experiencing difficulties right now. Please try again in a moment."
        )


@app.post("/recommendations", response_model=RecommendationsResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def get_episode_recommendations(
    request: Request,
    rec_request: RecommendationsRequest,
    rag: Any = Depends(get_rag_dependency)
):
    """Get episode recommendations based on query"""
    try:
        logger.info(f"Getting recommendations for: {rec_request.query[:50]}...")
        
        recommendations = rag.get_episode_recommendations(
            user_input=rec_request.query,
            k=rec_request.max_recommendations
        )
        
        # Convert to response model
        rec_models = [
            EpisodeRecommendation(
                episode_number=rec["episode_number"],
                title=rec["title"],
                guest_name=rec["guest_name"],
                url=rec["url"],
                relevance_score=rec["relevance_score"],
                reason=rec["reason"]
            )
            for rec in recommendations
        ]
        
        return RecommendationsResponse(
            query=rec_request.query,
            recommendations=rec_models,
            total_found=len(rec_models)
        )
        
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")


@app.post("/search", response_model=SearchResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def search_episodes(
    request: Request,
    search_request: SearchRequest,
    rag: Any = Depends(get_rag_dependency)
):
    """Search episodes with optional filters"""
    try:
        logger.info(f"Searching episodes for: {search_request.query[:50]}...")
        
        results = rag.search_episodes(
            query=search_request.query,
            filters=search_request.filters,
            k=search_request.max_results
        )
        
        # Convert to response model
        search_results = [
            SearchResult(
                episode_number=result["episode_number"],
                title=result["title"],
                guest_name=result["guest_name"],
                url=result["url"],
                relevance_score=result["relevance_score"],
                content_preview=result["content_preview"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        return SearchResponse(
            query=search_request.query,
            results=search_results,
            total_found=len(search_results),
            filters_applied=search_request.filters
        )
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/reset-conversation")
@limiter.limit("5/minute")
async def reset_conversation(
    request: Request,
    session_id: str = None,
    rag: Any = Depends(get_rag_dependency)
):
    """Reset conversation memory for a session"""
    try:
        rag.reset_conversation(session_id=session_id)
        return {"message": "Conversation reset successfully", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset conversation")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the YonEarth Gaia Chatbot API",
        "description": "Chat with Gaia, the spirit of Earth, using wisdom from YonEarth podcast episodes",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "recommendations": "/recommendations", 
            "search": "/search",
            "health": "/health"
        }
    }

@app.get("/test")
async def test():
    """Simple test endpoint"""
    return {"status": "ok", "message": "Test endpoint working"}


# Exception handlers
@app.exception_handler(500)
async def internal_server_error(request: Request, exc: Exception):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "Something went wrong on our end. Please try again later."
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )