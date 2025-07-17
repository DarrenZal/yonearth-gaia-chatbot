"""
BM25 RAG API endpoints for comparison and testing
"""
import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..rag.chain import get_rag_chain
from ..rag.bm25_chain import BM25RAGChain
from .bm25_models import (
    BM25ChatRequest, BM25ChatResponse, BM25Source,
    SearchMethodComparisonRequest, SearchMethodComparisonResponse, SearchMethodResult,
    BM25HealthResponse, BM25SearchRequest, BM25SearchResponse, ContentSearchResult,
    PerformanceComparisonResponse, RAGChainComparisonRequest, RAGChainComparisonResponse,
    RAGChainResult
)

logger = logging.getLogger(__name__)

# Create router for BM25 endpoints
router = APIRouter(prefix="/bm25", tags=["BM25 RAG"])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global BM25 chain instance
_bm25_chain: BM25RAGChain = None


def get_bm25_chain() -> BM25RAGChain:
    """Get or create BM25 RAG chain instance"""
    global _bm25_chain
    if _bm25_chain is None:
        logger.info("Initializing BM25 RAG chain...")
        _bm25_chain = BM25RAGChain(initialize_data=True)
    return _bm25_chain


@router.post("/chat", response_model=BM25ChatResponse)
@limiter.limit("10/minute")
async def bm25_chat(
    request: Request,
    chat_request: BM25ChatRequest
):
    """
    Chat with Gaia using BM25 hybrid RAG
    
    Features:
    - BM25 keyword search + semantic vector search
    - Reciprocal Rank Fusion for result combination
    - Cross-encoder reranking for improved relevance
    - Query-adaptive search strategy
    """
    start_time = time.time()
    
    try:
        bm25_chain = get_bm25_chain()
        
        # Process chat message
        result = bm25_chain.chat(
            message=chat_request.message,
            search_method=chat_request.search_method,
            k=chat_request.k,
            include_sources=chat_request.include_sources,
            personality_variant=chat_request.gaia_personality,
            temperature=chat_request.temperature,
            custom_prompt=chat_request.custom_prompt,
            category_threshold=chat_request.category_threshold
        )
        
        # Format sources - debug and fix the root cause
        formatted_sources = []
        logger.info(f"Processing {len(result.get('sources', []))} sources from BM25 chain")
        
        for i, source in enumerate(result.get('sources', [])):
            logger.info(f"--- Source {i+1} Debug Info ---")
            logger.info(f"Raw source keys: {list(source.keys())}")
            logger.info(f"content_type: {source.get('content_type', 'MISSING')}")
            logger.info(f"title: {source.get('title', 'MISSING')}")
            logger.info(f"episode_number: {source.get('episode_number', 'MISSING')}")
            
            try:
                # Try direct validation first
                bm25_source = BM25Source(**source)
                logger.info(f"✅ Source {i+1} validation SUCCESS")
                formatted_sources.append(bm25_source)
            except Exception as validation_error:
                logger.error(f"❌ Source {i+1} validation FAILED: {validation_error}")
                
                # Handle validation failures by creating proper format
                content_type = source.get('content_type', 'episode')
                logger.info(f"Creating fallback for content_type: {content_type}")
                
                if content_type == 'book':
                    # Create book source with proper format
                    formatted_source = BM25Source(
                        content_type='book',
                        title=source.get('title', 'Unknown Book'),
                        content_preview=source.get('content_preview', ''),
                        book_title=source.get('book_title', 'Unknown Book'),
                        author=source.get('author', 'Unknown Author'),
                        chapter_number=source.get('chapter_number'),
                        chapter_title=source.get('chapter_title', ''),
                        episode_number=source.get('episode_number', 'Book'),
                        url=source.get('url', '')
                    )
                    logger.info(f"Created book fallback: {formatted_source.title}")
                else:
                    # Create episode source with proper format  
                    formatted_source = BM25Source(
                        content_type='episode',
                        title=source.get('title', 'Unknown Title'),
                        content_preview=source.get('content_preview', ''),
                        episode_number=source.get('episode_number', 'Unknown'),
                        episode_id=source.get('episode_id', 'Unknown'),
                        guest_name=source.get('guest_name', 'Unknown Guest'),
                        url=source.get('url', '')
                    )
                    logger.info(f"Created episode fallback: {formatted_source.title}")
                
                formatted_sources.append(formatted_source)
        
        logger.info(f"Final formatted sources count: {len(formatted_sources)}")
        
        processing_time = time.time() - start_time
        
        return BM25ChatResponse(
            response=result.get('response', ''),
            sources=formatted_sources,
            episode_references=result.get('episode_references', []),
            search_method_used=result.get('search_method_used', 'unknown'),
            documents_retrieved=result.get('documents_retrieved', 0),
            bm25_stats=result.get('bm25_stats', {}),
            performance_stats=result.get('performance_stats', {}),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in BM25 chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-methods", response_model=SearchMethodComparisonResponse)
@limiter.limit("5/minute")
async def compare_search_methods(
    request: Request,
    comparison_request: SearchMethodComparisonRequest
):
    """
    Compare different search methods (BM25, semantic, hybrid) for the same query
    Useful for A/B testing and understanding search behavior
    """
    try:
        bm25_chain = get_bm25_chain()
        
        comparison = bm25_chain.compare_search_methods(
            query=comparison_request.query,
            k=comparison_request.k
        )
        
        # Format response
        methods = {}
        for method_name, method_data in comparison['methods'].items():
            if 'error' in method_data:
                methods[method_name] = SearchMethodResult(
                    documents_count=0,
                    content_referenced=[],
                    top_results=[],
                    error=method_data['error']
                )
            else:
                methods[method_name] = SearchMethodResult(
                    documents_count=method_data['documents_count'],
                    content_referenced=method_data['content_referenced'],
                    top_results=method_data['top_results']
                )
        
        return SearchMethodComparisonResponse(
            query=comparison['query'],
            methods=methods
        )
        
    except Exception as e:
        logger.error(f"Error comparing search methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=BM25SearchResponse)
@limiter.limit("15/minute")
async def bm25_search_episodes(
    request: Request,
    search_request: BM25SearchRequest
):
    """
    Search for episodes using BM25 hybrid search
    Returns episodes ranked by relevance with detailed scoring
    """
    start_time = time.time()
    
    try:
        bm25_chain = get_bm25_chain()
        
        episodes = bm25_chain.search_episodes(
            query=search_request.query,
            k=search_request.k,
            search_method=search_request.search_method
        )
        
        # Format content results (episodes and books)
        formatted_results = []
        for item in episodes:
            content_type = item.get('content_type', 'episode')
            
            if content_type == 'book':
                formatted_results.append(ContentSearchResult(
                    content_type='book',
                    content_id=item['content_id'],
                    book_title=item['book_title'],
                    author=item['author'],
                    chapter_number=item.get('chapter_number'),
                    chapter_title=item.get('chapter_title'),
                    title=item['title'],
                    chunks=item['chunks'],
                    max_score=item['max_score']
                ))
            else:
                formatted_results.append(ContentSearchResult(
                    content_type='episode',
                    episode_id=item['episode_id'],
                    episode_number=item['episode_number'],
                    title=item['title'],
                    url=item.get('url'),
                    chunks=item['chunks'],
                    max_score=item['max_score']
                ))
        
        processing_time = time.time() - start_time
        
        return BM25SearchResponse(
            query=search_request.query,
            search_method=search_request.search_method,
            results=formatted_results,
            total_found=len(formatted_results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in BM25 episode search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=BM25HealthResponse)
async def bm25_health_check():
    """
    Health check for BM25 RAG chain
    Returns detailed status of all components
    """
    try:
        bm25_chain = get_bm25_chain()
        health_data = bm25_chain.health_check()
        
        return BM25HealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Error in BM25 health check: {e}")
        return BM25HealthResponse(
            initialized=False,
            vectorstore_available=False,
            bm25_retriever_available=False,
            gaia_available=False,
            bm25_index_ready=False,
            reranker_available=False,
            performance_stats={},
            component_stats={"error": str(e)}
        )


@router.get("/performance", response_model=PerformanceComparisonResponse)
async def get_performance_stats():
    """
    Get performance statistics for BM25 RAG chain
    Useful for monitoring and optimization
    """
    try:
        bm25_chain = get_bm25_chain()
        stats = bm25_chain.get_performance_comparison()
        
        return PerformanceComparisonResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-chains", response_model=RAGChainComparisonResponse)
@limiter.limit("3/minute")
async def compare_rag_chains(
    request: Request,
    comparison_request: RAGChainComparisonRequest
):
    """
    Compare original RAG chain vs BM25 RAG chain for the same query
    Provides side-by-side analysis of both approaches
    """
    try:
        # Get both chains
        original_chain = get_rag_chain()
        bm25_chain = get_bm25_chain()
        
        # Test original RAG chain
        start_time = time.time()
        try:
            original_result = original_chain.chat(
                message=comparison_request.query,
                k=comparison_request.k
            )
            original_time = time.time() - start_time
            
            original_rag_result = RAGChainResult(
                response=original_result.get('response', ''),
                sources_count=len(original_result.get('sources', [])),
                content_referenced=original_result.get('episode_references', []),
                processing_time=original_time,
                method_used='original_hybrid',
                success=True
            )
        except Exception as e:
            original_rag_result = RAGChainResult(
                response='',
                sources_count=0,
                content_referenced=[],
                processing_time=0,
                method_used='original_hybrid',
                success=False,
                error=str(e)
            )
        
        # Test BM25 RAG chain
        start_time = time.time()
        try:
            bm25_result = bm25_chain.chat(
                message=comparison_request.query,
                search_method="hybrid",
                k=comparison_request.k
            )
            bm25_time = time.time() - start_time
            
            bm25_rag_result = RAGChainResult(
                response=bm25_result.get('response', ''),
                sources_count=len(bm25_result.get('sources', [])),
                content_referenced=bm25_result.get('episode_references', []),
                processing_time=bm25_time,
                method_used=bm25_result.get('search_method_used', 'bm25_hybrid'),
                success=True
            )
        except Exception as e:
            bm25_rag_result = RAGChainResult(
                response='',
                sources_count=0,
                content_referenced=[],
                processing_time=0,
                method_used='bm25_hybrid',
                success=False,
                error=str(e)
            )
        
        # Analysis
        analysis = {
            'performance_comparison': {
                'original_time': original_rag_result.processing_time,
                'bm25_time': bm25_rag_result.processing_time,
                'time_difference': bm25_rag_result.processing_time - original_rag_result.processing_time
            },
            'source_comparison': {
                'original_sources': original_rag_result.sources_count,
                'bm25_sources': bm25_rag_result.sources_count,
                'unique_original_content': list(set(original_rag_result.content_referenced) - set(bm25_rag_result.content_referenced)),
                'unique_bm25_content': list(set(bm25_rag_result.content_referenced) - set(original_rag_result.content_referenced)),
                'common_content': list(set(original_rag_result.content_referenced) & set(bm25_rag_result.content_referenced))
            }
        }
        
        # Recommendation
        recommendation = None
        if both_successful := (original_rag_result.success and bm25_rag_result.success):
            if len(bm25_rag_result.content_referenced) > len(original_rag_result.content_referenced):
                recommendation = "BM25 RAG found more diverse content sources"
            elif bm25_rag_result.processing_time < original_rag_result.processing_time:
                recommendation = "BM25 RAG was faster"
            elif len(analysis['source_comparison']['common_content']) > 0:
                recommendation = "Both chains found similar relevant content"
            else:
                recommendation = "Results differ significantly - manual review recommended"
        
        return RAGChainComparisonResponse(
            query=comparison_request.query,
            original_rag=original_rag_result,
            bm25_rag=bm25_rag_result,
            comparison_analysis=analysis,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Error comparing RAG chains: {e}")
        raise HTTPException(status_code=500, detail=str(e))