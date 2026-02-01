"""
Pydantic models for BM25 RAG API endpoints
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class CostDetail(BaseModel):
    """Cost detail for a specific service"""
    service: str = Field(..., description="Service name (e.g., 'OpenAI LLM', 'ElevenLabs Voice')")
    model: str = Field(..., description="Model used")
    usage: str = Field(..., description="Usage description")
    cost: str = Field(..., description="Cost in dollars")


class CostBreakdown(BaseModel):
    """Cost breakdown for API usage"""
    summary: str = Field(..., description="Total cost summary")
    details: List[CostDetail] = Field(default_factory=list, description="Detailed cost breakdown")


class BM25ChatRequest(BaseModel):
    """Request model for BM25 chat endpoint"""
    message: str = Field(..., description="User's message/question")
    search_method: Literal["auto", "bm25", "semantic", "hybrid"] = Field(
        default="auto",
        description="Search method to use"
    )
    k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source citations")
    gaia_personality: Optional[str] = Field(
        default="warm_mother",
        description="Gaia personality variant"
    )
    custom_prompt: Optional[str] = Field(None, description="Custom system prompt for Gaia (when personality is 'custom')", max_length=5000)
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Response creativity level"
    )
    category_threshold: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Category matching threshold for semantic search (0.6=broad, 0.7=normal, 0.8=strict)"
    )
    enable_voice: bool = Field(False, description="Enable voice generation for the response")
    voice_id: Optional[str] = Field(None, description="ElevenLabs voice ID to use for speech generation")
    mentioned_episodes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Episodes mentioned in previous conversation turns (for follow-up context)"
    )


class BM25Source(BaseModel):
    """Source citation with BM25 scores"""
    content_type: Literal["episode", "book"] = "episode"
    # Episode fields
    episode_id: Optional[str] = None
    episode_number: Optional[str] = None
    guest_name: Optional[str] = None
    url: Optional[str] = None
    # Book fields
    book_id: Optional[str] = None
    book_title: Optional[str] = None
    author: Optional[str] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    publication_year: Optional[int] = None
    # Common fields
    title: str
    content_preview: str
    keyword_score: Optional[float] = None
    semantic_score: Optional[float] = None
    final_score: Optional[float] = None


class BM25ChatResponse(BaseModel):
    """Response model for BM25 chat endpoint"""
    response: str
    sources: List[BM25Source] = []
    episode_references: List[str] = []  # Now includes book references too
    search_method_used: str
    documents_retrieved: int
    bm25_stats: Dict[str, Any] = {}
    performance_stats: Dict[str, Any] = {}
    success: bool = True
    processing_time: Optional[float] = None
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data if voice was enabled")
    cost_breakdown: Optional[CostBreakdown] = Field(None, description="Cost breakdown for this response")


class SearchMethodComparisonRequest(BaseModel):
    """Request model for search method comparison"""
    query: str = Field(..., description="Query to test with different methods")
    k: int = Field(default=5, ge=1, le=20, description="Number of results per method")


class SearchMethodResult(BaseModel):
    """Result for a single search method"""
    documents_count: int
    content_referenced: List[str]  # Episodes and books
    top_results: List[Dict[str, Any]]
    error: Optional[str] = None


class SearchMethodComparisonResponse(BaseModel):
    """Response comparing different search methods"""
    query: str
    methods: Dict[str, SearchMethodResult]


class BM25HealthResponse(BaseModel):
    """Health check response for BM25 chain"""
    initialized: bool
    vectorstore_available: bool
    bm25_retriever_available: bool
    gaia_available: bool
    bm25_index_ready: bool
    reranker_available: bool
    performance_stats: Dict[str, Any]
    component_stats: Dict[str, Any]


class BM25SearchRequest(BaseModel):
    """Request model for BM25 episode search"""
    query: str = Field(..., description="Search query")
    k: int = Field(default=10, ge=1, le=50, description="Number of episodes to return")
    search_method: Literal["bm25", "semantic", "hybrid"] = Field(
        default="hybrid", 
        description="Search method to use"
    )


class ContentSearchResult(BaseModel):
    """Content search result with BM25 scoring (episodes or books)"""
    content_type: Literal["episode", "book"] = "episode"
    # Episode fields
    episode_id: Optional[str] = None
    episode_number: Optional[str] = None
    url: Optional[str] = None
    # Book fields
    content_id: Optional[str] = None  # For books: book_title_ch#
    book_title: Optional[str] = None
    author: Optional[str] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    # Common fields
    title: str
    chunks: List[Dict[str, Any]]
    max_score: float
    relevance_summary: Optional[str] = None


class BM25SearchResponse(BaseModel):
    """Response for BM25 content search"""
    query: str
    search_method: str
    results: List[ContentSearchResult]
    total_found: int
    processing_time: Optional[float] = None


class PerformanceComparisonResponse(BaseModel):
    """Performance statistics comparison"""
    total_queries: int
    bm25_queries: int
    semantic_queries: int
    hybrid_queries: int
    reranked_queries: int
    bm25_percentage: Optional[float] = None
    semantic_percentage: Optional[float] = None
    hybrid_percentage: Optional[float] = None
    reranking_percentage: Optional[float] = None
    bm25_available: bool
    total_documents: int
    reranker_available: bool
    current_keyword_weight: float
    current_semantic_weight: float


class RAGChainComparisonRequest(BaseModel):
    """Request to compare original vs BM25 RAG chains"""
    query: str = Field(..., description="Query to test with both chains")
    k: int = Field(default=5, ge=1, le=20, description="Number of results")
    include_detailed_analysis: bool = Field(
        default=False, 
        description="Include detailed performance analysis"
    )


class RAGChainResult(BaseModel):
    """Result from a single RAG chain"""
    response: str
    sources_count: int
    content_referenced: List[str]  # Episodes and books
    processing_time: float
    method_used: str
    success: bool
    error: Optional[str] = None


class RAGChainComparisonResponse(BaseModel):
    """Comparison between original and BM25 RAG chains"""
    query: str
    original_rag: RAGChainResult
    bm25_rag: RAGChainResult
    comparison_analysis: Dict[str, Any]
    recommendation: Optional[str] = None