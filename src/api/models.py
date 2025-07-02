"""
Pydantic models for API requests and responses
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message to Gaia", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    personality: Optional[str] = Field(None, description="Gaia personality variant")
    custom_prompt: Optional[str] = Field(None, description="Custom system prompt for Gaia (when personality is 'custom')", max_length=5000)
    max_results: Optional[int] = Field(5, description="Maximum number of retrieved documents", ge=1, le=10)


class Citation(BaseModel):
    """Citation information"""
    episode_number: str
    title: str
    guest_name: str
    url: str
    relevance: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Gaia's response")
    personality: str = Field(..., description="Personality variant used")
    citations: List[Citation] = Field(default_factory=list, description="Episode citations")
    context_used: int = Field(..., description="Number of documents used for context")
    session_id: Optional[str] = Field(None, description="Session ID")
    retrieval_count: int = Field(0, description="Number of documents retrieved")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class EpisodeRecommendation(BaseModel):
    """Episode recommendation model"""
    episode_number: str
    title: str
    guest_name: str
    url: str
    relevance_score: float
    reason: str


class RecommendationsRequest(BaseModel):
    """Request model for episode recommendations"""
    query: str = Field(..., description="Query for recommendations", min_length=1, max_length=500)
    max_recommendations: Optional[int] = Field(3, description="Maximum recommendations", ge=1, le=10)


class RecommendationsResponse(BaseModel):
    """Response model for episode recommendations"""
    query: str
    recommendations: List[EpisodeRecommendation]
    total_found: int


class SearchRequest(BaseModel):
    """Request model for episode search"""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    max_results: Optional[int] = Field(10, description="Maximum results", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


class SearchResult(BaseModel):
    """Individual search result"""
    episode_number: str
    title: str
    guest_name: str
    url: str
    relevance_score: float
    content_preview: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for episode search"""
    query: str
    results: List[SearchResult]
    total_found: int
    filters_applied: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    rag_initialized: bool
    vectorstore_stats: Dict[str, Any]
    gaia_personality: Optional[str] = None


class ConversationMessage(BaseModel):
    """Individual conversation message"""
    role: str = Field(..., description="Role: 'user' or 'gaia'")
    content: str = Field(..., description="Message content")
    citations: Optional[List[Citation]] = Field(default_factory=list, description="Citations from response")


class ConversationRecommendationsRequest(BaseModel):
    """Request model for conversation-based recommendations"""
    conversation_history: List[ConversationMessage] = Field(..., description="Full conversation history")
    max_recommendations: Optional[int] = Field(4, description="Maximum recommendations", ge=1, le=10)
    session_id: Optional[str] = Field(None, description="Session ID for context")


class ConversationRecommendationsResponse(BaseModel):
    """Response model for conversation-based recommendations"""
    recommendations: List[Citation] = Field(..., description="Recommended content based on conversation")
    conversation_topics: List[str] = Field(default_factory=list, description="Extracted topics from conversation")
    total_found: int = Field(..., description="Total recommendations found")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None