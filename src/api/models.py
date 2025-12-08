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
    model: Optional[str] = Field(None, description="OpenAI model to use for response generation")
    enable_voice: bool = Field(False, description="Enable voice generation for the response")
    voice_id: Optional[str] = Field(None, description="ElevenLabs voice ID to use for speech generation")


class Citation(BaseModel):
    """Citation information"""
    episode_number: str
    title: str
    guest_name: str
    url: str
    relevance: str


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


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Gaia's response")
    personality: str = Field(..., description="Personality variant used")
    citations: List[Citation] = Field(default_factory=list, description="Episode citations")
    context_used: int = Field(..., description="Number of documents used for context")
    session_id: Optional[str] = Field(None, description="Session ID")
    retrieval_count: int = Field(0, description="Number of documents retrieved")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    model_used: Optional[str] = Field(None, description="OpenAI model used for response generation")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data if voice was enabled")
    cost_breakdown: Optional[CostBreakdown] = Field(None, description="Cost breakdown for this response")


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


class ConversationCitation(BaseModel):
    """Loose citation model for conversation history - accepts partial data from frontend"""
    episode_number: Optional[str] = None
    title: Optional[str] = None
    guest_name: Optional[str] = None
    url: Optional[str] = None
    relevance: Optional[str] = None
    content_type: Optional[str] = None
    book_title: Optional[str] = None
    author: Optional[str] = None
    chapter_number: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields from frontend


class ConversationMessage(BaseModel):
    """Individual conversation message"""
    role: str = Field(..., description="Role: 'user' or 'gaia'")
    content: str = Field(..., description="Message content")
    citations: Optional[List[ConversationCitation]] = Field(default_factory=list, description="Citations from response")


class ConversationRecommendationsRequest(BaseModel):
    """Request model for conversation-based recommendations"""
    conversation_history: List[ConversationMessage] = Field(..., description="Full conversation history")
    max_recommendations: Optional[int] = Field(4, description="Maximum recommendations", ge=1, le=10)
    session_id: Optional[str] = Field(None, description="Session ID for context")


class ConversationRecommendationsResponse(BaseModel):
    """Response model for conversation-based recommendations"""
    recommendations: List[ConversationCitation] = Field(default_factory=list, description="Recommended content based on conversation")
    conversation_topics: List[str] = Field(default_factory=list, description="Extracted topics from conversation")
    total_found: int = Field(..., description="Total recommendations found")


class ModelComparisonResponse(BaseModel):
    """Response model for model comparison endpoint"""
    comparison: bool = Field(True, description="Indicates this is a comparison response")
    models: Dict[str, ChatResponse] = Field(..., description="Responses from different models")
    processing_time: float = Field(..., description="Total processing time for all models")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    messageId: str = Field(..., description="Unique message ID")
    timestamp: str = Field(..., description="ISO timestamp of feedback")
    type: str = Field(..., description="Feedback type: helpful, not-helpful, detailed")
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Gaia's response")
    citations: List[Citation] = Field(default_factory=list, description="Citations included in response")
    sessionId: str = Field(..., description="Session ID")
    personality: str = Field(..., description="Selected personality")
    ragType: str = Field(..., description="RAG type used")
    modelType: str = Field(..., description="Model type used")
    relevanceRating: Optional[int] = Field(None, description="Relevance rating 1-5", ge=1, le=5)
    episodesCorrect: Optional[bool] = Field(None, description="Were the right episodes included")
    detailedFeedback: Optional[str] = Field(None, description="Detailed feedback text")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool = Field(..., description="Whether feedback was saved successfully")
    message: str = Field(..., description="Response message")


class VoiceGenerationRequest(BaseModel):
    """Request model for voice generation endpoint"""
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=5000)
    voice_settings: Optional[Dict[str, float]] = Field(None, description="Voice settings (stability, similarity_boost)")
    output_format: str = Field("mp3_44100_128", description="Audio output format")


class VoiceGenerationResponse(BaseModel):
    """Response model for voice generation endpoint"""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    text_length: int = Field(..., description="Length of text that was converted")