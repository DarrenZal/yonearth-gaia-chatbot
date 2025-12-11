"""
Pydantic models for GraphRAG Chat API endpoints.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class CommunityContext(BaseModel):
    """Context from a matched community cluster"""
    id: str = Field(..., description="Cluster ID (e.g., 'level_0_42')")
    name: str = Field(..., description="Community name")
    title: str = Field("", description="Human-readable title")
    summary: str = Field(..., description="Community summary text")
    level: int = Field(..., description="Hierarchy level (1=fine-grained, 2=themes)")
    entity_count: int = Field(..., description="Number of entities in community")
    relevance_score: float = Field(..., description="Relevance score (0-1)")


class EntityContext(BaseModel):
    """Context from a matched entity"""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (PERSON, ORGANIZATION, CONCEPT, etc.)")
    description: str = Field(..., description="Entity description")
    sources: List[str] = Field(default_factory=list, description="Source episodes/books")
    mention_count: int = Field(1, description="Number of mentions in corpus")
    relevance_score: float = Field(1.0, description="Relevance score (0-1)")


class RelationshipContext(BaseModel):
    """Context from a relationship between entities"""
    source: str = Field(..., description="Source entity name")
    predicate: str = Field(..., description="Relationship type (e.g., FOUNDED, WORKS_FOR)")
    target: str = Field(..., description="Target entity name")
    weight: float = Field(1.0, description="Relationship weight/strength")


class SourceCitation(BaseModel):
    """Source citation formatted for frontend display"""
    content_type: str = Field("episode", description="Content type: 'episode' or 'book'")
    episode_number: Optional[str] = Field(None, description="Episode number or 'Book: Title' for books")
    title: str = Field(..., description="Episode or chapter title")
    guest_name: Optional[str] = Field(None, description="Guest name or author")
    url: Optional[str] = Field(None, description="Primary URL (episode or ebook)")
    ebook_url: Optional[str] = Field(None, description="eBook URL for books")
    audiobook_url: Optional[str] = Field(None, description="Audiobook URL for books")
    print_url: Optional[str] = Field(None, description="Print book URL")


class GraphRAGChatRequest(BaseModel):
    """Request model for GraphRAG chat endpoint"""
    message: str = Field(..., description="User's message/question")
    search_mode: Literal["global", "local", "drift", "auto"] = Field(
        default="drift",
        description="Search mode: global (community summaries), local (entities), drift (both), auto (detect)"
    )
    community_level: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Community level to search (1=573 fine-grained, 2=73 themes)"
    )
    k_communities: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of communities to retrieve"
    )
    k_entities: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of entities to retrieve"
    )
    k_chunks: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of content chunks to retrieve for grounding"
    )
    personality: str = Field(
        default="warm_mother",
        description="Gaia personality variant"
    )
    custom_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt for Gaia",
        max_length=5000
    )
    # Voice parameters
    enable_voice: bool = Field(
        default=False,
        description="Enable text-to-speech voice generation for response"
    )
    voice_id: Optional[str] = Field(
        None,
        description="Voice ID to use for TTS (e.g., 'piper-kristin')"
    )


class GraphRAGChatResponse(BaseModel):
    """Response model for GraphRAG chat endpoint"""
    response: str = Field(..., description="Gaia's response")
    search_mode: str = Field(..., description="Search mode used")
    communities_used: List[CommunityContext] = Field(
        default_factory=list,
        description="Communities that provided context"
    )
    entities_matched: List[EntityContext] = Field(
        default_factory=list,
        description="Entities extracted from query"
    )
    relationships: List[RelationshipContext] = Field(
        default_factory=list,
        description="Relationships between matched entities"
    )
    source_episodes: List[str] = Field(
        default_factory=list,
        description="Episode sources referenced"
    )
    source_books: List[str] = Field(
        default_factory=list,
        description="Book sources referenced"
    )
    # Frontend-compatible sources format
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Formatted source citations for frontend display"
    )
    processing_time: float = Field(..., description="Response time in seconds")
    success: bool = Field(True, description="Whether the request succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    # Voice response
    audio_data: Optional[str] = Field(
        None,
        description="Base64-encoded audio data for TTS response"
    )


class GraphRAGCompareRequest(BaseModel):
    """Request model for comparing GraphRAG with BM25"""
    message: str = Field(..., description="User's message/question")
    include_bm25: bool = Field(True, description="Include BM25 response")
    include_graphrag: bool = Field(True, description="Include GraphRAG response")
    graphrag_mode: Literal["global", "local", "drift", "auto"] = Field(
        default="drift",
        description="GraphRAG search mode"
    )
    community_level: int = Field(default=1, ge=1, le=2)
    k_communities: int = Field(default=5, ge=1, le=20)
    k_entities: int = Field(default=10, ge=1, le=30)
    personality: str = Field(default="warm_mother")


class ComparisonResult(BaseModel):
    """Single system's response in a comparison"""
    system: str = Field(..., description="System name (bm25, graphrag)")
    response: str = Field(..., description="Response text")
    processing_time: float = Field(..., description="Processing time in seconds")
    success: bool = Field(True)
    error: Optional[str] = None
    # Metadata varies by system
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphRAGCompareResponse(BaseModel):
    """Response model for comparison endpoint"""
    query: str = Field(..., description="Original query")
    bm25_result: Optional[ComparisonResult] = Field(
        None,
        description="BM25 system response"
    )
    graphrag_result: Optional[ComparisonResult] = Field(
        None,
        description="GraphRAG system response"
    )
    comparison_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comparison metrics between systems"
    )


class GraphRAGHealthResponse(BaseModel):
    """Health check response for GraphRAG chain"""
    initialized: bool
    community_search_ready: bool
    local_search_ready: bool
    vectorstore_ready: bool
    gaia_ready: bool
    stats: Dict[str, Any] = Field(default_factory=dict)


class CommunityListResponse(BaseModel):
    """Response listing available communities"""
    level: int
    count: int
    communities: List[Dict[str, Any]]


class EntitySearchRequest(BaseModel):
    """Request to search for entities"""
    query: str = Field(..., description="Text to search for entities")
    k: int = Field(default=10, ge=1, le=50)


class EntitySearchResponse(BaseModel):
    """Response from entity search"""
    query: str
    entities_found: List[EntityContext]
    total_matches: int
