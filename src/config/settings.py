"""
Configuration settings for YonEarth Gaia Chatbot
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  # Ignore extra fields in .env
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model for chat")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    
    # Alternative LLM (Optional)
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key for Claude")
    anthropic_model: Optional[str] = Field(default="claude-3-haiku-20240307", description="Anthropic model")
    
    # Pinecone Configuration
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_environment: str = Field(default="gcp-starter", description="Pinecone environment")
    pinecone_index_name: str = Field(default="yonearth-episodes", description="Pinecone index name")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=True, description="Enable auto-reload for development")
    
    # CORS Settings
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000,https://yonearth.org",
        description="Allowed CORS origins (comma-separated)"
    )
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated origins into a list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=10, description="Rate limit per minute per IP")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Gaia Character Settings
    gaia_personality_variant: str = Field(default="warm_mother", description="Gaia personality variant")
    gaia_temperature: float = Field(default=0.7, description="LLM temperature for Gaia responses")
    gaia_max_tokens: int = Field(default=1000, description="Max tokens for Gaia responses")
    
    # Episode Processing
    episodes_to_process: int = Field(default=20, description="Number of episodes to process for MVP")
    chunk_size: int = Field(default=500, description="Token size for text chunks")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    
    # Development
    debug: bool = Field(default=True, description="Debug mode")
    
    # ElevenLabs Voice Configuration
    elevenlabs_api_key: Optional[str] = Field(default=None, description="ElevenLabs API key for voice generation")
    elevenlabs_voice_id: Optional[str] = Field(default="YcVr5DmTjJ2cEVwNiuhU", description="ElevenLabs voice ID")
    elevenlabs_model_id: str = Field(default="eleven_multilingual_v2", description="ElevenLabs model ID")
    enable_voice_generation: bool = Field(default=False, description="Enable voice generation for responses")
    
    # Knowledge Graph Configuration
    enable_graph_retrieval: bool = Field(default=True, description="Enable GraphRAG retrieval alongside hybrid search")
    graph_max_edges: int = Field(default=50, description="Maximum edges to expand in graph retrieval")
    graph_retrieval_k: int = Field(default=20, description="Number of graph chunks to retrieve")
    graph_retrieval_max_hops: int = Field(default=2, description="Maximum graph hops to explore for multi-hop expansion")
    graph_hop_decay: float = Field(default=0.7, description="Per-hop decay factor when scoring multi-hop neighbors")
    graph_enable_semantic_entity_match: bool = Field(
        default=True,
        description="Use lightweight semantic/fuzzy matching for entity aliases in addition to lexical match",
    )
    graph_semantic_entity_jaccard_threshold: float = Field(
        default=0.7,
        description="Minimum token-level Jaccard similarity to treat a multi-word alias as matched",
    )
    graph_enable_cluster_retrieval: bool = Field(
        default=False,
        description="Enable semantic cluster/topic retrieval using cluster_index.json when available",
    )
    graph_enable_adaptive_weight: bool = Field(
        default=True,
        description="Adapt graph fusion weight based on entity/cluster signal in the query",
    )
    graph_default_weight: float = Field(
        default=1.2,
        description="Default graph weight used in hybrid fusion when adaptive weighting is disabled",
    )
    graph_min_weight: float = Field(
        default=0.2,
        description="Minimum graph weight when adaptive weighting is enabled",
    )
    graph_max_weight: float = Field(
        default=2.0,
        description="Maximum graph weight when adaptive weighting is enabled",
    )
    graph_weight_entity_cap: int = Field(
        default=5,
        description="Number of matched entities at which graph weight saturates in adaptive mode",
    )
    graph_mode: str = Field(
        default="evidence_only",
        description="Graph usage mode: 'off' (disable), 'evidence_only' (evidence/triples only), or 'full' (recall + evidence).",
    )

    # Evidence Capping Configuration
    graph_evidence_max_total: int = Field(default=10, description="Maximum total evidence triples to include")
    graph_evidence_min_confidence: float = Field(default=0.6, description="Minimum p_true for evidence inclusion")
    graph_evidence_per_predicate: int = Field(default=2, description="Maximum evidence examples per predicate type")

    # Paths
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path"""
        return self.project_root / "data"
    
    @property
    def episodes_dir(self) -> Path:
        """Get episodes JSON directory"""
        return self.data_dir / "transcripts"
    
    @property
    def processed_dir(self) -> Path:
        """Get processed data directory"""
        return self.project_root / "data" / "processed"
    
    def get_episode_files(self, limit: Optional[int] = None) -> List[Path]:
        """Get list of episode JSON files"""
        episode_files = sorted(self.episodes_dir.glob("episode_*.json"))
        if limit:
            return episode_files[:limit]
        return episode_files


# Create singleton instance
settings = Settings()
