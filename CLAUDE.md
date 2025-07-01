# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Local Development
```bash
# Start development server
python scripts/start_local.py

# Or manual startup
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Run all tests
python tests/run_tests.py

# Or use pytest directly
pytest -v tests/ --tb=short --color=yes

# With coverage (if pytest-cov installed)
pytest --cov=src --cov-report=term-missing --cov-report=html:htmlcov tests/
```

### Docker Deployment
```bash
# Quick VPS deployment (includes nginx, SSL, Redis)
./deploy.sh

# Development with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

### Data Processing
```bash
# Process episode data (if needed)
python src/ingestion/process_episodes.py

# Test specific components
python scripts/test_api.py
python test_hybrid_search.py
```

## Architecture Overview

This is a **Hybrid RAG Chatbot** that allows users to chat with "Gaia" (the spirit of Earth) using knowledge from 172 YonEarth podcast episodes.

### Core Components

**API Layer** (`src/api/`):
- `main.py`: FastAPI application with endpoints for chat, search, recommendations
- `models.py`: Pydantic models for request/response validation
- Rate limiting, CORS, health checks included

**RAG System** (`src/rag/`):
- `hybrid_retriever.py`: Combines keyword frequency + semantic search
- `keyword_indexer.py`: Builds keyword frequency index for episode lookup
- `vectorstore.py`: Pinecone vector database wrapper
- `chain.py`: Main RAG orchestration that combines retrieval + generation

**Character System** (`src/character/`):
- `gaia.py`: Gaia character with memory, personality, citation handling
- `gaia_personalities.py`: Different personality variants (warm_mother, wise_elder, playful_spirit)

**Data Ingestion** (`src/ingestion/`):
- `episode_processor.py`: Processes YonEarth episode transcripts
- `chunker.py`: Text chunking with overlap for vectorization
- `process_episodes.py`: Main data processing pipeline

**Configuration** (`src/config/`):
- `settings.py`: Centralized settings using Pydantic Settings

### Hybrid Search Architecture

The system uses a novel **keyword frequency + semantic search** approach to solve the "citation hallucination" problem common in RAG systems:

1. **Keyword Indexer**: Builds frequency maps of important terms across episodes
2. **Semantic Search**: Vector similarity search using OpenAI embeddings  
3. **Hybrid Retriever**: Combines both approaches with weighted scoring
4. **Gaia Character**: Generates responses with proper episode citations

This ensures queries like "what is biochar?" correctly return Episodes 120, 122, 165 (which actually discuss biochar) rather than random episodes.

### Key Files to Understand

- `src/rag/chain.py`: Main entry point for the RAG pipeline
- `src/rag/hybrid_retriever.py`: Core hybrid search implementation
- `src/character/gaia.py`: Response generation with personality
- `src/api/main.py`: API endpoints and middleware
- `requirements.txt`: All Python dependencies

## Environment Setup

Required environment variables:
```bash
# API Keys (required)
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key-here

# Pinecone Configuration
PINECONE_INDEX_NAME=yonearth-episodes
PINECONE_ENVIRONMENT=gcp-starter

# Model Configuration
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GAIA_PERSONALITY_VARIANT=warm_mother
GAIA_TEMPERATURE=0.7

# Processing Settings
EPISODES_TO_PROCESS=172
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Common Development Tasks

### Adding New Personality Variants
Edit `src/character/gaia_personalities.py` to add new personality prompts.

### Modifying Search Behavior
- Keyword search: Edit `src/rag/keyword_indexer.py`
- Semantic search: Edit `src/rag/vectorstore.py` 
- Hybrid combination: Edit `src/rag/hybrid_retriever.py`

### API Endpoint Changes
Add new endpoints in `src/api/main.py` and corresponding models in `src/api/models.py`.

### Testing Search Accuracy
Use `test_hybrid_search.py` to test search results for specific queries and verify episode citations.

## Deployment Notes

- **VPS**: Use `./deploy.sh` for complete Docker setup with nginx, SSL, Redis
- **Render**: Use `render.yaml` blueprint for cloud deployment
- **Local**: Use `scripts/start_local.py` for development

The system is designed to handle 172 podcast episodes with ~1500 vectorized chunks and maintains episode citation accuracy through the hybrid search approach.