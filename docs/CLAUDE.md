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
```

## Architecture Overview

This is a **Dual RAG Chatbot** that allows users to chat with "Gaia" (the spirit of Earth) using knowledge from 172 YonEarth podcast episodes with two advanced search systems.

### Core Components

**API Layer** (`src/api/`):
- `main.py`: FastAPI application with endpoints for original RAG chat, search, recommendations
- `bm25_endpoints.py`: Advanced BM25 hybrid search endpoints with comparison features
- `models.py`: Pydantic models for original RAG API requests/responses
- `bm25_models.py`: Pydantic models for BM25 hybrid search API
- Rate limiting, CORS, health checks included

**Dual RAG System** (`src/rag/`):
- `chain.py`: Original RAG orchestration with hybrid keyword + semantic search
- `bm25_chain.py`: Advanced BM25 RAG with RRF, reranking, and query analysis
- `hybrid_retriever.py`: Original keyword frequency + semantic search combination
- `bm25_hybrid_retriever.py`: Advanced BM25 + semantic + cross-encoder reranking
- `keyword_indexer.py`: Builds keyword frequency index for episode lookup
- `vectorstore.py`: Pinecone vector database wrapper
- `pinecone_setup.py`: Pinecone index initialization and management

**Character System** (`src/character/`):
- `gaia.py`: Gaia character with memory, personality, citation handling, custom prompts
- `gaia_personalities.py`: Personality variants (warm_mother, wise_guide, earth_activist)

**Data Ingestion** (`src/ingestion/`):
- `episode_processor.py`: Processes YonEarth episode transcripts
- `chunker.py`: Text chunking with overlap for vectorization
- `process_episodes.py`: Main data processing pipeline

**Configuration** (`src/config/`):
- `settings.py`: Centralized settings using Pydantic Settings

**Web Interface** (`web/`):
- `index.html`: Beautiful responsive chat interface with personality selection
- `chat.js`: Advanced JavaScript with dual search modes, smart recommendations, custom prompts
- `styles.css`: Earth-themed CSS with comparison views and responsive design

### Dual Search Architecture

The system provides two complementary RAG approaches:

#### 🌿 Original RAG (Semantic + Keyword Hybrid)
1. **Keyword Indexer**: Builds frequency maps of important terms across episodes
2. **Semantic Search**: Vector similarity search using OpenAI embeddings  
3. **Hybrid Retriever**: Combines both approaches with weighted scoring
4. **Citation Accuracy**: Solves hallucination by finding episodes that actually contain search terms

#### 🔍 BM25 Advanced RAG (State-of-the-Art)
1. **BM25 Keyword Search**: Industry-standard keyword matching algorithm
2. **Semantic Vector Search**: OpenAI embedding similarity
3. **Reciprocal Rank Fusion (RRF)**: Mathematically optimal result combination
4. **Cross-Encoder Reranking**: MS-MARCO MiniLM final relevance scoring
5. **Query-Adaptive Strategy**: Automatically chooses optimal search method per query

### Smart Web Interface Features

**Personality System**:
- 3 predefined personalities (Nurturing Mother, Ancient Sage, Earth Guardian)
- Custom personality creation with editable system prompts
- localStorage persistence for custom prompts
- Template-based editing using existing personalities

**Search Method Selection**:
- 🌿 Original (Semantic Search): Meaning-based context understanding  
- 🔍 BM25 (Hybrid Search): Keyword + semantic + reranking
- ⚖️ Both (Comparison): Side-by-side results from both methods

**Smart Recommendations**:
- **Inline Citations**: Referenced episodes appear under each response
- **Dynamic Recommendations**: Bottom section evolves based on conversation context
- **Topic Tracking**: Automatically extracts conversation themes
- **Context Evolution**: "Based on our conversation about: permaculture, soil health..."
- **Related Suggestions**: "Try asking about: other episodes on composting"

### Key Files to Understand

**Core RAG Systems**:
- `src/rag/chain.py`: Original RAG pipeline entry point
- `src/rag/bm25_chain.py`: Advanced BM25 RAG pipeline
- `src/rag/hybrid_retriever.py`: Original hybrid search implementation
- `src/rag/bm25_hybrid_retriever.py`: Advanced BM25 + semantic + reranking

**API Endpoints**:
- `src/api/main.py`: Original RAG endpoints and middleware
- `src/api/bm25_endpoints.py`: BM25 endpoints with comparison features

**Character & Personality**:
- `src/character/gaia.py`: Response generation with personality and custom prompts
- `src/character/gaia_personalities.py`: Predefined personality prompts

**Web Interface**:
- `web/chat.js`: Advanced frontend with dual search, smart recommendations, custom prompts
- `web/index.html`: Responsive UI with personality selection and search method toggle

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
Edit `src/character/gaia_personalities.py` to add new personality prompts. The system supports both predefined and custom personalities.

### Modifying Search Behavior
**Original RAG**:
- Keyword search: Edit `src/rag/keyword_indexer.py`
- Semantic search: Edit `src/rag/vectorstore.py` 
- Hybrid combination: Edit `src/rag/hybrid_retriever.py`

**BM25 RAG**:
- BM25 parameters: Edit `src/rag/bm25_hybrid_retriever.py`
- RRF combination: Modify `reciprocal_rank_fusion()` method
- Reranking: Adjust cross-encoder model in `rerank_results()`

### API Endpoint Changes
**Original System**: Add endpoints in `src/api/main.py` and models in `src/api/models.py`

**BM25 System**: Add endpoints in `src/api/bm25_endpoints.py` and models in `src/api/bm25_models.py`

### Web Interface Updates
**Frontend Logic**: Edit `web/chat.js` for new features, search methods, or UI behavior

**Styling**: Update `web/styles.css` for visual changes

**HTML Structure**: Modify `web/index.html` for new UI components

### Testing Search Accuracy
Both RAG systems are designed to solve the citation hallucination problem. Test with queries like:
- "what is biochar?" → Should return Episodes 120, 122, 165
- "regenerative agriculture" → Should return relevant farming episodes
- "composting techniques" → Should return composting-focused episodes

## Deployment Notes

- **VPS**: Use `./deploy.sh` for complete Docker setup with nginx, SSL, Redis
- **Render**: Use `render.yaml` blueprint for cloud deployment  
- **Local**: Use `scripts/start_local.py` for development

## Technical Innovation

### Hybrid RAG Accuracy
Both systems solve the "citation hallucination" problem by ensuring retrieved episodes actually contain the searched content through:
- Keyword frequency indexing (Original)
- BM25 keyword matching (Advanced)
- Semantic similarity (Both)
- Intelligent combination algorithms

### Conversation Intelligence
The web interface provides advanced conversation features:
- **Topic Extraction**: Identifies themes from user messages and responses
- **Episode Tracking**: Remembers which episodes were discussed
- **Dynamic Context**: Recommendations evolve with conversation
- **Smart Suggestions**: Proactively suggests related topics to explore

### Personality System
- **Predefined Personalities**: 3 carefully crafted Earth-centered voices
- **Custom Prompts**: Users can create personalized Gaia personalities
- **Template Editing**: Use existing personalities as starting points
- **Persistent Storage**: Custom prompts saved in browser localStorage

The system handles 172 podcast episodes with ~1850 vectorized chunks and maintains high episode citation accuracy through both RAG approaches while providing an intelligent, conversation-aware user experience.