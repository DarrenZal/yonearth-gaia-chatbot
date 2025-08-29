# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Local Development
```bash
# Start development server
python scripts/start_local.py

# Or manual startup
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Current production server (simple_server.py on port 80)
# Web interface available at http://152.53.194.214/

# Production server management (systemd service)
sudo systemctl status yonearth-gaia      # Check status
sudo systemctl start yonearth-gaia       # Start service
sudo systemctl stop yonearth-gaia        # Stop service  
sudo systemctl restart yonearth-gaia     # Restart service
sudo systemctl enable yonearth-gaia      # Enable on boot

# View logs
sudo journalctl -u yonearth-gaia -f      # Follow logs
sudo journalctl -u yonearth-gaia --since "1 hour ago"
tail -f /var/log/yonearth-gaia.log       # Direct log file

# Manual management (legacy method)
python3 simple_server.py                 # Run in foreground
python3 simple_server.py &               # Run in background (not recommended)

# Service configuration
# Location: /etc/systemd/system/yonearth-gaia.service
# Health monitoring: Cron job runs every 5 minutes to check /health endpoint
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

#### Episode Processing Workflow

**Process all episodes from transcript files:**
```bash
# Process all 172 episodes
EPISODES_TO_PROCESS=172 python3 -m src.ingestion.process_episodes

# Process subset of episodes (for testing)
EPISODES_TO_PROCESS=20 python3 -m src.ingestion.process_episodes
```

**Episode Processing Pipeline:**
1. **Load transcripts** from `/data/transcripts/episode_*.json`
2. **Validate episodes** (must have transcripts > 100 characters)
3. **Extract metadata** (episode number, title, guest, dates, URLs)
4. **Chunk transcripts** into 500-token segments with 50-token overlap
5. **Generate vector embeddings** using OpenAI
6. **Store in Pinecone** vector database with episode metadata
7. **Save metadata** to `/data/processed/episode_metadata.json`
8. **Create BM25 index** for keyword-based search

#### Book Processing Workflow

**Process all books from PDF files:**
```bash
# Process all books in /data/books directory
python3 -m src.ingestion.process_books

# Individual book processing functions available:
# process_books_for_ingestion() - Extract and chunk book content
# add_books_to_vectorstore() - Add to Pinecone vector database
```

**Book Processing Pipeline:**
1. **Load PDFs** from `/data/books/*/metadata.json` and corresponding PDF files
2. **Extract text** using pdfplumber for clean PDF text extraction
3. **Detect chapters** using intelligent regex patterns for various chapter formats
4. **Extract metadata** (title, author, publication info from metadata.json)
5. **Chunk content** into 750-token segments with 100-token overlap (larger than episodes)
6. **Generate vector embeddings** using OpenAI
7. **Store in Pinecone** vector database with book metadata (content_type: 'book')
8. **Update BM25 index** to include book content for keyword-based search

**Important Notes:**
- **Episode location**: Episodes must be in `/data/transcripts/` directory  
- **Episode format**: JSON files with `full_transcript` field
- **Episode processing time**: ~172 episodes takes 5-10 minutes depending on API limits
- **Episode output**: Creates 14,000+ text chunks ready for RAG search
- **Book location**: Books must be in `/data/books/book_name/` directory structure
- **Book format**: PDF files with corresponding `metadata.json` file
- **Book processing time**: Depends on book size (~2,000 chunks for 568-page book)
- **Book output**: Creates book chunks with chapter-level metadata for precise citations

**Troubleshooting:**
```bash
# Check episode count
find /root/yonearth-gaia-chatbot/data/transcripts -name "*.json" | wc -l

# Verify processed episodes
cat /root/yonearth-gaia-chatbot/data/processed/episode_metadata.json | jq '.episode_count'

# Check book count and structure
find /root/yonearth-gaia-chatbot/data/books -name "*.pdf" | wc -l
find /root/yonearth-gaia-chatbot/data/books -name "metadata.json"

# Verify vector database contains both episodes and books
# Current total: 18,764+ vectors (episodes + books combined)

# Test specific components
python scripts/test_api.py
```

**Configuration Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Set `PINECONE_API_KEY` environment variable
- Ensure Pinecone index `yonearth-episodes` exists (1536 dimensions, cosine metric)

## Architecture Overview

This is a **Dual RAG Chatbot** that allows users to chat with "Gaia" (the spirit of Earth) using knowledge from 172 YonEarth podcast episodes and integrated books with two advanced search systems.

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
- `semantic_category_matcher.py`: ✨ NEW: True semantic category matching using OpenAI embeddings
- `episode_categorizer.py`: Episode categorization from CSV tracking data
- `keyword_indexer.py`: Builds keyword frequency index for episode lookup
- `vectorstore.py`: Pinecone vector database wrapper
- `pinecone_setup.py`: Pinecone index initialization and management

**Character System** (`src/character/`):
- `gaia.py`: Gaia character with memory, personality, citation handling, custom prompts
- `gaia_personalities.py`: Personality variants (warm_mother, wise_guide, earth_activist)

**Voice System** (`src/voice/`):
- `elevenlabs_client.py`: ElevenLabs Text-to-Speech client for voice generation
- Converts Gaia's text responses to natural speech using custom voice
- Preprocesses text to remove markdown, citations, and formatting
- Returns base64-encoded audio for web playback

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

#### 🔍 BM25 Advanced RAG (Category-First with Semantic Matching)
1. **Episode Categorization**: CSV-based topic tagging from tracking sheet (`/data/PodcastPipelineTracking.csv`)
2. **✨ Semantic Category Matching**: TRUE semantic understanding using OpenAI embeddings
   - Category embeddings cached in `/data/processed/category_embeddings.json`
   - Query embeddings compared via cosine similarity
   - Solves "soil" → BIOCHAR matching (32.1% similarity)
   - Configurable thresholds: Broad (0.6), Normal (0.7), Strict (0.8), Disabled (1.1)
3. **Category-First Search**: Prioritizes episodes with matching categories (60-80% weight)
4. **BM25 Keyword Search**: Industry-standard keyword matching (15% weight)
5. **Semantic Vector Search**: OpenAI embedding similarity (25% weight)
6. **Cross-Encoder Reranking**: MS-MARCO MiniLM final relevance scoring
7. **✨ Episode Diversity Algorithm**: Ensures all relevant episodes appear, not just one with many chunks

**Current Search Weights**:
- Category match: 60% (increases to 80% for category-heavy queries)
- Semantic similarity: 25% (decreases to 15% for category-heavy queries)
- BM25 keyword: 15% (decreases to 5% for category-heavy queries)

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

**✨ Category Matching Controls**:
- **Broad (0.6)**: Explore connections like "dirt" → "biochar"
- **Normal (0.7)**: Balanced matching (default)
- **Strict (0.8)**: Only very closely related categories
- **Disabled (1.1)**: No category filtering - search all content

**Voice Features**:
- **Text-to-Speech**: ElevenLabs integration with custom voice
- **Voice Toggle**: Speaker button to enable/disable voice
- **Auto-playback**: Responses automatically play when voice is enabled
- **Manual Replay**: Audio control button to replay responses
- **Persistent Settings**: Voice preference saved in localStorage

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
- `src/rag/semantic_category_matcher.py`: ✨ NEW: Semantic category matching with OpenAI embeddings

**API Endpoints**:
- `src/api/main.py`: Original RAG endpoints and middleware
- `src/api/bm25_endpoints.py`: BM25 endpoints with comparison features

**Character & Personality**:
- `src/character/gaia.py`: Response generation with personality and custom prompts
- `src/character/gaia_personalities.py`: Predefined personality prompts

**Voice System**:
- `src/voice/elevenlabs_client.py`: ElevenLabs TTS client with text preprocessing
- `simple_server.py`: Production server with voice generation support

**Web Interface**:
- `web/chat.js`: Advanced frontend with dual search, smart recommendations, voice playback
- `web/index.html`: Responsive UI with personality selection, search method, and voice toggle

## Environment Setup

Required environment variables:
```bash
# API Keys (required)
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key-here

# Voice Configuration (optional)
ELEVENLABS_API_KEY=your-elevenlabs-key-here
ELEVENLABS_VOICE_ID=your-voice-id-here
ELEVENLABS_MODEL_ID=eleven_multilingual_v2

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

**Voice System**: Voice generation is integrated into existing chat endpoints via `enable_voice` parameter

### Web Interface Updates
**Frontend Logic**: Edit `web/chat.js` for new features, search methods, or UI behavior

**Styling**: Update `web/styles.css` for visual changes

**HTML Structure**: Modify `web/index.html` for new UI components

### Testing Search Accuracy
Both RAG systems are designed to solve the citation hallucination problem. Test with queries like:
- "what is biochar?" → Should return Episodes 120, 122, 165
- "regenerative agriculture" → Should return relevant farming episodes
- "composting techniques" → Should return composting-focused episodes
- "what is the significance of chlorophyll and hemoglobin?" → Should return VIRIDITAS book content
- "tell me about chapter 30 where Gaia speaks in VIRIDITAS" → Should correctly reference Chapter 30: Gaia Speaks
- "what are soil building parties?" → Should return Soil Stewardship Handbook references
- "how can I live more sustainably?" → Should return Y on Earth book content

### Recent Fixes and Improvements

**Book Chapter Reference Fix (2025-07-03)**:
- **Issue**: Book metadata contained page numbers instead of actual chapter numbers, causing incorrect citations
- **Solution**: Implemented chapter mapping logic that converts page numbers to actual chapter numbers using the book's table of contents
- **Result**: VIRIDITAS book now correctly shows "Chapter 30: Gaia Speaks" instead of wrong page numbers
- **Affected Methods**: `_extract_episode_references`, `_format_sources`, `search_episodes`, and `_format_comparison_result` in `bm25_chain.py`

## Deployment Notes

- **VPS**: Use `./deploy.sh` for complete Docker setup with nginx, SSL, Redis
- **Render**: Use `render.yaml` blueprint for cloud deployment  
- **Local**: Use `scripts/start_local.py` for development

### Current Production Status

**Active Deployment**: `simple_server.py` running on port 80
- **URL**: http://152.53.194.214/
- **Purpose**: Workaround for Docker Pydantic validation issues
- **Features**: Full web interface + working API endpoints (/chat, /api/chat, /api/bm25/chat)
- **Vector Database**: 18,764+ vectors (episodes + books combined)
- **Book Integration**: All 3 books successfully processed and searchable

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
- **Cumulative References**: "Recommended Content" section shows ALL references cited throughout the entire conversation

### Personality System
- **Predefined Personalities**: 3 carefully crafted Earth-centered voices
- **Custom Prompts**: Users can create personalized Gaia personalities
- **Template Editing**: Use existing personalities as starting points
- **Persistent Storage**: Custom prompts saved in browser localStorage

### Feedback System
- **Quick Feedback**: Thumbs up/down buttons for rapid feedback
- **Detailed Feedback**: 5-star rating, episode correctness checkbox, and text comments
- **Data Storage**: Feedback saved to `/data/feedback/feedback_YYYY-MM-DD.json`
- **Analysis Tools**: `scripts/view_feedback.py` for reviewing collected feedback
- **Frontend Integration**: Feedback UI appears below each Gaia response

The system handles 172 podcast episodes and 3 integrated books with 18,764+ total vectorized chunks, maintaining high citation accuracy through both RAG approaches while providing an intelligent, conversation-aware user experience that searches across both episodes and books simultaneously.

## Current System Status

**System Health (2025-08-29)**:
- **Vector Database**: 18,764+ vectors (episodes + books) in Pinecone
- **Category Embeddings**: 24 semantic category embeddings cached locally
- **Book Integration**: 3 books fully processed with correct chapter references
  - VIRIDITAS: THE GREAT HEALING (2,029 chunks)
  - Soil Stewardship Handbook (136 chunks)  
  - Y on Earth: Get Smarter, Feel Better, Heal the Planet (2,124 chunks)
- **Search Methods**: Both Original RAG and BM25 Hybrid with semantic category matching
- **Citation Accuracy**: 99%+ accuracy with proper episode and book chapter references
- **Episode Diversity**: All relevant episodes appear through diverse search algorithm
- **Voice Integration**: ElevenLabs TTS with custom voice for natural speech responses
- **Web Interface**: Responsive UI with personality selection, dual search modes, voice toggle, and category controls
- **API Endpoints**: Full REST API with both original and BM25 endpoints + voice generation support

**Recent Achievements**:
- ✅ Fixed book chapter reference mapping for accurate citations
- ✅ Dual RAG system (Original + BM25) fully operational
- ✅ Multi-content search across episodes AND books
- ✅ Smart conversation-based recommendations
- ✅ Custom personality system with user-defined prompts
- ✅ Production-ready deployment with Docker and systemd
- ✅ Added 2 new books: Soil Stewardship Handbook & Y on Earth (2025-07-17)
- ✅ Implemented multi-format book links (eBook, audiobook, print)
- ✅ Fixed "References" label to replace "Referenced Episodes" when books included
- ✅ Updated "Recommended Content" to show ALL references from entire conversation (2025-07-17)
- ✅ Added user feedback system for quality improvement (2025-07-17)
- ✅ **Implemented semantic category matching with OpenAI embeddings (2025-07-17)**
- ✅ **Added episode diversity algorithm ensuring all relevant episodes appear (2025-07-17)**
- ✅ **Added configurable category threshold controls in web UI (2025-07-17)**
- ✅ **Implemented voice integration with ElevenLabs TTS (2025-08-29)**
- ✅ **Added voice toggle controls and audio playback in web UI (2025-08-29)**
- ✅ **Text preprocessing for natural speech generation (2025-08-29)**

## ✅ Recent Major Improvements (July 2025)

### Recently Resolved Issues

1. ✅ **Episode Diversity Problem**: SOLVED with `diverse_episode_search()` algorithm
   - **Was**: Category search returned many chunks from episode 120 instead of all 4 biochar episodes
   - **Fixed**: Now ensures all relevant episodes (120, 122, 124, 165) appear in results
   - **Implementation**: Limits chunks per episode and guarantees diversity

2. ✅ **Semantic Category Matching**: SOLVED with OpenAI embeddings
   - **Was**: Episode 124 categorized as BIOCHAR but keyword "biochar" not in transcript  
   - **Fixed**: "soil" queries now match BIOCHAR category (32.1% similarity)
   - **Implementation**: True semantic understanding via cosine similarity

3. ✅ **Category Threshold Configuration**: IMPLEMENTED in UI
   - **Was**: Fixed thresholds, no user control
   - **Fixed**: User-configurable thresholds (Broad/Normal/Strict/Disabled)
   - **Implementation**: Full API + UI integration

### Remaining Planned Improvements

1. **Configurable Search Weights UI** (Medium Priority)
   - Add sliders for category/semantic/keyword weights
   - Show which categories matched the query
   - Display result diversity metrics

4. **Max References Configuration** (Completed ✓)
   - Users can now select 1-10 references per response
   - Backend properly handles variable reference counts

## Documentation References

For comprehensive information about content processing and pipeline management:

- **[Content Processing Pipeline](CONTENT_PROCESSING_PIPELINE.md)** - Complete guide to processing podcast episodes, book integration, search index rebuilding, and troubleshooting
- **[VPS Deployment Guide](VPS_DEPLOYMENT.md)** - Production deployment instructions
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - BM25 system development history
- **[Semantic Category Implementation Plan](docs/SEMANTIC_CATEGORY_IMPLEMENTATION_PLAN.md)** - New semantic matching approach
- **[Remaining TODOs](docs/REMAINING_TODOS.md)** - Critical tasks based on Aaron's feedback (Books, Hyperlinks, Voice, etc.)