# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📂 Repository Structure

**For complete file-by-file documentation**, see **[REPO_STRUCTURE.md](REPO_STRUCTURE.md)**

This comprehensive guide describes every file in the repository, what it does, and how it fits into the system.

### Quick Navigation

- **Main Chat Flow**: `web/index.html` → `web/chat.js` → `src/api/bm25_endpoints.py` → `src/rag/bm25_chain.py` → `src/character/gaia.py`
- **Search System**: `src/rag/bm25_hybrid_retriever.py` + `src/rag/semantic_category_matcher.py` + `src/rag/vectorstore.py`
- **Voice Integration**: `src/voice/elevenlabs_client.py` + `src/api/voice_endpoints.py`
- **Data Processing**: `src/ingestion/` (episode_processor, book_processor, chunker)
- **Configuration**: `src/config/settings.py` (centralized settings from .env)
- **Documentation**: `docs/` (setup, features, deployment guides)
- **Active Scripts**: `scripts/` (6 utility scripts)
- **Archived Scripts**: `scripts/archive/` (43 historical scripts organized by category)

## Development Commands

### Local Development
```bash
# Start development server
python scripts/start_local.py

# Or manual startup
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment

**⚠️ CRITICAL: There are TWO production sites with DIFFERENT deployment paths!**

#### Site 1: earthdo.me (Docker-based)

The earthdo.me site is served by Docker containers:
- **yonearth-nginx** (nginx:alpine) - Serves web files on ports 80/443
- **yonearth-gaia-chatbot** - FastAPI application on port 8000
- **yonearth-redis** - Redis cache on port 6379

**earthdo.me Directory Structure:**
```bash
# Development (edit files here)
/home/claudeuser/yonearth-gaia-chatbot/

# Docker Production Web Files (what Docker nginx serves for earthdo.me)
/opt/yonearth-chatbot/web/          # Static files (HTML, JS, CSS, data/)
/opt/yonearth-chatbot/nginx.conf    # Docker nginx configuration
/opt/yonearth-chatbot/ssl/          # SSL certificates
```

#### Site 2: gaiaai.xyz (System nginx)

**⚠️ IMPORTANT: gaiaai.xyz uses DIFFERENT paths than earthdo.me!**

The gaiaai.xyz site is served by system nginx (not Docker):
- Root: `/var/www/symbiocenelabs/`
- GraphRAG 3D viewer: `/var/www/symbiocenelabs/YonEarth/graph/`
- GraphRAG data files: `/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/`

**gaiaai.xyz Directory Structure:**
```bash
# GraphRAG 3D Viewer files (HTML, JS)
/var/www/symbiocenelabs/YonEarth/graph/
  ├── index.html                          # LIMITED VIEW (Semantic, Structural, Voronoi only)
  ├── GraphRAG3D_EmbeddingView.html       # FULL VIEW (all visualization modes)
  └── GraphRAG3D_EmbeddingView.js         # Main viewer JavaScript

# GraphRAG data files (JSON hierarchy, layouts)
/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/

# ⚠️ IMPORTANT: community_id_mapping.json exists in TWO locations!
# The JS code tries BOTH paths in order - update BOTH when changing cluster titles:
/var/www/symbiocenelabs/YonEarth/graph/data/community_id_mapping.json  # PRIMARY (loaded first)
/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/community_id_mapping.json  # FALLBACK
```

**GraphRAG Viewer Pages:**
- **https://gaiaai.xyz/YonEarth/graph/** - Limited view with 3 core visualization modes (recommended for most users)
- **https://gaiaai.xyz/YonEarth/graph/GraphRAG3D_EmbeddingView.html** - Full view with all experimental modes

**Deploying GraphRAG changes to gaiaai.xyz:**

**⚠️ RECOMMENDED: Use the deployment script after regenerating clusters:**
```bash
# This script handles everything: deploys data, regenerates community_id_mapping.json,
# updates cache buster, and reloads nginx
./scripts/deploy_graphrag.sh
```

**Manual deployment (if needed):**
```bash
# Deploy GraphRAG viewer files
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/graph/index.html /var/www/symbiocenelabs/YonEarth/graph/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/graph/GraphRAG3D_EmbeddingView.html /var/www/symbiocenelabs/YonEarth/graph/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/graph/GraphRAG3D_EmbeddingView.js /var/www/symbiocenelabs/YonEarth/graph/

# Deploy GraphRAG data files
sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphsage_layout.json /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/

# ⚠️ CRITICAL: Regenerate community_id_mapping.json from the new hierarchy!
# The voronoi view uses this file for cluster titles. If not regenerated,
# you'll see OLD cluster names in the UI.
# Run: ./scripts/deploy_graphrag.sh (which does this automatically)

# Reload nginx to clear caches
sudo systemctl reload nginx
```

#### Legacy/Backup Paths (NOT actively served)
```bash
/var/www/yonearth/                  # Old system nginx location
/var/www/yonearth-migrated/         # earthdo.me migrated files (served by system nginx on earthdo.me)
```

#### Deploying Changes to Production

**1. Frontend changes (HTML/JS/CSS):**
```bash
# Copy web files to Docker mount
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/*.html /opt/yonearth-chatbot/web/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/*.js /opt/yonearth-chatbot/web/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/*.css /opt/yonearth-chatbot/web/

# Restart nginx container
sudo docker restart yonearth-nginx
```

**2. Backend API changes (Python):**
```bash
# Docker containers run from /root/yonearth-gaia-chatbot/
sudo cp /home/claudeuser/yonearth-gaia-chatbot/src/api/*.py /root/yonearth-gaia-chatbot/src/api/

# Restart API container
sudo docker restart yonearth-gaia-chatbot
```

**3. Adding new web content (wiki, handbook, data files):**
```bash
# IMPORTANT: Docker nginx only serves files from /opt/yonearth-chatbot/web/
# New directories MUST be copied here to be accessible

# Example: Adding a new wiki or data directory
sudo cp -r /var/www/yonearth/new-content /opt/yonearth-chatbot/web/

# If special routing needed, edit nginx config:
sudo nano /opt/yonearth-chatbot/nginx.conf

# Test and restart nginx
sudo docker exec yonearth-nginx nginx -t
sudo docker restart yonearth-nginx
```

**4. Full deployment (backend + frontend):**
```bash
# Backend
sudo cp -r /home/claudeuser/yonearth-gaia-chatbot/src/* /root/yonearth-gaia-chatbot/src/

# Frontend
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/* /opt/yonearth-chatbot/web/

# Restart both containers
sudo docker restart yonearth-gaia-chatbot yonearth-nginx
```

#### Production Container Management
```bash
# View running containers
sudo docker ps

# Check logs
sudo docker logs yonearth-nginx -f
sudo docker logs yonearth-gaia-chatbot -f

# Restart containers
sudo docker restart yonearth-nginx
sudo docker restart yonearth-gaia-chatbot

# Stop/start all services
cd /root/yonearth-gaia-chatbot
sudo docker-compose down
sudo docker-compose up -d
```

#### ⚠️ Browser Cache-Busting
When updating JS/CSS files, browsers may cache old versions. To force updates:
1. Add version query parameter to script/link tags in HTML:
   ```html
   <link rel="stylesheet" href="styles.css?v=2">
   <script src="app.js?v=2"></script>
   ```
2. Increment version number (v=2 → v=3) each time you update JS/CSS
3. This forces browsers to load the new version instead of using cached files

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

#### ✅ Re-Transcription with Precise Timestamps (COMPLETE)

**Status: All 172 episodes now have word-level timestamps!**

See **[docs/TRANSCRIPTION_SETUP.md](docs/TRANSCRIPTION_SETUP.md)** for complete setup instructions.

**Completed October 2025:**
- ✅ **172/172 episodes** transcribed with word-level timestamps
- ✅ **14 episodes** transcribed from YouTube (broken/missing audio URLs)
  - Episodes: 0, 16, 48, 53, 58, 62, 63, 73, 75, 101, 105, 165, 171, 172
- ✅ **158 episodes** transcribed from original audio
- ✅ Only episode #26 missing (doesn't exist in series)

**What This Provides:**
- **Exact timestamps** for every segment (no estimation)
- **Word-level timestamps** for ultra-precise navigation
- **Improved 3D map** - clicking nodes jumps to exact moments in audio
- **100% coverage** of all publishable episodes

**Implementation Used:**
- Lightweight Whisper (base model) - timestamps only, lower memory
- YouTube fallback for episodes with broken audio URLs
- ~3-4 minutes per episode processing time

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

#### Knowledge Graph Extraction Workflow

**Extract knowledge graph from episode transcripts:**
```bash
# Extract entities and relationships from all episodes
python3 scripts/extract_knowledge_graph_episodes.py

# Extract from specific episodes
python3 scripts/extract_remaining_7_episodes.py

# Build unified knowledge graph from extractions
python3 scripts/build_unified_knowledge_graph.py
```

**Knowledge Graph Extraction Pipeline:**
1. **Load transcripts** from `/data/transcripts/episode_*.json`
2. **Chunk transcripts** into 800-token segments with 100-token overlap
3. **Extract entities** using OpenAI with structured outputs (gpt-4o-mini)
   - Entity types: PERSON, ORGANIZATION, CONCEPT, PLACE, PRACTICE, PRODUCT, etc.
   - Pydantic schema validation ensures 100% valid JSON
4. **Extract relationships** between entities using structured outputs
   - Relationship types: FOUNDED, WORKS_FOR, PRACTICES, ADVOCATES_FOR, etc.
   - Context-aware relationship identification
5. **Store extractions** in `/data/knowledge_graph/entities/` and `/relationships/`
6. **Build unified graph** combining all episodes into single knowledge graph

**Extraction Performance:**
- **Model**: gpt-4o-mini (fast, cost-effective)
- **Rate limiting**: 0.05s delay between API calls (1,200 requests/min)
- **Speed**: ~30 seconds - 1 minute per episode
- **Output**: JSON files with entities, relationships, and metadata
- **Structured Outputs**: Guaranteed valid JSON using Pydantic schemas (no parsing errors)

**Knowledge Graph Output:**
- **Entity files**: `/data/knowledge_graph/entities/episode_*_extraction.json`
- **Relationship files**: `/data/knowledge_graph/relationships/episode_*_extraction.json`
- **Unified graph**: `/data/knowledge_graph/graph/unified_knowledge_graph.json`
- **Total episodes**: 172 (0-172, excluding #26)

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

**Deployment** (`/etc/systemd/system/`):
- `yonearth-api.service`: Systemd service for production uvicorn management
- Nginx configuration: `/etc/nginx/sites-enabled/yonearth`

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
- Voice integration available via `/api/chat` endpoint with `enable_voice` parameter

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

## Podcast Episode Visualization Maps

The system provides multiple interactive 2D visualizations of podcast episode embeddings, each using different dimensionality reduction algorithms. All maps use **6000 text chunks from 170 episodes** (standardized for fair comparison).

### Available Visualizations

#### 1. **t-SNE Map** (`/PodcastMap.html`)
- **Algorithm**: t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Parameters**:
  - `perplexity=30`: Balance between local and global structure
  - `n_iter=1000`: Number of optimization iterations
  - `n_clusters=9`: K-means clustering for topic groups
- **Script**: `scripts/archive/visualization/generate_map_semantic_topics.py`
- **Best For**: Discovering local cluster structures and fine-grained topic boundaries
- **Output**: `/data/processed/podcast_map_data.json`

#### 2. **UMAP Map** (`/PodcastMapUMAP.html`) ✨ **Interactive Parameters**
- **Algorithm**: UMAP (Uniform Manifold Approximation and Projection)
- **Default Parameters**:
  - `n_points=6000`: Number of text chunks to visualize
  - `n_neighbors=15`: Controls local vs. global structure (5-200)
  - `min_dist=0.1`: Minimum distance between points (0.0-0.99)
  - `n_epochs=500`: Optimization iterations
  - `n_clusters=9`: K-means clustering for topics
- **Interactive Controls**: Users can adjust `n_points`, `min_dist`, and `n_neighbors` in the UI
  - Higher `n_neighbors` (50-100): Pulls clusters closer together globally
  - Higher `min_dist` (0.3-0.5): Spreads points more evenly
  - Fewer points (3000): Cleaner aesthetics with tighter clusters
- **Script**: `scripts/archive/visualization/generate_map_umap_topics.py`
- **API Endpoints**:
  - `POST /api/regenerate_umap`: Trigger UMAP generation with custom parameters
  - `GET /api/umap_generation_status`: Poll generation progress
- **Best For**: Preserving global structure while maintaining local relationships
- **Output**: `/data/processed/podcast_map_umap_data.json`
- **Features**: Real-time progress monitoring, automatic page refresh on completion

#### 3. **Hierarchical Map** (`/PodcastMapHierarchical.html`)
- **Algorithm**: UMAP with hierarchical clustering
- **Parameters**:
  - `n_neighbors=15`
  - `min_dist=0.1`
  - `n_clusters_level1=3`: Top-level categories
  - `n_clusters_level2=9`: Mid-level topics
  - `n_clusters_level3=27`: Fine-grained subtopics
- **Script**: `scripts/archive/visualization/generate_map_hierarchical.py`
- **Best For**: Exploring topic hierarchies and nested relationships
- **Output**: `/data/processed/podcast_map_hierarchical_data.json`

#### 4. **Nomic Atlas Map** (`/PodcastMapNomic.html`)
- **Algorithm**: Nomic's proprietary UMAP implementation with automatic topic modeling
- **Parameters**:
  - `n_points=6000`: Standardized dataset size
  - Automatic hierarchical topic detection
- **Script**: `scripts/archive/visualization/upload_to_nomic_atlas.py`
- **Best For**: Cloud-based visualization with automatic topic labeling
- **Output**: `/data/processed/nomic_projections.json`
- **Features**: Hosted on Nomic Atlas cloud, automatic topic hierarchies

### UMAP Parameter Effects

**`n_neighbors`** (Number of Neighbors):
- **5-15**: Emphasizes local structure, creates tight small clusters
- **50-100**: Emphasizes global structure, pulls clusters closer together
- **Effect**: Higher values reduce gaps between distant clusters (e.g., "Accounting" cluster gap)

**`min_dist`** (Minimum Distance):
- **0.0-0.1**: Points cluster very tightly, clumped appearance
- **0.3-0.5**: Points spread more evenly, looser clusters
- **Effect**: Higher values create more aesthetically balanced layouts

**`n_points`** (Number of Data Points):
- **3000**: Faster generation (~2min), cleaner aesthetics, fewer outliers
- **6000**: Better content coverage (~3min), more comprehensive but may have outlier clusters
- **Effect**: Fewer points often produce more visually pleasing layouts

### Visualization Generation Workflow

1. **Fetch vectors** from Pinecone with embeddings
2. **Apply dimensionality reduction** (t-SNE, UMAP, or Hierarchical UMAP)
3. **Cluster** using K-means or hierarchical clustering
4. **Label clusters** using GPT-4 based on representative chunks
5. **Save** to `/data/processed/` as JSON
6. **Serve** via API endpoints for web visualization

### Common Visualization Tasks

**Regenerate UMAP with custom parameters**:
```bash
# Via environment variables
MAX_VECTORS=3000 UMAP_MIN_DIST=0.4 UMAP_N_NEIGHBORS=75 \
python3 scripts/archive/visualization/generate_map_umap_topics.py
```

**Upload new dataset to Nomic Atlas**:
```bash
python3 scripts/archive/visualization/upload_to_nomic_atlas.py
```

**Generate hierarchical visualization**:
```bash
python3 scripts/archive/visualization/generate_map_hierarchical.py
```

### WebGL Context Management

The 3D visualizations use Three.js with WebGL. Chrome limits active WebGL contexts (~16 across all tabs), which can cause "WebGL Not Available" errors after heavy usage or repeated page refreshes.

**Current solution**: The viewer includes a `dispose()` method that properly releases WebGL resources on page unload, helping Chrome recycle contexts faster.

**Future consideration**: WebGPU migration could provide significant benefits:
- Now supported in all major browsers (Chrome, Firefox, Safari as of 2025)
- Up to 10x faster rendering with Render Bundles
- GPU-accelerated compute for physics simulations
- Modern shader language (WGSL)
- Would require switching from `WebGLRenderer` to `WebGPURenderer` in Three.js
- See: https://web.dev/blog/webgpu-supported-major-browsers

## Deployment Notes

- **VPS**: Use `./deploy.sh` for complete Docker setup with nginx, SSL, Redis
- **Render**: Use `render.yaml` blueprint for cloud deployment
- **Local**: Use `scripts/start_local.py` for development

### Current Production Status

**Active Deployment**: Nginx + Uvicorn + Systemd
- **URL**: Production deployment (configure in deployment settings)
- **Architecture**:
  - Nginx (port 80): Serves static files and proxies API requests
  - Uvicorn (port 8000): FastAPI with 4 workers
  - Systemd service: `yonearth-api` (auto-restart, boot persistence)
- **Features**: Full web interface + working API endpoints (/api/chat, /api/bm25/chat)
- **Vector Database**: 18,764+ vectors (episodes + books combined)
- **Book Integration**: All 3 books successfully processed and searchable
- **Episode Coverage**: 172 episodes (0-172, excluding #26)

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

**System Health (2025-10-07)**:
- **Transcription Dataset**: ✅ **172/172 episodes with word-level timestamps (100% complete)**
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
- ✅ **COMPLETE: All 172 episodes re-transcribed with word-level timestamps (2025-10-07)**
  - 14 episodes transcribed from YouTube fallback (broken/missing audio)
  - 158 episodes transcribed from original audio
  - 100% coverage of all publishable episodes
  - Ready for 3D map navigation with precise timestamps
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