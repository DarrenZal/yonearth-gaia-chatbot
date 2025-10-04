# Complete Repository Structure

**Comprehensive file-by-file guide to the YonEarth Gaia Chatbot codebase**

---

## 📂 Root Directory

### Essential Files

**README.md**
Main project documentation. Overview of features, architecture, quick start guide, and deployment instructions. Start here for project understanding.

**CLAUDE.md**
Instructions for Claude Code development. Contains development commands, architecture details, deployment workflows, and technical implementation guides.

**CLEANUP_COMPLETE.md**
Summary of October 2025 repository cleanup. Documents what was deleted, archived, and reorganized. Can be archived after review.

**deploy.sh**
Main Docker deployment script. One-command setup for containerized deployment with nginx, Redis, and SSL support.

**docker-compose.yml**
Docker Compose configuration. Defines services (api, nginx, redis), volumes, networks, and environment variables for containerized deployment.

**requirements.txt**
Python dependencies. Complete list of packages required for the project (FastAPI, OpenAI, Pinecone, ElevenLabs, etc.).

**package.json** / **package-lock.json**
Node.js dependencies for Playwright browser automation (used for MCP integration). Minimal JavaScript tooling.

**.env.example**
Environment variable template. Shows required configuration (API keys, settings) without actual credentials. Copy to `.env` for local development.

**.gitignore**
Git ignore rules. Protects sensitive data (.env), large files (data directories), generated files (wiki, node_modules), and build artifacts.

---

## 📚 Documentation (`docs/`)

### Active Documentation

**docs/README.md**
Documentation index and navigation guide. Provides learning paths for beginners/intermediate/advanced users. Links to all major docs.

**docs/SETUP_AND_DEPLOYMENT.md** ⭐
Complete setup guide (400+ lines). Covers Docker quick start, local development, VPS production deployment, systemd configuration, nginx setup, and troubleshooting.

**docs/FEATURES_AND_USAGE.md** ⭐
Feature documentation (400+ lines). Explains search methods, personality system, voice integration, multi-content search, smart recommendations, feedback system, and best practices.

**docs/CONTENT_PROCESSING_PIPELINE.md**
Data processing workflows. Episode ingestion, book processing, knowledge graph extraction, and data pipeline management.

**docs/VOICE_INTEGRATION.md**
ElevenLabs voice integration guide. Setup, configuration, text preprocessing, and audio playback implementation.

**docs/COST_TRACKING.md**
Cost calculation and tracking. Methods for tracking API usage costs (OpenAI, ElevenLabs) and optimization strategies.

**docs/EPISODE_COVERAGE.md**
Complete episode inventory. List of all 172 episodes (0-172, excluding #26), scraping methodology, and data quality notes.

**docs/VPS_DEPLOYMENT.md**
Production server deployment guide. Ubuntu VPS setup, nginx configuration, systemd service management, and SSL/HTTPS setup.

**docs/REMAINING_TODOS.md**
Outstanding features and roadmap. Prioritized list of features to implement (hyperlinks, search weights UI, feedback analytics, etc.).

### Historical Archive (`docs/archive/`)

**docs/archive/IMPLEMENTATION_PLAN.md**
Original BM25 hybrid search implementation plan from July 2025. Historical reference for understanding BM25 system design decisions.

**docs/archive/SEMANTIC_CATEGORY_IMPLEMENTATION_PLAN.md**
Semantic category matching implementation plan. Shows evolution from keyword to semantic category understanding.

**docs/archive/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md**
Knowledge graph extraction planning. Strategy for entity and relationship extraction from episodes.

**docs/archive/VOICE_INTEGRATION_COMPLETE.md**
Voice integration completion report. Documents successful ElevenLabs integration (August 2025).

**docs/archive/EPISODE_SCRAPING_COMPLETE.md**
Episode scraping completion status. Documents successful scraping of all 172 episodes.

**docs/archive/UNIFIED_KNOWLEDGE_GRAPH_SUMMARY.md**
Knowledge graph statistics and summary. Total entities, relationships, and extraction completion status.

**docs/archive/INTEGRATION_GUIDE.md** / **INTEGRATION_GUIDE_FREE_PLAN.md**
Historical integration guides (superseded by current docs).

**docs/archive/PODCAST_MAP_COMPLETE.md** / **PODCAST_MAP_SUMMARY.md**
Podcast map visualization completion reports.

**docs/archive/KNOWLEDGE_GRAPH_FINAL_REPORT.md** / **KNOWLEDGE_GRAPH_METHODOLOGY_REPORT.md**
Detailed knowledge graph extraction reports with methodology and results.

**docs/archive/AGENT_*.md**
Historical agent reports from development sessions (October 2025).

---

## 🐍 Source Code (`src/`)

### API Layer (`src/api/`)

**src/api/main.py** ⭐
Main FastAPI application. Defines original RAG chat endpoints, search, recommendations, health checks, CORS, rate limiting, and middleware.

**src/api/models.py**
Pydantic models for original RAG API. Request/response schemas for chat, search, and recommendation endpoints.

**src/api/bm25_endpoints.py** ⭐
BM25 hybrid search API endpoints. Advanced category-first search with comparison features, configurable thresholds, and episode diversity.

**src/api/bm25_models.py**
Pydantic models for BM25 hybrid search. Schemas for BM25 chat requests, responses, and comparison results.

**src/api/voice_endpoints.py**
ElevenLabs voice generation endpoints. Text-to-speech API for converting Gaia responses to audio.

**src/api/podcast_map_route.py** / **podcast_map_route_local.py**
Podcast map visualization endpoints. Generate episode relationship maps and semantic clustering visualizations.

### RAG Systems (`src/rag/`)

**src/rag/chain.py** ⭐
Original RAG orchestration pipeline. Combines keyword + semantic search, generates responses with Gaia character, and formats citations.

**src/rag/bm25_chain.py** ⭐
BM25 RAG pipeline with category-first search. Advanced retrieval with RRF (Reciprocal Rank Fusion), cross-encoder reranking, and episode diversity.

**src/rag/hybrid_retriever.py**
Original hybrid retriever. Combines keyword frequency search with semantic vector search using weighted scoring.

**src/rag/bm25_hybrid_retriever.py** ⭐
Advanced BM25 hybrid retriever. Implements category matching (60-80%), semantic search (15-25%), BM25 keywords (5-15%), and cross-encoder reranking.

**src/rag/semantic_category_matcher.py** ⭐
Semantic category matching using OpenAI embeddings. TRUE semantic understanding (e.g., "soil" → BIOCHAR 32.1% similarity). Caches embeddings for performance.

**src/rag/episode_categorizer.py**
Episode categorization from CSV tracking data. Maps episodes to categories from `/data/PodcastPipelineTracking.csv`.

**src/rag/keyword_indexer.py**
Keyword frequency indexer. Builds term frequency maps across episodes for improved citation accuracy.

**src/rag/vectorstore.py**
Pinecone vector database wrapper. Handles embedding generation, vector upserts, similarity search, and metadata filtering.

**src/rag/pinecone_setup.py**
Pinecone index initialization and management. Creates indexes, configures dimensions (1536), and sets up cosine similarity metric.

### Character System (`src/character/`)

**src/character/gaia.py** ⭐
Gaia character implementation. Manages personality, memory, citation handling, custom prompts, and response generation with OpenAI.

**src/character/gaia_personalities.py**
Personality variant definitions. Three predefined personalities (warm_mother, wise_guide, earth_activist) with distinct system prompts.

### Voice System (`src/voice/`)

**src/voice/elevenlabs_client.py**
ElevenLabs Text-to-Speech client. Converts text responses to natural speech, preprocesses markdown/citations, returns base64-encoded audio.

### Data Ingestion (`src/ingestion/`)

**src/ingestion/episode_processor.py**
Episode transcript processing. Loads JSON files, validates data, extracts metadata (title, guest, URLs), and prepares for chunking.

**src/ingestion/book_processor.py**
Book PDF processing. Extracts text using pdfplumber, detects chapters, handles metadata, and prepares for vectorization.

**src/ingestion/chunker.py**
Text chunking utilities. Splits content into overlapping segments (500 tokens for episodes, 750 for books) for vector embedding.

**src/ingestion/process_episodes.py**
Main episode processing pipeline. Orchestrates loading, chunking, embedding, and Pinecone upload for all 172 episodes.

**src/ingestion/process_books.py**
Main book processing pipeline. Orchestrates PDF extraction, chunking, embedding, and Pinecone upload for integrated books.

### Configuration (`src/config/`)

**src/config/settings.py** ⭐
Centralized settings using Pydantic Settings. Loads environment variables, validates configuration, provides typed access to settings.

### Utilities (`src/utils/`)

**src/utils/cost_calculator.py**
API cost calculation utilities. Tracks OpenAI embedding costs, completion costs, ElevenLabs TTS costs, and provides per-response breakdowns.

### Knowledge Graph (`src/knowledge_graph/`)

**src/knowledge_graph/build_graph.py**
Main knowledge graph builder. Coordinates entity/relationship extraction and graph construction.

**src/knowledge_graph/unified_builder.py**
Unified graph builder. Combines multiple episode extractions into single coherent knowledge graph.

**src/knowledge_graph/ontology.py**
Knowledge graph ontology definitions. Entity types (PERSON, ORGANIZATION, CONCEPT, etc.) and relationship types.

**src/knowledge_graph/demo_queries.py**
Example knowledge graph queries. Demonstrates graph traversal and relationship finding.

#### Extractors (`src/knowledge_graph/extractors/`)

**entity_extractor.py**
Entity extraction using OpenAI structured outputs. Identifies people, organizations, concepts, places, practices, and products.

**relationship_extractor.py**
Relationship extraction using structured outputs. Finds connections between entities (FOUNDED, WORKS_FOR, PRACTICES, etc.).

**chunking.py**
Knowledge graph-specific chunking. 800-token chunks with 100-token overlap for extraction context.

**ontology.py**
Extractor ontology definitions. Pydantic schemas for structured output validation.

#### Graph (`src/knowledge_graph/graph/`)

**graph_builder.py**
Neo4j graph construction. Builds nodes and relationships from extracted entities.

**graph_queries.py**
Graph query utilities. Find paths, related entities, and execute Cypher queries.

**neo4j_client.py**
Neo4j database client. Connection management and query execution.

#### Visualization (`src/knowledge_graph/visualization/`)

**export_visualization.py**
Knowledge graph visualization export. Generates JSON/GML formats for external visualization tools.

#### Wiki (`src/knowledge_graph/wiki/`)

**wiki_builder.py**
Wiki site builder. Generates static wiki from knowledge graph data.

**markdown_generator.py**
Markdown page generation. Creates wiki pages for entities and relationships.

---

## 🌐 Web Interface (`web/`)

### Production Files (Active)

**web/index.html** ⭐
Main chat interface. Beautiful responsive UI with personality selection, search method toggles, voice controls, and category threshold settings.

**web/chat.js** ⭐
Frontend JavaScript logic (800+ lines). Handles dual search modes, smart recommendations, conversation management, voice playback, feedback submission, and localStorage persistence.

**web/styles.css** ⭐
Earth-themed styling (600+ lines). Responsive design, comparison views, dark/light themes, and accessibility features.

### Experimental/Debug Files

**web/debug_voice.html** / **test_voice.html** / **voice_manual_test.html**
Voice integration testing pages. Manual testing interfaces for ElevenLabs TTS functionality.

**web/KnowledgeGraph.html** / **KnowledgeGraph.js** / **KnowledgeGraph.css**
Knowledge graph visualization interface. Interactive graph explorer (experimental).

**web/PodcastMap.html** / **PodcastMap.js** / **PodcastMap.css**
Standard podcast map visualization. 2D episode clustering and similarity visualization.

**web/PodcastMap3D.html** / **PodcastMap3D.js** / **PodcastMap3D.css**
3D podcast map visualization. Three.js-based 3D episode clustering.

**web/PodcastMapHierarchical.html** / **PodcastMapHierarchical.js** / **PodcastMapHierarchical.css**
Hierarchical podcast map. Tree-based episode relationship visualization.

**web/PodcastMapUMAP.html** / **PodcastMapUMAP.js** / **PodcastMapUMAP.css**
UMAP-based podcast map. Advanced dimensionality reduction visualization.

**web/PodcastMapNomic.html** / **PodcastMapNomic.js** / **PodcastMapNomic.css**
Nomic Atlas integration. External embedding projection visualization.

---

## 🛠️ Scripts (`scripts/`)

### Active Utility Scripts

**scripts/start_local.py** ⭐
Development server launcher. Starts uvicorn with hot reload for local development.

**scripts/view_feedback.py** ⭐
Feedback analytics viewer. Display user feedback statistics and comments from JSON files.

**scripts/add_to_vectorstore.py**
Add content to Pinecone. Utility for manually adding new episodes or books to vector database.

**scripts/inspect_pinecone_books.py**
Inspect book data in Pinecone. Debug tool for verifying book chunks and metadata.

**scripts/fix_book_formatting.py**
Fix book formatting issues. Utility for correcting book metadata and chapter references.

**scripts/setup_data.py**
Initial data setup. One-time script for setting up data directories and downloading initial datasets.

### Deployment Scripts (`scripts/deployment/`)

**scripts/deployment/simple_server.py**
Simple HTTP server. Basic server for quick testing without full FastAPI setup.

**scripts/deployment/start_production.py**
Production server startup. Launches uvicorn with production workers and configuration.

**scripts/deployment/monitor_server.sh**
Server monitoring script. Checks server health and resource usage.

**scripts/deployment/run_server.sh**
Server runner script. Alternative server startup with environment configuration.

**scripts/deployment/deploy_podcast_map.sh**
Deploy podcast map visualization. Specialized deployment for map feature.

**scripts/deployment/deploy_soil_handbook_wiki.sh**
Deploy soil handbook wiki. Specialized deployment for wiki generator.

**scripts/deployment/test_podcast_map_local.sh**
Test podcast map locally. Local testing before production deployment.

### Archived Scripts (`scripts/archive/`)

#### Debug Scripts (`scripts/archive/debug/`)
5 debug scripts for troubleshooting API issues, metadata problems, and book formatting bugs. Used during development, kept for reference.

#### Testing Scripts (`scripts/archive/testing/`)
7 test scripts for API endpoints, voice integration, structured outputs, and book integration. Historical test files replaced by formal test suite.

#### Extraction Scripts (`scripts/archive/extraction/`)
16 one-time extraction scripts used to process all 172 episodes and extract knowledge graphs. Completed tasks, archived for reproducibility.

#### Scraping Scripts (`scripts/archive/scraping/`)
3 episode scraping scripts used to download transcripts from YonEarth website. Completed, archived for reference.

#### Visualization Scripts (`scripts/archive/visualization/`)
12 wiki and map generation scripts for various visualization experiments (Nomic, UMAP, BERTopic, hierarchical). Experimental features, some integrated into main codebase.

---

## 🗄️ Data Directories (`data/`)

**Note**: Data directories are gitignored (large files, regeneratable)

### data/transcripts/
Episode JSON files (172 episodes). Each file contains episode metadata and full transcript. ~7.2MB total.

### data/processed/
Processed data outputs:
- `episode_metadata.json` - Episode metadata index
- `category_embeddings.json` - Cached OpenAI embeddings for semantic category matching
- `podcast_map_data.json` - Podcast map visualization data

### data/knowledge_graph/
Knowledge graph extractions (~27MB):
- `entities/` - Entity extraction JSON files (per episode)
- `relationships/` - Relationship extraction JSON files (per episode)
- `graph/` - Unified knowledge graph outputs

### data/feedback/
User feedback data (~24KB):
- JSON files organized by date (`feedback_YYYY-MM-DD.json`)
- Contains ratings, comments, and episode correctness flags
- **Gitignored** (may contain PII)

### data/books/
Book PDF files and metadata (gitignored - large files):
- `VIRIDITAS/` - VIRIDITAS: THE GREAT HEALING
- `soil-stewardship-handbook/` - Soil Stewardship Handbook
- `y-on-earth/` - Y on Earth book

---

## 🧪 Tests (`tests/`)

**Note**: Test suite structure (to be expanded)

Basic test framework in place. Future expansion planned for:
- API endpoint testing
- RAG pipeline testing
- Character response testing
- Voice integration testing

---

## 🎯 Key File Relationships

### Chat Flow (User Query → Response)

1. **User → web/index.html** (UI)
2. **web/chat.js** (frontend logic)
3. **src/api/main.py** OR **src/api/bm25_endpoints.py** (API endpoint)
4. **src/rag/chain.py** OR **src/rag/bm25_chain.py** (RAG pipeline)
5. **src/rag/bm25_hybrid_retriever.py** (search + retrieval)
6. **src/rag/semantic_category_matcher.py** (category matching)
7. **src/rag/vectorstore.py** (Pinecone query)
8. **src/character/gaia.py** (response generation)
9. **src/voice/elevenlabs_client.py** (optional TTS)
10. **Response → web/chat.js → user**

### Data Processing Flow

1. **Raw data** (PDF/JSON)
2. **src/ingestion/episode_processor.py** OR **src/ingestion/book_processor.py**
3. **src/ingestion/chunker.py** (text chunking)
4. **src/rag/vectorstore.py** (embedding + upload)
5. **Pinecone vector database** (cloud storage)

### Configuration Flow

1. **.env** (local secrets)
2. **src/config/settings.py** (loaded settings)
3. **All modules** import from settings

---

## 📍 Where to Look for Common Tasks

**Add new API endpoint**: `src/api/main.py` or `src/api/bm25_endpoints.py`
**Modify search behavior**: `src/rag/bm25_hybrid_retriever.py`
**Change Gaia personality**: `src/character/gaia_personalities.py`
**Update UI**: `web/index.html`, `web/chat.js`, `web/styles.css`
**Add new book**: `src/ingestion/process_books.py`, then run processing
**Debug search issues**: `src/rag/semantic_category_matcher.py`, `src/rag/bm25_chain.py`
**Track costs**: `src/utils/cost_calculator.py`
**Voice issues**: `src/voice/elevenlabs_client.py`, `src/api/voice_endpoints.py`
**Deployment**: `docs/SETUP_AND_DEPLOYMENT.md`, `deploy.sh`

---

**Last Updated**: October 4, 2025
