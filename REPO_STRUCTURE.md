# Repository Structure

**Last Updated**: October 12, 2025  
**Status**: Clean and organized after Oct 2025 cleanup

This document provides a comprehensive overview of the yonearth-gaia-chatbot repository structure.

---

## 📁 Root Directory

```
yonearth-gaia-chatbot/
├── CLAUDE.md                 # Primary instructions for Claude Code
├── README.md                 # Project overview and quick start
├── REPO_STRUCTURE.md         # This file - complete repo documentation
├── requirements.txt          # Main Python dependencies
├── requirements-entity-resolution.txt  # Entity resolution dependencies
├── requirements-transcription.txt      # Transcription dependencies  
├── package.json              # Node.js dependencies (if any)
├── deploy.sh                 # Deployment script
├── .env                      # Environment variables (not in git)
├── .env.example              # Environment template
└── .gitignore                # Git ignore rules
```

**Key Files**:
- **CLAUDE.md**: Comprehensive guide for Claude Code - **READ THIS FIRST**
- **README.md**: User-facing documentation
- **REPO_STRUCTURE.md**: Detailed file structure (this document)

---

## 📂 `/src` - Main Application Code

```
src/
├── api/                      # FastAPI endpoints
│   ├── main.py              # Original RAG endpoints
│   ├── bm25_endpoints.py    # BM25 hybrid search endpoints
│   ├── voice_endpoints.py   # Voice generation endpoints
│   ├── models.py            # Original API models
│   └── bm25_models.py       # BM25 API models
├── rag/                      # Retrieval & search
│   ├── chain.py             # Original RAG pipeline
│   ├── bm25_chain.py        # BM25 hybrid RAG
│   ├── hybrid_retriever.py  # Original hybrid search
│   ├── bm25_hybrid_retriever.py  # BM25 + semantic search
│   ├── semantic_category_matcher.py  # Category matching with embeddings
│   ├── episode_categorizer.py  # Episode categorization
│   ├── keyword_indexer.py   # Keyword frequency indexing
│   ├── vectorstore.py       # Pinecone wrapper
│   └── pinecone_setup.py    # Pinecone initialization
├── character/               # Gaia AI character
│   ├── gaia.py             # Main character logic
│   └── gaia_personalities.py  # Personality variants
├── voice/                   # Voice generation
│   └── elevenlabs_client.py  # ElevenLabs TTS integration
├── ingestion/              # Data processing
│   ├── episode_processor.py  # Process podcast episodes
│   ├── book_processor.py    # Process books
│   └── chunker.py           # Text chunking utilities
├── config/                  # Configuration
│   └── settings.py          # Centralized settings
└── knowledge_graph/         # KG extraction (NEW)
    └── validators.py        # Entity/relationship validators
```

---

## 📂 `/scripts` - Utility Scripts

### Active Scripts (12 files)

```
scripts/
├── extract_kg_v3_2_2.py              # Main KG extraction (episodes)
├── extract_kg_v3_2_2_book_v4_comprehensive.py  # Book extraction V4
├── retry_failed_episodes.py          # Retry failed extractions
├── retranscribe_episodes_lightweight.py    # Re-transcribe with timestamps
├── retranscribe_episodes_with_timestamps.py  # Full re-transcription
├── add_to_vectorstore.py             # Add content to Pinecone
├── setup_data.py                     # Initial data setup
├── start_local.py                    # Start local dev server
├── view_feedback.py                  # View user feedback
├── fix_book_formatting.py            # Fix book metadata
├── inspect_pinecone_books.py         # Inspect Pinecone book data
└── run_extraction_wrapper.py         # Wrapper for extractions
```

### Archive (scripts/archive/)

```
scripts/archive/
├── old_extraction_versions/    # V1-V3 extraction scripts (4 files)
├── old_test_scripts/          # Test/experiment scripts (8 files)
├── old_processing_scripts/    # Old normalization/review scripts (11 files)
├── monitor_extraction.sh      # Monitoring script
├── run_book_extraction.sh     # Old book extraction
└── run_full_extraction.sh     # Old full extraction
```

**Note**: Archive contains 26 historical scripts for reference.

---

## 📂 `/data` - Data Storage

### Active Data (9 directories)

```
data/
├── books/                    # Book PDFs and metadata
│   ├── soil-stewardship-handbook/
│   ├── viriditas/
│   └── y-on-earth/
├── transcripts/             # Podcast episode transcripts (172 episodes)
├── knowledge_graph_v3_2_2/  # Main KG extraction output (episodes)
├── knowledge_graph/         # Unified knowledge graph
├── knowledge_graph_books_v3_2_2/  # Book KG extractions
├── knowledge_graph_books_v3_2_2_improved/  # V4 book extractions
├── processed/               # Processed episode metadata
├── feedback/                # User feedback data
└── cache/                   # API response cache
```

### Archive (data/archive/)

```
data/archive/
├── knowledge_graph_v2/          # Old V2 extraction
├── knowledge_graph_unified/     # Old unified attempt
├── knowledge_graph_dual_signal_test/   # Test outputs
├── knowledge_graph_gpt5_mini_test/
├── knowledge_graph_gpt5_nano_test/
├── knowledge_graph_two_pass_test/
├── refinement_output/           # Old refinement experiments
├── test_results/                # Test result files
└── test_extraction_results.json
```

**Note**: Archive contains 14 test directories for historical reference.

---

## 📂 `/docs` - Documentation

### Active Documentation (10 files)

```
docs/
├── README.md                        # Docs overview
├── CONTENT_PROCESSING_PIPELINE.md   # Content processing guide
├── TRANSCRIPTION_SETUP.md           # Transcription setup
├── FEATURES_AND_USAGE.md            # Feature documentation
├── SETUP_AND_DEPLOYMENT.md          # Setup instructions
├── VPS_DEPLOYMENT.md                # VPS deployment guide
├── VOICE_INTEGRATION.md             # Voice integration guide
├── COST_TRACKING.md                 # API cost tracking
├── EPISODE_COVERAGE.md              # Episode coverage stats
└── REMAINING_TODOS.md               # TODO list
```

### Knowledge Graph Documentation (docs/knowledge_graph/)

```
docs/knowledge_graph/
├── V4_COMPLETE_COMPARISON.md             # V4 vs V3 vs V2 vs V1 comparison
├── V4_EXTRACTION_QUALITY_ISSUES_REPORT.md  # V4 quality analysis
├── V4_ADDITIONAL_QUALITY_ISSUES.md       # Deep quality review
├── V5_IMPLEMENTATION_PLAN.md             # V5 implementation guide
├── ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md  # Entity resolution
├── ENTITY_RESOLUTION_GUIDE.md
├── EXTRACTION_PHILOSOPHY.md
└── COMPLEX_CLAIMS_AND_QUANTITATIVE_RELATIONSHIPS.md
```

### Archive (docs/archive/)

```
docs/archive/
├── extraction_history/          # Historical extraction docs (6 files)
│   ├── EXTRACTION_DEEP_INVESTIGATION_AND_A++_DESIGN.md
│   ├── EXTRACTION_IMPROVEMENT_COMPARISON.md
│   ├── EXTRACTION_QUALITY_MASTER_GUIDE.md
│   ├── KG_V3_2_2_QUICK_START.md
│   ├── RUN3_QUALITY_AND_COVERAGE_REPORT.md
│   └── V3_2_2_TEST_RESULTS.md
└── SETUP_NEW_PODCAST_PROMPT.md
```

---

## 📂 `/web` - Frontend Interface

```
web/
├── index.html               # Main chat interface
├── chat.js                  # Chat logic and API calls
├── styles.css               # Styling
├── PodcastMap.html          # t-SNE visualization
├── PodcastMapUMAP.html      # UMAP visualization
├── PodcastMapHierarchical.html  # Hierarchical clustering
├── PodcastMapNomic.html     # Nomic Atlas view
├── KnowledgeGraphBook.html  # KG visualization (book)
└── KnowledgeGraphBook.js    # KG visualization logic
```

---

## 📂 `/tests` - Test Suite

```
tests/
├── run_tests.py             # Test runner
├── test_api.py              # API tests
├── test_rag.py              # RAG tests
└── test_character.py        # Character tests
```

---

## 🗄️ Archive Directories Summary

### What's Archived?

1. **docs/archive/extraction_history/** (7 files)
   - Historical extraction documentation
   - V1-V3 design documents
   - Old quality reports

2. **scripts/archive/** (26 files)
   - Old extraction script versions (V1-V3)
   - Test and experiment scripts
   - Old normalization/processing scripts

3. **data/archive/** (14 directories)
   - Test extraction outputs
   - Old KG versions (V2)
   - Experimental data

**Total Archived**: 47 items  
**Purpose**: Historical reference, not actively used

---

## 🎯 Current System Status

### Active Extraction Systems

1. **Episode Extraction** (`extract_kg_v3_2_2.py`)
   - Status: Production-ready
   - Version: v3.2.2
   - Coverage: 172 episodes (100% with timestamps)
   - Output: `data/knowledge_graph_v3_2_2/`

2. **Book Extraction** (`extract_kg_v3_2_2_book_v4_comprehensive.py`)
   - Status: V4 complete, V5 planned
   - Version: v4_comprehensive
   - Books: 3 (Soil Stewardship, Viriditas, Y on Earth)
   - Output: `data/knowledge_graph_books_v3_2_2_improved/`
   - Quality: 57% issues identified, V5 will fix to <10%

### Data Completeness

- ✅ **Episodes**: 172/172 transcribed with word-level timestamps
- ✅ **Books**: 3 books processed
- ✅ **Vector Database**: 18,764+ vectors in Pinecone
- ✅ **Category Embeddings**: 24 semantic categories cached

---

## 📝 Key Configuration Files

```
.env                         # Environment variables (REQUIRED)
.env.example                # Environment template
requirements.txt            # Python dependencies
package.json               # Node.js dependencies (minimal)
```

**Required Environment Variables**:
- `OPENAI_API_KEY`: OpenAI API key for embeddings
- `PINECONE_API_KEY`: Pinecone vector database key
- `ELEVENLABS_API_KEY`: ElevenLabs voice API (optional)

---

## 🚀 Quick Navigation

**For Development**:
- Start here: [`CLAUDE.md`](CLAUDE.md)
- Setup: [`docs/SETUP_AND_DEPLOYMENT.md`](docs/SETUP_AND_DEPLOYMENT.md)
- Features: [`docs/FEATURES_AND_USAGE.md`](docs/FEATURES_AND_USAGE.md)

**For Knowledge Graph Work**:
- V5 Implementation: [`docs/knowledge_graph/V5_IMPLEMENTATION_PLAN.md`](docs/knowledge_graph/V5_IMPLEMENTATION_PLAN.md)
- V4 Analysis: [`docs/knowledge_graph/V4_COMPLETE_COMPARISON.md`](docs/knowledge_graph/V4_COMPLETE_COMPARISON.md)
- Quality Issues: [`docs/knowledge_graph/V4_EXTRACTION_QUALITY_ISSUES_REPORT.md`](docs/knowledge_graph/V4_EXTRACTION_QUALITY_ISSUES_REPORT.md)

**For Content Processing**:
- Pipeline Guide: [`docs/CONTENT_PROCESSING_PIPELINE.md`](docs/CONTENT_PROCESSING_PIPELINE.md)
- Transcription: [`docs/TRANSCRIPTION_SETUP.md`](docs/TRANSCRIPTION_SETUP.md)

---

## 📊 Repository Statistics

**Active Files**:
- Root documentation: 3 files
- Python scripts: 12 active scripts
- Source code: ~30 modules
- Documentation: 10 active docs + 8 KG docs
- Data directories: 9 active
- Frontend files: 9 files

**Archived Files**:
- Scripts: 26 historical scripts
- Documentation: 7 old docs
- Data: 14 test directories

**Total Repository Size**: ~450MB (mostly transcripts and KG data)

---

## 🔄 Recent Changes (October 2025)

### Cleanup (Oct 12, 2025)
- ✅ Deleted 49 log files and temp scripts
- ✅ Archived 44 old scripts, docs, and test data
- ✅ Organized V4/V5 reports into `docs/knowledge_graph/`
- ✅ Created clean root with only 3 essential .md files

### V4 Extraction (Oct 11-12, 2025)
- ✅ Completed V4 comprehensive extraction
- ✅ Identified 57% quality issues
- ✅ Created V5 implementation plan
- ✅ Detailed quality analysis reports

### Transcription (Oct 7, 2025)
- ✅ Re-transcribed all 172 episodes with word-level timestamps
- ✅ 100% episode coverage achieved

---

**For more details, see [`CLAUDE.md`](CLAUDE.md) - the primary reference for working with this codebase.**
