# Content Processing Pipeline Guide 📚

> Comprehensive documentation for processing and ingesting content into the YonEarth Gaia Chatbot

## Table of Contents

1. [Overview](#overview)
2. [Podcast Episode Processing](#podcast-episode-processing)
3. [Book Processing](#book-processing)
4. [Architecture & Data Flow](#architecture--data-flow)
5. [Configuration & Setup](#configuration--setup)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

## Overview

The YonEarth Gaia Chatbot uses a sophisticated content processing pipeline to transform various media types into searchable, contextual knowledge. The system currently supports podcast episodes and is designed to expand to books and other content types.

### Supported Content Types

| Content Type | Status | Input Format | Output |
|--------------|--------|--------------|--------|
| **Podcast Episodes** | ✅ Active | JSON transcripts | 14,475+ searchable chunks |
| **Books** | ✅ Active | PDF + JSON metadata | 4,289+ chapter-based chunks |
| **Articles** | 📋 Planned | HTML, Markdown | Topic-based segments |
| **Videos** | 📋 Planned | Transcript files | Time-stamped chunks |

### Processing Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Content  │───▶│  Content Parser  │───▶│   Text Chunks   │
│ Podcasts/Books  │    │ Extract/Validate │    │ 500-token segs  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌───────▼─────────┐
│ Search Indexes  │◀───│  Vector Database │◀───│   Embeddings    │
│ BM25 + Keyword  │    │    Pinecone      │    │ OpenAI API      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Podcast Episode Processing

### 🎯 Current Implementation (Active)

The podcast processing pipeline transforms YonEarth Community podcast episode transcripts into searchable knowledge chunks.

#### Data Sources
- **Location**: `/root/yonearth-gaia-chatbot/data/transcripts/`
- **Format**: JSON files with `full_transcript` field
- **Count**: 172 episode files
- **Total Content**: ~6.8M characters of transcript text

#### Processing Pipeline

**Step 1: Episode Discovery & Validation**
```bash
# Discover all episode files
find /root/yonearth-gaia-chatbot/data/transcripts -name "episode_*.json" | wc -l
# Output: 172 files found
```

**Step 2: Content Extraction**
```python
# Extract key fields from each episode JSON
{
  "title": "Episode 101 – William Karstens, Sacred Geometry",
  "episode_number": 101,
  "full_transcript": "...",  # Main content
  "audio_url": "https://...",
  "url": "https://yonearth.org/podcast/...",
  "subtitle": "...",
  "description": "...",
  "publish_date": "..."
}
```

**Step 3: Content Validation**
- ✅ **Transcript length**: Must be > 100 characters
- ✅ **Required fields**: title, episode_number, full_transcript
- ✅ **Text quality**: Remove empty episodes or corrupted transcripts

**Step 4: Text Chunking**
```python
# Chunking configuration
CHUNK_SIZE = 500        # tokens per chunk
CHUNK_OVERLAP = 50      # token overlap between chunks
CHUNKING_METHOD = "recursive_character"  # LangChain splitter
```

**Step 5: Metadata Enrichment**
```python
# Each chunk gets comprehensive metadata
{
  "content": "transcript text chunk...",
  "metadata": {
    "episode_number": "101",
    "title": "Episode 101 – William Karstens...",
    "guest_name": "William Karstens",
    "publish_date": "Unknown",
    "url": "https://yonearth.org/podcast/...",
    "audio_url": "https://media.blubrry.com/...",
    "subtitle": "Sacred Geometry & Subatomic Physics",
    "chunk_index": 15,
    "chunk_total": 136,
    "chunk_type": "speaker_turn"
  }
}
```

**Step 6: Vector Embedding Generation**
```python
# OpenAI embeddings for semantic search
model = "text-embedding-ada-002"
dimensions = 1536
api_cost = ~$0.10 per 1M tokens
```

**Step 7: Database Storage**
```python
# Pinecone vector database
index_name = "yonearth-episodes"
dimensions = 1536
metric = "cosine"
chunks_stored = 14475  # Total chunks from 172 episodes
```

**Step 8: Search Index Creation**
```python
# BM25 keyword search index
- Episode-level keyword frequency maps
- Important term extraction (TF-IDF)
- Cross-reference tables for hybrid search
```

#### Processing Commands

**Process All Episodes:**
```bash
# Full processing pipeline
cd /root/yonearth-gaia-chatbot
export EPISODES_TO_PROCESS=172
python3 -m src.ingestion.process_episodes
```

**Process Subset (Testing):**
```bash
# Process only first 20 episodes
export EPISODES_TO_PROCESS=20
python3 -m src.ingestion.process_episodes
```

**Verify Processing Results:**
```bash
# Check processed episode count
cat data/processed/episode_metadata.json | jq '.episode_count'

# Check chunk count
cat data/processed/chunks_preview.json | jq '.total_chunks'

# Verify Pinecone index stats
python3 -c "
from src.rag.pinecone_setup import get_index_stats
print(get_index_stats())
"
```

#### Processing Results

| Metric | Value | Description |
|--------|-------|-------------|
| **Episodes Processed** | 172 | All available transcripts |
| **Total Chunks** | 14,475 | 500-token segments |
| **Average Episode Length** | 84 chunks | ~42,000 characters |
| **Processing Time** | 5-10 minutes | Depends on API limits |
| **Storage Size** | ~50MB | Vector embeddings |
| **Search Index Size** | ~5MB | BM25 keyword maps |

## Book Processing

### ✅ Active Implementation

The book processing pipeline extends the current architecture to handle longer-form content with chapter-based organization. This system is now fully operational and integrated with the vector database.

#### Supported Book Formats
- **PDF**: ✅ Extract text using pdfplumber (fully implemented)
- **EPUB**: 📋 Parse with ebooklib for structured content (planned)
- **TXT**: 📋 Direct text processing with chapter detection (planned)
- **Markdown**: 📋 Structured processing with header-based sections (planned)

#### Book Processing Pipeline Design

**Step 1: Format Detection & Parsing**
```python
# Multi-format book parser
{
  "pdf": "pdfplumber.open(book_path)",
  "epub": "ebooklib.epub.read_epub(book_path)", 
  "txt": "open(book_path, 'r').read()",
  "md": "markdown.markdown(text, extensions=['toc'])"
}
```

**Step 2: Chapter Extraction**
```python
# Intelligent chapter detection
{
  "title": "VIRIDITAS: THE GREAT HEALING",
  "author": "Aaron William Perry",
  "chapters": [
    {
      "number": 1,
      "title": "Introduction to Viriditas",
      "content": "...",
      "page_range": [1, 15],
      "word_count": 3500
    },
    # ... more chapters
  ]
}
```

**Step 3: Smart Chunking Strategy**
```python
# Chapter-aware chunking
BOOK_CHUNK_SIZE = 750        # Larger chunks for books
BOOK_CHUNK_OVERLAP = 100     # More overlap for context
CHAPTER_BOUNDARY_RESPECT = True  # Don't split across chapters
```

**Step 4: Enhanced Metadata**
```python
# Rich book metadata per chunk
{
  "content": "book text chunk...",
  "metadata": {
    "book_title": "VIRIDITAS: THE GREAT HEALING",
    "author": "Aaron William Perry",
    "chapter_number": 3,
    "chapter_title": "The Healing Power of Nature",
    "page_number": 45,
    "section": "Regenerative Principles",
    "chunk_index": 12,
    "chunk_total": 89,
    "content_type": "book",
    "publication_year": 2023,
    "isbn": "...",
    "topics": ["regenerative agriculture", "healing", "nature"]
  }
}
```

**Step 5: Topic Modeling**
```python
# Automatic topic extraction for books
- Chapter-level topic identification
- Cross-reference with podcast episodes
- Thematic clustering for related content discovery
```

#### Book Processing Commands

**Process All Books:**
```bash
# Process all books with metadata.json files
python3 -m src.ingestion.process_books
```

**Add Books to Vector Database:**
```bash
# Process and add books to vector database
python3 -c "
from src.ingestion.process_books import add_books_to_vectorstore
add_books_to_vectorstore()
"
```

**Book-Specific Search:**
```bash
# Search within specific book using BM25 endpoint
curl -X POST http://localhost:8000/bm25/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does viriditas mean?", "search_method": "hybrid", "k": 5}'
```

#### Current Book Processing Results

**VIRIDITAS: THE GREAT HEALING** by Aaron William Perry:
- **Pages Processed**: 568 pages (PDF extraction)
- **Word Count**: 211,254 words
- **Chapters Detected**: 313 chapters (automatic detection)
- **Chunks Created**: 2,029 searchable chunks
- **Vector Database**: Successfully added to Pinecone
- **Processing Time**: ~2 minutes (PDF extraction + chunking)
- **Topics**: viriditas, regenerative agriculture, healing, nature connection, earth healing, ecological restoration

**Soil Stewardship Handbook** by Aaron William Perry:
- **Pages Processed**: 53 pages (PDF extraction)
- **Word Count**: 12,430 words
- **Chapters Detected**: 36 chapters (automatic detection)
- **Chunks Created**: 136 searchable chunks
- **Vector Database**: Successfully added to Pinecone
- **Processing Time**: ~30 seconds (PDF extraction + chunking)
- **Topics**: soil stewardship, regenerative agriculture, composting, biochar, victory gardens, soil health, permaculture

**Y on Earth: Get Smarter, Feel Better, Heal the Planet** by Aaron William Perry:
- **Pages Processed**: 547 pages (PDF extraction)
- **Word Count**: 209,649 words
- **Chapters Detected**: 488 chapters (automatic detection)
- **Chunks Created**: 2,124 searchable chunks
- **Vector Database**: Successfully added to Pinecone
- **Processing Time**: ~2 minutes (PDF extraction + chunking)
- **Topics**: sustainable living, mindfulness, community, environmental stewardship, personal wellness, regenerative practices

## Architecture & Data Flow

### Content Processing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT INGESTION LAYER                     │
├─────────────────────┬───────────────────┬───────────────────────┤
│   Podcast Episodes  │      Books        │    Future Content     │
│   ┌─────────────┐  │  ┌─────────────┐  │  ┌─────────────────┐  │
│   │ JSON Files  │  │  │ PDF/EPUB    │  │  │ Articles/Videos │  │
│   │ 172 Episodes│  │  │ Chapters    │  │  │ Web Scraping    │  │
│   └─────────────┘  │  └─────────────┘  │  └─────────────────┘  │
└─────────────────────┴───────────────────┴───────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                         │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Parser    │→ │   Chunker   │→ │    Metadata Enricher    │ │
│  │ Extract     │  │ 500-750     │  │ Episode/Chapter Info    │ │
│  │ Validate    │  │ tokens      │  │ Guest/Author Details    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE & INDEXING                          │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Vector Database │  │  Search Indexes │  │  Metadata DB    │ │
│  │   Pinecone      │  │ BM25 + Keyword  │  │   Episode Info  │ │
│  │ 14,475+ vectors │  │  Frequency Maps │  │   Chapter Info  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      RAG RETRIEVAL                             │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Semantic Search │  │ Keyword Search  │  │ Hybrid Fusion   │ │
│  │ OpenAI Embed.   │  │ BM25 Algorithm  │  │ RRF + Rerank    │ │
│  │ Cosine Similar. │  │ Term Frequency  │  │ Cross-Encoder   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
/root/yonearth-gaia-chatbot/
├── data/
│   ├── transcripts/           # 172 podcast episode JSON files
│   │   ├── episode_0.json
│   │   ├── episode_1.json
│   │   └── ...
│   ├── books/                 # Book files with metadata
│   │   └── veriditas/
│   │       ├── VIRIDITAS by AARON WILLIAM PERRY.pdf
│   │       └── metadata.json
│   └── processed/             # Generated processing outputs
│       ├── episode_metadata.json      # Episode processing results
│       ├── chunks_preview.json        # Episode chunk samples
│       ├── book_metadata.json         # Book processing results
│       └── book_chunks_preview.json   # Book chunk samples
├── src/
│   ├── ingestion/             # Content processing pipeline
│   │   ├── episode_processor.py       # Podcast episode processing
│   │   ├── book_processor.py          # Book processing (PDF extraction, chapter detection)
│   │   ├── process_books.py           # Main book processing pipeline
│   │   ├── chunker.py                 # Text chunking logic
│   │   └── process_episodes.py        # Main processing script
│   ├── rag/                   # Retrieval system
│   │   ├── vectorstore.py             # Pinecone interface
│   │   ├── bm25_hybrid_retriever.py   # Keyword + semantic search
│   │   └── pinecone_setup.py          # Database initialization
│   └── config/
│       └── settings.py                # Configuration management
└── docs/
    ├── CONTENT_PROCESSING_PIPELINE.md # This document
    ├── CLAUDE.md                      # Development guide
    └── VPS_DEPLOYMENT.md              # Deployment guide
```

## Configuration & Setup

### Environment Variables

```bash
# Required API Keys
export OPENAI_API_KEY="sk-..."          # OpenAI API for embeddings
export PINECONE_API_KEY="..."           # Pinecone vector database

# Processing Configuration
export EPISODES_TO_PROCESS=172          # Number of episodes to process
export BOOKS_TO_PROCESS=10              # Future: Number of books
export CHUNK_SIZE=500                   # Token size per chunk
export CHUNK_OVERLAP=50                 # Overlap between chunks

# Performance Settings
export MAX_CONCURRENT_REQUESTS=10       # API rate limiting
export BATCH_SIZE=100                   # Processing batch size
export RETRY_ATTEMPTS=3                 # Failed request retries
```

### Pinecone Setup

```python
# Create Pinecone index (one-time setup)
import pinecone

pinecone.init(api_key="your-key", environment="us-west1-gcp")

# Create index for content
pinecone.create_index(
    name="yonearth-episodes",
    dimension=1536,          # OpenAI embedding size
    metric="cosine",         # Similarity metric
    metadata_config={
        "indexed": [
            "episode_number",
            "book_title", 
            "content_type",
            "chapter_number"
        ]
    }
)
```

### Rebuilding Search Indexes

**When to rebuild indexes:**
- After processing new episodes
- When search results seem outdated  
- After cloning the repository
- If episode content isn't being found in searches

**Complete rebuild process:**
```bash
# 1. Install dependencies (if needed)
python3 -m pip install langchain-openai langchain-pinecone

# 2. Rebuild vector database and search indexes
python3 scripts/add_to_vectorstore.py

# 3. Restart services to reload updated data
docker-compose restart

# 4. Wait for services to initialize (15 seconds)
sleep 15

# 5. Verify system health
curl http://localhost:8000/health

# 6. Test with biochar query to verify new episodes
curl -X POST http://localhost:8000/bm25/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about biochar", "search_method": "hybrid", "k": 5}'
```

**Expected results after rebuild:**
- Vector database shows 16,000+ total vectors
- Biochar episodes (120, 122, 165) appear in search results
- All 172 episodes accessible through search
- BM25 hybrid search shows correct episode references

**Troubleshooting index rebuild:**
```bash
# Check if script exists
ls -la scripts/add_to_vectorstore.py

# Check vector database stats
python3 -c "
from src.rag.vectorstore import get_vectorstore
vs = get_vectorstore()
print(f'Vector count: {vs.get_stats()}')
"

# Test specific episode search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Episode 120 Rowdy Yeatts", "max_results": 3}'
```

### Dependencies

```bash
# Core processing libraries
pip install langchain>=0.1.0
pip install openai>=1.12.0
pip install pinecone-client>=3.0.0
pip install tiktoken>=0.5.2

# Text processing
pip install beautifulsoup4>=4.12.0
pip install nltk>=3.8.0
pip install sentence-transformers>=2.3.0

# Book processing (future)
pip install PyPDF2>=3.0.0
pip install pdfplumber>=0.9.0
pip install ebooklib>=0.18
pip install python-docx>=0.8.11
```

## Troubleshooting

### Common Issues & Solutions

#### Episode Processing Issues

**Issue: No episodes found**
```bash
# Check transcript directory
ls -la /root/yonearth-gaia-chatbot/data/transcripts/
# Expected: 172 episode_*.json files

# Verify file format
head -20 /root/yonearth-gaia-chatbot/data/transcripts/episode_1.json
# Should contain "full_transcript" field
```

**Issue: Processing fails on specific episodes**
```bash
# Check episode validation
python3 -c "
import json
from pathlib import Path

transcript_dir = Path('/root/yonearth-gaia-chatbot/data/transcripts')
for file_path in transcript_dir.glob('episode_*.json'):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        transcript = data.get('full_transcript', '')
        if len(transcript) < 100:
            print(f'{file_path.name}: INVALID - transcript too short ({len(transcript)} chars)')
        else:
            print(f'{file_path.name}: VALID - {len(transcript)} chars')
    except Exception as e:
        print(f'{file_path.name}: ERROR - {e}')
"
```

**Issue: Pinecone connection errors**
```bash
# Test Pinecone connection
python3 -c "
import pinecone
import os

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='us-west1-gcp'  # Adjust as needed
)

try:
    index = pinecone.Index('yonearth-episodes')
    stats = index.describe_index_stats()
    print(f'Index exists. Vector count: {stats.total_vector_count}')
except Exception as e:
    print(f'Pinecone error: {e}')
"
```

#### API Rate Limiting

**Issue: OpenAI rate limits**
```python
# Add rate limiting to processing
import time
from openai import RateLimitError

def process_with_retry(content, max_retries=3):
    for attempt in range(max_retries):
        try:
            return openai.Embedding.create(
                model="text-embedding-ada-002",
                input=content
            )
        except RateLimitError:
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
            continue
    raise Exception("Max retries exceeded")
```

#### Memory & Performance Issues

**Issue: Out of memory during processing**
```bash
# Process in smaller batches
export BATCH_SIZE=50           # Reduce from default 100
export MAX_CONCURRENT_REQUESTS=5  # Reduce from default 10

# Monitor memory usage
watch -n 1 'free -h && ps aux | grep python | head -5'
```

**Issue: Slow processing speed**
```python
# Optimize chunking performance
from concurrent.futures import ThreadPoolExecutor

def parallel_chunking(episodes, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_episode, episodes))
    return results
```

### Logging & Monitoring

**Enable detailed logging:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

**Monitor processing progress:**
```bash
# Watch processing logs
tail -f processing.log

# Monitor API usage
grep "OpenAI API" processing.log | wc -l
grep "Pinecone" processing.log | wc -l
```

## Performance Optimization

### Processing Speed Optimization

**Parallel Processing:**
```python
# Process multiple episodes simultaneously
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def process_episodes_parallel(episodes, max_workers=4):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, process_episode, episode)
            for episode in episodes
        ]
        results = await asyncio.gather(*tasks)
    return results
```

**Batch API Calls:**
```python
# Batch OpenAI embedding requests
def create_embeddings_batch(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=batch
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings
```

**Caching Strategy:**
```python
# Cache processed chunks to avoid reprocessing
import hashlib
import pickle

def get_content_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

def cache_chunks(episode_id, chunks):
    cache_path = f"cache/chunks_{episode_id}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(chunks, f)

def load_cached_chunks(episode_id):
    cache_path = f"cache/chunks_{episode_id}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None
```

### Storage Optimization

**Pinecone Index Optimization:**
```python
# Use metadata filtering for efficient queries
index.query(
    vector=query_embedding,
    top_k=10,
    filter={
        "content_type": {"$eq": "podcast"},
        "episode_number": {"$gte": 100}
    },
    include_metadata=True
)
```

**Compression Strategies:**
```python
# Compress large text chunks before storage
import gzip
import base64

def compress_text(text):
    compressed = gzip.compress(text.encode('utf-8'))
    return base64.b64encode(compressed).decode('utf-8')

def decompress_text(compressed_text):
    compressed = base64.b64decode(compressed_text.encode('utf-8'))
    return gzip.decompress(compressed).decode('utf-8')
```

### Cost Optimization

**OpenAI API Cost Management:**
```python
# Estimate processing costs
def estimate_processing_cost(episodes):
    total_tokens = sum(len(ep.transcript.split()) for ep in episodes)
    embedding_cost = (total_tokens / 1000) * 0.0001  # $0.0001 per 1K tokens
    return f"Estimated cost: ${embedding_cost:.2f}"

# Token counting
import tiktoken

def count_tokens(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

**Pinecone Cost Optimization:**
```python
# Optimize vector dimensions (if using custom models)
# Note: OpenAI embeddings are fixed at 1536 dimensions

# Use sparse vectors for keyword-heavy content
sparse_vector = {
    "indices": [1, 5, 10],           # Term positions
    "values": [0.5, 0.7, 0.2]        # TF-IDF scores
}
```

---

## Summary

This content processing pipeline provides a robust, scalable foundation for transforming diverse content types into searchable knowledge. The current podcast processing implementation demonstrates the architecture's effectiveness with 172 episodes processed into 14,475+ searchable chunks.

The planned book processing extension will leverage the same architecture while adding chapter-aware chunking and enhanced metadata for longer-form content.

**Key Benefits:**
- ✅ **Scalable**: Handle large content volumes efficiently
- ✅ **Accurate**: Maintain content context and citations
- ✅ **Flexible**: Support multiple content formats
- ✅ **Optimized**: Balance processing speed, accuracy, and cost

**Next Steps:**
1. Implement book processing pipeline for VIRIDITAS and other texts
2. Add article and video content support
3. Enhance cross-content recommendations
4. Implement advanced topic modeling and content clustering