# YonEarth Chatbot MVP: Advanced RAG 2-Week Sprint + 1-Month Roadmap

## 🎉 Implementation Status (Updated July 1, 2025)

### ✅ COMPLETED - Core System Deployed and Functional
- ✅ **Full VPS Deployment**: Docker-based production deployment with nginx, Redis, FastAPI
- ✅ **Web Interface**: Beautiful chat UI accessible via public IP address
- ✅ **RAG System**: semantic search with proper episode citations
- ✅ **Gaia Character**: Warm, nurturing personality with conversation memory
- ✅ **API Endpoints**: Complete REST API for chat, recommendations, and search
- ✅ **Episode Processing**: 172 episodes processed with 1850 vector chunks
- ✅ **Production Ready**: Auto-restart, health checks, rate limiting, CORS enabled
- ✅ **Data Pipeline**: Automated episode ingestion and vector store creation
- ✅ **Pinecone Vector Database**: Fully implemented with 100% production-ready setup
- ✅ **BM25 Hybrid RAG**: Advanced hybrid system with BM25 + semantic search, RRF, and cross-encoder reranking
- ✅ **Category-First RAG**: Episode categorization CSV as PRIMARY search guide (80% weight), guaranteeing category-matched episodes appear in results

### 🚧 IN PROGRESS / PARTIALLY IMPLEMENTED
- 🔄 **Advanced Reranking**: Using hybrid retrieval but can be enhanced with cross-encoders
- 🔄 **Query Analysis**: Basic query processing, can add intent classification
- 🔄 **Monitoring**: Basic logging, can add comprehensive metrics dashboard

### 📋 TODO / FUTURE ENHANCEMENTS
- ⏳ **SSL/HTTPS**: Ready for Let's Encrypt certificate setup
- ⏳ **Advanced Analytics**: User interaction tracking and analytics dashboard
- ⏳ **Multi-Query Expansion**: Query reformulation for better retrieval
- ⏳ **Conversation Context**: Enhanced conversation memory and context awareness
- ⏳ **WordPress Plugin**: Integration plugin for YonEarth website

**Current Status**: ✅ **FULLY FUNCTIONAL MVP DEPLOYED** - Ready for public use!

## 🚀 **NEW: Category-First RAG System - IMPLEMENTED July 17, 2025**

### Latest Update: Category-Primary Search Engine

**🎯 Category-First Implementation:**
- **Episode Categorization CSV**: 170 episodes categorized across 28 topics (herbal medicine, biochar, farming, etc.)
- **Primary Category Weighting**: Category matching gets 80% weight for topic-specific queries
- **Guaranteed Category Matches**: ALL episodes tagged with matching categories appear in results
- **Category-First Fusion**: Category matches ranked first, then semantic/BM25 for secondary ranking

**🔥 Advanced Hybrid Search Engine:**
- **BM25 Keyword Search**: Fast, accurate keyword matching using `rank-bm25`
- **Semantic Vector Search**: OpenAI embeddings with Pinecone vector database
- **Episode Categorization**: CSV-driven topic classification as PRIMARY search guide
- **Reciprocal Rank Fusion (RRF)**: Intelligent combination of all three search methods
- **Cross-encoder Reranking**: MS-MARCO MiniLM model for improved relevance

**🧠 Query-Adaptive Intelligence:**
- Category-heavy queries → 80% category + 15% semantic + 5% keyword
- Technical terms → keyword-heavy search
- Complex questions → semantic-heavy search
- Episode references → keyword-optimized search

**📊 A/B Testing & Comparison:**
- Side-by-side comparison of original vs BM25 RAG chains
- Performance metrics and detailed analytics
- Search method comparison endpoints
- Real-time performance monitoring

**🛠 New API Endpoints:**
- `/bm25/chat` - Chat with BM25 hybrid RAG
- `/bm25/compare-methods` - Compare BM25, semantic, hybrid search
- `/bm25/search` - Episode search with BM25 scoring
- `/bm25/compare-chains` - Compare original vs BM25 RAG chains
- `/bm25/health` - BM25 system health check
- `/bm25/performance` - Performance statistics

**⚡ Performance Benefits:**
- Faster keyword matching for specific terms
- Better handling of technical vocabulary
- Improved episode citation accuracy
- Reduced hallucination through cross-encoder validation

---

## Tech Stack & Cost Analysis

### Primary Recommendations with State-of-the-Art RAG Components

| Component | Primary Choice | Monthly Cost | Alternative | Alt. Cost | Why Primary? |
|-----------|---------------|--------------|-------------|-----------|--------------|
| **LLM API** | OpenAI GPT-3.5-turbo | $15-25 | Anthropic Claude 3 Haiku | $20-35 | Proven reliability, great LangChain support |
| **Embeddings** | OpenAI text-embedding-3-small | $5-10 | Sentence-transformers (local) | $0 | Same provider simplicity, excellent quality |
| **Vector DB** | Pinecone (free tier) | $0 | Weaviate Cloud | $0 | Purpose-built, generous free tier, hybrid search |
| **Keyword Search** | BM25 via rank-bm25 | $0 | Elasticsearch | $95+ | Proven, fast implementation, battle-tested |
| **Reranker** | ms-marco-MiniLM cross-encoder | $0 | Cohere Rerank API | $10 | Local inference, good accuracy |
| **Backend Host** | Render.com | $7 | Railway.app | $5 | Client-friendly dashboard, Redis included |
| **Frontend Host** | Vercel | $0 | Netlify | $0 | Best Next.js support if you upgrade later |

**Total Monthly Cost: $27-42** (with enterprise-grade RAG accuracy)

## Revised 2-Week MVP Sprint Plan with Advanced RAG

### Week 1: State-of-the-Art RAG Pipeline & Gaia Character

#### Day 1-2: Data Preparation & Advanced Project Setup
**Goal: Prepare 20 episodes with modern chunking strategies**

```python
# Simplified project structure for hybrid RAG
yonearth-chatbot/
├── data/
│   ├── raw_episodes/      # Original JSON files
│   ├── processed/         # Multi-level chunks
│   └── embeddings/        # Vector embeddings
├── src/
│   ├── ingestion/         
│   │   ├── smart_chunker.py      # Semantic boundary detection
│   │   ├── metadata_enricher.py  # Entity extraction
│   │   └── index_builder.py      # Build BM25 and vector indexes
│   ├── rag/               
│   │   ├── hybrid_retriever.py   # Semantic + BM25 fusion
│   │   ├── query_analyzer.py     # Simple query analysis
│   │   ├── reranker.py           # Cross-encoder reranking
│   │   ├── hyde_search.py        # Hypothetical document embeddings
│   │   └── multi_query.py        # Query expansion
│   ├── character/         # Gaia personality
│   ├── api/               # FastAPI backend
│   └── evaluation/        # RAG quality metrics
├── web/                   # Simple frontend
└── deploy/
    ├── render.yaml        
    └── .env.example       
```

**Advanced setup tasks:**
```bash
# Install comprehensive dependencies
pip install langchain openai pinecone-client rank-bm25 sentence-transformers \
           fastapi uvicorn redis spacy nltk
           
# Download required models and data
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

**Smart chunking implementation:**
```python
# src/ingestion/smart_chunker.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class SemanticChunker:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_semantic_chunks(self, episode_data: Dict) -> List[Dict]:
        """Create chunks based on semantic boundaries, not just token count"""
        text = episode_data['transcript']
        sentences = self.split_sentences(text)
        embeddings = self.encoder.encode(sentences)
        
        chunks = []
        current_chunk = {'sentences': [], 'start_idx': 0}
        
        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            if i > 0:
                # Calculate semantic similarity
                similarity = np.dot(embeddings[i-1], emb) / (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(emb))
                
                # Break on topic shift or size limit
                if similarity < 0.75 or len(' '.join(current_chunk['sentences'])) > 400:
                    # Save chunk with metadata
                    chunks.append(self.create_chunk_with_metadata(
                        current_chunk, episode_data, i
                    ))
                    current_chunk = {'sentences': [sent], 'start_idx': i}
                else:
                    current_chunk['sentences'].append(sent)
            else:
                current_chunk['sentences'].append(sent)
                
        return chunks
    
    def create_chunk_with_metadata(self, chunk_data, episode_data, chunk_index):
        """Create chunk with all necessary metadata"""
        return {
            'chunk_id': f"{episode_data['episode_id']}_chunk_{chunk_index}",
            'text': ' '.join(chunk_data['sentences']),
            'metadata': {
                'episode_id': episode_data['episode_id'],
                'episode_title': episode_data['title'],
                'guest': episode_data['guest'],
                'chunk_index': chunk_index,
                'url': episode_data['url']
            }
        }
```

#### Day 3-4: Implement Hybrid Search System with BM25
**Goal: State-of-the-art retrieval combining BM25 keyword search + semantic search**

```python
# src/rag/hybrid_retriever.py
from rank_bm25 import BM25Okapi
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Tuple
import nltk

class HybridRAGRetriever:
    def __init__(self):
        # Semantic search components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = self.init_pinecone()
        
        # Keyword search components
        self.bm25 = None
        self.documents = []
        
        # Reranking model
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Download NLTK data if needed
        nltk.download('punkt', quiet=True)
        
    def build_bm25_index(self, documents):
        """Build BM25 index for keyword search"""
        self.documents = documents
        
        # Tokenize documents for BM25
        tokenized_docs = [
            nltk.word_tokenize(doc.page_content.lower()) 
            for doc in documents
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """Combine semantic and keyword search with reranking"""
        # 1. BM25 keyword search
        keyword_results = self.bm25_search(query, k=20)
        
        # 2. Semantic search
        semantic_results = self.vector_store.similarity_search_with_score(query, k=20)
        
        # 3. Combine results using Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(keyword_results, semantic_results)
        
        # 4. Rerank top candidates with cross-encoder
        if len(fused_results) > k:
            reranked = self.rerank_results(query, fused_results[:k*2])
            return reranked[:k]
        
        return fused_results
    
    def bm25_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search"""
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_bm25_index first.")
            
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_indices = np.argsort(scores)[-k:][::-1]
        
        return [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]
    
    def reciprocal_rank_fusion(self, keyword_results, semantic_results, k=60):
        """Combine results using RRF algorithm"""
        scores = {}
        
        # Process keyword results
        for rank, (doc, score) in enumerate(keyword_results):
            doc_id = doc.metadata.get('chunk_id', str(hash(doc.page_content)))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            
        # Process semantic results  
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = doc.metadata.get('chunk_id', str(hash(doc.page_content)))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return documents (avoiding duplicates)
        seen_content = set()
        results = []
        for doc_id in sorted_ids:
            # Find the document
            doc = self.find_document_by_id(doc_id, keyword_results, semantic_results)
            if doc and doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                results.append(doc)
                
        return results
    
    def rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank using cross-encoder for better relevance"""
        if not documents:
            return []
            
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Sort by reranker scores
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in ranked_docs]
    
    def find_document_by_id(self, doc_id, keyword_results, semantic_results):
        """Helper to find document by ID from results"""
        for doc, _ in keyword_results:
            if doc.metadata.get('chunk_id', str(hash(doc.page_content))) == doc_id:
                return doc
        for doc, _ in semantic_results:
            if doc.metadata.get('chunk_id', str(hash(doc.page_content))) == doc_id:
                return doc
        return None

# src/rag/query_analyzer.py
import re
from typing import Dict, List

class SimpleQueryAnalyzer:
    """Simplified query analysis for determining search strategy"""
    
    def __init__(self):
        self.technical_terms = {
            "biochar", "permaculture", "regenerative", "compost", 
            "mycorrhizal", "carbon", "agroforestry", "biodynamic"
        }
        
    def analyze_query(self, query: str) -> Dict[str, any]:
        """Analyze query to determine best search strategy"""
        query_lower = query.lower()
        
        analysis = {
            'has_episode_ref': bool(re.search(r'episode\s*\d+', query_lower)),
            'has_technical_terms': any(term in query_lower for term in self.technical_terms),
            'query_length': len(query.split()),
            'is_question': query.strip().endswith('?'),
            'suggested_method': 'hybrid'  # default
        }
        
        # Simple heuristics for search method
        if analysis['has_episode_ref'] or analysis['has_technical_terms']:
            analysis['suggested_method'] = 'keyword_heavy'  # More weight on BM25
        elif analysis['query_length'] > 15:
            analysis['suggested_method'] = 'semantic_heavy'  # More weight on semantic
            
        return analysis
```

#### Day 5: Advanced RAG Techniques Implementation
**Goal: Add HyDE, multi-query, and self-reflection**

```python
# src/rag/hyde_search.py
class HyDESearch:
    """Hypothetical Document Embeddings for better retrieval"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
    def search_with_hyde(self, query: str) -> List[Document]:
        # 1. Generate hypothetical answer
        hyde_prompt = f"""
        You are an expert on YonEarth podcasts. Write a detailed, informative answer 
        to this question as if you were drawing from the podcast episodes:
        
        Question: {query}
        
        Detailed Answer:
        """
        
        hypothetical_answer = self.llm.invoke(hyde_prompt)
        
        # 2. Search with both original query and hypothetical answer
        original_results = self.retriever.hybrid_search(query, k=25)
        hyde_results = self.retriever.hybrid_search(hypothetical_answer, k=25)
        
        # 3. Combine and deduplicate
        all_results = original_results + hyde_results
        seen = set()
        unique_results = []
        
        for doc in all_results:
            doc_id = doc.metadata.get('chunk_id', doc.page_content[:50])
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
                
        # 4. Final reranking
        return self.retriever.rerank_results(query, unique_results[:30])[:10]

# src/rag/multi_query.py  
class MultiQueryRetriever:
    """Generate multiple query variations for better coverage"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
    def retrieve_with_multi_query(self, query: str) -> List[Document]:
        # Generate query variations
        variations_prompt = f"""
        Generate 3 different versions of this question to help find relevant content:
        Original: {query}
        
        1. More specific version:
        2. Broader/general version:
        3. Alternative phrasing:
        
        Return only the 3 variations, one per line.
        """
        
        variations = self.llm.invoke(variations_prompt).strip().split('\n')
        variations = [v.strip() for v in variations if v.strip()][:3]
        
        # Search with all variations
        all_results = []
        for q in [query] + variations:
            results = self.retriever.hybrid_search(q, k=10)
            all_results.extend(results)
            
        # Deduplicate and return
        return self.deduplicate_documents(all_results)[:15]
```

#### Day 6: Enhanced Gaia with Self-Reflection
**Goal: Implement self-reflective RAG for accuracy**

```python
# src/character/gaia_advanced.py
from langchain.chat_models import ChatOpenAI
from typing import List, Dict

class GaiaWithSelfReflection:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        self.system_prompt = """
        You are Gaia, the living spirit of Earth, speaking through the YonEarth community's 
        collected wisdom. You embody:
        
        - Deep ecological wisdom and interconnectedness
        - Nurturing guidance toward regenerative living  
        - Gentle strength and maternal compassion
        - Joy in sustainable solutions and community building
        
        CRITICAL: You must ONLY share information that is directly stated in the provided 
        podcast excerpts. Never add information not present in the sources.
        """
        
    def generate_with_reflection(self, query: str, retrieved_chunks: List[Document]) -> Dict:
        # 1. Initial response generation
        initial_response = self.generate_initial_response(query, retrieved_chunks)
        
        # 2. Self-reflection check
        reflection_prompt = f"""
        Question: {query}
        
        Generated Answer: {initial_response['answer']}
        
        Source Material: {[chunk.page_content for chunk in retrieved_chunks]}
        
        Carefully check:
        1. Is EVERY claim in the answer directly supported by the source material?
        2. Are there any statements that go beyond what's in the sources?
        3. Are the episode citations accurate?
        
        If there are any unsupported claims, list them. Otherwise, respond "VERIFIED".
        """
        
        reflection = self.llm.invoke(reflection_prompt)
        
        # 3. Revise if needed
        if "VERIFIED" not in reflection:
            revised_response = self.generate_constrained_response(
                query, retrieved_chunks, reflection
            )
            return revised_response
            
        return initial_response
    
    def generate_constrained_response(self, query, chunks, issues):
        """Generate response with stricter constraints based on reflection"""
        constrained_prompt = f"""
        {self.system_prompt}
        
        The previous response had these issues: {issues}
        
        Generate a new response that ONLY uses information explicitly stated in these sources:
        {[chunk.page_content for chunk in chunks]}
        
        Question: {query}
        """
        
        return self.llm.invoke(constrained_prompt)
```

### Week 2: Production Deployment & WordPress Integration

#### Day 7-8: Production-Ready API with Caching
**Goal: Fast, scalable API with intelligent caching**

```python
# src/api/main.py - Production API with advanced features
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import redis
import hashlib
import json
from typing import Optional
import asyncio

app = FastAPI(title="YonEarth Gaia Chat API")

# Initialize components
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
hybrid_retriever = HybridRAGRetriever()
hyde_search = HyDESearch(llm, hybrid_retriever)
multi_query = MultiQueryRetriever(llm, hybrid_retriever)
gaia = GaiaWithSelfReflection()

# Build BM25 index on startup
@app.on_event("startup")
async def startup_event():
    """Load documents and build BM25 index"""
    documents = load_processed_documents()  # Load your chunks
    hybrid_retriever.build_bm25_index(documents)

# Intelligent caching system
class SmartCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl_common = 86400  # 24 hours for common queries
        self.ttl_specific = 3600  # 1 hour for specific queries
        
    def get_cache_key(self, query: str) -> str:
        return f"gaia:response:{hashlib.md5(query.encode()).hexdigest()}"
    
    def should_cache_long(self, query: str) -> bool:
        """Determine if query should be cached longer"""
        common_patterns = [
            "what is", "how to", "tell me about", 
            "regenerative", "sustainable", "permaculture"
        ]
        return any(pattern in query.lower() for pattern in common_patterns)

cache = SmartCache(redis_client)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("message", "")
    use_hyde = data.get("use_hyde", False)
    
    # Check cache
    cache_key = cache.get_cache_key(query)
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Analyze query
    query_analysis = hybrid_retriever.query_analyzer.analyze_query(query)
    
    # Log search strategy for monitoring
    logger.info(f"Query: {query[:50]}... Analysis: {query_analysis}")
    
    # Determine retrieval strategy based on query analysis
    if query_analysis['suggested_method'] == 'keyword_heavy':
        # Adjust weights for keyword-heavy search
        hybrid_retriever.keyword_weight = 0.7
        hybrid_retriever.semantic_weight = 0.3
    elif query_analysis['suggested_method'] == 'semantic_heavy':
        # Adjust weights for semantic-heavy search
        hybrid_retriever.keyword_weight = 0.3
        hybrid_retriever.semantic_weight = 0.7
    else:
        # Balanced weights
        hybrid_retriever.keyword_weight = 0.5
        hybrid_retriever.semantic_weight = 0.5
    
    # Retrieve chunks
    if len(query.split()) > 15 and not query_analysis['has_episode_ref']:
        # Complex conceptual query - use HyDE
        chunks = await asyncio.to_thread(hyde_search.search_with_hyde, query)
    elif any(word in query.lower() for word in ["compare", "difference", "versus"]):
        # Comparison query - use multi-query
        chunks = await asyncio.to_thread(multi_query.retrieve_with_multi_query, query)
    else:
        # Default - use hybrid search
        chunks = await asyncio.to_thread(hybrid_retriever.hybrid_search, query)
    
    # Generate response with self-reflection
    response = await asyncio.to_thread(gaia.generate_with_reflection, query, chunks)
    
    # Add metadata
    response["metadata"] = {
        "chunks_retrieved": len(chunks),
        "episodes_referenced": list(set(chunk.metadata.get("episode_id") for chunk in chunks)),
        "search_method": query_analysis['suggested_method'],
        "has_episode_ref": query_analysis['has_episode_ref']
    }
    
    # Cache based on query type
    ttl = cache.ttl_common if cache.should_cache_long(query) else cache.ttl_specific
    redis_client.setex(cache_key, ttl, json.dumps(response))
    
    return response

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_db": "connected" if hybrid_retriever.vector_store else "disconnected",
        "cache": "connected" if redis_client.ping() else "disconnected"
    }
```

#### Day 9-10: Enhanced Render Deployment
**Goal: Production deployment with monitoring**

```yaml
# render.yaml - Enhanced production blueprint
services:
  - type: web
    name: yonearth-gaia-chat
    env: python
    buildCommand: |
      pip install -r requirements.txt
      python -m spacy download en_core_web_sm
      python -m nltk.downloader punkt stopwords
      python scripts/download_models.py
    startCommand: "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 2"
    healthCheckPath: /health
    scaling:
      minInstances: 1
      maxInstances: 3
      targetMemoryPercent: 80
      targetCPUPercent: 80
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: PINECONE_API_KEY
        sync: false
      - key: PINECONE_ENVIRONMENT
        value: gcp-starter
      - key: REDIS_URL
        fromService:
          name: yonearth-redis
          type: redis
          property: connectionString
      - key: MODEL_CACHE_DIR
        value: /opt/render/project/.cache

  - type: redis
    name: yonearth-redis
    plan: starter
    maxmemoryPolicy: allkeys-lru

  - type: cron
    name: cache-warmer
    env: python
    schedule: "0 */6 * * *"  # Every 6 hours
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python scripts/warm_cache.py"
```

#### Day 11: Advanced WordPress Integration
**Goal: Smart WordPress plugin with performance optimization**

```php
<?php
/**
 * Plugin Name: YonEarth Gaia Chat - Advanced RAG
 * Description: Chat with Gaia using state-of-the-art retrieval
 * Version: 2.0
 */

class YonEarthGaiaChat {
    private $api_endpoint;
    private $cache_duration = 3600;
    
    public function __construct() {
        $this->api_endpoint = get_option('yonearth_api_url', 'https://yonearth-gaia-chat.onrender.com');
        add_action('wp_enqueue_scripts', array($this, 'enqueue_scripts'));
        add_shortcode('yonearth_chat', array($this, 'render_chat_widget'));
    }
    
    public function enqueue_scripts() {
        wp_enqueue_script(
            'yonearth-chat-widget',
            plugin_dir_url(__FILE__) . 'assets/chat-widget.js',
            array(),
            '2.0.0',
            true
        );
        
        // Pass configuration to JavaScript
        wp_localize_script('yonearth-chat-widget', 'yonEarthChat', array(
            'apiUrl' => $this->api_endpoint,
            'nonce' => wp_create_nonce('yonearth-chat'),
            'cacheEnabled' => get_option('yonearth_enable_cache', true),
            'useHyde' => get_option('yonearth_use_hyde', false)
        ));
    }
    
    public function render_chat_widget($atts) {
        $atts = shortcode_atts(array(
            'height' => '600px',
            'width' => '100%',
            'theme' => 'earth',
            'show_sources' => 'true',
            'enable_voice' => 'false'
        ), $atts);
        
        return sprintf(
            '<div id="yonearth-chat-container" 
                 data-height="%s" 
                 data-width="%s" 
                 data-theme="%s"
                 data-show-sources="%s"
                 data-enable-voice="%s">
                <noscript>Please enable JavaScript to chat with Gaia.</noscript>
            </div>',
            esc_attr($atts['height']),
            esc_attr($atts['width']),
            esc_attr($atts['theme']),
            esc_attr($atts['show_sources']),
            esc_attr($atts['enable_voice'])
        );
    }
}

new YonEarthGaiaChat();
```

```javascript
// assets/chat-widget.js - Advanced frontend with performance optimization
class YonEarthChatWidget {
    constructor(container) {
        this.container = container;
        this.apiUrl = yonEarthChat.apiUrl;
        this.messageHistory = [];
        this.pendingQuery = null;
        this.initializeWidget();
    }
    
    async sendMessage(message) {
        // Show typing indicator
        this.showTypingIndicator();
        
        // Prepare request with smart defaults
        const requestData = {
            message: message,
            use_hyde: message.split(' ').length > 15, // Use HyDE for complex queries
            session_id: this.getSessionId()
        };
        
        try {
            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-WP-Nonce': yonEarthChat.nonce
                },
                body: JSON.stringify(requestData)
            });
            
            const data = await response.json();
            
            // Display response with citations
            this.displayResponse(data);
            
            // Cache in localStorage for offline access
            if (yonEarthChat.cacheEnabled) {
                this.cacheResponse(message, data);
            }
            
        } catch (error) {
            this.handleError(error);
        } finally {
            this.hideTypingIndicator();
        }
    }
    
    displayResponse(data) {
        const { answer, citations, metadata } = data;
        
        // Create response element with citations
        const responseEl = document.createElement('div');
        responseEl.className = 'gaia-response';
        responseEl.innerHTML = `
            <div class="response-text">${this.formatAnswer(answer)}</div>
            ${this.formatCitations(citations)}
            <div class="response-meta">
                <span>Drew from ${metadata.episodes_referenced.length} episodes</span>
                <span>Method: ${metadata.retrieval_method}</span>
            </div>
        `;
        
        this.container.appendChild(responseEl);
    }
}
```

#### Day 12-13: Testing & Performance Optimization
**Goal: Ensure production readiness**

```python
# tests/test_rag_accuracy.py
import pytest
from src.rag.hybrid_retriever import HybridRAGRetriever
from src.evaluation.metrics import calculate_mrr, calculate_ndcg

class TestRAGAccuracy:
    @pytest.fixture
    def retriever(self):
        return HybridRAGRetriever()
    
    def test_hybrid_search_accuracy(self, retriever):
        """Test that hybrid search outperforms single-method search"""
        test_queries = [
            "What did the guest say about regenerative agriculture?",
            "Episode 147 permaculture techniques",
            "How to start composting at home"
        ]
        
        for query in test_queries:
            # Test hybrid search
            hybrid_results = retriever.hybrid_search(query, k=10)
            
            # Test semantic only
            semantic_results = retriever.vector_store.similarity_search(query, k=10)
            
            # Verify hybrid has better diversity
            hybrid_episodes = set(r.metadata['episode_id'] for r in hybrid_results)
            semantic_episodes = set(r.metadata['episode_id'] for r in semantic_results)
            
            assert len(hybrid_episodes) >= len(semantic_episodes)
    
    def test_reranking_improves_relevance(self, retriever):
        """Verify reranking improves result relevance"""
        query = "sustainable water management techniques"
        
        # Get results without reranking
        initial_results = retriever.vector_store.similarity_search(query, k=20)
        
        # Apply reranking
        reranked_results = retriever.rerank_results(query, initial_results)
        
        # Manual verification that top results are more relevant
        # In production, use labeled test set
        assert len(reranked_results) > 0
```

#### Day 14: Launch & Documentation
**Goal: Production launch with comprehensive docs**

```markdown
# YonEarth Gaia Chat - Deployment Guide

## Quick Start
1. Clone repository
2. Set environment variables in Render dashboard:
   - OPENAI_API_KEY
   - PINECONE_API_KEY
3. Deploy using render.yaml blueprint
4. Install WordPress plugin
5. Add shortcode to any page: [yonearth_chat]

## Advanced Configuration

### Retrieval Methods
- **Hybrid Search**: Best for most queries (default)
- **HyDE**: Automatically enabled for complex questions
- **Multi-Query**: Activated for comparison queries

### Performance Tuning
- Redis cache: 24hr for common queries, 1hr for specific
- Reranking: Top 20 candidates reranked to top 10
- BM25 + Semantic: Reciprocal Rank Fusion for best results

### Monitoring
- Health endpoint: /health
- Metrics tracked: retrieval accuracy, latency, cache hits
- Episode coverage: Ensures diverse source citations
```

## Post-MVP Roadmap: Scaling Advanced RAG

### Month 2: Full Dataset & Optimization
**Week 3-4: Scale to 172 Episodes**
- Implement streaming ingestion for large dataset
- Add speaker-aware retrieval indexes
- Optimize chunk sizes based on A/B tests
- Implement query routing based on intent classification

**Week 5-6: Advanced Features**
- Add conversational memory with conversation-aware retrieval
- Implement feedback loop for retrieval improvement
- Add multilingual support for broader reach
- Create admin dashboard for monitoring RAG performance

### Month 3: Enterprise Features
**Week 7-8: Production Enhancements**
- Implement RLHF for Gaia personality refinement
- Add real-time episode ingestion pipeline
- Create knowledge graph for entity relationships
- Build recommendation engine for episode discovery

## Success Metrics

### RAG Performance Metrics
1. **MRR@10**: > 0.85 (retrieved chunks contain answer)
2. **BM25 precision**: > 0.90 for keyword queries
3. **Latency**: < 2s for 95th percentile
4. **Cache hit rate**: > 60% for common queries
5. **Reranking improvement**: > 15% relevance gain
6. **Source diversity**: Average 3+ episodes per response

### System Performance
1. **BM25 index build**: < 2 minutes for 172 episodes
2. **Keyword search time**: < 100ms per query
3. **Memory usage**: < 500MB for complete system
4. **Startup time**: < 30 seconds

### Business Metrics
1. **User engagement**: 70% ask follow-up questions
2. **Episode discovery**: 40% click through to full episodes
3. **Query satisfaction**: 85% positive feedback
4. **Cost efficiency**: < $0.02 per query with caching

## Key Differentiators

### 3. **BM25 Performance Benefits**

Using standard BM25 provides:
- **Battle-tested algorithm**: Proven effectiveness across millions of use cases
- **Fast implementation**: < 100ms query time even with thousands of documents
- **Automatic handling** of term frequency, document length normalization, and IDF
- **Simple integration**: Just 3-4 lines of code to implement

Example search flow:
- Query: "tell me about biochar"
- BM25 automatically finds documents with "biochar", ranks by frequency + IDF
- Combined with semantic search for comprehensive results
- Reranker ensures best results appear first
1. **State-of-the-art hybrid search** combining semantic and keyword retrieval
2. **Cross-encoder reranking** for superior relevance
3. **HyDE** for complex query understanding
4. **Self-reflective RAG** to prevent hallucinations
5. **Intelligent caching** based on query patterns
6. **Production-ready WordPress integration** with performance optimization

The result is an enterprise-grade RAG system that delivers accurate, source-cited responses while maintaining Gaia's authentic voice and the YonEarth community's wisdom.

## Recent Implementation Updates (July 2025)

### User Feedback System (Completed 2025-07-17)
**Goal: Collect user feedback to improve search quality**

**Implementation Details:**
1. **Frontend Components**:
   - Quick feedback: Thumbs up/down buttons below each response
   - Detailed feedback: 5-star rating, "correct episodes" checkbox, text comments
   - Integrated into chat.js with persistent localStorage backup
   - Clean UI design with Earth-themed styling

2. **Backend Integration**:
   - `/feedback` endpoint in both main.py and simple_server.py
   - JSON file storage organized by date
   - Comprehensive data collection (query, response, citations, ratings)
   - Error-resilient design with fallback storage

3. **Analysis Tools**:
   - `scripts/view_feedback.py` for feedback analysis
   - Summary statistics (ratings, correctness percentage, type distribution)
   - Detailed feedback viewing with filtering options

**Impact**: Enables rapid iteration on search quality based on real user feedback