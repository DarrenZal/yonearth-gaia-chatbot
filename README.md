# KGC - Knowledge Graph Construction 📚

> Production-ready system for extracting high-quality knowledge graphs from books and documents

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

## 🎯 Current Achievement

**Grade:** A+ (Quality Gate Passed!)
- **0 CRITICAL issues**
- **≤2 HIGH issues**
- **≤2% issue rate**
- **116 high-quality relationships** from 30 pages
- **93% improvement** from baseline

## ⚡ Quick Start

```bash
# Clone and install
git clone https://github.com/DarrenZal/KGC.git
cd KGC
pip install -r requirements.txt

# Extract knowledge graph
export OPENAI_API_KEY="your-key-here"
python scripts/extract_kg_v14_3_8_incremental.py \
  --book our_biggest_deal \
  --section front_matter \
  --pages 1-30 \
  --author "Aaron William Perry"
```

## 🌟 Key Features

### 🧠 **Advanced Category-First RAG System**
- **✨ Semantic Category Matching**: TRUE semantic understanding using OpenAI embeddings
  - Solves "soil" → BIOCHAR matching automatically (32.1% similarity)
  - Cached embeddings for performance (`/data/processed/category_embeddings.json`)
  - User-configurable thresholds: Broad (0.6), Normal (0.7), Strict (0.8), Disabled (1.1)
- **Episode Diversity Algorithm**: Ensures all relevant episodes appear, not just one with many chunks
- **Dual Search Methods**: 
  - 🌿 **Original (Semantic Search)**: Meaning-based context understanding
  - 🔍 **BM25 (Category-First Hybrid)**: Category matching (80%) + semantic (15%) + keyword (5%)
  - ⚖️ **Side-by-Side Comparison**: Compare both methods simultaneously
- **Guaranteed Category Matches**: ALL episodes tagged with matching categories appear in results
- **Multi-Content Integration**: Search across both podcast episodes AND books simultaneously
- **Accurate Citations**: No more hallucinated episode references
- **Smart Recommendations**: Dynamic suggestions from episodes and books based on conversation context

### 🌱 **Gaia Character Personalities**
- **🤱 Nurturing Mother**: Warm, caring, and patient guidance
- **🧙‍♂️ Ancient Sage**: Deep wisdom from Earth's timeless perspective
- **⚡ Earth Guardian**: Passionate activist for ecological justice
- **✨ Custom**: Create your own personalized Gaia with custom system prompts

### 🎯 **Smart Conversation Features**
- **Conversation-Aware Recommendations**: Episodes update based on topics discussed
- **Topic Tracking**: Automatically extracts and follows conversation themes
- **Duplicate Prevention**: Clean, non-redundant episode suggestions
- **Context Evolution**: Recommendations improve as conversations develop
- **User Feedback System**: Collect feedback on response quality with ratings and detailed comments
- **Cost Tracking**: Transparent breakdown of API costs for each response

### 🎤 **Voice Integration**
- **Text-to-Speech**: Hear Gaia's responses spoken with ElevenLabs AI voice technology
- **Custom Voice**: Uses a specially cloned voice for authentic, natural speech
- **Toggle Control**: Enable/disable voice with a simple speaker button
- **Auto-play**: Responses automatically play when voice is enabled
- **Browser Support**: Works across all modern browsers
- **Persistent Preference**: Voice settings saved in browser localStorage

### 🚀 **Production Ready**
- **Docker Deployment**: One-command setup with nginx, Redis, web interface
- **Beautiful Web UI**: Responsive chat interface with personality selection
- **Dual API Endpoints**: Both original and BM25 search methods available
- **Rate Limiting**: Prevent abuse with configurable limits
- **Health Monitoring**: Built-in health checks and logging
- **SSL Ready**: Easy HTTPS setup with Let's Encrypt

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Hybrid Search   │───▶│   Gaia LLM      │
│ "what is        │    │ BM25 + Semantic  │    │   Response      │
│  biochar?"      │    │ + Reranking      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                       ┌────────▼────────┐               │
                       │ Episode Sources │               │
                       │ 120: High Plains│               │
                       │ 122: Dr. Laird  │               │
                       │ 165: Kelpie W.  │               │
                       └─────────────────┘               │
                                                         ▼
                              ┌─────────────────────────────────────┐
                              │ "Biochar is a carbon-rich material  │
                              │ created through pyrolysis...        │
                              │                                     │
                              │ Referenced Episodes:                │
                              │ • Episode 120: High Plains Biochar │
                              │ • Episode 122: Dr. David Laird     │
                              │                                     │
                              │ Recommended Episodes:               │
                              │ Based on our conversation about:    │
                              │ biochar, carbon, farming            │
                              └─────────────────────────────────────┘
```

## 🎯 How It Works

### Dual Search Methods
1. **🌿 Original (Semantic Search)**: Uses OpenAI embeddings for meaning-based search
2. **🔍 BM25 (Category-First Hybrid)**: Episode categorization table (80%) + semantic (15%) + keyword (5%) with cross-encoder reranking
3. **⚖️ Comparison Mode**: See both methods side-by-side to compare approaches

### Smart Recommendations
- **Inline Citations**: See which episodes and books Gaia referenced for each response
- **Cumulative References**: "Recommended Content" section shows ALL references from entire conversation
- **Dynamic Recommendations**: Bottom section updates based on your conversation topics
- **Context Awareness**: "Based on our conversation about: permaculture, soil health..."
- **Related Suggestions**: "Try asking about: other episodes on composting"

### Custom Personalities
- Select from 3 pre-built personalities or create your own
- Custom prompts stored in browser localStorage
- Use any existing personality as a template for editing
- Works with both search methods

### 📝 User Feedback System
- **Quick Feedback**: Simple thumbs up/down buttons for each response
- **Detailed Feedback**: Optional detailed feedback with:
  - 5-star relevance rating
  - "Were the right episodes included?" checkbox
  - Free-text feedback area
- **Data Collection**: Feedback stored in JSON files for analysis
- **Persistent Storage**: Saved both locally and on server
- **Analysis Tools**: Script to view and analyze collected feedback

### 📚 Integrated Books
Currently includes three books by Aaron William Perry:
- **VIRIDITAS: THE GREAT HEALING** - Exploration of viriditas (life force) and regenerative practices
- **Soil Stewardship Handbook** - Practical guide for soil regeneration, composting, and biochar
- **Y on Earth: Get Smarter, Feel Better, Heal the Planet** - Sustainable living, mindfulness, and community connection

#### Book Integration Features
- **PDF Processing**: Automatically extracts and processes PDF books from `/data/books`
- **Chapter Detection**: Intelligent parsing of chapter boundaries and titles
- **Unified Search**: Books and podcast episodes searched together seamlessly
- **Enhanced Citations**: Shows book title, author, and chapter information with clickable links
- **Multiple Formats**: Links to eBook, audiobook, and print versions when available
- **Optimized Chunking**: Larger chunks for books (750 tokens) vs episodes (500 tokens)

## 🚀 Deployment Options

### Option 1: VPS Docker Deployment (Recommended)
```bash
# Clone and deploy in one command
git clone https://github.com/DarrenZal/yonearth-gaia-chatbot.git
cd yonearth-gaia-chatbot
./deploy.sh
```

**Includes**: Docker, nginx, Redis, SSL certificates, monitoring

### Option 2: Render.com (Cloud)
1. Fork this repository
2. Connect to Render using `render.yaml` blueprint
3. Add API keys as environment variables
4. Deploy!

### Option 3: Local Development
```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
uvicorn src.api.main:app --reload
```

## 🔑 Required API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Pinecone API Key**: Get from [Pinecone Console](https://app.pinecone.io/)
3. **ElevenLabs API Key** (Optional): Get from [ElevenLabs Console](https://elevenlabs.io/) for voice features

Create a Pinecone index:
- **Name**: `yonearth-episodes`
- **Dimensions**: `1536`
- **Metric**: `cosine`

## 🧪 Testing

### Web Interface
Visit your deployment URL and try these category-first queries:
- "What is biochar?" (should reference Episodes 120, 122, 165 - ALL BIOCHAR category episodes)
- "herbal medicine" (should reference Episodes 19, 108, 90, 115, 98 - ALL HERBAL MEDICINE category episodes)
- "Tell me about regenerative agriculture"
- "How can I start composting?"
- "What is the significance of chlorophyll and hemoglobin?" (references VIRIDITAS book)
- "What are soil building parties?" (references Soil Stewardship Handbook)
- "How can I live more sustainably?" (references Y on Earth book)

**🎯 Category Testing**: Use **🔍 BM25 Hybrid Search** mode to test category-first functionality

### API Testing
```bash
# Test Original search method
curl -X POST http://YOUR_SERVER_IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "what is biochar?",
    "max_results": 5,
    "session_id": "test",
    "personality": "warm_mother"
  }'

# Test BM25 hybrid search
curl -X POST http://YOUR_SERVER_IP:8000/api/bm25/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "what is biochar?",
    "search_method": "hybrid",
    "k": 5,
    "gaia_personality": "wise_guide"
  }'

# Test chat with voice generation
curl -X POST http://YOUR_SERVER_IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about composting",
    "enable_voice": true,
    "max_citations": 3,
    "personality": "warm_mother"
  }'

# Test voice endpoint
curl http://YOUR_SERVER_IP:8000/api/voice/test
```

## 📊 API Endpoints

### Original RAG System
```bash
POST /chat
{
  "message": "Tell me about regenerative agriculture",
  "max_results": 5,
  "session_id": "optional",
  "personality": "warm_mother",
  "custom_prompt": "optional custom system prompt",
  "enable_voice": false  # Enable voice generation
}
```

### BM25 Hybrid System
```bash
POST /bm25/chat
{
  "message": "Tell me about soil health",
  "search_method": "hybrid",  # auto, bm25, semantic, hybrid
  "k": 5,
  "gaia_personality": "earth_activist",
  "custom_prompt": "optional custom system prompt",
  "enable_voice": true,  # Enable voice generation
  "category_threshold": 0.7  # Semantic category matching threshold
}
```

### Search Comparison
```bash
POST /bm25/compare-methods
{
  "query": "permaculture techniques",
  "k": 5
}
```

### User Feedback
```bash
POST /feedback
{
  "messageId": "msg-123456789",
  "timestamp": "2025-07-17T17:40:47.113Z",
  "type": "detailed",  # helpful, not-helpful, detailed
  "query": "what is biochar?",
  "response": "Biochar is a carbon-rich material...",
  "citations": [...],
  "sessionId": "session_abc123",
  "personality": "warm_mother",
  "ragType": "bm25",
  "modelType": "gpt-3.5-turbo",
  "relevanceRating": 5,  # 1-5 stars
  "episodesCorrect": true,
  "detailedFeedback": "Great response with accurate citations!"
}
```

### Episode Search
```bash
POST /search
{
  "query": "permaculture techniques",
  "max_results": 10,
  "filters": {"guest_name": "specific_guest"}
}
```

### Voice Test
```bash
GET /api/voice/test
# Returns voice client status and test generation result
```

## ⚙️ Configuration

Key environment variables:
```bash
# Required
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here

# Optional Voice Features
ELEVENLABS_API_KEY=your_key_here       # For voice generation
ELEVENLABS_VOICE_ID=your_voice_id      # Custom voice ID
ELEVENLABS_MODEL_ID=eleven_multilingual_v2  # Voice model

# Optional Configuration
GAIA_PERSONALITY_VARIANT=warm_mother  # warm_mother, wise_guide, earth_activist
GAIA_TEMPERATURE=0.7                   # 0.0-1.0 (accuracy vs creativity)
EPISODES_TO_PROCESS=172                # Number of episodes to index
ALLOWED_ORIGINS=https://your-domain.com
RATE_LIMIT_PER_MINUTE=20

# Book Processing
CHUNK_SIZE=500                         # Base chunk size for episodes
CHUNK_OVERLAP=50                       # Overlap between chunks
# Books use CHUNK_SIZE + 250 (750 tokens) and CHUNK_OVERLAP + 50 (100 tokens)
```

## 🔧 Management

```bash
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Update application
git pull origin main
docker-compose build app
docker-compose up -d app

# Monitor resources
docker stats
```

## 📈 Performance

- **Response Time**: < 2 seconds for most queries
- **Search Accuracy**: 99%+ episode citation accuracy with BM25 hybrid
- **Scalability**: Redis caching, multi-worker support
- **Uptime**: Health checks, auto-restart, systemd integration
- **Conversation Intelligence**: Dynamic recommendations based on chat context

## 🌍 Live Demo Results

Try these queries that demonstrate the system's accuracy:

**Query**: "What is biochar?"
- **Before (broken)**: Referenced Episode 111 (no biochar content)
- **After (fixed)**: References Episodes 120, 122, 165 (actual biochar episodes)
- **Category-First Search**: Prioritizes ALL episodes tagged with BIOCHAR category (80% weight)

**Query**: "herbal medicine"
- **Category-First Results**: Episodes 19 (Brigitte Mars), 108 (Ann Armbrecht), 90 (Vera Herbals), 115 (Y on Earth), 98 (Biodynamic)
- **Guaranteed Coverage**: ALL 18 episodes tagged with HERBAL MEDICINE category appear in results
- **Smart Ranking**: Category matches (80%) + semantic relevance (15%) + keyword matching (5%)

**Query**: "What is the significance of chlorophyll and hemoglobin?"
- **Multi-Content Search**: References both podcast episodes AND book content from VIRIDITAS
- **Book Integration**: Shows chapter-specific citations with author information and clickable links

### Content Database
- **172 Podcast Episodes**: 14,475+ searchable chunks with full transcripts
- **3 Books**: 4,289+ searchable chunks with chapter-level citations
  - VIRIDITAS: THE GREAT HEALING (2,029 chunks)
  - Soil Stewardship Handbook (136 chunks)
  - Y on Earth (2,124 chunks)
- **Total Vectors**: 18,764+ indexed for hybrid search

## 🎨 Web Interface Features

- **🌿 Personality Selection**: Choose Gaia's voice and communication style
- **⚙️ Search Method Toggle**: Switch between Original, BM25, or Both
- **🔊 Voice Toggle**: Enable/disable text-to-speech for Gaia's responses
- **📋 Smart Recommendations**: Dynamic episode suggestions
- **💬 Conversation Memory**: Gaia remembers your chat context
- **✨ Custom Prompts**: Create personalized Gaia personalities
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **🎤 Audio Controls**: Automatic playback with manual replay option

## 🧠 Technical Innovation

### Category-First Hybrid RAG Architecture
- **Episode Categorization**: CSV-driven topic classification as PRIMARY search guide (80% weight)
- **Category-First Fusion**: Guarantees ALL category-matching episodes appear in results
- **BM25 Keyword Search**: Finds episodes that actually contain search terms
- **Semantic Vector Search**: Understanding context and meaning  
- **Reciprocal Rank Fusion**: Intelligently combines category + keyword + semantic results
- **Cross-Encoder Reranking**: Final relevance scoring for optimal results
- **Query-Adaptive Strategy**: Automatically chooses best search approach
- **Multi-Content Integration**: Seamlessly searches across podcast episodes AND books

### Book Processing Pipeline
- **PDF Text Extraction**: Advanced processing using pdfplumber for clean text extraction
- **Chapter Detection**: Intelligent regex-based chapter boundary detection
- **Optimized Chunking**: Larger chunks (750 tokens) for book content vs episodes (500 tokens)
- **Metadata Preservation**: Author, title, chapter information maintained throughout pipeline
- **Unified Vector Storage**: Books and episodes stored together in same vector database

### Conversation Intelligence
- **Topic Extraction**: Automatically identifies conversation themes
- **Episode Tracking**: Remembers which episodes were discussed
- **Dynamic Context**: Recommendations evolve with conversation
- **Duplicate Prevention**: Clean, non-redundant suggestions
- **Cross-Content Citations**: References both podcast episodes and book chapters

## 📚 Documentation

For detailed documentation, visit the [`docs/` folder](docs/):

- **[🚀 VPS Deployment Guide](docs/VPS_DEPLOYMENT.md)** - Complete step-by-step deployment
- **[🔧 Development Guide](CLAUDE.md)** - Technical architecture and development
- **[🎤 Voice Integration Guide](docs/VOICE_INTEGRATION.md)** - ElevenLabs TTS setup and usage
- **[💰 Cost Tracking Guide](docs/COST_TRACKING.md)** - API usage cost transparency
- **[📋 Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** - BM25 system development history

## 🚧 Roadmap & Upcoming Features

### High Priority
- **✅ Content Categorization**: ✅ COMPLETED - Category-first search with 170 episodes categorized across 28 topics
- **📊 Recommended Content Alignment**: Ensure recommended episodes precisely match referenced sources
- **✅ Cost Calculator**: ✅ COMPLETED - Track and display response generation costs for budget management

### Medium Priority  
- **✅ Voice Integration**: ✅ COMPLETED - Voice responses using ElevenLabs API with custom voice
- **🔗 Knowledge Graph Links**: Implement hyperlinks within responses linking to YonEarth resources and related content
- **✅ Max References Setting**: ✅ COMPLETED - Configurable limit for maximum episode/book references per response

### Long Term
- **🧠 Advanced Knowledge Graph**: Create interconnected content relationships for deeper context discovery
  - Neo4j graph database integration for relationship traversal queries
  - JSON-LD/RDF format for structured data interoperability
  - LLM-enhanced wiki generation: Wikipedia-style articles combining structured data with narrative text from source material
  - Automated wiki page creation using entity data + original context chunks for rich, engaging content
- **📱 Mobile App**: Native mobile application for better mobile experience
- **🔍 Advanced Search Filters**: Filter by guest, topic, date range, and content type
- **📈 Analytics Dashboard**: Usage statistics and popular topics tracking

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **YonEarth Community**: For the incredible podcast content and regenerative wisdom
- **OpenAI**: GPT models and embeddings API
- **Pinecone**: Vector database infrastructure
- **FastAPI**: Modern Python API framework
- **LangChain**: RAG pipeline components

---

**Ready to chat with Gaia?** 🌱 Deploy your chatbot and start exploring Earth's wisdom through the power of hybrid search and conversation intelligence!