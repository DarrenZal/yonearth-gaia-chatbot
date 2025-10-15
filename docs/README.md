# YonEarth Gaia Chatbot Documentation

Welcome to the comprehensive documentation for the YonEarth Gaia Chatbot project.

---

## 📚 Documentation Index

### Getting Started

**[Setup and Deployment Guide](SETUP_AND_DEPLOYMENT.md)**
- Docker quick start
- Local development setup
- VPS production deployment
- Environment configuration
- Service management
- Troubleshooting

### Using the Chatbot

**[Features and Usage Guide](FEATURES_AND_USAGE.md)**
- Search methods (Original, BM25, Both)
- Personality system (predefined + custom)
- Voice integration
- Multi-content search (episodes + books)
- Smart recommendations
- Feedback system
- Cost tracking
- Best practices

### Technical Documentation

**[Architecture Overview](ARCHITECTURE.md)** *(Coming Soon)*
- System architecture
- RAG pipeline details
- Database schema
- API design
- Frontend/backend interaction

**[Development Guide](DEVELOPMENT.md)** *(Coming Soon)*
- Development workflow
- Code structure
- Adding features
- Testing
- API reference

**[Content Processing](CONTENT_PROCESSING_PIPELINE.md)**
- Episode processing workflow
- Book processing workflow
- Knowledge graph extraction
- Data pipeline management

**[Knowledge Graph System](knowledge_graph/README.md)** ⭐ **NEW**
- Complete v3.2.2 extraction system
- 172 episodes, 45,478 relationships
- Three-stage pipeline (Extract → Validate → Score)
- Production-ready visualization

### Specialized Topics

**[Voice Integration](VOICE_INTEGRATION.md)**
- ElevenLabs setup
- Voice configuration
- Text preprocessing
- Audio playback

**[Cost Tracking](COST_TRACKING.md)**
- Cost calculation methods
- API usage tracking
- Cost optimization strategies

**[Episode Coverage](EPISODE_COVERAGE.md)**
- Complete episode list (0-172)
- Scraping methodology
- Data quality notes

**[VPS Deployment](VPS_DEPLOYMENT.md)**
- Production server setup
- Nginx configuration
- Systemd service management
- SSL/HTTPS setup

### Planning and TODOs

**[Remaining TODOs](REMAINING_TODOS.md)**
- Outstanding features
- Planned improvements
- Priority roadmap

---

## 🗂️ Project Structure

```
yonearth-gaia-chatbot/
├── docs/                           # 📚 Documentation (you are here)
│   ├── README.md                   # This file
│   ├── SETUP_AND_DEPLOYMENT.md     # Setup guide
│   ├── FEATURES_AND_USAGE.md       # Feature documentation
│   ├── CONTENT_PROCESSING_PIPELINE.md
│   ├── VOICE_INTEGRATION.md
│   ├── COST_TRACKING.md
│   ├── EPISODE_COVERAGE.md
│   ├── VPS_DEPLOYMENT.md
│   ├── REMAINING_TODOS.md
│   └── archive/                    # Historical docs
│
├── src/                            # 🔧 Source code
│   ├── api/                        # FastAPI endpoints
│   │   ├── main.py                 # Main API with original RAG
│   │   ├── bm25_endpoints.py       # BM25 hybrid search
│   │   ├── models.py               # Pydantic models
│   │   └── voice_endpoints.py      # Voice generation
│   │
│   ├── rag/                        # RAG systems
│   │   ├── chain.py                # Original RAG pipeline
│   │   ├── bm25_chain.py           # BM25 RAG pipeline
│   │   ├── hybrid_retriever.py     # Original hybrid search
│   │   ├── bm25_hybrid_retriever.py # BM25 + semantic + reranking
│   │   ├── semantic_category_matcher.py # Semantic category matching
│   │   └── vectorstore.py          # Pinecone wrapper
│   │
│   ├── character/                  # Gaia personality
│   │   ├── gaia.py                 # Character implementation
│   │   └── gaia_personalities.py   # Personality variants
│   │
│   ├── voice/                      # Voice system
│   │   └── elevenlabs_client.py    # TTS client
│   │
│   ├── ingestion/                  # Data processing
│   │   ├── episode_processor.py    # Episode ingestion
│   │   ├── book_processor.py       # Book ingestion
│   │   └── chunker.py              # Text chunking
│   │
│   └── config/                     # Configuration
│       └── settings.py             # Centralized settings
│
├── web/                            # 🌐 Web interface
│   ├── index.html                  # Chat UI
│   ├── chat.js                     # Frontend logic
│   └── styles.css                  # Styling
│
├── scripts/                        # 🛠️ Utility scripts
│   ├── start_local.py              # Start dev server
│   ├── test_api.py                 # Test endpoints
│   ├── view_feedback.py            # View user feedback
│   └── archive/                    # Historical scripts
│
├── data/                           # 📊 Data storage
│   ├── transcripts/                # Episode JSON files
│   ├── processed/                  # Processed data
│   ├── knowledge_graph/            # KG extractions (historical)
│   ├── knowledge_graph_v3_2_2/     # v3.2.2 extraction (45,478 relationships)
│   └── feedback/                   # User feedback
│
├── tests/                          # 🧪 Test suite
│
├── README.md                       # Main project README
├── CLAUDE.md                       # Claude Code instructions
├── requirements.txt                # Python dependencies
└── docker-compose.yml              # Docker configuration
```

---

## 🚀 Quick Links

### For New Users
1. **[Setup Guide](SETUP_AND_DEPLOYMENT.md)** - Get started
2. **[Features Guide](FEATURES_AND_USAGE.md)** - Learn what it can do
3. **[Main README](../README.md)** - Project overview

### For Developers
1. **[CLAUDE.md](../CLAUDE.md)** - Development commands
2. **[Content Processing](CONTENT_PROCESSING_PIPELINE.md)** - Data pipeline
3. **[VPS Deployment](VPS_DEPLOYMENT.md)** - Production setup

### For Contributors
1. **[Remaining TODOs](REMAINING_TODOS.md)** - What needs doing
2. **[Cost Tracking](COST_TRACKING.md)** - Understanding costs
3. **[Archive](archive/)** - Historical context

---

## 📖 Learning Path

### Beginner (Just Want to Use It)
1. **[Setup and Deployment](SETUP_AND_DEPLOYMENT.md)** - Get it running
2. **[Features and Usage](FEATURES_AND_USAGE.md)** - Learn the features
3. Start chatting!

### Intermediate (Want to Customize)
1. Read Beginner docs first
2. **[CLAUDE.md](../CLAUDE.md)** - Development environment
3. **[Main README](../README.md)** - Architecture overview
4. Experiment with personality customization

### Advanced (Want to Contribute)
1. Read Beginner + Intermediate docs
2. **[Content Processing](CONTENT_PROCESSING_PIPELINE.md)** - Data pipeline
3. **[VPS Deployment](VPS_DEPLOYMENT.md)** - Production deployment
4. **[Remaining TODOs](REMAINING_TODOS.md)** - Contribution opportunities
5. Review code in `src/` directory

---

## 🔍 Finding What You Need

### Common Questions

**"How do I install this?"**
→ [Setup and Deployment](SETUP_AND_DEPLOYMENT.md#quick-start-docker)

**"What can this chatbot do?"**
→ [Features and Usage](FEATURES_AND_USAGE.md#core-features)

**"How do I change Gaia's personality?"**
→ [Features: Personality System](FEATURES_AND_USAGE.md#gaia-personality-system)

**"How do I deploy to production?"**
→ [VPS Deployment](VPS_DEPLOYMENT.md)

**"How do I add more episodes?"**
→ [Content Processing](CONTENT_PROCESSING_PIPELINE.md#episode-processing-workflow)

**"What's the system architecture?"**
→ [Main README: Architecture](../README.md#architecture)

**"How much does this cost to run?"**
→ [Cost Tracking](COST_TRACKING.md)

**"Can users hear Gaia speak?"**
→ [Voice Integration](VOICE_INTEGRATION.md)

**"What features are planned?"**
→ [Remaining TODOs](REMAINING_TODOS.md)

---

## 📂 Archive

The `archive/` folder contains historical documentation:
- Completed implementation plans
- Agent reports from development
- Old integration guides
- Completed status reports

These are kept for reference but are no longer actively maintained.

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Read [Remaining TODOs](REMAINING_TODOS.md) for current priorities
2. Review [CLAUDE.md](../CLAUDE.md) for development setup
3. Test your changes locally
4. Submit pull request with description

---

## 📞 Getting Help

- **General questions**: Check this documentation index
- **Setup issues**: [Setup Guide Troubleshooting](SETUP_AND_DEPLOYMENT.md#troubleshooting)
- **Feature requests**: See [Remaining TODOs](REMAINING_TODOS.md)
- **Bug reports**: Open an issue on GitHub

---

**Last Updated**: October 4, 2025

*Documentation maintained by the YonEarth Gaia Chatbot team*
