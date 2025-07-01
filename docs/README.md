# YonEarth Gaia Chatbot Documentation

This folder contains detailed documentation for the YonEarth Gaia Chatbot project.

## 📚 Documentation Index

### Deployment & Setup
- **[VPS Deployment Guide](VPS_DEPLOYMENT.md)** - Complete step-by-step VPS deployment with Docker
- **[Quick Start](../README.md#-quick-deployment)** - One-command deployment instructions

### Development
- **[Claude Code Guide](CLAUDE.md)** - Technical documentation for Claude Code development
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Original BM25 implementation plan and progress

### Architecture & Features
- **[Main README](../README.md)** - Overview, features, and architecture
- **[API Documentation](../README.md#-api-endpoints)** - Endpoint specifications and examples

## 🔧 Development Workflow

1. **Local Development**: See [CLAUDE.md](CLAUDE.md) for development commands and architecture
2. **Testing**: API testing examples in [main README](../README.md#-testing)
3. **Deployment**: Production deployment via [VPS_DEPLOYMENT.md](VPS_DEPLOYMENT.md)

## 🏗️ Project Structure

```
yonearth-gaia-chatbot/
├── docs/                    # 📚 Documentation
│   ├── README.md           # This file
│   ├── VPS_DEPLOYMENT.md   # Deployment guide
│   ├── CLAUDE.md           # Development guide
│   └── IMPLEMENTATION_PLAN.md # BM25 implementation plan
├── src/                    # 🔧 Source code
│   ├── api/               # FastAPI endpoints
│   ├── rag/               # RAG systems (Original + BM25)
│   ├── character/         # Gaia personality system
│   ├── ingestion/         # Data processing
│   └── config/            # Configuration
├── web/                   # 🌐 Web interface
│   ├── index.html         # Chat UI
│   ├── chat.js           # Frontend logic
│   └── styles.css        # Styling
├── tests/                 # 🧪 Test suite
└── scripts/              # 🛠️ Utility scripts
```

## 🚀 Quick Links

- **Live Demo**: http://152.53.194.214:8000
- **Deploy Your Own**: `git clone` → `./deploy.sh` → Add API keys
- **API Docs**: Visit `/docs` on your deployment
- **Issues & Contributing**: See main [README](../README.md#-contributing)

## 📖 Learning Path

1. **Start Here**: [Main README](../README.md) for overview and quick start
2. **Deploy**: Follow [VPS_DEPLOYMENT.md](VPS_DEPLOYMENT.md) for production setup
3. **Develop**: Use [CLAUDE.md](CLAUDE.md) for technical development
4. **Extend**: See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for advanced features

---

**Need help?** Check the [main README](../README.md) or [VPS deployment guide](VPS_DEPLOYMENT.md) for troubleshooting and support.