# YonEarth Gaia Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that embodies Gaia, the spirit of Earth, providing wisdom from the YonEarth Community Podcast episodes.

## Project Structure

```
yonearth-chatbot/
├── src/
│   ├── ingestion/       # Data processing and chunking
│   ├── rag/            # RAG pipeline with vector search
│   ├── character/      # Gaia personality and prompts
│   ├── api/            # FastAPI backend
│   └── config/         # Configuration and settings
├── web/                # Frontend chat interface
├── deploy/            # Deployment configurations
├── tests/             # Test suite
└── requirements.txt   # Python dependencies
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

3. Process episodes for vector database:
   ```bash
   python -m src.ingestion.process_episodes
   ```

4. Run the API server:
   ```bash
   uvicorn src.api.main:app --reload
   ```

5. Open web interface:
   ```bash
   open web/index.html
   ```

## Features

- **Gaia Character**: Earth goddess personality providing wisdom from podcast content
- **Episode Citations**: Every response includes episode references with timestamps
- **Smart Retrieval**: Semantic search across 172 YonEarth podcast episodes
- **Conversation Memory**: Maintains context across chat sessions
- **WordPress Ready**: Includes plugin for easy WordPress integration

## Configuration

See `.env.example` for required environment variables:
- `OPENAI_API_KEY`: For embeddings and chat completion
- `PINECONE_API_KEY`: For vector database
- `PINECONE_ENVIRONMENT`: Your Pinecone environment
- `REDIS_URL`: For caching (optional)

## Deployment

The project includes a `render.yaml` blueprint for easy deployment to Render.com.

## License

MIT License - See LICENSE file for details