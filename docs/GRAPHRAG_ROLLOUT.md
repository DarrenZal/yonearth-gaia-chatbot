# GraphRAG v2 Rollout Guide

## Current Status (2025-12-08)

**GraphRAG v2 is now the PRIMARY search backend for gaiaai.xyz/YonEarth/**

- `/api/chat` routes to `/api/graphrag/chat` (nginx config)
- Knowledge Graph: 26,219 entities, 39,118 relationships, 45,673 aliases
- Model: `gpt-4.1-mini` with `temperature=0` for consistent responses
- Single uvicorn worker for stable graph state

## Overview

GraphRAG v2 includes significant improvements:
- Entity noise filtering (no more single-letter matches)
- Typed relationship rendering (FOUNDED, WORKS_FOR, ADVOCATES_FOR, etc.)
- Community disambiguation with token-overlap gating
- KG-guided chunk retrieval with configurable boost
- Entity descriptions in context (e.g., "Samantha Power - Founder and Director of BioFi Project")
- Proper episode overlap calculation

## Configuration

### Environment Variables

```bash
# Backend version: "v2" (new) or "v1" (fallback)
GRAPHRAG_BACKEND_VERSION=v2

# KG boost factor (default: 1.3)
# Increase if KG matches aren't surfacing; decrease if irrelevant sources dominate
GRAPHRAG_KG_BOOST_FACTOR=1.3

# LLM settings for consistent responses
OPENAI_MODEL=gpt-4.1-mini
GAIA_TEMPERATURE=0
```

### Quick Rollback

To revert to v1 if issues arise:

```bash
# Option 1: Environment variable (requires restart)
export GRAPHRAG_BACKEND_VERSION=v1
sudo kill -HUP $(pgrep -f "uvicorn.*8001")

# Option 2: Edit .env file
echo "GRAPHRAG_BACKEND_VERSION=v1" >> .env
sudo kill -HUP $(pgrep -f "uvicorn.*8001")
```

## Pre-Deployment Checklist

1. **Run smoke tests**:
   ```bash
   python3 scripts/smoke_test_graphrag.py --api-url http://127.0.0.1:8001
   ```

2. **Check health endpoint**:
   ```bash
   curl http://127.0.0.1:8001/api/graphrag/health | jq
   ```

3. **Verify all components ready**:
   - `initialized: true`
   - `community_search_ready: true`
   - `local_search_ready: true`
   - `vectorstore_ready: true`
   - `gaia_ready: true`

## Post-Deployment Monitoring

### Key Metrics to Watch

1. **KG Boost Effectiveness** (in logs):
   ```
   KG boost: 3/5 KG-matched chunks in top-4 (positions: [0, 1, 3])
   ```
   - If frequently "0/N KG-matched chunks reached top-k", consider increasing `GRAPHRAG_KG_BOOST_FACTOR`

2. **Episode Overlap** (comparison endpoint):
   ```bash
   curl -X POST http://127.0.0.1:8001/api/graphrag/compare \
     -H "Content-Type: application/json" \
     -d '{"message": "What is biochar?"}' | jq '.comparison_metrics'
   ```
   - Expect non-zero `episode_overlap_ratio`

3. **Response Latency**:
   - Target: < 8s average
   - Current baseline: ~4s

### Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Overlap always 0% | Episode ID type mismatch | Fixed in v2 |
| Short entity noise ("O", "pi") | Missing alias filter | Fixed in v2 |
| All RELATED_TO relationships | Wrong field name | Fixed in v2 |
| KG matches not in top-k | Boost too low | Increase `GRAPHRAG_KG_BOOST_FACTOR` to 1.4-1.5 |
| Irrelevant KG sources | Boost too high | Decrease `GRAPHRAG_KG_BOOST_FACTOR` to 1.1-1.2 |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/graphrag/health` | Health check |
| `POST /api/graphrag/chat` | Single GraphRAG query |
| `POST /api/graphrag/compare` | Side-by-side BM25 vs GraphRAG |
| `GET /api/graphrag/communities/{level}` | List communities |
| `GET /api/graphrag/entities/search?q=...` | Search entities |

## Frontend

The comparison UI is available at:
- **Production**: https://gaiaai.xyz/YonEarth/graphrag/
- **Local**: http://localhost:8001/graphrag/

## Files Modified in v2

- `src/rag/graphrag_local_search.py` - Entity filtering, relationship types
- `src/rag/graphrag_community_search.py` - Token overlap disambiguation
- `src/rag/graphrag_chain.py` - KG-guided retrieval, query classifier
- `src/api/graphrag_chat_endpoints.py` - Episode overlap normalization
- `src/config/settings.py` - Feature flags

## Rollback Procedure

If critical issues occur:

1. **Immediate** (seconds):
   ```bash
   export GRAPHRAG_BACKEND_VERSION=v1
   sudo kill -HUP $(pgrep -f "uvicorn.*8001")
   ```

2. **Verify rollback**:
   ```bash
   curl http://127.0.0.1:8001/api/graphrag/health
   ```

3. **Investigate** using logs:
   ```bash
   sudo journalctl -u yonearth-api-migrated -f
   ```

4. **Report issue** with:
   - Query that failed
   - Error message
   - Timestamp

## Nginx Routing

The main chat endpoint is routed via nginx:

```nginx
# /etc/nginx/sites-enabled/gaiaai.xyz
location = /api/chat {
    proxy_pass http://localhost:8001/api/graphrag/chat;
    ...
}
```

To switch back to BM25:
```bash
sudo sed -i 's|/api/graphrag/chat|/api/bm25/chat|' /etc/nginx/sites-enabled/gaiaai.xyz
sudo nginx -t && sudo systemctl reload nginx
```

## Rebuilding the Knowledge Graph Index

If you need to rebuild the graph index from the unified graph:

```bash
# Rebuild from unified_v2.json
python3 scripts/rebuild_graph_index_from_unified.py

# Add relationships to cluster registry for viewer
python3 scripts/add_relationships_to_cluster_registry.py

# Deploy to viewer
./scripts/deploy_graphrag.sh
```
