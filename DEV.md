# Local development

Run the full YOE Gaia stack on your laptop in one terminal. Hits the same
production Pinecone index for retrieval (read-only by default; ingestion
scripts have explicit flags).

## One-time setup (~15 min)

```bash
# 1. Create a Python 3.11 venv (3.10 also fine; 3.13/3.14 likely break langchain 0.1.x)
cd /Users/darrenzal/projects/yonearth-gaia-chatbot
python3.11 -m venv .venv

# 2. Install deps + pin to server's langchain 0.1.x line
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install \
  'langchain==0.1.20' 'langchain-community==0.0.38' \
  'langchain-core==0.1.53' 'langchain-openai==0.0.5' \
  'langchain-pinecone==0.0.3' 'langchain-text-splitters==0.0.2' \
  'pinecone-client==3.2.2' 'openai==1.109.1' 'fastapi==0.121.1'

# 3. Pull the secrets-bearing .env from the production server
scp claudeuser@152.53.194.214:/home/claudeuser/yonearth-gaia-chatbot/.env .env
# (.env is gitignored; never commit it)
```

## Daily run

```bash
.venv/bin/python3 scripts/dev.py            # default port 8765
.venv/bin/python3 scripts/dev.py --reload   # auto-reload on src/ + web/ changes
```

Then open **http://localhost:8765/guide/** in a browser. First boot takes ~30s
(builds BM25 keyword index over 182 episodes); subsequent runs reuse the
cached index at `data/indexes/keyword_index.pkl`.

## What the dev script does

- Mounts `web/` at `/guide/` (matches production nginx alias)
- Aliases `/guide/yoe_taxonomy.json` → `web/data/yoe_taxonomy.json` (prod
  serves it flat from `/var/www/yonearth-guide/`)
- Redirects `/` → `/guide/`
- Resolves the KG visualization data file via env override
  (`YOE_KG_DATA`) or repo-relative fallback when the production absolute
  path doesn't exist locally — see `_kg_data_file()` in `src/api/main.py`

## Iteration workflow

1. Edit `web/*.{html,js,css}` or `src/**/*.py`
2. With `--reload`, the server picks up backend changes automatically.
   Frontend changes need only a browser hard-refresh (Cmd-Shift-R).
3. Test in browser at `localhost:8765/guide/`
4. When satisfied, run `scripts/deploy-prod.sh` to push to the live site
5. Verify on `earthdo.me/guide/`

## What's safe to run locally vs production

| Action | Local OK? | Notes |
|---|---|---|
| Browse `/guide/` + KG render | ✅ | Reads local `data/knowledge_graph/visualization_data.json` |
| Chat queries (`/api/chat`) | ✅ | Reads from production Pinecone — same vectors as live site |
| `scripts/build_yoe_taxonomy.py` | ✅ | Pulls from Aaron's Google Sheet, writes `web/data/yoe_taxonomy.json` |
| `scripts/build_merge_candidates.py` | ✅ | Read-only against `data/knowledge_graph/visualization_data.json` |
| `scripts/apply_kg_merges.py` | ⚠️ | Mutates `data/knowledge_graph/visualization_data.json` locally — does NOT push to server (use `scripts/deploy-prod.sh` for that) |
| `scripts/ingest_new_resources.py` | 🚫 | Writes to **production** Pinecone — only run when you mean to |
| `scripts/scrape_yoe_resources.py` | ✅ | Just writes JSON to `data/transcripts/` |

## Troubleshooting

- **Port 8765 in use**: pass `--port 8770` (or any free port).
- **`tiktoken: Could not automatically map text-embedding-3-small`**: already
  patched in `src/rag/vectorstore.py` (cl100k_base fallback).
- **404 on `/api/knowledge-graph/data`**: confirm
  `data/knowledge_graph/visualization_data.json` exists locally. If not,
  `scp claudeuser@152.53.194.214:/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/visualization_data.json data/knowledge_graph/`.
- **`langchain.schema not found`**: you have langchain 1.x; downgrade per
  setup step 2 above.
- **Slow startup**: first boot rebuilds the BM25 index (~30s). Subsequent
  starts use the pickle cache.

## Production deploy

The deploy script is unchanged:

```bash
scripts/deploy-prod.sh \
  -t claudeuser@152.53.194.214:/var/www/yonearth-guide/ \
  -m "what changed" \
  web/<files>...
```

Backend changes (`src/api/*.py`) need a separate FastAPI service restart on
the server:

```bash
ssh claudeuser@152.53.194.214 'sudo systemctl restart yonearth-fastapi.service'
```
