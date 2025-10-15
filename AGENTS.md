# Repository Guidelines

## Project Structure & Module Organization
Core backend modules live in `src/`: `src/api/` (FastAPI routers), `src/rag/` (retrieval logic), `src/ingestion/` (episode and book pipelines), and `src/character/` (Gaia persona). The static chat UI and knowledge-graph explorers are in `web/`. Operational helpers, including `scripts/start_local.py` and extraction tooling, sit in `scripts/`. Tests mirror the package layout in `tests/`, while transcripts, embeddings, and KG outputs are housed in `data/`. Consult `docs/` for deployment runbooks.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: set up a virtualenv before installs.
- `pip install -r requirements.txt`: install backend deps; use the specialized requirement files only when a job needs them.
- `python scripts/start_local.py`: perform preflight checks, launch FastAPI with autoreload, and open the static UI.
- `uvicorn src.api.main:app --reload`: quick manual server spin-up during iteration.
- `python tests/run_tests.py` or `python -m pytest tests -v --tb=short`: run the suite; coverage lands in `htmlcov/` when `pytest-cov` is present.
- `./deploy.sh`: full-stack wrapper that mirrors production.

## Coding Style & Naming Conventions
Match the existing Python tone: PEP 8 spacing, 4-space indents, and type hints on new public functions. Stick to snake_case for functions and vars, PascalCase for Pydantic models, and keep settings in `src/config/settings.py`. Reuse shared helpers instead of cloning logic. Frontend scripts (`web/*.js`) use camelCase and scoped selectors; centralize styling in `web/styles.css`. Document any new formatter or linter in `docs/` before enforcing it.

## Testing Guidelines
`pytest` powers the suite, with shared fixtures in `tests/conftest.py`. Mirror package paths when creating tests (e.g., `src/rag/foo.py` → `tests/test_rag_foo.py`) and name cases after behavior (`test_chunker_handles_empty_episode`). Keep fixtures reusable and move bulky payloads to `tests/fixtures/`. Use `pytest.mark.asyncio` or the `async_client` fixture for async work, and share coverage or failure logs in reviews.

## Commit & Pull Request Guidelines
Git history favors emoji-prefixed, imperative titles (`✨ Add BM25 reranker guardrails`). Use the body only to capture essential rationale. Pull requests should summarize scope, list validation (`pytest -v`, manual UI checks), and link issues or KG tasks; attach screenshots or logs whenever behavior shifts. For production rollouts, follow `CLAUDE.md` to sync `/root/yonearth-gaia-chatbot/` and restart `yonearth-api`.

## Environment & Configuration
Copy `.env.example` to `.env`, then set `OPENAI_API_KEY`, `PINECONE_API_KEY`, and add `ELEVENLABS_API_KEY` for voice. Run `scripts/start_local.py` to confirm configuration. Keep secrets and large assets out of git, and coordinate new ingestion feeds to manage storage budgets.
