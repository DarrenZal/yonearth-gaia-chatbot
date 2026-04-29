#!/usr/bin/env python3
"""
dev.py — local development server for the YOE Gaia chatbot.

Runs the FastAPI backend AND serves the `/web/` static frontend on a single
port. In production, nginx handles the static side; locally we bundle them
so you only need one terminal.

Usage:
  cd /Users/darrenzal/projects/yonearth-gaia-chatbot
  .venv/bin/python3 scripts/dev.py            # default :8000
  .venv/bin/python3 scripts/dev.py --port 8001
  .venv/bin/python3 scripts/dev.py --reload   # auto-reload on file change

Then open http://localhost:8000/guide/ in a browser.

Notes:
  - Reads .env from repo root (must contain OPENAI_API_KEY + PINECONE_API_KEY).
  - Hits the SAME production Pinecone index `yonearth-episodes`. Don't run
    ingestion scripts against this from the dev server unless you mean to.
  - The /guide/ path is mounted to /web/ — same paths as production behind
    nginx. The KG iframe loads /guide/KnowledgeGraph.html, which works
    because the StaticFiles mount preserves directory structure.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
WEB_DIR = REPO / "web"

# Ensure repo root is importable
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--reload", action="store_true", help="auto-reload on src/ changes")
    args = ap.parse_args()

    if not WEB_DIR.is_dir():
        print(f"ERROR: {WEB_DIR} not found — run from repo root", file=sys.stderr)
        return 2
    if not (REPO / ".env").is_file():
        print(f"ERROR: {REPO}/.env not found — copy from .env.local or pull from server", file=sys.stderr)
        return 2

    # Import the live app and mount static frontend
    from src.api.main import app

    # Mount /guide/ → web/, alias /guide/yoe_taxonomy.json (prod nginx
    # serves it flat from /var/www/yonearth-guide/), redirect / → /guide/.
    _wire_static_routes(app)

    print(f"\n→ http://{args.host}:{args.port}/guide/", file=sys.stderr)
    print(f"  serving frontend from: {WEB_DIR}", file=sys.stderr)
    print(f"  reload: {args.reload}", file=sys.stderr)
    print(file=sys.stderr)

    import uvicorn

    if args.reload:
        # uvicorn's --reload requires an import string, not the app instance
        uvicorn.run("scripts.dev:_make_app", host=args.host, port=args.port, reload=True, factory=True, reload_dirs=[str(REPO / "src"), str(WEB_DIR)])
    else:
        uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _wire_static_routes(app):
    """Mount /guide/ → web/ AND alias the flat-root files that nginx serves
    in production (yoe_taxonomy.json is fetched as /guide/yoe_taxonomy.json
    but lives at web/data/yoe_taxonomy.json in the repo)."""
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse, FileResponse

    @app.get("/guide/yoe_taxonomy.json", include_in_schema=False)
    async def _taxonomy_alias():
        p = WEB_DIR / "data" / "yoe_taxonomy.json"
        if p.exists():
            return FileResponse(p, media_type="application/json")
        return RedirectResponse("/guide/data/yoe_taxonomy.json")

    app.mount("/guide", StaticFiles(directory=str(WEB_DIR), html=True), name="guide")

    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return RedirectResponse(url="/guide/", status_code=302)


def _make_app():
    """Factory used when --reload is enabled."""
    from src.api.main import app
    _wire_static_routes(app)
    return app


if __name__ == "__main__":
    sys.exit(main())
