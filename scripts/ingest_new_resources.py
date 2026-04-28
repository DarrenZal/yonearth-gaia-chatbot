#!/usr/bin/env python3
"""
ingest_new_resources.py — chunk + embed + upload SPECIFIC episode-shaped JSON
files to the production Pinecone index. Skips the full re-ingest path that
process_episodes.py would trigger.

Inputs (default): the two files emitted by scripts/scrape_yoe_resources.py
  - data/transcripts/episode_resource_soilwerks.json
  - data/transcripts/episode_resource_welewaters.json

Pass `--dry-run` to chunk + cost-estimate without writing to Pinecone.
Pass `--files <path> [<path> ...]` to override the default file set.

Cost model (text-embedding-3-small): $0.00002 per 1K tokens. The scraped
content for these two sources totals ~10KB, so the embedding cost is sub-cent.

Effects on production:
  - Writes ~N new vectors to Pinecone index `yonearth-episodes`.
  - Existing vectors are NOT modified.
  - The local BM25 cache (data/cache/bm25_index.pkl) will rebuild from
    Pinecone on next API startup — no separate update needed.

NOT a deploy script. The FastAPI service does NOT need a restart for new
Pinecone vectors to appear in retrieval (LangChain queries Pinecone live).
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

# repo root on sys.path so `src.*` imports work when run as a script
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.ingestion.chunker import TranscriptChunker  # noqa: E402
from src.ingestion.episode_processor import Episode  # noqa: E402

DEFAULTS = [
    REPO / "data" / "transcripts" / "episode_resource_soilwerks.json",
    REPO / "data" / "transcripts" / "episode_resource_welewaters.json",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ingest_new_resources")


def load_episode(fp: pathlib.Path) -> Episode:
    with fp.open(encoding="utf-8") as f:
        d = json.load(f)
    ep = Episode(d)
    if not ep.has_transcript:
        raise ValueError(
            f"{fp.name}: full_transcript too short ({len(ep.transcript)} chars). Min 100."
        )
    return ep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="chunk + cost-estimate only; no Pinecone write")
    ap.add_argument("--files", nargs="+", help="override default file set", default=None)
    args = ap.parse_args()

    files = [pathlib.Path(p) for p in (args.files or DEFAULTS)]
    for fp in files:
        if not fp.exists():
            log.error(f"missing: {fp}")
            return 2

    log.info(f"Loading {len(files)} episode-shaped files")
    episodes = [load_episode(fp) for fp in files]
    for ep in episodes:
        log.info(
            f"  {ep.episode_number}  '{ep.title[:60]}'  "
            f"transcript={len(ep.transcript)} chars"
        )

    log.info("Chunking…")
    chunker = TranscriptChunker()
    documents = chunker.chunk_episodes(episodes)

    # Tag every chunk's metadata with source_type from the source JSON so
    # downstream filtering can find them later.
    for fp, ep in zip(files, episodes):
        with fp.open(encoding="utf-8") as f:
            raw = json.load(f)
        st = raw.get("source_type")
        su = raw.get("source_url")
        for d in documents:
            if d.metadata.get("episode_number") == str(ep.episode_number):
                if st: d.metadata["source_type"] = st
                if su: d.metadata["source_url"] = su

    log.info(f"  produced {len(documents)} chunks")
    total_chars = sum(len(d.page_content) for d in documents)
    log.info(f"  total chars across chunks: {total_chars}")

    if args.dry_run:
        log.info("--dry-run: skipping Pinecone write")
        log.info("Sample chunk metadata:")
        for d in documents[:3]:
            log.info(f"  {d.metadata}")
        return 0

    # Live ingest — import vectorstore late so dry-run path does not require
    # full settings env to load.
    from src.rag.vectorstore import YonEarthVectorStore  # noqa: WPS433

    log.info("Connecting to Pinecone…")
    vs = YonEarthVectorStore()
    cost = vs.estimate_embedding_cost(documents)
    log.info(f"Embedding cost estimate: ${cost['estimated_cost_usd']:.4f}")

    log.info("Writing to Pinecone…")
    ids = vs.add_documents(documents)
    log.info(f"DONE — wrote {len(ids)} vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
