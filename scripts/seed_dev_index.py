#!/usr/bin/env python3
"""
Provision the dev Pinecone index and seed it with ~50 representative vectors
carrying SYNTHETIC legacy category metadata.

Purpose: give migrate_categories.py real data to prove itself against. Prod has
zero vectors with stale labels (see 2026-04-15 audit), so without synthetic
seeding the migration would be a no-op and plan tests #3/#12 couldn't
exercise the rewrite path.

Provisions yonearth-dev if missing (cosine, 1536d, smallest serverless tier).
Fetches ~50 random vectors from prod, injects a mix of stale and current
category labels into their metadata, and upserts into the dev index.

Usage:
  python scripts/seed_dev_index.py [--source-index yonearth-episodes] [--dev-index yonearth-dev] [--count 50]
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


LEGACY_CATEGORY_POOL = [
    "REGENERATIVE",
    "COMPOSTING",
    "WATER",
    "ENERGY",
    "PERMACULTURE",       # legacy spelling
    "SUSTAINABILITY",     # legacy spelling
    "POLICY & GOVERNMENT",  # legacy spelling
]
CURRENT_CATEGORY_POOL = [
    "BIOCHAR",
    "SOIL",
    "HERBAL MEDICINE",
    "CLIMATE & SCIENCE",
    "FARMING & FOOD",
    "COMMUNITY",
]


def _get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def ensure_dev_index(pc: Pinecone, dev_index_name: str, dimension: int) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if dev_index_name in existing:
        logger.info(f"Dev index {dev_index_name!r} already exists")
        return
    logger.info(f"Creating dev index {dev_index_name!r} (dim={dimension}, metric=cosine, serverless)")
    pc.create_index(
        name=dev_index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for it to be ready
    import time
    for _ in range(30):
        existing = [idx.name for idx in pc.list_indexes()]
        if dev_index_name in existing:
            break
        time.sleep(1)
    time.sleep(3)  # Give it a little more to fully provision
    logger.info(f"Dev index {dev_index_name!r} created")


def synth_metadata(rng: random.Random, base: dict) -> dict:
    md = dict(base)
    n_legacy = rng.randint(1, 3)
    n_current = rng.randint(0, 2)
    legacy = rng.sample(LEGACY_CATEGORY_POOL, min(n_legacy, len(LEGACY_CATEGORY_POOL)))
    current = rng.sample(CURRENT_CATEGORY_POOL, min(n_current, len(CURRENT_CATEGORY_POOL)))
    md["categories"] = legacy + current
    return md


def fetch_source_sample(pc: Pinecone, source_index_name: str, count: int) -> list[dict]:
    """Grab `count` random vectors from the source (prod) index."""
    idx = pc.Index(source_index_name)
    stats = idx.describe_index_stats()
    dim = _get(stats, "dimension", 0)
    # Zero-vector query returns arbitrary ordering — effectively random for our seeding.
    res = idx.query(vector=[0.0] * dim, top_k=count, include_values=True, include_metadata=True)
    matches = _get(res, "matches", []) or []
    records: list[dict] = []
    for m in matches:
        records.append(
            {
                "id": f"seed-{_get(m, 'id', 'unknown')}",
                "values": list(_get(m, "values", []) or []),
                "metadata": dict(_get(m, "metadata", {}) or {}),
            }
        )
    return records, dim


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser(description="Provision + seed the dev Pinecone index")
    parser.add_argument("--source-index", default=os.environ.get("PINECONE_INDEX_NAME", "yonearth-episodes"))
    parser.add_argument("--dev-index", default=os.environ.get("PINECONE_INDEX_NAME_DEV", "yonearth-dev"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible synthetic metadata")
    args = parser.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    pc = Pinecone(api_key=api_key)
    rng = random.Random(args.seed)

    records, dim = fetch_source_sample(pc, args.source_index, args.count)
    logger.info(f"Fetched {len(records)} source vectors from {args.source_index!r} (dim={dim})")
    if not records:
        raise SystemExit(f"Source index {args.source_index!r} returned no vectors; cannot seed")

    ensure_dev_index(pc, args.dev_index, dim)

    # Inject synthetic legacy metadata
    for r in records:
        r["metadata"] = synth_metadata(rng, r["metadata"])

    dev_idx = pc.Index(args.dev_index)
    # Clear existing dev vectors first for idempotency
    try:
        dev_idx.delete(delete_all=True)
        logger.info(f"Cleared dev index {args.dev_index!r}")
    except Exception as e:
        # Empty index delete_all raises on some Pinecone versions — ignore
        logger.info(f"Clear-all returned: {type(e).__name__} (likely empty — ok)")

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(records), batch_size):
        dev_idx.upsert(vectors=records[i : i + batch_size])
    logger.info(f"Seeded {len(records)} vectors into {args.dev_index!r} with synthetic legacy categories")

    # Show a sample so the operator can eyeball
    sample = records[0]
    logger.info(f"Sample seeded metadata: id={sample['id']}  categories={sample['metadata'].get('categories')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
