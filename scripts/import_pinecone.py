#!/usr/bin/env python3
"""
Restore a Pinecone index from a JSONL export — rollback half of the migration cycle.

Reads the JSONL produced by export_pinecone.py and upserts every record back
into the target index, preserving id / namespace / values / metadata exactly.
Respects the migration lockfile (refuses to write if one is present) unless
--force-bypass-lock is passed (used only by migrate_categories.py itself).

Usage:
  python scripts/import_pinecone.py --index yonearth-dev --in /tmp/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone

from src.utils.migration_lock import abort_if_migration_in_progress

logger = logging.getLogger(__name__)


def _get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def import_index(index_name: str, in_path: Path, batch_size: int = 100) -> dict[str, int]:
    """Upsert every record in the JSONL back into the index."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Group records by namespace for batched upsert
    per_ns: dict[str, list[dict]] = defaultdict(list)
    with open(in_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            per_ns[rec.get("namespace", "")].append(rec)

    written_counts: dict[str, int] = {}
    for ns, records in per_ns.items():
        written = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            vectors = [
                {
                    "id": r["id"],
                    "values": r["values"],
                    "metadata": r.get("metadata", {}),
                }
                for r in batch
            ]
            index.upsert(vectors=vectors, namespace=ns)
            written += len(vectors)
        written_counts[ns] = written
        logger.info(f"  [{ns or '(default)'}] restored {written} vectors")
    return written_counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser(description="Restore a Pinecone index from JSONL")
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--in", dest="in_path", required=True, type=Path, help="Input JSONL path")
    parser.add_argument(
        "--force-bypass-lock",
        action="store_true",
        help="Bypass the .pinecone-migration-lock check (only migrate_categories.py should use this)",
    )
    args = parser.parse_args()

    if not args.force_bypass_lock:
        abort_if_migration_in_progress()

    written = import_index(args.index, args.in_path)
    total = sum(written.values())
    logger.info(f"Import complete: {total} vectors restored into {args.index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
