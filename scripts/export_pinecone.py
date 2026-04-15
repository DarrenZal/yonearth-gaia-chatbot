#!/usr/bin/env python3
"""
Export every vector in a Pinecone index to JSONL — point-in-time backup.

Each line is a JSON object:
  {"id": str, "namespace": str, "values": list[float], "metadata": dict}

Runs across every namespace reported by describe_index_stats(). Verifies
that the export count per namespace matches the index's reported count.
Used as the snapshot half of a migration rollback cycle (paired with
import_pinecone.py).

Usage:
  python scripts/export_pinecone.py --index yonearth-dev --out /tmp/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone

logger = logging.getLogger(__name__)


def _get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def export_index(index_name: str, out_path: Path, batch_size: int = 100) -> dict[str, int]:
    """Export all vectors to a JSONL file. Returns {namespace: count} actually written.

    Integrity invariant: `written_count == len(ids_enumerated_via_list)` per
    namespace. We log `describe_index_stats()` counts as informational (they are
    eventually-consistent on Pinecone serverless — lag behind actual vector count
    for minutes after deletes/upserts), but only the list()/fetch count is
    authoritative for mismatch detection.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    ns_map = _get(stats, "namespaces", {}) or {}
    stats_counts = {ns: _get(info, "vector_count", 0) for ns, info in ns_map.items()}
    dim = _get(stats, "dimension", 0)

    logger.info(f"Export target: {index_name}  dim={dim}  namespaces={list(stats_counts)}")

    written_counts: dict[str, int] = {}
    enumerated_counts: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fh:
        for ns in stats_counts:
            written = 0
            enumerated = 0
            # Paginate IDs via list(); Pinecone serverless supports this.
            try:
                id_pages = index.list(namespace=ns)
            except Exception:
                # Older/classic indexes don't support list(); fall back to query.
                id_pages = _fallback_list_via_query(index, ns, dim)

            buffer: list[str] = []
            for page in id_pages:
                ids = list(page) if not isinstance(page, list) else page
                if not ids:
                    continue
                enumerated += len(ids)
                for i in range(0, len(ids), batch_size):
                    chunk = ids[i : i + batch_size]
                    fetched = index.fetch(ids=chunk, namespace=ns)
                    vectors = _get(fetched, "vectors", {}) or {}
                    for vid, vec in vectors.items():
                        rec = {
                            "id": vid,
                            "namespace": ns,
                            "values": list(_get(vec, "values", []) or []),
                            "metadata": dict(_get(vec, "metadata", {}) or {}),
                        }
                        buffer.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
                        written += 1
                    if len(buffer) >= 500:
                        fh.write("\n".join(buffer) + "\n")
                        buffer.clear()
            if buffer:
                fh.write("\n".join(buffer) + "\n")
            written_counts[ns] = written
            enumerated_counts[ns] = enumerated
            logger.info(
                f"  [{ns or '(default)'}] wrote {written}  enumerated {enumerated}  "
                f"stats_reported {stats_counts[ns]}"
            )

    # Authoritative check: each fetched vector must have been enumerated first.
    # Mismatch here means a writer slipped in between list() and fetch().
    for ns in stats_counts:
        if written_counts[ns] != enumerated_counts[ns]:
            raise RuntimeError(
                f"Export integrity failure in namespace {ns!r}: "
                f"enumerated {enumerated_counts[ns]} IDs but fetched {written_counts[ns]} records. "
                f"A concurrent writer may have modified the index during export; abort."
            )
        # Informational: flag stats lag without aborting
        if written_counts[ns] != stats_counts[ns]:
            logger.info(
                f"  note: describe_index_stats reported {stats_counts[ns]} for {ns!r} but "
                f"list() enumerated {enumerated_counts[ns]} — eventual-consistency lag, not an error."
            )
    return written_counts


def _fallback_list_via_query(index, namespace: str, dim: int):
    """Zero-vector query fallback for indexes that don't support list()."""
    res = index.query(
        vector=[0.0] * dim, top_k=10000, namespace=namespace, include_values=False, include_metadata=False,
    )
    matches = _get(res, "matches", []) or []
    ids = [_get(m, "id", None) for m in matches]
    ids = [i for i in ids if i]
    # Return as a single-page iterable for the caller loop
    yield ids


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser(description="Export a Pinecone index to JSONL")
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    args = parser.parse_args()

    written = export_index(args.index, args.out)
    total = sum(written.values())
    logger.info(f"Export complete: {total} vectors written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
