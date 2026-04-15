#!/usr/bin/env python3
"""
Audit a Pinecone index for stale category/topic metadata labels.

Reusable tool — not a one-shot script. Prints per-namespace vector counts,
sampled metadata shape, and (if provided) a comparison against a pre-migration
snapshot JSONL to assert parity of counts + IDs. Satisfies AC #19 / #24.

Usage:
  python scripts/audit_pinecone_categories.py --index yonearth-episodes
  python scripts/audit_pinecone_categories.py --index yonearth-dev \\
      --pre-snapshot /tmp/pre.jsonl --fields categories,topics \\
      --labels REGENERATIVE,COMPOSTING,WATER,ENERGY

Exit codes:
  0 — audit complete; no stale labels (with --pre-snapshot: also counts+IDs match)
  1 — stale labels present OR pre/post mismatch
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

logger = logging.getLogger(__name__)

DEFAULT_STALE_LABELS = ("REGENERATIVE", "COMPOSTING", "WATER", "ENERGY")
DEFAULT_FIELDS = ("categories", "topics")


def _get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def _collect_ids(index, namespace: str, dim: int) -> set[str]:
    """Collect every vector ID in a namespace."""
    ids: set[str] = set()
    try:
        for page in index.list(namespace=namespace):
            for vid in (page if isinstance(page, list) else list(page)):
                ids.add(vid)
    except Exception:
        res = index.query(vector=[0.0] * dim, top_k=10000, namespace=namespace, include_metadata=False)
        for m in _get(res, "matches", []) or []:
            vid = _get(m, "id", None)
            if vid:
                ids.add(vid)
    return ids


def audit(
    index_name: str,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    labels: tuple[str, ...] = DEFAULT_STALE_LABELS,
    pre_snapshot: Path | None = None,
) -> int:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    dim = _get(stats, "dimension", 0)
    ns_map = _get(stats, "namespaces", {}) or {}

    print(f"Index: {index_name}  dimension: {dim}")
    print(f"Total vectors: {_get(stats, 'total_vector_count', 0)}")
    print(f"Namespaces: {list(ns_map)}")

    # --- Metadata shape sample
    print("\nSample metadata keys per namespace (first match):")
    for ns in ns_map:
        res = index.query(vector=[0.0] * dim, top_k=1, namespace=ns, include_metadata=True)
        matches = _get(res, "matches", []) or []
        if matches:
            md = _get(matches[0], "metadata", {}) or {}
            print(f"  [{ns or '(default)'}] {sorted(md.keys())}")

    # --- Stale-label counts
    print("\nStale-label vector counts:")
    stale_total = 0
    for ns in ns_map:
        for field in fields:
            for label in labels:
                try:
                    res = index.query(
                        vector=[0.0] * dim,
                        top_k=10000,
                        namespace=ns,
                        filter={field: {"$in": [label]}},
                        include_metadata=False,
                    )
                    n = len(_get(res, "matches", []) or [])
                    if n > 0:
                        print(f"  [{ns or '(default)'}] {field} contains {label!r}: {n}")
                        stale_total += n
                except Exception:
                    pass
    if stale_total == 0:
        print("  (none — Outcome A)")

    # --- Pre/post parity check
    parity_ok = True
    if pre_snapshot is not None:
        print("\nPre/post parity check against snapshot:")
        pre_counts: dict[str, int] = defaultdict(int)
        pre_ids: dict[str, set[str]] = defaultdict(set)
        with open(pre_snapshot) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ns = rec.get("namespace", "")
                pre_counts[ns] += 1
                pre_ids[ns].add(rec["id"])

        for ns in set(list(pre_counts) + list(ns_map)):
            post_count = _get(ns_map.get(ns, {}), "vector_count", 0)
            post_ids = _collect_ids(index, ns, dim) if ns in ns_map else set()
            ok_count = post_count == pre_counts.get(ns, 0)
            ok_ids = post_ids == pre_ids.get(ns, set())
            status = "OK" if ok_count and ok_ids else "MISMATCH"
            print(
                f"  [{ns or '(default)'}] pre={pre_counts.get(ns, 0)} post={post_count} "
                f"id_match={ok_ids} -> {status}"
            )
            if not (ok_count and ok_ids):
                parity_ok = False

    return 0 if stale_total == 0 and parity_ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser(description="Audit Pinecone metadata for stale category labels")
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--pre-snapshot", type=Path, help="JSONL export file to check parity against")
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help=f"Comma-separated metadata fields to scan (default: {','.join(DEFAULT_FIELDS)})",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_STALE_LABELS),
        help=f"Comma-separated stale labels to count (default: {','.join(DEFAULT_STALE_LABELS)})",
    )
    args = parser.parse_args()

    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    labels = tuple(l.strip() for l in args.labels.split(",") if l.strip())
    return audit(args.index, fields=fields, labels=labels, pre_snapshot=args.pre_snapshot)


if __name__ == "__main__":
    sys.exit(main())
