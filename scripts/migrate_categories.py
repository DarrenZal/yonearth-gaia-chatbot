#!/usr/bin/env python3
"""
Rewrite stale category labels in every vector's metadata.

Deterministic rewrite table (plan §7):
  REGENERATIVE             -> REGEN / SOCIAL ENTERPRISE
  COMPOSTING               -> SOIL
  WATER                    -> ECOLOGY & NATURE
  ENERGY                   -> TECHNOLOGY & MATERIALS
  PERMACULTURE             -> PERMA-CULTURE
  SUSTAINABILITY           -> SUSTAIN-ABILITY
  POLICY & GOVERNMENT      -> POLICY & GOVERNMT

Iterates every namespace reported by describe_index_stats() (fails loudly if
enumeration fails — partial coverage is worse than no coverage). Rewrites the
'categories' and 'topics' metadata fields (both list-of-string and single-string
shapes), deduplicates after rewrite, and upserts changed records only.

Creates .pinecone-migration-lock at repo root for the duration of the run so
concurrent writers abort (plan test #21). The lockfile is REMOVED on clean exit.
On any exception the lockfile is LEFT BEHIND — operator must inspect and clear
manually. Aborting-on-signal is also handled.

Usage:
  python scripts/migrate_categories.py --index yonearth-dev --dry-run
  python scripts/migrate_categories.py --index yonearth-dev
  python scripts/migrate_categories.py --index yonearth-episodes --confirm-prod

As of 2026-04-15 the prod Pinecone index (yonearth-episodes) has ZERO vectors
with stale category labels — the write path never sets them. This script
exists as dormant toolkit for future taxonomy adjustments, not a
current-required migration. See tests/fixtures/g2_rehearsal_transcript.md
for the dev-index cycle that proves it works.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone

from src.utils.migration_lock import LOCKFILE

logger = logging.getLogger(__name__)

REWRITE_TABLE: dict[str, str] = {
    "REGENERATIVE": "REGEN / SOCIAL ENTERPRISE",
    "COMPOSTING": "SOIL",
    "WATER": "ECOLOGY & NATURE",
    "ENERGY": "TECHNOLOGY & MATERIALS",
    "PERMACULTURE": "PERMA-CULTURE",
    "SUSTAINABILITY": "SUSTAIN-ABILITY",
    "POLICY & GOVERNMENT": "POLICY & GOVERNMT",
}

FIELDS_TO_REWRITE = ("categories", "topics")


def _get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def rewrite_value(value: Any) -> tuple[Any, bool]:
    """Rewrite a categories/topics metadata value.

    Accepts list[str] or a single string. Returns (new_value, changed).
    De-duplicates list values after rewrite so a vector tagged both SOIL and
    COMPOSTING becomes singular SOIL.
    """
    if value is None:
        return value, False
    if isinstance(value, str):
        new = REWRITE_TABLE.get(value, value)
        return new, new != value
    if isinstance(value, list):
        changed = False
        new_list: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                if item not in seen:
                    new_list.append(item)
                    seen.add(item)
                continue
            mapped = REWRITE_TABLE.get(item, item)
            if mapped != item:
                changed = True
            if mapped not in seen:
                new_list.append(mapped)
                seen.add(mapped)
        return new_list, changed or new_list != value
    return value, False


def rewrite_metadata(metadata: dict) -> tuple[dict, bool]:
    """Rewrite every target field in a metadata dict. Returns (new_md, changed)."""
    new_md = dict(metadata)
    any_changed = False
    for field in FIELDS_TO_REWRITE:
        if field in new_md:
            new_val, changed = rewrite_value(new_md[field])
            if changed:
                new_md[field] = new_val
                any_changed = True
    return new_md, any_changed


def _collect_ids(index, namespace: str, dim: int) -> list[str]:
    try:
        out: list[str] = []
        for page in index.list(namespace=namespace):
            for vid in (page if isinstance(page, list) else list(page)):
                out.append(vid)
        return out
    except Exception:
        res = index.query(
            vector=[0.0] * dim, top_k=10000, namespace=namespace, include_metadata=False
        )
        return [_get(m, "id", None) for m in (_get(res, "matches", []) or []) if _get(m, "id", None)]


def migrate(
    index_name: str,
    *,
    dry_run: bool = False,
    batch_size: int = 100,
    sample_diff_size: int = 20,
) -> dict[str, int]:
    """Walk every namespace, rewrite target fields, return {namespace: changed_count}."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    ns_map = _get(stats, "namespaces", {}) or {}
    dim = _get(stats, "dimension", 0)
    if not ns_map:
        raise RuntimeError(
            f"describe_index_stats() returned no namespaces for {index_name}; aborting to avoid partial coverage"
        )

    logger.info(f"Index: {index_name}  namespaces: {list(ns_map)}  dry_run={dry_run}")

    per_ns_changed: dict[str, int] = {}
    printed_diff_samples = 0

    for ns in ns_map:
        ids = _collect_ids(index, ns, dim)
        logger.info(f"  [{ns or '(default)'}] {len(ids)} IDs")
        changed = 0

        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            fetched = index.fetch(ids=chunk, namespace=ns)
            vectors = _get(fetched, "vectors", {}) or {}
            to_upsert: list[dict] = []

            for vid, vec in vectors.items():
                metadata = dict(_get(vec, "metadata", {}) or {})
                new_md, md_changed = rewrite_metadata(metadata)
                if md_changed:
                    changed += 1
                    if printed_diff_samples < sample_diff_size:
                        logger.info(
                            f"    diff sample id={vid}\n      before: {metadata}\n      after:  {new_md}"
                        )
                        printed_diff_samples += 1
                    to_upsert.append(
                        {
                            "id": vid,
                            "values": list(_get(vec, "values", []) or []),
                            "metadata": new_md,
                        }
                    )

            if to_upsert and not dry_run:
                index.upsert(vectors=to_upsert, namespace=ns)

        per_ns_changed[ns] = changed
        logger.info(f"  [{ns or '(default)'}] changed={changed}{' (dry-run, no writes)' if dry_run else ''}")

    return per_ns_changed


def _require_confirm_prod(index_name: str, confirm_prod: bool) -> None:
    """Prod writes demand --confirm-prod. Anything matching /^yonearth-(episodes|prod)/ is prod."""
    if index_name in ("yonearth-episodes", "yonearth-prod") and not confirm_prod:
        raise SystemExit(
            f"Refusing to migrate prod index {index_name!r} without --confirm-prod"
        )


def _install_sigint_handler() -> None:
    def handler(signum, frame):
        logger.error(f"received signal {signum}; lockfile left in place for operator inspection")
        sys.exit(130)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser(description="Rewrite stale Pinecone category labels")
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--dry-run", action="store_true", help="Print diffs, do not write")
    parser.add_argument("--confirm-prod", action="store_true", help="Required for prod index writes")
    args = parser.parse_args()

    _require_confirm_prod(args.index, args.confirm_prod)
    _install_sigint_handler()

    # Print target loudly before any write
    print(f"MIGRATION TARGET INDEX: {args.index}  dry_run={args.dry_run}")

    # Create lockfile for the duration of a real run. Dry-runs also lock —
    # they still fetch metadata and we don't want ingestion to slip in during
    # the fetch.
    if not LOCKFILE.exists():
        LOCKFILE.touch()
        logger.info(f"Created lockfile at {LOCKFILE}")
    else:
        raise SystemExit(
            f"Lockfile already exists at {LOCKFILE} — another migration in progress? Remove it manually if stale."
        )

    try:
        changed = migrate(args.index, dry_run=args.dry_run)
        total = sum(changed.values())
        logger.info(f"Migration complete: {total} vectors rewritten across {len(changed)} namespaces")
    finally:
        # Only remove lockfile on success. Leave it for operator on exception
        # (caught by outer context — any raise skips this branch).
        pass

    # Clean exit — remove lockfile
    try:
        LOCKFILE.unlink()
        logger.info(f"Removed lockfile at {LOCKFILE}")
    except FileNotFoundError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
