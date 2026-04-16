"""
Plan test #12 — end-to-end migration integrity on the dev Pinecone index.

Runs only when PINECONE_INDEX_NAME_DEV is set (else skipped).

Cycle exercised:
  1. Snapshot pre-migration: per-namespace counts + vector ID set.
  2. Export to temp JSONL.
  3. Run migrate_categories.migrate() on the dev index.
  4. Snapshot post-migration: counts must equal pre (exact); ID sets must equal
     pre-sets (exact); only metadata may differ.
  5. Spot-check 10 random vectors: stale labels rewritten per the deterministic table.
  6. Run import_pinecone.import_index() to restore from the pre-migration export.
  7. Snapshot post-rollback: counts + ID sets + metadata exactly match pre.

This is slow (~30-60s against a real dev index). Run with:
  PINECONE_INDEX_NAME_DEV=yonearth-dev pytest tests/test_migrate_categories_integrity.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    not os.environ.get("PINECONE_INDEX_NAME_DEV"),
    reason="PINECONE_INDEX_NAME_DEV not set — integrity tests require a dev Pinecone index",
)


def _get(obj, key, default=None):
    return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)


def _snapshot_index(index) -> dict:
    """Return {namespace: {'count': int, 'ids': set[str], 'meta_by_id': dict}}.

    Uses list()+fetch() as ground truth — describe_index_stats() is eventually
    consistent on Pinecone serverless and lags behind actual vector counts
    for minutes after upserts/deletes, so we don't trust it for assertions.
    """
    stats = index.describe_index_stats()
    ns_map = _get(stats, "namespaces", {}) or {}
    dim = _get(stats, "dimension", 0)
    out = {}
    for ns in ns_map:
        ids = set()
        try:
            for page in index.list(namespace=ns):
                for vid in (page if isinstance(page, list) else list(page)):
                    ids.add(vid)
        except Exception:
            res = index.query(vector=[0.0] * dim, top_k=10000, namespace=ns, include_metadata=False)
            for m in _get(res, "matches", []) or []:
                ids.add(_get(m, "id", None))
            ids.discard(None)

        meta_by_id = {}
        ids_list = list(ids)
        batch = 100
        for i in range(0, len(ids_list), batch):
            chunk = ids_list[i : i + batch]
            fetched = index.fetch(ids=chunk, namespace=ns)
            for vid, vec in (_get(fetched, "vectors", {}) or {}).items():
                meta_by_id[vid] = dict(_get(vec, "metadata", {}) or {})

        out[ns] = {"count": len(ids), "ids": ids, "meta_by_id": meta_by_id}
    return out


@pytest.fixture(scope="module")
def dev_index():
    from pinecone import Pinecone

    api_key = os.environ["PINECONE_API_KEY"]
    name = os.environ["PINECONE_INDEX_NAME_DEV"]
    pc = Pinecone(api_key=api_key)
    return pc.Index(name)


def test_full_export_migrate_import_cycle(dev_index):
    """Pre/post/rollback parity with byte-equal metadata at every checkpoint."""
    from scripts.export_pinecone import export_index
    from scripts.import_pinecone import import_index
    from scripts.migrate_categories import migrate

    dev_name = os.environ["PINECONE_INDEX_NAME_DEV"]

    # 1. Pre-snapshot
    pre = _snapshot_index(dev_index)
    assert sum(ns["count"] for ns in pre.values()) > 0, "dev index is empty — run scripts/seed_dev_index.py first"

    with tempfile.TemporaryDirectory() as td:
        export_path = Path(td) / "pre.jsonl"

        # 2. Export
        exported = export_index(dev_name, export_path)
        for ns, n in exported.items():
            assert n == pre[ns]["count"], f"export count mismatch in {ns!r}"

        # 3. Migrate
        changed = migrate(dev_name, dry_run=False)
        # Should rewrite at least one vector if seeded correctly
        assert sum(changed.values()) > 0, "seeded dev index had no stale labels — re-run seed_dev_index.py"

        # 4. Post-migration snapshot
        post = _snapshot_index(dev_index)
        for ns, pre_info in pre.items():
            assert post[ns]["count"] == pre_info["count"], f"count drift in {ns!r}"
            assert post[ns]["ids"] == pre_info["ids"], f"ID set drift in {ns!r}"

        # 5. Spot-check: stale labels in metadata are gone after migrate
        stale = {"REGENERATIVE", "COMPOSTING", "WATER", "ENERGY"}
        for ns, info in post.items():
            for vid, md in info["meta_by_id"].items():
                for field in ("categories", "topics"):
                    val = md.get(field)
                    if isinstance(val, list):
                        assert not (set(val) & stale), f"stale label in {ns}/{vid}/{field}: {val}"
                    elif isinstance(val, str):
                        assert val not in stale, f"stale label in {ns}/{vid}/{field}: {val}"

        # 6. Rollback via import
        import_index(dev_name, export_path)

        # 7. Post-rollback snapshot — must match pre exactly (incl. metadata)
        rolled_back = _snapshot_index(dev_index)
        for ns, pre_info in pre.items():
            assert rolled_back[ns]["count"] == pre_info["count"]
            assert rolled_back[ns]["ids"] == pre_info["ids"]
            # Metadata equality across all IDs
            for vid, pre_md in pre_info["meta_by_id"].items():
                assert rolled_back[ns]["meta_by_id"][vid] == pre_md, (
                    f"metadata drift after rollback in {ns}/{vid}"
                )
