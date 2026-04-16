"""
Plan test #3 — unit tests for the category rewrite logic + mocked namespace coverage.

Covers:
- Rewrite correctness for each table entry
- Deduplication after rewrite (list tagged both SOIL + COMPOSTING -> singular SOIL)
- Idempotency: running rewrite twice on same data -> zero net changes on second pass
- Namespace coverage: a fake Pinecone fixture with vectors in 3 namespaces
  ("", "books", "episodes") -> all 3 processed, assertion counts match.

Does NOT hit the live Pinecone — that's plan test #12.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.migrate_categories import (  # noqa: E402
    REWRITE_TABLE,
    rewrite_metadata,
    rewrite_value,
    migrate,
)


class TestRewriteValue:
    def test_single_string_stale_label_rewritten(self):
        for stale, replacement in REWRITE_TABLE.items():
            new, changed = rewrite_value(stale)
            assert new == replacement
            assert changed is True

    def test_single_string_current_label_unchanged(self):
        for current in ("BIOCHAR", "SOIL", "CLIMATE & SCIENCE", "POLICY & GOVERNMT"):
            new, changed = rewrite_value(current)
            assert new == current
            assert changed is False

    def test_list_of_labels_rewritten(self):
        new, changed = rewrite_value(["REGENERATIVE", "BIOCHAR", "WATER"])
        assert new == ["REGEN / SOCIAL ENTERPRISE", "BIOCHAR", "ECOLOGY & NATURE"]
        assert changed is True

    def test_dedup_after_rewrite(self):
        """Vector tagged both SOIL and COMPOSTING -> singular SOIL."""
        new, changed = rewrite_value(["SOIL", "COMPOSTING"])
        assert new == ["SOIL"]
        assert changed is True

    def test_dedup_preserves_order_of_first_occurrence(self):
        new, changed = rewrite_value(["COMPOSTING", "SOIL", "BIOCHAR"])
        # COMPOSTING rewrites to SOIL; SOIL already present after; so output = [SOIL, BIOCHAR]
        assert new == ["SOIL", "BIOCHAR"]
        assert changed is True

    def test_none_returns_none(self):
        new, changed = rewrite_value(None)
        assert new is None
        assert changed is False

    def test_empty_list_unchanged(self):
        new, changed = rewrite_value([])
        assert new == []
        assert changed is False

    def test_non_string_items_preserved(self):
        """Defensive: if somehow a non-string lurks in the list, pass it through."""
        new, changed = rewrite_value([42, "REGENERATIVE", 42])  # 42 deduped
        assert 42 in new
        assert "REGEN / SOCIAL ENTERPRISE" in new


class TestRewriteMetadata:
    def test_rewrites_categories_field(self):
        md = {"title": "x", "categories": ["REGENERATIVE", "SOIL"]}
        new, changed = rewrite_metadata(md)
        assert new["categories"] == ["REGEN / SOCIAL ENTERPRISE", "SOIL"]
        assert new["title"] == "x"
        assert changed is True

    def test_rewrites_topics_field(self):
        md = {"topics": ["WATER"], "other": "y"}
        new, changed = rewrite_metadata(md)
        assert new["topics"] == ["ECOLOGY & NATURE"]
        assert changed is True

    def test_rewrites_both_fields(self):
        md = {"categories": ["COMPOSTING"], "topics": ["ENERGY"]}
        new, changed = rewrite_metadata(md)
        assert new["categories"] == ["SOIL"]
        assert new["topics"] == ["TECHNOLOGY & MATERIALS"]
        assert changed is True

    def test_unchanged_metadata_reports_no_change(self):
        md = {"title": "x", "categories": ["BIOCHAR", "SOIL"]}
        new, changed = rewrite_metadata(md)
        assert new == md
        assert changed is False

    def test_idempotent_second_pass(self):
        """Running rewrite twice on same data == running once."""
        md = {"categories": ["REGENERATIVE", "BIOCHAR", "COMPOSTING"]}
        once, _ = rewrite_metadata(md)
        twice, second_changed = rewrite_metadata(once)
        assert once == twice
        assert second_changed is False


# ---- Namespace coverage test via mocked Pinecone fixture ----


def _fake_pinecone_client(fixture: dict[str, list[dict]]):
    """Build a fake Pinecone client where `pc.Index(name)` returns a stub index.

    The stub index provides describe_index_stats, list, fetch, upsert per the
    namespace fixture {ns -> [{"id", "values", "metadata"}, ...]}.
    """

    class FakeIndex:
        def __init__(self):
            self.upserts: list[tuple[str, list[dict]]] = []  # (namespace, vectors)

        def describe_index_stats(self):
            return {
                "dimension": 3,
                "total_vector_count": sum(len(v) for v in fixture.values()),
                "namespaces": {ns: {"vector_count": len(vs)} for ns, vs in fixture.items()},
            }

        def list(self, namespace: str = ""):
            yield [v["id"] for v in fixture.get(namespace, [])]

        def fetch(self, ids, namespace: str = ""):
            lookup = {v["id"]: v for v in fixture.get(namespace, [])}
            return {"vectors": {i: lookup[i] for i in ids if i in lookup}}

        def upsert(self, vectors, namespace: str = ""):
            self.upserts.append((namespace, list(vectors)))
            # Mutate fixture so subsequent reads see the rewrite (for idempotency test)
            ns_records = {v["id"]: v for v in fixture.get(namespace, [])}
            for v in vectors:
                ns_records[v["id"]] = v
            fixture[namespace] = list(ns_records.values())
            return {"upserted_count": len(vectors)}

    fake_index = FakeIndex()
    pc = SimpleNamespace(Index=lambda name: fake_index)
    return pc, fake_index


def test_migrate_walks_every_namespace(monkeypatch):
    fixture = {
        "": [
            {"id": "d1", "values": [0.1, 0.2, 0.3], "metadata": {"categories": ["REGENERATIVE"]}},
        ],
        "books": [
            {"id": "b1", "values": [0.1, 0.2, 0.3], "metadata": {"categories": ["COMPOSTING", "SOIL"]}},
            {"id": "b2", "values": [0.1, 0.2, 0.3], "metadata": {"categories": ["BIOCHAR"]}},
        ],
        "episodes": [
            {"id": "e1", "values": [0.1, 0.2, 0.3], "metadata": {"topics": ["WATER", "ENERGY"]}},
        ],
    }
    pc, fake_index = _fake_pinecone_client(fixture)

    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    with patch("scripts.migrate_categories.Pinecone", return_value=pc):
        changed = migrate("fake-index", dry_run=False)

    assert set(changed.keys()) == {"", "books", "episodes"}
    assert changed[""] == 1           # REGENERATIVE -> REGEN / SOCIAL ENTERPRISE
    assert changed["books"] == 1      # b1 had COMPOSTING -> SOIL+dedup; b2 unchanged
    assert changed["episodes"] == 1   # WATER+ENERGY -> ECOLOGY & NATURE + TECHNOLOGY & MATERIALS

    # Verify upsert payload correctness
    all_upserted = [(ns, v) for ns, vs in fake_index.upserts for v in vs]
    ids = {v["id"] for _, v in all_upserted}
    assert ids == {"d1", "b1", "e1"}
    # b2 was unchanged -> should NOT be upserted
    assert "b2" not in ids


def test_migrate_is_idempotent(monkeypatch):
    """Run migrate twice; the second run must report zero changes."""
    fixture = {
        "": [{"id": "d1", "values": [0.1, 0.2, 0.3], "metadata": {"categories": ["REGENERATIVE"]}}],
    }
    pc, _ = _fake_pinecone_client(fixture)

    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    with patch("scripts.migrate_categories.Pinecone", return_value=pc):
        first = migrate("fake-index", dry_run=False)
        second = migrate("fake-index", dry_run=False)

    assert first[""] == 1
    assert second[""] == 0


def test_migrate_raises_on_empty_namespace_map(monkeypatch):
    """Partial coverage is worse than none — abort if describe returns no namespaces."""

    class EmptyIndex:
        def describe_index_stats(self):
            return {"dimension": 3, "total_vector_count": 0, "namespaces": {}}

    pc = SimpleNamespace(Index=lambda name: EmptyIndex())
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    with patch("scripts.migrate_categories.Pinecone", return_value=pc):
        with pytest.raises(RuntimeError, match="no namespaces"):
            migrate("fake-index")


def test_dry_run_performs_no_upserts(monkeypatch):
    fixture = {
        "": [{"id": "d1", "values": [0.1, 0.2, 0.3], "metadata": {"categories": ["REGENERATIVE"]}}],
    }
    pc, fake_index = _fake_pinecone_client(fixture)

    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    with patch("scripts.migrate_categories.Pinecone", return_value=pc):
        changed = migrate("fake-index", dry_run=True)
    assert changed[""] == 1
    assert fake_index.upserts == []
