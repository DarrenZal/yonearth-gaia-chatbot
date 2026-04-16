"""
Plan test #7 — schema fidelity of GET /api/taxonomy (AC #18).

Exercises the actual FastAPI route via TestClient to verify the wire contract:
5 pillars, 22 secondaries (no duplicates), every secondary has a description,
and Aaron's spreadsheet spellings are present verbatim.
"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OFFICIAL_PILLARS = {"COMMUNITY", "CULTURE", "ECONOMY", "ECOLOGY", "HEALTH"}


@pytest.fixture(scope="module")
def client():
    """Build a minimal FastAPI app that mounts only the taxonomy router.

    Avoids pulling in the full main.py, which eagerly initializes the RAG chain
    (OpenAI + Pinecone), so tests run without external credentials.
    """
    from src.api.taxonomy_endpoint import router as taxonomy_router

    app = FastAPI()
    app.include_router(taxonomy_router)
    return TestClient(app)


def test_endpoint_returns_200(client):
    resp = client.get("/api/taxonomy")
    assert resp.status_code == 200


def test_five_pillars(client):
    data = client.get("/api/taxonomy").json()
    assert len(data["pillars"]) == 5
    assert set(data["pillars"]) == OFFICIAL_PILLARS


def test_twenty_two_secondaries_no_duplicates(client):
    data = client.get("/api/taxonomy").json()
    all_secondaries = [s for secs in data["taxonomy"].values() for s in secs]
    assert len(all_secondaries) == 22
    assert len(set(all_secondaries)) == 22


def test_every_pillar_key_is_official(client):
    data = client.get("/api/taxonomy").json()
    assert set(data["taxonomy"].keys()) == OFFICIAL_PILLARS


def test_every_secondary_has_a_description(client):
    data = client.get("/api/taxonomy").json()
    all_secondaries = {s for secs in data["taxonomy"].values() for s in secs}
    description_keys = set(data["descriptions"].keys())
    assert description_keys == all_secondaries
    assert len(description_keys) == 22


def test_aaron_spellings_verbatim(client):
    data = client.get("/api/taxonomy").json()
    all_secondaries = {s for secs in data["taxonomy"].values() for s in secs}
    for must_have in ("POLICY & GOVERNMT", "PERMA-CULTURE", "SUSTAIN-ABILITY"):
        assert must_have in all_secondaries
        assert must_have in data["descriptions"]


def test_removed_labels_absent(client):
    """AC #9: zero occurrences of REGENERATIVE / COMPOSTING / WATER / ENERGY as labels."""
    data = client.get("/api/taxonomy").json()
    all_secondaries = {s for secs in data["taxonomy"].values() for s in secs}
    description_keys = set(data["descriptions"].keys())
    for stale in ("REGENERATIVE", "COMPOSTING", "WATER", "ENERGY"):
        assert stale not in all_secondaries
        assert stale not in description_keys


def test_regen_social_enterprise_replaces_regenerative(client):
    data = client.get("/api/taxonomy").json()
    all_secondaries = [s for secs in data["taxonomy"].values() for s in secs]
    assert "REGEN / SOCIAL ENTERPRISE" in all_secondaries
    # And it lives under ECONOMY per the plan mapping
    assert "REGEN / SOCIAL ENTERPRISE" in data["taxonomy"]["ECONOMY"]
