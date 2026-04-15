"""
Plan test #5 — taxonomy loader fail-fast behavior.

The app must NOT silently boot with an empty or missing taxonomy. Both the RAG
loader and the API-endpoint loader must raise a clear RuntimeError with the
file path in the message when the JSON is absent or malformed.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _get_loaders():
    """Import both loaders lazily so import errors surface inside the test."""
    from src.api.taxonomy_endpoint import _load_taxonomy_from_disk
    from src.rag.semantic_category_matcher import _load_taxonomy

    return _load_taxonomy, _load_taxonomy_from_disk


def test_missing_file_raises_with_path_in_message(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    rag_load, api_load = _get_loaders()

    with pytest.raises(RuntimeError) as exc_rag:
        rag_load(missing)
    assert str(missing) in str(exc_rag.value)
    assert "not found" in str(exc_rag.value)

    with pytest.raises(RuntimeError) as exc_api:
        api_load(missing)
    assert str(missing) in str(exc_api.value)
    assert "not found" in str(exc_api.value)


def test_malformed_json_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    rag_load, api_load = _get_loaders()

    with pytest.raises(RuntimeError) as exc_rag:
        rag_load(bad)
    assert "malformed" in str(exc_rag.value).lower()
    assert str(bad) in str(exc_rag.value)

    with pytest.raises(RuntimeError) as exc_api:
        api_load(bad)
    assert "malformed" in str(exc_api.value).lower()


def test_missing_required_top_level_key_raises(tmp_path):
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"pillars": [], "taxonomy": {}}))  # missing descriptions
    rag_load, api_load = _get_loaders()

    with pytest.raises(RuntimeError) as exc_rag:
        rag_load(incomplete)
    assert "descriptions" in str(exc_rag.value)

    with pytest.raises(RuntimeError) as exc_api:
        api_load(incomplete)
    assert "descriptions" in str(exc_api.value)


def test_valid_file_loads_five_pillars_and_twenty_two_secondaries():
    """Sanity-check the committed data/yoe_taxonomy.json."""
    rag_load, api_load = _get_loaders()
    data = rag_load()

    assert data["pillars"] == ["COMMUNITY", "CULTURE", "ECONOMY", "ECOLOGY", "HEALTH"]
    all_secondaries = [s for secs in data["taxonomy"].values() for s in secs]
    assert len(all_secondaries) == 22
    assert len(set(all_secondaries)) == 22  # no duplicates
    assert set(data["descriptions"].keys()) == set(all_secondaries)

    # Aaron's spreadsheet spellings — verbatim
    assert "POLICY & GOVERNMT" in all_secondaries
    assert "PERMA-CULTURE" in all_secondaries
    assert "SUSTAIN-ABILITY" in all_secondaries


def test_module_level_constants_populated():
    """semantic_category_matcher exposes CATEGORY_DESCRIPTIONS, SECONDARY_TO_PILLAR, YOE_PILLARS."""
    from src.rag import semantic_category_matcher as m

    assert len(m.CATEGORY_DESCRIPTIONS) == 22
    assert len(m.SECONDARY_TO_PILLAR) == 22
    assert m.YOE_PILLARS == ["COMMUNITY", "CULTURE", "ECONOMY", "ECOLOGY", "HEALTH"]
    # Removed-from-vocab labels must NOT appear
    for stale in ("REGENERATIVE", "COMPOSTING", "WATER", "ENERGY", "PERMACULTURE", "SUSTAINABILITY"):
        assert stale not in m.CATEGORY_DESCRIPTIONS
    # Pillar assignment sanity-check
    assert m.SECONDARY_TO_PILLAR["SOIL"] == "ECOLOGY"
    assert m.SECONDARY_TO_PILLAR["REGEN / SOCIAL ENTERPRISE"] == "ECONOMY"
