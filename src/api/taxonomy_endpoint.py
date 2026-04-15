"""
Taxonomy endpoint — serves data/yoe_taxonomy.json as the single runtime source
for both backend consumers and the frontend Guide UI.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["taxonomy"])

TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "yoe_taxonomy.json"


def _load_taxonomy_from_disk(path: Optional[Path] = None) -> Dict:
    """Load the YOE taxonomy JSON. Fail fast on missing/malformed file."""
    if path is None:
        path = TAXONOMY_PATH
    if not path.exists():
        raise RuntimeError(
            f"yoe_taxonomy.json not found at {path}; deploy artifact is incomplete"
        )
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"yoe_taxonomy.json at {path} is malformed: {e}"
        ) from e
    for key in ("pillars", "taxonomy", "descriptions"):
        if key not in data:
            raise RuntimeError(
                f"yoe_taxonomy.json missing required top-level key '{key}' (path: {path})"
            )
    return data


# Load once at import (fail-fast on malformed/missing). The JSON file is frozen
# at deploy time — edits require a redeploy, so in-process caching is fine.
_TAXONOMY_CACHE: Dict = _load_taxonomy_from_disk()


@router.get("/taxonomy")
async def get_taxonomy() -> Dict:
    """Return the full YOE taxonomy (pillars, secondary→pillar mapping, descriptions)."""
    return _TAXONOMY_CACHE
