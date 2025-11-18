"""
GraphRAG index loader and helpers for fast entity-centric retrieval.
Loads precomputed JSON files from settings.data_dir / graph_index.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings


@dataclass
class GraphIndex:
    entities_lexicon: Dict[str, Any]
    chunk_entity_map: Dict[str, List[str]]
    entity_chunk_map: Dict[str, List[str]]
    hierarchy: Dict[str, Any]
    adjacency: Dict[str, List[Dict[str, Any]]]
    previews: Dict[str, str] = field(default_factory=dict)
    # Optional semantic cluster/topic index: expected shape
    # {"clusters": {"cluster_id": {"chunk_ids": [...], "summary": str, "embedding": [float, ...]}, ...}}
    cluster_index: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "GraphIndex":
        base = base_dir or (settings.data_dir / "graph_index")

        def read_json(name: str) -> Any:
            path = base / name
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        entities_lexicon = read_json("entities_lexicon.json")
        chunk_entity_map = read_json("chunk_entity_map.json")
        entity_chunk_map = read_json("entity_chunk_map.json")
        hierarchy = read_json("hierarchy.json")
        adjacency = read_json("adjacency.json")
        # Optional previews.json for lightweight text snippets
        try:
            previews: Dict[str, str] = read_json("previews.json")
        except Exception:
            previews = {}

        # Optional cluster_index.json for semantic cluster/topic retrieval
        try:
            cluster_index: Dict[str, Any] = read_json("cluster_index.json")
        except Exception:
            cluster_index = {}

        return cls(
            entities_lexicon=entities_lexicon,
            chunk_entity_map=chunk_entity_map,
            entity_chunk_map=entity_chunk_map,
            hierarchy=hierarchy,
            adjacency=adjacency,
            previews=previews,
            cluster_index=cluster_index,
        )

    def alias_index(self) -> Dict[str, str]:
        return self.entities_lexicon.get("alias_index", {})

    def find_entities_in_text(self, text: str) -> List[str]:
        """Extract entity IDs by simple lexical match of names/aliases in the text.

        Applies lightweight guards to reduce noise (skip very short/common aliases),
        and optionally a small fuzzy/semantic fallback for multi-word aliases.
        """
        text_lower = (text or "").lower()
        found: List[str] = []
        seen: set[str] = set()
        stop_aliases = {"i", "p", "dc", "and", "or", "the", "in", "on", "of", "a", "an"}

        def _tokenize(s: str) -> List[str]:
            return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]

        # Pre-tokenize query text once for fuzzy matching
        query_tokens = set(_tokenize(text_lower))

        for alias_lower, eid in self.alias_index().items():
            al = (alias_lower or "").strip().lower()
            if not al:
                continue
            if al in stop_aliases or len(al) < 3:
                continue
            # Require at least one alphabetic character
            if not any(c.isalpha() for c in al):
                continue

            # Exact/substring match first (fast path)
            if al in text_lower:
                if eid not in seen:
                    seen.add(eid)
                    found.append(eid)
                continue

            # Optional semantic/fuzzy fallback for multi-word aliases
            if not settings.graph_enable_semantic_entity_match:
                continue

            alias_tokens = _tokenize(al)
            # Only attempt fuzzy matching for aliases with 2+ tokens to avoid noisy short matches
            if len(alias_tokens) < 2:
                continue

            if not alias_tokens or not query_tokens:
                continue

            intersection = len(query_tokens.intersection(alias_tokens))
            if not intersection:
                continue
            union = len(query_tokens.union(alias_tokens))
            jaccard = intersection / union if union else 0.0

            if jaccard >= settings.graph_semantic_entity_jaccard_threshold and eid not in seen:
                seen.add(eid)
                found.append(eid)
        return found
