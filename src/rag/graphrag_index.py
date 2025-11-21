"""
GraphRAG index loader and helpers for fast entity-centric retrieval.
Loads precomputed JSON files from settings.data_dir / graph_index.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..config import settings


@dataclass
class GraphIndex:
    entities_lexicon: Dict[str, Any]
    chunk_entity_map: Dict[str, List[str]]
    entity_chunk_map: Dict[str, List[str]]
    hierarchy: Dict[str, Any]
    adjacency: Dict[str, List[Dict[str, Any]]]

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
        return cls(
            entities_lexicon=entities_lexicon,
            chunk_entity_map=chunk_entity_map,
            entity_chunk_map=entity_chunk_map,
            hierarchy=hierarchy,
            adjacency=adjacency,
        )

    def alias_index(self) -> Dict[str, str]:
        return self.entities_lexicon.get("alias_index", {})

    def find_entities_in_text(self, text: str) -> List[str]:
        """Extract entity IDs by simple lexical match of names/aliases in the text."""
        text_lower = text.lower()
        found: List[str] = []
        seen = set()
        for alias_lower, eid in self.alias_index().items():
            if alias_lower in text_lower and eid not in seen:
                seen.add(eid)
                found.append(eid)
        return found

