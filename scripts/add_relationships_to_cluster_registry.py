#!/usr/bin/env python3
"""
Populate cluster_registry.json with relationships from unified_v2.json.

This is a viewer-side helper: the existing cluster_registry has entity lists
but empty relationships, so edges do not render in the 3D viewer. We reuse the
already-processed unified_v2 graph and merge map to normalize IDs and attach
edges per cluster.

Defaults target the deployed viewer path; override with --registry-path/--output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
DEFAULT_UNIFIED_PATH = ROOT / "data/knowledge_graph_unified/unified_v2.json"
DEFAULT_MERGES_PATH = ROOT / "data/knowledge_graph_unified/entity_merges.json"
DEFAULT_REGISTRY_PATH = Path("/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/cluster_registry.json")
DEFAULT_OUTPUT_PATH = ROOT / "data/graphrag_hierarchy/cluster_registry_with_relationships.json"


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def build_alias_map(entities: Dict[str, Dict], merges: Dict[str, Any]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for eid, data in entities.items():
        alias_map[eid.lower()] = eid
        name = data.get("name")
        if name:
            alias_map[name.lower()] = eid
        for alias in data.get("aliases", []) or []:
            alias_map[alias.lower()] = eid

    merge_map = merges.get("merges", merges if isinstance(merges, dict) else {})
    for old, canonical in merge_map.items():
        if old == "stats":
            continue
        alias_map[str(old).lower()] = str(canonical)
    return alias_map


def normalize_name(name: str, alias_map: Dict[str, str]) -> str | None:
    if not name:
        return None
    return alias_map.get(name.lower())


def normalize_relationships(relationships: List[Dict], alias_map: Dict[str, str]) -> List[Dict]:
    dedup: Dict[Tuple[str, str, str], Dict] = {}
    for rel in relationships:
        src = normalize_name(rel.get("source"), alias_map)
        tgt = normalize_name(rel.get("target"), alias_map)
        if not src or not tgt:
            continue
        predicate = rel.get("predicate") or "RELATES_TO"
        key = (src, predicate, tgt)
        entry = dedup.setdefault(
            key,
            {"source": src, "predicate": predicate, "target": tgt, "mention_count": 0, "sources": set()},
        )
        entry["mention_count"] += rel.get("mention_count", 1)
        for s in rel.get("sources", []) or []:
            entry["sources"].add(s)

    normalized = []
    for entry in dedup.values():
        normalized.append(
            {
                "source": entry["source"],
                "predicate": entry["predicate"],
                "target": entry["target"],
                "mention_count": entry["mention_count"],
                "sources": sorted(entry["sources"]),
            }
        )
    return normalized


def main(args: argparse.Namespace):
    unified = load_json(args.unified_path)
    entities = unified.get("entities", {})
    relationships = unified.get("relationships", [])
    merges = load_json(args.merges_path) if args.merges_path.exists() else {}
    alias_map = build_alias_map(entities, merges)
    normalized_rels = normalize_relationships(relationships, alias_map)

    registry = load_json(args.registry_path)

    # Pre-index relationships by endpoint to reduce scanning.
    rel_by_entity: Dict[str, List[Dict]] = {}
    for rel in normalized_rels:
        rel_by_entity.setdefault(rel["source"], []).append(rel)
        rel_by_entity.setdefault(rel["target"], []).append(rel)

    for cluster_id, cluster in registry.items():
        ent_ids: Set[str] = set()
        for ent in cluster.get("entities", []):
            canon = normalize_name(ent, alias_map) or ent
            ent_ids.add(canon)

        # Collect edges where both endpoints are inside the cluster entity set.
        seen_keys: Set[Tuple[str, str, str]] = set()
        rels: List[Dict] = []
        for eid in ent_ids:
            for rel in rel_by_entity.get(eid, []):
                key = (rel["source"], rel["predicate"], rel["target"])
                if key in seen_keys:
                    continue
                if rel["source"] in ent_ids and rel["target"] in ent_ids:
                    rels.append(rel)
                    seen_keys.add(key)

        cluster["relationships"] = rels

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(registry, f, indent=2)
    print(f"Cluster registry with relationships saved to: {args.output}")
    print(f"Clusters processed: {len(registry)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate cluster_registry.json with relationships from unified_v2.")
    parser.add_argument("--unified-path", type=Path, default=DEFAULT_UNIFIED_PATH, help="Path to unified_v2.json")
    parser.add_argument("--merges-path", type=Path, default=DEFAULT_MERGES_PATH, help="Path to entity_merges.json")
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH, help="Path to cluster_registry.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for enriched cluster registry (do not overwrite live file unless ready)",
    )
    args = parser.parse_args()
    main(args)
