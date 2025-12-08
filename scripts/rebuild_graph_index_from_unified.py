#!/usr/bin/env python3
"""
Rebuild graph_index JSONs from the processed unified_v2 graph without re-extraction.

Inputs:
  - data/knowledge_graph_unified/unified_v2.json
  - data/knowledge_graph_unified/entity_merges.json
  - data/batch_jobs/results/extraction_results.json
  - data/batch_jobs/parent_chunks.json
  - data/batch_jobs/child_chunks.json (optional)

Outputs (default: data/graph_index_v2/):
  - entities_lexicon.json
  - adjacency.json
  - entity_chunk_map.json
  - chunk_entity_map.json
  - hierarchy.json
  - rebuild_stats.json

The script normalizes relationship endpoints using a lowercased alias/merge map,
deduplicates edges, builds bidirectional adjacency (with inverse predicates),
and maps entities to chunks based on the existing extraction_results.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
DEFAULT_UNIFIED_PATH = ROOT / "data/knowledge_graph_unified/unified_v2.json"
DEFAULT_MERGES_PATH = ROOT / "data/knowledge_graph_unified/entity_merges.json"
DEFAULT_RESULTS_PATH = ROOT / "data/batch_jobs/results/extraction_results.json"
DEFAULT_PARENT_CHUNKS_PATH = ROOT / "data/batch_jobs/parent_chunks.json"
DEFAULT_CHILD_CHUNKS_PATH = ROOT / "data/batch_jobs/child_chunks.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/graph_index_v2"


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def build_alias_map(entities: Dict[str, Dict], merges: Dict[str, Any]) -> Dict[str, str]:
    """Lowercase alias → canonical entity id."""
    alias_map: Dict[str, str] = {}
    for eid, data in entities.items():
        alias_map[eid.lower()] = eid
        name = data.get("name")
        if name:
            alias_map[name.lower()] = eid
        for alias in data.get("aliases", []) or []:
            alias_map[alias.lower()] = eid

    merge_map = merges.get("merges", merges if isinstance(merges, dict) else {})
    # Some merge files include a top-level 'stats' key; skip it.
    for old, canonical in merge_map.items():
        if old == "stats":
            continue
        alias_map[str(old).lower()] = str(canonical)

    return alias_map


def normalize_name(name: str, alias_map: Dict[str, str]) -> str | None:
    if not name:
        return None
    return alias_map.get(name.lower())


def normalize_relationships(
    relationships: List[Dict], alias_map: Dict[str, str]
) -> Tuple[List[Dict], Dict[str, int]]:
    """Canonicalize relationship endpoints, drop unresolved, dedupe, and merge sources."""
    dedup: Dict[Tuple[str, str, str], Dict] = {}
    stats = {"total": 0, "kept": 0, "dropped_unresolved": 0}

    for rel in relationships:
        stats["total"] += 1
        src = normalize_name(rel.get("source"), alias_map)
        tgt = normalize_name(rel.get("target"), alias_map)
        if not src or not tgt:
            stats["dropped_unresolved"] += 1
            continue

        predicate = rel.get("predicate") or "RELATES_TO"
        key = (src, predicate, tgt)
        entry = dedup.setdefault(
            key,
            {
                "source": src,
                "predicate": predicate,
                "target": tgt,
                "mention_count": 0,
                "sources": set(),
            },
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
        stats["kept"] += 1

    return normalized, stats


def build_adjacency(relationships: List[Dict]) -> Dict[str, List[Dict]]:
    """Create bidirectional adjacency with simple contexts."""
    adj: Dict[str, List[Dict]] = defaultdict(list)
    for rel in relationships:
        src = rel["source"]
        tgt = rel["target"]
        pred = rel.get("predicate", "RELATES_TO")
        mention_count = rel.get("mention_count", 1)
        sources = rel.get("sources", [])

        def add_edge(a: str, b: str, predicate: str):
            context = ""
            if sources:
                context = f"{predicate} ({', '.join(sources[:3])})"
            adj[a].append(
                {
                    "neighbor_id": b,
                    "predicate": predicate,
                    "count": mention_count,
                    "p_true": 1.0,
                    "contexts": [context] if context else [],
                }
            )

        add_edge(src, tgt, pred)
        add_edge(tgt, src, f"inverse_of:{pred}")

    return adj


def build_chunk_mappings(
    extraction_results_path: Path, alias_map: Dict[str, str]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, int]]:
    """Build entity→chunk and chunk→entity maps from extraction results."""
    results = load_json(extraction_results_path)
    entity_chunk: Dict[str, Set[str]] = defaultdict(set)
    chunk_entity: Dict[str, Set[str]] = defaultdict(set)
    stats = {"chunks": 0, "entity_mentions": 0, "unresolved_entities": 0}

    for chunk_id, payload in results.items():
        stats["chunks"] += 1
        for ent in payload.get("entities", []):
            stats["entity_mentions"] += 1
            canon = normalize_name(ent.get("name"), alias_map)
            if not canon:
                stats["unresolved_entities"] += 1
                continue
            entity_chunk[canon].add(chunk_id)
            chunk_entity[chunk_id].add(canon)

    # Convert sets to sorted lists
    entity_chunk_map = {k: sorted(v) for k, v in entity_chunk.items()}
    chunk_entity_map = {k: sorted(v) for k, v in chunk_entity.items()}
    return entity_chunk_map, chunk_entity_map, stats


def build_entities_lexicon(entities: Dict[str, Dict], alias_map: Dict[str, str]) -> Dict[str, Any]:
    alias_index = dict(alias_map)
    entities_list = []
    for eid, data in entities.items():
        entities_list.append(
            {
                "id": eid,
                "name": data.get("name", eid),
                "aliases": data.get("aliases", []) or [],
                "type": data.get("type", "UNKNOWN"),
                "freq": data.get("mention_count", 1),
            }
        )
    return {"entities": entities_list, "alias_index": alias_index}


def build_hierarchy(parent_chunks_path: Path, child_chunks_path: Path | None = None) -> Dict[str, Dict]:
    parent_chunks = load_json(parent_chunks_path)
    hierarchy: Dict[str, Dict] = {}
    for parent in parent_chunks:
        chunk_id = parent.get("id")
        if not chunk_id:
            continue
        meta = {
            "source_type": parent.get("source_type"),
            "source_id": parent.get("source_id"),
            "token_count": parent.get("token_count"),
            "content_preview": parent.get("content_preview"),
        }
        meta.update(parent.get("metadata", {}) or {})
        hierarchy[chunk_id] = meta

    # Child chunks are optional; if present, add them too.
    if child_chunks_path and child_chunks_path.exists():
        child_chunks = load_json(child_chunks_path)
        for child in child_chunks:
            cid = child.get("id")
            if not cid:
                continue
            meta = {
                "parent_id": child.get("parent_id"),
                "start_offset": child.get("start_offset"),
                "end_offset": child.get("end_offset"),
                "source_type": child.get("metadata", {}).get("source_type"),
                "source_id": child.get("metadata", {}).get("source_id"),
            }
            hierarchy[cid] = meta

    return hierarchy


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def main(args: argparse.Namespace):
    unified = load_json(args.unified_path)
    entities: Dict[str, Dict] = unified.get("entities", {})
    relationships: List[Dict] = unified.get("relationships", [])
    merges = load_json(args.merges_path) if args.merges_path.exists() else {}

    alias_map = build_alias_map(entities, merges)

    normalized_rels, rel_stats = normalize_relationships(relationships, alias_map)
    adjacency = build_adjacency(normalized_rels)
    entity_chunk_map, chunk_entity_map, chunk_stats = build_chunk_mappings(
        args.extraction_results_path, alias_map
    )
    entities_lexicon = build_entities_lexicon(entities, alias_map)
    hierarchy = build_hierarchy(args.parent_chunks_path, args.child_chunks_path)

    # Write outputs
    out = args.output_dir
    save_json(entities_lexicon, out / "entities_lexicon.json")
    save_json(adjacency, out / "adjacency.json")
    save_json(entity_chunk_map, out / "entity_chunk_map.json")
    save_json(chunk_entity_map, out / "chunk_entity_map.json")
    save_json(hierarchy, out / "hierarchy.json")

    stats = {
        "entities": len(entities),
        "relationships_original": len(relationships),
        "relationships_normalized": len(normalized_rels),
        "relationships_dropped_unresolved": rel_stats["dropped_unresolved"],
        "alias_index_size": len(alias_map),
        "entity_chunk_entries": len(entity_chunk_map),
        "chunk_entity_entries": len(chunk_entity_map),
        "chunks_processed": chunk_stats["chunks"],
        "entity_mentions_total": chunk_stats["entity_mentions"],
        "entity_mentions_unresolved": chunk_stats["unresolved_entities"],
        "adjacency_nodes": len(adjacency),
    }
    save_json(stats, out / "rebuild_stats.json")

    print("Rebuild complete. Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput directory: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild graph_index from unified_v2.")
    parser.add_argument(
        "--unified-path",
        type=Path,
        default=DEFAULT_UNIFIED_PATH,
        help="Path to unified_v2.json",
    )
    parser.add_argument(
        "--merges-path",
        type=Path,
        default=DEFAULT_MERGES_PATH,
        help="Path to entity_merges.json",
    )
    parser.add_argument(
        "--extraction-results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to extraction_results.json",
    )
    parser.add_argument(
        "--parent-chunks-path",
        type=Path,
        default=DEFAULT_PARENT_CHUNKS_PATH,
        help="Path to parent_chunks.json",
    )
    parser.add_argument(
        "--child-chunks-path",
        type=Path,
        default=DEFAULT_CHILD_CHUNKS_PATH,
        help="Path to child_chunks.json (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write graph_index JSONs",
    )
    args = parser.parse_args()
    main(args)
