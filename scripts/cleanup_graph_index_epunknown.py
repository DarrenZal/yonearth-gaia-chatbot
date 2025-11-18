#!/usr/bin/env python3
"""
Cleanup script to remove legacy 'epunknown:*' chunks from the merged graph index.

Edits these files under data/graph_index:
- chunk_entity_map.json: remove any entries whose chunk_id startswith 'epunknown:'
- entity_chunk_map.json: remove any chunk_ids that startwith 'epunknown:' from lists
- hierarchy.json: remove any keys starting with 'epunknown:'
- previews.json: remove any keys starting with 'epunknown:'

Runs in-place and prints counts of removed items.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / 'data' / 'graph_index'


def read_json(p: Path):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(p: Path, data):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cleanup():
    target_prefix = 'epunknown:'
    cem_p = INDEX_DIR / 'chunk_entity_map.json'
    ecm_p = INDEX_DIR / 'entity_chunk_map.json'
    hier_p = INDEX_DIR / 'hierarchy.json'
    prev_p = INDEX_DIR / 'previews.json'

    removed_cem = 0
    removed_ecm_refs = 0
    removed_hier = 0
    removed_prev = 0

    # chunk_entity_map.json
    try:
        cem = read_json(cem_p)
        keys = list(cem.keys())
        for k in keys:
            if isinstance(k, str) and k.startswith(target_prefix):
                del cem[k]
                removed_cem += 1
        write_json(cem_p, cem)
    except Exception:
        pass

    # entity_chunk_map.json
    try:
        ecm = read_json(ecm_p)
        for ent, lst in list(ecm.items()):
            if not isinstance(lst, list):
                continue
            new_lst = [cid for cid in lst if not (isinstance(cid, str) and cid.startswith(target_prefix))]
            removed_ecm_refs += (len(lst) - len(new_lst))
            ecm[ent] = new_lst
        write_json(ecm_p, ecm)
    except Exception:
        pass

    # hierarchy.json
    try:
        hier = read_json(hier_p)
        keys = list(hier.keys())
        for k in keys:
            if isinstance(k, str) and k.startswith(target_prefix):
                del hier[k]
                removed_hier += 1
        write_json(hier_p, hier)
    except Exception:
        pass

    # previews.json
    try:
        prev = read_json(prev_p)
        keys = list(prev.keys())
        for k in keys:
            if isinstance(k, str) and k.startswith(target_prefix):
                del prev[k]
                removed_prev += 1
        write_json(prev_p, prev)
    except Exception:
        pass

    print(json.dumps({
        'removed_chunk_entity_map': removed_cem,
        'removed_entity_chunk_map_refs': removed_ecm_refs,
        'removed_hierarchy': removed_hier,
        'removed_previews': removed_prev,
        'index_dir': str(INDEX_DIR)
    }, indent=2))


if __name__ == '__main__':
    cleanup()

