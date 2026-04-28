#!/usr/bin/env python3
"""
patch_kg_domain_colors.py — swap old bright domain hex codes for the new
earth-tone palette across visualization_data.json (top-level `domains` array
+ every node's `domain_colors` array).

Apr 28 mapping (from Aaron's earth-tones palette JPG):
  Community: #4CAF50 → #E1CAB2
  Culture:   #9C27B0 → #BA986D
  Economy:   #FF9800 → #829591
  Ecology:   #2196F3 → #86A37C
  Health:    #F44336 → #AF9D66

Usage:
  python3 scripts/patch_kg_domain_colors.py [--input PATH] [--output PATH]

Default: read & write data/knowledge_graph/visualization_data.json (in place,
with .bak backup written first).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO / "data" / "knowledge_graph" / "visualization_data.json"

OLD_TO_NEW = {
    # Old hex (uppercase to match file) → new hex
    "#4CAF50": "#E1CAB2",  # community
    "#9C27B0": "#BA986D",  # culture
    "#FF9800": "#829591",  # economy
    "#2196F3": "#86A37C",  # ecology
    "#F44336": "#AF9D66",  # health
}

# Top-level domain entries — overwrite by name, not hex (in case file already
# partially patched).
NAME_TO_NEW = {
    "Community": "#E1CAB2",
    "Culture":   "#BA986D",
    "Economy":   "#829591",
    "Ecology":   "#86A37C",
    "Health":    "#AF9D66",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_PATH))
    ap.add_argument("--output", default=None, help="default: same as input (in-place with .bak)")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output) if args.output else in_path

    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        return 2

    print(f"→ loading {in_path}", file=sys.stderr)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    # 1. Top-level `domains` array — match by name
    top_changed = 0
    for d in data.get("domains", []):
        name = d.get("name")
        if name in NAME_TO_NEW and d.get("color") != NAME_TO_NEW[name]:
            d["color"] = NAME_TO_NEW[name]
            top_changed += 1
    print(f"  top-level `domains` entries patched: {top_changed}", file=sys.stderr)

    # 2. Per-node `domain_colors` arrays — replace by hex (case-insensitive)
    nodes_changed = 0
    arrays_changed = 0
    for n in data.get("nodes", []):
        cols = n.get("domain_colors")
        if not isinstance(cols, list):
            continue
        new_cols = []
        any_changed = False
        for c in cols:
            up = (c or "").upper()
            if up in OLD_TO_NEW:
                new_cols.append(OLD_TO_NEW[up])
                any_changed = True
            else:
                new_cols.append(c)
        if any_changed:
            n["domain_colors"] = new_cols
            arrays_changed += 1
            nodes_changed += 1
    print(f"  per-node `domain_colors` arrays patched: {arrays_changed}", file=sys.stderr)

    # Backup before write
    if out_path == in_path:
        bak = in_path.with_suffix(in_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(in_path, bak)
            print(f"  backup → {bak.name}", file=sys.stderr)

    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✓ wrote {out_path}", file=sys.stderr)

    # Verify
    fresh = json.loads(out_path.read_text(encoding="utf-8"))
    print("  verification — top-level domains now:", file=sys.stderr)
    for d in fresh.get("domains", []):
        print(f"    {d}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
