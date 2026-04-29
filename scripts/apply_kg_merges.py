#!/usr/bin/env python3
"""
apply_kg_merges.py — apply approved entity merges from candidate_merges.csv
into visualization_data.json.

Inputs:
  data/candidate_merges.csv                           (status filled in)
  data/knowledge_graph/visualization_data.json        (canonical KG)
  data/knowledge_graph/visualization_data.json.bak    (existing pre-color-patch
                                                       backup; we add a fresh
                                                       pre-merge backup too)
  data/yoe_canonical_entities.json                    (alias map; for canonical
                                                       name overrides)
  /Users/darrenzal/projects/ecoscene/aaron-2026-04-28/YOE Sponsors & Partners.xlsx
                                                      (for is_sponsor + is_yoe_enterprise flags)

Output:
  data/knowledge_graph/visualization_data.json        (rewritten in place)
  data/knowledge_graph/visualization_data.json.pre-merge.bak (new backup)
  data/knowledge_graph/merge_log.json                 (audit trail)

Algorithm per cluster (where ≥1 member has status=accept):
  1. Identify canonical node: the row in cluster with status='canonical'.
  2. Collect member nodes whose status='accept'.
  3. For each member:
       - Add member.name to canonical.aliases (if not already present)
       - Add member.aliases to canonical.aliases
       - Union member.episodes into canonical.episodes
       - canonical.mention_count += member.mention_count
       - canonical.importance = max(canonical.importance, member.importance)
       - Union member.domains and aligned domain_colors into canonical
       - Append a merge_provenance entry under canonical
  4. Repoint all `links[]` with source=member.id or target=member.id → canonical.id
  5. Drop self-links and dedupe (source, target, type) tuples.
  6. Delete member node.
  7. If alias map specifies a different canonical_name (e.g., 'Aaron Perry' vs
     mc-winner 'Aaron William Perry'), rename canonical node and push the old
     name into aliases.

Sponsor flagging:
  - For all 38 canonical sponsors from xlsx: set is_sponsor=true on the matched
    KG node (after all merges).
  - For 3 'YES' YOE enterprises (Earth Water Press, Soil Werks, Wele Waters):
    set is_yoe_enterprise=true.

Verification (built-in):
  - pre/post id count delta == accepted-merge count
  - every node still has required keys
  - no orphan node references in links

Usage:
  python3 scripts/apply_kg_merges.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
KG_PATH = REPO / "data" / "knowledge_graph" / "visualization_data.json"
BAK_PRE_MERGE = REPO / "data" / "knowledge_graph" / "visualization_data.json.pre-merge.bak"
CSV_PATH = REPO / "data" / "candidate_merges.csv"
ALIAS_PATH = REPO / "data" / "yoe_canonical_entities.json"
SPONSOR_XLSX = pathlib.Path(
    "/Users/darrenzal/projects/ecoscene/aaron-2026-04-28/YOE Sponsors & Partners.xlsx"
)
LOG_PATH = REPO / "data" / "knowledge_graph" / "merge_log.json"

REQUIRED_NODE_KEYS = {"id", "name", "type"}


def load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: pathlib.Path, obj: dict):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_sponsors() -> tuple[set[str], set[str]]:
    """Return (sponsor_names, yoe_enterprise_names)."""
    try:
        import openpyxl  # type: ignore
    except ImportError:
        print("WARN: openpyxl not installed; skipping sponsor flagging", file=sys.stderr)
        return set(), set()
    wb = openpyxl.load_workbook(SPONSOR_XLSX, data_only=True)
    ws = wb.active
    sponsors: set[str] = set()
    yoe_ent: set[str] = set()
    for r in ws.iter_rows(values_only=True):
        if not r or not r[0]:
            continue
        name = str(r[0]).strip()
        if (not name) or name.upper().startswith("Y ON EARTH COMMUNITY SPONSORS") or name.startswith("YOE Enterprise"):
            continue
        sponsors.add(name)
        flag = (str(r[1] if len(r) > 1 else "")).strip().upper() if (len(r) > 1 and r[1]) else ""
        if flag == "YES":
            yoe_ent.add(name)
    return sponsors, yoe_ent


def parse_csv() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Return (canon_by_cluster, members_by_cluster) where members include
    ONLY rows with status=accept.
    """
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    canon: dict[str, dict] = {}
    members: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cid = r["cluster_id"]
        if r.get("status") == "canonical":
            canon[cid] = r
        elif r.get("status") == "accept":
            members[cid].append(r)
    return canon, dict(members)


def union_keep_order(*lists: list) -> list:
    seen: set = set()
    out: list = []
    for L in lists:
        for x in (L or []):
            if x is None:
                continue
            key = json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x
            if key not in seen:
                seen.add(key)
                out.append(x)
    return out


def union_domains_aligned(canonical: dict, member: dict) -> tuple[list[str], list[str]]:
    """Union (domains, domain_colors) preserving alignment. domain_colors[i]
    corresponds to domains[i]. Use canonical's color if present; else member's."""
    canon_d = canonical.get("domains", []) or []
    canon_c = canonical.get("domain_colors", []) or []
    mem_d = member.get("domains", []) or []
    mem_c = member.get("domain_colors", []) or []

    by_name: dict[str, str] = {}
    for d, c in zip(canon_d, canon_c):
        by_name[d] = c
    for d, c in zip(mem_d, mem_c):
        by_name.setdefault(d, c)

    domains = list(by_name.keys())
    colors = [by_name[d] for d in domains]
    return domains, colors


def apply_merges(kg: dict, canon_rows: dict, members_rows: dict, alias_map: dict) -> dict:
    """Mutate kg in place. Return audit log dict."""
    nodes_by_id: dict[str, dict] = {n["id"]: n for n in kg["nodes"]}
    links: list[dict] = kg.get("links", [])

    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "merges": [],
        "rename": [],
        "sponsor_flags": [],
        "yoe_enterprise_flags": [],
    }

    # Build alias_to_canon: (lower_name, type) → canonical_name
    alias_to_canon: dict[tuple[str, str], str] = {}
    for type_, mappings in alias_map.items():
        if str(type_).startswith("_"):
            continue
        for canon_name, aliases in mappings.items():
            alias_to_canon[(canon_name.lower().strip(), type_)] = canon_name
            for a in aliases:
                alias_to_canon[(a.lower().strip(), type_)] = canon_name

    # Step 1: cluster-by-cluster merge
    redirect: dict[str, str] = {}  # member_id → canonical_id
    for cid, members in members_rows.items():
        canon_row = canon_rows.get(cid)
        if not canon_row:
            print(f"  WARN: cluster {cid} has accepted members but no canonical row; skipping", file=sys.stderr)
            continue
        canon_id = canon_row["canonical_id"]
        canon = nodes_by_id.get(canon_id)
        if not canon:
            print(f"  WARN: cluster {cid} canonical_id={canon_id!r} not found in nodes; skipping", file=sys.stderr)
            continue

        merge_record = {
            "cluster_id": cid,
            "canonical_id": canon_id,
            "canonical_name_before": canon.get("name"),
            "merged": [],
        }

        # Provenance log on canonical (preserves what was merged in)
        prov = canon.setdefault("merge_provenance", [])

        for m_row in members:
            mid = m_row["member_id"]
            if mid == canon_id:
                continue  # safety
            mem = nodes_by_id.get(mid)
            if not mem:
                continue

            # Aliases
            new_aliases = (canon.get("aliases") or [])
            new_aliases = union_keep_order(new_aliases, [mem.get("name")], mem.get("aliases") or [])
            canon["aliases"] = [a for a in new_aliases if a and a != canon.get("name")]

            # Episodes
            canon_eps = canon.get("episodes") or []
            mem_eps = mem.get("episodes") or []
            canon["episodes"] = sorted(set(canon_eps) | set(mem_eps), key=lambda x: (isinstance(x, str), x))
            canon["episode_count"] = len(canon["episodes"])

            # Counts
            canon["mention_count"] = (canon.get("mention_count", 0) or 0) + (mem.get("mention_count", 0) or 0)

            # Importance
            canon["importance"] = max(canon.get("importance", 0) or 0, mem.get("importance", 0) or 0)

            # Domains + colors aligned
            canon["domains"], canon["domain_colors"] = union_domains_aligned(canon, mem)

            # Provenance entry
            prov.append({
                "from_id": mid,
                "from_name": mem.get("name"),
                "from_mention_count": mem.get("mention_count", 0),
                "from_episodes": mem_eps,
                "match_via": m_row.get("match_via", ""),
                "fuzzy_or_semantic_score": m_row.get("fuzzy_or_semantic_score", ""),
                "auto_reason": m_row.get("auto_reason", ""),
            })
            merge_record["merged"].append({
                "id": mid,
                "name": mem.get("name"),
                "auto_reason": m_row.get("auto_reason", ""),
            })
            redirect[mid] = canon_id
            del nodes_by_id[mid]

        # Optional: rename canonical to alias-map preferred name
        canon_name_csv = (canon_row.get("canonical_name") or "").strip()
        if canon_name_csv and canon_name_csv != canon.get("name"):
            old_name = canon.get("name", "")
            canon["name"] = canon_name_csv
            new_aliases = canon.get("aliases") or []
            if old_name and old_name not in new_aliases:
                canon["aliases"] = [old_name] + new_aliases
            log["rename"].append({"cluster_id": cid, "id": canon_id, "from": old_name, "to": canon_name_csv})

        # Also normalize via alias map if applicable
        canon_lower = (canon.get("name") or "").lower().strip()
        canon_type = canon.get("type", "")
        mapped = alias_to_canon.get((canon_lower, canon_type))
        if mapped and mapped != canon.get("name"):
            old_name = canon.get("name", "")
            canon["name"] = mapped
            new_aliases = canon.get("aliases") or []
            if old_name and old_name not in new_aliases:
                canon["aliases"] = [old_name] + new_aliases
            log["rename"].append({"cluster_id": cid, "id": canon_id, "from": old_name, "to": mapped})

        merge_record["canonical_name_after"] = canon.get("name")
        log["merges"].append(merge_record)

    # Step 2: rewrite kg.nodes from nodes_by_id
    kg["nodes"] = list(nodes_by_id.values())

    # Step 3: repoint links from removed members to their canonicals; drop self-loops + dedupe
    new_links: list[dict] = []
    seen_links: set[tuple] = set()
    dropped_self = 0
    deduped = 0
    repointed = 0
    for link in links:
        # source/target may be string id (in viz_data they typically are)
        src = link.get("source")
        tgt = link.get("target")
        # Sometimes link objects in d3-style data carry the resolved node dict;
        # for safety, accept both.
        if isinstance(src, dict): src = src.get("id")
        if isinstance(tgt, dict): tgt = tgt.get("id")
        new_src = redirect.get(src, src)
        new_tgt = redirect.get(tgt, tgt)
        if new_src != src or new_tgt != tgt:
            repointed += 1
        if new_src == new_tgt:
            dropped_self += 1
            continue
        # Skip if either side now references a non-existent node (orphan)
        if new_src not in nodes_by_id or new_tgt not in nodes_by_id:
            dropped_self += 1
            continue
        rel = link.get("type") or link.get("relationship") or ""
        key = (new_src, new_tgt, rel)
        if key in seen_links:
            deduped += 1
            continue
        seen_links.add(key)
        new_link = dict(link)
        new_link["source"] = new_src
        new_link["target"] = new_tgt
        new_links.append(new_link)
    kg["links"] = new_links
    log["link_stats"] = {
        "before": len(links),
        "after": len(new_links),
        "repointed": repointed,
        "dropped_orphan_or_self": dropped_self,
        "deduped": deduped,
    }

    return log


def flag_sponsors(kg: dict, sponsor_names: set[str], yoe_enterprise: set[str], log: dict):
    sponsor_lower = {s.lower(): s for s in sponsor_names}
    enterprise_lower = {s.lower(): s for s in yoe_enterprise}
    for n in kg["nodes"]:
        nm_lower = (n.get("name") or "").lower().strip()
        all_aliases = [(a or "").lower().strip() for a in (n.get("aliases") or [])]
        if nm_lower in sponsor_lower or any(a in sponsor_lower for a in all_aliases):
            if not n.get("is_sponsor"):
                n["is_sponsor"] = True
                log["sponsor_flags"].append(n["id"])
        if nm_lower in enterprise_lower or any(a in enterprise_lower for a in all_aliases):
            if not n.get("is_yoe_enterprise"):
                n["is_yoe_enterprise"] = True
                log["yoe_enterprise_flags"].append(n["id"])


def validate(kg: dict, expected_drop: int, pre_id_count: int) -> list[str]:
    errors: list[str] = []
    nodes = kg["nodes"]
    post_id_count = len(nodes)
    if (pre_id_count - post_id_count) != expected_drop:
        errors.append(
            f"id-drop mismatch: pre={pre_id_count} post={post_id_count} "
            f"actual_drop={pre_id_count - post_id_count} expected={expected_drop}"
        )
    seen_ids: set[str] = set()
    for n in nodes:
        for k in REQUIRED_NODE_KEYS:
            if k not in n:
                errors.append(f"node {n.get('id','?')!r} missing required key: {k}")
        nid = n.get("id")
        if nid in seen_ids:
            errors.append(f"duplicate node id: {nid!r}")
        seen_ids.add(nid)
    # Link integrity
    for link in kg.get("links", []):
        for side in ("source", "target"):
            v = link.get(side)
            if isinstance(v, dict):
                v = v.get("id")
            if v not in seen_ids:
                errors.append(f"link references unknown node {side}={v!r}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not KG_PATH.exists() or not CSV_PATH.exists() or not ALIAS_PATH.exists():
        print("ERROR: missing inputs", file=sys.stderr); return 2

    print(f"→ loading {KG_PATH.name}", file=sys.stderr)
    kg = load_json(KG_PATH)
    pre_id_count = len(kg["nodes"])
    pre_link_count = len(kg.get("links", []))
    print(f"  pre: {pre_id_count} nodes, {pre_link_count} links", file=sys.stderr)

    print(f"→ loading CSV decisions", file=sys.stderr)
    canon_rows, members_rows = parse_csv()
    expected_drop = sum(len(v) for v in members_rows.values())
    print(f"  {len(canon_rows)} clusters with canonical rows; {expected_drop} accepted members", file=sys.stderr)

    print(f"→ loading alias map", file=sys.stderr)
    alias_map = load_json(ALIAS_PATH)

    print(f"→ loading sponsor xlsx", file=sys.stderr)
    sponsors, yoe_ent = load_sponsors()
    print(f"  {len(sponsors)} sponsors / {len(yoe_ent)} YOE enterprises", file=sys.stderr)

    print(f"→ applying merges", file=sys.stderr)
    log = apply_merges(kg, canon_rows, members_rows, alias_map)
    print(f"  merged {len(log['merges'])} clusters; {sum(len(m['merged']) for m in log['merges'])} members folded", file=sys.stderr)
    print(f"  links: {log['link_stats']}", file=sys.stderr)
    print(f"  renames: {len(log['rename'])}", file=sys.stderr)

    print(f"→ flagging sponsors", file=sys.stderr)
    flag_sponsors(kg, sponsors, yoe_ent, log)
    print(f"  is_sponsor set on {len(log['sponsor_flags'])} nodes", file=sys.stderr)
    print(f"  is_yoe_enterprise set on {len(log['yoe_enterprise_flags'])} nodes", file=sys.stderr)

    print(f"→ validating", file=sys.stderr)
    errors = validate(kg, expected_drop, pre_id_count)
    if errors:
        print(f"  ✗ {len(errors)} validation errors", file=sys.stderr)
        for e in errors[:20]:
            print(f"    - {e}", file=sys.stderr)
        if not args.dry_run:
            print("  refusing to write — fix script + re-run", file=sys.stderr)
            return 3
    else:
        print(f"  ✓ validation clean", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: no files written", file=sys.stderr)
        return 0

    # Snapshot
    if not BAK_PRE_MERGE.exists():
        shutil.copy2(KG_PATH, BAK_PRE_MERGE)
        print(f"  backup → {BAK_PRE_MERGE.name}", file=sys.stderr)

    # Write KG + log
    write_json(KG_PATH, kg)
    write_json(LOG_PATH, log)
    print(f"\n✓ wrote {KG_PATH}", file=sys.stderr)
    print(f"✓ wrote {LOG_PATH}", file=sys.stderr)
    print(f"\n  post: {len(kg['nodes'])} nodes  {len(kg.get('links',[]))} links", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
