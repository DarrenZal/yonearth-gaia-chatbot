#!/usr/bin/env python3
"""
Fold remaining Wele/Waylay Waters transcript variants into the canonical
Wele Waters KG node.

This is intentionally narrower than the general candidate merge pipeline:
the April merge already handled the main cluster, but a few one-off spelling
variants remained in visualization_data.json. Keep "Whale Waters Soaking Salts"
separate because that is a product variant with its own node.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
KG_PATH = REPO / "data" / "knowledge_graph" / "visualization_data.json"
LOG_PATH = REPO / "data" / "knowledge_graph" / "merge_log.json"

CANONICAL_ID = "Wele Waters"
LEGACY_CANONICAL_ID = "Waylay Waters"
VARIANT_IDS = [
    LEGACY_CANONICAL_ID,
    "whey-lay waters",
    "Wailay Waters",
    "Willay Waters",
    "Wheylay Waters",
    "WayLayWaters",
    "Whale Waters",
    "Weylay Waters Community",
]


def replace_text(value: str | None) -> str | None:
    if value is None:
        return value
    return value.replace(LEGACY_CANONICAL_ID, CANONICAL_ID)


def union_keep_order(*lists: list) -> list:
    seen = set()
    out = []
    for values in lists:
        for value in values or []:
            if value is None:
                continue
            key = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
    return out


def sorted_episodes(*lists: list) -> list:
    values = set()
    for episodes in lists:
        values.update(episodes or [])
    return sorted(values, key=lambda x: (isinstance(x, str), x))


def union_domains_aligned(canonical: dict, member: dict) -> tuple[list[str], list[str]]:
    by_name: dict[str, str] = {}
    for domain, color in zip(canonical.get("domains", []) or [], canonical.get("domain_colors", []) or []):
        by_name[domain] = color
    for domain, color in zip(member.get("domains", []) or [], member.get("domain_colors", []) or []):
        by_name.setdefault(domain, color)
    domains = list(by_name.keys())
    return domains, [by_name[d] for d in domains]


def main() -> int:
    kg = json.loads(KG_PATH.read_text(encoding="utf-8"))
    nodes_by_id = {node["id"]: node for node in kg["nodes"]}
    canonical = nodes_by_id.get(CANONICAL_ID)
    legacy = nodes_by_id.get(LEGACY_CANONICAL_ID)
    renamed_canonical = False
    if not canonical and legacy:
        canonical = legacy
        canonical["id"] = CANONICAL_ID
        canonical["name"] = CANONICAL_ID
        canonical["description"] = replace_text(canonical.get("description"))
        nodes_by_id[CANONICAL_ID] = canonical
        del nodes_by_id[LEGACY_CANONICAL_ID]
        renamed_canonical = True
    if not canonical:
        raise SystemExit(f"missing canonical node: {CANONICAL_ID}")

    merged = []
    redirect = {LEGACY_CANONICAL_ID: CANONICAL_ID}
    canonical["description"] = replace_text(canonical.get("description"))
    canonical["aliases"] = [
        alias for alias in union_keep_order(
            [LEGACY_CANONICAL_ID],
            canonical.get("aliases") or [],
        )
        if alias and alias != canonical.get("name")
    ]
    for variant_id in VARIANT_IDS:
        member = nodes_by_id.get(variant_id)
        if not member:
            continue

        canonical["aliases"] = [
            alias for alias in union_keep_order(
                canonical.get("aliases") or [],
                [member.get("name")],
                member.get("aliases") or [],
            )
            if alias and alias != canonical.get("name")
        ]
        canonical["episodes"] = sorted_episodes(
            canonical.get("episodes") or [],
            member.get("episodes") or [],
        )
        canonical["episode_count"] = len(canonical["episodes"])
        canonical["mention_count"] = (
            (canonical.get("mention_count") or 0) + (member.get("mention_count") or 0)
        )
        canonical["importance"] = max(
            canonical.get("importance") or 0,
            member.get("importance") or 0,
        )
        canonical["domains"], canonical["domain_colors"] = union_domains_aligned(canonical, member)
        canonical.setdefault("merge_provenance", []).append({
            "from_id": member["id"],
            "from_name": member.get("name"),
            "from_type": member.get("type"),
            "from_mention_count": member.get("mention_count", 0),
            "from_episodes": member.get("episodes") or [],
            "match_via": "manual Wele/Waylay Waters cleanup",
            "auto_reason": "Residual transcript spelling variant of YOE sponsor Wele Waters",
        })
        merged.append({
            "id": member["id"],
            "name": member.get("name"),
            "type": member.get("type"),
        })
        redirect[member["id"]] = CANONICAL_ID
        del nodes_by_id[member["id"]]

    new_links = []
    seen_links = set()
    link_stats = {
        "before": len(kg.get("links", [])),
        "after": 0,
        "repointed": 0,
        "dropped_orphan_or_self": 0,
        "deduped": 0,
    }
    for link in kg.get("links", []):
        source = link.get("source")
        target = link.get("target")
        if isinstance(source, dict):
            source = source.get("id")
        if isinstance(target, dict):
            target = target.get("id")
        new_source = redirect.get(source, source)
        new_target = redirect.get(target, target)
        if new_source != source or new_target != target:
            link_stats["repointed"] += 1
        if new_source == new_target or new_source not in nodes_by_id or new_target not in nodes_by_id:
            link_stats["dropped_orphan_or_self"] += 1
            continue
        relationship = link.get("type") or link.get("relationship") or ""
        key = (new_source, new_target, relationship)
        if key in seen_links:
            link_stats["deduped"] += 1
            continue
        seen_links.add(key)
        new_link = dict(link)
        new_link["source"] = new_source
        new_link["target"] = new_target
        new_links.append(new_link)

    kg["nodes"] = list(nodes_by_id.values())
    description_updates = 0
    for node in kg["nodes"]:
        old_description = node.get("description")
        new_description = replace_text(old_description)
        if new_description != old_description:
            description_updates += 1
        node["description"] = new_description
    kg["links"] = new_links
    link_stats["after"] = len(new_links)

    flag_updates = 0
    for key in ("sponsor_flags", "yoe_enterprise_flags"):
        if key in kg and isinstance(kg[key], list):
            new_values = [CANONICAL_ID if value == LEGACY_CANONICAL_ID else value for value in kg[key]]
            if new_values != kg[key]:
                flag_updates += 1
            kg[key] = new_values

    node_ids = {node["id"] for node in kg["nodes"]}
    orphans = [
        link for link in kg["links"]
        if link.get("source") not in node_ids or link.get("target") not in node_ids
    ]
    if orphans:
        raise SystemExit(f"refusing to write: {len(orphans)} orphan links remain")

    changed = (
        renamed_canonical
        or bool(merged)
        or link_stats["repointed"] > 0
        or link_stats["dropped_orphan_or_self"] > 0
        or link_stats["deduped"] > 0
        or description_updates > 0
        or flag_updates > 0
    )
    if not changed:
        print(f"no changes needed for {CANONICAL_ID}")
        return 0

    KG_PATH.write_text(json.dumps(kg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log = json.loads(LOG_PATH.read_text(encoding="utf-8")) if LOG_PATH.exists() else {}
    for key in ("sponsor_flags", "yoe_enterprise_flags"):
        if key in log and isinstance(log[key], list):
            log[key] = [CANONICAL_ID if value == LEGACY_CANONICAL_ID else value for value in log[key]]
    log.setdefault("manual_cleanups", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "reason": "Canonicalize Wele Waters KG spelling and fold transcript variants",
        "canonical_id": CANONICAL_ID,
        "merged": merged,
        "link_stats": link_stats,
        "renamed_canonical": renamed_canonical,
        "description_updates": description_updates,
    })
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"merged {len(merged)} variants into {CANONICAL_ID}")
    print(json.dumps({"merged": merged, "link_stats": link_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
