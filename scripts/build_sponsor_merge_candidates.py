#!/usr/bin/env python3
"""
build_sponsor_merge_candidates.py — generate candidate_merges.csv for KG entity
cleanup against the Apr 28 sponsor list.

Inputs:
  - data/knowledge_graph/visualization_data.json  (canonical KG, 10770 nodes)
  - /Users/darrenzal/projects/ecoscene/aaron-2026-04-28/YOE Sponsors & Partners.xlsx

Output:
  - data/candidate_merges.csv  (review by Darren before merge)

Each row:
  canonical_name, canonical_is_yoe_enterprise,
  candidate_name, candidate_id, candidate_type, candidate_mention_count,
  jw_score, type_match, episode_overlap_existing, recommendation, status

Recommendation rules (plan):
  jw >= 0.92                         AND type ∈ {ORGANIZATION,GROUP,...} → 'auto-flag'
  0.85 <= jw < 0.92                  AND same conditions               → 'low-confidence'
  Below 0.85 not surfaced.
  + known-dupe overrides (pre-marked 'accept' in `status`).

Type-match rule: candidate type must be one of ORG_TYPES (we treat sponsors
as org-like). Other types appear with type_match=False so reviewer knows.

Usage:
  python3 scripts/build_sponsor_merge_candidates.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import openpyxl

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
KG_PATH = REPO_ROOT / "data" / "knowledge_graph" / "visualization_data.json"
SPONSOR_XLSX = pathlib.Path(
    "/Users/darrenzal/projects/ecoscene/aaron-2026-04-28/YOE Sponsors & Partners.xlsx"
)
OUT_CSV = REPO_ROOT / "data" / "candidate_merges.csv"

ORG_TYPES = {"ORGANIZATION", "GROUP", "PROGRAM", "NETWORK", "PROJECT", "INITIATIVE", "PLATFORM", "PRODUCT"}

# Pre-accept rule: ONLY exact case-insensitive name match on ORG-typed nodes.
# Substring/known-dupe pre-accepts moved to NOTE-only — user reviews everything.
# This avoids over-accepting EVENT-type podcasts that share a substring.

# --- Jaro-Winkler (small inline impl, no scipy/jellyfish dep) ---

def _jaro(a: str, b: str) -> float:
    if a == b: return 1.0
    if not a or not b: return 0.0
    match_window = max(len(a), len(b)) // 2 - 1
    if match_window < 0: match_window = 0
    a_match = [False] * len(a)
    b_match = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo = max(0, i - match_window)
        hi = min(len(b), i + match_window + 1)
        for j in range(lo, hi):
            if not b_match[j] and ch == b[j]:
                a_match[i] = True
                b_match[j] = True
                matches += 1
                break
    if matches == 0: return 0.0
    transpositions = 0
    k = 0
    for i, ch in enumerate(a):
        if a_match[i]:
            while not b_match[k]:
                k += 1
            if ch != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2
    m = matches
    return ((m / len(a)) + (m / len(b)) + ((m - transpositions) / m)) / 3


def jaro_winkler(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    j = _jaro(a, b)
    prefix = 0
    for x, y in zip(a, b):
        if x == y: prefix += 1
        else: break
        if prefix >= 4: break
    return j + prefix * 0.1 * (1 - j)


# --- xlsx parse ---

def load_sponsors() -> list[tuple[str, bool]]:
    wb = openpyxl.load_workbook(SPONSOR_XLSX, data_only=True)
    ws = wb.active
    out = []
    for r in ws.iter_rows(values_only=True):
        if not r or not r[0]:
            continue
        name = str(r[0]).strip()
        if not name or name.upper().startswith("Y ON EARTH COMMUNITY SPONSORS"):
            continue
        if name.startswith("YOE Enterprise"):
            continue
        flag_cell = (r[1] if len(r) > 1 else "")
        is_yoe_ent = (str(flag_cell).strip().upper() == "YES") if flag_cell else False
        out.append((name, is_yoe_ent))
    # Synthesize canonical entries that are NOT in the xlsx but are
    # known-dupe targets per the Apr 24 meeting notes.
    out.insert(0, ("Y on Earth Community", True))
    out.insert(0, ("Vera Herbals", False))
    return out


def main() -> int:
    if not KG_PATH.exists():
        print(f"ERROR: {KG_PATH} not found — scp from server first", file=sys.stderr)
        return 2

    print(f"→ loading {KG_PATH.name}", file=sys.stderr)
    kg = json.loads(KG_PATH.read_text(encoding="utf-8"))
    nodes = kg["nodes"]
    print(f"  {len(nodes)} nodes", file=sys.stderr)

    sponsors = load_sponsors()
    print(f"→ loaded {len(sponsors)} canonical sponsors from xlsx", file=sys.stderr)

    rows = []
    for canonical_name, is_yoe_ent in sponsors:
        cn_lower = canonical_name.lower()
        for node in nodes:
            cand_name = (node.get("name") or "").strip()
            if not cand_name:
                continue
            jw = jaro_winkler(canonical_name, cand_name)
            if jw < 0.85:
                continue
            cand_type = node.get("type", "")
            type_match = cand_type in ORG_TYPES
            mention_count = node.get("mention_count", 0)
            cand_id = node.get("id", "")

            # Recommendation logic
            recommendation = "auto-flag" if jw >= 0.92 and type_match else (
                "low-confidence" if jw >= 0.85 and type_match else "type-mismatch"
            )

            # Pre-accept ONLY exact case-insensitive name match AND ORG type
            pre_status = ""
            if cand_name.lower() == cn_lower and type_match:
                pre_status = "accept"

            rows.append({
                "canonical_name": canonical_name,
                "canonical_is_yoe_enterprise": "yes" if is_yoe_ent else "no",
                "candidate_name": cand_name,
                "candidate_id": cand_id,
                "candidate_type": cand_type,
                "candidate_mention_count": mention_count,
                "jw_score": f"{jw:.3f}",
                "type_match": "yes" if type_match else "no",
                "recommendation": recommendation,
                "status": pre_status,
            })

    rows.sort(key=lambda r: (r["canonical_name"], -float(r["jw_score"])))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    # Summary
    print(f"\n→ wrote {OUT_CSV}", file=sys.stderr)
    print(f"  {len(rows)} candidate rows", file=sys.stderr)
    auto_flag = sum(1 for r in rows if r["recommendation"] == "auto-flag")
    low_conf = sum(1 for r in rows if r["recommendation"] == "low-confidence")
    type_mm = sum(1 for r in rows if r["recommendation"] == "type-mismatch")
    pre_accept = sum(1 for r in rows if r["status"] == "accept")
    print(f"  auto-flag: {auto_flag}", file=sys.stderr)
    print(f"  low-confidence: {low_conf}", file=sys.stderr)
    print(f"  type-mismatch: {type_mm}", file=sys.stderr)
    print(f"  pre-accepted: {pre_accept}", file=sys.stderr)

    # Show top 30 in stderr for quick scan
    print("\n--- preview (top 30) ---", file=sys.stderr)
    for r in rows[:30]:
        print(
            f"  {r['canonical_name']!r:42}  vs  {r['candidate_name']!r:42}  "
            f"jw={r['jw_score']}  type={r['candidate_type']:14}  "
            f"mc={r['candidate_mention_count']:3}  rec={r['recommendation']:14}  status={r['status']!r}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
