#!/usr/bin/env python3
"""
build_yoe_taxonomy.py — generate web/data/yoe_taxonomy.json from Aaron's YOE
taxonomy Google Sheet.

Source of truth:
  https://docs.google.com/spreadsheets/d/1LpBxvzLD7rbR2XZV7gEBmf6NykhGhSEHJwBu2qDfduM

Structure of the sheet (as of 2026-04-22):
  Cols A–D  : episode #, guest, org, location
  Cols E–G  : admin flags (transcript done / BTS page / map)
  Cols H–L  : PRIMARY PILLARS  — Community · Culture · Economy · Ecology · Health
  Cols M–AK : SECONDARY CATEGORIES (themes) — Biochar · Bio-dynamics · Business ·
              Climate & Science · Community · Ecology & Nature · Education ·
              Esoterica · Farming & Food · Green Building · Green Faith ·
              Health & Wellness · Herbal Medicine · Impact Investing ·
              Indigenous Wisdom · Media Books & Content · Perma-culture ·
              Policy & Govt · Regen / Social Enterprise · Soil · Sustain-ability ·
              Technology & Materials · (blank) · Colorado & Wyoming
  An 'X' or '1' in a cell = "this episode is tagged with this pillar/theme".

The emitted JSON is consumed by the /guide/ KG simple-mode theme chip strip
(see plan: ~/.claude/plans/yoe-guide-design2-themes.md). It's data-only;
frontend joins against each KG entity's `episodes: [N, ...]` field client-side.

Usage:
    python3 scripts/build_yoe_taxonomy.py
    # writes to web/data/yoe_taxonomy.json

Dependencies: stdlib only. The sheet is link-shared, so no Google auth is
required — a plain HTTP GET to the CSV export endpoint works.
"""
from __future__ import annotations

import csv
import io
import json
import pathlib
import sys
import urllib.request
from datetime import datetime, timezone

SHEET_ID = "1LpBxvzLD7rbR2XZV7gEBmf6NykhGhSEHJwBu2qDfduM"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0"

# The two header rows (index 0 and 1). Row 1 (0-indexed 0) carries top-level
# section labels ("PILLARS" etc. in column H). Row 2 (0-indexed 1) carries
# per-column names. We key everything off row-2 column names.
HEADER_ROW_IDX = 1

# Primary pillars in the exact order / color we want them surfaced.
# Colors match web/KnowledgeGraph.js domain palette so KG and taxonomy agree.
PILLAR_COLORS = {
    "COMMUNITY": "#4caf50",
    "CULTURE":   "#9c27b0",
    "ECONOMY":   "#ff9800",
    "ECOLOGY":   "#2196f3",
    "HEALTH":    "#f44336",
}

# Columns whose headers we want to EXCLUDE from the "themes" list even when
# they carry data. These are either duplicates of pillar names, admin
# columns, or region tags that would confuse the theme chip strip.
EXCLUDED_HEADERS = {
    "", "PILLARS", "COMMUNITY",  # duplicate "COMMUNITY" in secondary row re-uses the pillar name
    "TRANSCRIPT COMPLETE & DELIVERED?",
    "BEHIND THE SCENES IN AMBASSADOR RESOURCES PAGE?",
    "GUEST ON GLOBAL RESOURCES MAP?",
    "COLORAOD & WYOMING",   # typo in source; region tag, not a theme
    "COLORADO & WYOMING",
}


def fetch_csv_rows(url: str) -> list[list[str]]:
    """GET the sheet as CSV and return a list of rows (each row = list[str])."""
    req = urllib.request.Request(url, headers={"User-Agent": "build_yoe_taxonomy/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return list(csv.reader(io.StringIO(body)))


def is_tagged(cell: str) -> bool:
    """A cell counts as a tag when it's a non-empty string like 'X' or '1'."""
    return bool((cell or "").strip())


def parse_episode_number(raw: str) -> int | None:
    """Column A carries the episode number. Return int or None."""
    try:
        return int((raw or "").strip())
    except (TypeError, ValueError):
        return None


def build(rows: list[list[str]]) -> dict:
    if len(rows) <= HEADER_ROW_IDX:
        raise ValueError(f"CSV has fewer than {HEADER_ROW_IDX + 1} rows — cannot find header row")

    header = [h.strip().upper() for h in rows[HEADER_ROW_IDX]]
    data_rows = rows[HEADER_ROW_IDX + 1 :]

    # Locate the column index for each pillar + each theme we care about.
    pillar_col_idx: dict[str, int] = {}
    for pillar in PILLAR_COLORS:
        try:
            pillar_col_idx[pillar] = header.index(pillar)
        except ValueError:
            print(f"warn: pillar {pillar!r} not found in header — skipping", file=sys.stderr)

    # Themes are every column after the last pillar, minus excluded ones.
    if pillar_col_idx:
        first_theme_col = max(pillar_col_idx.values()) + 1
    else:
        first_theme_col = 12  # sane default: col M
    theme_cols: list[tuple[int, str]] = []
    for i in range(first_theme_col, len(header)):
        name = header[i]
        if name and name not in EXCLUDED_HEADERS:
            theme_cols.append((i, name))

    # Normalize theme display names: "CLIMATE & SCIENCE" -> "Climate & Science",
    # "BIO-DYNAMICS" -> "Bio-Dynamics", "IMPACT INVESTING" -> "Impact Investing".
    def display_name(raw: str) -> str:
        parts = raw.replace("&", " & ").split()
        out = []
        for w in parts:
            if w == "&":
                out.append("&")
            elif w.isupper() and len(w) > 1:
                out.append(w.title())
            else:
                out.append(w)
        return " ".join(out)

    pillars_out: dict[str, dict] = {
        p: {"color": c, "episode_ids": []} for p, c in PILLAR_COLORS.items() if p in pillar_col_idx
    }
    themes_out: dict[str, dict] = {
        name: {"display": display_name(name), "episode_ids": []}
        for (_, name) in theme_cols
    }
    episodes_out: dict[str, dict] = {}

    for row in data_rows:
        if not row:
            continue
        ep_num = parse_episode_number(row[0] if row else "")
        if ep_num is None:
            continue
        ep_key = str(ep_num)
        guest = (row[1] if len(row) > 1 else "").strip()
        org = (row[2] if len(row) > 2 else "").strip()
        loc = (row[3] if len(row) > 3 else "").strip()

        ep_pillars: list[str] = []
        for pillar, col in pillar_col_idx.items():
            if col < len(row) and is_tagged(row[col]):
                ep_pillars.append(pillar)
                pillars_out[pillar]["episode_ids"].append(ep_num)

        ep_themes: list[str] = []
        for col, name in theme_cols:
            if col < len(row) and is_tagged(row[col]):
                ep_themes.append(name)
                themes_out[name]["episode_ids"].append(ep_num)

        episodes_out[ep_key] = {
            "episode_number": ep_num,
            "guest": guest,
            "org": org,
            "location": loc,
            "pillars": ep_pillars,
            "themes": ep_themes,
        }

    # Sorted + de-duped episode_ids for stable output
    for bucket in list(pillars_out.values()) + list(themes_out.values()):
        bucket["episode_ids"] = sorted(set(bucket["episode_ids"]))

    # Themes list with counts, sorted by count desc then name — drives the
    # default display order of the chip strip.
    themes_list = [
        {"name": n, "display": info["display"], "episode_ids": info["episode_ids"], "count": len(info["episode_ids"])}
        for n, info in themes_out.items()
    ]
    themes_list.sort(key=lambda t: (-t["count"], t["name"]))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": {
            "google_sheet_id": SHEET_ID,
            "url": f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit",
        },
        "pillars": pillars_out,
        "themes": themes_list,
        "episodes": episodes_out,
        "stats": {
            "episode_count": len(episodes_out),
            "pillar_count": len(pillars_out),
            "theme_count": len(themes_list),
        },
    }


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out_path = repo_root / "web" / "data" / "yoe_taxonomy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"→ fetching {CSV_URL}", file=sys.stderr)
    rows = fetch_csv_rows(CSV_URL)
    print(f"  {len(rows)} rows pulled", file=sys.stderr)

    data = build(rows)
    print(
        f"  {data['stats']['episode_count']} episodes, "
        f"{data['stats']['pillar_count']} pillars, "
        f"{data['stats']['theme_count']} themes",
        file=sys.stderr,
    )

    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✓ wrote {out_path.relative_to(repo_root)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
