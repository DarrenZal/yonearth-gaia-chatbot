#!/usr/bin/env python3
"""
auto_decide_merges.py — fill in `status` column on data/candidate_merges.csv
using description + episode + naming heuristics. For genuinely uncertain
cases, generate data/aaron_merge_review.md with episode + transcript
citations so Aaron can adjudicate without us.

Inputs:
  data/candidate_merges.csv               (output of build_merge_candidates.py)
  data/knowledge_graph/visualization_data.json
  data/transcripts/episode_*.json         (used for citation snippets)

Outputs (overwrites in place):
  data/candidate_merges.csv               (status filled per heuristic)
  data/aaron_merge_review.md              (uncertain cases for Aaron)

Decision logic per non-canonical row:
  ACCEPT if any:
    - match_via contains 'alias'              (hand-curated map; trusted)
    - case-insensitive identical names
    - member name is morphological variant of canonical
        (Levenshtein ≤ 2  OR  member-name ⊆ canonical-name as substring)
    - match_via = 'fuzzy+semantic' AND fuzzy score ≥ 0.95
    - description has ≥ 60% keyword overlap with canonical's description
    - all member's episodes are a subset of canonical's episodes
        (only fires when canonical has ≥ 5 episodes — avoids tiny-overlap traps)

  REJECT if any:
    - member description names a distinct entity-defining keyword absent from canonical
        (per a small DISTINCT_KEYWORDS map: e.g., "perennial grain" → Land Institute)
    - member name is in NOT_A_DUPE blacklist (curated from user's review)
    - description keyword overlap with canonical is < 20%
    - member is a generic single-word noun that doesn't share canonical's stem
        (e.g., "corporation" → not B Corp; "foundation" → not Lidge)

  NEEDS_REVIEW: everything else
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "candidate_merges.csv"
KG_PATH = REPO / "data" / "knowledge_graph" / "visualization_data.json"
TRANSCRIPTS_DIR = REPO / "data" / "transcripts"
REVIEW_PATH = REPO / "data" / "aaron_merge_review.md"

# Curated from user's Apr 28 review feedback. Member names that should NOT
# merge into canonical regardless of fuzzy/semantic score.
NOT_A_DUPE: dict[str, set[str]] = {
    "Y on Earth Community":     {"One Earth Community", "Community Finders"},
    "Y on Earth":               {"One Earth Community", "Community Finders"},
    "B Corp":                   {"corporation"},
    "Lidge Family Foundation":  {"Village Family Foundation"},
    "Rodale Institute":         {"Verdeo Institute", "Land Institute"},
}

# Member names that are PHONETIC LLM TRANSCRIPTION NOISE. The user said
# "not sure" about these — likely Whisper misheard "Y on Earth Community" but
# we want Aaron to confirm with citation rather than auto-rejecting.
# Force these into NEEDS_REVIEW even if heuristics would otherwise reject.
FORCE_REVIEW = {
    "Y-Energ community", "Y-Earth community", "Y-Earth Community",
    "Y Honors Community", "Y Honors community",
    "Wieners community", "Winers community", "Weiner Community", "Wiener Community",
    "White Honors Community", "Why Honors Community",
    "Wine and Earth Community", "Wine Community",
    "YNRF community", "Wired Earth Community Network",
}

# Keyword patterns that signal a member is a distinct entity, not a typo.
# (canonical_name → list of regex patterns that, if found in member's description
# but NOT in canonical's description, indicate the member is a separate entity.)
DISTINCT_KEYWORDS: dict[str, list[str]] = {
    "Rodale Institute": [r"\bperennial grain"],
    # Add more as Aaron pushes back on later rounds.
}

GENERIC_SINGLE_WORDS = {
    "foundation", "institute", "corporation", "corporations", "community",
    "communities", "podcast", "network", "company", "council", "alliance",
    "association", "society", "group", "fund", "project", "platform",
}


# --- Levenshtein (small) ---

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --- description keyword overlap ---

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "and", "or", "but", "in", "on", "at", "by", "for", "with",
    "as", "from", "that", "this", "these", "those", "it", "its", "their",
    "they", "them", "we", "our", "us", "i", "me", "my",
    "focused", "dedicated", "promotes", "promoting",
    "organization", "company", "community", "foundation", "institute",
}


def keywords(text: str) -> set[str]:
    if not text: return set()
    toks = re.findall(r"[a-z]{4,}", text.lower())
    return {t for t in toks if t not in STOPWORDS}


def description_overlap(a: str, b: str) -> float:
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb: return 0.0
    return len(ka & kb) / max(len(ka), len(kb))


# --- transcript citation lookup ---

_transcript_cache: dict[int, dict] = {}


def get_episode_transcript(ep_num) -> dict | None:
    try:
        n = int(ep_num)
    except (TypeError, ValueError):
        return None
    if n in _transcript_cache:
        return _transcript_cache[n]
    fp = TRANSCRIPTS_DIR / f"episode_{n}.json"
    if not fp.exists():
        _transcript_cache[n] = None  # type: ignore
        return None
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        _transcript_cache[n] = None  # type: ignore
        return None
    _transcript_cache[n] = d
    return d


def _segment_citation(d: dict, ep: int, needle: str) -> str | None:
    for seg in d.get("segments", []):
        t = (seg.get("text") or "").lower()
        if needle in t:
            start = seg.get("start", 0)
            mins, secs = int(start // 60), int(start % 60)
            quote = (seg.get("text") or "").strip()
            if len(quote) > 220:
                quote = quote[:217] + "…"
            return f"[ep {ep} at {mins:02d}:{secs:02d}] \"{quote}\""
    return None


def find_citation(name: str, episode_nums: list[int], canonical_name: str | None = None) -> str | None:
    """Return a markdown citation. Tries (1) verbatim member name, (2) longest
    distinctive token from member name, (3) canonical name as fallback (for
    LLM-transcription-noise cases). Returns None if nothing found.
    """
    needle = name.lower().strip()
    if len(needle) < 3:
        return None

    # Pre-compute fallback search terms
    distinct_tokens = sorted(
        (t for t in re.findall(r"[a-z]+", needle) if len(t) >= 4 and t not in {"community", "podcast", "earth"}),
        key=lambda t: -len(t),
    )
    canon_needle = (canonical_name or "").lower().strip() if canonical_name else None

    for ep in episode_nums[:5]:
        d = get_episode_transcript(ep)
        if not d:
            continue
        ep_title = d.get("title", "")
        # 1. Verbatim member name
        cite = _segment_citation(d, ep, needle)
        if cite:
            return cite
        # 2. Distinctive token from member name
        for tok in distinct_tokens[:2]:
            cite = _segment_citation(d, ep, tok)
            if cite:
                return cite + f"\n  - _(member name not verbatim; matched token `{tok}`)_"
        # 3. Canonical name fallback (LLM transcription noise case)
        if canon_needle and canon_needle != needle:
            cite = _segment_citation(d, ep, canon_needle)
            if cite:
                return cite + f"\n  - _(member name not in transcript; likely Whisper drift — found canonical `{canonical_name}` instead)_"
        # 4. Bare episode-title fallback
        if ep_title:
            return f"[ep {ep}] _(name not in transcript verbatim; likely LLM extraction artifact)_  Episode: \"{ep_title[:120]}\""
    return None


# --- decision engine ---

def decide(row: dict, canon_node: dict, member_node: dict) -> tuple[str, str]:
    """Return (status, reason). status ∈ {'accept','reject','review'}."""
    canon_name = (row["canonical_name"] or "").strip()
    member_name = (row["member_name"] or "").strip()
    via = row.get("match_via") or ""
    score = float(row["fuzzy_or_semantic_score"] or 0.0)

    # 0. Explicit user blacklist (NOT_A_DUPE) — highest priority reject
    if canon_name in NOT_A_DUPE and member_name in NOT_A_DUPE[canon_name]:
        return "reject", "user-flagged not-a-dupe"

    # 0b. FORCE_REVIEW — phonetic LLM noise the user wants Aaron to verify
    if member_name in FORCE_REVIEW:
        return "review", "user-flagged for Aaron citation review (phonetic noise)"

    # 0a. Distinct-keyword reject
    canon_desc = canon_node.get("description") or ""
    member_desc = member_node.get("description") or ""
    for can_key, patterns in DISTINCT_KEYWORDS.items():
        if can_key.lower() in canon_name.lower():
            for pat in patterns:
                if re.search(pat, member_desc, flags=re.I) and not re.search(pat, canon_desc, flags=re.I):
                    return "reject", f"distinct-keyword: {pat!r} present in member, absent from canonical"

    # 1. ACCEPT — alias-driven (hand-curated)
    if "alias" in via:
        return "accept", "alias map (hand-curated)"

    # 2. ACCEPT — case-insensitive identical names
    if canon_name.lower() == member_name.lower():
        return "accept", "case-only difference"

    # 3. Generic single-word reject (unless name is contained in canonical with stem)
    if member_name.lower() in GENERIC_SINGLE_WORDS:
        # Check if member is a reasonable substring of canonical (e.g., "corporations" in "B Corporation")
        # — only allow if canonical CLEARLY contains member token-for-token AND fuzzy score is high
        if not (member_name.lower() in canon_name.lower() and score >= 0.95):
            return "reject", f"generic single word: {member_name!r}"

    # 4. ACCEPT — morphological variants (Levenshtein ≤ 2 OR substring)
    a, b = canon_name.lower().strip(), member_name.lower().strip()
    if a and b:
        if b in a or a in b:
            return "accept", "name is substring of canonical"
        if levenshtein(a, b) <= 2:
            return "accept", f"Levenshtein={levenshtein(a, b)} (typo)"

    # 5. ACCEPT — fuzzy+semantic with high fuzzy
    if "fuzzy" in via and "semantic" in via and score >= 0.95:
        return "accept", f"fuzzy+semantic, score={score:.3f}"

    # 6. Description overlap signals
    overlap = description_overlap(canon_desc, member_desc)
    if overlap >= 0.6:
        return "accept", f"description overlap {overlap:.2f}"
    if overlap < 0.20 and member_desc and canon_desc:
        return "reject", f"description overlap {overlap:.2f} (too low)"

    # 7. Episode subset — only if canonical has lots of episodes
    canon_eps = set(canon_node.get("episodes") or [])
    member_eps = set(member_node.get("episodes") or [])
    if len(canon_eps) >= 5 and member_eps and member_eps.issubset(canon_eps):
        return "accept", "all member episodes ⊆ canonical episodes"

    # 8. Default: review
    return "review", f"via={via} score={score:.3f} overlap={overlap:.2f}"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        return 2
    if not KG_PATH.exists():
        print(f"ERROR: {KG_PATH} not found", file=sys.stderr)
        return 2

    print(f"→ loading KG", file=sys.stderr)
    kg = json.loads(KG_PATH.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in kg["nodes"]}

    print(f"→ loading {CSV_PATH.name}", file=sys.stderr)
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    print(f"  {len(rows)} rows", file=sys.stderr)

    counts = defaultdict(int)
    review_clusters: dict[str, list[dict]] = defaultdict(list)

    for r in rows:
        if r["status"] == "canonical":
            counts["canonical"] += 1
            continue
        canon = nodes.get(r["canonical_id"])
        member = nodes.get(r["member_id"])
        if not canon or not member:
            r["status"] = "reject"
            r["auto_reason"] = "missing node lookup"
            counts["reject"] += 1
            continue
        status, reason = decide(r, canon, member)
        r["status"] = status if status != "review" else ""  # blank for review
        r["auto_reason"] = reason
        counts[status] += 1
        if status == "review":
            review_clusters[r["cluster_id"]].append({
                "row": r,
                "canon": canon,
                "member": member,
            })

    # Write CSV (with new auto_reason column)
    fieldnames = list(rows[0].keys())
    if "auto_reason" not in fieldnames:
        fieldnames.append("auto_reason")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write Aaron review markdown
    print(f"→ generating {REVIEW_PATH.name}", file=sys.stderr)
    with REVIEW_PATH.open("w", encoding="utf-8") as f:
        f.write("# YOE Knowledge Graph — Aaron's Merge Review\n\n")
        f.write(
            "Below are entity-merge candidates that automated heuristics couldn't decide. "
            "For each cluster, the **canonical** name is what the merged entity will be called. "
            "Members listed under it would be merged INTO the canonical IF approved.\n\n"
            "For each member, we provide the LLM-generated description from the KG plus a "
            "transcript citation showing where the name was actually mentioned in a podcast. "
            "Mark each one **YES** (merge into canonical) or **NO** (separate entity).\n\n"
            "---\n\n"
        )
        # Sort clusters by size (most members first)
        ordered = sorted(review_clusters.items(), key=lambda kv: -len(kv[1]))
        for ci, items in ordered:
            first_canon_id = items[0]["row"]["canonical_id"]
            canon = nodes[first_canon_id]
            f.write(f"## `{ci}` — Canonical: **{items[0]['row']['canonical_name']}** ({canon.get('type','?')})\n\n")
            f.write(f"_Canonical KG description:_ {(canon.get('description') or '_(none)_').strip()}\n\n")
            for it in items:
                m = it["member"]
                row = it["row"]
                m_name = m.get("name", "")
                m_desc = (m.get("description") or "_(no description)_").strip()
                m_eps = m.get("episodes") or []
                cite = find_citation(m_name, [int(x) for x in m_eps if isinstance(x, (int, float))], canonical_name=row.get("canonical_name"))
                cite_line = cite if cite else f"_(no transcript citation found; eps {m_eps[:5]})_"
                f.write(f"- **Member:** `{m_name}` (mc={m.get('mention_count',0)}, eps={m_eps[:5]})\n")
                f.write(f"  - _Description:_ {m_desc}\n")
                f.write(f"  - _Heuristic:_ {row.get('auto_reason','')}\n")
                f.write(f"  - _Source:_ {cite_line}\n")
                f.write(f"  - **Merge?** ☐ YES   ☐ NO\n\n")
            f.write("---\n\n")

    print(f"\n✓ updated {CSV_PATH.name}", file=sys.stderr)
    print(f"✓ wrote {REVIEW_PATH.name}", file=sys.stderr)
    print(f"\n=== auto-decision counts ===", file=sys.stderr)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10}: {v}", file=sys.stderr)
    print(f"\n  Aaron review clusters: {len(review_clusters)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
