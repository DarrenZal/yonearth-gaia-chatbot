#!/usr/bin/env python3
"""
build_merge_candidates.py — 5-phase entity dedup for YOE KG.

Pattern lifted from koi-processor's EntityResolver (5-tier waterfall). Phases:
  1. Quality gate     — drop generic-noun + bad-pattern names
  2. Canonical aliases— Tier 1.5 deterministic map (data/yoe_canonical_entities.json)
  3. Fuzzy            — Tier 1.x JW (PERSON) / token-sort (ORG), no length prefilter
  4. Semantic         — Tier 2 OpenAI text-embedding-3-small + per-type cosine threshold
  5. Merge & cluster  — union-find over all flagged pairs → single CSV with cluster_id

Output: data/candidate_merges.csv (single merged review CSV, one row per
non-canonical member). Reviewer marks `status` accept/reject before the
apply script runs.

Per-type thresholds (calibrated for OpenAI -small embeddings; lower than
koi-processor's because -small produces tighter cosine distributions):
  PERSON         0.86      ORGANIZATION   0.90
  CONCEPT        0.88      PLACE          0.88
  PRACTICE       0.88      PRODUCT        0.88
  TECHNOLOGY     0.88      EVENT          0.88
  SPECIES        0.92      (default)      0.90

Fuzzy thresholds:
  PERSON         0.86 (JW; deliberately lower than koi-processor's 0.93 so
                       short variants like "Aaron"↔"Aaron Perry" match)
  ORGANIZATION   0.85 (token_sort_ratio scaled 0–1)
  (default)      0.90

Cost: ~$0.02 OpenAI for 10770-node sweep; cache makes re-runs free.

Usage (on server with OPENAI_API_KEY in .env):
  cd /home/claudeuser/yonearth-gaia-chatbot
  set -a && source .env && set +a
  .venv/bin/python3 scripts/build_merge_candidates.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import sys
import time
from collections import defaultdict
from typing import Iterable

REPO = pathlib.Path(__file__).resolve().parent.parent
KG_PATH = REPO / "data" / "knowledge_graph" / "visualization_data.json"
ALIAS_PATH = REPO / "data" / "yoe_canonical_entities.json"
EMBED_CACHE = REPO / "data" / "entity_embeddings.json"
OUT_CSV = REPO / "data" / "candidate_merges.csv"

EMBED_MODEL = "text-embedding-3-small"

# --- Phase 1: quality gate (lifted from koi-processor api/quality_gates.py) ---

BLOCKED_NAMES_LOWER = {
    # Pronouns / generic
    "he", "she", "it", "they", "them", "his", "her", "its", "their",
    "this", "that", "these", "those", "who", "what", "which", "where",
    "the", "a", "an", "someone", "something", "everyone", "everything",
    "nothing", "nobody", "anybody", "anything", "none", "other", "others",
    "people", "person", "thing", "things", "group", "groups",
    "organization", "company", "project", "concept", "location",
    "unknown", "n/a", "na", "null", "undefined", "tbd",
    # YOE-specific generic concepts
    "water", "soil", "system", "user", "guest", "host", "podcast",
    "knowledge", "knowledge graph", "graph",
}

BAD_NAME_PATTERNS = [
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^[a-zA-Z]$"),
    re.compile(r"^\d+$"),
    re.compile(r"^[^a-zA-Z]*$"),
    re.compile(r"^.{0,1}$"),
    re.compile(r"^.{200,}$"),
]


def is_blocked(name: str) -> tuple[bool, str | None]:
    """Return (blocked, reason)."""
    if not name or not name.strip():
        return True, "empty"
    n = name.strip()
    if n.lower() in BLOCKED_NAMES_LOWER:
        return True, f"blocklist({n.lower()})"
    for p in BAD_NAME_PATTERNS:
        if p.search(n):
            return True, f"pattern({p.pattern[:30]})"
    return False, None


# --- Phase 3: fuzzy (no length prefilter; per-type) ---

PER_TYPE_FUZZY_THRESHOLD = {
    "PERSON":       0.93,   # tightened — was 0.86, caught "John Smith" ↔ "John Adams"
    "ORGANIZATION": 0.88,   # was 0.85
    "CONCEPT":      0.92,
    "PRACTICE":     0.92,
    "PRODUCT":      0.92,
    "PLACE":        0.92,
    "TECHNOLOGY":   0.92,
    "EVENT":        0.92,
    "SPECIES":      0.95,
}
DEFAULT_FUZZY_THRESHOLD = 0.93


def jaro(a: str, b: str) -> float:
    if a == b: return 1.0
    if not a or not b: return 0.0
    mw = max(len(a), len(b)) // 2 - 1
    if mw < 0: mw = 0
    am = [False] * len(a)
    bm = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo = max(0, i - mw)
        hi = min(len(b), i + mw + 1)
        for j in range(lo, hi):
            if not bm[j] and ch == b[j]:
                am[i] = True
                bm[j] = True
                matches += 1
                break
    if matches == 0: return 0.0
    transpositions = 0
    k = 0
    for i, ch in enumerate(a):
        if am[i]:
            while not bm[k]: k += 1
            if ch != b[k]: transpositions += 1
            k += 1
    transpositions //= 2
    m = matches
    return ((m / len(a)) + (m / len(b)) + ((m - transpositions) / m)) / 3


def jaro_winkler(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    j = jaro(a, b)
    prefix = 0
    for x, y in zip(a, b):
        if x == y: prefix += 1
        else: break
        if prefix >= 4: break
    return j + prefix * 0.1 * (1 - j)


def token_sort_ratio(a: str, b: str) -> float:
    """0–1 score; rapidfuzz-equivalent token sort."""
    ta = " ".join(sorted(re.findall(r"\w+", a.lower())))
    tb = " ".join(sorted(re.findall(r"\w+", b.lower())))
    if not ta or not tb: return 0.0
    # Use Jaro on the sorted token strings.
    return jaro(ta, tb)


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def fuzzy_score(a: str, b: str, type_: str) -> float:
    """Score 0–1. For PERSON: require ≥2 shared tokens OR identical names
    (case-folded), else return 0 to suppress first-name-only matches like
    'John Perkins' ↔ 'John Adams'.
    """
    if type_ == "PERSON":
        ta, tb = _tokens(a), _tokens(b)
        if a.lower().strip() != b.lower().strip() and len(ta & tb) < 2:
            return 0.0
        return jaro_winkler(a, b)
    return token_sort_ratio(a, b)


# --- Phase 4: semantic (OpenAI embeddings, cosine) ---

PER_TYPE_SEMANTIC_THRESHOLD = {
    "PERSON":       0.93,   # tightened — was 0.86
    "ORGANIZATION": 0.92,   # was 0.90
    "CONCEPT":      0.93,   # tightened — was 0.88, caught "B Corp" ↔ "Desertification"
    "PLACE":        0.92,
    "PRACTICE":     0.92,
    "PRODUCT":      0.92,
    "TECHNOLOGY":   0.92,
    "EVENT":        0.92,
    "SPECIES":      0.94,
}
DEFAULT_SEMANTIC_THRESHOLD = 0.92


def cosine(a: list[float], b: list[float]) -> float:
    s = 0.0; na = 0.0; nb = 0.0
    for x, y in zip(a, b):
        s += x * y; na += x * x; nb += y * y
    if na == 0 or nb == 0: return 0.0
    return s / (math.sqrt(na) * math.sqrt(nb))


def build_embed_text(node: dict) -> str:
    name = (node.get("name") or "").strip()
    desc = (node.get("description") or "").strip().replace("\n", " ")
    return f"{name}. {desc[:300]}".strip(". ").strip()


def batched(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def fetch_embeddings(texts: list[str], cache: dict[str, list[float]]) -> dict[str, list[float]]:
    miss = [t for t in texts if t and t not in cache]
    if not miss:
        return {t: cache[t] for t in texts if t in cache}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        for i, group in enumerate(batched(miss, 100)):
            print(f"  embed batch {i+1}: {len(group)} texts", file=sys.stderr)
            resp = client.embeddings.create(model=EMBED_MODEL, input=group)
            for t, item in zip(group, resp.data):
                cache[t] = item.embedding
            time.sleep(0.05)
    except ImportError:
        import urllib.request
        for i, group in enumerate(batched(miss, 100)):
            print(f"  embed batch {i+1}: {len(group)} texts (urllib)", file=sys.stderr)
            req = urllib.request.Request(
                "https://api.openai.com/v1/embeddings",
                data=json.dumps({"model": EMBED_MODEL, "input": group}).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            for t, item in zip(group, body["data"]):
                cache[t] = item["embedding"]
            time.sleep(0.05)

    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return {t: cache[t] for t in texts if t in cache}


# --- Union-find for cluster aggregation across phases ---

class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, x: str):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        self.add(a); self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# --- Pipeline ---

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="dev: process first N nodes")
    ap.add_argument("--no-semantic", action="store_true", help="skip Phase 4 (no OpenAI call)")
    args = ap.parse_args()

    print(f"→ loading {KG_PATH.name}", file=sys.stderr)
    kg = json.loads(KG_PATH.read_text(encoding="utf-8"))
    nodes = kg["nodes"]
    if args.limit:
        nodes = nodes[: args.limit]
    print(f"  {len(nodes)} nodes", file=sys.stderr)

    aliases = json.loads(ALIAS_PATH.read_text(encoding="utf-8")) if ALIAS_PATH.exists() else {}

    # ----- Phase 1: quality gate -----
    print("→ Phase 1: quality gate", file=sys.stderr)
    blocked: dict[str, str] = {}  # id → reason
    kept_nodes: list[dict] = []
    for n in nodes:
        b, reason = is_blocked(n.get("name") or "")
        if b:
            blocked[n["id"]] = reason or ""
        else:
            kept_nodes.append(n)
    print(f"  blocked {len(blocked)}; kept {len(kept_nodes)}", file=sys.stderr)

    # Build name→nodes lookup (case-insensitive) — needed for Phases 2 and case folding
    by_name_ci: dict[str, list[dict]] = defaultdict(list)
    for n in kept_nodes:
        by_name_ci[(n.get("name") or "").lower().strip()].append(n)

    # ----- Phase 2: canonical alias resolution -----
    print("→ Phase 2: canonical aliases", file=sys.stderr)
    # Build a map: (lower_name, type) → canonical_name
    alias_to_canon: dict[tuple[str, str], str] = {}
    for type_, mappings in aliases.items():
        if type_.startswith("_"):
            continue
        for canon, alias_list in mappings.items():
            alias_to_canon[(canon.lower().strip(), type_)] = canon
            for a in alias_list:
                alias_to_canon[(a.lower().strip(), type_)] = canon

    uf = UnionFind()
    for n in kept_nodes:
        uf.add(n["id"])

    # Phase 2a: link nodes that match alias→canonical (any type variation considered)
    alias_links = 0
    for type_, mappings in aliases.items():
        if type_.startswith("_"):
            continue
        for canon, alias_list in mappings.items():
            # Find ALL nodes whose name matches canon or any alias (case-insensitive)
            members: list[dict] = []
            seen_ids: set[str] = set()
            for nm in [canon] + list(alias_list):
                for n in by_name_ci.get(nm.lower().strip(), []):
                    if n["id"] not in seen_ids:
                        members.append(n)
                        seen_ids.add(n["id"])
            if len(members) >= 2:
                # Union all members
                base = members[0]["id"]
                for m in members[1:]:
                    uf.union(base, m["id"])
                alias_links += len(members) - 1
    print(f"  alias-driven unions: {alias_links}", file=sys.stderr)

    # Phase 2b: case-fold within type — "Water" / "water" if same type
    case_links = 0
    for ci_name, group in by_name_ci.items():
        if len(group) < 2:
            continue
        # group all same-type members
        by_type: dict[str, list[dict]] = defaultdict(list)
        for n in group:
            by_type[n.get("type", "")].append(n)
        for t, lst in by_type.items():
            if len(lst) < 2: continue
            base = lst[0]["id"]
            for m in lst[1:]:
                uf.union(base, m["id"])
                case_links += 1
    print(f"  case-fold unions: {case_links}", file=sys.stderr)

    # ----- Phase 3: fuzzy (no length prefilter, per-type) -----
    print("→ Phase 3: fuzzy (Jaro-Winkler / token-sort, no length prefilter)", file=sys.stderr)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for n in kept_nodes:
        by_type[n.get("type", "")].append(n)

    fuzzy_pairs: list[tuple[str, str, float]] = []  # (id_a, id_b, score)
    for type_, group in by_type.items():
        thr = PER_TYPE_FUZZY_THRESHOLD.get(type_, DEFAULT_FUZZY_THRESHOLD)
        # O(n^2) per type — manageable since the largest bucket is ~3200 (CONCEPT).
        # 3200^2 / 2 = 5.1M comparisons; with simple JW that's ~5s in Python.
        names = [(n["id"], (n.get("name") or "").strip()) for n in group]
        n_count = len(names)
        for i in range(n_count):
            id_a, name_a = names[i]
            for j in range(i + 1, n_count):
                id_b, name_b = names[j]
                # Tiny optimization: skip if names share no first letter
                if name_a and name_b and name_a[0].lower() != name_b[0].lower():
                    # Still allow because token-sort can match across different first letters.
                    # Keep this skip for PERSON only (Jaro-Winkler weights prefix).
                    if type_ == "PERSON":
                        continue
                score = fuzzy_score(name_a, name_b, type_)
                if score >= thr:
                    fuzzy_pairs.append((id_a, id_b, score))
                    uf.union(id_a, id_b)
        print(f"  {type_:14}: {n_count:5} nodes  fuzzy_pairs_so_far={len(fuzzy_pairs)}", file=sys.stderr)
    print(f"  total fuzzy pairs: {len(fuzzy_pairs)}", file=sys.stderr)

    # ----- Phase 4: semantic embeddings -----
    semantic_pairs: list[tuple[str, str, float]] = []
    if not args.no_semantic:
        print("→ Phase 4: semantic embeddings (OpenAI text-embedding-3-small)", file=sys.stderr)
        cache = json.loads(EMBED_CACHE.read_text(encoding="utf-8")) if EMBED_CACHE.exists() else {}
        print(f"  embed cache: {len(cache)} pre-existing", file=sys.stderr)

        text_by_id: dict[str, str] = {n["id"]: build_embed_text(n) for n in kept_nodes}
        unique_texts = sorted({t for t in text_by_id.values() if t})
        embeds = fetch_embeddings(unique_texts, cache)
        print(f"  embeds ready: {len(embeds)} unique texts", file=sys.stderr)

        # Build per-type id→vec map
        for type_, group in by_type.items():
            thr = PER_TYPE_SEMANTIC_THRESHOLD.get(type_, DEFAULT_SEMANTIC_THRESHOLD)
            id_vec: list[tuple[str, list[float]]] = []
            for n in group:
                t = text_by_id.get(n["id"])
                if t and t in embeds:
                    id_vec.append((n["id"], embeds[t]))
            id_to_name = {n["id"]: (n.get("name") or "") for n in group}
            is_person = (type_ == "PERSON")
            for i in range(len(id_vec)):
                ia, va = id_vec[i]
                name_a = id_to_name.get(ia, "")
                tokens_a = _tokens(name_a) if is_person else None
                for j in range(i + 1, len(id_vec)):
                    ib, vb = id_vec[j]
                    if is_person:
                        name_b = id_to_name.get(ib, "")
                        if name_a.lower().strip() != name_b.lower().strip():
                            if len(tokens_a & _tokens(name_b)) < 2:
                                continue
                    s = cosine(va, vb)
                    if s >= thr:
                        semantic_pairs.append((ia, ib, s))
                        uf.union(ia, ib)
            print(f"  {type_:14}: semantic_pairs_so_far={len(semantic_pairs)}", file=sys.stderr)
        print(f"  total semantic pairs: {len(semantic_pairs)}", file=sys.stderr)

    # ----- Phase 5: collect clusters + write CSV -----
    print("→ Phase 5: clusters + CSV", file=sys.stderr)
    clusters: dict[str, list[dict]] = defaultdict(list)
    id_to_node = {n["id"]: n for n in kept_nodes}
    for n in kept_nodes:
        clusters[uf.find(n["id"])].append(n)

    # Filter to clusters with ≥2 members
    nontrivial = [v for v in clusters.values() if len(v) >= 2]
    print(f"  {len(nontrivial)} clusters with ≥2 members", file=sys.stderr)

    # Build pair-evidence index for "match_via" annotation
    pair_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)
    for a, b, _s in fuzzy_pairs:
        pair_evidence[_key(a, b)].append("fuzzy")
    for a, b, _s in semantic_pairs:
        pair_evidence[_key(a, b)].append("semantic")

    # Write CSV — one row per non-canonical member; cluster_id grouping
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "cluster_id", "type", "canonical_name", "canonical_id", "canonical_mention_count",
            "member_name", "member_id", "member_mention_count", "match_via", "fuzzy_or_semantic_score",
            "status",
        ])
        nontrivial_sorted = sorted(nontrivial, key=lambda c: (-len(c), -max((n.get("mention_count", 0) for n in c), default=0)))
        for ci, cluster in enumerate(nontrivial_sorted):
            cluster_sorted = sorted(cluster, key=lambda n: (-n.get("mention_count", 0), -len(n.get("name", ""))))
            canon = cluster_sorted[0]
            cid = f"c{ci:04d}"
            type_ = canon.get("type", "")
            canon_name = canon.get("name", "")
            # Bias: if canon's name (lowercased) is in alias_to_canon for this type,
            # prefer the alias-mapped canonical name as display.
            mapped = alias_to_canon.get((canon_name.lower().strip(), type_))
            if mapped:
                canon_name = mapped
            for member in cluster_sorted:
                if member is canon:
                    via = "canonical"
                    score_str = "1.000"
                else:
                    via_set = set(pair_evidence.get(_key(canon["id"], member["id"]), []))
                    # If pair not directly linked to canon, find which other cluster member is its evidence
                    if not via_set:
                        for other in cluster_sorted:
                            if other is member: continue
                            ev = pair_evidence.get(_key(other["id"], member["id"]), [])
                            via_set.update(ev)
                    if not via_set:
                        # Member came in via Phase 2 (alias / case-fold)
                        via_set.add("alias")
                    via = "+".join(sorted(via_set))
                    # Score: report best fuzzy/semantic score we have for this pair to canon
                    best_score = 0.0
                    for x, y, s in fuzzy_pairs + semantic_pairs:
                        if {x, y} == {canon["id"], member["id"]}:
                            best_score = max(best_score, s)
                    score_str = f"{best_score:.3f}" if best_score else ""
                w.writerow([
                    cid,
                    type_,
                    canon_name,
                    canon["id"],
                    canon.get("mention_count", 0),
                    member.get("name", ""),
                    member["id"],
                    member.get("mention_count", 0),
                    via,
                    score_str,
                    "canonical" if member is canon else "",
                ])

    print(f"\n✓ wrote {OUT_CSV}", file=sys.stderr)

    # Console summary: 8 largest clusters
    print("\n--- preview: 8 largest clusters ---", file=sys.stderr)
    for cluster in sorted(nontrivial, key=lambda c: -len(c))[:8]:
        cluster_sorted = sorted(cluster, key=lambda n: (-n.get("mention_count", 0), -len(n.get("name", ""))))
        canon = cluster_sorted[0]
        type_ = canon.get("type", "?")
        print(f"  [{type_:14}] canon: {canon['name']!r:42}  mc={canon.get('mention_count',0):>3}  size={len(cluster)}", file=sys.stderr)
        for m in cluster_sorted[1:6]:
            print(f"      → {m['name']!r:42}  mc={m.get('mention_count',0):>3}", file=sys.stderr)
        if len(cluster_sorted) > 6:
            print(f"      … and {len(cluster_sorted) - 6} more", file=sys.stderr)

    # Stats by type
    by_type_stats: dict[str, int] = defaultdict(int)
    for c in nontrivial:
        t = c[0].get("type", "?")
        by_type_stats[t] += len(c) - 1  # non-canonical members
    print("\n--- merges by type ---", file=sys.stderr)
    for t, n in sorted(by_type_stats.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {t:14}: {n} merges", file=sys.stderr)

    print("\n--- blocked entities (top 10) ---", file=sys.stderr)
    for nid, reason in list(blocked.items())[:10]:
        node = next((x for x in nodes if x["id"] == nid), {})
        print(f"  {node.get('name','?')!r:32} {node.get('type','?'):14} reason={reason}", file=sys.stderr)
    print(f"  … total blocked: {len(blocked)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
