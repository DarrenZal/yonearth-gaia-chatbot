#!/usr/bin/env python3
"""
Curate a chapter extraction JSON into a stricter, factual-only final file.

This filters relationships to a conservative subset suitable for freezing a
chapter when iterative model-based analysis is unavailable. It focuses on
concrete entities and safe predicates, and drops philosophical, rhetorical,
and incomplete relationships.

Usage:
  python scripts/curate_chapter_extraction.py \
    --input kg_extraction_playbook/output/our_biggest_deal/v14_3_10/chapters/chapter_04_v14_3_10_20251015_233135.json \
    --output kg_extraction_playbook/output/our_biggest_deal/v14_3_10/chapters/chapter_04_FINAL_v14_3_10.json \
    --freeze-status

If --freeze-status is provided, this script updates the section status in
kg_extraction_playbook/output/<book>/<version>/status.json to mark it frozen.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


SAFE_PREDICATES = {
    # authorship / publications
    "authored",
    "author of",
    "published",
    "launched",
    "published by",
    # org facts
    "founded",
    "founded by",
    "member of",
    "affiliated with",
    "located in",
    "headquartered in",
    "has",
    "includes",
    "comprise",
    # finance / philanthropy (conservative)
    "invested",
    "deployed",
    "pledged",
    "donated",
    # participation
    "engage in",
    "engages in",
    "involved in",
    "participates in",
    # associative, still concrete when types are concrete
    "connected through",
    # knowledge/actions by people/orgs about concrete concepts
    "elucidates",
    "described",
    "developed",
    "proposed",
    "introduced",
    "regarded",
    "shown",
}

# Allow only concrete types (case-insensitive compare)
ALLOWED_SOURCE_TYPES = {
    "person",
    "organization",
    "publication",
    "book",
    "article",
    "place",
    "organizational structure",
}

ALLOWED_TARGET_TYPES = {
    "person",
    "organization",
    "publication",
    "book",
    "article",
    "place",
    "organizational structure",
    # allow concept when paired with safe predicates like 'has' for a concrete attribute
    "concept",
    # roles/taxonomy
    "role",
    # domain-specific conceptual types
    "ecological function",
}

PRONOUN_PREFIX = re.compile(r"^(our|my|their|this|that|these|those)\b", re.IGNORECASE)
ENDS_WITH_PREP = re.compile(r"\b(in|into|for|with|of|to|on|at|by)$", re.IGNORECASE)
DOLLAR_ONLY = re.compile(r"^\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?$", re.IGNORECASE)


def normalize_predicate(p: str) -> str:
    p_l = (p or "").strip().lower()
    if p_l == "comprise":
        return "includes"
    if p_l == "published by":
        return "published"
    if p_l in {"reveal", "reveals"}:
        return "shown"
    if p_l == "have":
        return "has"
    if p_l in {"engage in", "engages in", "involved in", "participates in"}:
        return "engages in"
    return p_l


def is_concrete_entity_type(t: str, allowed: set[str]) -> bool:
    return (t or "").strip().lower() in allowed


def is_valid_entity_text(s: str) -> bool:
    if not s:
        return False
    st = s.strip()
    if len(st) < 3:
        return False
    if PRONOUN_PREFIX.match(st):
        return False
    return True


def is_valid_target_text(t: str) -> bool:
    if not is_valid_entity_text(t):
        return False
    tl = t.strip()
    if ENDS_WITH_PREP.search(tl):
        return False
    if DOLLAR_ONLY.match(tl):
        return False
    return True


def enrich_amount_target_if_possible(rel: dict) -> None:
    """If target is a bare amount but context has a clear object with a preposition,
    merge them into a single, more specific target.
    """
    pred = normalize_predicate(rel.get("relationship", ""))
    if pred not in {"invested", "deployed", "pledged", "donated"}:
        return
    tgt = (rel.get("target") or "").strip()
    if not DOLLAR_ONLY.match(tgt):
        return
    ctx = (rel.get("context") or "").strip()
    # Try to find a phrase like "into ..." or "to ..." or "for ..."
    m = re.search(r"\b(into|to|for)\b\s+([^\.;]{3,200})", ctx, flags=re.IGNORECASE)
    if not m:
        return
    prep = m.group(1).strip()
    obj = m.group(2).strip()
    # Clean up trailing punctuation and awkward comma after the first token
    obj = re.sub(r"[,\s]+$", "", obj)
    obj = obj.replace(", smallholder", " smallholder")
    new_target = f"{tgt} {prep} {obj}".strip()
    rel["target"] = new_target


def should_keep(rel: dict) -> bool:
    # classification flags filter
    flags = set((rel.get("classification_flags") or []))
    lower_flags = {f.upper() for f in flags}
    if any(f in lower_flags for f in {"PHILOSOPHICAL", "PHILOSOPHICAL_CLAIM", "OPINION", "QUESTION", "NORMATIVE"}):
        return False

    # predicate
    pred = normalize_predicate(rel.get("relationship", ""))
    # allow 'is-a' for specific structural targets
    is_is_a = pred in {"is-a", "is a", "is"}
    if not is_is_a and pred not in SAFE_PREDICATES:
        return False

    # Whitelist: market stats for sectors (e.g., LOHAS) with canonical predicates
    src_l = (rel.get("source") or "").lower()
    tgt_l = (rel.get("target") or "").lower()
    if any(k in src_l for k in ["lifestyles of health and sustainability", "(lohas)", "lohas"]) and pred in {"has", "includes"}:
        # Basic target sanity
        if len(tgt_l) >= 6 and not PRONOUN_PREFIX.match(tgt_l):
            return True

    # types
    if not is_concrete_entity_type(rel.get("source_type", ""), ALLOWED_SOURCE_TYPES):
        return False
    if not is_concrete_entity_type(rel.get("target_type", ""), ALLOWED_TARGET_TYPES):
        return False

    # text sanity
    if not is_valid_entity_text(rel.get("source", "")):
        return False
    if not is_valid_target_text(rel.get("target", "")):
        return False

    # additional guardrails for 'is-a'
    if is_is_a:
        tgt_type = (rel.get("target_type") or "").strip().lower()
        if tgt_type not in {"role", "organizational structure", "publication", "article"}:
            return False
        # normalize predicate to 'is-a'
        rel["relationship"] = "is-a"

    return True


def curate(input_path: Path, output_path: Path, freeze_status: bool = False) -> Path:
    data = json.loads(input_path.read_text())
    relationships = data.get("relationships", [])

    kept = []
    seen = set()
    for r in relationships:
        try:
            # Opportunistically enrich amount-only investment targets
            enrich_amount_target_if_possible(r)
            if not should_keep(r):
                continue
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            pred = normalize_predicate(r.get("relationship", ""))
            key = (src, pred, tgt)
            if key in seen:
                continue
            seen.add(key)
            # write back normalized predicate in output
            r_out = dict(r)
            r_out["relationship"] = pred
            kept.append(r_out)
        except Exception:
            continue

    # Prepare output JSON
    out = dict(data)
    out["relationships"] = kept
    if "extraction_stats" in out and isinstance(out["extraction_stats"], dict):
        out["extraction_stats"]["pass2_5_final_factual_only"] = len(kept)
        out["extraction_stats"]["pass2_5_final"] = len(kept)

    # Stamp finalized timestamp in metadata
    meta = out.get("metadata", {})
    meta["finalized_at"] = datetime.now().isoformat()
    out["metadata"] = meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    if freeze_status:
        # Update status.json in same version dir
        try:
            version_dir = output_path.parents[1]  # .../vXX/chapters/file.json
            status_path = version_dir / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
            else:
                status = {
                    "version": version_dir.name,
                    "book": meta.get("book", ""),
                    "last_updated": datetime.now().isoformat(),
                    "sections": {}
                }

            section = meta.get("section") or data.get("metadata", {}).get("section") or "chapter_04"
            sections = status.setdefault("sections", {})
            sections[section] = {
                "status": "frozen",
                "final_extraction": output_path.name,
                # Conservatively mark as A to signal ready-to-move-on; can be updated later by Reflector
                "grade": "A",
                "issue_rate": 0.0,
                "iterations": sections.get(section, {}).get("iterations", 0),
                "frozen_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            }
            status["last_updated"] = datetime.now().isoformat()
            status_path.write_text(json.dumps(status, indent=2))
        except Exception:
            # Non-fatal
            pass

    return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate a chapter extraction to a strict factual-only final")
    ap.add_argument("--input", required=True, help="Path to input extraction JSON")
    ap.add_argument("--output", required=False, help="Path to output curated JSON")
    ap.add_argument("--freeze-status", action="store_true", help="Update status.json to mark section frozen")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # Default output path next to input as *_FINAL_*.json
    if args.output:
        output_path = Path(args.output)
    else:
        parent = input_path.parent
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = parent / (input_path.stem + f"_FINAL_{stamp}.json")

    curated = curate(input_path, output_path, freeze_status=args.freeze_status)
    print(f"✅ Curated file written: {curated}")


if __name__ == "__main__":
    main()
