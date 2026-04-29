#!/usr/bin/env python3
"""
apply_my_review_decisions.py — fill in the 77 NEEDS_REVIEW rows in
data/candidate_merges.csv with my own context-based judgments. Each cluster
has a one-line rationale recorded in the auto_reason column.

Decisions made by reviewing data/aaron_merge_review.md case-by-case using
description + transcript citation + canonical name evidence.
"""
from __future__ import annotations

import csv
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "candidate_merges.csv"

# Cluster-level decisions (clean cases where all members go the same way).
ACCEPT_CLUSTERS: dict[str, str] = {
    # Y on Earth Community variants — Whisper transcription noise; citations
    # confirm canonical name appears in same episodes.
    "c0000": "all members are Whisper transcription variants of Y on Earth Community",
    "c0001": "domain-name typos (whyoners.org / yhonner.org) of YonEarth.org",
    "c0002": "podcast title variants (capitalization, plural, 'The' prefix, lowercase)",
    "c0004": "Lich/Litch/Ich Family Foundation are Whisper variants of Lidge",
    "c0005": "Rodeil = Whisper variant of Rodale; same description; mc=1",
    "c0007": "Wele/Whaley Waters are Wele Waters product variants",
    "c0010": "Weyland Waters = Whisper variant; citation confirms 'Waylay and Waters'",
    "c0013": "Doctor Bronners = Dr. Bronner's (just expansion of abbreviation)",
    "c0023": "Weylay/Weyley Waters Soaking Salts variants",
    "c0031": "Association of Water Schools = Whisper noise for Waldorf Schools",
    "c0035": "Mycelium Network singular = Mycelial Networks plural; same biology concept",
    "c0043": "Silvo Pasture = silvopasture (real agroforestry term); same concept",
    "c0067": "HOA abbreviation = Homeowners Associations expansion",
    "c0082": "Planting Trees = tree planting (word order only)",
    "c0083": "Earth X Film Festival = EarthX Film (same event, fuller name)",
    "c0089": "Judith D. Schwartz = Judith Schwartz (middle initial)",
    "c0091": "loss of biodiversity = Biodiversity Loss (word order)",
    "c0114": "Grow Domes = Growing Dome (citation: 'company that makes these grow domes')",
    "c0125": "income inequality is core component of economic inequality; same KG concept",
    "c0141": "Self-Driving Cars = Autonomous Vehicles (synonyms)",
    "c0146": "Chelsea Green Publishers = Chelsea Green Publishing (same company)",
    "c0179": "Industrialized Agriculture = Industrial Agriculture (morphology)",
    "c0230": "connection with nature = Nature Connection (word order)",
    "c0251": "River Ganges = Ganges River (word order)",
    "c0397": "Aggregated Soil = Soil Aggregates (morphology)",
    "c0451": "Carbon Nitrogen Ratio = Carbon to Nitrogen Ratio (preposition elision)",
    "c0465": "sixth great extinction = Sixth Grade Extinction (canonical itself is typo of 'great'); same scientific concept",
    "c0485": "Organic Regenerative Farming = Regenerative Organic Farming (word order)",
    "c0494": "Biodynamic Preps = Biodynamic Soil Preps (abbreviation)",
    "c0496": "Dr. Bronner's chocolate = Dr. Bronner's Magic Chocolate (apostrophe + brand-name omission)",
    "c0517": "YODE Earth Community Podcast = Wyon Earth Community Podcast (both Whisper variants of Y on Earth Community Podcast)",
    "c0568": "Rhine Mystic Movement = Rhineland Mystic movement (Rhineland → Rhine abbreviation)",
    "c0610": "February Conference = Conference in February (word order)",
    "c0643": "pharmaceutical industry = pharmaceutical industries (singular/plural)",
    "c0653": "Democratic Republic of Congo = Democratic Republic of the Congo (article elision)",
    "c0657": "Ring of Fire Kiln = Ring of Fire Biochar Kiln (abbreviation)",
    "c0690": "Judeo-Christian creation story = Judeo-Christian creation stories (singular/plural)",
    "c0702": "Recycled Materials ~ Recyclable Materials (close enough; LLM extraction noise)",
    "c0725": "Writers Community podcast = Winner's Community Podcast (both Whisper variants of Y on Earth Community Podcast)",
    "c0727": "Mesoamerican Reef → Rift is Whisper noise; citation confirms 'reef' in source ('Mesoamerican reef that actually represent almost the 15% of biodiversity')",
}

REJECT_CLUSTERS: dict[str, str] = {
    "c0075": "Iowa State + Ohio State Universities are DIFFERENT institutions from Colorado State",
    "c0118": "American Bar Association (lawyers) ≠ American Medical Association (physicians)",
    "c0121": "Sonoma State (California) ≠ Oklahoma State (different states + universities)",
    "c0507": "Corn Residues are a SUBSET/specific type of Crop Residues; keep distinct in KG",
    "c0508": "HP Biochar (High Plains Biochar company) ≠ biocharco.op (different domain/entity)",
    "c0736": "Norwegian government ≠ Singaporean government (different countries)",
    "c0744": "neo-colonial vs post-colonial economics — distinct academic concepts (perpetuation vs critique)",
}


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    fieldnames = list(rows[0].keys())
    if "auto_reason" not in fieldnames:
        fieldnames.append("auto_reason")

    counts = {"accept_review": 0, "reject_review": 0, "untouched": 0, "unknown_cluster": 0}
    cluster_seen: set[str] = set()

    for r in rows:
        if r.get("status") == "canonical":
            counts["untouched"] += 1
            continue
        cid = r.get("cluster_id", "")
        # Only override when the row was previously left for review (status == "")
        if r.get("status") not in ("", None):
            counts["untouched"] += 1
            continue
        if cid in ACCEPT_CLUSTERS:
            r["status"] = "accept"
            r["auto_reason"] = f"my-review: {ACCEPT_CLUSTERS[cid]}"
            counts["accept_review"] += 1
            cluster_seen.add(cid)
        elif cid in REJECT_CLUSTERS:
            r["status"] = "reject"
            r["auto_reason"] = f"my-review: {REJECT_CLUSTERS[cid]}"
            counts["reject_review"] += 1
            cluster_seen.add(cid)
        else:
            counts["unknown_cluster"] += 1

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Updated rows: accept={counts['accept_review']}  reject={counts['reject_review']}  unknown_cluster={counts['unknown_cluster']}", file=sys.stderr)
    print(f"Clusters covered: {len(cluster_seen)} / {len(ACCEPT_CLUSTERS) + len(REJECT_CLUSTERS)}", file=sys.stderr)

    if counts["unknown_cluster"]:
        print("\nWARN: rows still with no decision (cluster not in my dicts):", file=sys.stderr)
        for r in rows:
            if r.get("status") in ("", None) and r.get("status") != "canonical":
                print(f"  cluster={r['cluster_id']} canonical={r['canonical_name']!r} member={r['member_name']!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
