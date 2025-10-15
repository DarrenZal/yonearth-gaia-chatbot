#!/usr/bin/env python3
"""
Check V8 Extraction Results

Displays V8 extraction results and compares to V7 baseline.
"""

import json
from pathlib import Path

print("="*80)
print("📊 V8 EXTRACTION RESULTS")
print("="*80)
print()

# Check for V8 output
v8_output_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v8")
v8_files = list(v8_output_dir.glob("soil_stewardship_handbook_v8.json"))

if not v8_files:
    print("❌ V8 extraction not complete yet")
    print()
    print("Check progress:")
    print("  tail -50 v8_extraction.log | grep -E '(PASS|Chunk|Batch|✅|COMPLETE)'")
    print()
    print("Or wait and run this script again when extraction completes.")
    exit(0)

v8_file = v8_files[0]
print(f"✅ Found V8 output: {v8_file.name}")
print()

# Load V8 results
with open(v8_file) as f:
    v8_results = json.load(f)

# Display V8 results
print("="*80)
print("V8 CURATOR-ENHANCED EXTRACTION RESULTS")
print("="*80)
print()

print(f"📚 Book: {v8_results['book_title']}")
print(f"📅 Extracted: {v8_results['timestamp']}")
print(f"⏱️  Time: {v8_results['extraction_time_minutes']} minutes")
print(f"📄 Pages processed: {v8_results['pages']}")
print(f"📊 Page coverage: {v8_results['page_coverage_percentage']}%")
print()

print("EXTRACTION PIPELINE:")
print(f"  Pass 1 candidates: {v8_results['pass1_candidates']}")
print(f"  Pass 2 evaluated: {v8_results['pass2_evaluated']}")
print(f"  Pass 2.5 final: {v8_results['pass2_5_final']}")
print()

# Pass 2.5 stats (V8 enhancements)
stats = v8_results['pass2_5_stats']
print("✨ V8 CURATOR-ENHANCED PASS 2.5 STATS:")
print(f"  - ✨ Praise quotes corrected: {stats.get('praise_quotes_corrected', 0)}")
print(f"  - ✨ Authorships reversed: {stats['authorship_reversed']}, Endorsements: {stats['endorsements_detected']}, Dedications: {stats.get('dedications_corrected', 0)}")
print(f"  - ✨ Pronouns resolved (enhanced): {stats['pronouns_resolved']} anaphoric + {stats['generic_pronouns_resolved']} generic")
print(f"  - ✨ Vague entities replaced (context-aware): {stats['entities_enriched']}")
print(f"  - ✨ Lists split (with 'and'): {stats['lists_split']}")
print(f"  - ✨ Predicates normalized: {stats['predicates_normalized']}, semantically corrected: {stats.get('predicates_semantically_corrected', 0)}")
print(f"  - ✨ Metaphors normalized: {stats.get('metaphors_normalized', 0)}")
print()

print("QUALITY METRICS:")
print(f"  High confidence (p≥0.75): {v8_results['high_confidence_count']} ({v8_results['high_confidence_count']/v8_results['pass2_5_final']*100:.1f}%)")
print(f"  Medium confidence: {v8_results['medium_confidence_count']} ({v8_results['medium_confidence_count']/v8_results['pass2_5_final']*100:.1f}%)")
print(f"  Low confidence: {v8_results['low_confidence_count']} ({v8_results['low_confidence_count']/v8_results['pass2_5_final']*100:.1f}%)")
print(f"  Conflicts detected: {v8_results['conflicts_detected']}")
print()

# Compare to V7 if available
v7_output_dir = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v7")
v7_files = list(v7_output_dir.glob("soil_stewardship_handbook_v7.json"))

if v7_files:
    print("="*80)
    print("V8 vs V7 COMPARISON")
    print("="*80)
    print()

    with open(v7_files[0]) as f:
        v7_results = json.load(f)

    v7_stats = v7_results['pass2_5_stats']

    print("RELATIONSHIP COUNTS:")
    print(f"  V7: {v7_results['pass2_5_final']} relationships")
    print(f"  V8: {v8_results['pass2_5_final']} relationships")
    diff = v8_results['pass2_5_final'] - v7_results['pass2_5_final']
    print(f"  Difference: {diff:+d} ({diff/v7_results['pass2_5_final']*100:+.1f}%)")
    print()

    print("PASS 2.5 CORRECTIONS COMPARISON:")
    metrics = [
        ('Authorship reversed', 'authorship_reversed'),
        ('Endorsements detected', 'endorsements_detected'),
        ('Pronouns resolved', 'pronouns_resolved'),
        ('Vague entities enriched', 'entities_enriched'),
        ('Lists split', 'lists_split'),
        ('Predicates normalized', 'predicates_normalized'),
    ]

    for label, key in metrics:
        v7_val = v7_stats.get(key, 0)
        v8_val = stats.get(key, 0)
        diff = v8_val - v7_val
        print(f"  {label:30s}: V7={v7_val:3d}, V8={v8_val:3d}, Δ={diff:+3d}")

    print()
    print("✨ V8 NEW ENHANCEMENTS:")
    v8_new = [
        ('Praise quotes corrected', 'praise_quotes_corrected'),
        ('Dedications corrected', 'dedications_corrected'),
        ('Predicates semantically corrected', 'predicates_semantically_corrected'),
        ('Metaphors normalized', 'metaphors_normalized'),
    ]

    for label, key in v8_new:
        v8_val = stats.get(key, 0)
        print(f"  {label:35s}: {v8_val:3d} (NEW in V8)")

    print()

print("="*80)
print("✅ V8 EXTRACTION COMPLETE")
print("="*80)
print()

print("NEXT STEPS:")
print("  1. Run Reflector on V8 to measure quality improvements")
print("     python scripts/run_reflector_on_v8.py")
print()
print("  2. Compare V8 vs V7 quality metrics")
print("     Expected: 6.71% → 2.7% issue rate (60% reduction)")
print()
print("  3. If <3% issue rate achieved, V8 becomes PRODUCTION SYSTEM")
print()
