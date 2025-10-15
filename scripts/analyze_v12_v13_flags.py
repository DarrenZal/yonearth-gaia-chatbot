#!/usr/bin/env python3
"""
Analyze classification_flags differences between V12 and V13
"""

import json
from collections import Counter
from pathlib import Path

# Paths
v12_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v12/soil_stewardship_handbook_v12.json")
v13_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v13/soil_stewardship_handbook_v13_from_v12.json")

# Load both outputs
with open(v12_path) as f:
    v12 = json.load(f)

with open(v13_path) as f:
    v13 = json.load(f)

print("="*80)
print("🔍 V12 vs V13 CLASSIFICATION FLAGS ANALYSIS")
print("="*80)
print()

# Count flags in each version
v12_flags = []
for rel in v12['relationships']:
    v12_flags.extend(rel.get('classification_flags', []))

v13_flags = []
for rel in v13['relationships']:
    v13_flags.extend(rel.get('classification_flags', []))

v12_counts = Counter(v12_flags)
v13_counts = Counter(v13_flags)

print("📊 V12 CLASSIFICATION FLAGS:")
for flag, count in v12_counts.most_common():
    print(f"   {flag}: {count}")

print()
print("📊 V13 CLASSIFICATION FLAGS:")
for flag, count in v13_counts.most_common():
    print(f"   {flag}: {count}")

print()
print("📈 FLAG DIFFERENCES:")
all_flags = set(v12_counts.keys()) | set(v13_counts.keys())
for flag in sorted(all_flags):
    v12_count = v12_counts.get(flag, 0)
    v13_count = v13_counts.get(flag, 0)
    diff = v13_count - v12_count
    if diff != 0:
        print(f"   {flag}: {v12_count} → {v13_count} ({diff:+d})")

print()
print("="*80)
print("🔍 NORMATIVE FLAG INVESTIGATION")
print("="*80)
print()

# Find relationships with NORMATIVE in V12
v12_normative_rels = [rel for rel in v12['relationships'] if 'NORMATIVE' in rel.get('classification_flags', [])]

print(f"Found {len(v12_normative_rels)} relationships with NORMATIVE flag in V12")
print()

if v12_normative_rels:
    print("Sample NORMATIVE relationships from V12:")
    for i, rel in enumerate(v12_normative_rels[:5], 1):
        print(f"\n{i}. ({rel['source']}) --[{rel['relationship']}]--> ({rel['target']})")
        print(f"   p_true: {rel['p_true']:.2f}")
        print(f"   flags: {rel.get('classification_flags', [])}")
        print(f"   context: {rel['context'][:150]}...")

# Check if those same relationships exist in V13 and what flags they have
print()
print("="*80)
print("🔍 WHERE DID NORMATIVE RELATIONSHIPS GO IN V13?")
print("="*80)
print()

# Create lookup by source-relationship-target
v13_lookup = {}
for rel in v13['relationships']:
    key = (rel['source'], rel['relationship'], rel['target'])
    v13_lookup[key] = rel

matched = 0
missing = 0
for v12_rel in v12_normative_rels:
    key = (v12_rel['source'], v12_rel['relationship'], v12_rel['target'])

    if key in v13_lookup:
        matched += 1
        v13_rel = v13_lookup[key]

        v12_flags = v12_rel.get('classification_flags', [])
        v13_flags = v13_rel.get('classification_flags', [])

        if v12_flags != v13_flags:
            print(f"⚠️  Flags changed:")
            print(f"   ({v12_rel['source']}) --[{v12_rel['relationship']}]--> ({v12_rel['target']})")
            print(f"   V12 flags: {v12_flags}")
            print(f"   V13 flags: {v13_flags}")
            print(f"   V12 p_true: {v12_rel['p_true']:.2f} | V13 p_true: {v13_rel['p_true']:.2f}")
            print()
    else:
        missing += 1

print(f"Matched: {matched}/{len(v12_normative_rels)} NORMATIVE relationships found in V13")
print(f"Missing: {missing}/{len(v12_normative_rels)} NORMATIVE relationships not found in V13")
print()

print("="*80)
print("🎯 ROOT CAUSE ANALYSIS")
print("="*80)
print()

print("V12 classification_flags options (from prompt):")
print("   Likely includes: FACTUAL, TESTABLE_CLAIM, NORMATIVE, PHILOSOPHICAL_CLAIM, etc.")
print()

print("V13 classification_flags options (from prompt):")
print('   ["FACTUAL" | "TESTABLE_CLAIM" | "PHILOSOPHICAL_CLAIM" | "METAPHOR" | "OPINION" | "ABSTRACT_CONCEPT"]')
print()
print("🚨 FINDING: V13 prompt REMOVED 'NORMATIVE' from classification_flags options!")
print("   - The prompt text mentions detecting 'normative prescriptions'")
print("   - But 'NORMATIVE' is NOT in the allowed flags list")
print("   - LLM had no way to classify normative claims")
print()
