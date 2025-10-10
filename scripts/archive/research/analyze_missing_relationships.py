#!/usr/bin/env python3
"""
Analyze Missing Relationships: Dual-Signal vs Single-Signal

Investigates what relationships appear in single-signal but NOT in dual-signal.
Determines if these are:
1. Noise/low-quality (good to filter)
2. Valid relationships (bad to miss)
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
DUAL_DIR = DATA_DIR / "knowledge_graph_dual_signal_test"
SINGLE_DIR = DATA_DIR / "knowledge_graph_v2"

def normalize_relationship(source, rel, target):
    """Normalize for comparison (case-insensitive, whitespace-normalized)"""
    return (
        source.lower().strip(),
        rel.lower().strip(),
        target.lower().strip()
    )

def load_dual_signal(ep_num):
    """Load dual-signal extraction"""
    path = DUAL_DIR / f"episode_{ep_num}_dual_signal.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    # Extract relationships as normalized tuples
    relationships = {}
    for r in data.get('relationships', []):
        key = normalize_relationship(r['source'], r['relationship'], r['target'])
        relationships[key] = r

    return relationships

def load_single_signal(ep_num):
    """Load single-signal extraction"""
    path = SINGLE_DIR / f"episode_{ep_num}_extraction.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    # Extract relationships as normalized tuples
    relationships = {}
    for r in data.get('relationships', []):
        key = normalize_relationship(r['source'], r['relationship'], r['target'])
        relationships[key] = r

    return relationships

def analyze_missing(ep_num):
    """Analyze what's missing in dual-signal vs single-signal"""
    print(f"\n{'='*80}")
    print(f"EPISODE {ep_num}: MISSING RELATIONSHIP ANALYSIS")
    print('='*80)

    dual = load_dual_signal(ep_num)
    single = load_single_signal(ep_num)

    if not dual or not single:
        print("⚠️  Data not available")
        return None

    # Find relationships in single but NOT in dual
    missing = []
    for key, rel in single.items():
        if key not in dual:
            missing.append(rel)

    # Find relationships in dual but NOT in single (new discoveries)
    new_in_dual = []
    for key, rel in dual.items():
        if key not in single:
            new_in_dual.append(rel)

    print(f"\n📊 COUNTS:")
    print(f"  Single-signal total: {len(single)}")
    print(f"  Dual-signal total: {len(dual)}")
    print(f"  Missing in dual: {len(missing)} ({len(missing)/len(single)*100:.1f}%)")
    print(f"  New in dual: {len(new_in_dual)} ({len(new_in_dual)/len(dual)*100:.1f}%)")

    # Categorize missing relationships by confidence
    high_conf_missing = []
    med_conf_missing = []
    low_conf_missing = []

    for rel in missing:
        # Calculate average confidence from single-signal
        avg_conf = (
            rel.get('source_confidence', 1.0) +
            rel.get('relationship_confidence', 1.0) +
            rel.get('target_confidence', 1.0)
        ) / 3

        if avg_conf >= 0.75:
            high_conf_missing.append((avg_conf, rel))
        elif avg_conf >= 0.5:
            med_conf_missing.append((avg_conf, rel))
        else:
            low_conf_missing.append((avg_conf, rel))

    print(f"\n🔍 MISSING BREAKDOWN BY SINGLE-SIGNAL CONFIDENCE:")
    print(f"  High confidence (≥0.75): {len(high_conf_missing)}")
    print(f"  Medium confidence (0.5-0.75): {len(med_conf_missing)}")
    print(f"  Low confidence (<0.5): {len(low_conf_missing)}")

    # Show examples of high-confidence missing relationships
    if high_conf_missing:
        print(f"\n⚠️  HIGH-CONFIDENCE RELATIONSHIPS MISSING IN DUAL-SIGNAL:")
        print(f"   (These might be VALID relationships we're losing!)\n")

        # Sort by confidence (tuples sort by first element)
        high_conf_missing.sort(key=lambda x: x[0], reverse=True)

        for i, (conf, rel) in enumerate(high_conf_missing[:10], 1):
            print(f"  {i}. {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
            print(f"     Single-signal confidence: {conf:.2f}")
            print(f"     Context: {rel.get('context', 'N/A')[:100]}...")
            print()

    # Show examples of low-confidence missing relationships
    if low_conf_missing:
        print(f"\n✅ LOW-CONFIDENCE RELATIONSHIPS MISSING IN DUAL-SIGNAL:")
        print(f"   (These are likely NOISE - good to filter!)\n")

        # Sort by confidence (lowest first)
        low_conf_missing.sort(key=lambda x: x[0])

        for i, (conf, rel) in enumerate(low_conf_missing[:5], 1):
            print(f"  {i}. {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
            print(f"     Single-signal confidence: {conf:.2f}")
            print()

    return {
        'episode': ep_num,
        'total_single': len(single),
        'total_dual': len(dual),
        'missing_count': len(missing),
        'missing_high_conf': len(high_conf_missing),
        'missing_med_conf': len(med_conf_missing),
        'missing_low_conf': len(low_conf_missing),
        'new_in_dual': len(new_in_dual)
    }

def main():
    """Analyze all test episodes"""
    print("="*80)
    print("🔬 MISSING RELATIONSHIP ANALYSIS")
    print("Investigating what dual-signal filters out vs single-signal")
    print("="*80)

    test_episodes = [10, 39, 50, 75, 100, 112, 120, 122, 150, 165]

    results = []
    for ep in test_episodes:
        result = analyze_missing(ep)
        if result:
            results.append(result)

    # Overall summary
    print(f"\n{'='*80}")
    print("📊 OVERALL SUMMARY")
    print('='*80)

    total_single = sum(r['total_single'] for r in results)
    total_dual = sum(r['total_dual'] for r in results)
    total_missing = sum(r['missing_count'] for r in results)
    total_high_conf_missing = sum(r['missing_high_conf'] for r in results)
    total_med_conf_missing = sum(r['missing_med_conf'] for r in results)
    total_low_conf_missing = sum(r['missing_low_conf'] for r in results)

    print(f"\nAcross all {len(results)} episodes:")
    print(f"  Single-signal total: {total_single}")
    print(f"  Dual-signal total: {total_dual}")
    print(f"  Missing in dual: {total_missing} ({total_missing/total_single*100:.1f}%)")

    print(f"\nMissing breakdown:")
    print(f"  High confidence (≥0.75): {total_high_conf_missing} ({total_high_conf_missing/total_missing*100:.1f}%)")
    print(f"  Medium confidence (0.5-0.75): {total_med_conf_missing} ({total_med_conf_missing/total_missing*100:.1f}%)")
    print(f"  Low confidence (<0.5): {total_low_conf_missing} ({total_low_conf_missing/total_missing*100:.1f}%)")

    print(f"\n🎯 INTERPRETATION:")

    if total_high_conf_missing > total_missing * 0.3:
        print("  ⚠️  WARNING: Dual-signal is filtering out many HIGH-confidence relationships!")
        print("     This suggests we're LOSING valid data. We need to fix this.")
    elif total_low_conf_missing > total_missing * 0.5:
        print("  ✅ GOOD: Dual-signal is primarily filtering LOW-confidence relationships")
        print("     This is NOISE REDUCTION - exactly what we want!")
    else:
        print("  ⚠️  MIXED: Dual-signal filters both high and low confidence relationships")
        print("     Need to investigate further to understand the pattern")

    # Calculate what percentage of single-signal HIGH conf is being kept
    print(f"\n📈 RETENTION RATES:")
    print(f"  Overall retention: {total_dual/total_single*100:.1f}%")
    print(f"  High-conf relationships lost: {total_high_conf_missing}")
    print(f"  Low-conf relationships lost: {total_low_conf_missing}")

if __name__ == "__main__":
    main()
