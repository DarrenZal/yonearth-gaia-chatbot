#!/usr/bin/env python3
"""
Compare GPT-5-mini vs GPT-4o-mini for Dual-Signal Extraction

Key metrics:
1. Entity pair coverage (gpt-4o-mini: 10.2%)
2. Conflict detection (gpt-4o-mini: 6 conflicts)
3. Relationship count
4. Quality of error detection
"""

import json
from pathlib import Path

DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
GPT5_DIR = DATA_DIR / "knowledge_graph_gpt5_mini_test"
GPT4O_DIR = DATA_DIR / "knowledge_graph_dual_signal_test"
SINGLE_SIGNAL_DIR = DATA_DIR / "knowledge_graph_v2"


def normalize_entity(entity):
    """Normalize entity name for comparison"""
    return entity.lower().strip()


def get_entity_pairs(relationships):
    """Extract (source, target) pairs from relationships"""
    pairs = set()
    for r in relationships:
        key = (normalize_entity(r['source']), normalize_entity(r['target']))
        pairs.add(key)
    return pairs


def load_episode(directory, ep_num, suffix=""):
    """Load episode extraction from directory"""
    if suffix:
        path = directory / f"episode_{ep_num}_{suffix}.json"
    else:
        path = directory / f"episode_{ep_num}_extraction.json"

    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    return data.get('relationships', [])


def analyze_episode(ep_num):
    """Compare GPT-5-mini vs GPT-4o-mini for one episode"""
    print(f"\n{'='*80}")
    print(f"EPISODE {ep_num} COMPARISON")
    print('='*80)

    # Load data
    gpt5_rels = load_episode(GPT5_DIR, ep_num, "gpt5_mini")
    gpt4o_rels = load_episode(GPT4O_DIR, ep_num, "dual_signal")
    single_rels = load_episode(SINGLE_SIGNAL_DIR, ep_num)

    if not gpt5_rels or not gpt4o_rels or not single_rels:
        print("⚠️  Missing data")
        return None

    # Get entity pairs
    gpt5_pairs = get_entity_pairs(gpt5_rels)
    gpt4o_pairs = get_entity_pairs(gpt4o_rels)
    single_pairs = get_entity_pairs(single_rels)

    # Calculate coverage
    gpt5_coverage = len(gpt5_pairs & single_pairs) / len(single_pairs) if single_pairs else 0
    gpt4o_coverage = len(gpt4o_pairs & single_pairs) / len(single_pairs) if single_pairs else 0

    # Count conflicts and type violations
    gpt5_conflicts = sum(1 for r in gpt5_rels if r.get('signals_conflict', False))
    gpt4o_conflicts = sum(1 for r in gpt4o_rels if r.get('signals_conflict', False))

    gpt5_type_viols = sum(1 for r in gpt5_rels if r.get('type_constraint_violated', False))
    gpt4o_type_viols = sum(1 for r in gpt4o_rels if r.get('type_constraint_violated', False))

    print(f"\n📊 RELATIONSHIP COUNTS:")
    print(f"  Single-signal (baseline): {len(single_rels)} relationships, {len(single_pairs)} entity pairs")
    print(f"  GPT-4o-mini: {len(gpt4o_rels)} relationships, {len(gpt4o_pairs)} entity pairs")
    print(f"  GPT-5-mini:  {len(gpt5_rels)} relationships, {len(gpt5_pairs)} entity pairs")

    print(f"\n🎯 ENTITY PAIR COVERAGE (vs single-signal baseline):")
    print(f"  GPT-4o-mini: {gpt4o_coverage*100:.1f}% ({len(gpt4o_pairs & single_pairs)}/{len(single_pairs)})")
    print(f"  GPT-5-mini:  {gpt5_coverage*100:.1f}% ({len(gpt5_pairs & single_pairs)}/{len(single_pairs)})")

    if gpt5_coverage > gpt4o_coverage:
        improvement = (gpt5_coverage - gpt4o_coverage) * 100
        print(f"  ✅ GPT-5-mini is {improvement:.1f} percentage points better!")
    elif gpt5_coverage < gpt4o_coverage:
        decline = (gpt4o_coverage - gpt5_coverage) * 100
        print(f"  ⚠️  GPT-5-mini is {decline:.1f} percentage points worse")
    else:
        print(f"  ⚖️  Same coverage")

    print(f"\n🚨 ERROR DETECTION:")
    print(f"  GPT-4o-mini conflicts: {gpt4o_conflicts}, type violations: {gpt4o_type_viols}")
    print(f"  GPT-5-mini conflicts:  {gpt5_conflicts}, type violations: {gpt5_type_viols}")

    # Show examples of relationships GPT-5 found that GPT-4o missed
    gpt5_only = gpt5_pairs - gpt4o_pairs - (gpt5_pairs - single_pairs)
    if gpt5_only:
        print(f"\n✨ ENTITY PAIRS GPT-5-MINI FOUND (that GPT-4o-mini missed):")
        for i, (src, tgt) in enumerate(list(gpt5_only)[:5], 1):
            # Find the relationship
            rel = next((r for r in gpt5_rels if
                       normalize_entity(r['source']) == src and
                       normalize_entity(r['target']) == tgt), None)
            if rel:
                print(f"  {i}. {src} --[{rel['relationship']}]--> {tgt}")

    return {
        'episode': ep_num,
        'gpt5_coverage': gpt5_coverage,
        'gpt4o_coverage': gpt4o_coverage,
        'gpt5_conflicts': gpt5_conflicts,
        'gpt4o_conflicts': gpt4o_conflicts,
        'gpt5_pairs': len(gpt5_pairs),
        'gpt4o_pairs': len(gpt4o_pairs),
        'single_pairs': len(single_pairs)
    }


def main():
    """Compare all test episodes"""
    print("="*80)
    print("🔬 GPT-5-MINI vs GPT-4O-MINI COMPARISON")
    print("="*80)

    test_episodes = [10, 39, 50, 75, 100, 112, 120, 122, 150, 165]

    results = []
    for ep in test_episodes:
        result = analyze_episode(ep)
        if result:
            results.append(result)

    # Overall summary
    print(f"\n{'='*80}")
    print("📊 OVERALL SUMMARY")
    print('='*80)

    total_single_pairs = sum(r['single_pairs'] for r in results)
    total_gpt4o_pairs = sum(r['gpt4o_pairs'] for r in results)
    total_gpt5_pairs = sum(r['gpt5_pairs'] for r in results)

    gpt4o_overall_coverage = total_gpt4o_pairs / total_single_pairs if total_single_pairs else 0
    gpt5_overall_coverage = total_gpt5_pairs / total_single_pairs if total_single_pairs else 0

    total_gpt4o_conflicts = sum(r['gpt4o_conflicts'] for r in results)
    total_gpt5_conflicts = sum(r['gpt5_conflicts'] for r in results)

    print(f"\nAcross {len(results)} episodes:")
    print(f"\n📊 ENTITY PAIR COVERAGE:")
    print(f"  Single-signal baseline: {total_single_pairs} pairs")
    print(f"  GPT-4o-mini: {total_gpt4o_pairs} pairs ({gpt4o_overall_coverage*100:.1f}% coverage)")
    print(f"  GPT-5-mini:  {total_gpt5_pairs} pairs ({gpt5_overall_coverage*100:.1f}% coverage)")

    print(f"\n🚨 ERROR DETECTION:")
    print(f"  GPT-4o-mini: {total_gpt4o_conflicts} conflicts detected")
    print(f"  GPT-5-mini:  {total_gpt5_conflicts} conflicts detected")

    print(f"\n🎯 VERDICT:")

    if gpt5_overall_coverage > gpt4o_overall_coverage * 1.5:
        print("  ✅ GPT-5-MINI IS SIGNIFICANTLY BETTER (>50% improvement)")
        print("     Recommendation: Use gpt-5-mini for full extraction")
    elif gpt5_overall_coverage > gpt4o_overall_coverage * 1.2:
        print("  ✅ GPT-5-MINI IS BETTER (>20% improvement)")
        print("     Recommendation: Use gpt-5-mini for full extraction")
    elif gpt5_overall_coverage > gpt4o_overall_coverage:
        print("  ✅ GPT-5-MINI IS SLIGHTLY BETTER")
        print("     Recommendation: Consider using gpt-5-mini if cost is similar")
    elif gpt5_overall_coverage == gpt4o_overall_coverage:
        print("  ⚖️  SAME PERFORMANCE")
        print("     Recommendation: Use cheaper model (gpt-4o-mini)")
    else:
        print("  ⚠️  GPT-5-MINI IS WORSE")
        print("     Recommendation: Stick with gpt-4o-mini OR try two-pass approach")

    # Check if EITHER model achieves good coverage
    if max(gpt5_overall_coverage, gpt4o_overall_coverage) > 0.5:
        print(f"\n  ✨ Coverage is good enough for single-pass approach!")
    else:
        print(f"\n  ⚠️  Coverage still poor ({max(gpt5_overall_coverage, gpt4o_overall_coverage)*100:.1f}%)")
        print("     Consider Option B (two-pass) or Option C (validate v2)")


if __name__ == "__main__":
    main()
