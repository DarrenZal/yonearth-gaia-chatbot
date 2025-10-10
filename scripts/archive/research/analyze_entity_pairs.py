#!/usr/bin/env python3
"""
Analyze Entity Pairs: Dual-Signal vs Single-Signal

Compare based on source-target entity pairs, ignoring relationship wording.
This tells us if dual-signal is missing FACTS, not just using different words.
"""

import json
from pathlib import Path

DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
DUAL_DIR = DATA_DIR / "knowledge_graph_dual_signal_test"
SINGLE_DIR = DATA_DIR / "knowledge_graph_v2"

def normalize_entity(entity):
    """Normalize entity name for comparison"""
    return entity.lower().strip()

def get_entity_pairs(relationships):
    """Extract (source, target) pairs from relationships"""
    pairs = {}
    for r in relationships:
        key = (normalize_entity(r['source']), normalize_entity(r['target']))
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(r)
    return pairs

def load_dual_signal(ep_num):
    """Load dual-signal extraction"""
    path = DUAL_DIR / f"episode_{ep_num}_dual_signal.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('relationships', [])

def load_single_signal(ep_num):
    """Load single-signal extraction"""
    path = SINGLE_DIR / f"episode_{ep_num}_extraction.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('relationships', [])

def analyze_entity_pairs(ep_num):
    """Analyze entity pair coverage"""
    print(f"\n{'='*80}")
    print(f"EPISODE {ep_num}: ENTITY PAIR ANALYSIS")
    print(f"(Comparing source-target pairs, ignoring relationship wording)")
    print('='*80)

    dual_rels = load_dual_signal(ep_num)
    single_rels = load_single_signal(ep_num)

    if not dual_rels or not single_rels:
        print("⚠️  Data not available")
        return None

    dual_pairs = get_entity_pairs(dual_rels)
    single_pairs = get_entity_pairs(single_rels)

    # Find pairs in single but not in dual
    missing_pairs = {}
    for key in single_pairs:
        if key not in dual_pairs:
            missing_pairs[key] = single_pairs[key]

    # Find pairs in dual but not in single
    new_pairs = {}
    for key in dual_pairs:
        if key not in single_pairs:
            new_pairs[key] = dual_pairs[key]

    # Find pairs in BOTH (might have different relationship wordings)
    shared_pairs = {}
    for key in dual_pairs:
        if key in single_pairs:
            shared_pairs[key] = {
                'dual': dual_pairs[key],
                'single': single_pairs[key]
            }

    print(f"\n📊 ENTITY PAIR COVERAGE:")
    print(f"  Single-signal pairs: {len(single_pairs)}")
    print(f"  Dual-signal pairs: {len(dual_pairs)}")
    print(f"  Shared pairs: {len(shared_pairs)} ({len(shared_pairs)/len(single_pairs)*100:.1f}% of single)")
    print(f"  Missing in dual: {len(missing_pairs)} ({len(missing_pairs)/len(single_pairs)*100:.1f}%)")
    print(f"  New in dual: {len(new_pairs)} ({len(new_pairs)/len(dual_pairs)*100:.1f}%)")

    # Analyze missing pairs by confidence
    if missing_pairs:
        high_conf_missing = []
        for key, rels in missing_pairs.items():
            for r in rels:
                avg_conf = (
                    r.get('source_confidence', 1.0) +
                    r.get('relationship_confidence', 1.0) +
                    r.get('target_confidence', 1.0)
                ) / 3
                if avg_conf >= 0.75:
                    high_conf_missing.append((avg_conf, key, r))

        print(f"\n⚠️  HIGH-CONFIDENCE ENTITY PAIRS MISSING:")
        print(f"   Total: {len(high_conf_missing)}")

        if high_conf_missing:
            high_conf_missing.sort(key=lambda x: x[0], reverse=True)
            print(f"\n   Examples:")
            for i, (conf, (src, tgt), rel) in enumerate(high_conf_missing[:5], 1):
                print(f"   {i}. {src} --> {tgt}")
                print(f"      Single-signal: {rel['relationship']} (conf: {conf:.2f})")
                print()

    # Show examples of shared pairs with different wording
    if shared_pairs:
        print(f"\n✅ SHARED ENTITY PAIRS (same facts, different wording):")
        print(f"   Examples of how dual-signal phrases relationships:\n")

        for i, (key, rels) in enumerate(list(shared_pairs.items())[:5], 1):
            src, tgt = key
            dual_rel = rels['dual'][0]
            single_rel = rels['single'][0]

            print(f"   {i}. {src} --> {tgt}")
            print(f"      Single: '{single_rel['relationship']}'")
            print(f"      Dual:   '{dual_rel['relationship']}'")
            print(f"      Text conf: {dual_rel.get('text_confidence', 'N/A')}, Knowledge: {dual_rel.get('knowledge_plausibility', 'N/A')}")
            print()

    return {
        'episode': ep_num,
        'single_pairs': len(single_pairs),
        'dual_pairs': len(dual_pairs),
        'shared_pairs': len(shared_pairs),
        'missing_pairs': len(missing_pairs),
        'new_pairs': len(new_pairs),
        'high_conf_missing': len(high_conf_missing) if missing_pairs else 0
    }

def main():
    """Analyze all test episodes"""
    print("="*80)
    print("🔬 ENTITY PAIR ANALYSIS")
    print("Comparing source-target pairs (ignoring relationship wording)")
    print("="*80)

    test_episodes = [10, 39, 50, 75, 100, 112, 120, 122, 150, 165]

    results = []
    for ep in test_episodes:
        result = analyze_entity_pairs(ep)
        if result:
            results.append(result)

    # Overall summary
    print(f"\n{'='*80}")
    print("📊 OVERALL SUMMARY")
    print('='*80)

    total_single_pairs = sum(r['single_pairs'] for r in results)
    total_dual_pairs = sum(r['dual_pairs'] for r in results)
    total_shared_pairs = sum(r['shared_pairs'] for r in results)
    total_missing_pairs = sum(r['missing_pairs'] for r in results)
    total_new_pairs = sum(r['new_pairs'] for r in results)
    total_high_conf_missing = sum(r['high_conf_missing'] for r in results)

    print(f"\nAcross all {len(results)} episodes:")
    print(f"  Single-signal entity pairs: {total_single_pairs}")
    print(f"  Dual-signal entity pairs: {total_dual_pairs}")
    print(f"  Shared pairs: {total_shared_pairs} ({total_shared_pairs/total_single_pairs*100:.1f}% of single)")
    print(f"  Missing in dual: {total_missing_pairs} ({total_missing_pairs/total_single_pairs*100:.1f}%)")
    print(f"  New in dual: {total_new_pairs}")
    print(f"  High-conf pairs missing: {total_high_conf_missing}")

    print(f"\n🎯 INTERPRETATION:")

    coverage = total_shared_pairs / total_single_pairs if total_single_pairs > 0 else 0

    if coverage >= 0.8:
        print(f"  ✅ EXCELLENT: Dual-signal covers {coverage*100:.1f}% of single-signal facts")
        print("     The differences are mainly in relationship phrasing, not missing facts")
    elif coverage >= 0.6:
        print(f"  ✅ GOOD: Dual-signal covers {coverage*100:.1f}% of single-signal facts")
        print("     Some facts are missing, but most are captured")
    else:
        print(f"  ⚠️  WARNING: Dual-signal only covers {coverage*100:.1f}% of single-signal facts")
        print("     Many entity pairs are missing - this is a problem")

    if total_high_conf_missing > 0:
        print(f"\n  ⚠️  {total_high_conf_missing} high-confidence entity pairs are missing")
        print("     These represent real facts that dual-signal failed to extract")

if __name__ == "__main__":
    main()
