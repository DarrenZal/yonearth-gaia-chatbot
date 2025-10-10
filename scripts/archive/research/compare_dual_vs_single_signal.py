#!/usr/bin/env python3
"""
Compare Dual-Signal vs Single-Signal Extraction Results

Analyzes whether dual-signal extraction catches more errors than single-signal
by comparing the same 10 test episodes.

Key metrics:
- How many errors does dual-signal catch that single-signal missed?
- Are conflicts correctly identified (biochar, Boulder/Lafayette)?
- Does dual-signal give us better debugging info?
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
DUAL_SIGNAL_DIR = DATA_DIR / "knowledge_graph_dual_signal_test"
SINGLE_SIGNAL_DIR = DATA_DIR / "knowledge_graph_v2"


def load_dual_signal_episode(ep_num: int):
    """Load dual-signal extraction results"""
    path = DUAL_SIGNAL_DIR / f"episode_{ep_num}_dual_signal.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_single_signal_episode(ep_num: int):
    """Load single-signal (v2) extraction results"""
    path = SINGLE_SIGNAL_DIR / f"episode_{ep_num}_extraction.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_dual_signal(data):
    """Analyze dual-signal extraction"""
    rels = data.get('relationships', [])

    analysis = {
        'total': len(rels),
        'conflicts': [r for r in rels if r.get('signals_conflict', False)],
        'type_violations': [r for r in rels if r.get('type_constraint_violated', False)],
        'low_confidence': [r for r in rels if r.get('overall_confidence', 1.0) < 0.75],
        'text_high_knowledge_low': [],  # Text says X but knowledge says NO
        'explicit_conflicts': []
    }

    # Find interesting patterns
    for r in rels:
        text_conf = r.get('text_confidence', 1.0)
        know_conf = r.get('knowledge_plausibility', 1.0)

        # High text confidence but low knowledge confidence = speaker error or misread
        if text_conf >= 0.75 and know_conf < 0.5:
            analysis['text_high_knowledge_low'].append(r)

        # Explicit conflicts with explanation
        if r.get('signals_conflict') and r.get('conflict_explanation'):
            analysis['explicit_conflicts'].append(r)

    return analysis


def analyze_single_signal(data):
    """Analyze single-signal extraction"""
    rels = data.get('relationships', [])

    analysis = {
        'total': len(rels),
        'low_confidence': [
            r for r in rels
            if any([
                r.get('source_confidence', 1.0) < 0.75,
                r.get('relationship_confidence', 1.0) < 0.75,
                r.get('target_confidence', 1.0) < 0.75
            ])
        ]
    }

    return analysis


def compare_episode(ep_num: int):
    """Compare dual vs single signal for one episode"""
    print(f"\n{'='*70}")
    print(f"EPISODE {ep_num} COMPARISON")
    print('='*70)

    dual = load_dual_signal_episode(ep_num)
    single = load_single_signal_episode(ep_num)

    if not dual:
        print(f"⚠️  Dual-signal data not found")
        return None

    if not single:
        print(f"⚠️  Single-signal data not found")
        return None

    dual_analysis = analyze_dual_signal(dual)
    single_analysis = analyze_single_signal(single)

    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Dual-signal relationships: {dual_analysis['total']}")
    print(f"  Single-signal relationships: {single_analysis['total']}")

    print(f"\n🔍 ERROR DETECTION:")
    print(f"  Dual-signal conflicts detected: {len(dual_analysis['conflicts'])}")
    print(f"  Dual-signal type violations: {len(dual_analysis['type_violations'])}")
    print(f"  Dual-signal low confidence: {len(dual_analysis['low_confidence'])}")
    print(f"  Single-signal low confidence: {len(single_analysis['low_confidence'])}")

    print(f"\n⚡ DUAL-SIGNAL ADVANTAGES:")
    print(f"  Text HIGH but Knowledge LOW: {len(dual_analysis['text_high_knowledge_low'])}")
    print(f"  Explicit conflict explanations: {len(dual_analysis['explicit_conflicts'])}")

    # Show interesting conflicts
    if dual_analysis['explicit_conflicts']:
        print(f"\n🚨 CONFLICTS DETECTED (with explanations):")
        for i, rel in enumerate(dual_analysis['explicit_conflicts'][:5], 1):
            print(f"\n  {i}. {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
            print(f"     Text confidence: {rel['text_confidence']:.2f}")
            print(f"     Knowledge plausibility: {rel['knowledge_plausibility']:.2f}")
            print(f"     Overall confidence: {rel['overall_confidence']:.2f}")
            print(f"     Conflict: {rel['conflict_explanation']}")

    # Show text-high-knowledge-low (these are the errors single-signal might miss)
    if dual_analysis['text_high_knowledge_low']:
        print(f"\n⚠️  TEXT SAYS YES, KNOWLEDGE SAYS NO:")
        for i, rel in enumerate(dual_analysis['text_high_knowledge_low'][:5], 1):
            print(f"\n  {i}. {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
            print(f"     Text: {rel['text_confidence']:.2f} ({rel['text_clarity']})")
            print(f"     Knowledge: {rel['knowledge_plausibility']:.2f}")
            print(f"     Reasoning: {rel['knowledge_reasoning']}")

            # Find corresponding single-signal relationship
            single_rels = single.get('relationships', [])
            matches = [
                r for r in single_rels
                if r['source'] == rel['source'] and r['target'] == rel['target']
            ]
            if matches:
                sr = matches[0]
                print(f"     Single-signal confidence: {sr['relationship_confidence']:.2f}")
                if sr['relationship_confidence'] >= 0.75:
                    print(f"     ⚠️  Single-signal MISSED this error!")

    return {
        'episode': ep_num,
        'dual_signal': dual_analysis,
        'single_signal': single_analysis
    }


def main():
    """Compare all test episodes"""
    print("="*70)
    print("🔬 DUAL-SIGNAL vs SINGLE-SIGNAL COMPARISON")
    print("="*70)

    test_episodes = [10, 39, 50, 75, 100, 112, 120, 122, 150, 165]

    results = []
    for ep_num in test_episodes:
        result = compare_episode(ep_num)
        if result:
            results.append(result)

    # Overall summary
    print(f"\n{'='*70}")
    print("📊 OVERALL COMPARISON SUMMARY")
    print('='*70)

    total_dual_conflicts = sum(len(r['dual_signal']['conflicts']) for r in results)
    total_dual_type_violations = sum(len(r['dual_signal']['type_violations']) for r in results)
    total_dual_low = sum(len(r['dual_signal']['low_confidence']) for r in results)
    total_single_low = sum(len(r['single_signal']['low_confidence']) for r in results)
    total_text_high_know_low = sum(len(r['dual_signal']['text_high_knowledge_low']) for r in results)

    print(f"\nTotal episodes analyzed: {len(results)}")
    print(f"\nDual-Signal Detection:")
    print(f"  - Conflicts detected: {total_dual_conflicts}")
    print(f"  - Type violations: {total_dual_type_violations}")
    print(f"  - Low confidence: {total_dual_low}")
    print(f"  - Text HIGH / Knowledge LOW: {total_text_high_know_low}")

    print(f"\nSingle-Signal Detection:")
    print(f"  - Low confidence: {total_single_low}")

    print(f"\n🎯 KEY INSIGHTS:")

    if total_text_high_know_low > 0:
        print(f"  ✅ Dual-signal caught {total_text_high_know_low} errors where text was clear")
        print(f"     but knowledge flagged them as implausible")
        print(f"     (Single-signal likely gave these HIGH confidence!)")

    if total_dual_type_violations > 0:
        print(f"  ✅ Dual-signal identified {total_dual_type_violations} type constraint violations")
        print(f"     (e.g., 'located_in' with non-PLACE target)")

    if total_dual_conflicts > 0:
        print(f"  ✅ Dual-signal detected {total_dual_conflicts} conflicts with explanations")
        print(f"     (Provides debugging info: WHY is confidence low)")

    # Decision recommendation
    print(f"\n{'='*70}")
    print("🤔 RECOMMENDATION")
    print('='*70)

    if total_text_high_know_low > total_single_low * 0.2:
        print("✅ PROCEED WITH FULL DUAL-SIGNAL EXTRACTION")
        print("   Dual-signal catches significantly more errors than single-signal")
        print("   Worth the $5 cost for 172 episodes")
        print(f"   Estimated improvement: {total_text_high_know_low} additional errors caught per 10 episodes")
    elif total_dual_conflicts > 0:
        print("✅ CONSIDER DUAL-SIGNAL EXTRACTION")
        print("   Dual-signal provides better debugging info (conflict explanations)")
        print("   May not catch many MORE errors, but helps understand WHY errors occur")
    else:
        print("⚠️  SINGLE-SIGNAL MAY BE SUFFICIENT")
        print("   Dual-signal not showing significant advantages in this test")
        print("   Consider implementing validation passes instead")

    print(f"\n📁 Detailed results: {DUAL_SIGNAL_DIR}")
    print(f"📁 Comparison baseline: {SINGLE_SIGNAL_DIR}")


if __name__ == "__main__":
    main()
