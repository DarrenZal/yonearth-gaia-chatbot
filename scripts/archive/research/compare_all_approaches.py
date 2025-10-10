#!/usr/bin/env python3
"""
Compare All Extraction Approaches

Compares:
1. Single-signal (v2) - baseline
2. Dual-signal gpt-4o-mini - original dual-signal test
3. Dual-signal gpt-5-mini - smarter model test
4. Two-pass gpt-4o-mini - Option B test

Key metrics:
- Entity pair coverage vs baseline
- Conflict detection
- Relationship counts
- Quality indicators
"""

import json
from pathlib import Path

DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
SINGLE_DIR = DATA_DIR / "knowledge_graph_v2"
GPT4O_DUAL_DIR = DATA_DIR / "knowledge_graph_dual_signal_test"
GPT5_MINI_DIR = DATA_DIR / "knowledge_graph_gpt5_mini_test"
GPT5_NANO_DIR = DATA_DIR / "knowledge_graph_gpt5_nano_test"
TWO_PASS_DIR = DATA_DIR / "knowledge_graph_two_pass_batched_test"


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


def load_episode(directory, ep_num, suffix):
    """Load episode extraction from directory"""
    if suffix == "single":
        path = directory / f"episode_{ep_num}_extraction.json"
    else:
        path = directory / f"episode_{ep_num}_{suffix}.json"

    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    return data.get('relationships', [])


def analyze_episode(ep_num):
    """Compare all approaches for one episode"""
    print(f"\n{'='*80}")
    print(f"EPISODE {ep_num} COMPARISON")
    print('='*80)

    # Load data
    single_rels = load_episode(SINGLE_DIR, ep_num, "single")
    gpt4o_dual_rels = load_episode(GPT4O_DUAL_DIR, ep_num, "dual_signal")
    gpt5_mini_rels = load_episode(GPT5_MINI_DIR, ep_num, "gpt5_mini")
    gpt5_nano_rels = load_episode(GPT5_NANO_DIR, ep_num, "gpt5_nano")
    two_pass_rels = load_episode(TWO_PASS_DIR, ep_num, "two_pass_batched")  # Updated for batched test
    if not two_pass_rels:
        # Try alternative naming pattern
        two_pass_rels = load_episode(TWO_PASS_DIR, ep_num, "extraction")

    if not single_rels:
        print("⚠️  Baseline (single-signal) data missing")
        return None

    # Get entity pairs
    single_pairs = get_entity_pairs(single_rels)

    results = {
        'episode': ep_num,
        'single': {
            'rels': len(single_rels),
            'pairs': len(single_pairs)
        }
    }

    approaches = []

    if gpt4o_dual_rels:
        gpt4o_pairs = get_entity_pairs(gpt4o_dual_rels)
        gpt4o_coverage = len(gpt4o_pairs & single_pairs) / len(single_pairs)
        gpt4o_conflicts = sum(1 for r in gpt4o_dual_rels if r.get('signals_conflict', False))
        approaches.append(('gpt-4o-mini (dual)', gpt4o_dual_rels, gpt4o_pairs, gpt4o_coverage, gpt4o_conflicts))
        results['gpt4o_dual'] = {
            'rels': len(gpt4o_dual_rels),
            'pairs': len(gpt4o_pairs),
            'coverage': gpt4o_coverage,
            'conflicts': gpt4o_conflicts
        }

    if gpt5_mini_rels:
        gpt5_mini_pairs = get_entity_pairs(gpt5_mini_rels)
        gpt5_mini_coverage = len(gpt5_mini_pairs & single_pairs) / len(single_pairs)
        gpt5_mini_conflicts = sum(1 for r in gpt5_mini_rels if r.get('signals_conflict', False))
        approaches.append(('gpt-5-mini (dual)', gpt5_mini_rels, gpt5_mini_pairs, gpt5_mini_coverage, gpt5_mini_conflicts))
        results['gpt5_mini'] = {
            'rels': len(gpt5_mini_rels),
            'pairs': len(gpt5_mini_pairs),
            'coverage': gpt5_mini_coverage,
            'conflicts': gpt5_mini_conflicts
        }

    if gpt5_nano_rels:
        gpt5_nano_pairs = get_entity_pairs(gpt5_nano_rels)
        gpt5_nano_coverage = len(gpt5_nano_pairs & single_pairs) / len(single_pairs)
        gpt5_nano_conflicts = sum(1 for r in gpt5_nano_rels if r.get('signals_conflict', False))
        approaches.append(('gpt-5-nano (dual)', gpt5_nano_rels, gpt5_nano_pairs, gpt5_nano_coverage, gpt5_nano_conflicts))
        results['gpt5_nano'] = {
            'rels': len(gpt5_nano_rels),
            'pairs': len(gpt5_nano_pairs),
            'coverage': gpt5_nano_coverage,
            'conflicts': gpt5_nano_conflicts
        }

    if two_pass_rels:
        two_pass_pairs = get_entity_pairs(two_pass_rels)
        two_pass_coverage = len(two_pass_pairs & single_pairs) / len(single_pairs)
        two_pass_conflicts = sum(1 for r in two_pass_rels if r.get('signals_conflict', False))
        approaches.append(('two-pass (gpt-4o-mini)', two_pass_rels, two_pass_pairs, two_pass_coverage, two_pass_conflicts))
        results['two_pass'] = {
            'rels': len(two_pass_rels),
            'pairs': len(two_pass_pairs),
            'coverage': two_pass_coverage,
            'conflicts': two_pass_conflicts
        }

    # Print comparison
    print(f"\n📊 RELATIONSHIP COUNTS:")
    print(f"  Single-signal (baseline): {len(single_rels)} relationships, {len(single_pairs)} entity pairs")
    for name, rels, pairs, _, _ in approaches:
        print(f"  {name:25s}: {len(rels):3d} relationships, {len(pairs):3d} entity pairs")

    print(f"\n🎯 ENTITY PAIR COVERAGE (vs single-signal baseline):")
    for name, _, pairs, coverage, _ in approaches:
        match_count = len(pairs & single_pairs)
        print(f"  {name:25s}: {coverage*100:5.1f}% ({match_count}/{len(single_pairs)})")

    print(f"\n🚨 ERROR DETECTION:")
    for name, _, _, _, conflicts in approaches:
        print(f"  {name:25s}: {conflicts} conflicts detected")

    # Find best approach
    if approaches:
        best = max(approaches, key=lambda x: x[3])  # Sort by coverage
        print(f"\n🏆 BEST COVERAGE: {best[0]} with {best[3]*100:.1f}%")

    return results


def main():
    """Compare all approaches across all test episodes"""
    print("="*80)
    print("🔬 COMPREHENSIVE APPROACH COMPARISON")
    print("="*80)
    print("\nComparing:")
    print("  1. Single-signal (v2) - baseline")
    print("  2. Dual-signal gpt-4o-mini - original test")
    print("  3. Dual-signal gpt-5-mini - smarter model")
    print("  4. Dual-signal gpt-5-nano - fastest model")
    print("  5. Two-pass gpt-4o-mini - Option B")
    print("="*80)

    test_episodes = [10, 39, 50, 75, 100, 112, 120, 122, 150, 165]

    all_results = []
    for ep in test_episodes:
        result = analyze_episode(ep)
        if result:
            all_results.append(result)

    # Overall summary
    print(f"\n{'='*80}")
    print("📊 OVERALL SUMMARY")
    print('='*80)

    # Calculate totals
    total_single_pairs = sum(r['single']['pairs'] for r in all_results)

    summary = {
        'single': {'pairs': total_single_pairs, 'rels': sum(r['single']['rels'] for r in all_results)}
    }

    approaches_summary = []

    # gpt-4o-mini dual
    if all(r.get('gpt4o_dual') for r in all_results):
        total_pairs = sum(r['gpt4o_dual']['pairs'] for r in all_results)
        total_conflicts = sum(r['gpt4o_dual']['conflicts'] for r in all_results)
        coverage = total_pairs / total_single_pairs if total_single_pairs else 0
        approaches_summary.append(('gpt-4o-mini (dual)', total_pairs, coverage, total_conflicts))
        summary['gpt4o_dual'] = {'pairs': total_pairs, 'coverage': coverage, 'conflicts': total_conflicts}

    # gpt-5-mini dual
    if all(r.get('gpt5_mini') for r in all_results):
        total_pairs = sum(r['gpt5_mini']['pairs'] for r in all_results)
        total_conflicts = sum(r['gpt5_mini']['conflicts'] for r in all_results)
        coverage = total_pairs / total_single_pairs if total_single_pairs else 0
        approaches_summary.append(('gpt-5-mini (dual)', total_pairs, coverage, total_conflicts))
        summary['gpt5_mini'] = {'pairs': total_pairs, 'coverage': coverage, 'conflicts': total_conflicts}

    # gpt-5-nano dual
    if all(r.get('gpt5_nano') for r in all_results):
        total_pairs = sum(r['gpt5_nano']['pairs'] for r in all_results)
        total_conflicts = sum(r['gpt5_nano']['conflicts'] for r in all_results)
        coverage = total_pairs / total_single_pairs if total_single_pairs else 0
        approaches_summary.append(('gpt-5-nano (dual)', total_pairs, coverage, total_conflicts))
        summary['gpt5_nano'] = {'pairs': total_pairs, 'coverage': coverage, 'conflicts': total_conflicts}

    # two-pass
    if all(r.get('two_pass') for r in all_results):
        total_pairs = sum(r['two_pass']['pairs'] for r in all_results)
        total_conflicts = sum(r['two_pass']['conflicts'] for r in all_results)
        coverage = total_pairs / total_single_pairs if total_single_pairs else 0
        approaches_summary.append(('two-pass (gpt-4o-mini)', total_pairs, coverage, total_conflicts))
        summary['two_pass'] = {'pairs': total_pairs, 'coverage': coverage, 'conflicts': total_conflicts}

    print(f"\nAcross {len(all_results)} episodes:")
    print(f"\n📊 ENTITY PAIR COVERAGE:")
    print(f"  Single-signal baseline: {total_single_pairs} pairs")
    for name, pairs, coverage, _ in approaches_summary:
        print(f"  {name:25s}: {pairs:4d} pairs ({coverage*100:5.1f}% coverage)")

    print(f"\n🚨 ERROR DETECTION:")
    for name, _, _, conflicts in approaches_summary:
        print(f"  {name:25s}: {conflicts} conflicts detected")

    print(f"\n{'='*80}")
    print("🎯 FINAL VERDICT:")
    print('='*80)

    if approaches_summary:
        # Find best coverage
        best = max(approaches_summary, key=lambda x: x[2])
        print(f"\n✅ BEST COVERAGE: {best[0]}")
        print(f"   Coverage: {best[2]*100:.1f}%")
        print(f"   Conflicts detected: {best[3]}")

        # Check if any approach achieves good coverage
        if best[2] >= 0.7:
            print(f"\n   🎉 Excellent coverage (≥70%)!")
            print(f"   Recommendation: Use {best[0]} for full extraction")
        elif best[2] >= 0.5:
            print(f"\n   ✅ Good coverage (≥50%)")
            print(f"   Recommendation: Use {best[0]} for full extraction")
        else:
            print(f"\n   ⚠️  Coverage still below 50%")
            print(f"   All dual-signal approaches struggle with coverage")
            print(f"   Recommendation: Consider Option C (validate existing v2) or investigate further")

    # Save summary
    output_path = DATA_DIR / "extraction_comparison_summary.json"
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': '2025-10-10',
            'test_episodes': test_episodes,
            'summary': summary,
            'detailed_results': all_results
        }, f, indent=2)

    print(f"\n📁 Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
