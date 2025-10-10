#!/usr/bin/env python3
"""
Compare GPT-5-nano vs GPT-5-mini dual-signal extraction results.

This script analyzes:
1. Extraction coverage (total relationships extracted)
2. Conflict detection rates (dual-signal disagreements)
3. Speed and efficiency
4. Per-episode breakdown
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_episode_results(results_dir: Path, model_name: str) -> Dict[int, Dict]:
    """Load all episode extraction results from a directory."""
    episode_data = {}

    # Try different filename patterns
    patterns = [
        f"episode_*_{model_name}.json",
        "episode_*_extraction.json",
        "episode_*.json"
    ]

    json_files = []
    for pattern in patterns:
        files = list(results_dir.glob(pattern))
        if files:
            json_files = files
            break

    if not json_files:
        return {}

    for json_file in json_files:
        # Skip summary files
        if "summary" in json_file.stem:
            continue

        # Extract episode number from filename
        parts = json_file.stem.split("_")
        episode_num = int(parts[1])

        with open(json_file) as f:
            data = json.load(f)

        episode_data[episode_num] = {
            "relationships": data.get("relationships", []),
            "entity_count": len(data.get("entities", [])),
            "relationship_count": len(data.get("relationships", [])),
        }

    return episode_data

def count_conflicts(relationships: List[Dict]) -> Tuple[int, int]:
    """Count conflicts and type violations in relationships."""
    conflicts = 0
    type_violations = 0

    for rel in relationships:
        # Check for dual-signal conflicts
        text_conf = rel.get("text_confidence", 1.0)
        knowledge_plaus = rel.get("knowledge_plausibility", 1.0)

        # Conflict: high text confidence but low knowledge plausibility
        if text_conf >= 0.8 and knowledge_plaus < 0.5:
            conflicts += 1

        # Type violation: relationship type doesn't match entity types
        if rel.get("type_violation", False):
            type_violations += 1

    return conflicts, type_violations

def analyze_model_results(results_dir: Path, model_name: str) -> Dict:
    """Analyze results for a single model."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name} results...")
    print(f"{'='*60}")

    episode_data = load_episode_results(results_dir, model_name)

    if not episode_data:
        print(f"❌ No results found in {results_dir}")
        return {}

    total_relationships = 0
    total_conflicts = 0
    total_type_violations = 0

    episode_stats = []

    for episode_num in sorted(episode_data.keys()):
        data = episode_data[episode_num]
        relationships = data["relationships"]
        rel_count = len(relationships)
        conflicts, type_viols = count_conflicts(relationships)

        total_relationships += rel_count
        total_conflicts += conflicts
        total_type_violations += type_viols

        episode_stats.append({
            "episode": episode_num,
            "relationships": rel_count,
            "conflicts": conflicts,
            "type_violations": type_viols,
            "conflict_rate": (conflicts / rel_count * 100) if rel_count > 0 else 0
        })

        print(f"Episode {episode_num:3d}: {rel_count:3d} relationships, "
              f"{conflicts:2d} conflicts ({conflicts/rel_count*100:5.1f}%), "
              f"{type_viols} type violations")

    avg_relationships = total_relationships / len(episode_data) if episode_data else 0
    conflict_rate = (total_conflicts / total_relationships * 100) if total_relationships > 0 else 0

    print(f"\n📊 {model_name} Summary:")
    print(f"   Episodes: {len(episode_data)}")
    print(f"   Total relationships: {total_relationships}")
    print(f"   Average per episode: {avg_relationships:.1f}")
    print(f"   Total conflicts: {total_conflicts} ({conflict_rate:.1f}%)")
    print(f"   Type violations: {total_type_violations}")

    return {
        "model": model_name,
        "episodes": len(episode_data),
        "total_relationships": total_relationships,
        "avg_per_episode": avg_relationships,
        "total_conflicts": total_conflicts,
        "conflict_rate": conflict_rate,
        "type_violations": total_type_violations,
        "episode_stats": episode_stats
    }

def compare_models(nano_stats: Dict, mini_stats: Dict):
    """Generate comparison between nano and mini models."""
    print(f"\n{'='*60}")
    print("🔬 GPT-5 MODEL COMPARISON")
    print(f"{'='*60}\n")

    # Coverage comparison
    print("📈 EXTRACTION COVERAGE:")
    print(f"   gpt-5-nano: {nano_stats['total_relationships']:4d} relationships "
          f"({nano_stats['avg_per_episode']:.1f} avg/episode)")
    print(f"   gpt-5-mini: {mini_stats['total_relationships']:4d} relationships "
          f"({mini_stats['avg_per_episode']:.1f} avg/episode)")

    coverage_diff = mini_stats['total_relationships'] - nano_stats['total_relationships']
    coverage_pct = (coverage_diff / nano_stats['total_relationships'] * 100) if nano_stats['total_relationships'] > 0 else 0

    if coverage_diff > 0:
        print(f"   ✅ gpt-5-mini extracts {coverage_diff} MORE relationships (+{coverage_pct:.1f}%)")
    else:
        print(f"   ⚠️  gpt-5-nano extracts {abs(coverage_diff)} MORE relationships (+{abs(coverage_pct):.1f}%)")

    # Conflict detection comparison
    print(f"\n⚠️  CONFLICT DETECTION (dual-signal disagreements):")
    print(f"   gpt-5-nano: {nano_stats['total_conflicts']:3d} conflicts ({nano_stats['conflict_rate']:.1f}%)")
    print(f"   gpt-5-mini: {mini_stats['total_conflicts']:3d} conflicts ({mini_stats['conflict_rate']:.1f}%)")

    if mini_stats['conflict_rate'] > nano_stats['conflict_rate']:
        print(f"   ✅ gpt-5-mini detects MORE conflicts ({mini_stats['conflict_rate']:.1f}% vs {nano_stats['conflict_rate']:.1f}%)")
    else:
        print(f"   ⚠️  gpt-5-nano detects MORE conflicts ({nano_stats['conflict_rate']:.1f}% vs {mini_stats['conflict_rate']:.1f}%)")

    # Per-episode comparison
    print(f"\n📊 PER-EPISODE COMPARISON:")
    print(f"{'Episode':>8} {'nano':>8} {'mini':>8} {'Diff':>8} {'nano conf':>10} {'mini conf':>10}")
    print(f"{'-'*60}")

    for nano_ep, mini_ep in zip(nano_stats['episode_stats'], mini_stats['episode_stats']):
        ep = nano_ep['episode']
        nano_rels = nano_ep['relationships']
        mini_rels = mini_ep['relationships']
        diff = mini_rels - nano_rels
        nano_conf = nano_ep['conflicts']
        mini_conf = mini_ep['conflicts']

        diff_sign = "+" if diff > 0 else ""
        print(f"{ep:8d} {nano_rels:8d} {mini_rels:8d} {diff_sign}{diff:7d} "
              f"{nano_conf:6d} ({nano_ep['conflict_rate']:4.1f}%) "
              f"{mini_conf:6d} ({mini_ep['conflict_rate']:4.1f}%)")

    # Winner determination
    print(f"\n{'='*60}")
    print("🏆 VERDICT:")
    print(f"{'='*60}")

    if coverage_pct > 100:
        print("✅ gpt-5-mini is the CLEAR WINNER!")
        print(f"   • Extracts 2x MORE relationships (+{coverage_pct:.1f}%)")
        print(f"   • {mini_stats['avg_per_episode']:.1f} avg/episode vs {nano_stats['avg_per_episode']:.1f}")
        print(f"   • Conflict rate similar ({mini_stats['conflict_rate']:.1f}% vs {nano_stats['conflict_rate']:.1f}%)")
        print(f"   • MUCH better coverage for comprehensive knowledge graphs")
        print(f"\n💡 RECOMMENDATION: Use gpt-5-mini for full 172-episode extraction")
    elif coverage_pct > 50 and mini_stats['conflict_rate'] >= nano_stats['conflict_rate']:
        print("✅ gpt-5-mini is the WINNER!")
        print(f"   • Extracts {coverage_pct:.1f}% MORE relationships")
        print(f"   • Maintains good conflict detection ({mini_stats['conflict_rate']:.1f}%)")
        print(f"   • Better coverage for comprehensive knowledge graphs")
    elif coverage_pct < 0:
        print("⚠️  gpt-5-nano extracts MORE but may not be better:")
        print(f"   • Extracts {abs(coverage_pct):.1f}% MORE relationships")
        print(f"   • But lower conflict rate ({nano_stats['conflict_rate']:.1f}% vs {mini_stats['conflict_rate']:.1f}%)")
        print(f"   • May be over-extracting without enough validation")
    else:
        print("⚖️  MIXED RESULTS - both models have strengths:")
        print(f"   • gpt-5-mini: {coverage_pct:+.1f}% coverage difference")
        print(f"   • gpt-5-nano: {nano_stats['conflict_rate']:.1f}% conflict rate")
        print(f"   • gpt-5-mini: {mini_stats['conflict_rate']:.1f}% conflict rate")

def main():
    nano_dir = project_root / "data" / "knowledge_graph_gpt5_nano_test"
    mini_dir = project_root / "data" / "knowledge_graph_gpt5_mini_test"

    # Analyze both models
    nano_stats = analyze_model_results(nano_dir, "gpt-5-nano")
    mini_stats = analyze_model_results(mini_dir, "gpt-5-mini")

    if nano_stats and mini_stats:
        compare_models(nano_stats, mini_stats)
    else:
        print("\n❌ Cannot compare - missing results for one or both models")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
