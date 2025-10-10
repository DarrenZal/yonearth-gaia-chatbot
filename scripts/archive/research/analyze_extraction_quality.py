#!/usr/bin/env python3
"""
Analyze extraction quality and identify problematic patterns
"""

import json
import glob
from pathlib import Path
from collections import defaultdict, Counter
import re

DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_v2")

def analyze_relationships():
    """Comprehensive analysis of all extracted relationships"""

    all_relationships = []
    low_confidence_rels = []
    geographic_rels = []
    suspicious_patterns = []

    geo_terms = ['located_in', 'part_of', 'contains', 'in', 'based_in']

    print("📊 Analyzing 15,201 relationships from 172 episodes...\n")

    for json_file in sorted(DATA_DIR.glob("episode_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            for rel in data.get('relationships', []):
                all_relationships.append(rel)

                # Track low confidence
                if rel['relationship_confidence'] < 0.75:
                    low_confidence_rels.append({
                        'episode': data['episode'],
                        'rel': rel
                    })

                # Track geographic relationships
                if any(term in rel['relationship'].lower() for term in geo_terms):
                    geographic_rels.append({
                        'episode': data['episode'],
                        'rel': rel
                    })

                # Detect suspicious patterns
                source = rel['source'].lower()
                target = rel['target'].lower()

                # Pattern 1: Generic/vague entities in geographic relations
                if any(term in rel['relationship'].lower() for term in geo_terms):
                    if any(word in target for word in ['unknown', 'unspecified', 'unclear']):
                        suspicious_patterns.append({
                            'type': 'vague_location',
                            'episode': data['episode'],
                            'rel': rel,
                            'issue': f"Vague location: {target}"
                        })

                    # Pattern 2: Concepts as locations (biochar, Lebanese, etc)
                    abstract_terms = ['biochar', 'sustainability', 'concept', 'principle']
                    if any(term in target for term in abstract_terms):
                        suspicious_patterns.append({
                            'type': 'abstract_as_location',
                            'episode': data['episode'],
                            'rel': rel,
                            'issue': f"Abstract concept used as location: {target}"
                        })

                # Pattern 3: Relationship type mismatch
                if 'part_of' in rel['relationship'].lower():
                    # grandmother part_of Lebanese is wrong - should be "is"
                    if any(word in source for word in ['grandmother', 'mother', 'father', 'person']):
                        if any(word in target for word in ['lebanese', 'american', 'european']):
                            suspicious_patterns.append({
                                'type': 'wrong_relationship_type',
                                'episode': data['episode'],
                                'rel': rel,
                                'issue': f"Should be 'is' not 'part_of': {source} -> {target}"
                            })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print(f"✅ Analyzed {len(all_relationships)} relationships\n")

    # Report findings
    print("="*70)
    print("🔍 LOW CONFIDENCE RELATIONSHIPS (< 0.75)")
    print("="*70)
    print(f"Found {len(low_confidence_rels)} low-confidence relationships\n")

    # Group by confidence range
    conf_ranges = {
        '0.60-0.65': [],
        '0.65-0.70': [],
        '0.70-0.75': []
    }

    for item in low_confidence_rels:
        conf = item['rel']['relationship_confidence']
        if 0.60 <= conf < 0.65:
            conf_ranges['0.60-0.65'].append(item)
        elif 0.65 <= conf < 0.70:
            conf_ranges['0.65-0.70'].append(item)
        elif 0.70 <= conf < 0.75:
            conf_ranges['0.70-0.75'].append(item)

    for range_name, items in conf_ranges.items():
        if items:
            print(f"\n📉 Confidence {range_name} ({len(items)} relationships):")
            for item in items[:10]:  # Show first 10
                rel = item['rel']
                print(f"  Ep {item['episode']}: {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
                print(f"    Confidences: src={rel['source_confidence']:.2f}, rel={rel['relationship_confidence']:.2f}, tgt={rel['target_confidence']:.2f}")

    # Suspicious patterns
    print("\n" + "="*70)
    print("⚠️  SUSPICIOUS PATTERNS DETECTED")
    print("="*70)

    pattern_counts = Counter(p['type'] for p in suspicious_patterns)
    for pattern_type, count in pattern_counts.most_common():
        print(f"\n🚨 {pattern_type.upper().replace('_', ' ')}: {count} instances")

        examples = [p for p in suspicious_patterns if p['type'] == pattern_type][:5]
        for ex in examples:
            rel = ex['rel']
            print(f"  Ep {ex['episode']}: {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
            print(f"    Issue: {ex['issue']}")
            print(f"    Confidence: {rel['relationship_confidence']:.2f}")

    # Geographic relationship analysis
    print("\n" + "="*70)
    print("🗺️  GEOGRAPHIC RELATIONSHIP ANALYSIS")
    print("="*70)
    print(f"Total geographic relationships: {len(geographic_rels)}\n")

    # Find potentially reversed
    potentially_reversed = []
    for item in geographic_rels:
        rel = item['rel']
        # Low confidence + generic terms might indicate direction uncertainty
        if rel['relationship_confidence'] < 0.75:
            potentially_reversed.append(item)

    print(f"⚠️  Low-confidence geographic relationships: {len(potentially_reversed)}")
    for item in potentially_reversed[:15]:
        rel = item['rel']
        print(f"  Ep {item['episode']}: {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
        print(f"    Confidence: {rel['relationship_confidence']:.2f}")

    # Summary statistics
    print("\n" + "="*70)
    print("📈 SUMMARY STATISTICS")
    print("="*70)

    avg_source_conf = sum(r['source_confidence'] for r in all_relationships) / len(all_relationships)
    avg_rel_conf = sum(r['relationship_confidence'] for r in all_relationships) / len(all_relationships)
    avg_target_conf = sum(r['target_confidence'] for r in all_relationships) / len(all_relationships)

    print(f"Average Confidences:")
    print(f"  Source Entities: {avg_source_conf:.3f}")
    print(f"  Relationships: {avg_rel_conf:.3f}")
    print(f"  Target Entities: {avg_target_conf:.3f}")

    print(f"\nQuality Indicators:")
    print(f"  High confidence (>0.85): {sum(1 for r in all_relationships if r['relationship_confidence'] > 0.85)} ({sum(1 for r in all_relationships if r['relationship_confidence'] > 0.85)/len(all_relationships)*100:.1f}%)")
    print(f"  Medium confidence (0.75-0.85): {sum(1 for r in all_relationships if 0.75 <= r['relationship_confidence'] <= 0.85)} ({sum(1 for r in all_relationships if 0.75 <= r['relationship_confidence'] <= 0.85)/len(all_relationships)*100:.1f}%)")
    print(f"  Low confidence (<0.75): {len(low_confidence_rels)} ({len(low_confidence_rels)/len(all_relationships)*100:.1f}%)")

    print(f"\nSuspicious Patterns:")
    print(f"  Total flagged: {len(suspicious_patterns)} ({len(suspicious_patterns)/len(all_relationships)*100:.1f}%)")

    # Save detailed report
    report = {
        'total_relationships': len(all_relationships),
        'low_confidence': len(low_confidence_rels),
        'geographic_relationships': len(geographic_rels),
        'suspicious_patterns': len(suspicious_patterns),
        'pattern_breakdown': dict(pattern_counts),
        'avg_confidences': {
            'source': avg_source_conf,
            'relationship': avg_rel_conf,
            'target': avg_target_conf
        },
        'examples': {
            'low_confidence': [
                {
                    'episode': item['episode'],
                    'source': item['rel']['source'],
                    'relationship': item['rel']['relationship'],
                    'target': item['rel']['target'],
                    'confidence': item['rel']['relationship_confidence']
                }
                for item in low_confidence_rels[:50]
            ],
            'suspicious': [
                {
                    'type': p['type'],
                    'episode': p['episode'],
                    'source': p['rel']['source'],
                    'relationship': p['rel']['relationship'],
                    'target': p['rel']['target'],
                    'issue': p['issue'],
                    'confidence': p['rel']['relationship_confidence']
                }
                for p in suspicious_patterns[:50]
            ]
        }
    }

    report_path = DATA_DIR / "quality_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_path}")

    return report

if __name__ == "__main__":
    analyze_relationships()
