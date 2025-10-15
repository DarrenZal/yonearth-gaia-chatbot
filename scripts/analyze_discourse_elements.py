#!/usr/bin/env python3
"""
Analyze V14.3.1 Pass 1 output to check if discourse elements are being extracted.

Checks for:
1. Attribution patterns (argues, claims, states, proposes)
2. Direct aspirational patterns (we can, we should, we must)
3. Comparison with V14.3 to see what changed
"""

import json
import re
from pathlib import Path
from collections import Counter

def analyze_discourse_patterns(relationships):
    """Analyze relationships for discourse and aspirational patterns"""

    discourse_patterns = {
        'argues': [],
        'claims': [],
        'states': [],
        'proposes': [],
        'suggests': [],
        'demonstrates': [],
        'shows': [],
        'found': [],
        'authored': [],
        'written_by': []
    }

    aspirational_patterns = {
        'we_can': [],
        'we_should': [],
        'we_must': [],
        'we_will': [],
        'can_help': [],
        'should_be': []
    }

    for rel in relationships:
        # Check relationship predicate for discourse markers
        rel_text = rel.get('relationship', '').lower()
        context = rel.get('context', '').lower()
        source = rel.get('source', '').lower()
        target = rel.get('target', '').lower()

        # Discourse patterns
        for pattern in discourse_patterns.keys():
            if pattern in rel_text or pattern in context[:100]:
                discourse_patterns[pattern].append({
                    'source': rel.get('source'),
                    'relationship': rel.get('relationship'),
                    'target': rel.get('target'),
                    'context_snippet': context[:150]
                })

        # Aspirational patterns (should be rare/absent)
        full_text = f"{source} {rel_text} {target} {context}".lower()
        if 'we can' in full_text or 'humanity can' in full_text:
            aspirational_patterns['we_can'].append(rel)
        if 'we should' in full_text or 'humanity should' in full_text:
            aspirational_patterns['we_should'].append(rel)
        if 'we must' in full_text or 'humanity must' in full_text:
            aspirational_patterns['we_must'].append(rel)
        if 'we will' in full_text or 'humanity will' in full_text:
            aspirational_patterns['we_will'].append(rel)

    return discourse_patterns, aspirational_patterns


def main():
    # Find V14.3.1 Pass 1 checkpoint
    output_dir = Path("kg_extraction_playbook/output/v14_3_1")
    checkpoints = list(output_dir.glob("*_pass1_checkpoint.json"))

    if not checkpoints:
        print("❌ No V14.3.1 Pass 1 checkpoint found")
        return

    checkpoint = checkpoints[0]
    print(f"📖 Analyzing: {checkpoint.name}")
    print()

    with open(checkpoint) as f:
        relationships = json.load(f)

    print(f"Total candidates: {len(relationships)}")
    print()

    # Analyze patterns
    discourse, aspirational = analyze_discourse_patterns(relationships)

    # Report discourse elements
    print("=" * 80)
    print("✅ DISCOURSE ELEMENTS (Should be present)")
    print("=" * 80)

    total_discourse = 0
    for pattern, examples in discourse.items():
        count = len(examples)
        total_discourse += count
        if count > 0:
            print(f"\n{pattern.upper()}: {count} instances")
            for ex in examples[:3]:  # Show first 3
                print(f"  - {ex['source']} → {ex['relationship']} → {ex['target']}")
                print(f"    Context: {ex['context_snippet'][:100]}...")

    print()
    print(f"TOTAL DISCOURSE ELEMENTS: {total_discourse}")
    print()

    # Report aspirational patterns
    print("=" * 80)
    print("❌ ASPIRATIONAL PATTERNS (Should be rare/absent)")
    print("=" * 80)

    total_aspirational = 0
    for pattern, examples in aspirational.items():
        count = len(examples)
        total_aspirational += count
        if count > 0:
            print(f"\n{pattern.upper()}: {count} instances")
            for ex in examples[:2]:  # Show first 2
                print(f"  - {ex.get('source')} → {ex.get('relationship')} → {ex.get('target')}")

    print()
    print(f"TOTAL ASPIRATIONAL: {total_aspirational}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total candidates: {len(relationships)}")
    print(f"Discourse elements: {total_discourse} ({100*total_discourse/len(relationships):.1f}%)")
    print(f"Aspirational patterns: {total_aspirational} ({100*total_aspirational/len(relationships):.1f}%)")
    print()

    if total_discourse > 20:
        print("✅ GOOD: Extracting discourse elements properly")
    else:
        print("⚠️  WARNING: Low discourse element count - may be filtering too aggressively")

    if total_aspirational < 10:
        print("✅ GOOD: Successfully filtering aspirational content")
    else:
        print("⚠️  WARNING: Still extracting aspirational content")


if __name__ == "__main__":
    main()
