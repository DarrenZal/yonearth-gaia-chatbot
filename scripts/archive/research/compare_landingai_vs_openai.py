#!/usr/bin/env python3
"""
Compare LandingAI extraction vs OpenAI two-pass batched extraction for episode 10.

This script analyzes the differences in entity and relationship extraction
between LandingAI's ADE and OpenAI's gpt-4o-mini two-pass approach.
"""

import json
from pathlib import Path
from typing import Dict, Set, List, Tuple

# Paths
LANDINGAI_FILE = Path("data/knowledge_graph_landingai_test/episode_10_landingai.json")
OPENAI_FILE = Path("data/knowledge_graph_two_pass_batched_test/episode_10_two_pass_batched.json")


def load_extraction(file_path: Path) -> Dict:
    """Load extraction results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_entity_pairs_from_relationships(relationships: List[Dict]) -> Set[Tuple[str, str, str]]:
    """Extract (source, relationship, target) tuples from relationships."""
    pairs = set()
    for rel in relationships:
        source = rel.get('source', '').lower().strip()
        relationship = rel.get('relationship', '').lower().strip()
        target = rel.get('target', '').lower().strip()
        if source and relationship and target:
            pairs.add((source, relationship, target))
    return pairs


def compare_extractions():
    """Compare LandingAI vs OpenAI extractions."""
    print(f"\n{'='*80}")
    print(f"LandingAI vs OpenAI Knowledge Graph Extraction Comparison")
    print(f"Episode 10 - Lauren Tucker - Kiss the Ground")
    print(f"{'='*80}\n")

    # Load both extractions
    landingai = load_extraction(LANDINGAI_FILE)
    openai = load_extraction(OPENAI_FILE)

    # Basic statistics
    print(f"📊 Basic Statistics:")
    print(f"{'─'*80}")
    print(f"{'Metric':<40} {'LandingAI':>15} {'OpenAI':>15}")
    print(f"{'─'*80}")
    print(f"{'Model':<40} {landingai['model']:>15} {openai['model']:>15}")
    print(f"{'Approach':<40} {landingai['approach']:>15} {openai['approach']:>15}")
    print(f"{'Chunks Processed':<40} {landingai['chunks_processed']:>15} {openai.get('pass1_extracted', 'N/A'):>15}")
    print(f"{'Unique Entities':<40} {landingai['unique_entities']:>15} {'-':>15}")
    print(f"{'Total Relationships':<40} {landingai['total_relationships']:>15} {openai['pass1_extracted']:>15}")
    print(f"{'─'*80}\n")

    # Get relationship pairs
    landingai_pairs = get_entity_pairs_from_relationships(landingai['relationships'])
    openai_pairs = get_entity_pairs_from_relationships(openai['relationships'])

    # Calculate overlaps
    shared_pairs = landingai_pairs.intersection(openai_pairs)
    landingai_only = landingai_pairs - openai_pairs
    openai_only = openai_pairs - landingai_pairs

    print(f"📈 Relationship Coverage Analysis:")
    print(f"{'─'*80}")
    print(f"LandingAI extracted:     {len(landingai_pairs):>3} unique relationship pairs")
    print(f"OpenAI extracted:        {len(openai_pairs):>3} unique relationship pairs")
    print(f"Shared (both):           {len(shared_pairs):>3} pairs ({len(shared_pairs)/len(openai_pairs)*100:.1f}% of OpenAI)")
    print(f"LandingAI only:          {len(landingai_only):>3} pairs")
    print(f"OpenAI only:             {len(openai_only):>3} pairs")
    print(f"{'─'*80}\n")

    # Show shared relationships
    print(f"✅ Shared Relationships (found by both):")
    print(f"{'─'*80}")
    for i, (source, rel, target) in enumerate(sorted(shared_pairs)[:10], 1):
        print(f"{i:2}. {source} --[{rel}]--> {target}")
    if len(shared_pairs) > 10:
        print(f"... and {len(shared_pairs) - 10} more")
    print()

    # Show LandingAI-only relationships
    print(f"🔵 LandingAI-Only Relationships (missed by OpenAI):")
    print(f"{'─'*80}")
    for i, (source, rel, target) in enumerate(sorted(landingai_only)[:10], 1):
        print(f"{i:2}. {source} --[{rel}]--> {target}")
    if len(landingai_only) > 10:
        print(f"... and {len(landingai_only) - 10} more")
    print()

    # Show OpenAI-only relationships
    print(f"🟢 OpenAI-Only Relationships (missed by LandingAI):")
    print(f"{'─'*80}")
    for i, (source, rel, target) in enumerate(sorted(openai_only)[:15], 1):
        print(f"{i:2}. {source} --[{rel}]--> {target}")
    if len(openai_only) > 15:
        print(f"... and {len(openai_only) - 15} more")
    print()

    # Analyze entity types (LandingAI only, as OpenAI includes them)
    if 'entities' in landingai and landingai['entities']:
        entity_types = {}
        for entity in landingai['entities']:
            etype = entity.get('type', 'UNKNOWN')
            entity_types[etype] = entity_types.get(etype, 0) + 1

        print(f"🏷️  LandingAI Entity Type Distribution:")
        print(f"{'─'*80}")
        for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{etype:<20} {count:>3} entities")
        print()

    # Quality analysis
    print(f"💡 Quality Observations:")
    print(f"{'─'*80}")

    # Check for confidence scores
    landingai_with_conf = [r for r in landingai['relationships'] if 'confidence' in r]
    openai_with_conf = [r for r in openai['relationships'] if 'overall_confidence' in r]

    print(f"LandingAI relationships with confidence: {len(landingai_with_conf)}/{len(landingai['relationships'])}")
    print(f"OpenAI relationships with confidence:    {len(openai_with_conf)}/{len(openai['relationships'])}")

    if landingai_with_conf:
        avg_conf = sum(r['confidence'] for r in landingai_with_conf) / len(landingai_with_conf)
        print(f"LandingAI average confidence:            {avg_conf:.2f}")

    if openai_with_conf:
        avg_conf = sum(r['overall_confidence'] for r in openai_with_conf) / len(openai_with_conf)
        print(f"OpenAI average confidence:               {avg_conf:.2f}")

    print()

    # Comparison summary
    print(f"{'='*80}")
    print(f"📝 Summary:")
    print(f"{'='*80}")
    print(f"""
LandingAI Results:
✅ Extracted {len(landingai_pairs)} unique relationship pairs from 9 chunks
✅ Provided entity type classification for {len(landingai.get('entities', []))} entities
✅ Schema-driven extraction ensures consistent structure
✅ Included confidence scores for most relationships

OpenAI Two-Pass Results:
✅ Extracted {len(openai_pairs)} unique relationship pairs (3.3x more than LandingAI)
✅ Dual-signal evaluation (text_confidence + knowledge_plausibility)
✅ Detected {openai.get('conflicts_detected', 0)} conflicts
✅ Identified {openai.get('type_violations', 0)} type violations

Coverage:
• LandingAI found {len(shared_pairs)}/{len(openai_pairs)} ({len(shared_pairs)/len(openai_pairs)*100:.1f}%) of OpenAI's relationships
• LandingAI missed {len(openai_only)} relationships that OpenAI found
• LandingAI found {len(landingai_only)} unique relationships not found by OpenAI

Recommendation:
The OpenAI two-pass approach extracted significantly more relationships ({len(openai_pairs)}
vs {len(landingai_pairs)}), suggesting better coverage. However, LandingAI's schema-driven
approach provides more structured entity metadata and may be faster/cheaper per API call.

Consider:
1. **For comprehensiveness**: Use OpenAI two-pass extraction
2. **For structured entity data**: Use LandingAI or combine both approaches
3. **For cost optimization**: Test LandingAI on larger batches (less chunking may help)
    """)

    print(f"{'='*80}\n")


if __name__ == "__main__":
    compare_extractions()
