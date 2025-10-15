#!/usr/bin/env python3
"""
Compare V14 (baseline) vs V14.2 (improved) extraction quality on real book chunks.

This script extracts relationships from challenging chunks using both versions
and compares the results to determine which version produces better quality.

V14.2 improvements over V14:
- Pass 1: Entity resolution (demonstrative/personal pronouns)
- Pass 1: Semantic abstraction (rhetorical → factual)
- Pass 1: Complex parsing (multi-clause sentences)
- Pass 2: Reduced confidence penalties (V14.1 changes)
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load prompts for both versions
PROMPTS_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts")

V14_PASS1 = (PROMPTS_DIR / "pass1_extraction_v14.txt").read_text()
V14_PASS2 = (PROMPTS_DIR / "pass2_evaluation_v14.txt").read_text()

V14_2_PASS1 = (PROMPTS_DIR / "pass1_extraction_v14_2.txt").read_text()
V14_2_PASS2 = (PROMPTS_DIR / "pass2_evaluation_v14_2.txt").read_text()

# Pydantic models
class ExtractedRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    source_type: str
    target_type: str
    context: str
    page: int
    entity_specificity_score: float = None

class ExtractionResult(BaseModel):
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

class EvaluatedRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    source_type: str
    target_type: str
    context: str
    page: int
    p_true: float
    text_confidence: float
    knowledge_plausibility: float
    entity_specificity_score: float = None
    classification_flags: List[str] = Field(default_factory=list)

class EvaluationResult(BaseModel):
    relationships: List[EvaluatedRelationship] = Field(default_factory=list)


# Sample challenging chunks from the book
SAMPLE_CHUNKS = [
    {
        "id": "chunk_1_demonstrative",
        "page": 10,
        "text": """Soil stewardship encompasses a holistic approach to land management.
        This approach opens doors to sustainable farming practices that can regenerate degraded landscapes."""
    },
    {
        "id": "chunk_2_pronoun_resolution",
        "page": 15,
        "text": """By working with soil rather than against it, farmers can achieve remarkable results.
        They report improved yields, reduced input costs, and healthier ecosystems."""
    },
    {
        "id": "chunk_3_rhetorical_language",
        "page": 20,
        "text": """Soil is the answer to many of our environmental challenges.
        Through regenerative practices, we can mitigate climate change and restore biodiversity."""
    },
    {
        "id": "chunk_4_complex_parsing",
        "page": 25,
        "text": """By getting our hands in the living soil, we literally enhance our immune systems,
        reduce stress, and increase our production of beneficial hormones like serotonin."""
    },
    {
        "id": "chunk_5_bibliographic",
        "page": 6,
        "text": """Perry, Aaron William. Soil Stewardship Handbook: A Practical Guide to Regenerative Agriculture.
        Boulder: Y on Earth Community, 2018. Foreword by Joel Salatin."""
    },
    {
        "id": "chunk_6_mixed_content",
        "page": 12,
        "text": """Cover cropping enhances soil structure, prevents erosion, and sequesters carbon.
        These practices are essential for regenerative agriculture systems."""
    },
    {
        "id": "chunk_7_entity_specificity",
        "page": 18,
        "text": """The connection with the soil has been preserved in the Slovenian countryside.
        Their land remains productive and ecologically vibrant."""
    },
    {
        "id": "chunk_8_semantic_abstraction",
        "page": 22,
        "text": """Soil degradation threatens humanity's ability to feed itself.
        Regenerative practices can restore agricultural soils to productively vital states."""
    }
]


def extract_v14(chunk_text: str, page: int) -> Dict[str, Any]:
    """Run V14 baseline pipeline on a chunk"""
    # Pass 1
    response1 = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": V14_PASS1.format(text=chunk_text)}],
        response_format=ExtractionResult,
        temperature=0.3
    )
    pass1_rels = response1.choices[0].message.parsed.relationships

    if not pass1_rels:
        return {"pass1_count": 0, "pass2_count": 0, "final_count": 0, "relationships": []}

    # Pass 2
    rels_json = json.dumps([{
        'source': r.source,
        'relationship': r.relationship,
        'target': r.target,
        'source_type': r.source_type,
        'target_type': r.target_type,
        'context': r.context,
        'entity_specificity_score': r.entity_specificity_score
    } for r in pass1_rels], indent=2)

    response2 = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": V14_PASS2.format(
            batch_size=len(pass1_rels),
            relationships_json=rels_json
        )}],
        response_format=EvaluationResult,
        temperature=0.0
    )
    pass2_rels = response2.choices[0].message.parsed.relationships

    # Filter by p_true >= 0.5
    final_rels = [r for r in pass2_rels if r.p_true >= 0.5]

    return {
        "pass1_count": len(pass1_rels),
        "pass2_count": len(pass2_rels),
        "final_count": len(final_rels),
        "relationships": [{
            "source": r.source,
            "relationship": r.relationship,
            "target": r.target,
            "p_true": r.p_true,
            "flags": r.classification_flags
        } for r in final_rels]
    }


def extract_v14_2(chunk_text: str, page: int) -> Dict[str, Any]:
    """Run V14.2 full pipeline on a chunk"""
    # Pass 1 (with V14.2 improvements)
    response1 = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": V14_2_PASS1.format(text=chunk_text)}],
        response_format=ExtractionResult,
        temperature=0.3
    )
    pass1_rels = response1.choices[0].message.parsed.relationships

    if not pass1_rels:
        return {"pass1_count": 0, "pass2_count": 0, "final_count": 0, "relationships": []}

    # Pass 2 (with V14.1 penalties)
    rels_json = json.dumps([{
        'source': r.source,
        'relationship': r.relationship,
        'target': r.target,
        'source_type': r.source_type,
        'target_type': r.target_type,
        'context': r.context,
        'entity_specificity_score': r.entity_specificity_score
    } for r in pass1_rels], indent=2)

    response2 = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": V14_2_PASS2.format(
            batch_size=len(pass1_rels),
            relationships_json=rels_json
        )}],
        response_format=EvaluationResult,
        temperature=0.0
    )
    pass2_rels = response2.choices[0].message.parsed.relationships

    # Filter by p_true >= 0.5
    final_rels = [r for r in pass2_rels if r.p_true >= 0.5]

    return {
        "pass1_count": len(pass1_rels),
        "pass2_count": len(pass2_rels),
        "final_count": len(final_rels),
        "relationships": [{
            "source": r.source,
            "relationship": r.relationship,
            "target": r.target,
            "p_true": r.p_true,
            "flags": r.classification_flags
        } for r in final_rels]
    }


def main():
    print("="*80)
    print("🔬 V14 (BASELINE) vs V14.2 (IMPROVED) REAL EXTRACTION COMPARISON")
    print("="*80)
    print()
    print(f"Testing {len(SAMPLE_CHUNKS)} challenging chunks from Soil Stewardship Handbook")
    print()

    results = []

    for i, chunk in enumerate(SAMPLE_CHUNKS, 1):
        print(f"[{i}/{len(SAMPLE_CHUNKS)}] {chunk['id']}...")
        print(f"   Text: {chunk['text'][:80]}...")

        # Run V14 baseline
        print(f"   Running V14 (baseline)...")
        v14_result = extract_v14(chunk['text'], chunk['page'])

        # Run V14.2
        print(f"   Running V14.2 (improved)...")
        v14_2_result = extract_v14_2(chunk['text'], chunk['page'])

        # Compare
        v14_final = v14_result['final_count']
        v14_2_final = v14_2_result['final_count']

        if v14_2_final > v14_final:
            winner = "V14.2 ✅"
        elif v14_final > v14_2_final:
            winner = "V14 ✅"
        else:
            winner = "TIE"

        print(f"   V14: {v14_final} relationships | V14.2: {v14_2_final} relationships | Winner: {winner}")
        print()

        results.append({
            "chunk_id": chunk['id'],
            "page": chunk['page'],
            "text_preview": chunk['text'][:150],
            "v14": v14_result,
            "v14_2": v14_2_result,
            "winner": winner
        })

    # Summary
    print("="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print()

    v14_total = sum(r['v14']['final_count'] for r in results)
    v14_2_total = sum(r['v14_2']['final_count'] for r in results)

    v14_wins = sum(1 for r in results if "V14 ✅" == r['winner'])
    v14_2_wins = sum(1 for r in results if "V14.2 ✅" == r['winner'])
    ties = sum(1 for r in results if r['winner'] == "TIE")

    print(f"Total Relationships Extracted:")
    print(f"  V14: {v14_total} relationships")
    print(f"  V14.2: {v14_2_total} relationships")
    print()
    print(f"Chunk-by-Chunk Wins:")
    print(f"  V14 wins: {v14_wins}/{len(SAMPLE_CHUNKS)}")
    print(f"  V14.2 wins: {v14_2_wins}/{len(SAMPLE_CHUNKS)}")
    print(f"  Ties: {ties}/{len(SAMPLE_CHUNKS)}")
    print()

    # Detailed comparison
    print("="*80)
    print("📋 DETAILED CHUNK-BY-CHUNK COMPARISON")
    print("="*80)
    print()

    for r in results:
        print(f"## {r['chunk_id']} (Page {r['page']})")
        print(f"Text: {r['text_preview']}...")
        print()
        print(f"V14 Pipeline: {r['v14']['pass1_count']} → {r['v14']['pass2_count']} → {r['v14']['final_count']}")
        print(f"V14.2 Pipeline: {r['v14_2']['pass1_count']} → {r['v14_2']['pass2_count']} → {r['v14_2']['final_count']}")
        print()

        print(f"V14 Relationships ({r['v14']['final_count']}):")
        for rel in r['v14']['relationships']:
            flags = ", ".join(rel['flags']) if rel['flags'] else "NONE"
            print(f"  • ({rel['source']}) --[{rel['relationship']}]--> ({rel['target']}) [p={rel['p_true']:.2f}, flags={flags}]")
        if not r['v14']['relationships']:
            print(f"  (none)")
        print()

        print(f"V14.2 Relationships ({r['v14_2']['final_count']}):")
        for rel in r['v14_2']['relationships']:
            flags = ", ".join(rel['flags']) if rel['flags'] else "NONE"
            print(f"  • ({rel['source']}) --[{rel['relationship']}]--> ({rel['target']}) [p={rel['p_true']:.2f}, flags={flags}]")
        if not r['v14_2']['relationships']:
            print(f"  (none)")
        print()
        print(f"Winner: {r['winner']}")
        print()
        print("-"*80)
        print()

    # Recommendation
    print("="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    print()

    if v14_2_total > v14_total * 1.1:
        print("✅ V14.2 extracts significantly more relationships (>10% increase)")
        print("   RECOMMENDATION: Use V14.2 for full extraction")
    elif v14_total > v14_2_total * 1.1:
        print("✅ V14 extracts significantly more relationships (>10% increase)")
        print("   RECOMMENDATION: Use V14 for full extraction")
    else:
        print("⚖️  V14 and V14.2 produce similar volumes")
        print("   RECOMMENDATION: Review detailed relationships above for quality assessment")
        print("   Consider: Which version produces more accurate/useful relationships?")

    # Save results
    output_path = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/v14_vs_v14_2_comparison.json")
    with open(output_path, 'w') as f:
        json.dump({
            "summary": {
                "total_chunks": len(SAMPLE_CHUNKS),
                "v14_total_relationships": v14_total,
                "v14_2_total_relationships": v14_2_total,
                "v14_wins": v14_wins,
                "v14_2_wins": v14_2_wins,
                "ties": ties
            },
            "detailed_results": results
        }, f, indent=2)

    print()
    print(f"📁 Detailed results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
