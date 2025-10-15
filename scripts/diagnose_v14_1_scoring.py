#!/usr/bin/env python3
"""
Diagnose V14.1 scoring behavior to understand why penalty reductions had no effect.

This script examines:
1. Actual p_true values assigned in Pass 2
2. Whether confidence penalties are being applied
3. Which relationships fall near the 0.5 threshold
4. What filtering mechanism removes relationships (p_true vs postprocessing)
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "kg_extraction_playbook" / "prompts"
ANALYSIS_DIR = PROJECT_ROOT / "kg_extraction_playbook" / "analysis_reports"
DATA_DIR = PROJECT_ROOT / "data" / "knowledge_graph_v3_2_2"

PASS1_PROMPT_PATH = PROMPTS_DIR / "pass1_extraction_v14_1.txt"
PASS2_PROMPT_PATH = PROMPTS_DIR / "pass2_evaluation_v14_1.txt"

# Initialize OpenAI client
client = AsyncOpenAI()

# Pydantic models for structured outputs
class ExtractedRelationship(BaseModel):
    """Pass 1 extraction output"""
    source: str = Field(..., description="Source entity")
    predicate: str = Field(..., description="Relationship predicate")
    target: str = Field(..., description="Target entity")
    context: str = Field(..., description="Supporting context from text")

class ExtractedRelationships(BaseModel):
    """Container for Pass 1 output"""
    relationships: List[ExtractedRelationship]

class EvaluatedRelationship(BaseModel):
    """Pass 2 evaluation output"""
    source: str
    predicate: str
    target: str
    context: str
    p_true: float = Field(..., ge=0.0, le=1.0, description="Probability this relationship is factually true")
    classification_flags: List[str] = Field(default_factory=list, description="Applied classification flags")
    reasoning: str = Field(..., description="Explanation of confidence score and classifications")

class EvaluatedRelationships(BaseModel):
    """Container for Pass 2 output"""
    relationships: List[EvaluatedRelationship]

# Test chunks that should PASS but are currently FAILING
TEST_CHUNKS = {
    "TC004": {
        "text": "Soil stewardship enables sustainable farming practices.",
        "expected": ("soil stewardship", "enables", "sustainable farming practices"),
        "severity": "HIGH",
        "issue": "Should extract demonstrative concept, currently missing"
    },
    "TC008": {
        "text": "By working with soil, individuals can cultivate their own human potential.",
        "expected": ("individuals", "can cultivate", "human potential"),
        "severity": "MEDIUM",
        "issue": "Generic 'their' should resolve to 'individuals'"
    },
    "TC012": {
        "text": "Through regenerative soil management practices, we can mitigate climate change.",
        "expected": ("soil management", "can mitigate", "climate change"),
        "severity": "MEDIUM",
        "issue": "Testable scientific claim with hedging"
    },
    "TC013": {
        "text": "Research suggests getting your hands in the soil may enhance immune systems.",
        "expected": ("getting hands in soil", "may enhance", "immune systems"),
        "severity": "MEDIUM",
        "issue": "Testable claim with confidence hedging"
    },
    "TC017": {
        "text": "Some farmers believe living soil may enhance cognitive performance.",
        "expected": ("living soil", "may enhance", "cognitive performance"),
        "severity": "MILD",
        "issue": "Opinion with hedging should be preserved"
    },
}


async def run_pass1_extraction(chunk_text: str) -> List[Dict[str, Any]]:
    """Run Pass 1 extraction on a chunk"""
    with open(PASS1_PROMPT_PATH, 'r') as f:
        pass1_prompt = f.read()

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": pass1_prompt},
            {"role": "user", "content": f"Extract knowledge graph relationships from this text:\n\n{chunk_text}"}
        ],
        response_format=ExtractedRelationships,
        temperature=0
    )

    result = response.choices[0].message.parsed
    return [rel.model_dump() for rel in result.relationships]


async def run_pass2_evaluation(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run Pass 2 evaluation on extracted relationships"""
    with open(PASS2_PROMPT_PATH, 'r') as f:
        pass2_prompt = f.read()

    rels_text = "\n\n".join([
        f"Relationship {i+1}:\n"
        f"  Source: {rel['source']}\n"
        f"  Predicate: {rel['predicate']}\n"
        f"  Target: {rel['target']}\n"
        f"  Context: {rel['context']}"
        for i, rel in enumerate(relationships)
    ])

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": pass2_prompt},
            {"role": "user", "content": f"Evaluate these relationships:\n\n{rels_text}"}
        ],
        response_format=EvaluatedRelationships,
        temperature=0
    )

    result = response.choices[0].message.parsed
    return [rel.model_dump() for rel in result.relationships]


async def diagnose_chunk(test_id: str, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnose a single test chunk"""
    print(f"\n{'='*80}")
    print(f"🔍 Diagnosing {test_id}: {chunk_data['issue']}")
    print(f"{'='*80}")
    print(f"Text: {chunk_data['text']}")
    print(f"Expected: {chunk_data['expected']}")

    # Run Pass 1
    print(f"\n📥 Running Pass 1 extraction...")
    pass1_rels = await run_pass1_extraction(chunk_data['text'])
    print(f"   Extracted {len(pass1_rels)} relationships")

    if not pass1_rels:
        return {
            "test_id": test_id,
            "severity": chunk_data["severity"],
            "issue": chunk_data["issue"],
            "pass1_count": 0,
            "pass2_count": 0,
            "filtered_count": 0,
            "diagnosis": "EXTRACTION_FAILURE - Pass 1 extracted nothing",
            "pass1_relationships": [],
            "pass2_relationships": []
        }

    # Run Pass 2
    print(f"📊 Running Pass 2 evaluation...")
    pass2_rels = await run_pass2_evaluation(pass1_rels)
    print(f"   Evaluated {len(pass2_rels)} relationships")

    # Analyze Pass 2 scores
    print(f"\n🎯 Pass 2 Confidence Scores:")
    for i, rel in enumerate(pass2_rels):
        flags_str = ", ".join(rel['classification_flags']) if rel['classification_flags'] else "NONE"
        above_threshold = "✅ PASS" if rel['p_true'] >= 0.5 else "❌ FAIL"
        print(f"   {i+1}. {rel['source']} --[{rel['predicate']}]--> {rel['target']}")
        print(f"      p_true: {rel['p_true']:.3f} {above_threshold}")
        print(f"      Flags: {flags_str}")
        print(f"      Reasoning: {rel['reasoning'][:120]}...")

    # Filter by threshold
    filtered_rels = [rel for rel in pass2_rels if rel['p_true'] >= 0.5]

    # Diagnose
    diagnosis = []
    if len(pass1_rels) == 0:
        diagnosis.append("EXTRACTION_FAILURE")
    elif len(pass2_rels) == 0:
        diagnosis.append("EVALUATION_FAILURE")
    elif len(filtered_rels) == 0:
        diagnosis.append("THRESHOLD_FILTERING")
        max_score = max(rel['p_true'] for rel in pass2_rels)
        diagnosis.append(f"Highest p_true: {max_score:.3f} (below 0.5 threshold)")

        # Check if any have TESTABLE_CLAIM, OPINION, NORMATIVE, PHILOSOPHICAL_CLAIM flags
        for rel in pass2_rels:
            if any(flag in rel['classification_flags'] for flag in ['TESTABLE_CLAIM', 'OPINION', 'NORMATIVE', 'PHILOSOPHICAL_CLAIM']):
                diagnosis.append(f"Confidence penalty applied: {rel['classification_flags']}")
    else:
        diagnosis.append("EXTRACTED_SUCCESSFULLY")

    return {
        "test_id": test_id,
        "severity": chunk_data["severity"],
        "issue": chunk_data["issue"],
        "pass1_count": len(pass1_rels),
        "pass2_count": len(pass2_rels),
        "filtered_count": len(filtered_rels),
        "diagnosis": " | ".join(diagnosis),
        "pass1_relationships": pass1_rels,
        "pass2_relationships": pass2_rels
    }


async def main():
    """Run diagnostic on selected test cases"""
    print("="*80)
    print("🔬 V14.1 SCORING DIAGNOSTIC")
    print("="*80)
    print(f"\nTesting {len(TEST_CHUNKS)} failing test cases to understand scoring behavior")

    results = []

    for test_id, chunk_data in TEST_CHUNKS.items():
        result = await diagnose_chunk(test_id, chunk_data)
        results.append(result)
        await asyncio.sleep(0.05)  # Rate limiting

    # Summary analysis
    print(f"\n\n{'='*80}")
    print("📊 DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")

    extraction_failures = [r for r in results if r['pass1_count'] == 0]
    threshold_failures = [r for r in results if r['pass1_count'] > 0 and r['filtered_count'] == 0]
    successes = [r for r in results if r['filtered_count'] > 0]

    print(f"\nFailure Breakdown:")
    print(f"  Extraction Failures (Pass 1): {len(extraction_failures)}/{len(results)}")
    print(f"  Threshold Failures (p_true < 0.5): {len(threshold_failures)}/{len(results)}")
    print(f"  Successes: {len(successes)}/{len(results)}")

    if threshold_failures:
        print(f"\n🎯 Threshold Failure Analysis:")
        scores = []
        for r in threshold_failures:
            if r['pass2_relationships']:
                max_score = max(rel['p_true'] for rel in r['pass2_relationships'])
                scores.append(max_score)
                print(f"  {r['test_id']}: Highest p_true = {max_score:.3f}")

        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\n  Average highest score: {avg_score:.3f}")
            print(f"  Gap to threshold: {0.5 - avg_score:.3f}")

            # Calculate recommended threshold
            recommended_threshold = max(scores) * 0.95  # Allow top scores to pass
            print(f"\n  💡 INSIGHT: Lower threshold to {recommended_threshold:.3f} would allow top relationships through")

    # Check if penalties are being applied
    print(f"\n🏷️  Classification Flag Analysis:")
    flag_counts = {}
    for r in results:
        for rel in r['pass2_relationships']:
            for flag in rel['classification_flags']:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    for flag, count in sorted(flag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {flag}: {count} occurrences")

    # Save detailed results
    output_path = ANALYSIS_DIR / "v14_1_scoring_diagnosis.json"
    with open(output_path, 'w') as f:
        json.dump({
            "summary": {
                "total_tests": len(results),
                "extraction_failures": len(extraction_failures),
                "threshold_failures": len(threshold_failures),
                "successes": len(successes)
            },
            "results": results
        }, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_path}")

    # Recommendations
    print(f"\n\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}")

    if extraction_failures:
        print(f"1. ⚠️  Pass 1 extraction too conservative - {len(extraction_failures)} chunks extracted nothing")
        print(f"   → Need to relax entity/predicate constraints in Pass 1 prompt")

    if threshold_failures:
        avg_gap = 0.5 - (sum(scores) / len(scores)) if scores else 0
        print(f"2. 📉 p_true scores below threshold by average of {avg_gap:.3f}")
        if avg_gap > 0.15:
            print(f"   → Penalties still too severe - recommend further reduction")
        elif avg_gap > 0.05:
            print(f"   → Consider lowering threshold from 0.5 to 0.45 or 0.40")
        else:
            print(f"   → Scores very close to threshold - minor adjustment needed")

    if flag_counts:
        penalty_flags = {k: v for k, v in flag_counts.items() if k in ['TESTABLE_CLAIM', 'OPINION', 'NORMATIVE', 'PHILOSOPHICAL_CLAIM']}
        if penalty_flags:
            print(f"3. 🏷️  Confidence penalties being applied: {sum(penalty_flags.values())} times")
            print(f"   → Penalties ARE working, but may need further reduction")


if __name__ == "__main__":
    asyncio.run(main())
