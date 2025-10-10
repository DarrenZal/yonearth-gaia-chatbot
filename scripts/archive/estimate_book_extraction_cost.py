#!/usr/bin/env python3
"""
Estimate cost and time for extracting a 540-page book using different approaches.

Assumptions:
- 540 pages
- ~300 words per page = 162,000 words
- ~200,000 tokens total (1.23 tokens/word average)
- Chunk size: 800 tokens with 100 overlap
- ~285 chunks (200,000 / 800 * 1.14 overlap factor)
"""

import json

# Book parameters
PAGES = 540
WORDS_PER_PAGE = 300
TOTAL_WORDS = PAGES * WORDS_PER_PAGE
TOKENS_PER_WORD = 1.23
TOTAL_TOKENS = int(TOTAL_WORDS * TOKENS_PER_WORD)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
OVERLAP_FACTOR = 1.14  # Chunks overlap, so more chunks than naive calculation
TOTAL_CHUNKS = int((TOTAL_TOKENS / CHUNK_SIZE) * OVERLAP_FACTOR)

print(f"📖 BOOK PARAMETERS")
print(f"{'='*60}")
print(f"Pages: {PAGES}")
print(f"Words: {TOTAL_WORDS:,}")
print(f"Tokens: {TOTAL_TOKENS:,}")
print(f"Chunk size: {CHUNK_SIZE} tokens")
print(f"Total chunks: {TOTAL_CHUNKS}")
print()

# OpenAI Pricing (as of October 2025)
# Source: https://openai.com/api/pricing/

PRICING = {
    "gpt-4o-mini": {
        "input": 0.150 / 1_000_000,   # $0.150 per 1M input tokens
        "output": 0.600 / 1_000_000,  # $0.600 per 1M output tokens
        "name": "gpt-4o-mini"
    },
    "gpt-5-mini": {
        "input": 2.00 / 1_000_000,    # $2.00 per 1M input tokens (estimated)
        "output": 8.00 / 1_000_000,   # $8.00 per 1M output tokens (estimated)
        "name": "gpt-5-mini"
    }
}

# Estimated tokens per API call based on test data
# From episode tests: avg chunk = 800 tokens input, generates ~200 tokens output per relationship

def estimate_single_pass_gpt5_mini():
    """Single-pass dual-signal extraction with gpt-5-mini."""
    print(f"{'='*60}")
    print(f"APPROACH 1: Single-Pass Dual-Signal (gpt-5-mini)")
    print(f"{'='*60}")

    # From test data: gpt-5-mini extracts ~197.9 relationships per episode
    # Episodes avg ~13 chunks, so ~15.2 relationships per chunk
    rels_per_chunk = 15.2

    # Input: system prompt + chunk text
    system_prompt_tokens = 500  # Dual-signal prompt
    chunk_tokens = 800
    input_tokens_per_chunk = system_prompt_tokens + chunk_tokens

    # Output: JSON with relationships (avg ~200 tokens per relationship)
    # But Pydantic structured output has overhead, estimate 250 tokens per rel
    tokens_per_relationship = 250
    output_tokens_per_chunk = int(rels_per_chunk * tokens_per_relationship)

    total_input_tokens = TOTAL_CHUNKS * input_tokens_per_chunk
    total_output_tokens = TOTAL_CHUNKS * output_tokens_per_chunk

    pricing = PRICING["gpt-5-mini"]
    input_cost = total_input_tokens * pricing["input"]
    output_cost = total_output_tokens * pricing["output"]
    total_cost = input_cost + output_cost

    # Time estimation: ~2 seconds per API call (observed from tests)
    seconds_per_chunk = 2.0
    total_seconds = TOTAL_CHUNKS * seconds_per_chunk
    total_hours = total_seconds / 3600

    # Total relationships
    total_relationships = int(TOTAL_CHUNKS * rels_per_chunk)

    print(f"\n📊 Token Usage:")
    print(f"   Input tokens:  {total_input_tokens:,} ({TOTAL_CHUNKS} chunks × {input_tokens_per_chunk:,})")
    print(f"   Output tokens: {total_output_tokens:,} (~{rels_per_chunk:.1f} rels/chunk × {tokens_per_relationship} tokens)")
    print(f"   Total tokens:  {total_input_tokens + total_output_tokens:,}")

    print(f"\n💰 Cost Breakdown:")
    print(f"   Input cost:  ${input_cost:.2f} ({total_input_tokens:,} tokens @ ${pricing['input']*1_000_000:.2f}/1M)")
    print(f"   Output cost: ${output_cost:.2f} ({total_output_tokens:,} tokens @ ${pricing['output']*1_000_000:.2f}/1M)")
    print(f"   TOTAL COST:  ${total_cost:.2f}")

    print(f"\n⏱️  Time Estimation:")
    print(f"   API calls: {TOTAL_CHUNKS}")
    print(f"   Time per call: {seconds_per_chunk:.1f}s")
    print(f"   TOTAL TIME: {total_hours:.1f} hours ({total_seconds/60:.0f} minutes)")

    print(f"\n📈 Expected Output:")
    print(f"   Relationships: ~{total_relationships:,}")
    print(f"   Conflicts: ~{int(total_relationships * 0.045)} (4.5% rate)")

    return {
        "approach": "Single-Pass gpt-5-mini",
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_cost": total_cost,
        "total_hours": total_hours,
        "total_relationships": total_relationships,
        "api_calls": TOTAL_CHUNKS
    }

def estimate_two_pass_batched_gpt4o_mini():
    """Two-pass batched extraction with gpt-4o-mini."""
    print(f"\n{'='*60}")
    print(f"APPROACH 2: Two-Pass Batched (gpt-4o-mini)")
    print(f"{'='*60}")

    # From test data: two-pass extracts ~280-313 relationships per episode
    # Episodes avg ~14 chunks, so ~21 relationships per chunk
    rels_per_chunk = 21.0

    # PASS 1: Extract relationships (comprehensive)
    print(f"\n🔄 PASS 1: Extraction")
    system_prompt_tokens = 400  # Simpler extraction prompt
    chunk_tokens = 800
    p1_input_tokens_per_chunk = system_prompt_tokens + chunk_tokens

    tokens_per_relationship = 200  # Simpler schema (no dual-signal yet)
    p1_output_tokens_per_chunk = int(rels_per_chunk * tokens_per_relationship)

    p1_total_input = TOTAL_CHUNKS * p1_input_tokens_per_chunk
    p1_total_output = TOTAL_CHUNKS * p1_output_tokens_per_chunk

    pricing = PRICING["gpt-4o-mini"]
    p1_input_cost = p1_total_input * pricing["input"]
    p1_output_cost = p1_total_output * pricing["output"]
    p1_total_cost = p1_input_cost + p1_output_cost

    p1_seconds_per_chunk = 1.5  # gpt-4o-mini is faster
    p1_total_seconds = TOTAL_CHUNKS * p1_seconds_per_chunk
    p1_total_hours = p1_total_seconds / 3600

    print(f"   Input tokens:  {p1_total_input:,}")
    print(f"   Output tokens: {p1_total_output:,}")
    print(f"   Cost: ${p1_total_cost:.2f}")
    print(f"   Time: {p1_total_hours:.1f} hours")
    print(f"   Relationships extracted: ~{int(TOTAL_CHUNKS * rels_per_chunk):,}")

    # PASS 2: Batched evaluation (50 relationships per batch)
    print(f"\n🔄 PASS 2: Batched Evaluation")
    total_relationships = int(TOTAL_CHUNKS * rels_per_chunk)
    batch_size = 50
    num_batches = int(total_relationships / batch_size) + 1

    # Each batch: system prompt + 50 relationships as JSON
    p2_system_prompt_tokens = 600  # Dual-signal evaluation prompt
    tokens_per_rel_in_batch = 100  # Compact JSON representation
    p2_input_tokens_per_batch = p2_system_prompt_tokens + (batch_size * tokens_per_rel_in_batch)

    # Output: 50 evaluations with dual-signal scores
    tokens_per_evaluation = 150  # Dual-signal result
    p2_output_tokens_per_batch = batch_size * tokens_per_evaluation

    p2_total_input = num_batches * p2_input_tokens_per_batch
    p2_total_output = num_batches * p2_output_tokens_per_batch

    p2_input_cost = p2_total_input * pricing["input"]
    p2_output_cost = p2_total_output * pricing["output"]
    p2_total_cost = p2_input_cost + p2_output_cost

    p2_seconds_per_batch = 120  # 2 minutes per batch (observed)
    p2_total_seconds = num_batches * p2_seconds_per_batch
    p2_total_hours = p2_total_seconds / 3600

    print(f"   Batches: {num_batches} (50 rels each)")
    print(f"   Input tokens:  {p2_total_input:,}")
    print(f"   Output tokens: {p2_total_output:,}")
    print(f"   Cost: ${p2_total_cost:.2f}")
    print(f"   Time: {p2_total_hours:.1f} hours")

    # TOTALS
    total_input_tokens = p1_total_input + p2_total_input
    total_output_tokens = p1_total_output + p2_total_output
    total_cost = p1_total_cost + p2_total_cost
    total_hours = p1_total_hours + p2_total_hours
    total_api_calls = TOTAL_CHUNKS + num_batches

    conflicts = int(total_relationships * 0.063)  # 6.3% observed rate

    print(f"\n📊 COMBINED TOTALS:")
    print(f"   Input tokens:  {total_input_tokens:,}")
    print(f"   Output tokens: {total_output_tokens:,}")
    print(f"   TOTAL COST: ${total_cost:.2f}")
    print(f"   TOTAL TIME: {total_hours:.1f} hours")
    print(f"   API calls: {total_api_calls:,} ({TOTAL_CHUNKS} Pass 1 + {num_batches} Pass 2)")

    print(f"\n📈 Expected Output:")
    print(f"   Relationships: ~{total_relationships:,}")
    print(f"   Conflicts: ~{conflicts} (6.3% rate)")

    return {
        "approach": "Two-Pass Batched gpt-4o-mini",
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_cost": total_cost,
        "total_hours": total_hours,
        "total_relationships": total_relationships,
        "api_calls": total_api_calls
    }

def compare_approaches(approach1, approach2):
    """Compare the two approaches."""
    print(f"\n{'='*60}")
    print(f"🔬 COMPARISON")
    print(f"{'='*60}\n")

    print(f"{'Metric':<25} {'gpt-5-mini':>18} {'gpt-4o-mini':>18} {'Difference':>15}")
    print(f"{'-'*80}")

    # Cost
    cost_diff = approach1['total_cost'] - approach2['total_cost']
    cost_pct = (cost_diff / approach2['total_cost'] * 100) if approach2['total_cost'] > 0 else 0
    print(f"{'💰 Total Cost':<25} ${approach1['total_cost']:>16.2f} ${approach2['total_cost']:>16.2f} ${cost_diff:>13.2f} ({cost_pct:+.0f}%)")

    # Time
    time_diff = approach1['total_hours'] - approach2['total_hours']
    time_pct = (time_diff / approach2['total_hours'] * 100) if approach2['total_hours'] > 0 else 0
    print(f"{'⏱️  Total Time (hours)':<25} {approach1['total_hours']:>18.1f} {approach2['total_hours']:>18.1f} {time_diff:>14.1f} ({time_pct:+.0f}%)")

    # Relationships
    rel_diff = approach1['total_relationships'] - approach2['total_relationships']
    rel_pct = (rel_diff / approach2['total_relationships'] * 100) if approach2['total_relationships'] > 0 else 0
    print(f"{'📊 Relationships':<25} {approach1['total_relationships']:>18,} {approach2['total_relationships']:>18,} {rel_diff:>14,} ({rel_pct:+.0f}%)")

    # API calls
    api_diff = approach1['api_calls'] - approach2['api_calls']
    api_pct = (api_diff / approach2['api_calls'] * 100) if approach2['api_calls'] > 0 else 0
    print(f"{'🔌 API Calls':<25} {approach1['api_calls']:>18,} {approach2['api_calls']:>18,} {api_diff:>14,} ({api_pct:+.0f}%)")

    # Cost per relationship
    cost_per_rel_1 = approach1['total_cost'] / approach1['total_relationships']
    cost_per_rel_2 = approach2['total_cost'] / approach2['total_relationships']
    print(f"{'💵 Cost per Relationship':<25} ${cost_per_rel_1:>17.4f} ${cost_per_rel_2:>17.4f}")

    # Verdict
    print(f"\n{'='*60}")
    print(f"🏆 VERDICT")
    print(f"{'='*60}\n")

    if cost_diff < 0 and rel_diff > 0:
        print(f"✅ {approach1['approach']} WINS on BOTH cost AND coverage!")
        print(f"   • ${abs(cost_diff):.2f} CHEAPER ({abs(cost_pct):.0f}%)")
        print(f"   • {rel_diff:,} MORE relationships ({abs(rel_pct):.0f}%)")
    elif cost_diff > 0 and rel_diff > 0:
        print(f"⚖️  TRADEOFF: {approach1['approach']} costs more but extracts more")
        print(f"   • ${cost_diff:.2f} more expensive (+{cost_pct:.0f}%)")
        print(f"   • {rel_diff:,} MORE relationships (+{rel_pct:.0f}%)")
        print(f"   • Worth it? ${cost_diff:.2f} / {rel_diff:,} rels = ${cost_diff/rel_diff:.4f} per extra relationship")
    elif cost_diff < 0 and rel_diff < 0:
        print(f"⚖️  TRADEOFF: {approach1['approach']} cheaper but extracts less")
        print(f"   • ${abs(cost_diff):.2f} CHEAPER ({abs(cost_pct):.0f}%)")
        print(f"   • {abs(rel_diff):,} FEWER relationships ({abs(rel_pct):.0f}%)")
    else:
        print(f"✅ {approach2['approach']} WINS on BOTH cost AND coverage!")
        print(f"   • ${abs(cost_diff):.2f} CHEAPER ({abs(cost_pct):.0f}%)")
        print(f"   • {abs(rel_diff):,} MORE relationships ({abs(rel_pct):.0f}%)")

    print(f"\n💡 RECOMMENDATION:")
    if cost_diff > 0 and rel_diff < 0:
        print(f"   Use {approach2['approach']} - cheaper AND more comprehensive!")
    elif cost_diff < 0 and rel_diff > 0:
        print(f"   Use {approach1['approach']} - cheaper AND more comprehensive!")
    elif cost_diff > 0:
        extra_cost_per_rel = cost_diff / abs(rel_diff) if rel_diff != 0 else 0
        if extra_cost_per_rel < 0.01:
            print(f"   Use {approach1['approach']} - only ${extra_cost_per_rel:.4f} per extra relationship")
        else:
            print(f"   Use {approach2['approach']} - ${abs(cost_diff):.2f} savings outweigh {abs(rel_diff)} fewer relationships")
    else:
        print(f"   Use {approach2['approach']} - better value for money")

def main():
    approach1 = estimate_single_pass_gpt5_mini()
    approach2 = estimate_two_pass_batched_gpt4o_mini()
    compare_approaches(approach1, approach2)

if __name__ == "__main__":
    main()
