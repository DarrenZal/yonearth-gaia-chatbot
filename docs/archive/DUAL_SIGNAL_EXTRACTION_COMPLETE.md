# 🔬 Dual-Signal Knowledge Graph Extraction: Complete Guide

**Status:** Testing in progress (October 2025)
**Current Tests:** gpt-4o-mini, gpt-5-mini, gpt-5-nano, two-pass extraction

---

## Table of Contents
1. [The Core Insight](#the-core-insight)
2. [Test Results & The Coverage Problem](#test-results--the-coverage-problem)
3. [Solution Options](#solution-options)
4. [Current Testing Status](#current-testing-status)
5. [Implementation Details](#implementation-details)

---

## The Core Insight

### The Fundamental Question

> "When we extract knowledge, should the LLM only consider the content, or also use its own knowledge?
> For example, if text says 'biochar is a type of place', should we return:
> - What the text says (100% confidence from content)
> - What the LLM knows (90% confidence it's NOT a place)
> - Or both?"

**This is the fundamental tension in knowledge extraction!**

### The Two Philosophies

#### Philosophy A: Pure Information Extraction (Single-Signal)
```
Task: "Extract what the text says"

Text: "Boulder is located in Lafayette"
Output: Boulder --[located_in]--> Lafayette
Confidence: 0.85 (high, because text is clear)

Problem: We extracted what was SAID, but it's WRONG!
```

**Treats LLM as:** A reader (comprehension task)
**Truth definition:** What the document claims
**Ignores:** LLM's world knowledge

#### Philosophy B: Knowledge-Grounded Extraction (Dual-Signal)
```
Task: "Extract what the text says AND validate against your knowledge"

Text: "Boulder is located in Lafayette"

Output:
  text_signal: Boulder --[located_in]--> Lafayette
  text_confidence: 0.85 (text clearly states this)

  knowledge_signal: This contradicts my training
  knowledge_confidence: 0.95 (I know Boulder > Lafayette)

  conflict_detected: TRUE
  overall_confidence: 0.20 (LOW due to conflict!)
```

**Treats LLM as:** Reader + Fact Checker
**Truth definition:** What's actually true in the world
**Leverages:** LLM's world knowledge

### The Biochar Example That Started It All

```python
# Scenario: Transcript contains confused statement
Text: "The International Biochar Initiative is located in biochar"

# Single-Signal (v2):
{
    "source": "International Biochar Initiative",
    "relationship": "located_in",
    "target": "biochar",
    "confidence": 0.80  # HIGH because text is clear!
}
# Result: We confidently extract WRONG information!

# Dual-Signal:
{
    "source": "International Biochar Initiative",
    "relationship": "located_in",
    "target": "biochar",

    # Signal 1: What the text says
    "text_confidence": 0.80,
    "text_clarity": "explicit",

    # Signal 2: What LLM knows
    "knowledge_plausibility": 0.05,  # I know biochar is NOT a place!
    "knowledge_reasoning": "biochar is a soil amendment, not a location",

    # Conflict detection
    "signals_conflict": true,
    "overall_confidence": 0.15,  # LOW due to conflict
}
```

**This catches the error at extraction time!**

---

## Test Results & The Coverage Problem

### What We Discovered

Our initial dual-signal tests revealed a critical problem:

**Coverage Analysis (10 test episodes):**
- Single-signal extracted: **883 entity pairs**
- Dual-signal (gpt-4o-mini) extracted: **691 entity pairs**
- **Shared pairs: Only 90 (10.2%)**
- **Missing in dual: 793 pairs (89.8%)**

**Examples of Missing High-Confidence Facts:**
1. "Lauren Tucker" → "American University" (graduated_from, 0.95 conf)
2. "Boulder" → "Colorado" (located_in, 0.95 conf)
3. "Kelpie Wilson" → "International Biochar Initiative" (works_at, 0.94 conf)

### Why This Happens

**Hypothesis 1: Prompt Complexity**
- Dual-signal prompt asks for 5 simultaneous tasks per relationship:
  1. Extract relationships (generative)
  2. Evaluate text clarity (analytical)
  3. Check knowledge plausibility (recall)
  4. Identify entity types (classification)
  5. Detect conflicts (reasoning)
- Model becomes overwhelmed and extracts fewer relationships

**Hypothesis 2: Conservative Behavior**
- Emphasis on conflict detection makes model overly cautious
- Model skips uncertain relationships to avoid mistakes
- Sacrifices completeness for accuracy

**Hypothesis 3: Different Granularity**
- Dual-signal extracts at different detail level than single-signal
- May find DIFFERENT relationships, not the SAME ones
- Complementary information rather than overlapping

### Initial Test Results

**Episode 10 Comparison:**

| Model | Relationships | Entity Pairs | Coverage vs v2 | Conflicts |
|-------|--------------|--------------|----------------|-----------|
| Single-signal (v2) | 64 | 63 | 100% (baseline) | N/A |
| gpt-4o-mini (dual) | 45 | 44 | 12.7% | 0 |
| gpt-5-mini (dual) | 136 | 136 | 11.1% | 3 |
| Two-pass (Pass 1) | 233 | TBD | TBD | TBD |

**Key Finding:** gpt-5-mini extracted 3x MORE relationships but still had low coverage overlap with baseline.

**Conflicts Detected by gpt-5-mini:**
1. "Kiss the Ground" social media handle error (transcription issue)
2. Innate soil detection claim (implausible biological claim)
3. Urban soil quality superlative (overgeneralization)

---

## Solution Options

### Option A: Modified Single-Pass (Comprehensive Dual-Signal)

**Approach:** Fix the prompt to emphasize extraction over filtering

```python
DUAL_SIGNAL_PROMPT_V2 = """
CRITICAL INSTRUCTION: Extract ALL relationships you observe in the text,
even if you have doubts about their plausibility. The dual-signal system
is for EVALUATION, not FILTERING during extraction.

Your job is to:
1. Extract EVERY relationship mentioned in the text (be comprehensive!)
2. For each one, provide honest dual-signal assessment
3. Mark conflicts when text and knowledge disagree
4. Let the validation system handle filtering later

DO NOT skip relationships because they seem implausible - extract them
and mark them with low knowledge_plausibility instead!
"""
```

**Pros:**
- ✅ Single LLM call per chunk (cost-efficient)
- ✅ Maintains dual-signal conflict detection
- ✅ Should achieve better coverage

**Cons:**
- ⚠️ May still have coverage gaps
- ⚠️ Cognitive load still high

**Cost:** $5, 1.5 hours for full extraction

**Status:** Not tested (smarter models tested instead)

---

### Option B: Two-Pass Extraction (Extract, Then Evaluate)

**Approach:** Separate extraction from evaluation into two distinct passes

**Pass 1 - Comprehensive Extraction:**
```python
EXTRACTION_PROMPT = """
Extract ALL relationships from this text. Be comprehensive and thorough.
Include every fact, connection, and relationship you observe.

For each relationship, provide:
- source, relationship, target
- context (where in text)
"""
```

**Pass 2 - Dual-Signal Evaluation:**
```python
EVALUATION_PROMPT = """
For this extracted relationship:
  {source} --[{relationship}]--> {target}

Provide dual-signal assessment:
1. Text confidence: How clearly does the text state this?
2. Knowledge plausibility: Is this plausible based on your knowledge?
3. Conflict detection: Do these signals disagree?
"""
```

**Pros:**
- ✅ **Best coverage** - extraction is separate from evaluation
- ✅ Focused cognitive tasks (one at a time)
- ✅ Catches errors while maintaining comprehensiveness
- ✅ Clear separation of concerns

**Cons:**
- ❌ **2x API calls** = 2x cost ($10 instead of $5)
- ❌ **2x time** = 3 hours instead of 1.5 hours

**Cost:** $10, 3 hours for full extraction

**Status:** ✅ Currently testing on 10 episodes with gpt-4o-mini

**Early Results:**
- Episode 10 Pass 1: Extracted **233 relationships** (3.6x more than v2!)
- Pass 2: In progress (evaluating 30/233)

---

### Option C: Hybrid Approach (Validate Existing v2)

**Approach:** Keep single-signal extraction, add dual-signal validation layer

**Step 1:** Use existing single-signal extraction (already done!)
- 172 episodes
- 21,336 entities, 15,201 relationships
- $5 cost (already paid)

**Step 2:** Run dual-signal VALIDATION on extracted relationships
```python
VALIDATION_PROMPT = """
You are validating an extracted relationship:
  {source} --[{relationship}]--> {target}

Assess:
1. Is this factually plausible based on your knowledge?
2. Do the entity types make sense for this relationship?
3. Are there any type constraint violations?

Provide:
- knowledge_plausibility (0-1)
- type_violation (bool)
- reasoning (why/why not)
"""
```

**Pros:**
- ✅ **Comprehensive coverage** (uses existing extraction)
- ✅ Adds error detection to existing data
- ✅ Can validate in batches (efficient)
- ✅ No re-extraction needed

**Cons:**
- ⚠️ Inherits any issues from v2 extraction
- ⚠️ Won't find new relationships v2 missed

**Cost:** ~$3 for validation pass (15,201 relationships)

**Status:** Not yet tested

---

### Option D: Smarter Models (gpt-5-mini, gpt-5-nano)

**Approach:** Test if more capable models handle dual-signal better

**Hypothesis:** Smarter/faster models might:
- Handle complex dual-signal prompts better
- Achieve better coverage without sacrificing quality
- Process faster (gpt-5-nano)

**Models Tested:**
1. **gpt-5-mini** (smarter than gpt-4o-mini)
   - Status: Testing in progress (episode 50/10)
   - Speed: ~70 seconds per API call (3x slower than gpt-4o-mini)
   - Episode 10: 136 relationships, 3 conflicts
   - Episode 39: 195 relationships, 13 conflicts

2. **gpt-5-nano** (fastest GPT-5 model)
   - Status: Testing started (episode 10, chunk 0/9)
   - Speed: TBD (checking if truly faster)
   - Expected: Similar or better speed than gpt-4o-mini

**Pros:**
- ✅ Single-pass approach (cost-efficient)
- ✅ Better conflict detection than gpt-4o-mini
- ✅ More comprehensive extraction

**Cons:**
- ⚠️ Still shows coverage problem (~11% vs baseline)
- ⚠️ gpt-5-mini is 3x slower
- ⚠️ Extracts DIFFERENT relationships, not same ones

**Cost:** Same as Option A ($5, but longer time for gpt-5-mini)

---

## Current Testing Status

### Active Tests (Running in Parallel)

#### 1. gpt-5-mini Dual-Signal
- **Status:** Episode 50/10 in progress
- **Progress:** ~20% complete
- **Speed:** ~70 seconds per API call
- **Results so far:**
  - Episode 10: 136 relationships, 3 conflicts
  - Episode 39: 195 relationships, 13 conflicts
- **Estimated completion:** ~60-90 minutes remaining

#### 2. Two-Pass (gpt-4o-mini)
- **Status:** Episode 10, Pass 2 in progress
- **Progress:** Evaluating 30/233 relationships (~13%)
- **Speed:** ~3-4 seconds per evaluation
- **Results so far:**
  - Pass 1: 233 relationships extracted
  - Pass 2: Evaluating each with dual-signal
- **Estimated completion:** ~60 minutes remaining

#### 3. gpt-5-nano Dual-Signal
- **Status:** Just started (episode 10, chunk 0/9)
- **Progress:** <1% complete
- **Speed:** TBD (waiting for first chunks)
- **Hypothesis:** Should be faster than gpt-4o-mini
- **Estimated completion:** ~30-45 minutes (if fast)

### Comparison Analysis Ready

**Script created:** `scripts/compare_all_approaches.py`

**Will compare:**
1. Single-signal (v2) - baseline
2. Dual-signal gpt-4o-mini - original test (10.2% coverage)
3. Dual-signal gpt-5-mini - smarter model (TBD)
4. Dual-signal gpt-5-nano - fastest model (TBD)
5. Two-pass gpt-4o-mini - Option B (TBD)

**Metrics:**
- Entity pair coverage vs baseline
- Conflict detection rate
- Relationship counts
- Quality indicators
- Speed/cost tradeoffs

---

## Implementation Details

### Dual-Signal Schema

```python
class DualSignalRelationship(BaseModel):
    """Relationship with separated text and knowledge signals"""

    # The extraction
    source: str
    relationship: str
    target: str
    context: Optional[str]

    # Signal 1: Text-based (reading comprehension ONLY)
    text_confidence: float  # 0.0-1.0
    text_clarity: Literal["explicit", "implicit", "inferred", "unclear"]

    # Signal 2: Knowledge-based (world knowledge ONLY)
    knowledge_plausibility: float  # 0.0-1.0
    knowledge_reasoning: str

    # Entity type checking (from knowledge)
    source_type: Optional[str]  # PERSON/ORG/PLACE/CONCEPT/etc
    target_type: Optional[str]
    type_constraint_violated: bool

    # Conflict detection
    signals_conflict: bool
    conflict_explanation: Optional[str]

    # Combined assessment
    overall_confidence: float  # Should be LOW if conflict detected

    # Provenance
    episode_number: int
```

### Error Types Caught

#### Type 1: Extraction Mistakes (LLM misread)
```
Text: "Boulder, located near Lafayette..."
Bad extraction: Boulder located_in Lafayette

Dual signal catches:
  text_confidence: 0.60 (unclear, says "near" not "in")
  knowledge_plausibility: 0.05 (I know this is wrong)
  → overall_confidence: 0.10 ❌
```

#### Type 2: Speaker/Content Errors (text is wrong)
```
Text: "Boulder is in Lafayette" (speaker misspoke)
Pure extraction: Boulder located_in Lafayette (0.90 confidence!)

Dual signal catches:
  text_confidence: 0.90 (text is clear)
  knowledge_plausibility: 0.05 (but I know it's backwards!)
  conflict_detected: true
  → overall_confidence: 0.15 ❌
```

#### Type 3: Type Violations
```
Text: "Located in biochar"
Pure extraction: X located_in biochar (0.80)

Dual signal catches:
  text_confidence: 0.80 (text states this)
  knowledge_plausibility: 0.05 (biochar is not a place!)
  type_constraint_violated: true
  → overall_confidence: 0.10 ❌
```

### Research Validation

This approach is based on:
- **FaR Method (2024)**: 23.5% reduction in calibration error through fact-and-reflection
- **Knowledge-Grounded IE (2023)**: 15-20% precision improvement using external KB validation
- **Provenance-Aware Extraction**: Separates text evidence from world knowledge

---

## Decision Framework

### When Tests Complete

**Run comprehensive comparison:**
```bash
python3 scripts/compare_all_approaches.py
```

**Evaluate based on:**
1. **Coverage:** Which approach achieves ≥70% entity pair coverage vs v2?
2. **Conflict Detection:** Which finds meaningful errors?
3. **Speed:** What's the time/cost tradeoff?
4. **Quality:** Which provides best debugging information?

### Decision Criteria

**If Two-Pass (Option B) shows ≥80% coverage:**
- ✅ Use two-pass for full extraction
- Cost: $10, Time: 3 hours
- Best quality, worth the 2x cost

**If a smarter model (gpt-5-mini/nano) shows ≥70% coverage:**
- ✅ Use that model for single-pass dual-signal
- Cost: $5-7, Time: 1.5-3 hours
- Good balance of coverage and error detection

**If all dual-signal approaches show <50% coverage:**
- ⚠️ Use Option C (validate existing v2)
- Cost: $3, Time: 1.5 hours
- Add validation to existing comprehensive extraction

**If all approaches fail:**
- Consider hybrid: v2 extraction + targeted validation on low-confidence only
- Investigate why dual-signal struggles with coverage

---

## Files Reference

### Documentation
- `/docs/DUAL_SIGNAL_EXTRACTION_COMPLETE.md` - This document (consolidated)
- ~~`/docs/DUAL_SIGNAL_EXTRACTION_ULTRATHOUGHT.md`~~ - Archived (merged here)
- ~~`/docs/DUAL_SIGNAL_TEST_PLAN.md`~~ - Archived (merged here)
- ~~`/docs/DUAL_SIGNAL_PROBLEM_AND_SOLUTIONS.md`~~ - Archived (merged here)

### Test Scripts
- `/scripts/test_dual_signal_extraction.py` - gpt-4o-mini test
- `/scripts/test_dual_signal_gpt5_mini.py` - gpt-5-mini test
- `/scripts/test_dual_signal_gpt5_nano.py` - gpt-5-nano test
- `/scripts/test_two_pass_extraction.py` - Two-pass (Option B) test
- `/scripts/compare_all_approaches.py` - Comprehensive comparison

### Analysis Scripts
- `/scripts/analyze_entity_pairs.py` - Entity pair coverage analysis
- `/scripts/analyze_missing_relationships.py` - Missing relationship investigation
- `/scripts/compare_dual_vs_single_signal.py` - Dual vs single comparison
- `/scripts/compare_gpt5_vs_gpt4o_mini.py` - Model comparison

### Output Directories
- `/data/knowledge_graph_v2/` - Single-signal baseline (v2)
- `/data/knowledge_graph_dual_signal_test/` - gpt-4o-mini dual-signal
- `/data/knowledge_graph_gpt5_mini_test/` - gpt-5-mini dual-signal
- `/data/knowledge_graph_gpt5_nano_test/` - gpt-5-nano dual-signal
- `/data/knowledge_graph_two_pass_test/` - Two-pass extraction

---

## Next Steps

1. **Wait for tests to complete** (~60-90 minutes)
2. **Run comprehensive comparison** to evaluate all approaches
3. **Make final decision** based on coverage, quality, and cost
4. **Implement chosen approach** for full 172-episode extraction
5. **Build unified knowledge graph** from best extraction results

**Last Updated:** October 10, 2025
**Tests Running:** 3 parallel tests in progress
**Expected Decision:** Within 2 hours
