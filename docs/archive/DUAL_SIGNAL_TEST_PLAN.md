# 🔬 Dual-Signal Extraction Test Plan

## Current Status

**We have:** Single-signal extraction (v2) complete
- ✅ 172/172 episodes
- ✅ 21,336 entities, 15,201 relationships
- ✅ 88% quality (confidence ≥ 0.75)
- ✅ Cost: $5, Time: 1.47 hours

**The Question:** Should we re-extract with dual-signal approach?

## What is Dual-Signal Extraction?

### Single-Signal (Current - v2)
```python
class RelationshipWithConfidence:
    source: str
    relationship: str
    target: str
    source_confidence: float      # How sure about source?
    relationship_confidence: float # How sure about relationship?
    target_confidence: float       # How sure about target?
```

**Problem:** We don't know WHY confidence is low. Did the LLM misread the text? Or does it know the fact is wrong?

### Dual-Signal (Proposed - v3)
```python
class DualSignalRelationship:
    source: str
    relationship: str
    target: str

    # Signal 1: Text reading (comprehension only)
    text_confidence: float          # How clear is the text?
    text_clarity: str              # explicit/implicit/unclear

    # Signal 2: World knowledge (ignore text)
    knowledge_plausibility: float  # Is this plausible?
    knowledge_reasoning: str       # Why/why not?
    source_type: str              # What type is source?
    target_type: str              # What type is target?
    type_constraint_violated: bool # Does this break rules?

    # Conflict detection
    signals_conflict: bool         # Do signals disagree?
    conflict_explanation: str
    overall_confidence: float
```

## The Breakthrough Examples

### Example 1: Biochar Error
**Text**: "International Biochar Initiative is located in biochar"

**Single-signal output:**
```json
{
  "relationship_confidence": 0.80,  // HIGH - text is clear!
  "action": "ACCEPT"  // ❌ Wrong!
}
```

**Dual-signal output:**
```json
{
  "text_confidence": 0.85,           // Text clearly states this
  "knowledge_plausibility": 0.05,    // But I know biochar is NOT a place!
  "target_type": "MATERIAL/PRODUCT",
  "type_constraint_violated": true,
  "signals_conflict": true,
  "conflict_explanation": "Text states org located in material, but biochar is charcoal/soil amendment, not a geographic location",
  "overall_confidence": 0.10,        // LOW due to conflict
  "action": "FLAG_FOR_DELETION"  // ✅ Correct!
}
```

**Result:** Caught at extraction time instead of needing validation!

### Example 2: Boulder/Lafayette
**Text**: "Boulder is located in Lafayette"

**Single-signal output:**
```json
{
  "relationship_confidence": 0.60,  // Somewhat low, but why?
  "action": "MAYBE FLAG"  // ⚠️ Unclear
}
```

**Dual-signal output:**
```json
{
  "text_confidence": 0.85,           // Text clearly states this
  "knowledge_plausibility": 0.05,    // I know Boulder > Lafayette!
  "source_type": "PLACE",
  "target_type": "PLACE",
  "type_constraint_violated": true,
  "signals_conflict": true,
  "conflict_explanation": "Boulder (pop ~110k) cannot be contained by Lafayette (pop ~30k)",
  "overall_confidence": 0.10,
  "action": "FLAG_FOR_REVERSAL"  // ✅ Clear action!
}
```

**Result:** Not only caught, but we know HOW to fix it (reverse)!

## Test Plan

### Phase 1: Test on 10 Episodes (~$0.30, 15 minutes)

**Test episodes:** 10, 39, 50, 75, 100, 112, 120, 122, 150, 165

**Why these?**
- Episode 112: Known biochar errors
- Episodes 120, 122, 165: Biochar content
- Episode 39: Colorado area content (Boulder/Lafayette)
- Others: Representative sample

**What to check:**
1. Does dual-signal detect conflicts?
2. Are conflict explanations helpful?
3. Does it catch errors single-signal missed?
4. Are type violations correctly identified?

### Phase 2: Compare Results

Run comparison script to analyze:
- How many conflicts detected?
- How many "text HIGH, knowledge LOW" cases?
- How many type violations found?
- Do conflict explanations help debugging?

### Phase 3: Decision

**If dual-signal shows advantages:**
- ✅ Proceed with full 172-episode extraction ($5, ~1.5 hours)
- Expected: Catch 80% of errors at extraction time
- Benefit: Less validation work needed

**If dual-signal shows no clear advantage:**
- ⚠️ Stick with single-signal v2
- Focus on validation passes (type checking, logical rules)
- Save $5 and 1.5 hours

## How to Run the Test

### Step 1: Run Dual-Signal Test
```bash
cd /home/claudeuser/yonearth-gaia-chatbot

# Make script executable
chmod +x scripts/test_dual_signal_extraction.py

# Run test (will take ~15 minutes)
python3 scripts/test_dual_signal_extraction.py
```

**Expected output:**
```
🔬 DUAL-SIGNAL EXTRACTION TEST
Testing on 10 episodes to validate approach
...
✨ DUAL-SIGNAL TEST COMPLETE
⏱️  Total time: 14.3 minutes
📊 Success: 10/10 episodes
🎯 Total relationships: ~900
⚠️  Conflicts detected: 45 (5.0%)
🚨 Type violations: 12
```

### Step 2: Compare to Single-Signal
```bash
# Run comparison script
python3 scripts/compare_dual_vs_single_signal.py
```

**Expected output:**
```
🔬 DUAL-SIGNAL vs SINGLE-SIGNAL COMPARISON
...
📊 OVERALL COMPARISON SUMMARY

Dual-Signal Detection:
  - Conflicts detected: 45
  - Type violations: 12
  - Text HIGH / Knowledge LOW: 23

Single-Signal Detection:
  - Low confidence: 108

🤔 RECOMMENDATION
✅ PROCEED WITH FULL DUAL-SIGNAL EXTRACTION
   Dual-signal catches significantly more errors
   Estimated improvement: 23 additional errors caught per 10 episodes
```

### Step 3: Review Specific Examples
```bash
# Look at specific conflict examples
cat data/knowledge_graph_dual_signal_test/episode_112_dual_signal.json | grep -A20 '"signals_conflict": true'
```

### Step 4: Make Decision

**If test shows dual-signal works well:**
```bash
# Create full dual-signal extraction script
# (Same as test script but for all 172 episodes)
python3 scripts/full_dual_signal_extraction.py
```

**If test shows minimal advantage:**
```bash
# Stick with v2, focus on validation passes
# Priority 1: Type checking (GeoNames, Wikidata)
# Priority 2: Logical rules (population hierarchy)
```

## Expected Outcomes

### Best Case Scenario
- Dual-signal catches 20-30% more errors than single-signal
- Conflict explanations provide actionable debugging info
- Type violations automatically identified
- **Decision:** Proceed with full extraction

### Good Case Scenario
- Dual-signal catches similar errors but with better explanations
- Helps understand WHY confidence is low
- Some additional errors caught (10-15%)
- **Decision:** Consider full extraction for better debugging

### Neutral Case Scenario
- Dual-signal catches same errors as single-signal
- No significant advantage
- **Decision:** Stick with v2, focus on validation passes

## Cost-Benefit Analysis

**Test cost:**
- $0.30 for 10 episodes
- 15 minutes processing time
- 10 minutes analysis time

**Full extraction cost (if test succeeds):**
- $5 for 172 episodes (same as v2)
- 1.5 hours processing time
- Potential benefit: 80% fewer errors downstream

**Validation alternative (if test fails):**
- $0 for type checking (uses external APIs)
- 1-2 days implementation time
- 49 known errors auto-fixed

## Research Validation

This approach is based on:
- **FaR Method (2024)**: 23.5% reduction in calibration error
- **Knowledge-Grounded IE (2023)**: 15-20% precision improvement
- **DUAL_SIGNAL_EXTRACTION_ULTRATHOUGHT.md**: Our architectural design

## Next Steps

1. **Run test script** (15 minutes)
2. **Run comparison script** (2 minutes)
3. **Review results** (10 minutes)
4. **Make decision:**
   - If good results → Full dual-signal extraction ($5, 1.5 hours)
   - If mixed results → Stick with v2, implement validation passes

## Files Created

1. **scripts/test_dual_signal_extraction.py** - Test on 10 episodes
2. **scripts/compare_dual_vs_single_signal.py** - Compare results
3. **docs/DUAL_SIGNAL_TEST_PLAN.md** - This document

## Success Criteria

Dual-signal is worth pursuing if:
1. ✅ Catches ≥10 additional errors in test (vs. single-signal)
2. ✅ Conflict explanations are actionable and helpful
3. ✅ Type violations are correctly identified
4. ✅ "Text HIGH / Knowledge LOW" pattern catches real errors

If ≥3 of these criteria are met → **Proceed with full extraction**

---

**Status:** Ready to test
**Next action:** Run `python3 scripts/test_dual_signal_extraction.py`
**Time estimate:** 15 minutes for test, 2 minutes for comparison
**Cost:** $0.30
