# 🔬 Dual-Signal Extraction: Problem Analysis & Solutions

## 📊 Test Results Summary

**Coverage Analysis:**
- Single-signal extracted: **883 entity pairs** (unique source→target relationships)
- Dual-signal extracted: **691 entity pairs**
- **Shared pairs: Only 90 (10.2%)**
- **Missing in dual: 793 pairs (89.8%)**
- **High-confidence pairs missing: 777**

## 🎯 The Core Problem

### What We Discovered

Dual-signal is **TOO CONSERVATIVE** and misses 90% of valid facts, including high-confidence relationships like:

**Examples of Missing Facts:**
1. "Lauren Tucker" → "American University" (graduated_from, 0.95 conf)
2. "Boulder" → "Colorado" (located_in, 0.95 conf)
3. "Kelpie Wilson" → "International Biochar Initiative" (works_at, 0.94 conf)
4. "Craig Hospital" → "Denver" (located_in, 0.94 conf)

These are all **valid, important facts** that should be in the knowledge graph!

### Why Is This Happening?

**Hypothesis 1: Prompt Complexity**
The dual-signal prompt is VERY complex:
- Asks model to separate text reading from knowledge validation
- Multiple fields to fill (text_confidence, knowledge_plausibility, types, violations, conflicts)
- Model may be getting overwhelmed and extracting fewer relationships

**Hypothesis 2: Conservative Behavior**
The dual-signal prompt emphasizes **conflict detection**, which may make the model:
- Skip relationships it's uncertain about
- Focus on "obvious" facts only
- Avoid relationships that might trigger conflicts

**Hypothesis 3: Cognitive Load**
The model is being asked to:
1. Extract relationships (generative task)
2. Evaluate text clarity (analytical task)
3. Check knowledge plausibility (recall task)
4. Identify entity types (classification task)
5. Detect conflicts (reasoning task)

This is 5 simultaneous tasks per relationship! The model may sacrifice completeness for accuracy.

## ✨ Solution Options

### Option A: Modified Single-Pass (Comprehensive Dual-Signal)

**Approach:** Fix the dual-signal prompt to be more comprehensive

**Changes:**
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
...
"""
```

**Pros:**
- ✅ Single LLM call per chunk (cost-efficient)
- ✅ Maintains dual-signal conflict detection
- ✅ Should achieve better coverage

**Cons:**
- ⚠️ May still have some coverage gaps
- ⚠️ Cognitive load still high for model

**Cost:** $5, 1.5 hours for full extraction

---

### Option B: Two-Pass Extraction (Extract, Then Evaluate)

**Approach:** Separate extraction from evaluation

**Pass 1 - Extract (Comprehensive):**
```python
EXTRACTION_PROMPT = """
Extract ALL relationships from this text. Be comprehensive and thorough.
Include every fact, connection, and relationship you observe.

For each relationship, provide:
- source, relationship, target
- context (where in text)
"""
```

**Pass 2 - Evaluate (Dual-Signal):**
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
- ✅ Can extract with single-signal, evaluate with dual-signal
- ✅ Catches errors while maintaining comprehensiveness

**Cons:**
- ❌ **2x API calls** = 2x cost ($10 instead of $5)
- ❌ **2x time** = 3 hours instead of 1.5 hours

**Cost:** $10, 3 hours for full extraction

---

### Option C: Hybrid Approach (Use Single-Signal + Validation)

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
- ✅ No extraction errors - just evaluating existing facts

**Cons:**
- ⚠️ Requires processing 15,201 existing relationships
- ⚠️ Additional cost for validation pass

**Cost:** ~$3 for validation pass (15,201 relationships)

---

### Option D: Targeted Dual-Signal (Smart Hybrid)

**Approach:** Use single-signal extraction, apply dual-signal only to uncertain cases

**Step 1:** Extract with single-signal (comprehensive)

**Step 2:** Apply dual-signal validation to:
- Low confidence relationships (< 0.75)
- Known error-prone patterns (biochar, geographic, type mismatches)
- Relationships flagged by automated checks

**Pros:**
- ✅ **Best ROI** - focused dual-signal where it helps most
- ✅ Comprehensive coverage
- ✅ Lower cost than full dual-signal
- ✅ Catches errors where they're most likely

**Cons:**
- ⚠️ May miss some errors in high-confidence extractions
- ⚠️ More complex pipeline

**Cost:** ~$1-2 for targeted validation

---

## 🎯 Recommendation

### Best Overall: **Option B (Two-Pass)** or **Option C (Hybrid)**

**Why Two-Pass (Option B)?**
- Guarantees comprehensive coverage (separate extraction from evaluation)
- Focused cognitive tasks = better quality
- Worth the 2x cost for quality knowledge graph
- Clear separation of concerns

**Why Hybrid (Option C)?**
- Leverages existing work (v2 extraction already done)
- Adds error detection without re-extraction
- Cost-effective ($3 vs $10)
- Can improve existing graph immediately

### Recommendation Decision Tree

```
Do you need to re-extract anyway (e.g., for different chunking)?
├─ YES → Use Option B (Two-Pass)
│         Best quality, worth the 2x cost
│
└─ NO → Use Option C (Hybrid)
          Validate existing v2 extraction
          Add dual-signal scoring to current graph
```

## 📋 Implementation Plan (Option B - Two-Pass)

### Phase 1: Create Scripts

**1. Two-pass extraction script**
```python
# scripts/extract_knowledge_graph_two_pass.py
# Pass 1: Comprehensive extraction
# Pass 2: Dual-signal evaluation
```

**2. Test on 10 episodes**
- Verify coverage matches single-signal
- Confirm dual-signal detects conflicts
- Check cost/time estimates

**3. Compare to single-signal**
- Entity pair coverage should be ~90%+
- Should detect similar conflicts as dual-signal v1
- Should maintain error detection while improving coverage

### Phase 2: Full Extraction (if test succeeds)

**Run two-pass on all 172 episodes**
- Cost: $10
- Time: 3 hours
- Output: Comprehensive graph with dual-signal scores

### Phase 3: Analysis

**Compare three approaches:**
1. Single-signal (v2)
2. Dual-signal single-pass (current test)
3. Dual-signal two-pass (new)

**Metrics:**
- Coverage (entity pairs extracted)
- Error detection (conflicts found)
- Quality (confidence scores)
- Cost-benefit

## 📋 Implementation Plan (Option C - Hybrid)

### Phase 1: Validation Script

**1. Create validation script**
```python
# scripts/validate_existing_relationships.py
# Load v2 extraction
# For each relationship: get dual-signal assessment
# Add scores to existing data
```

**2. Test on 10 episodes**
- Verify it adds useful scores
- Check conflict detection works
- Estimate cost/time

### Phase 2: Full Validation

**Run on all 15,201 relationships**
- Cost: $3
- Time: 1.5 hours
- Output: Enhanced v2 graph with dual-signal scores

### Phase 3: Filter & Fix

**Use dual-signal scores to:**
- Flag conflicts for manual review
- Identify type violations
- Prioritize correction efforts

## 🎯 Next Steps

1. **User Decision:** Which option to pursue?
   - Option A: Modified single-pass (try to fix current approach)
   - Option B: Two-pass extraction (comprehensive + evaluation)
   - Option C: Hybrid (validate existing v2 extraction)
   - Option D: Targeted validation (smart hybrid)

2. **Test Implementation**
   - Create scripts for chosen approach
   - Test on 10 episodes
   - Compare coverage and quality

3. **Make Final Decision**
   - If coverage ≥80% and error detection works → proceed
   - If not → try different option

## 📊 Expected Outcomes

### Option B (Two-Pass) Success Criteria:
- ✅ Coverage: ≥80% of single-signal entity pairs
- ✅ Error detection: Finds conflicts like dual-signal v1
- ✅ Quality: Both comprehensive AND validated

### Option C (Hybrid) Success Criteria:
- ✅ Adds useful validation scores to existing extraction
- ✅ Identifies conflicts and type violations
- ✅ Costs < $5
- ✅ Improves existing graph without re-extraction

---

**Status:** Awaiting user decision on which approach to implement
**Estimated Total Cost:**
- Option A: $5 (re-run with fixed prompt)
- Option B: $10 (two-pass extraction)
- Option C: $3 (validate existing)
- Option D: $1-2 (targeted validation)
