# V14.3 Results and Next Steps

**Date:** 2025-10-15 01:10 UTC
**Status:** Phase 1 Complete - B Grade Achieved
**Next Phase:** V14.3.1 Prompt Enhancement (Target: A- Grade)

---

## QUICK SUMMARY

**V14.3 Result:** **B grade (16.2% issue rate)** instead of expected A- (8.6%)

**Root Cause:** LLM stochastic variance - fresh extraction produced 738 candidates vs V13.1's 861 checkpoint candidates (14% fewer)

**Assessment:** ✅ **SUCCESS** - Huge improvement from V14.2's catastrophic C+ (27.3%), validates V12+V13.1 configuration

**Recommendation:** Proceed with **V14.3.1** - enhance Pass 1 prompt to reduce subjective content extraction

---

## V14.3 RESULTS BREAKDOWN

### Configuration Used

```
Pass 1:  V12 prompt (fresh extraction)
Pass 2:  V13.1 prompt (dual-signal evaluation)
Pass 2.5: 12 modules (V13 pipeline)
```

### Extraction Statistics

| Metric | V14.3 | V13.1 | Difference |
|--------|-------|-------|------------|
| **Pass 1 Candidates** | 738 | 861 (checkpoint) | -123 (-14%) |
| **Final Relationships** | 722 | 873 | -151 (-17%) |
| **High Confidence** | 584 (80.9%) | 703 (80.5%) | Similar |
| **Total Issues** | 117 | 75 | +42 |
| **Issue Rate** | 16.2% | 8.6% | +7.6pp |
| **Grade** | **B** | **A-** | -1 letter |

### Quality Breakdown

| Severity | Count | % of 722 | Top Categories |
|----------|-------|----------|----------------|
| **Critical** | 0 | 0% | ✅ None |
| **High** | 12 | 1.7% | Unresolved pronouns (12) |
| **Medium** | 56 | 7.8% | Vague entities (38), Broad predicates (18) |
| **Mild** | 49 | 6.8% | Philosophical content (67), Praise quotes (11) |
| **TOTAL** | **117** | **16.2%** | **Mixed** |

---

## KEY FINDINGS

### ✅ Strengths (What Worked)

1. **Zero critical factual errors** - No reversed authorship, wrong facts, or impossible relationships
2. **Excellent deduplication** - Zero duplicate relationships
3. **Good predicate normalization** - 119 unique predicates (well below 150 threshold)
4. **Effective content flagging** - System correctly identifies subjective content (PHILOSOPHICAL_CLAIM, FIGURATIVE_LANGUAGE)
5. **Strong bibliographic parsing** - Correct author, publisher, date extraction
6. **Validated configuration** - Proves V12 + V13.1 + 12 modules works

### ⚠️ Weaknesses (What Needs Improvement)

1. **67 MILD philosophical/metaphorical relationships (9.3%)** - System flags but doesn't filter
2. **38 MEDIUM vague entities (5.3%)** - "experience", "framework", "impact" without context
3. **18 MEDIUM overly broad predicates (2.5%)** - "can", "helps", "makes" without specificity
4. **12 HIGH unresolved pronouns (1.7%)** - "we", "us" not resolved to "humanity"
5. **Fresh extraction variance** - 14% fewer candidates than V12 checkpoint

---

## ROOT CAUSE ANALYSIS

### Why V14.3 Got B Instead of A-

**Discovery:** V13.1 didn't run fresh Pass 1 extraction - it used a **saved V12 checkpoint**!

```json
// V13.1 metadata
"source": "V12 Pass 1 checkpoint (861 candidates)",
"v12_checkpoint": ".../book_soil_handbook_v12_20251014_044425_pass1_checkpoint.json"
```

V14.3 ran a **fresh** V12 Pass 1 extraction and got different results:
- **V12 Checkpoint:** 861 candidates (deterministic - saved file)
- **V14.3 Fresh:** 738 candidates (stochastic - LLM variance)

### LLM Variance Impact

**Same prompt, different extraction:**
- Temperature sampling introduces randomness
- Attention patterns vary slightly between runs
- Time-based API behavior variation
- Long prompts may be interpreted differently

**Result:** Fresh extraction picked up more philosophical/aspirational content, fewer factual relationships

---

## COMPARISON: V14.2 vs V14.3 vs V13.1

| Version | Config | Grade | Issue Rate | Key Problem |
|---------|--------|-------|------------|-------------|
| **V14.2** | V14 + V14 + V13.1 | **C+** | 27.3% | ❌ Predicate fragmentation (100 issues) |
| **V14.3** | V12 + V13.1 + V13.1 | **B** | 16.2% | ⚠️ Philosophical content (67 issues) |
| **V13.1** | V12 ckpt + V13.1 + V13.1 | **A-** | 8.6% | ✅ Baseline (checkpoint) |

**Progress:** V14.2 (C+) → V14.3 (B) = **-11.1pp improvement** in one cycle!

---

## STRATEGIC OPTIONS

### Option A: Use V12 Checkpoint (Guaranteed A-)

**Approach:** Load V12 checkpoint instead of running Pass 1

**Pros:**
- ✅ Guaranteed A- grade (identical to V13.1 inputs)
- ✅ Fast iteration (skip 20-min Pass 1 extraction)

**Cons:**
- ❌ Not a "true" fresh extraction
- ❌ Doesn't validate robustness to variance
- ❌ Reuses old data

**Use Case:** Quick validation that Pass 2/2.5 changes work

---

### Option B: Accept Variance, Improve Prompts (Recommended)

**Approach:** Acknowledge LLM variance, build robust prompts that handle it

**Pros:**
- ✅ Realistic - production will have variance
- ✅ Forces robust prompt engineering
- ✅ Validates system works with different inputs

**Cons:**
- ❌ Requires 2-3 more cycles to reach A-
- ❌ Can't guarantee exact A- match

**Use Case:** Build production-ready system that handles variance

---

## RECOMMENDATION: V14.3.1 PROMPT ENHANCEMENT

**Goal:** Reach A- grade (8.6%) on fresh extraction by improving Pass 1 prompt

### Changes for V14.3.1

**1. Enhance V12 Pass 1 Prompt** (Highest Impact: -11.4%)

Add three constraints:

```
CONSTRAINT 1: Extract only FACTUAL relationships about what IS, not aspirational
statements about what COULD BE or SHOULD BE. Exclude philosophical claims,
metaphors, and motivational language unless they convey verifiable facts.

CONSTRAINT 2: Entities must be specific and unambiguous. Include disambiguating
context: use "living soil" not "soil", "human health" not "health",
"environmental impact" not "impact". Avoid abstract nouns like "experience",
"framework", "activities" unless qualified.

CONSTRAINT 3: Do not extract relationships with pronouns (we, us, our, it) as
source or target. Resolve pronouns to antecedents before extraction. If
antecedent unclear, use generic term ("humanity" for "we" in philosophical contexts).
```

**Expected Impact:**
- Aspirational content: 67 → ~10 (-8.0%)
- Vague entities: 38 → ~15 (-3.2%)
- Unresolved pronouns: 12 → ~2 (-1.4%)
- **Total: -12.6%** issue rate

**2. Add Pass 2.5 Config Filtering** (Medium Impact: -9.8%)

```yaml
# config/extraction_config.yaml
filter_subjective_content: true
filter_rules:
  - condition: "p_true < 0.5 AND (PHILOSOPHICAL_CLAIM OR FIGURATIVE_LANGUAGE OR METAPHOR)"
    action: exclude
  - condition: "OPINION OR SEMANTIC_INCOMPATIBILITY"
    action: exclude
```

**Expected Impact:**
- Philosophical content: 67 → 0 (-9.3%)
- Opinion statements: 3 → 0 (-0.4%)
- Semantic mismatches: 1 → 0 (-0.1%)
- **Total: -9.8%** issue rate

**3. Enhance PronounResolver Module** (Low Impact: -1.7%)

```python
# modules/pass2_5_postprocessing/pronoun_resolver.py
FALLBACK_RESOLUTIONS = {
    'we': 'humanity',
    'us': 'humanity',
    'our': 'human'
}
# Apply when no antecedent found within 2 sentences
```

**Expected Impact:**
- Unresolved pronouns: 12 → 0 (-1.7%)

### Combined Expected Result

**V14.3 Current:** 16.2% issue rate (B grade)

**V14.3.1 Estimated:**
- Prompt enhancement: -12.6%
- Config filtering: -9.8%
- Pronoun resolver: -1.7%
- **Overlap adjustment:** +5-7% (some fixes overlap)

**Final Estimate:** **5-8% issue rate (A- to A grade)**

---

## IMPLEMENTATION PLAN

### Phase 1.5: V14.3.1 Prompt Enhancement (1-2 hours)

**Tasks:**
1. ✏️ Update V12 Pass 1 prompt with 3 new constraints
2. ✏️ Add few-shot examples showing factual vs aspirational content
3. 🔧 Create Pass 2.5 config filtering module
4. 🔧 Enhance PronounResolver with fallback rules
5. ▶️ Run V14.3.1 extraction (~40 minutes)
6. 🔍 Run Reflector analysis
7. 📊 Compare V14.3.1 vs V14.3 vs V13.1

**Success Criteria:** A- grade (5-9% issue rate) or better

**If Successful:** Proceed to Phase 2 (enhance PredicateNormalizer)

**If Not:** Analyze remaining issues, iterate on prompts

---

### Phase 2: Predicate Normalization (Post A-)

Once A- achieved, enhance predicate quality:
1. Add context-aware normalization rules
2. 'can' + action → 'can [action]'
3. 'helps' + context → more specific verb
4. Target: A grade (5-8% issue rate)

---

### Phase 3: Advanced Filtering (Post A)

Add new modules:
1. PhilosophicalClaimFilter (reduce philosophical content)
2. MetadataFilter (improve entity typing)
3. ConfidenceFilter (threshold on p_true scores)
4. Target: A+ grade (<5% issue rate)

---

## FILES CREATED

1. **V14.3 Extraction Script**
   - `scripts/extract_kg_v14_3_book.py`
   - Full V13.1 configuration (V12 + V13.1 + 12 modules)

2. **V14.3 Extraction Output**
   - `kg_extraction_playbook/output/v14_3/soil_stewardship_handbook_v14_3.json`
   - 722 relationships, 584 high confidence (80.9%)

3. **V14.3 Reflector Analysis**
   - `kg_extraction_playbook/analysis_reports/reflection_v14.3_20251015_010510.json`
   - Comprehensive quality analysis with recommendations

4. **V14.3 LLM Variance Analysis**
   - `kg_extraction_playbook/V14_3_LLM_VARIANCE_ANALYSIS.md`
   - Root cause investigation and strategic options

5. **V14.3 Reflector Script**
   - `scripts/run_reflector_on_v14_3.py`
   - Automated quality analysis for V14.3

6. **Extraction Log**
   - `kg_v14_3_extraction.log`
   - Complete extraction trace (39.6 minutes)

---

## CONCLUSION

### What We Learned

1. **V14.3 validates the V12 + V13.1 configuration** - B grade (16.2%) is excellent progress from V14.2's C+ (27.3%)
2. **LLM variance is real and significant** - Same prompt can produce 14% different candidate counts
3. **V13.1's A- grade was partially luck** - Used a good V12 checkpoint, not fresh extraction
4. **Checkpoints vs Fresh:** Checkpoints guarantee consistency, fresh extractions test robustness
5. **B grade is not failure** - Clear path to A- through prompt enhancement

### Next Steps

**RECOMMENDED:** Implement **V14.3.1** with enhanced Pass 1 prompt

**Expected Outcome:** A- grade (5-8% issue rate) on fresh extraction

**Timeline:** 2-3 hours total
- 1-2 hours: Prompt enhancement + module updates
- 40 minutes: Extraction run
- 5 minutes: Reflector analysis

**If V14.3.1 achieves A-:** Proceed to Phase 2 (enhance PredicateNormalizer for A grade)

**If V14.3.1 achieves B+:** Iterate on prompt constraints, retry

**Target:** A+ grade (<5% issue rate) within 3-4 more cycles

---

## APPENDIX: Reflector Top Recommendations

1. **HIGH - Prompt Enhancement:** Add factual-only constraint to Pass 1 (-11.4% impact)
2. **HIGH - Config Filtering:** Exclude flagged subjective content (-9.8% impact)
3. **HIGH - Pronoun Resolution:** Add fallback rules for abstract contexts (-1.7% impact)
4. **MEDIUM - Entity Specificity:** Require disambiguating context in Pass 1
5. **MEDIUM - Predicate Normalization:** Context-aware rules for vague predicates

**Total Estimated Impact of Top 3:** -23% issue rate → **5-8% final issue rate (A- to A grade)**
