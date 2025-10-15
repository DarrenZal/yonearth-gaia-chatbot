# V14 Performance Analysis - Input for Curator

**Analysis Date**: 2025-10-14
**Source Version**: V13.1 (Grade: B/B-, 14.5% issue rate)
**Target Version**: V14 (tested, not deployed)
**Target Grade**: A (< 5% issue rate)

---

## Executive Summary

V14 was tested on 17 specific problematic cases from V13.1. Results show **excellent filtering** of invalid relationships (3/3 REJECT tests passed) but **over-aggressive filtering** of valid relationships (14/17 tests failed, 17.6% pass rate).

**Key Finding**: V14's confidence penalties are too severe, causing valid hedged claims and testable statements to be filtered out.

---

## V14 Test Results

### Overall Performance
- **Pass Rate**: 3/17 (17.6%)
- **Fail Rate**: 14/17 (82.4%)
- **Pipeline Reduction**: 52.4% (42 → 20 relationships)

### By Severity
| Severity | Passed | Total | Pass Rate |
|----------|--------|-------|-----------|
| HIGH     | 1      | 4     | 25%       |
| MEDIUM   | 2      | 10    | 20%       |
| MILD     | 0      | 3     | 0%        |

### Successes (What Worked)
✅ **TC003**: Filtered "aspects of life" (vague abstract)
✅ **TC011**: Filtered "cosmically sacred" (philosophical)
✅ **TC014**: Filtered "thrive and heal" (normative)

**Analysis**: V14's entity specificity constraints and classification_flags WORK PERFECTLY for rejecting invalid relationships.

### Failures (What Didn't Work)

#### Category 1: Over-Aggressive Confidence Penalties (8-10 cases)
- TC012: "soil management can mitigate climate change" → FILTERED (likely TESTABLE_CLAIM penalty)
- TC013: "getting hands in soil may enhance immune systems" → FILTERED (TESTABLE_CLAIM penalty)
- TC017: "living soil may enhance cognitive performance" → FILTERED (OPINION penalty)

**Pattern**: Many chunks showed Pass 2.5: 0 relationships (everything filtered out)

**Evidence**:
- [2/17] Pass 2.5: 0 relationships (from 1 in Pass 2) → 100% filtered
- [4/17] Pass 2.5: 0 relationships (from 2 in Pass 2) → 100% filtered
- [6/17] Pass 2.5: 0 relationships (from 2 in Pass 2) → 100% filtered
- [7/17] Pass 2.5: 0 relationships (from 2 in Pass 2) → 100% filtered

#### Category 2: Postprocessing Not Correcting (4-6 cases)
- TC001: Copyright statement not excluded from praise detection
- TC002: Endorsement direction not swapped
- TC004: Demonstrative pronoun "this" not resolved
- TC005-006: Predicates not normalized ("is" vs "is-a", tense variations)
- TC007-010: Possessive/generic pronouns not resolved

#### Category 3: Extraction Too Conservative (2-3 cases)
- TC015: "soil degradation threatens humanity" not extracted
- TC016: "agricultural soils restores productively vital states" flagged as vague

---

## Root Cause Analysis

### Root Cause #1: Confidence Penalties Too Severe (PRIMARY ISSUE)

**V14 Penalties**:
```
TESTABLE_CLAIM: -0.15
OPINION: -0.25
NORMATIVE: -0.30
PHILOSOPHICAL_CLAIM: -0.50
```

**Problem**: These penalties push valid hedged claims below p_true = 0.5 threshold.

**Examples of Valid Relationships Being Filtered**:
1. "soil management **can mitigate** climate change" → TESTABLE_CLAIM (-0.15) → p_true < 0.5
2. "getting hands in soil **may enhance** immune systems" → TESTABLE_CLAIM (-0.15) → p_true < 0.5
3. "living soil **may enhance** cognitive performance" → OPINION (-0.25) → p_true < 0.5

**Key Insight**: Hedged claims with "may", "can", "could" should be PRESERVED with moderate confidence, not heavily penalized.

### Root Cause #2: Postprocessing Modules Not Triggering

**Issue**: V14 code modules were updated but not being invoked correctly OR patterns not matching.

**Specific Failures**:
1. praise_quote_detector.py: Copyright exclusion patterns may not match actual text on page 6
2. bibliographic_citation_parser.py: Endorsement direction validation not triggering
3. pronoun_resolver.py: Demonstrative pronouns blocked entirely instead of attempted resolution
4. predicate_normalizer.py: Normalization patterns too narrow (missing tense variations)

### Root Cause #3: Entity Specificity Too Strict

**Issue**: V14's forbidden patterns and specificity scoring too conservative.

**Examples**:
- "productively vital states" flagged as vague (compound phrases penalized)
- "this" auto-rejected instead of attempting resolution
- Threshold of 0.90 may be too high

---

## Recommendations for V14.1

### Priority 1: Adjust Confidence Penalties (CRITICAL)

**Recommended Reductions** (50-67% reduction):
```
TESTABLE_CLAIM: -0.05  (reduce from -0.15)  ← Preserve hedged scientific claims
OPINION: -0.10  (reduce from -0.25)         ← Preserve hedged opinions
NORMATIVE: -0.15  (reduce from -0.30)       ← Preserve prescriptive statements with context
PHILOSOPHICAL_CLAIM: -0.30  (reduce from -0.50)  ← Still heavily penalize but not eliminate
```

**Rationale**:
- Hedged claims ("may", "can", "could") should have p_true ~0.4-0.6, not <0.3
- Let postprocessing and final threshold do the heavy lifting
- V14's classification is working; penalties are the problem

**Alternative**: Lower p_true threshold from 0.5 to 0.4 (keep penalties as-is)

### Priority 2: Fix Postprocessing Triggers

**Not prompt changes - these are code/config issues**:
1. Debug why praise_quote_detector copyright patterns don't match
2. Check bibliographic_citation_parser endorsement validation logic
3. Allow demonstrative pronoun resolution attempts before rejection
4. Expand predicate_normalizer patterns

**For Curator**: Note that these are code fixes, not prompt changes. Prompts should still guide extraction quality, but mention that postprocessing should handle corrections.

### Priority 3: Relax Entity Specificity Constraints

**Forbidden Pattern Refinement**:
- Don't auto-reject demonstratives; extract them and let pronoun_resolver fix
- Allow compound phrases like "productively vital states"
- Consider context: "this" in "this enables sustainable farming" may be resolvable

**Entity Specificity Score**:
- Reduce threshold from 0.90 to 0.85 (if used in Pass 1)
- Or remove entity_specificity_score from Pass 1 entirely (keep for Pass 2 only)

---

## Success Metrics for V14.1

To achieve **A-grade** (< 5% issue rate):

1. **Test Pass Rate**: ≥ 90% on V13.1 issue tests (currently 17.6%)
2. **Maintain Filtering**: Still correctly filter truly vague/philosophical/normative (3/3)
3. **Enable Preservation**: Don't over-filter hedged scientific claims
4. **Balanced Reduction**: Target 30-40% pipeline reduction (currently 52.4%)

**Expected Pipeline for V14.1**:
```
Pass 1: 42 relationships (same as V14)
Pass 2: ~38 relationships (10% reduction, less aggressive penalties)
Pass 2.5: ~28 relationships (30% total reduction)
```

---

## V14 Prompt Context

### V14 Pass 1 Enhancements (Added)
- Entity specificity constraints with FORBIDDEN_ENTITY_PATTERNS
- Semantic predicate validation
- Demonstrative pronoun blocking

### V14 Pass 2 Enhancements (Added)
- Confidence scoring rules with penalties
- Philosophical/rhetorical claim filtering (p_true < 0.3)
- classification_flags: FACTUAL, TESTABLE_CLAIM, PHILOSOPHICAL_CLAIM, NORMATIVE, OPINION

---

## Curator Instructions

Please generate V14.1 improvements that:

1. **REDUCE confidence penalties by 50-67%** as specified in Priority 1
2. **Maintain classification_flags** (they work correctly)
3. **Refine entity specificity constraints** to allow resolvable demonstratives
4. **Balance filtering vs preservation** (goal: preserve valid hedged claims)
5. **Keep all other V14 improvements** (semantic validation, contextual flags)

**Key Principle**: V14's **detection** is excellent. The problem is **penalty severity**. Adjust penalties while keeping the classification logic.

---

## Test Cases for V14.1 Validation

Must pass ≥15/17 tests (90%+) to proceed with full extraction.

**Critical Tests to Fix**:
- TC012-013: Testable claims with hedging ("may", "can")
- TC017: Opinion with hedging
- TC001-002: Postprocessing corrections (HIGH severity)
- TC004: Demonstrative pronoun resolution (HIGH severity)

**Must Still Pass** (filtering):
- TC003: Vague abstract entity
- TC011: Philosophical claim
- TC014: Normative statement

---

**End of Analysis**
