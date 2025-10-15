# V14 Root Cause Analysis - Manual Pattern Analysis

**Date**: 2025-10-14
**Test Results**: 3/17 PASS (17.6%), 14/17 FAIL
**Analyst**: Manual analysis based on test patterns

---

## Executive Summary

V14 improvements achieved **excellent filtering** (3/3 REJECT tests passed) but suffered from **over-aggressive filtering** causing 14/17 failures. The root issue is V14's confidence penalties combined with the p_true >= 0.5 threshold filter out valid relationships.

---

## Pipeline Flow Analysis

```
Pass 1 (Extract)    →   Pass 2 (Evaluate)   →   Pass 2.5 (Postprocess)
42 relationships    →   44 relationships    →   20 relationships (52.4% loss)
```

**Key Observation**: 52.4% of extracted relationships are filtered out, suggesting overly strict filtering.

---

## Root Causes by Pattern

### 1. ✅ **SUCCESSES - Filtering works perfectly** (3 tests)

**TC003**: Vague Abstract Entity
- V13.1: `(soil stewardship) --[enhances]--> (aspects of life)`
- V14: ✅ Correctly **filtered** "aspects of life" (vague abstract)
- **Why it worked**: V14's entity specificity constraints detected vague target

**TC011**: Philosophical Claim - High Confidence
- V13.1: `(soil) --[is]--> (cosmically sacred)`
- V14: ✅ Correctly **filtered** philosophical/metaphysical claim
- **Why it worked**: V14's classification_flags detected PHILOSOPHICAL_CLAIM

**TC014**: Normative Statement as Factual
- V13.1: `(humanity) --[must]--> (thrive and heal)`
- V14: ✅ Correctly **filtered** normative/prescriptive statement
- **Why it worked**: V14's NORMATIVE flag + confidence penalty

---

### 2. ❌ **ROOT CAUSE #1: Over-Aggressive Confidence Penalties** (Estimated 8-10 failures)

**Pattern**: Many failed tests showed `Pass 2.5: 0 relationships` - everything filtered out.

**Evidence from test logs**:
- [1/17] Pass 2.5: 1 relationship (from 2 in Pass 2) → 50% filtered
- [2/17] Pass 2.5: 0 relationships (from 1 in Pass 2) → 100% filtered
- [4/17] Pass 2.5: 0 relationships (from 2 in Pass 2) → 100% filtered
- [6/17] Pass 2.5: 0 relationships (from 2 in Pass 2) → 100% filtered
- [7/17] Pass 2.5: 0 relationships (from 2 in Pass 2) → 100% filtered
- [8/17] Pass 2.5: 0 relationships (from 2 in Pass 2) → 100% filtered
- [9/17] Pass 2.5: 0 relationships (from 3 in Pass 2) → 100% filtered
- [10/17] Pass 2.5: 0 relationships (from 1 in Pass 2) → 100% filtered

**Likely Cause**: V14's confidence penalties are too severe:
- TESTABLE_CLAIM: -0.15
- OPINION: -0.25
- NORMATIVE: -0.30
- PHILOSOPHICAL_CLAIM: -0.50

These penalties push many valid relationships below p_true = 0.5 threshold.

**Affected Test Categories**:
- TC012: Philosophical Claim - Abstract (FAILED)
  - Expected: `(soil management) --[can mitigate]--> (climate change)`
  - Issue: Likely assigned TESTABLE_CLAIM flag, p_true reduced below 0.5

- TC013: Testable Claim Without Confidence Adjustment (FAILED)
  - Expected: `(getting hands in soil) --[may enhance]--> (immune systems)`
  - Issue: TESTABLE_CLAIM penalty applied, but relationship is valid

- TC017: Opinion Statement as Fact (FAILED)
  - Expected: `(living soil) --[may enhance]--> (cognitive performance)`
  - Issue: OPINION penalty too severe for hedged claims

---

### 3. ❌ **ROOT CAUSE #2: Postprocessing Modules Not Triggering** (Estimated 4-6 failures)

**Pattern**: Expected corrections (pronoun resolution, predicate normalization) didn't happen.

**TC001**: Praise Quote False Positive (FAILED) - HIGH severity
- V13.1: `(Soil Stewardship Handbook) --[authored by]--> (Aaron William Perry)`
- Expected V14: `(Aaron William Perry) --[authored]--> (Soil Stewardship Handbook)`
- **Issue**: praise_quote_detector.py didn't correct direction
- **Root cause**: Copyright exclusion patterns may not match actual text

**TC002**: Reversed Endorsement Direction (FAILED) - HIGH severity
- V13.1: `(Soil Stewardship Handbook) --[endorsed by]--> (Lily Sophia von Übergarten)`
- Expected V14: `(Lily Sophia von Übergarten) --[wrote foreword for]--> (Soil Stewardship Handbook)`
- **Issue**: bibliographic_citation_parser.py didn't swap direction
- **Root cause**: Endorsement validation logic not triggering

**TC004**: Vague Demonstrative Pronoun (FAILED) - HIGH severity
- V13.1: `(this) --[enables]--> (sustainable farming practices)`
- Expected V14: `(soil stewardship) --[enables]--> (sustainable farming practices)`
- **Issue**: pronoun_resolver.py didn't resolve "this"
- **Root cause**: Demonstrative pronouns in FORBIDDEN_PATTERNS may block extraction entirely

**TC005-006**: Predicate Fragmentation (FAILED) - MEDIUM severity
- TC005: "is" vs "is-a" not normalized
- TC006: Tense variations ("preserved" vs "preserves") not normalized
- **Issue**: predicate_normalizer.py not converting properly
- **Root cause**: Normalization patterns may be too narrow

**TC007-010**: Unresolved Pronouns (FAILED) - MEDIUM severity
- TC007-008: Possessive pronouns ("my people" → "Slovenians")
- TC009-010: Generic pronouns ("we" → "humanity", "you" → "humans")
- **Issue**: pronoun_resolver.py failed to resolve
- **Root cause**: Insufficient context or missing nationality metadata

---

### 4. ❌ **ROOT CAUSE #3: Extraction Too Conservative** (Estimated 2-3 failures)

**Pattern**: Some expected relationships never extracted in Pass 1.

**TC015**: Semantic Predicate Mismatch (FAILED) - MILD severity
- Expected: `(soil degradation) --[threatens]--> (humanity)`
- **Issue**: May not have been extracted due to V14's semantic validation
- **Root cause**: FORBIDDEN_ENTITY_PATTERNS or specificity constraints too strict

**TC016**: Incomplete List Splitting (FAILED) - MILD severity
- Expected: `(agricultural soils) --[restores]--> (productively vital states)`
- **Issue**: Complex multi-word target may be flagged as vague
- **Root cause**: Entity specificity scoring too harsh on compound phrases

---

## Quantitative Root Cause Breakdown

Based on pattern analysis:

| Root Cause | Est. Tests Affected | % of Failures |
|-----------|---------------------|---------------|
| **Over-aggressive confidence penalties** (Pass 2) | 8-10 | 57-71% |
| **Postprocessing modules not triggering** (Pass 2.5) | 4-6 | 29-43% |
| **Extraction too conservative** (Pass 1) | 2-3 | 14-21% |

**Primary Issue**: V14's confidence penalties (Pass 2) are the dominant failure mode.

---

## Recommendations for V14.1 (Next ACE Cycle)

### Priority 1: Adjust Confidence Penalties

**Current V14 penalties**:
```
TESTABLE_CLAIM: -0.15
OPINION: -0.25
NORMATIVE: -0.30
PHILOSOPHICAL_CLAIM: -0.50
```

**Recommended V14.1 penalties**:
```
TESTABLE_CLAIM: -0.05  (reduce from -0.15)
OPINION: -0.10  (reduce from -0.25)
NORMATIVE: -0.15  (reduce from -0.30)
PHILOSOPHICAL_CLAIM: -0.30  (reduce from -0.50)
```

**Rationale**: Hedged claims like "may enhance" or "can mitigate" should be preserved with moderate confidence, not heavily penalized.

### Priority 2: Fix Postprocessing Triggers

1. **praise_quote_detector.py**: Test copyright exclusion patterns against actual V13.1 page 6 text
2. **bibliographic_citation_parser.py**: Debug why endorsement direction validation doesn't trigger
3. **pronoun_resolver.py**: Add fallback for demonstrative pronouns when resolution fails
4. **predicate_normalizer.py**: Expand normalization patterns for tense and verb forms

### Priority 3: Relax Entity Specificity Constraints

1. Allow compound phrases like "productively vital states" (not inherently vague)
2. Don't auto-reject demonstrative pronouns - try to resolve first
3. Reduce entity_specificity_score threshold from 0.90 to 0.80 (if scoring is used in Pass 1)

### Priority 4: Alternative Approach - Adjust p_true Threshold

Instead of modifying penalties, consider lowering the final filtering threshold:
- **Current**: p_true >= 0.5
- **Alternative**: p_true >= 0.4

**Trade-off**: This would preserve more relationships but may increase issue rate slightly.

---

## Success Metrics for V14.1

To achieve **A-grade** (issue rate < 5%):

1. **Test pass rate**: ≥ 90% on V13.1 issue tests (currently 17.6%)
2. **Maintain filtering successes**: Still correctly filter vague/philosophical/normative relationships
3. **Enable corrections**: Postprocessing modules successfully apply fixes
4. **Balanced extraction**: Extract valid relationships without being over-conservative

**Target pipeline reduction**: 30-40% (currently 52.4%)
- Pass 1: 42 relationships
- Pass 2: ~38 relationships (10% reduction)
- Pass 2.5: ~28 relationships (30% reduction total)

---

## Next Steps

1. ✅ **Create V14.1 changeset** based on these recommendations
2. **Test V14.1** on V13.1 issues (target: 90%+ pass rate)
3. **Run full V14.1 extraction** if tests pass
4. **Run Reflector on V14.1** to measure actual improvement

---

## Appendix: Test-by-Test Analysis

### HIGH Severity Failures (1/4 passed, 3 failed)

| Test ID | Category | Status | Root Cause |
|---------|----------|--------|-----------|
| TC001 | Praise Quote False Positive | FAIL | Postprocessing not triggering |
| TC002 | Reversed Endorsement Direction | FAIL | Postprocessing not triggering |
| TC003 | Vague Abstract Entity | **PASS** | ✅ Filtering worked |
| TC004 | Vague Demonstrative Pronoun | FAIL | Pronoun resolution failed |

### MEDIUM Severity Failures (2/10 passed, 8 failed)

| Test ID | Category | Status | Root Cause |
|---------|----------|--------|-----------|
| TC005 | Predicate Fragmentation - is-a vs is | FAIL | Normalization not working |
| TC006 | Predicate Fragmentation - Tense | FAIL | Normalization not working |
| TC007 | Unresolved Possessive Pronoun | FAIL | Resolution failed |
| TC008 | Unresolved Possessive Pronoun | FAIL | Resolution failed |
| TC009 | Unresolved Generic Pronoun | FAIL | Resolution failed |
| TC010 | Unresolved Generic Pronoun | FAIL | Resolution failed |
| TC011 | Philosophical Claim - High Confidence | **PASS** | ✅ Filtering worked |
| TC012 | Philosophical Claim - Abstract | FAIL | Over-aggressive penalty |
| TC013 | Testable Claim Without Adjustment | FAIL | Over-aggressive penalty |
| TC014 | Normative Statement as Factual | **PASS** | ✅ Filtering worked |

### MILD Severity Failures (0/3 passed, 3 failed)

| Test ID | Category | Status | Root Cause |
|---------|----------|--------|-----------|
| TC015 | Semantic Predicate Mismatch | FAIL | Extraction too conservative |
| TC016 | Incomplete List Splitting | FAIL | Extraction too conservative |
| TC017 | Opinion Statement as Fact | FAIL | Over-aggressive penalty |

---

**End of Manual Root Cause Analysis**
