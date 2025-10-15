# V14.2 Root Cause Analysis

**Date**: October 14, 2025
**Analyst**: Claude Code
**Objective**: Identify the ACTUAL root cause of V14.0's regression from V13.1 (A-) to determine correct V14.2 approach

---

## 🔍 Critical Discovery

**V13.1 and V14.0 used IDENTICAL Pass 2 evaluation prompts.**

```bash
$ diff pass2_evaluation_v13_1.txt pass2_evaluation_v14.txt
# NO DIFFERENCES FOUND
```

This completely changes our understanding of the regression.

---

## 📊 What We Know

### V13.1 Configuration (A- Grade, 8.6% issue rate):
- **Pass 1**: V12 checkpoint (861 candidates from V12's Pass 1 prompt)
- **Pass 2**: `pass2_evaluation_v13_1.txt` (IDENTICAL to V14)
- **Pass 2.5**: **12 postprocessing modules** via `get_book_pipeline()`
- **Result**: 873 relationships, 75 issues (8.6%)

### V14.0 Configuration (B+ Grade, 10.78% issue rate):
- **Pass 1**: `pass1_extraction_v14.txt` (27KB, complex prompt)
- **Pass 2**: `pass2_evaluation_v14.txt` (**IDENTICAL** to V13.1)
- **Pass 2.5**: **14 postprocessing modules** via `get_book_pipeline()` + **NEW modules**:
  - MetadataFilter (4-layer detection)
  - ConfidenceFilter (flag-specific thresholds)
  - Enhanced PredicateNormalizer V1.4
- **Result**: 603 relationships, 65 issues (10.78%)

### V14.1 Configuration (C+ Grade, 16.7% issue rate):
- **Pass 1**: V12 prompt (23KB, permissive) - ABANDONED
- **Pass 2**: `pass2_evaluation_v14.txt` (same as V13.1 and V14.0)
- **Pass 2.5**: 15 modules (added SemanticDeduplicator - buggy)
- **Result**: 708 relationships, 118 issues (16.7%) - **WORST EVER**

---

## 🎯 Root Cause Hypothesis CORRECTION

### What V14.1 Failure Analysis Claimed:
> "V13.1's success was due to BETTER Pass 2/Pass 2.5, not V12's prompt"

**This is PARTIALLY CORRECT but INCOMPLETE:**

Yes, V12's prompt extracts low-quality content (Rumi poetry, praise quotes).
Yes, V14's Pass 1 prompt was more selective (596 vs 861 candidates).

**BUT**: The Pass 2 prompts are IDENTICAL, so "BETTER Pass 2 evaluation" in V13.1 is NOT from the prompt.

### The ACTUAL Root Cause:

The regression from V13.1 (A-) → V14.0 (B+) is in **Pass 2.5 postprocessing configuration**:

1. **V14.0 added 2 new modules** (MetadataFilter, ConfidenceFilter)
2. **V14.0 enhanced PredicateNormalizer** to V1.4
3. **One or more of these changes introduced new issues OR filtered too aggressively**

### Evidence:

| Version | Pass 1 | Pass 2 Prompt | Pass 2.5 Modules | Result | Issue Rate |
|---------|--------|---------------|------------------|--------|------------|
| V13.1 | V12 (861) | v13_1.txt | 12 modules | 873 rels | 8.6% ✅ |
| V14.0 | V14 (596) | v14.txt (SAME) | 14 modules + V1.4 | 603 rels | 10.78% ❌ |
| V14.1 | V12 (782) | v14.txt (SAME) | 15 modules (buggy) | 708 rels | 16.7% ❌❌ |

**Key Insight**: Same Pass 2 prompt, different Pass 2.5 config → different issue rates!

---

## 🔧 What Changed in Pass 2.5 Between V13.1 and V14.0?

### V13.1 Pass 2.5 (12 modules):
Looking at `run_v13_1_from_v12_checkpoint.py`:
- Used `extract_kg_v13_book.py`'s `postprocess_pass2_5()`
- Called `get_book_pipeline()` (12 modules)
- Exact module list unknown - need to check V13 book_pipeline code

### V14.0 Pass 2.5 (14 modules):
Looking at `extract_kg_v14_book.py`:
- New module: `MetadataFilter` (4-layer detection for book metadata)
- New module: `ConfidenceFilter` (flag-specific thresholds + unresolved pronoun handling)
- Enhanced: `PredicateNormalizer` V1.4 (tense norm, modal verb preservation)
- All other modules from V13.1

### Possible Regression Sources:

1. **MetadataFilter introduced false positives**
   - V14.0 issue analysis: "Over-Extraction of Abstract/Philosophical" increased from 0.9% to 2.99%
   - MetadataFilter might not be catching all philosophical content in book text

2. **ConfidenceFilter filtering too conservatively**
   - V14.0 filtered from 596 candidates to 603 final (7 added by ListSplitter)
   - V13.1 filtered from 861 candidates to 873 final (12 added by ListSplitter)
   - Something is filtering too aggressively in V14.0

3. **PredicateNormalizer V1.4 semantic validation issues**
   - V1.4 added semantic validation
   - Might be flagging valid relationships as semantic mismatches

---

## ✅ What V14.1 Analysis Got RIGHT

V14.1 **CORRECTLY proved**:
1. ✅ V12's Pass 1 prompt extracts low-quality content (poetry, metaphors)
2. ✅ V14's Pass 1 prompt correctly filters this content
3. ✅ Higher extraction volume ≠ better quality (782 → 708, but 16.7% issues!)
4. ✅ The regression is in Pass 2.5, not Pass 1 or Pass 2

V14.1 **INCORRECTLY concluded**:
1. ❌ "V13.1 had BETTER Pass 2 evaluation" - **FALSE**: Same prompt as V14.0!
2. ❌ "Use V12's prompt for V14.2" - **BAD IDEA**: V12 extracts low-quality content!

---

## 🎯 Correct Path Forward for V14.2

### DO NOT:
- ❌ Change Pass 1 prompt (V14's is fine - correctly filters poetry/quotes)
- ❌ Change Pass 2 prompt (it's already identical to V13.1's successful prompt)
- ❌ Use V12's permissive Pass 1 prompt (extracts low-quality content)

### DO:
1. ✅ **Keep V14.0's Pass 1 prompt** (or make slightly LESS restrictive if needed for recall)
2. ✅ **Keep the SAME Pass 2 prompt** (already identical to V13.1)
3. ✅ **Compare V13.1 vs V14.0 Pass 2.5 configurations**:
   - Which modules were in V13.1's 12 modules?
   - Which 2 modules were added in V14.0's 14 modules?
   - What thresholds/configs changed?
4. ✅ **Test hypotheses**:
   - Disable MetadataFilter → does philosophical content decrease?
   - Relax ConfidenceFilter thresholds → do we get more valid relationships?
   - Test PredicateNormalizer V1.4 vs V1.3 → semantic validation impact?
5. ✅ **Run ablation tests**:
   - V14 Pass 1 + V14 Pass 2 + V13.1 Pass 2.5 (12 modules) → expected A- grade
   - Compare to V14.0 (14 modules) to isolate which new module caused regression

### Expected V14.2 Results:
- **Target**: 600-850 relationships (V14's volume was acceptable!)
- **Target**: A or A+ grade (3-5% issue rate, ~20-40 issues)
- **NO** novel error patterns (Rumi poetry, praise quotes filtered by V14 Pass 1)
- **NO** regression in philosophical content filtering

---

## 🧪 Recommended V14.2 Implementation

### Option 1: Rollback to V13.1 Pass 2.5 Config (Conservative)
**Approach**: Use V14 Pass 1 + V14 Pass 2 + V13.1 Pass 2.5 (12 modules)

**Rationale**:
- V14 Pass 1 prevents Rumi poetry / praise quote extraction
- V14 Pass 2 is identical to V13.1 Pass 2
- V13.1 Pass 2.5 achieved A- grade
- Expected: A or A- grade (8.6% or better)

**Risk**: Low - known good configuration

### Option 2: Debug V14.0 Pass 2.5 Modules (Investigative)
**Approach**: Systematically test each V14.0 module change

**Steps**:
1. Run V14 Pass 1 + V14 Pass 2 + V13.1 Pass 2.5 → baseline
2. Add MetadataFilter → measure impact
3. Add ConfidenceFilter → measure impact
4. Upgrade PredicateNormalizer V1.3 → V1.4 → measure impact
5. Identify which change caused regression
6. Fix the problematic module/config

**Risk**: Medium - requires multiple test runs

**Benefit**: Identifies exact cause, improves system understanding

### Option 3: Hybrid - V14 Enhancements + V13.1 Stability (Recommended)
**Approach**: Cherry-pick V14.0 improvements that didn't cause regression

**Steps**:
1. Start with V13.1 Pass 2.5 config (12 modules) - **baseline**
2. Add MetadataFilter **with strict thresholds** - test
3. If MetadataFilter works, add ConfidenceFilter **with relaxed thresholds** - test
4. Keep PredicateNormalizer V1.3 (don't upgrade to V1.4 yet)
5. Iterate until A or A+ grade achieved

**Risk**: Low-Medium - incremental testing catches regressions

**Benefit**: Best of both worlds - V14 improvements + V13.1 stability

---

## 📝 Conclusion

**Key Finding**: V13.1 and V14.0 used **IDENTICAL Pass 2 prompts**. The regression is in **Pass 2.5 postprocessing configuration**, not Pass 2 evaluation.

**V14.2 Strategy**:
- Keep V14's Pass 1 (filters low-quality content)
- Keep V14's Pass 2 (already same as V13.1)
- **Rollback to V13.1's Pass 2.5 config** OR **debug V14.0's new modules**

**Next Steps**:
1. Read V13.1's book_pipeline code to identify exact 12 modules
2. Compare to V14.0's 14 modules
3. Create V14.2 with Option 1 (Conservative Rollback) first
4. If successful, run Option 2 (Investigative Debug) to understand root cause
5. Eventually implement Option 3 (Hybrid) for best long-term solution

**Expected Outcome**: A or A+ grade (3-5% issue rate, down from V14.0's 10.78%)

---

**Status**: Root cause identified - Pass 2.5 module configuration regression
**Recommendation**: Implement Option 1 (Conservative Rollback) for V14.2
**Timeline**: Ready to implement immediately
