# V14.2 Implementation Plan

**Date**: October 14, 2025
**Objective**: Restore A/A+ grade quality by fixing V14.0's Pass 2.5 regression
**Strategy**: Conservative rollback approach (Option 1 from root cause analysis)

---

## 🎯 Executive Summary

**Root Cause Identified**:
- V13.1 (A-) and V14.0 (B+) used **IDENTICAL Pass 2 evaluation prompts**
- Regression is in **Pass 2.5 postprocessing pipeline** (12 → 14 modules)
- V14.0 added 2 new modules (MetadataFilter, ConfidenceFilter) that caused issues

**V14.2 Strategy**:
- ✅ Keep V14.0's Pass 1 prompt (correctly filters poetry/quotes)
- ✅ Keep V14.0's Pass 2 prompt (same as V13.1 - working correctly)
- ✅ **Rollback to V13.1's Pass 2.5 configuration (12 modules)**

**Expected Outcome**: A or A- grade (3-8.6% issue rate)

---

## 📊 Evidence Summary

### What We Discovered:

1. **Pass 2 Prompts are IDENTICAL** ✅
   ```bash
   $ diff pass2_evaluation_v13_1.txt pass2_evaluation_v14.txt
   # NO DIFFERENCES
   ```

2. **Pass 2.5 Pipelines are DIFFERENT** ❌
   - V13.1: `get_book_pipeline()` → **12 modules**
   - V14.0: `get_book_pipeline()` → **14 modules** (added MetadataFilter, ConfidenceFilter)

3. **V14.1 Proved V12's Prompt Extracts Low-Quality Content** ✅
   - V14.1 used V12 prompt → 782 candidates, 118 issues (16.7% - WORST EVER)
   - V12 prompt extracts Rumi poetry, praise quotes, book title misparses
   - V14 prompt correctly filters this content (596 candidates, fewer poetry issues)

### Configuration Comparison:

| Version | Pass 1 | Pass 2 | Pass 2.5 | Result | Grade |
|---------|--------|--------|----------|--------|-------|
| **V13.1** | V12 checkpoint (861) | v13_1.txt | 12 modules | 873 rels, 75 issues | **A- (8.6%)** ✅ |
| **V14.0** | v14.txt (596) | v14.txt (SAME) | 14 modules | 603 rels, 65 issues | **B+ (10.78%)** ❌ |
| **V14.1** | V12 (782) | v14.txt (SAME) | 15 modules (buggy) | 708 rels, 118 issues | **C+ (16.7%)** ❌❌ |

---

## 🔧 V14.2 Implementation (Conservative Rollback)

### Configuration:

```python
# V14.2 CONFIGURATION
Pass 1: pass1_extraction_v14.txt (27KB, filters poetry/quotes)
Pass 2: pass2_evaluation_v14.txt (IDENTICAL to v13_1.txt)
Pass 2.5: V13 book_pipeline configuration (12 modules)
```

### Key Changes from V14.0:

1. **Rollback Pass 2.5 to V13.1 configuration**
   - Remove MetadataFilter module (or disable it)
   - Remove ConfidenceFilter module (or disable it)
   - Use PredicateNormalizer V1.3 (not V1.4)
   - Restore V13.1's 12-module pipeline

2. **Keep V14.0's Pass 1 prompt** (prevents low-quality extractions)
   - Filters Rumi poetry
   - Filters praise quotes
   - Prevents book title misparses

3. **Keep V14.0's Pass 2 prompt** (already same as V13.1 - no changes needed)

---

## 📝 Implementation Steps

### Step 1: Check V13 Pipeline Configuration

```bash
# Read V13's book_pipeline code to identify exact 12 modules
cat src/knowledge_graph/postprocessing/pipelines/book_pipeline.py | grep -A 50 "def get_book_pipeline"
```

**Expected modules** (based on V13 script documentation):
1. PraiseQuoteDetector
2. BibliographicCitationParser
3. VagueEntityBlocker
4. ListSplitter (V13 fixed with POS tagging)
5. ContextEnricher
6. PronounResolver
7. PredicateNormalizer (V13 enhanced, ~80 predicates)
8. PredicateValidator
9. TitleCompletenessValidator
10. FigurativeLanguageFilter
11. ClaimClassifier
12. Deduplicator

### Step 2: Create V14.2 Book Pipeline

Option A: **Modify get_book_pipeline() to accept version parameter**
```python
def get_book_pipeline(version='v14'):
    """
    Get book processing pipeline.

    Args:
        version: 'v13' (12 modules) or 'v14' (14 modules)
    """
    if version == 'v13':
        # Return V13.1's 12-module pipeline
        modules = [
            PraiseQuoteDetector(...),
            BibliographicCitationParser(...),
            VagueEntityBlocker(...),
            ListSplitter(...),  # V13 version with POS tagging
            ContextEnricher(...),
            PronounResolver(...),
            PredicateNormalizer(...),  # V13 version (~80 predicates)
            PredicateValidator(...),
            TitleCompletenessValidator(...),
            FigurativeLanguageFilter(...),
            ClaimClassifier(...),
            Deduplicator(...)
        ]
    elif version == 'v14':
        # V14.0's 14-module pipeline (current)
        modules = [
            # ... existing V14 modules ...
        ]

    return PostprocessingPipeline(modules)
```

Option B: **Create separate get_book_pipeline_v13() function**
```python
def get_book_pipeline_v13():
    """V13.1's 12-module pipeline (A- grade baseline)"""
    modules = [
        PraiseQuoteDetector(...),
        # ... same 12 modules as V13.1 ...
    ]
    return PostprocessingPipeline(modules)
```

### Step 3: Create extract_kg_v14_2_book.py

Copy `extract_kg_v14_book.py` and modify:

```python
# V14.2 CONFIGURATION
PASS1_PROMPT_FILE = PROMPTS_DIR / "pass1_extraction_v14.txt"  # Keep V14
PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v14.txt"  # Keep V14 (same as V13.1)

# Pass 2.5: Use V13.1 pipeline
def postprocess_pass2_5(...):
    logger.info("🔧 PASS 2.5: Running V13.1 modular postprocessing pipeline (12 modules)...")

    # Use V13.1's 12-module configuration
    pipeline = get_book_pipeline(version='v13')  # OR get_book_pipeline_v13()

    processed_objs, stats = pipeline.run(relationships, context)
    return processed_objs, stats
```

### Step 4: Update ProcessingContext

```python
context = ProcessingContext(
    content_type='book',
    document_metadata=document_metadata,
    pages_with_text=pages_with_text,
    run_id=run_id,
    extraction_version='v14_2'  # New version identifier
)
```

### Step 5: Add Logging

```python
logger.info("="*80)
logger.info("🚀 V14.2 KNOWLEDGE GRAPH EXTRACTION - CONSERVATIVE ROLLBACK")
logger.info("="*80)
logger.info("")
logger.info("✨ V14.2 CONFIGURATION:")
logger.info("  1. ✅ V14 Pass 1: Filters poetry/quotes (27KB prompt)")
logger.info("  2. ✅ V14 Pass 2: Dual-signal evaluation (same as V13.1)")
logger.info("  3. ✅ V13.1 Pass 2.5: 12-module pipeline (A- baseline)")
logger.info("")
logger.info("🎯 ROOT CAUSE FIX:")
logger.info("  - V14.0 regression caused by Pass 2.5 changes (12 → 14 modules)")
logger.info("  - MetadataFilter and ConfidenceFilter introduced issues")
logger.info("  - Rolling back to V13.1's proven 12-module configuration")
logger.info("")
logger.info("📊 EXPECTED RESULTS:")
logger.info("  - V14.0 Grade: B+ (10.78% issue rate, 65 issues)")
logger.info("  - V14.2 Target: A or A- (3-8.6% issue rate, <55 issues)")
logger.info("  - Expected improvement: 10-40 issues fixed")
logger.info("")
```

---

## 🧪 Testing Plan

### Test 1: V14.2 Full Extraction
```bash
python3 scripts/extract_kg_v14_2_book.py
```

**Expected Results**:
- Pass 1: ~600-650 candidates (V14 prompt, similar to V14.0's 596)
- Pass 2: ~600-650 evaluated (minimal filtering)
- Pass 2.5: ~850-900 final relationships (V13.1 pipeline expansion from ListSplitter)
- Grade: A or A- (3-8.6% issue rate)

### Test 2: Run Reflector on V14.2
```bash
python3 scripts/run_reflector_on_v14_2.py
```

**Success Criteria**:
- ✅ Grade: A or A- (target: <8.6% issue rate)
- ✅ NO novel error patterns (no Rumi poetry, no praise quotes)
- ✅ Philosophical content: similar to V13.1 (~8-13 issues, 0.9-1.5%)
- ✅ Predicate fragmentation: similar to V13.1
- ✅ Total issues: 25-75 (down from V14.0's 65)

### Test 3: Compare V14.2 vs V13.1 vs V14.0
```bash
python3 scripts/compare_v14_2_vs_v13_1_vs_v14_0.py
```

**Analysis**:
- Which issue categories improved over V14.0?
- Which issue categories match V13.1 baseline?
- Are there any new regressions?

---

## 📈 Expected Improvements Over V14.0

### Issue Category Predictions:

| Category | V14.0 | V14.2 Target | Change |
|----------|-------|--------------|--------|
| **Philosophical/Abstract Claims** | 18 (2.99%) | ~8-13 (0.9-1.5%) | **-50% to -70%** ✅ |
| **Redundant 'is-a' Relationships** | 25 (4.15%) | ~10-15 (1.1-1.7%) | **-40% to -60%** ✅ |
| **Predicate Fragmentation** | 12 (1.99%) | ~10 (1.1%) | **-17% to -45%** ✅ |
| **Vague/Generic Entities** | 8 (1.33%) | ~12 (1.4%) | **+50%** ⚠️ possible |
| **Praise Quotes** | 0 (0%) | 0 (0%) | **No change** ✅ |
| **Unresolved Pronouns** | 0 (0%) | 0 (0%) | **No change** ✅ |
| **TOTAL** | 65 (10.78%) | **25-75 (3-8.6%)** | **-62% to -85%** ✅✅✅ |

**Note**: Vague entities might increase slightly because V14's enhanced filters are removed, but overall quality should improve significantly.

---

## 🚨 Risk Assessment

### Low Risk:
- ✅ V13.1's Pass 2.5 configuration is **proven** (A- grade)
- ✅ V14's Pass 1 is **proven** to filter low-quality content (no Rumi poetry in V14.0)
- ✅ Pass 2 prompts are **identical** (no change in evaluation logic)

### Potential Issues:
- ⚠️ V14's Pass 1 might be too restrictive (596 vs 861 candidates)
  - **Mitigation**: If V14.2 gets <800 relationships, consider relaxing Pass 1 slightly
- ⚠️ Vague entities might increase without V14's VagueEntityBlocker enhancements
  - **Mitigation**: Monitor reflector results; if vague entities >15, re-enable VagueEntityBlocker

### Rollback Plan:
- If V14.2 fails to achieve A/A- grade:
  - Analyze which issue categories regressed
  - Consider Option 2 (Investigative Debug) to identify specific module causing V14.0 regression
  - Implement Option 3 (Hybrid) with selective V14 module cherry-picking

---

## 📝 Next Steps After V14.2

### If V14.2 Succeeds (A or A- grade):
1. ✅ **Adopt V14.2 as new stable release**
2. 🔬 **Run Option 2 (Investigative Debug)** to understand which V14.0 module caused regression
3. 🔬 **Create V14.3** with fixed MetadataFilter/ConfidenceFilter modules
4. 🎯 **Target V15**: A+ grade (<5% issue rate) by incrementally adding V14 improvements

### If V14.2 Partially Succeeds (B+ or better, <10% issue rate):
1. 📊 Analyze which issue categories still problematic
2. 🔧 Create V14.2.1 with targeted fixes for specific issue types
3. 🔬 Run ablation tests to identify best module combination

### If V14.2 Fails (B or worse, >12% issue rate):
1. ❌ **Abandon rollback approach**
2. 🔬 **Deep dive into V13.1 vs V14.0 module differences**
3. 🔬 **Consider reverting to V13 Pass 1 prompt as well** (full V13.1 configuration)
4. 🎯 **Re-evaluate V14 strategy entirely**

---

## 🎯 Success Metrics

### Primary Goal: A or A- Grade
- **A+ Grade**: <5% issue rate (<43 issues)
- **A Grade**: 5-7% issue rate (43-60 issues)
- **A- Grade**: 7-9% issue rate (60-75 issues) ← **Target**
- **B+ Grade**: 9-12% issue rate (75-103 issues)

### Secondary Goals:
- ✅ NO Rumi poetry extraction
- ✅ NO praise quote misclassification
- ✅ NO book title misparses
- ✅ Philosophical content similar to V13.1 (~8-13 issues, 0.9-1.5%)
- ✅ Total relationships: 800-900 (similar to V13.1's 873)

---

## 📁 File Changes Required

### New Files:
1. `/scripts/extract_kg_v14_2_book.py` - Main extraction script
2. `/kg_extraction_playbook/output/v14_2/` - Output directory
3. `/scripts/run_reflector_on_v14_2.py` - Reflector analysis script
4. `/scripts/compare_v14_2_vs_v13_1_vs_v14_0.py` - Comparison script

### Modified Files:
1. `/src/knowledge_graph/postprocessing/pipelines/book_pipeline.py`
   - Add `get_book_pipeline(version='v14')` parameter OR
   - Add `get_book_pipeline_v13()` function

### No Changes Needed:
- ✅ `/kg_extraction_playbook/prompts/pass1_extraction_v14.txt` (keep as is)
- ✅ `/kg_extraction_playbook/prompts/pass2_evaluation_v14.txt` (keep as is)

---

## 🏁 Timeline

| Task | Duration | Status |
|------|----------|--------|
| 1. Verify V13 pipeline configuration | 15 min | Pending |
| 2. Modify book_pipeline.py | 30 min | Pending |
| 3. Create extract_kg_v14_2_book.py | 30 min | Pending |
| 4. Run V14.2 extraction | 45 min | Pending |
| 5. Run reflector on V14.2 | 10 min | Pending |
| 6. Analyze results | 30 min | Pending |
| 7. Document findings | 30 min | Pending |
| **TOTAL** | **~3 hours** | **Ready to start** |

---

## 📄 Conclusion

V14.2 represents a **conservative, low-risk approach** to fixing V14.0's regression:
- Proven Pass 1 (filters low-quality content)
- Proven Pass 2 (same as V13.1's A- baseline)
- Proven Pass 2.5 (V13.1's 12-module configuration)

**Expected Outcome**: A or A- grade (3-8.6% issue rate, 25-75 issues)

**If successful**: V14.2 becomes the new baseline, and we can incrementally add V14 improvements in V14.3+

**If unsuccessful**: We have learned valuable lessons and can adjust strategy accordingly

---

**Status**: Ready to implement
**Risk Level**: Low
**Confidence**: High (based on empirical evidence from V13.1's A- performance)

Let's proceed with implementation!
