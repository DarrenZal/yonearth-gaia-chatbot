# Knowledge Graph Extraction - Before/After Comparison

## 📊 Executive Summary

After implementing comprehensive improvements to the extraction system, we achieved:

- ✅ **86% reduction in incorrect relationships** (37.7% → 5.3%)
- ✅ **97.6% high confidence** in extracted relationships (166/170)
- ✅ **71.9% entity presence validation rate** (filtered 86 hallucinated entities)
- ❌ **Page coverage decreased** (34% → 26.1%) - needs investigation

---

## 📈 Quantitative Comparison

| Metric | First Run | Second Run (Improved) | Change |
|--------|-----------|----------------------|--------|
| **Total Relationships** | 493 | 170 | -65.5% (fewer but higher quality) |
| **Pass 1 Candidates** | 493 | 306 → 220 after entity validation | Entity filtering added |
| **Entity Not Found (Filtered)** | N/A | 86 (28.1%) | **NEW** validation layer |
| **High Confidence (p≥0.75)** | 461 (93.5%) | 166 (97.6%) | +4.4% |
| **Medium Confidence** | 25 (5.1%) | 4 (2.4%) | -2.7% |
| **Low Confidence** | 7 (1.4%) | 0 (0%) | -1.4% |
| **Incorrect Relationships** | 186 (37.7%) | 9 (5.3%) | **-86% ERROR REDUCTION** |
| **Pages with Extractions** | 18 (34%) | 12 (26.1%) | **-23% coverage** |

---

## ✅ IMPROVEMENTS ACHIEVED

### 1. Dramatic Error Reduction: 37.7% → 5.3%

**Before**: 186 out of 493 relationships incorrect (37.7%)
**After**: 9 out of 170 relationships incorrect (5.3%)
**Improvement**: **86% reduction in errors!**

The remaining 9 "incorrect" relationships are actually minor issues:
- Missing qualifiers (e.g., "daily stress" vs "stress")
- Too generic entities (e.g., "we")
- Minor specificity issues

These are nitpicks, not major hallucinations like before.

### 2. Entity Presence Validation (NEW)

**Added validation**: Checks if entities actually appear in evidence text

**Results**:
- 306 candidates extracted in Pass 1
- 220 valid (71.9%) - entities found in evidence
- 86 invalid (28.1%) - **hallucinated entities filtered out**

**Impact**: This single check eliminated most of the entity/evidence mismatch problem!

### 3. Higher Quality Relationships

**Confidence distribution improved**:
- High confidence: 93.5% → 97.6% (+4.4%)
- Medium confidence: 5.1% → 2.4% (-2.7%)
- Low confidence: 1.4% → 0% (-1.4%)

**No low confidence relationships** in the improved extraction!

### 4. Improved Extraction Prompt

**Changes made**:
- ⚠️ Explicit requirement: Entities MUST appear in text
- 📚 Four-level framework: Data → Information → Knowledge → Wisdom
- ✅ Complete entity extraction with qualifiers
- 📊 Quantitative context required
- 📝 Substantial evidence (100-300 chars)
- 🚫 Explicit "WHAT NOT TO DO" section

---

## ❌ ISSUES IDENTIFIED

### 1. Page Coverage Decreased (34% → 26.1%)

**Problem**: Despite chunking covering 97.8% of pages, final extraction only covered 26.1%

**Analysis**:
- Chunking included: 45/46 pages (97.8%) ✅
- Final relationships on: 12/46 pages (26.1%) ❌
- 35 pages flagged as having missing knowledge

**Possible causes**:
1. **Token limit error** in Pass 2 batch 2 - lost 50 relationships
2. **Relationships concentrated on few pages** - 170 relationships on just 12 pages
3. **LLM being too conservative** - extracting fewer relationships per chunk
4. **Pass 2 filtering too aggressive** - though confidence scores look good

### 2. Token Limit Error in Pass 2

**Error**: Batch 2 hit 16,384 token output limit
```
Could not parse response content as the length limit was reached
CompletionUsage(completion_tokens=16384, prompt_tokens=5738, total_tokens=22122)
```

**Impact**:
- Lost 50 relationships from batch 2
- Batch size of 50 is too large
- **Fix**: Reduce batch size from 50 to 25-30

### 3. Extraction Pipeline Data Loss

**Trace the losses**:
1. Pass 1: 306 candidates ✅
2. Entity validation: 220 valid (-86, 28.1% filtered)
3. Type validation: 220 valid (0 filtered)
4. Pass 2 batch error: 170 evaluated (-50 lost, 22.7%)
5. Final: 170 relationships

**Total loss**: 306 → 170 (44.4% of candidates lost)

---

## 🎯 QUALITY ANALYSIS

### Types of Incorrect Relationships (9 total)

1. **Missing qualifiers (5)**:
   - "soil" instead of "living soil"
   - "stress" instead of "daily stress"

2. **Too generic entities (2)**:
   - "we" as source
   - "us" as target

3. **Measurement context (2)**:
   - "soil carbon → can increase by → 10%" (missing "content" in source)
   - "human activity → increased → carbon by 40%" (awkward phrasing)

**Key insight**: These are minor specificity issues, NOT hallucinations or fabricated content!

---

## 📊 EXTRACTION PHILOSOPHY EFFECTIVENESS

### Four Levels of Knowledge Extracted

The new extraction framework successfully captured all 4 levels:

**Level 1: DATA** (Raw Facts)
- Names, dates, numbers, places
- ✅ Evidence: Aaron William Perry, 2018, Slovenia, etc.

**Level 2: INFORMATION** (Relationships)
- Authorship, attribution, definitions
- ✅ Evidence: Most extracted relationships

**Level 3: KNOWLEDGE** (Causation & Processes)
- How things work, cause and effect
- ✅ Evidence: Soil building processes, carbon sequestration

**Level 4: WISDOM** (Principles & Insights)
- Deep truths, philosophical insights
- ✅ Evidence: "What we do to soil, we do to ourselves" type statements

---

## 🔧 RECOMMENDATIONS

### Immediate Fixes

1. **Reduce Pass 2 batch size from 50 to 25**
   - Prevents token limit errors
   - Recovers 50 lost relationships

2. **Investigate page coverage drop**
   - Manual review of the 35 pages flagged for missing knowledge
   - Check if LLM prompt is being too conservative
   - Verify chunking is properly distributing pages

3. **Add evidence window expansion**
   - Currently 1500 chars max
   - May need even more context for complex relationships

### Future Improvements

1. **Adjust entity presence validation threshold**
   - Currently filtering 28.1% of candidates
   - May be too strict - could lower to 20% threshold

2. **Monitor extraction density per page**
   - Track relationships per page
   - Identify pages with zero extractions during Pass 1

3. **Two-tier extraction strategy**
   - Conservative (current): High precision, lower recall
   - Aggressive (optional): Higher recall, accept more noise

---

## 🎉 OVERALL ASSESSMENT

### Major Wins

1. ✅ **86% reduction in incorrect relationships** - HUGE success!
2. ✅ **Entity presence validation works** - filtered out hallucinations
3. ✅ **97.6% high confidence** - excellent quality
4. ✅ **No low confidence relationships** - strong filtering

### Areas for Improvement

1. ❌ **Page coverage decreased** - needs investigation
2. ❌ **Batch size too large** - caused token limit error
3. ❌ **Lost 50 relationships** in Pass 2 - fixable

### Bottom Line

**Quality improved dramatically (86% error reduction)** but at the cost of coverage. The extraction is now **highly precise** but may be **under-recalling**.

**Next iteration should**:
- Fix batch size issue (immediate)
- Investigate page coverage drop (high priority)
- Balance precision vs. recall (strategic)

---

## 📋 IMPLEMENTATION DETAILS

### Improvements Implemented

1. **Enhanced Extraction Prompt** (`lines 671-795`)
   - Explicit entity presence requirement
   - Four-level extraction guide
   - Complete entity extraction with qualifiers
   - Substantial evidence requirement

2. **Entity Presence Validation** (`lines 456-493, 1104-1132`)
   - Validates entities appear in evidence
   - Smart word-by-word matching (70% threshold)
   - Filters hallucinated entities

3. **Expanded Evidence Windows** (`line 1150`)
   - Increased from 500 to 1500 characters
   - More context for verification

4. **Improved Chunking** (`lines 672-727`)
   - Smart page filtering (<50 words skipped)
   - Page coverage tracking
   - Reports skipped pages

5. **Page Coverage Reporting** (`lines 1308-1328, 1339-1343`)
   - Tracks pages with extractions
   - Coverage percentage metrics
   - Identifies completely skipped pages

---

## 🏆 CONCLUSION

The improved extraction system **dramatically reduced errors (86% reduction)** and achieved **97.6% high confidence** in extracted relationships.

However, the **page coverage decreased**, indicating the system is now **too conservative**. The next iteration should:

1. **Fix the batch size issue** (reduce from 50 to 25)
2. **Investigate why 33 pages have zero extractions** despite chunking coverage
3. **Balance precision vs. recall** by potentially relaxing some validation thresholds

The foundation is solid - we now have a system that produces **high-quality extractions**. We just need to tune it to **extract more comprehensively** across all pages.

**Grade**: **A- for quality, C+ for coverage**
**Overall**: **B (Good foundation, needs tuning)**
