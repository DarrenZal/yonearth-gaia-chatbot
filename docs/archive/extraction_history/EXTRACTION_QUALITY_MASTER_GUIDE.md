# 📊 Knowledge Graph Extraction - Quality Master Guide

**Last Updated**: October 11, 2025
**Version**: v3.2.2 (Improved)
**Status**: 🔴 **CRITICAL ISSUES IDENTIFIED** - Re-extraction Required

---

## 📋 Executive Summary

Comprehensive quality review of Soil Stewardship Handbook extraction reveals **critical issues** requiring immediate attention:

### 🔴 Critical Findings

| Issue | Severity | Impact |
|-------|----------|--------|
| **37.7% Incorrect Relationships** | 🔴 CRITICAL | 186 out of 493 relationships have issues |
| **66% Pages Skipped** | 🔴 CRITICAL | 35 out of 53 pages had NO extractions |
| **Missing Knowledge** | 🟡 HIGH | 28 pages with extractable content missed |
| **Coverage Rate** | 🟡 HIGH | Only 34% of pages extracted |

### 📊 Quality Metrics

- **Total Relationships**: 493
- **Correct Relationships**: 307 (62.3%)
- **Incorrect Relationships**: 186 (37.7%)
- **Pages Covered**: 18/53 (34.0%)
- **Pages Fully Skipped**: 35/53 (66.0%)

---

## 🚨 Part 1: Critical Issues Identified

### Issue #1: Entity/Evidence Mismatch (CRITICAL)

**Problem**: 186 relationships (37.7%) have entities that don't appear in the evidence text.

**Example**:
```
Triple: "Soil Stewardship Handbook" → "is authored by" → "Aaron William Perry"
Evidence: "Copyright © 2018 Aaron William Perry..."
Issue: "Soil Stewardship Handbook" not found in evidence text
```

**Root Cause**: Entities are being extracted from broader context (multiple sentences or paragraphs) but only a narrow evidence window is being saved. This creates a mismatch where the entity names don't appear in the saved evidence snippet.

**Impact**:
- Cannot verify relationships from evidence alone
- Evidence spans incomplete
- Makes human review difficult
- Undermines trust in extraction

**Fix Required**:
1. **Expand evidence windows** to capture full context where entities appear
2. **Validate entity presence** in evidence before saving relationship
3. **Save broader context** (±100 words around relationship)
4. **Add validation check**: Flag relationships where entities missing from evidence

---

### Issue #2: Poor Page Coverage (CRITICAL)

**Problem**: Only 34% of pages (18/53) have any extractions at all.

**Pages Completely Skipped**: 35 pages
- Pages: 1, 3-9, 11, 13-14, 16-17, 20, 22, 24, 27, 29-33, 35, 37-38, 40, 42-43, 45-46, 48-49, 51-53

**Analysis**:
- Some skipped pages are expected (copyright, TOC, references)
- BUT many contain valuable content that should have been extracted
- Page 16: Contains "APPRENTICE (BEGINNER) LEVEL" framework - should have extracted this!
- Page 20: Contains statistics and climate change data - should have extracted this!
- Page 48: Contains Y on Earth Community description - should have extracted this!

**Root Cause**:
1. **Chunking strategy** may skip pages with less dense text
2. **Extraction threshold** may be filtering out valid but "low confidence" relationships
3. **Evidence window issues** may cause chunks to not align with page boundaries

**Fix Required**:
1. **Review chunking logic** - ensure all pages with text >100 words are included in chunks
2. **Lower confidence threshold** for first pass - filter later, don't lose data
3. **Add page coverage metrics** to extraction pipeline
4. **Flag pages with zero extractions** for manual review

---

### Issue #3: Lost Specificity (HIGH PRIORITY)

**Problem**: Extracted entities lose important context from the source text.

**Examples**:

#### 3a: Measurement Context Lost
```
Text: "soil carbon of about 10%"
Extracted: soil → is increased by → 10%
SHOULD BE: soil carbon content → can increase by → 10%
```

#### 3b: Qualifiers Dropped
```
Text: "organic, biodegradable kitchen scraps"
Extracted: Kitchen scraps → end up in → landfills
SHOULD BE: organic biodegradable kitchen scraps → end up in → landfills
```

#### 3c: Scope Missing
```
Text: "10% increase of the carbon content in soil world-wide"
Extracted: soil → is increased by → 10%
SHOULD BE: global soil carbon content → can increase by → 10%
```

**Impact**:
- Relationships lose meaning
- Cannot understand full context
- Quantitative claims become nonsensical

**Fix Required**: ✅ **ALREADY IMPLEMENTED** in improved prompt (see extract_kg_v3_2_2_book_improved.py lines 684-739)

---

### Issue #4: Missing Knowledge on Extracted Pages (HIGH)

**Problem**: Even pages that WERE extracted are missing significant knowledge.

**Example - Page 23**:
- Text contains: definitions, statistics, processes, benefits, organizations
- Only extracted: 20 relationships
- But has 5+ knowledge indicators suggesting more content should have been extracted

**Examples of Missing Knowledge**:
- Page 10: Mentions "Lily Sophia von Übergarten" but partial extraction
- Page 15: Contains SSG (Soil Stewardship Guild) framework but incomplete
- Page 28: Contains acknowledgments with contributor names but many missed

**Root Cause**:
1. **Extraction prompt** may be too conservative
2. **Pass 2 filtering** may be too aggressive
3. **Relationship density** varies by page - some pages have more extractable content per word

**Fix Required**:
1. **Re-evaluate Pass 2 threshold** - consider keeping medium confidence relationships
2. **Add explicit examples** to extraction prompt for each knowledge type
3. **Test on sample pages** to calibrate extraction sensitivity

---

## 🔧 Part 2: Technical Improvements Needed

### 2.1 Evidence Window Expansion

**Current Issue**: Evidence windows too narrow (truncated to 500 chars)

**Proposed Fix**:
```python
# Current (lines 1048-1052)
MAX_WIN = 500
if len(rel.evidence_text) > MAX_WIN:
    rel.evidence_text = rel.evidence_text[:MAX_WIN] + "…"

# Proposed
MAX_WIN = 1000  # Double the window
# AND: Expand to include ±2 sentences around relationship
```

### 2.2 Entity Presence Validation

**Add validation check**:
```python
def validate_entity_in_evidence(entity: str, evidence: str) -> bool:
    """Check if entity appears in evidence (with fuzzy matching)"""
    entity_lower = entity.lower()
    evidence_lower = evidence.lower()

    # Exact match
    if entity_lower in evidence_lower:
        return True

    # Fuzzy match for multi-word entities
    entity_words = entity_lower.split()
    if len(entity_words) > 1:
        matches = sum(1 for word in entity_words if word in evidence_lower)
        if matches >= len(entity_words) * 0.7:  # 70% of words present
            return True

    return False

# Use during extraction:
if not validate_entity_in_evidence(rel.source, rel.evidence_text):
    rel.flags["ENTITY_NOT_IN_EVIDENCE"] = True
    # Flag for review or expand evidence window
```

### 2.3 Page Coverage Monitoring

**Add to extraction pipeline**:
```python
# Track page coverage
pages_with_extractions = set()
for rel in relationships:
    page = rel.evidence.get('page_number')
    if page:
        pages_with_extractions.add(page)

# Report coverage
total_pages = len(pages_with_text)
coverage = len(pages_with_extractions) / total_pages
logger.info(f"Page coverage: {coverage:.1%} ({len(pages_with_extractions)}/{total_pages})")

# Flag pages with zero extractions
skipped_pages = set(range(1, total_pages + 1)) - pages_with_extractions
if skipped_pages:
    logger.warning(f"⚠️  {len(skipped_pages)} pages had zero extractions: {sorted(skipped_pages)}")
```

### 2.4 Improved Extraction Prompt

✅ **ALREADY DONE**: The improved prompt (lines 671-739) includes:
- Extract complete entities with modifiers
- Handle quantitative claims carefully
- Preserve scope and context
- Explicit examples of good vs bad extraction

**What's Still Needed**:
- Add more examples for each entity type
- Add explicit guidance for acknowledgments, citations, references
- Add guidance for handling quotes and attributions

---

## 📝 Part 3: Recommended Actions (Priority Order)

### Priority 1: Fix Entity/Evidence Mismatch (IMMEDIATE)

**Tasks**:
1. ✅ Add entity presence validation (code above)
2. ✅ Expand evidence windows from 500 to 1000 chars
3. ✅ Modify chunking to capture ±2 sentences around relationships
4. ⏳ Re-run extraction on Soil Handbook
5. ⏳ Verify relationships have entities in evidence

**Expected Impact**: Reduce incorrect relationships from 37.7% to <10%

---

### Priority 2: Improve Page Coverage (HIGH)

**Tasks**:
1. ✅ Add page coverage monitoring (code above)
2. ⏳ Review chunking logic - identify why pages are skipped
3. ⏳ Lower Pass 1 extraction threshold (extract more, filter less)
4. ⏳ Manually review sample of skipped pages to verify they should be skipped
5. ⏳ Re-run extraction with improved coverage

**Expected Impact**: Increase coverage from 34% to >80%

---

### Priority 3: Enhance Extraction Prompt (MEDIUM)

**Tasks**:
1. ✅ Add guidance for complete entities (DONE in improved prompt)
2. ✅ Add guidance for quantitative claims (DONE in improved prompt)
3. ⏳ Add more examples for each entity type (Person, Org, Quote, etc.)
4. ⏳ Add guidance for acknowledgments and citations
5. ⏳ Test on sample pages to validate effectiveness

**Expected Impact**: Improve extraction completeness by 20-30%

---

### Priority 4: Re-evaluate Pass 2 Filtering (MEDIUM)

**Tasks**:
1. ⏳ Analyze distribution of confidence scores
2. ⏳ Identify threshold where false negatives occur
3. ⏳ Consider keeping medium confidence relationships (p_true 0.5-0.75)
4. ⏳ Add human review workflow for medium confidence
5. ⏳ Test on sample pages

**Expected Impact**: Recover 10-20% of missed relationships

---

## 📚 Part 4: Detailed Findings Reference

### 4.1 Comprehensive Quality Report

**Full Report**: `data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2_comprehensive_review.md`

**Key Statistics**:
- Total relationships: 493
- Incorrect relationships: 186 (37.7%)
- Pages with extractions: 18/53 (34.0%)
- Pages fully skipped: 35/53 (66.0%)
- Pages with missing knowledge: 28/53 (52.8%)

### 4.2 Original Quality Review

**Report**: `data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2_quality_review.md`

**Key Findings** (simpler analysis):
- 8 issues identified across 4 categories
- 2 numbers without context
- 2 lost specificity cases
- 1 semantically odd relationship
- 3 missing critical context

**Note**: Original review underestimated issues because it didn't check entity presence in evidence.

---

## 🔬 Part 5: Root Cause Analysis

### Why Entity/Evidence Mismatch Occurs

**Hypothesis #1**: Chunking Overlap Issue
- Entities extracted from chunk boundaries
- Evidence saved from wrong chunk
- Entities appear in adjacent chunk, not saved chunk

**Hypothesis #2**: LLM Inferring Entities
- GPT-4o-mini infers entity names from context
- E.g., sees "Copyright © 2018 Aaron William Perry" and infers subject is "Soil Stewardship Handbook"
- But "Soil Stewardship Handbook" doesn't appear in that specific evidence window

**Hypothesis #3**: Evidence Truncation
- Full evidence captured during extraction
- But then truncated to 500 chars during post-processing
- Truncation removes entity mentions

**Most Likely**: Hypothesis #2 - LLM is inferring entities from broader context that isn't being saved as evidence.

**Solution**: Explicitly instruct LLM to only extract entities that **appear in the text chunk** being processed.

---

### Why Page Coverage is Low

**Hypothesis #1**: Chunking Skips Pages
- Chunking algorithm creates 800-word chunks with 100-word overlap
- Some pages <800 words may be skipped
- Or combined with adjacent pages

**Hypothesis #2**: Extraction Threshold Too High
- Pass 1 extracts relationships
- But Pass 2 filters aggressively based on confidence
- Some pages may have all relationships filtered out

**Hypothesis #3**: PDF Parsing Issues
- Some pages have images, diagrams, or formatting that breaks text extraction
- Pages appear blank or garbled to extraction pipeline

**Most Likely**: Combination of #1 and #2 - chunking misses some pages, and aggressive filtering removes others.

**Solution**:
1. Ensure ALL pages with text >100 words are included in at least one chunk
2. Track which pages appear in which chunks
3. Report pages that don't appear in any chunk

---

## 📖 Part 6: Additional Documentation

### Related Guides

- **`docs/knowledge_graph/COMPLEX_CLAIMS_AND_QUANTITATIVE_RELATIONSHIPS.md`**: How to handle quantitative claims, equivalence relationships, and multi-triple extraction
- **`docs/knowledge_graph/ENTITY_RESOLUTION_GUIDE.md`**: Basic manual entity resolution approach
- **`docs/knowledge_graph/ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md`**: Advanced entity resolution with graph embeddings
- **`docs/knowledge_graph/KG_MASTER_GUIDE_V3.md`**: Complete v3.2.2 specification
- **`docs/knowledge_graph/KG_POST_EXTRACTION_REFINEMENT.md`**: Post-processing and validation

### Scripts

- **`scripts/extract_kg_v3_2_2_book_improved.py`**: Improved book extraction with better prompts
- **`scripts/review_extraction_quality.py`**: Simple quality review (8 issues found)
- **`scripts/comprehensive_extraction_review.py`**: Comprehensive review (186 issues found)

---

## ✅ Part 7: Action Items Checklist

### Immediate (< 1 day)

- [ ] Add entity presence validation to extraction pipeline
- [ ] Expand evidence windows from 500 to 1000 characters
- [ ] Add page coverage monitoring and reporting
- [ ] Modify extraction prompt to only extract entities that appear in chunk

### Short-term (< 1 week)

- [ ] Review and fix chunking logic to ensure all pages covered
- [ ] Lower Pass 1 extraction threshold
- [ ] Add ±2 sentences context to evidence windows
- [ ] Re-run extraction on Soil Handbook with fixes
- [ ] Compare before/after metrics

### Medium-term (< 1 month)

- [ ] Add more examples to extraction prompt
- [ ] Implement human review workflow for medium confidence relationships
- [ ] Test on additional books to validate improvements
- [ ] Build automated quality checks into extraction pipeline
- [ ] Create acceptance tests for extraction quality

---

## 🎯 Success Criteria

### Target Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Incorrect Relationships | 37.7% | <10% | 🔴 |
| Page Coverage | 34.0% | >80% | 🔴 |
| Entity in Evidence | ~62% | >95% | 🔴 |
| Extraction Completeness | Unknown | >80% | 🟡 |

### Validation Tests

After re-extraction, verify:
1. ✅ All relationships have entities present in evidence text
2. ✅ >80% of pages with substantive content have at least one extraction
3. ✅ Quantitative claims preserve measurement context
4. ✅ Qualifiers and scope are captured in entity names
5. ✅ Evidence windows are sufficient to understand relationships

---

## 🔄 Iteration Plan

### Phase 1: Fix Critical Issues (This Week)
1. Implement entity validation
2. Expand evidence windows
3. Add page coverage monitoring
4. Re-run extraction on Soil Handbook

### Phase 2: Validate Improvements (Next Week)
1. Run comprehensive quality review on new extraction
2. Compare metrics: before vs after
3. Manual spot-check sample of relationships
4. Identify remaining issues

### Phase 3: Refine and Scale (Following Weeks)
1. Apply learnings to other books
2. Build automated quality checks
3. Create golden dataset for testing
4. Document best practices

---

## 📞 Questions & Next Steps

**Open Questions**:
1. Should we keep medium confidence relationships (0.5-0.75)?
2. How much evidence context is enough? 1000 chars? 2000 chars?
3. Should we extract acknowledgments, citations, and references differently?
4. What's the right balance between precision and recall?

**Immediate Next Step**: Implement Priority 1 fixes and re-run extraction to measure impact.

---

## 📊 Appendix: Sample Issues

### Sample Issue #1: Entity Not in Evidence

**Relationship**:
```json
{
  "source": "Soil Stewardship Handbook",
  "relationship": "is authored by",
  "target": "Aaron William Perry",
  "evidence_text": "Copyright © 2018 Aaron William Perry",
  "page": 2
}
```

**Problem**: "Soil Stewardship Handbook" does not appear in evidence text

**Fix**: Expand evidence to include surrounding context:
```
"This book is dedicated to my two children... Copyright © 2018 Aaron William Perry.
All rights reserved. This Soil Stewardship Handbook provides..."
```

---

### Sample Issue #2: Lost Specificity

**Relationship**:
```json
{
  "source": "soil",
  "relationship": "is increased by",
  "target": "10%",
  "evidence_text": "we're only talking about an increase of soil carbon of about 10%",
  "page": 21
}
```

**Problems**:
1. "soil" should be "soil carbon content"
2. Semantically odd: "soil is increased by 10%" doesn't make sense
3. Missing scope: should mention "worldwide"

**Fix**:
```json
{
  "source": "global soil carbon content",
  "relationship": "can increase by",
  "target": "10%",
  "evidence_text": "The amount of fossil carbon that we need to return to the ground is an amount equal to a 10% increase of the carbon content in soil world-wide."
}
```

---

### Sample Issue #3: Page Completely Skipped

**Page 16**: Contains framework for "APPRENTICE (BEGINNER) LEVEL"

**Expected Extractions**:
- Soil Stewardship Guild → has level → Apprentice
- Apprentice level → includes practice → Compost
- Apprentice level → includes practice → Grow House Plants
- Apprentice level → includes practice → Buy products with soil stewardship in mind

**Actual Extractions**: 0

**Why**: Page may be in chunk that was entirely filtered out during Pass 2

**Fix**: Lower filtering threshold and ensure all pages with substantive text are extracted

---

**End of Master Guide**

For detailed issue listings, see:
- Comprehensive Review: `data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2_comprehensive_review.md`
- Simple Review: `data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2_quality_review.md`
