# 🔬 Deep Investigation & A++ Extraction System Design

**Date**: October 11, 2025
**Investigation**: Why 73.3% of pages were lost in Run #2
**Goal**: Design an A++ extraction system that is generic and reusable

---

## 📊 Investigation Summary

### Critical Findings

#### Finding #1: **73.3% Page Loss** (CRITICAL)
- **Data**: 45 pages chunked, but only 12 pages have relationships (26.1% coverage)
- **Lost pages**: 33 pages with ZERO extractions
- **Problem**: Main content pages (4-35) have almost no extractions
- **Problem**: References pages (36-41) dominate with 87% of all relationships (148/170)

**This is backwards!** References should contribute less, main content should dominate.

#### Finding #2: **Prompt is Too Restrictive**
- **Master Guide says**: "Extract EVERYTHING...don't worry if correct...be exhaustive"
- **Our improved prompt says**: "⚠️ CRITICAL: You MUST ONLY extract entities that ACTUALLY APPEAR in the text"
- **Result**: LLM became too conservative and skipped most content

**Hypothesis**: The restrictive prompt discouraged extraction from main content pages.

#### Finding #3: **Entity Presence Validation Too Aggressive**
- **Filtered**: 86 out of 306 candidates (28.1%)
- **Threshold**: 70% word match for multi-word entities
- **Master Guide**: No entity presence validation layer exists
- **Result**: Lost valid relationships due to overly strict matching

#### Finding #4: **Batch Size Too Large → Token Limit Error**
- **Batch size**: 50 relationships
- **Error**: Hit 16,384 token output limit in Pass 2 batch 2
- **Lost**: 50 relationships (22.7% of valid candidates)
- **Master Guide**: Uses batch size of 50 but may need reduction for books

#### Finding #5: **Incomplete Entity Extraction**
- **Examples**:
  - "soil" instead of "soil carbon content"
  - "10%" without context instead of "10% increase in global soil carbon"
  - Missing qualifiers: organic, global, annual, daily

**Root cause**: Prompt didn't emphasize complete multi-word concept extraction enough.

---

## 🎯 Root Cause Analysis

### Why did main content pages get skipped?

**Theory #1: Over-Restrictive Prompt** (MOST LIKELY)
```
❌ Our prompt: "MUST ONLY extract entities that ACTUALLY APPEAR in the text"
✅ Master guide: "Extract EVERYTHING...don't worry if correct...be exhaustive"
```

**Impact**: LLM interpreted "ACTUALLY APPEAR" too literally and skipped:
- Concepts that span multiple sentences
- Entities that are paraphrased
- Relationships that require inference

**Theory #2: Entity Presence Validation Filtered Too Much**
- 28.1% filtering rate is high
- Many valid relationships may have been lost
- 70% word match threshold may be too strict

**Theory #3: Prompt Discouraged "Easy" Extractions**
- References pages have simple, explicit relationships (X authored Y, X located in Y)
- Main content pages have complex conceptual relationships
- Restrictive prompt made LLM skip complex relationships

---

## 🏆 A++ Extraction System Design

### Core Principles (From Master Guide v3.2.2)

1. **Two-Pass Architecture**
   - ✅ Pass 1: HIGH RECALL - extract EVERYTHING
   - ✅ Pass 2: HIGH PRECISION - evaluate each relationship

2. **Simple > Complex**
   - ✅ Simple extraction prompt for Pass 1
   - ✅ Dual-signal evaluation for Pass 2
   - ✅ Pattern priors from existing graph

3. **Evidence-Linked**
   - ✅ Every relationship linked to exact text span
   - ✅ Evidence traceable to source

4. **Robust & Generic**
   - ✅ NDJSON format for partial failure recovery
   - ✅ Batching with proper size (25-30)
   - ✅ Caching with scorer-aware keys

---

## 🔧 A++ System Design Decisions

### Decision #1: Return to Master Guide Prompts

**Pass 1 Prompt** (From Master Guide):
```
Extract ALL relationships you can find in this text.
Don't worry about whether they're correct or make sense.
Just extract everything - we'll validate later.

For each relationship, provide:
- source entity
- relationship type
- target entity
- the exact quote supporting this (important!)

Be exhaustive. It's better to extract too much than too little.
```

**Why**: Simple, encouraging, high recall. No restrictions that make LLM conservative.

**Pass 2 Prompt** (From Master Guide):
```
Evaluate these extracted relationships.

For EACH relationship, provide TWO INDEPENDENT evaluations:

1. TEXT SIGNAL (ignore world knowledge):
   - How clearly does the text state this relationship?
   - Score 0.0-1.0 based purely on text clarity

2. KNOWLEDGE SIGNAL (ignore the text):
   - Is this relationship plausible given world knowledge?
   - What types are the source and target entities?
   - Score 0.0-1.0 based purely on plausibility

If the signals conflict (text says X but knowledge says Y):
- Set signals_conflict = true
- Include conflict_explanation
- Include suggested_correction if known

CRITICAL: Return candidate_uid UNCHANGED in every output object.
```

**Why**: Separates concerns, provides calibratable signals, catches conflicts.

### Decision #2: Remove Entity Presence Validation

**Rationale**:
- Master guide doesn't have this validation layer
- 28.1% filtering rate is too aggressive
- False positives are better than false negatives (high recall in Pass 1)
- Pass 2 dual-signal evaluation will catch bad relationships

**Alternative**: Keep validation but make it optional and less strict (50% threshold instead of 70%).

### Decision #3: Reduce Batch Size to 25

**Rationale**:
- Batch of 50 hit 16,384 token limit
- Lost 50 relationships
- Smaller batches = safer, more recoverable
- Master guide uses 50 but books may be more verbose

**Implementation**: `batch_size=25` in `evaluate_batch_robust()`

### Decision #4: Add Multi-Word Concept Extraction Guidance

**Add to Pass 1 prompt**:
```
IMPORTANT: Extract entities as COMPLETE concepts:
✅ "soil carbon content" NOT just "soil"
✅ "organic matter" NOT just "matter"
✅ "fossil fuel emissions" NOT just "emissions"
✅ "global temperature increase" NOT just "temperature"

Keep adjectives/modifiers that change meaning:
- Scope: global, worldwide, regional, local
- Type: organic, chemical, natural, synthetic
- Time: annual, daily, long-term, short-term
- State: active, stable, labile, total
```

**Why**: Addresses the incomplete entity extraction issue.

### Decision #5: Add Quantitative Relationship Handling

**Add to Pass 1 prompt**:
```
SPECIAL: For claims with numbers/percentages:

1. Extract the FULL measurable quantity
   ✅ "soil carbon content → can increase by → 10%"
   ❌ "soil → is increased by → 10%"

2. Include SCOPE
   ✅ "global soil carbon" or "worldwide soil carbon"
   ❌ just "soil carbon"

3. Capture EQUIVALENCE relationships
   When text says "amount equal to" or "same as":
   - Extract as: (Thing A, equals, Thing B)

Example:
Text: "Goal is an amount equal to a 10% increase in soil carbon worldwide"
Extract:
- fossil carbon sequestration goal → equals → 10% global soil carbon increase
- soil carbon content → can increase by → 10% (scope: worldwide)
```

**Why**: Fixes the 9 "incorrect" relationships with numbers/percentages.

### Decision #6: Add Page Coverage Monitoring During Pass 1

**Implementation**:
```python
# Track which pages have extractions during Pass 1
pages_with_candidates = set()
for candidate in all_candidates:
    page = candidate.evidence.get('page_number')
    if page:
        pages_with_candidates.add(page)

# Alert if many pages have zero extractions
total_substantive_pages = len([p for p, t in pages_with_text if len(t.split()) >= 50])
coverage = len(pages_with_candidates) / total_substantive_pages
if coverage < 0.7:
    logger.warning(f"Low page coverage: {coverage:.1%} ({len(pages_with_candidates)}/{total_substantive_pages})")
```

**Why**: Early detection of extraction issues.

### Decision #7: Implement Chunk-Level Extraction Monitoring

**Problem**: We chunk 45 pages but only 12 have extractions.

**Solution**: Track extractions per chunk during Pass 1:
```python
for i, (page_nums, chunk) in enumerate(text_chunks):
    candidates = pass1_extract_book(chunk, doc_sha256, page_nums)

    if len(candidates) == 0:
        logger.warning(f"Chunk {i} (pages {page_nums}) extracted ZERO candidates")
    elif len(candidates) < 3:
        logger.warning(f"Chunk {i} (pages {page_nums}) extracted only {len(candidates)} candidates")

    all_candidates.extend(candidates)
```

**Why**: Identifies problematic chunks in real-time.

---

## 📋 Complete A++ Extraction Pipeline

### Stage 1: PDF Text Extraction & Chunking
```python
# Extract text from PDF (pdfplumber)
full_text, pages_with_text = extract_text_from_pdf(pdf_path)

# Smart chunking (800 words, 100 overlap, min 50 words/page)
text_chunks = chunk_book_text(pages_with_text, chunk_size=800, overlap=100, min_page_words=50)

# Verify coverage
logger.info(f"Pages included: {len(pages_included)}/{len(pages_with_text)} ({coverage:.1%})")
```

### Stage 2: Pass 1 - High Recall Extraction
```python
# Simple prompt: "Extract EVERYTHING...don't worry if correct...be exhaustive"
all_candidates = []
pages_with_candidates = set()

for i, (page_nums, chunk) in enumerate(text_chunks):
    candidates = pass1_extract_book(chunk, doc_sha256, page_nums)

    # Monitor extractions per chunk
    if len(candidates) == 0:
        logger.warning(f"Chunk {i} (pages {page_nums}): ZERO candidates")

    # Track page coverage
    for c in candidates:
        if c.evidence.get('page_number'):
            pages_with_candidates.add(c.evidence['page_number'])

    all_candidates.extend(candidates)

logger.info(f"Pass 1: {len(all_candidates)} candidates from {len(pages_with_candidates)} pages")
```

### Stage 3: Type Validation (Soft - Master Guide)
```python
# Only filter KNOWN type violations (not unknowns)
valid_candidates = []
for candidate in all_candidates:
    validated = type_validate(candidate)
    if not validated.flags.get("TYPE_VIOLATION"):
        valid_candidates.append(validated)

logger.info(f"Type validation: {len(valid_candidates)} valid ({len(all_candidates)-len(valid_candidates)} filtered)")
```

### Stage 4: Pass 2 - Dual-Signal Evaluation
```python
# Batch size: 25 (not 50 - avoid token limit)
validated_relationships = []
for batch in chunks(valid_candidates, size=25):
    evaluations = evaluate_batch_robust(
        batch=batch,
        model="gpt-4o-mini",
        prompt=DUAL_SIGNAL_EVALUATION_PROMPT,
        prompt_version="v3.2.2_A++",
        format="ndjson"
    )
    validated_relationships.extend(evaluations)

logger.info(f"Pass 2: {len(validated_relationships)} relationships evaluated")
```

### Stage 5: Post-Processing & Calibration
```python
# Canonicalize, compute pattern priors, geo validation
alias_resolver = SimpleAliasResolver()
priors = SmoothedPatternPriors(existing_graph) if existing_graph else None

for rel in validated_relationships:
    # Save surface forms
    rel.evidence["source_surface"] = rel.source
    rel.evidence["target_surface"] = rel.target

    # Canonicalize
    rel.source = alias_resolver.resolve(rel.source)
    rel.target = alias_resolver.resolve(rel.target)

    # Generate claim UID
    rel.claim_uid = generate_claim_uid(rel)

    # Compute pattern prior
    rel.pattern_prior = priors.get_prior(rel.source, rel.relationship, rel.target) if priors else 0.5

    # Calibrated probability
    rel.p_true = compute_p_true(rel.text_confidence, rel.knowledge_plausibility, rel.pattern_prior, rel.signals_conflict)

    # Geo validation
    geo_validation = validate_geographic_relationship(rel)
    if geo_validation.get("valid") is False:
        rel.p_true = max(0.0, rel.p_true - geo_validation.get("confidence_penalty", 0.0))
```

---

## ✅ Implementation Checklist

### High Priority (Must Fix)
- [ ] Replace Pass 1 prompt with master guide simple prompt
- [ ] Replace Pass 2 prompt with master guide dual-signal prompt
- [ ] Reduce batch size from 50 to 25
- [ ] Remove entity presence validation (or make it 50% threshold + optional)
- [ ] Add multi-word concept extraction guidance to Pass 1
- [ ] Add quantitative relationship guidance to Pass 1
- [ ] Add chunk-level extraction monitoring

### Medium Priority (Should Fix)
- [ ] Add page coverage monitoring during Pass 1
- [ ] Alert if >30% of pages have zero extractions
- [ ] Track extractions per page during Pass 1
- [ ] Log warning for chunks with <3 candidates

### Low Priority (Nice to Have)
- [ ] Add evidence window expansion to 2000 chars (from 1500)
- [ ] Add relationship type distribution monitoring
- [ ] Add entity type distribution monitoring

---

## 📊 Expected Results from A++ System

### Coverage Metrics
| Metric | Run #1 | Run #2 | Run #3 (A++) | Target |
|--------|--------|--------|--------------|--------|
| Total Relationships | 493 | 170 | 350-450 | 400+ |
| Page Coverage | 34% | 26% | **80%+** | 80%+ |
| Main Content % | Unknown | 13% | **60%+** | 60%+ |
| References % | Unknown | 87% | **40%** | 30-40% |

### Quality Metrics
| Metric | Run #1 | Run #2 | Run #3 (A++) | Target |
|--------|--------|--------|--------------|--------|
| Incorrect % | 37.7% | 5.3% | **<10%** | <10% |
| High Confidence % | 93.5% | 97.6% | **90%+** | 85%+ |
| Entity in Evidence | ~62% | ~72% | **85%+** | 90%+ |

### Balanced Approach
- **Run #1**: High recall (493 rels), low precision (37.7% incorrect)
- **Run #2**: High precision (5.3% incorrect), low recall (170 rels, 26% coverage)
- **Run #3**: **BALANCED** - Good recall (400+ rels, 80% coverage) + Good precision (<10% incorrect)

---

## 🎯 Generic & Reusable Design

### Adaptable to Different Content Types

**Books**:
- ✅ Page-aware chunking
- ✅ Chapter detection (future)
- ✅ Figure/table extraction (future)

**Podcasts** (Already Implemented):
- ✅ Word-level timestamps
- ✅ Speaker attribution
- ✅ Audio deep links

**Articles/Papers**:
- ✅ Section-aware chunking
- ✅ Citation extraction
- ✅ Abstract/conclusion emphasis

**Videos**:
- ✅ Frame-level timestamps
- ✅ Visual entity extraction
- ✅ Multimodal fusion

### Configurable Parameters

```python
EXTRACTION_CONFIG = {
    # Content-specific
    "content_type": "book",  # book, podcast, article, video
    "chunk_size": 800,  # words per chunk
    "chunk_overlap": 100,  # word overlap
    "min_page_words": 50,  # skip pages with fewer words

    # Extraction behavior
    "pass1_model": "gpt-4o-mini",
    "pass2_model": "gpt-4o-mini",
    "batch_size": 25,
    "enable_entity_validation": False,  # Toggle entity presence check
    "entity_match_threshold": 0.5,  # If enabled, 50% word match

    # Quality thresholds
    "min_text_confidence": 0.0,  # No filtering in Pass 1
    "min_p_true": 0.5,  # Filter in final output
    "max_conflicts": 999,  # Keep all conflicts for review

    # Coverage monitoring
    "min_page_coverage": 0.7,  # Alert if <70% pages have extractions
    "min_chunk_candidates": 3,  # Warn if chunk extracts <3
}
```

### Extensible Architecture

**Add new content types**:
```python
class BookExtractor(BaseExtractor):
    def chunk_content(self, content):
        return chunk_book_text(content, self.config)

    def extract_evidence(self, rel, content):
        return extract_book_evidence(rel, content)

class PodcastExtractor(BaseExtractor):
    def chunk_content(self, content):
        return chunk_transcript(content, self.config)

    def extract_evidence(self, rel, content):
        return extract_podcast_evidence_with_timestamps(rel, content)
```

**Add new validators**:
```python
validators = [
    TypeValidator(config),
    GeoValidator(config),
    QuantitativeValidator(config),  # NEW
    TemporalValidator(config),  # NEW
]

for rel in relationships:
    for validator in validators:
        rel = validator.validate(rel)
```

---

## 🚀 Implementation Plan

### Phase 1: Fix Critical Issues (30 minutes)
1. Update Pass 1 prompt (use master guide version)
2. Update Pass 2 prompt (use master guide version)
3. Reduce batch size to 25
4. Remove entity presence validation

### Phase 2: Add Improvements (30 minutes)
1. Add multi-word concept guidance to Pass 1
2. Add quantitative relationship guidance to Pass 1
3. Add chunk-level monitoring
4. Add page coverage alerts

### Phase 3: Test & Run (15 minutes)
1. Quick syntax check
2. Run extraction script
3. Monitor logs for issues
4. Wait for completion (~20 minutes)

**Total time**: ~1 hour 15 minutes

---

## 🎉 A++ System Goals

### What Makes It A++?

1. **✅ High Coverage** - Extracts from 80%+ of pages
2. **✅ High Quality** - Maintains <10% error rate
3. **✅ Balanced** - Both recall and precision are good
4. **✅ Generic** - Works for books, podcasts, articles, videos
5. **✅ Reusable** - Configurable parameters for different use cases
6. **✅ Robust** - NDJSON format, batching, caching, monitoring
7. **✅ Evidence-Linked** - Every fact traceable to source
8. **✅ Calibrated** - p_true scores are accurate
9. **✅ Extensible** - Easy to add new validators and content types

---

## 📝 Conclusion

The A++ system returns to the master guide principles while incorporating lessons learned:

1. **Simple prompts** that encourage high recall
2. **Dual-signal evaluation** for precision
3. **Proper batch sizing** to avoid token limits
4. **Multi-word concepts** for complete extraction
5. **Quantitative handling** for numbers/percentages
6. **Coverage monitoring** to detect issues early
7. **Generic architecture** for reusability

**Expected outcome**: 400+ relationships, 80% page coverage, <10% error rate.

Let's implement it!
