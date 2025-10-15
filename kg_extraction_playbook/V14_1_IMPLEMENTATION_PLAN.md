# V14.1 Implementation Plan - Path to A+ Grade

**Date**: 2025-10-14
**Status**: DESIGN PHASE
**Target**: A+ grade (<3.0% issue rate, ideally ~2.9%)
**Strategy**: Hybrid approach - restore extraction volume + targeted issue fixes

---

## 🎯 Success Criteria

**Quantitative Targets:**
- Extract ~870 relationships (restore V13.1 volume)
- Reduce to ~25 total issues (down from V14.0's 65)
- Achieve <3.0% issue rate (targeting 2.9%)
- Grade: **A or A+**

**Qualitative Targets:**
- Zero critical issues (maintained)
- Zero high-priority issues (maintained from V14.0)
- <5 medium-priority issues
- All philosophical/metaphorical relationships filtered
- All redundant 'is-a' relationships consolidated
- All vague entities caught and blocked

---

## 📊 V14.0 Analysis Summary

**What Worked (Keep):**
- ✅ Pass 2 claim classification (FACTUAL/NORMATIVE/PHILOSOPHICAL scoring)
- ✅ Eliminated ALL high-priority issues (8 → 0)
- ✅ MetadataFilter module (working, just needs validation)
- ✅ ConfidenceFilter architecture (concept is sound)
- ✅ Pronoun resolver enhancement (integrated into ConfidenceFilter)

**What Failed (Fix):**
- ❌ Pass 1 prompt TOO RESTRICTIVE (596 vs ~870 candidates)
- ❌ No deduplication → 25 redundant 'is-a' relationships
- ❌ Philosophical filter weak → still 18 philosophical relationships
- ❌ Vague entity blocker incomplete → still 8 vague entities
- ❌ Predicate Normalizer V1.4 bugs → 12 inconsistencies
- ❌ List splitter semantic issues → 5 awkward splits

---

## 🔧 V14.1 Changes (8 Total)

### **CHANGE 001: Fix Pass 1 Prompt (Relaxed Extraction)**
**Priority**: CRITICAL
**Impact**: +270 relationships (restore to ~870)
**Target Fix**: Extract more relationships while maintaining quality

**Problem**:
V14.0's Pass 1 prompt was too restrictive with entity specificity requirements, causing the LLM to skip 270 valid relationships.

**Solution**:
Rebalance the prompt to be **permissive in extraction, strict in evaluation**:

1. **RELAX entity specificity** (make it guidance, not requirements):
   - Change: "Extract ONLY entities that are..." → "PREFER entities that are..."
   - Keep the guidance but make it non-blocking
   - Add: "When in doubt, extract the relationship - Pass 2 will evaluate quality"

2. **KEEP extraction scope** (domain knowledge vs metadata):
   - This guidance was valuable
   - Just make it less aggressive

3. **ADD permissive extraction principle**:
   - "Be generous in extraction. Extract relationships even if you're uncertain about specificity."
   - "Pass 2 will filter low-quality relationships - your job is comprehensive extraction"

4. **REMOVE blocking language**:
   - Remove: "DO NOT extract...", "EXCLUDE...", "NEVER extract..."
   - Replace with: "Be cautious with...", "Pass 2 will scrutinize..."

**Expected Impact**:
- Extract ~870 candidates (up from 596)
- Pass 2 will filter appropriately
- Dilution effect will help issue rate

**Files Modified**:
- `kg_extraction_playbook/prompts/pass1_extraction_v14_1.txt`

---

### **CHANGE 002: Add Semantic Deduplication Module**
**Priority**: CRITICAL
**Impact**: -20 issues (eliminate redundant 'is-a' relationships)
**Target Fix**: 25 redundant relationships → ~5 remaining

**Problem**:
25 semantically redundant 'is-a' relationships (4.2% of issues). Examples:
- "X is-a source of Y for Z1", "X is-a source of Y for Z2", "X is-a source of Y for Z3"
- Multiple variations of same relationship with slight wording differences

**Solution**:
Create new `SemanticDeduplicator` module that:

1. **Embed all relationships** using sentence-transformers:
   - Use `all-MiniLM-L6-v2` or OpenAI embeddings
   - Create embedding for full relationship triple: (source, predicate, target)

2. **Find semantic duplicates** using cosine similarity:
   - Threshold: 0.85-0.90 similarity = duplicates
   - Group relationships by source entity first (for efficiency)

3. **Consolidation rules**:
   - Keep the relationship with HIGHEST p_true score
   - If 'is-a source of X for [Y1, Y2, Y3]', consolidate to 'provides X'
   - If multiple 'is-a' with same source/target, keep most general predicate
   - Preserve all metadata from kept relationship

4. **Special handling**:
   - Don't deduplicate if predicates are different (even if entities same)
   - Don't deduplicate if context significantly different
   - Add deduplication flag to metadata

**Expected Impact**:
- Remove ~20 redundant relationships
- Issue count: 65 → 45
- Will work on expanded 870 relationship set too

**Files Created**:
- `src/knowledge_graph/postprocessing/universal/semantic_deduplicator.py`

**Files Modified**:
- `src/knowledge_graph/postprocessing/universal/__init__.py` (export)
- `src/knowledge_graph/postprocessing/pipelines/book_pipeline.py` (add to pipeline)

**Priority in Pipeline**: 115 (after Deduplicator, before ConfidenceFilter)

---

### **CHANGE 003: Add Vague Metaphor Pattern Blocking**
**Priority**: HIGH (was CRITICAL, but simpler than originally planned)
**Impact**: -15 issues (filter vague metaphorical patterns)
**Target Fix**: 18 vague metaphors → ~3 remaining

**Problem**:
18 issues from "Over-Extraction of Abstract/Philosophical Relationships" - but these are specifically VAGUE metaphorical patterns, not all philosophical content.

**Key Learning from V13.1**:
V13.1 showed that **classification > filtering** for philosophical content. V13.1 (A-) was better than V12 precisely because we STOPPED penalizing philosophical claims and just classified them. The problem isn't philosophical relationships per se, but specifically vague metaphorical patterns with no concrete information.

**Solution - Targeted Pattern Blocking in Pass 1**:

Add "🚫 RED FLAG PATTERNS" section to Pass 1 prompt:

```markdown
## 🚫 RED FLAG PATTERNS - Prefer to Avoid

These patterns are too vague and provide no concrete information. If you encounter them, try to extract a more specific relationship from context, or skip if no concrete information is present:

❌ **Vague Metaphorical Patterns**:
- "X is sacred" → No concrete information, purely subjective
- "X is the answer to Y" → Metaphor, extract specific mechanism if available
- "X is the key to Y" → Metaphor, extract specific relationship if available
- "X is medicine" → Only extract if literal pharmaceutical context
- "X is the foundation of Y" → Extract specific functional relationship instead
- "X transforms Y" → Extract specific change/effect instead

✅ **BUT: Keep Concrete Philosophical Claims**:
- "regenerative agriculture addresses climate change" ✅ (concrete claim)
- "soil stewardship improves ecosystem health" ✅ (testable)
- "biochar enhances soil carbon sequestration" ✅ (specific mechanism)

**Decision Rule**: Does this relationship provide CONCRETE, ACTIONABLE information?
- YES → Extract it (even if philosophical)
- NO → Skip or find more specific alternative
```

**Why NO Separate Module**:
1. **ConfidenceFilter already handles this**: PHILOSOPHICAL_CLAIM and METAPHOR flags → 0.85 threshold
2. **Pass 2 classification working**: Correctly identifying these patterns
3. **Don't want to filter all philosophical content**: Some is valuable domain knowledge
4. **Simpler is better**: Just guide Pass 1 extraction, let existing pipeline handle it

**Expected Impact**:
- Pass 1 extracts fewer vague metaphors (~15 fewer)
- ConfidenceFilter catches any remaining (0.85 threshold)
- Concrete philosophical claims still extracted and kept
- Issue count: 45 → 30 (from previous fix)

**Files Modified**:
- `kg_extraction_playbook/prompts/pass1_extraction_v14_1.txt` (add red flag patterns section)

**No New Modules**: Use existing classification + ConfidenceFilter architecture

---

### **CHANGE 004: Fix Predicate Normalizer V1.4 Bugs**
**Priority**: HIGH
**Impact**: -10 issues (prevent normalization inconsistencies)
**Target Fix**: 12 inconsistencies → ~2 remaining

**Problem**:
12 predicate normalization inconsistencies introduced by V1.4. Issues:
- Inconsistent 'is' → 'is-a' mapping (some normalized, some not)
- Semantic mismatches after normalization
- Modal verb preservation causing unexpected behavior

**Solution - Three Fixes**:

#### 4A. Consistent 'is' normalization:
- Rule: ALL 'is' variations must map to 'is-a' consistently
- Add normalization logging to track what's being changed
- Add unit tests for 'is' variations

#### 4B. Semantic validation enhancement:
Improve `validate_semantic_compatibility()`:
- After normalization, check if relationship still makes sense
- Use simple heuristics:
  - Person + physical action → likely valid
  - Abstract concept + physical action → likely invalid
  - Organization + 'produces' → valid
  - Individual + 'produces' (abstract) → suspect

#### 4C. Modal verb preservation review:
- Add logging for modal verb preservation
- Verify: "can work with" → should preserve "can"
- Verify: "may provide" → should preserve "may"
- If causing issues, add special handling for "can be-a" patterns

#### 4D. Expand normalization rules:
Add 30+ new normalization rules for common predicate families:
- All 'contain' forms → 'contains'
- All 'provide' forms → 'provides'
- All 'support' forms → 'supports'
- All 'enhance' forms → 'enhances'
- All 'release' forms → 'releases'

**Expected Impact**:
- Prevent ~10 inconsistencies
- Issue count: 30 → 20

**Files Modified**:
- `src/knowledge_graph/postprocessing/universal/predicate_normalizer.py`

**Version**: Bump to V1.5

---

### **CHANGE 005: Expand Vague Entity Blocker**
**Priority**: HIGH
**Impact**: -6 issues (catch vague/generic entities)
**Target Fix**: 8 vague → ~2 remaining

**Problem**:
Still 8 vague entity issues despite VagueEntityBlocker + Pass 1 guidance. Missing patterns:
- "the answer", "the key", "the way", "the solution"
- Gerund phrases: "being connected to X", "having Y"
- Overly generic: "individuals", "people" (when more specific entity available)

**Solution - Three Enhancements**:

#### 5A. Expand VagueEntityBlocker patterns:
Add to existing pattern list:
```python
vague_patterns = [
    # Existing patterns
    r'^the answer$',
    r'^the key$',
    r'^the way$',
    r'^the solution$',
    r'^the foundation$',
    r'^the source$',

    # Gerund phrases
    r'^being \w+',
    r'^having \w+',
    r'^doing \w+',

    # Overly generic
    r'^individuals$',
    r'^people$',
    r'^things$',
    r'^stuff$',
    r'^elements$',
    r'^aspects$',
]
```

#### 5B. Context-aware specificity check:
- If entity is generic (e.g., "individuals") AND more specific entity mentioned in context (e.g., "farmers"), block generic
- Use NER to find more specific alternatives
- Flag for manual review if uncertain

#### 5C. Gerund phrase detection:
- Use POS tagging to detect gerund phrases (VBG at start)
- Block gerund phrases as entities (they're usually predicates or actions)
- Exception: if gerund is a well-known process name (e.g., "composting")

**Expected Impact**:
- Catch ~6 vague entities
- Issue count: 20 → 14

**Files Modified**:
- `src/knowledge_graph/postprocessing/universal/vague_entity_blocker.py`

**Version**: Bump to V1.2

---

### **CHANGE 006: Improve List Splitter Semantic Coherence**
**Priority**: MEDIUM
**Impact**: -4 issues (prevent awkward splits)
**Target Fix**: 5 awkward splits → ~1 remaining

**Problem**:
5 awkward list splitting issues. Examples:
- Splitting "land and soil" (semantic unit)
- Inconsistent splitting (splits some items but not all)
- Breaking compound objects that should stay together

**Solution - Two Enhancements**:

#### 6A. Semantic unit detection:
Before splitting, check if items form semantic unit:
- Use word embeddings to check similarity between split candidates
- If cosine similarity > 0.7, DON'T split (e.g., "land and soil", "farmers and ranchers")
- Maintain as compound entity

#### 6B. Consistency enforcement:
If splitting "A, B and C":
- EITHER split all three → three relationships
- OR don't split at all → one relationship with list
- NO partial splits

#### 6C. Dependency parsing:
Use spaCy dependency parsing to identify true list structures:
- True list: "A, B, and C" with conj dependencies
- Compound object: "A and B" as single unit
- Apply splitting only to true lists

**Expected Impact**:
- Prevent ~4 awkward splits
- Issue count: 14 → 10

**Files Modified**:
- `src/knowledge_graph/postprocessing/universal/list_splitter.py`

**Version**: Bump to V1.2

---

### **CHANGE 007: Enhance Pass 2 Abstract Relationship Penalization**
**Priority**: MEDIUM
**Impact**: -5 issues (lower scores for abstract relationships)
**Target Fix**: Additional filtering at evaluation stage

**Problem**:
Pass 2 isn't penalizing abstract/philosophical relationships enough. They're getting p_true scores high enough to pass ConfidenceFilter.

**Solution**:
Enhance Pass 2 prompt with explicit abstract penalization:

1. **Add guidance section**:
"Low-Utility Abstract Relationships (score 0.3-0.5):
- Philosophical claims without concrete information
- Metaphorical statements that don't provide actionable knowledge
- Relationships that describe spiritual/mystical properties
- 'X is the key to Y' (unless literal key)
- 'X is sacred' (unless documenting cultural belief as fact)

Even if textually accurate, these relationships should receive LOW p_true scores because they don't provide concrete, verifiable information useful for knowledge graph queries."

2. **Add few-shot examples**:
Show examples of abstract relationships that should score low, even if textually accurate.

**Expected Impact**:
- Lower p_true scores for abstract relationships
- More will be filtered by ConfidenceFilter
- Issue count: 10 → 5

**Files Modified**:
- `kg_extraction_playbook/prompts/pass2_evaluation_v14_1.txt`

---

### **CHANGE 008: Add Extraction Pipeline Metrics**
**Priority**: LOW (quality-of-life improvement)
**Impact**: Better debugging and validation

**Problem**:
Can't easily see Pass 1/Pass 2/Pass 2.5 filtering breakdown, making debugging difficult.

**Solution**:
Add comprehensive metrics logging:

1. **Log to metadata**:
```python
'pipeline_metrics': {
    'pass1_candidates': 870,
    'pass2_evaluated': 870,
    'pass2_acceptance_rate': 1.0,
    'pass2_5_input': 870,
    'pass2_5_output': 850,
    'pass2_5_removal_rate': 0.023,
    'modules_executed': 14,
    'filtering_by_module': {
        'PhilosophicalBlocker': 15,
        'SemanticDeduplicator': 20,
        'ConfidenceFilter': 5,
        ...
    }
}
```

2. **Add progress logging**:
- Log after each major stage
- Show acceptance/rejection counts
- Help identify which modules are working

**Expected Impact**:
- Easier debugging
- Better validation of changes
- No direct quality impact

**Files Modified**:
- `scripts/extract_kg_v14_1_book.py` (add metrics collection)

---

## 📋 Implementation Checklist

### Phase 1: Core Extraction & Deduplication (Critical)
- [ ] Change 001: Fix Pass 1 prompt (relax extraction)
- [ ] Change 002: Add SemanticDeduplicator module
- [ ] Change 008: Add pipeline metrics logging
- [ ] Create `extract_kg_v14_1_book.py` script
- [ ] Test extraction on Soil Stewardship Handbook
- [ ] Verify ~870 relationships extracted

### Phase 2: Filtering & Quality (Critical/High)
- [ ] Change 003: Add red flag patterns to Pass 1 prompt (vague metaphors)
- [ ] Change 004: Fix Predicate Normalizer V1.4 → V1.5
- [ ] Change 005: Expand VagueEntityBlocker patterns
- [ ] Change 006: Improve ListSplitter semantic coherence
- [ ] Update book pipeline with new modules (just SemanticDeduplicator)
- [ ] Test postprocessing on V14.1 output

### Phase 3: Evaluation Enhancement (Medium)
- [ ] Change 007: Enhance Pass 2 abstract penalization
- [ ] Test Pass 2 scoring on sample relationships
- [ ] Verify low scores for abstract relationships

### Phase 4: Validation (Final)
- [ ] Run V14.1 extraction
- [ ] Run Reflector analysis
- [ ] Compare V14.1 vs V13.1 vs V14.0
- [ ] Validate targets achieved
- [ ] Document results

---

## 🎯 Expected V14.1 Results

**Quantitative Predictions:**
| Metric | V14.0 | V14.1 Target | Improvement |
|--------|-------|--------------|-------------|
| Relationships | 603 | ~850-870 | +44% |
| Total Issues | 65 | ~25 | -62% |
| Issue Rate | 10.78% | ~2.9% | -73% |
| Grade | B+ | A or A+ | +1-2 grades |
| Critical | 0 | 0 | = |
| High | 0 | 0 | = |
| Medium | 18 | <5 | -72% |
| Mild | 47 | ~20 | -57% |

**Issue Category Predictions:**
- Redundant 'is-a': 25 → ~5 (-80%)
- Philosophical: 18 → ~3 (-83%)
- Predicate norm: 12 → ~2 (-83%)
- Vague entities: 8 → ~2 (-75%)
- List splitting: 5 → ~1 (-80%)
- **Total**: 68 → ~13 (-81%)

**Why this will work:**
1. **Volume restoration** (+270 rels) creates dilution effect
2. **Deduplication** (-20 rels, -20 issues) = direct 1:1 issue reduction
3. **Philosophical blocker** (-15 issues) = 23% of current issues
4. **Multiple filters compound** = issues caught by multiple mechanisms
5. **Remaining issues distributed** across mild categories

---

## 🚀 Next Steps

1. **Implement Phase 1** (core extraction + deduplication)
   - Critical path to restore extraction volume
   - Deduplication has biggest single impact

2. **Test after Phase 1**
   - Should already see major improvement
   - Validates approach before continuing

3. **Implement Phases 2-3** (filtering + evaluation)
   - Build on Phase 1 foundation
   - Each module provides incremental improvement

4. **Run full validation** (Phase 4)
   - Compare against all previous versions
   - Document learnings for future iterations

**Estimated Implementation Time**: 3-4 hours total
- Phase 1: 1.5 hours
- Phase 2: 1.5 hours
- Phase 3: 0.5 hours
- Phase 4: 0.5 hours

---

**Ready to implement? Let's start with Phase 1!** 🎯
