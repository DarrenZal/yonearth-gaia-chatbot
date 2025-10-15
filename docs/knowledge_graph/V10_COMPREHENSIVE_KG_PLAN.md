# V10 Comprehensive Knowledge Graph Extraction Plan

**Date**: October 13, 2025
**Goal**: Extract ALL valuable factual relationships (not hitting arbitrary numbers)
**Focus**: Meaningful data, information, knowledge, and wisdom

---

## 🎯 V10 Objectives

### Quality Targets
- **Issue rate**: <3% (A++ grade for general knowledge graph)
- **High confidence**: 80%+ of relationships
- **Attribution**: 100% (maintain from V9)
- **Classification**: 100% (maintain from V9)

### Comprehensiveness Targets (Evidence-Based)
V9 is missing **valuable factual relationships** that V8 extracted:

| Relationship Type | V8 Count | V9 Count | Missing | Value |
|-------------------|----------|----------|---------|-------|
| **Bibliographic** (authored, published) | 304 | 127 | **177** | ✅ Essential |
| **Categorical** (is-a, is) | 93 | 27 | **66** | ✅ Important |
| **Compositional** (contains, includes) | 29 | 10 | **19** | ✅ Important |
| **Functional** (provides, produces, enhances) | 32 | 8 | **24** | ✅ Important |
| **Organizational** (affiliated with, collaborates) | 15 | 3 | **12** | ✅ Important |

**Total missing valuable relationships**: ~298

**Target**: Extract these without lowering quality standards

---

## 📊 V9 Analysis Summary

### What V9 Did Well ✅
1. **100% attribution coverage** - every claim traced to source
2. **100% classification coverage** - all statements labeled by type
3. **99.3% classification accuracy** - only 2 misclassifications
4. **Reduced true issues** to 5.8% (24 issues) vs V8's 8.35% (91 issues)
5. **List splitter inheritance** - automatic attribution + classification

### What V9 Missed ❌
1. **Bibliographic relationships** - 177 fewer citations (authored, published)
2. **Categorical relationships** - 66 fewer definitions (is-a, is)
3. **Compositional relationships** - 19 fewer (contains, includes, provides)
4. **Functional relationships** - 24 fewer (produces, enhances, stimulates)

### V9's True Quality Issues (24 total, 5.8%)
1. **Possessive pronouns** (8 issues, 1.9%) - "my people" not resolved
2. **Vague entities** (10 issues, 2.4%) - "thousands" instead of "thousands of people"
3. **Dedication parsing** (6 issues, 1.4%) - list splitter context blindness

---

## 🔧 V10 Implementation Strategy

### Phase 1: Fix Quality Issues (from Curator)

**CRITICAL Fixes** (eliminates all 6 CRITICAL + 8 HIGH issues):

1. **Dedication Parser Enhancement** (fixes 6 CRITICAL)
   - File: `modules/pass2_5_postprocessing/bibliographic_parser.py`
   - Add `process_dedication()` method
   - Detect dedication syntax: "dedicated to [person/list]"
   - Correct attribution: author → dedicatee (not vice versa)

2. **Possessive Pronoun Resolution** (fixes 8 HIGH)
   - File: `modules/pass2_5_postprocessing/pronoun_resolver.py`
   - Add `_resolve_possessive()` method
   - Handle: "my X", "our X", "their X"
   - Use context window to find antecedent

3. **Vague Entity Filter** (fixes 10 MEDIUM)
   - File: `prompts/pass1_extraction_v9.txt`
   - Add constraint: "Avoid vague quantifiers ('thousands', 'millions'). Be specific: 'thousands of people', 'millions of humans'"
   - Add constraint: "Avoid vague determiners ('the land', 'the sea'). Be specific: 'agricultural land', 'ocean ecosystems'"

**Expected Impact**: Reduces issues from 5.8% → 0.7% (3 residual issues)

---

### Phase 2: Increase Comprehensiveness

**ROOT CAUSE**: V9's Pass 1 prompt is too restrictive

**Current V9 Pass 1 Problem**:
```
EXTRACT (Be comprehensive):
✅ Factual relationships
✅ Testable claims
✅ Philosophical statements
✅ Metaphors
```

**But in practice**, V9 extracted:
- Only 114 "authored" relationships (vs V8's 214)
- Only 13 "published" relationships (vs V8's 90)
- Only 26 "is-a" relationships (vs V8's 79)

**Why?** The prompt says "be comprehensive" but doesn't give examples of what to extract.

---

### Phase 2A: Enhanced Pass 1 Prompt

**File**: `prompts/pass1_extraction_v10.txt`

**Changes**:

1. **Add Explicit Relationship Type Examples**:
```markdown
## RELATIONSHIP TYPES TO EXTRACT

### 1. BIBLIOGRAPHIC (Authorship & Publication)
✅ Extract: (Person)-[authored]->(Book/Paper)
✅ Extract: (Book)-[published by]->(Publisher)
✅ Extract: (Paper)-[published in]->(Journal, Year)
Example: "Suzuki, David. Sacred Balance. Vancouver: Greystone, 1997."
→ (David Suzuki, authored, Sacred Balance)
→ (Sacred Balance, published by, Greystone)
→ (Sacred Balance, published in, 1997)

### 2. CATEGORICAL (Definitions & Classifications)
✅ Extract: (Entity)-[is-a]->(Category)
✅ Extract: (Entity)-[is]->(Definition)
Example: "Soil is a complex ecosystem containing bacteria, fungi, and minerals."
→ (soil, is-a, complex ecosystem)
→ (soil, contains, bacteria)
→ (soil, contains, fungi)

### 3. COMPOSITIONAL (Parts & Contents)
✅ Extract: (Whole)-[contains]->(Part)
✅ Extract: (Whole)-[includes]->(Component)
✅ Extract: (Resource)-[provides]->(Service/Benefit)
Example: "Compost contains nitrogen, phosphorus, and beneficial microorganisms."
→ (compost, contains, nitrogen)
→ (compost, contains, phosphorus)
→ (compost, contains, beneficial microorganisms)

### 4. FUNCTIONAL (Processes & Effects)
✅ Extract: (Agent)-[produces]->(Product)
✅ Extract: (Action)-[enhances]->(Outcome)
✅ Extract: (Practice)-[stimulates]->(Effect)
Example: "Cover cropping enhances soil structure and prevents erosion."
→ (cover cropping, enhances, soil structure)
→ (cover cropping, prevents, erosion)

### 5. ORGANIZATIONAL (Affiliations & Roles)
✅ Extract: (Person)-[affiliated with]->(Organization)
✅ Extract: (Person)-[directs]->(Organization)
✅ Extract: (Organization)-[collaborates with]->(Organization)
Example: "Dr. Jane Smith directs the Soil Health Institute."
→ (Dr. Jane Smith, directs, Soil Health Institute)
```

2. **Add Few-Shot Examples**:
```markdown
## EXTRACTION EXAMPLES

Example 1 - Bibliographic Citation:
Input: "Capra, Fritjof. The Web of Life. New York: Anchor Books, 1996."
Output:
- (Fritjof Capra, authored, The Web of Life)
- (The Web of Life, published by, Anchor Books)
- (The Web of Life, published in, 1996)

Example 2 - Practical Instructions:
Input: "Choose appropriate seasons and species best suited for your region."
Output:
- (farmers, should choose, appropriate seasons for planting)
- (farmers, should choose, species suited for region)

Example 3 - Scientific Facts:
Input: "Mycorrhizal fungi form symbiotic relationships with plant roots, enhancing nutrient uptake."
Output:
- (mycorrhizal fungi, form symbiosis with, plant roots)
- (mycorrhizal fungi, enhance, nutrient uptake)
```

---

### Phase 2B: Adjust Pass 2 Evaluation

**File**: `prompts/pass2_evaluation_v10.txt`

**Changes**:

1. **Recalibrate Knowledge Plausibility for Bibliographic Facts**:
```markdown
## KNOWLEDGE PLAUSIBILITY CALIBRATION

**FACTUAL (0.9-1.0)**: Verifiable citations
- Bibliographic citations: (Author, authored, Book) → 0.95
- Publication info: (Book, published by, Publisher) → 0.95
- Organizational roles: (Person, directs, Organization) → 0.90

**FACTUAL (0.7-0.9)**: Verifiable relationships
- Scientific processes: (Practice, enhances, Outcome) → 0.80
- Compositional: (Whole, contains, Part) → 0.80
- Categorical: (Entity, is-a, Category) → 0.75
```

2. **Don't Over-Penalize Bibliographic Relationships**:
```markdown
## BIBLIOGRAPHIC CITATIONS - SPECIAL HANDLING

Bibliographic citations are HIGH VALUE even if they seem "list-like":
- (Author, authored, Book) ✅ Extract
- (Book, published by, Publisher) ✅ Extract
- (Book, published in, Year) ✅ Extract

Do NOT flag as "too simple" or "low value"
```

---

## 📋 V10 Implementation Checklist

### Code Changes

- [ ] **bibliographic_parser.py**: Add `process_dedication()` method
- [ ] **pronoun_resolver.py**: Add `_resolve_possessive()` method
- [ ] **list_splitter.py**: Add context-awareness for dedications
- [ ] **extract_kg_v10_book.py**: Copy V9, apply changes above

### Prompt Changes

- [ ] **pass1_extraction_v10.txt**:
  - Add explicit relationship type examples
  - Add few-shot extraction examples
  - Add constraints against vague entities

- [ ] **pass2_evaluation_v10.txt**:
  - Recalibrate knowledge_plausibility for bibliographic facts
  - Add bibliographic citation special handling
  - Don't over-penalize valuable relationship types

### System Changes

- [ ] Set `MIN_P_TRUE_THRESHOLD = None` (keep from V9)
- [ ] Maintain attribution + classification system (keep from V9)
- [ ] Ensure list splitter inherits classification/attribution (keep from V9)

---

## 🎯 Success Criteria

### Quantitative
1. **True issues**: <3% (down from V9's 5.8%)
2. **High confidence**: 80%+ (up from V9's 72.7%)
3. **Bibliographic coverage**: 200+ (vs V9's 127, V8's 304)
4. **Categorical coverage**: 60+ (vs V9's 27, V8's 93)
5. **Total relationships**: Natural outcome, not a target (likely 600-800)

### Qualitative
1. **Comprehensive extraction** of valuable factual knowledge
2. **Accurate attribution** for all relationships (100%)
3. **Proper classification** of all statements (100%)
4. **Meaningful relationships** - every extracted relationship should be useful

### Not Measured By
- ❌ Raw relationship count (900+ was arbitrary)
- ❌ Matching V8's exact count (1,090 included quality issues)
- ❌ Extraction speed or efficiency

---

## 🔄 Testing Strategy

### Unit Tests
1. Test dedication parsing with 5 real examples
2. Test possessive pronoun resolution with 5 cases
3. Test bibliographic extraction with 10 citations

### Integration Tests
1. Run V10 on Soil Stewardship Handbook (full book)
2. Compare V10 vs V9:
   - Bibliographic relationships: should increase significantly
   - Quality issues: should decrease to <3%
   - Attribution/classification: maintain 100%

### Success Validation
1. **Reflector Analysis**: Run on V10 output, should show:
   - <3% issue rate (A++ grade for general KG)
   - No major categories of missing relationships
   - Praise for comprehensiveness

2. **Manual Review**: Sample 50 relationships:
   - All should be meaningful
   - All should have proper attribution
   - All should have accurate classification

---

## 📈 Expected V10 Results

| Metric | V8 | V9 | V10 Target | Rationale |
|--------|----|----|------------|-----------|
| **Total Relationships** | 1,090 | 414 | 650-750 | Natural outcome of comprehensive extraction |
| **High Confidence** | 83.1% | 72.7% | 80-85% | Better quality control |
| **True Issues** | 8.35% | 5.8% | <3% | Fix all major issue categories |
| **Bibliographic** | 304 | 127 | 250+ | Comprehensive citation extraction |
| **Categorical** | 93 | 27 | 70+ | Include all definitions |
| **Attribution** | 0% | 100% | 100% | Maintain |
| **Classification** | 0% | 100% | 100% | Maintain |

---

## 🎓 Key Lessons Learned

1. **Quantity ≠ Quality**: V8's 1,090 relationships had 91 issues (8.35%)
2. **Numbers are arbitrary**: Focus on extracting valuable relationships, not hitting counts
3. **Comprehensiveness requires examples**: Saying "be comprehensive" isn't enough - show what to extract
4. **Citation matter**: 177 missing bibliographic relationships is a real gap
5. **Preserve innovations**: V9's attribution + classification system is excellent

---

## 🚀 Next Steps

1. **Review this plan** with stakeholders
2. **Implement V10 changes** (estimated 2-3 hours)
3. **Run V10 extraction** on Soil Stewardship Handbook
4. **Run Reflector** to validate improvements
5. **If A++ achieved**: Deploy to full corpus (172 episodes + 3 books)

---

**Prepared By**: Claude Code
**Date**: October 13, 2025
**Status**: Ready for Implementation
