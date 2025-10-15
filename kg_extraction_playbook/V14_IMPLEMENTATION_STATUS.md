# V14.0 Implementation Status

## 📊 Overview

**Target**: Implement V14.0 improvements to reach A grade (6.0% issue rate)
**Source**: V13.1 (A- grade, 8.6% issue rate, 75 issues)
**Changeset**: `/kg_extraction_playbook/changesets/v13_1_to_v14_IMPROVED_20251014_102000.json`

## ✅ Completed (1/7)

### Change 001: Metadata Filter Module ✅
- **File**: `src/knowledge_graph/postprocessing/content_specific/books/metadata_filter.py`
- **Status**: ✅ COMPLETE
- **What it does**: Filters book metadata (praise quotes, dedications, publication info)
- **Multi-layer detection**:
  - Layer 1: Flag-based (PRAISE_QUOTE_CORRECTED, DEDICATION_DETECTED)
  - Layer 2: Predicate-based (endorsed, dedicated, published by/in)
  - Layer 3: Page-based (front/back matter detection)
  - Layer 4: Combined heuristic (book title + person name on early page)
- **Priority**: 11 (runs after PraiseQuoteDetector)
- **Dependencies**: PraiseQuoteDetector
- **Expected Impact**: Eliminates 10 metadata relationships (1.1%)

## 🔨 In Progress (0/7)

None currently

## ⏳ Remaining (6/7)

### Change 003+004: Update Pass 1 Prompt (HIGH Priority)
- **Files**:
  - `kg_extraction_playbook/prompts/pass1_extraction_v14.txt` (NEW)
  - Copy from V13: `kg_extraction_playbook/prompts/pass1_extraction_v13.txt`
- **Changes Needed**:
  1. **Entity Specificity Requirements** (Change 003):
     - Add section after "ENTITY EXTRACTION RULES"
     - Heading: "🎯 ENTITY SPECIFICITY REQUIREMENTS"
     - Prohibit: 'aspects of life', 'the answer', 'this approach', 'the way'
     - Rules: Scan 2-3 sentences for referent, skip if vague, prefer concrete nouns
     - Test: Can you define entity in 1-2 sentences? If no → too vague

  2. **Extraction Scope** (Change 004):
     - Add section at beginning after introduction
     - Heading: "📚 EXTRACTION SCOPE: DOMAIN KNOWLEDGE ONLY"
     - Principle: Extract SUBJECT MATTER, NOT book metadata
     - Skip: Praise quotes, dedications, publication info, author bio, front/back matter
     - Extract: Domain facts, scientific claims, practical knowledge, citations
     - Decision rule: "Is this about SOIL STEWARDSHIP or THE BOOK?"

- **Real V13.1 Failures to Include**:
  - ❌ ('soil stewardship', 'affects', 'aspects of life')
  - ❌ ('regenerative agriculture', 'is', 'the answer')
  - ❌ ('Michael Bowman', 'endorsed', 'Soil Stewardship Handbook')

- **Expected Impact**: Prevents 18-20 issues at source (vague entities + metadata)

### Change 002: Update Pass 2 Prompt (HIGH Priority)
- **File**:
  - `kg_extraction_playbook/prompts/pass2_evaluation_v14.txt` (NEW)
  - Copy from V13.1: `kg_extraction_playbook/prompts/pass2_evaluation_v13_1.txt`
- **Changes Needed**:
  - Insert after "EVALUATION CRITERIA", before "OUTPUT FORMAT"
  - Heading: "⚠️ CLAIM TYPE CLASSIFICATION"
  - Three types: FACTUAL, NORMATIVE, PHILOSOPHICAL
  - Definitions with p_true ranges:
    - FACTUAL: Testable/measurable, p_true = 0.7-0.95
    - NORMATIVE: Prescriptive (should/ought/can), p_true = 0.3-0.5
    - PHILOSOPHICAL: Abstract/spiritual, p_true = 0.1-0.3
  - Scoring rules for each type
  - **Real V13.1 failures** as examples:
    - ❌ ('soil', 'is-a', 'cosmically sacred') → p_true=0.8 WRONG, should be p_true=0.2 PHILOSOPHICAL
    - ❌ ('humanity', 'should connect with', 'soil') → p_true=0.75 WRONG, should be p_true=0.4 NORMATIVE
    - ✅ ('soil management', 'can mitigate', 'climate change') → p_true=0.75 FACTUAL (testable)
- **Expected Impact**: Fixes 10-12 philosophical/normative issues (1.1-1.4%)

### Change 006: Update Filtering Thresholds (MEDIUM Priority)
- **File**:
  - `config/filtering_thresholds.yaml` (CHECK IF EXISTS, may need to create)
  - OR modify filtering logic in extraction script
- **Changes Needed**:
  - **CONSERVATIVE approach**: base_threshold = 0.5 (NOT 0.7!)
  - Flag-specific thresholds:
    - PHILOSOPHICAL_CLAIM: 0.85
    - METAPHOR / FIGURATIVE_LANGUAGE: 0.85
    - OPINION: 0.9
    - signals_conflict=true: 0.75
- **Rationale**: V13.1 calibrated for 0.5. Raising to 0.7 would over-filter valid knowledge.
- **Expected Impact**: Filters 8-10 low-quality philosophical/metaphorical relationships

### Change 005: Enhance Predicate Normalizer (MEDIUM Priority)
- **File**: `src/knowledge_graph/postprocessing/universal/predicate_normalizer.py`
- **Changes Needed**:
  1. **Tense Normalization**:
     - 'has preserved' → 'preserved'
     - 'has enabled' → 'enabled'
     - 'is produced' → 'produces'

  2. **'is-X' Variant Consolidation**:
     - 'is about' → 'relates to' or 'concerns'
     - 'is of' → 'is-a' if type relationship
     - 'is characterized by' → extract core verb

  3. **CRITICAL: DO NOT strip modal verbs**:
     - ✅ KEEP: 'can', 'may', 'might', 'could'
     - ❌ DO NOT normalize 'can help' → 'helps' (loses epistemic info)
     - Rationale: Modal verbs indicate uncertainty, critical for scientific claims

  4. **Semantic Validation**:
     - Add validate_semantic_compatibility() method
     - Check abstract source + physical predicate → FLAG
     - Check 'is-a' with incompatible types → REJECT

- **Expected Impact**: Reduces 133 unique predicates → 90-100, fixes 5-8 semantic mismatches

### Change 007: Fix Pronoun Resolver (MEDIUM Priority)
- **File**: `src/knowledge_graph/postprocessing/universal/pronoun_resolver.py`
- **Changes Needed**:
  1. Ensure resolve_pronouns() called for BOTH source AND target
  2. Apply same context window (3 sentences) to targets as sources
  3. For possessives ('our countryside'):
     - Extract entity from context
     - Example: 'our countryside' + 'Slovenia' → 'Slovenian countryside'

  4. **Fallback for unresolvable**:
     - If resolution fails after checking 3 sentences → add PRONOUN_UNRESOLVED flag
     - Filter relationship if p_true < 0.7 AND unresolved pronoun present

  5. Resolution strategy:
     - Possessive: Extract from context
     - Generic ('we', 'us'): → 'humanity', 'individuals', or specific group
     - Demonstrative ('this', 'that'): → nearest concrete noun in previous sentence

- **Expected Impact**: Resolves 5-7 remaining pronoun issues

### Integration: Add MetadataFilter to Book Pipeline
- **File**: `src/knowledge_graph/postprocessing/pipelines/book_pipeline.py`
- **Change Needed**:
  - Import: `from ..content_specific.books.metadata_filter import MetadataFilter`
  - Add to pipeline AFTER bibliographic_citation_parser
  - Position: After PraiseQuoteDetector, before other universal modules
- **Status**: ⏳ Pending (depends on Change 001 ✅)

## 📈 Expected Impact Summary

| Metric | V13.1 (Current) | V14.0 (Target) | Improvement |
|--------|-----------------|----------------|-------------|
| **Grade** | A- | A | +1 grade level |
| **Issue Rate** | 8.6% | 6.0% | -2.6 percentage points |
| **Total Issues** | 75 | <55 | -20 issues (27% reduction) |
| **High Priority** | 8 | 0-2 | -6 to -8 issues |
| **Medium Priority** | 22 | 8-12 | -10 to -14 issues |
| **Mild Priority** | 45 | <20 | -25 issues |

## 🔧 Implementation Order (Recommended)

1. ✅ **DONE**: Create metadata_filter.py (Change 001)
2. ⏳ **NEXT**: Update Pass 1 prompt (Changes 003 + 004) - HIGH impact, low risk
3. ⏳ Update Pass 2 prompt (Change 002) - HIGH impact, low risk
4. ⏳ Add MetadataFilter to book_pipeline.py - Integration
5. ⏳ Update filtering thresholds (Change 006) - MEDIUM impact, low risk
6. ⏳ Enhance predicate_normalizer.py (Change 005) - MEDIUM impact, medium risk
7. ⏳ Fix pronoun_resolver.py (Change 007) - MEDIUM impact, low risk

## 📝 Testing Plan

After implementation:

1. **Run V14.0 extraction** on Soil Stewardship Handbook
2. **Compare to V13.1** using Reflector
3. **Success Criteria**:
   - Issue rate < 6.5%
   - High priority issues < 3
   - No new critical issues
   - Grade ≥ A
4. **Rollback Plan**: If worse than V13.1, keep only metadata_filter, revert other changes

## 📁 Key Files

### Changesets
- Original Curator: `kg_extraction_playbook/changesets/v13_1_to_v14_20251014_100326.json`
- **Improved V14.0**: `kg_extraction_playbook/changesets/v13_1_to_v14_IMPROVED_20251014_102000.json` ✅

### Analysis Reports
- V13.1 Reflector: `kg_extraction_playbook/analysis_reports/reflection_v13.1_20251014_095254.json`
- V14.3 test results: `kg_extraction_playbook/analysis_reports/v14_3_pipeline_test.json` (17.6% pass rate - REJECTED)
- V14.2 test results: `kg_extraction_playbook/analysis_reports/v14_2_pipeline_test.json` (23.5% pass rate - REJECTED)

### Prompts
- V13.1 Pass 1: `kg_extraction_playbook/prompts/pass1_extraction_v13.txt` (base for V14)
- V13.1 Pass 2: `kg_extraction_playbook/prompts/pass2_evaluation_v13_1.txt` (base for V14)
- **V14 Pass 1**: `kg_extraction_playbook/prompts/pass1_extraction_v14.txt` (TO CREATE)
- **V14 Pass 2**: `kg_extraction_playbook/prompts/pass2_evaluation_v14.txt` (TO CREATE)

## 🎯 Critical Improvements Over Curator's V14.1

1. ✅ **Preserved epistemic hedging** (don't strip 'can', 'may', 'might')
2. ✅ **Conservative filtering** (0.5 base threshold, not 0.7)
3. ✅ **Lower risk** (7 changes vs 10)
4. ✅ **Real V13.1 examples** in prompts (not generic)
5. ✅ **Multi-layer metadata detection** (4 layers, not just flags)
6. ✅ **Unresolvable pronoun fallback** (filter if can't resolve)

## 📊 Risk Assessment

| Change | Risk Level | Rationale |
|--------|------------|-----------|
| 001: MetadataFilter | **LOW** ✅ | New module, won't break existing functionality |
| 003: Entity Specificity | **LOW** | Prompt enhancement, easy to revert |
| 004: Extraction Scope | **LOW** | Prompt enhancement, easy to revert |
| 002: Claim Classification | **LOW** | Prompt enhancement with clear guidelines |
| 006: Filtering Thresholds | **LOW** | Conservative approach, easy to tune |
| 005: Predicate Normalizer | **MEDIUM** | Code changes, but preserves modal verbs |
| 007: Pronoun Resolver | **LOW** | Extends existing logic, adds fallback |

**Overall Risk**: LOW (mostly prompt enhancements with 1 new module, 2 code improvements)

## 🚀 Next Steps

**Option 1: Continue Implementation Now**
- Continue with Changes 003+004 (Pass 1 prompt updates)
- Then 002 (Pass 2 prompt), 006 (config), 005 (code), 007 (code)
- Test and compare to V13.1

**Option 2: Review and Resume Later**
- Review this status document
- Continue implementation in fresh session
- All context preserved in this document + changeset JSON

---

*Last Updated*: 2025-10-14 10:25
*Status*: 1/7 changes complete, 6 remaining
*Next*: Update Pass 1 prompt (Changes 003+004)
