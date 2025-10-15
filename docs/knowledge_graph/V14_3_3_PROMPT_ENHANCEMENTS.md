# V14.3.3 Prompt Enhancements - Phase 2 Upstream Prevention

## Overview

**Version**: V14.3.3 (Planned)
**Phase**: Phase 2 - Prompt Enhancements
**Goal**: Prevent issues upstream in Pass 1 extraction before they reach postprocessing
**Target**: Fix 8 HIGH + 12 MEDIUM issues from V14.3.2 analysis

## Rationale

**Why Prompt Improvements?**
- Upstream prevention > downstream fixing
- Issues prevented in Pass 1 don't need postprocessing cleanup
- More maintainable than adding complex postprocessing modules
- Gives LLM better guidance for edge cases

**V14.3.2 Issues Addressable by Prompts**:
- 8 HIGH: Unresolved pronouns ("our immune systems" → "human immune systems")
- 12 MEDIUM: Vague abstract entities ("cognitive performance" → "human cognitive performance")
- 2 CRITICAL: Document structure awareness (foreword vs authorship) - reinforcement

## Prompt Enhancement Strategy

### Enhancement 1: Pronoun Resolution Instructions
**Target Issues**: 8 HIGH (unresolved pronouns)
**File**: `kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt`

**Add to Constraints Section**:
```
5. PRONOUN RESOLUTION:
   - NEVER extract pronouns as entities
   - ALWAYS resolve pronouns BEFORE extraction

   Possessive pronouns ("our", "my", "their"):
   - Add "human" qualifier for human-related terms
   - Examples:
     * "our immune systems" → "human immune systems"
     * "our stress levels" → "human stress levels"
     * "our intelligence" → "human intelligence"
     * "their practices" → "human practices" (if referring to people)

   Personal pronouns ("we", "us", "they"):
   - Use "humans" or "humanity" for general references
   - Use specific group if mentioned in context
   - Examples:
     * "We enhance our health" → "humans enhance human health"
     * "They developed techniques" → "[specific group] developed techniques"

   Demonstrative pronouns ("this", "that", "these", "those"):
   - Replace with the antecedent noun from previous sentence
   - Examples:
     * "This process creates soil" → "composting process creates soil"
     * "These methods improve yields" → "organic farming methods improve yields"

   IMPORTANT: If you cannot resolve a pronoun with certainty, skip that relationship.
```

**Expected Impact**:
- Fixes: "soil stewardship → enhances → intelligence" → "soil stewardship → enhances → human intelligence"
- Fixes: "soil → boosts → immune systems" → "soil → boosts → human immune systems"
- Fixes: "soil → enhances → serotonin levels" → "soil → enhances → human serotonin levels"
- Fixes: "soil → reduces → stress levels" → "soil → reduces → human stress levels"
- Reduces HIGH issues from 8 → 0-2 (some edge cases may remain)

---

### Enhancement 2: Entity Specificity Requirements
**Target Issues**: 12 MEDIUM (vague abstract entities)
**File**: `kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt`

**Add to Extraction Guidelines**:
```
6. ENTITY SPECIFICITY:
   - Extract CONCRETE, SPECIFIC entities only
   - Avoid vague/abstract terms without qualifiers

   Common vague patterns to avoid:

   Performance/Levels (add WHO/WHAT):
   - ❌ "cognitive performance" → ✅ "human cognitive performance"
   - ❌ "stress levels" → ✅ "human stress levels"
   - ❌ "serotonin levels" → ✅ "human serotonin levels"
   - ❌ "energy levels" → ✅ "human energy levels"

   Impact/Effects (add ON WHAT):
   - ❌ "community impact" → ✅ "community health impact" or "local community well-being"
   - ❌ "environmental impact" → ✅ "ecosystem health impact"
   - ❌ "positive effects" → ✅ "positive effects on human health"

   Processes (add SPECIFIC TYPE):
   - ❌ "the process" → ✅ "composting process" or "soil-building process"
   - ❌ "the technique" → ✅ "no-till farming technique"
   - ❌ "the method" → ✅ "biodynamic farming method"

   Generic descriptors (add CONTEXT):
   - ❌ "quality" → ✅ "soil quality" or "crop quality"
   - ❌ "diversity" → ✅ "microbial diversity" or "crop diversity"
   - ❌ "resilience" → ✅ "ecosystem resilience" or "farm resilience"

   Rule of thumb:
   - If an entity contains "performance", "impact", "levels", "quality", "diversity",
     "resilience", "the process", "the method", etc. → ADD QUALIFIERS to make it specific
   - If you're unsure whether an entity is specific enough, ADD CONTEXT
```

**Expected Impact**:
- Reduces MEDIUM vague entity issues from 12 → 2-4
- Makes entities more useful for knowledge graph queries
- Improves semantic clarity

---

### Enhancement 3: Document Structure Awareness
**Target Issues**: 2 CRITICAL (foreword misattribution) - Reinforcement
**File**: `kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt`

**Add to Extraction Guidelines**:
```
7. DOCUMENT STRUCTURE AWARENESS:
   - Identify the section type before extracting relationships

   Section Types:

   Front Matter (pages 1-15):
   - Foreword, dedication, praise quotes, endorsements
   - Contributors who are NOT the author
   - Handle specially:
     * Foreword signatures: "wrote foreword for", NOT "authored"
     * Endorsement quotes: "endorsed", NOT "authored"
     * Dedication: "dedicated to", NOT other relationships

   Main Content (after front matter):
   - Primary information source
   - Extract domain knowledge relationships
   - This is where "authored" relationships are appropriate

   Back Matter (last 10-20 pages):
   - Appendices, resources, bibliography, acknowledgments
   - Extract relevant domain knowledge, but be cautious of references

   Authorship Rules:
   - ONLY the actual book author (from title page/copyright) gets "authored" relationship
   - Foreword writers: "wrote foreword for"
   - Introduction writers (not author): "wrote introduction for"
   - Endorsers: "endorsed"

   Examples:

   ✅ CORRECT:
   - "With Love, Jane Doe" at end of foreword (page 10)
     → (Jane Doe, wrote foreword for, Book Title)

   - "Author Name, 2018" in book introduction (page 15)
     → (Author Name, authored, Book Title)

   - "This book is dedicated to my parents"
     → (Author Name, dedicated to, parents)

   ❌ INCORRECT:
   - "With Love, Jane Doe" at end of foreword
     → (Jane Doe, authored, Book Title)  # WRONG - she wrote foreword, not book

   - Endorsement quote by Expert
     → (Expert, authored, Book Title)  # WRONG - should be "endorsed"
```

**Expected Impact**:
- Reinforces postprocessing fixes (PraiseQuoteDetector, FrontMatterDetector)
- Prevents issues at source rather than fixing downstream
- Reduces CRITICAL issues by preventing extraction errors

---

## Implementation Plan

### Step 1: Create v14_3_3 Prompts

**Files to Create**:
1. `kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt`
   - Based on v14_3_2 (or v14.3.1) Pass 1 prompt
   - Add Enhancement 1 (Pronoun Resolution)
   - Add Enhancement 2 (Entity Specificity)
   - Add Enhancement 3 (Document Structure Awareness)

2. `kg_extraction_playbook/prompts/pass2_evaluation_v14_3_3.txt`
   - Keep same as v14_3_2 (or v14.3.1) Pass 2 prompt
   - No changes needed (evaluation logic is solid)

### Step 2: Update Extraction Script

**File**: `scripts/extract_kg_v14_3_3_book.py`
- Copy from `extract_kg_v14_3_2_book.py`
- Update to load v14_3_3 prompts
- Keep temperature=0.0 (deterministic)
- Keep v14_3_2 pipeline (14 modules with Phase 1 fixes)

### Step 3: Test on Sample Chapter

**Approach**: Incremental testing
- Extract introduction/foreword only (pages 1-15)
- Run Reflector analysis
- Verify pronoun resolution and entity specificity improvements
- Iterate if needed before full book extraction

### Step 4: Full Book Extraction

**Once sample achieves A or A+ grade**:
- Run full book extraction with v14_3_3 prompts
- Run Reflector analysis
- Compare with V14.3.2 and V14.3.1
- Verify grade improvement (B+ → A or A+)

---

## Expected Results

### Issue Reduction

| Issue Category | V14.3.2 | V14.3.3 (Target) | Improvement |
|----------------|---------|------------------|-------------|
| CRITICAL | 2 | 0 | -2 ✅ (Phase 1) |
| HIGH | 8 | 0-2 | -6 to -8 ✅ (Phase 2) |
| MEDIUM (pronouns) | 0 | 0 | 0 (fixed by pronouns) |
| MEDIUM (vague entities) | 12 | 2-4 | -8 to -10 ✅ |
| MEDIUM (list splitting) | 15 | 15 | 0 (Phase 3) |
| MILD | 28 | 28 | 0 (user accepts) |
| **Total** | **53 (11.1%)** | **17-21 (3.6-4.4%)** | **-32 to -36** ✅ |

### Grade Progression

| Version | Issues | Issue Rate | Grade |
|---------|--------|-----------|-------|
| V14.3.2 | 53 | 11.1% | B+ |
| V14.3.2.1 (Phase 1) | 51 | 10.7% | B+ → A- |
| V14.3.3 (Phase 2) | 17-21 | 3.6-4.4% | **A** ✅ |
| V14.3.4 (Phase 3) | 8-12 | 1.7-2.5% | **A+** 🎯 |

---

## Phase 3 Preview: Postprocessing Improvements

**After Phase 2 completes**, if additional cleanup is needed:

### Fix 1: Enhanced PronounResolver
- Catch remaining possessive pronouns that Pass 1 missed
- Fallback rules for edge cases
- **Target**: Remaining HIGH pronoun issues

### Fix 2: Context-Aware ListSplitter
- Don't split on "and" within quotes
- Don't split on "and" after colons (titles)
- Don't split if result starts with "and"
- **Target**: 15 MEDIUM list splitting issues

**Expected Final Result**: A+ grade (1.7-2.5% issue rate, 8-12 total issues)

---

## Prompt Engineering Principles Applied

1. **Specificity**: Concrete examples for each pattern
2. **Structure**: Clear sections with ✅/❌ indicators
3. **Context**: Explain WHY (not just WHAT)
4. **Examples**: Multiple examples for each rule
5. **Clarity**: Simple, actionable instructions
6. **Reinforcement**: Multi-level guidance (rules + examples + edge cases)

---

## Testing Strategy

### Validation Approach
1. **Unit tests**: Test individual patterns with mock data
2. **Sample extraction**: Test on book introduction/foreword only
3. **Reflector analysis**: Verify issue reduction
4. **Iterative refinement**: Adjust prompts based on results
5. **Full extraction**: Once sample achieves target grade

### Success Criteria
- HIGH issues: 0-2 (down from 8)
- MEDIUM vague entity issues: 2-4 (down from 12)
- Total issues: 17-21 (down from 53)
- Grade: A (3.6-4.4% issue rate)

---

## Notes

- **Phase 2 is purely prompt-based**: No code changes needed
- **Prompt enhancements are cumulative**: Each builds on previous
- **Testing is critical**: Validate improvements before full extraction
- **Iteration expected**: May need 1-2 refinement cycles to achieve target
- **Provenance tracking**: Save all prompt versions and extraction results

---

## Next Actions

1. ✅ Document prompt enhancements (this file)
2. Create v14_3_3 prompts from v14_3_2 base
3. Update extraction script to use v14_3_3 prompts
4. Test on sample chapter (introduction/foreword)
5. Analyze results and iterate if needed
6. Run full extraction once sample achieves A grade
