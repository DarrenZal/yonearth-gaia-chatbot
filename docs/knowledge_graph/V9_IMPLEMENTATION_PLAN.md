# V9 Implementation Plan

## Overview

Based on the V8 Reflector analysis, V9 will implement 5 CRITICAL and HIGH priority fixes to reduce issue rate from 8.35% to <5% (targeting 3-4%).

## V8 Quality Issues Summary

**Current State**: 91 issues (8.35%) across 1090 relationships
- CRITICAL: 3 issues (0.28%)
- HIGH: 18 issues (1.65%)
- MEDIUM: 42 issues (3.85%)
- MILD: 28 issues (2.57%)

**Regression from V7**: +29 issues (+1.64% increase)
- Root causes: Incomplete curator fixes, over-extraction, no quality threshold

---

## CRITICAL Priority Fixes

### 1. Fix Praise Quote Attribution Logic ⭐⭐⭐

**Problem**: PraiseQuoteDetector changes relationship type from "authored" to "endorsed" but doesn't identify the correct SPEAKER as the source.

**Examples**:
- **BAD**: `(Perry, endorsed, Soil Stewardship Handbook)`
  - Perry is the AUTHOR being praised, not the endorser
- **GOOD**: `(Adrian Del Caro, endorsed, Soil Stewardship Handbook)`
  - Adrian Del Caro is the SPEAKER in the attribution marker

**Current Implementation** (line 241-309 in extract_kg_v8_book.py):
```python
# Only changes relationship type, doesn't fix attribution
if any(verb in relationship_type for verb in self.authorship_verbs):
    if self.is_praise_quote_context(evidence, page):
        rel.relationship = 'endorsed'  # ✅ Fixed type
        # ❌ But source is still wrong!
```

**V9 Fix**:
1. Parse attribution marker pattern: `—Name, Title`
2. Extract SPEAKER name from attribution
3. Check if current source is the AUTHOR being praised (appears in evidence with "has given us", "provides", etc.)
4. If source is author, swap to make SPEAKER the source
5. Handle credential bylines (e.g., "—Adrian Del Caro, Author of X") to identify AUTHORED works separately

**Implementation**:
```python
def fix_praise_quote_attribution(self, rel, evidence_text):
    """
    Extract the SPEAKER from attribution marker and fix source.

    Patterns:
    - "X has given us" → X is the author being praised
    - "—Speaker Name, Title" → Speaker is the endorser
    """
    # Extract speaker from attribution
    attribution_match = re.search(r'—([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', evidence_text)
    if not attribution_match:
        return rel

    speaker_name = attribution_match.group(1)

    # Check if current source is the subject being praised
    subject_patterns = [
        r'(\w+)\s+has given us',
        r'(\w+)\s+provides',
        r'(\w+)\'s?\s+(?:handbook|book|work)',
        r'with (?:his|her)\s+.*?,\s+(\w+)\s+'
    ]

    for pattern in subject_patterns:
        match = re.search(pattern, evidence_text, re.IGNORECASE)
        if match and match.group(1).lower() in rel.source.lower():
            # Current source is the subject, swap to speaker
            rel.source = speaker_name
            rel.flags['PRAISE_ATTRIBUTION_FIXED'] = True
            break

    # Handle credential bylines (Author of X)
    if 'Author of' in evidence_text and speaker_name in evidence_text:
        # This is a credentials line, extract authored work
        work_match = re.search(r'Author of (.+?)(?:\n|$|University)', evidence_text)
        if work_match:
            # Create separate authorship relationship
            authored_work = work_match.group(1).strip()
            # Note: This requires creating a new relationship

    return rel
```

**Expected Impact**: Fixes 3 CRITICAL errors (0.28%)

---

### 2. Add Quality Threshold Filtering ⭐⭐⭐

**Problem**: Relationships with `p_true < 0.5` (low confidence) are still being extracted. These are typically philosophical/abstract statements that Pass 2 correctly flagged as questionable.

**Examples**:
- `p_true=0.40`: "spiritual flourishing depends on earth care" (philosophical)
- `p_true=0.35`: "being connected to soil is what it means to be human" (subjective definition)

**V9 Fix**: Add filtering after Pass 2 evaluation before Pass 2.5 postprocessing.

**Implementation** (in `build_relationship_from_eval` function):
```python
# After Pass 2 evaluation
if p_true < 0.5:  # ⭐ V9 NEW: Quality threshold
    logger.debug(f"   Filtered low-confidence relationship (p_true={p_true:.2f}): {eval_rel.source} → {eval_rel.target}")
    stats['low_confidence_filtered'] += 1
    return None  # Skip this relationship

# Otherwise continue to Pass 2.5
```

**Configuration**:
```python
# V9 Constants
MIN_P_TRUE_THRESHOLD = 0.5  # Filter relationships below this confidence
```

**Expected Impact**: Removes ~8 philosophical/abstract statements (0.73%)

---

## HIGH Priority Fixes

### 3. Enhanced Possessive Pronoun Resolution ⭐⭐

**Problem**: Pronoun resolver handles simple pronouns (he/she/it) but not possessive constructions (my X, our X, their X).

**Examples**:
- `"my people" → "Slovenians"` (based on context: "my people...in Slovenia")
- `"Our land" → "Slovenia"` (based on context)
- `"our connection" → "Slovenian connection"` (partial resolution)

**V9 Fix**: Extend pronoun resolver to detect and resolve possessive patterns.

**Implementation** (enhance `PronounResolver` class):
```python
class PronounResolver:
    def __init__(self):
        # Existing patterns
        self.simple_pronouns = ['he', 'she', 'it', 'they', 'we', 'i']

        # ⭐ V9 NEW: Possessive patterns
        self.possessive_patterns = [
            (r'\bmy\s+(\w+)', 'author_possessive'),     # my people, my land
            (r'\bour\s+(\w+)', 'collective_possessive'), # our connection, our land
            (r'\btheir\s+(\w+)', 'third_person_possessive')  # their practices
        ]

    def resolve_possessive_pronoun(self, text, entity, context_window):
        """
        Resolve possessive pronouns using context.

        Examples:
        - "my people...in Slovenia" → "Slovenians"
        - "Our land of glorious mountains...Slovenia" → "Slovenia"
        """
        for pattern, poss_type in self.possessive_patterns:
            match = re.search(pattern, entity, re.IGNORECASE)
            if not match:
                continue

            possessed_noun = match.group(1)  # e.g., "people", "land"

            # Strategy 1: Look for proper nouns in context
            if poss_type == 'author_possessive':
                # "my X" likely refers to author's background
                # Look for country/place names in context
                place_match = re.search(r'\b([A-Z][a-z]+(?:ia|land|stan))\b', context_window)
                if place_match and possessed_noun in ['people', 'ancestors']:
                    return place_match.group(1) + 'ans'  # Slovenia → Slovenians
                if place_match and possessed_noun == 'land':
                    return place_match.group(1)  # Slovenia

            elif poss_type == 'collective_possessive':
                # "our X" likely refers to collective entity mentioned earlier
                # Similar to "my" but look for plural/collective nouns
                place_match = re.search(r'\b([A-Z][a-z]+(?:ia|land|stan))\b', context_window)
                if place_match:
                    if possessed_noun in ['land', 'country', 'region']:
                        return place_match.group(1)
                    elif possessed_noun in ['people', 'tradition', 'connection']:
                        return f"{place_match.group(1)} {possessed_noun}"

        return entity  # No resolution found
```

**Expected Impact**: Fixes 12 HIGH possessive pronoun errors (1.10%)

---

### 4. Enhanced Pass 1 Extraction Prompt ⭐⭐

**Problem**: Pass 1 prompt is too permissive ("Extract ALL relationships") leading to over-extraction of pronouns, abstractions, and philosophical statements.

**Current Wording**:
> "Extract ALL relationships you can find in this text. Don't worry about whether they're correct or make sense - just extract EVERYTHING. We'll validate later in a separate pass."

**V9 Fix**: Add explicit constraints and few-shot examples.

**New Prompt Enhancements**:
```text
## ⚠️ CRITICAL EXTRACTION RULES ⚠️

**EXTRACT**:
✅ Specific named entities (people, places, organizations, books)
✅ Concrete, verifiable relationships
✅ Factual claims (not opinions or philosophical statements)

**AVOID**:
❌ Pronouns as sources or targets (I, we, he, she, they, my X, our X)
❌ Overly abstract concepts ("spiritual flourishing", "the crossroads")
❌ Metaphors and figurative language
❌ Philosophical definitions ("what it means to be human")
❌ Overly verbose entities (full sentences as entities)

## ENTITY CONCISENESS

Entities should be **CONCISE NOUN PHRASES** (2-5 words typically):
- ✅ GOOD: "soil ecosystem complexity"
- ❌ BAD: "hardly anything is as complex as the living web of interconnectedness found in our planet's soil"
- ✅ GOOD: "new generation of farmers"
- ❌ BAD: "new generation of farmers, gardeners and citizens are leading the way"

## FEW-SHOT EXAMPLES

### Example 1: Avoid Pronouns
**Text**: "I hail from Slovenia. My people have lived there for centuries."
❌ BAD: (I, hails from, Slovenia)
❌ BAD: (my people, lived in, Slovenia)
✅ GOOD: (Aaron William Perry, hails from, Slovenia)  # Resolve "I" to author name
✅ GOOD: (Slovenians, lived in, Slovenia)  # Resolve "my people" to specific group

### Example 2: Avoid Philosophical Statements
**Text**: "Being connected to land and soil is what it means to be human."
❌ BAD: (being connected to land and soil, is what it means to be, human)
✅ SKIP: This is a philosophical opinion, not a factual relationship

### Example 3: Concrete vs. Abstract
**Text**: "Spiritual flourishing is wedded to how we take care of the earth."
❌ BAD: (spiritual flourishing, depends on, how we take care of the earth)
✅ BETTER (if extracting): (human wellbeing, depends on, environmental stewardship)  # Use concrete terms

### Example 4: Entity Conciseness
**Text**: "Learning about soil and engaging in soil-building practices is an exciting way to positively influence families."
❌ BAD: (Learning about the power of soil, is-a, exciting way to positively influence your families)
✅ GOOD: (soil education, benefits, families)  # Concise entities, clear relationship
```

**Expected Impact**: Reduces pronoun sources (6 HIGH), vague entities (24 MEDIUM), philosophical statements (18 MEDIUM) at extraction time. Estimated 40-50% reduction = ~24 issues prevented.

---

### 5. Enhanced Pass 2 Evaluation Prompt ⭐⭐

**Problem**: Knowledge plausibility signal is miscalibrated - philosophical statements get moderate scores (0.5-0.7) when they should get low scores (0.0-0.3).

**Current Wording**:
> "KNOWLEDGE SIGNAL (ignore the text): Is this relationship plausible given world knowledge? Score 0.0-1.0 based purely on plausibility"

**V9 Fix**: Add explicit scoring guidelines and examples.

**New Evaluation Criteria**:
```text
## KNOWLEDGE PLAUSIBILITY SIGNAL

Score based on **VERIFIABILITY and CONCRETENESS**:

### 0.0-0.3: Unverifiable / Philosophical
- Metaphors ("our land is a veritable Eden")
- Philosophical claims ("spiritual flourishing depends on earth care")
- Subjective definitions ("being connected to soil is what it means to be human")
- Abstract opinions not grounded in facts

**Examples**:
- (spiritual flourishing, depends on, earth care) → 0.2 (philosophical, not verifiable)
- (being connected to soil, is what it means to be, human) → 0.1 (subjective definition)
- (Our land, is, a veritable Eden) → 0.3 (metaphor, not literal fact)

### 0.4-0.6: Debatable / Abstract
- Causal claims with some factual basis but debatable
- Abstract concepts with empirical support
- General statements that are plausible but not easily verified

**Examples**:
- (soil health, affects, community wellbeing) → 0.5 (causal claim with some evidence)
- (composting, improves, soil quality) → 0.6 (generally accepted, some evidence)

### 0.7-0.9: Concrete Facts / Minor Uncertainty
- Geographic facts
- Historical events
- Scientific consensus
- Well-documented relationships

**Examples**:
- (Slovenia, is located in, eastern Alpine region) → 0.8 (geographic fact)
- (biochar, sequesters, carbon) → 0.75 (scientific consensus)

### 0.9-1.0: Easily Verifiable Facts
- Authorship (who wrote what)
- Organizational roles (who founded/works for what)
- Basic factual claims

**Examples**:
- (Aaron William Perry, authored, Soil Stewardship Handbook) → 1.0 (verifiable authorship)
- (Y on Earth, is, nonprofit organization) → 0.95 (easily verified)

## CONFLICT DETECTION

Set `signals_conflict=true` when:

1. **Text confidence high (>0.7) but knowledge plausibility low (<0.4)**
   - Example: Text clearly states a metaphor, but it's not literally true
   - Conflict: "The text clearly says it, but it's a metaphor/opinion"

2. **Text confidence low (<0.4) but knowledge plausibility high (>0.7)**
   - Example: Text is vague but the relationship is a well-known fact
   - Conflict: "Text doesn't clearly state it, but we know it's true"

3. **Philosophical/abstract despite clear text**
   - Example: "being connected to soil is what it means to be human"
   - Conflict: "Text is clear, but this is a philosophical claim, not a fact"

**Provide `conflict_explanation`** describing WHY the signals conflict.
```

**Expected Impact**: Improves Pass 2 filtering of philosophical statements. Combined with p_true threshold (Fix #2), should remove most of the 18 MEDIUM philosophical issues.

---

## V9 Implementation Summary

### Files to Modify:
1. `scripts/extract_kg_v9_book.py` (copy from V8, apply fixes)
2. `kg_extraction_playbook/prompts/pass1_extraction_v9.txt` (new version)
3. `kg_extraction_playbook/prompts/pass2_evaluation_v9.txt` (new version)

### Expected Quality Improvement:
- **V8 Baseline**: 91 issues (8.35%)
- **Fix #1 (Praise quotes)**: -3 issues
- **Fix #2 (p_true threshold)**: -8 issues
- **Fix #3 (Possessive pronouns)**: -12 issues
- **Fix #4 (Pass 1 prompt)**: -24 issues (estimated 50% reduction of 48)
- **Fix #5 (Pass 2 prompt)**: -9 issues (estimated 50% of remaining 18 philosophical)

**V9 Target**: ~35-40 issues (3.2-3.7%) ✅ **Below 5% production threshold**

### Testing Strategy:
1. Run V9 extraction on Soil Stewardship Handbook
2. Compare V9 vs V8 output (relationships, flags, statistics)
3. Run Reflector on V9 results
4. Validate <5% issue rate achieved

---

## Implementation Steps

1. ✅ Create V9 implementation plan (this document)
2. ⏳ Copy V8 script to V9 with version updates
3. ⏳ Implement Fix #1: Praise quote attribution logic
4. ⏳ Implement Fix #2: p_true threshold filtering
5. ⏳ Implement Fix #3: Possessive pronoun resolution
6. ⏳ Create Fix #4: Enhanced Pass 1 prompt v9
7. ⏳ Create Fix #5: Enhanced Pass 2 prompt v9
8. ⏳ Test V9 extraction
9. ⏳ Run Reflector on V9 results
10. ⏳ Document V9 results
