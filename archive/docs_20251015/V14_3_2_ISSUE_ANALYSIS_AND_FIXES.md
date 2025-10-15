# V14.3.2 Issue Analysis and Targeted Fixes

## Overview

**V14.3.2 Result**: B+ grade (11.1% issue rate, 53 total issues)
**Expected**: A+ grade (1-2% issue rate)
**Root Cause**: Temperature > 0 causing LLM variance + new issues emerged

## Temperature Fix ✅

**CRITICAL FIX APPLIED**:
```python
# BEFORE:
temperature=0.3  # Pass 1 ❌
temperature=0.2  # Pass 2 ❌

# AFTER:
temperature=0.0  # Pass 1 ✅ Deterministic
temperature=0.0  # Pass 2 ✅ Deterministic
```

**Impact**: This will eliminate LLM variance between runs, giving us reproducible results.

## Issue Breakdown (V14.3.2)

### CRITICAL Issues (2) - Must Fix

#### 1. Foreword Misattribution
**Issue**: "Lily Sophia von Übergarten → authored → Soil Stewardship Handbook"
- **What's wrong**: She wrote the foreword, not the book
- **Evidence**: "With Love and Hope, Lily Sophia von Übergarten Slovenia, 2018" (page 10)
- **Root cause**: Pass 1 doesn't distinguish foreword signatures from authorship

**Fix Strategy**:
- **Prompt Fix** (Pass 1): Add document structure awareness
- **Code Fix** (Pass 2.5): Create FrontMatterDetector module

#### 2. Author Signature Over-Corrected
**Issue**: "Aaron William Perry → endorsed → Soil Stewardship Handbook"
- **What's wrong**: Aaron IS the author, but PraiseQuoteDetector changed "authored" to "endorsed"
- **Evidence**: "Gratefully in service and celebration, Aaron William Perry..." (page 12)
- **Root cause**: PraiseQuoteDetector too aggressive, no author whitelist

**Fix Strategy**:
- **Code Fix** (Pass 2.5): Add author whitelist check to PraiseQuoteDetector

### HIGH Issues (8) - Upstream Prevention

#### Unresolved Pronouns
**Issues**:
- "soil stewardship → enhances → intelligence" (should be "human intelligence")
- "soil → boosts → immune systems" (should be "human immune systems")
- "soil → enhances → serotonin levels" (should be "human serotonin levels")
- "soil → reduces → stress levels" (should be "human stress levels")

**Root cause**: Possessive pronouns ("our") not resolved

**Fix Strategy**:
- **Prompt Fix** (Pass 1): Add explicit pronoun resolution instruction
- **Code Fix** (Pass 2.5): Enhance PronounResolver to catch possessive pronouns

### MEDIUM Issues (27) - Smarter Postprocessing

#### List Splitting Artifacts (15)
**Issues**:
- "chapter on Soil-Building Explained: Practical" + "Awesome!" (should be one entity)
- "agricultural soils → restored to → and productively vital states" (grammatically broken)

**Root cause**: ListSplitter splits on "and" within titles/compound nouns

**Fix Strategy**:
- **Code Fix** (Pass 2.5): Context-aware splitting (don't split within quotes/after colons)

#### Vague Abstract Entities (12)
**Issues**:
- "cognitive performance" (should be "human cognitive performance")
- "community impact" (too vague)

**Root cause**: Pass 1 allows abstract entities, VagueEntityBlocker not catching all

**Fix Strategy**:
- **Prompt Fix** (Pass 1): Emphasize concrete, specific entities
- **Code Fix** (Pass 2.5): Lower VagueEntityBlocker threshold or enhance patterns

### MILD Issues (28) - User Accepts

#### Metaphorical/Figurative Language
**Issues**:
- "soil → is-a → medicine" (metaphor)
- "soil microbiome → magically does → work" (figurative)

**Status**: **User accepts these** ("ok with stuff like 'soil embodies wisdom'")

**Fix Strategy**: Optional prompt enhancement, low priority

## Targeted Fixes - Implementation Plan

### Priority 1: CRITICAL Fixes (Immediate)

#### Fix 1.1: PraiseQuoteDetector Author Whitelist

**File**: `src/knowledge_graph/postprocessing/content_specific/books/praise_quote_detector.py`

**Enhancement**:
```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    super().__init__(config)

    # V14.3.2.1 NEW: Author whitelist to prevent over-correction
    self.known_authors = set()  # Populated from document metadata

def set_document_metadata(self, context: ProcessingContext):
    """V14.3.2.1: Extract known authors from metadata"""
    if context.document_metadata:
        author = context.document_metadata.get('author')
        if author:
            # Handle both "First Last" and "First Middle Last"
            self.known_authors.add(author.lower())
            # Add variants
            parts = author.split()
            if len(parts) >= 2:
                self.known_authors.add(f"{parts[0].lower()} {parts[-1].lower()}")

def is_praise_quote(self, rel: Any) -> Tuple[bool, str]:
    """Enhanced with author whitelist check"""

    # V14.3.2.1 FIX: Don't flag actual author as praise quote
    if rel.source.lower() in self.known_authors:
        return False, "source_is_known_author"

    # Original logic continues...
```

**Expected Impact**: Fixes 1 CRITICAL issue (Aaron Perry case)

#### Fix 1.2: FrontMatterDetector Module (NEW)

**File**: `src/knowledge_graph/postprocessing/content_specific/books/front_matter_detector.py`

**Purpose**: Detect and handle foreword/praise section signatures

**Logic**:
```python
class FrontMatterDetector(PostProcessingModule):
    """
    Detect relationships extracted from front matter (foreword, praise, dedication).

    Front matter signatures should be:
    - "wrote foreword for" instead of "authored"
    - "endorsed" instead of "authored" (for non-authors)
    """

    priority = 12  # After MetadataFilter (11), before BibliographicCitationParser (20)

    FRONT_MATTER_KEYWORDS = [
        'foreword', 'preface', 'introduction',
        'with love and hope', 'endorsement', 'praise',
        'what people are saying'
    ]

    def is_front_matter_signature(self, evidence: str, page: int) -> bool:
        """Detect if evidence is from front matter"""
        evidence_lower = evidence.lower()

        # Check for front matter keywords
        if any(keyword in evidence_lower for keyword in self.FRONT_MATTER_KEYWORDS):
            return True

        # Check page number (front matter usually < page 15)
        if page < 15:
            # Check for signature patterns
            if re.search(r'(with love|gratefully|sincerely|in service)', evidence_lower):
                return True

        return False

    def correct_front_matter_authorship(self, rel: Any) -> Any:
        """Convert authorship to 'wrote foreword for' if from front matter"""

        if rel.relationship != 'authored':
            return rel

        if not self.is_front_matter_signature(rel.evidence_text, rel.page):
            return rel

        # Check if source is the actual author (should keep as 'authored')
        if self.is_known_author(rel.source):
            return rel

        # Convert to foreword
        new_rel = copy.deepcopy(rel)
        new_rel.relationship = 'wrote foreword for'
        new_rel.flags['FRONT_MATTER_CORRECTED'] = True
        new_rel.flags['original_relationship'] = 'authored'

        return new_rel
```

**Expected Impact**: Fixes 1 CRITICAL issue (Lily Sophia case)

### Priority 2: HIGH Fixes (Pronoun Resolution)

#### Fix 2.1: Enhanced Pronoun Resolution in Pass 1 Prompt

**File**: `kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt`

**Addition to Constraints Section**:
```
5. PRONOUN RESOLUTION:
   - NEVER extract pronouns as entities
   - Resolve pronouns BEFORE extraction:
     * "our", "my", "their" → add "human" qualifier (e.g., "our immune systems" → "human immune systems")
     * "we", "us" → use "humans" or "humanity" or the specific group mentioned
     * "this", "that", "these", "those" → replace with the antecedent noun
   - Examples:
     * "Our stress levels are reduced" → Extract: "soil reduces human stress levels"
     * "We enhance our intelligence" → Extract: "soil stewardship enhances human intelligence"
     * "This process creates healthy soil" → Extract: "composting process creates healthy soil"
```

**Expected Impact**: Prevents 8 HIGH pronoun issues upstream

#### Fix 2.2: Enhanced PronounResolver Module

**File**: `src/knowledge_graph/postprocessing/universal/pronoun_resolver.py`

**Enhancement**:
```python
# V14.3.2.1 NEW: Possessive pronoun patterns
POSSESSIVE_PATTERNS = {
    'our': 'human',
    'my': 'individual',
    'their': 'human',
    'its': 'the',
}

def resolve_possessive_pronouns(self, entity: str) -> Optional[str]:
    """V14.3.2.1: Resolve possessive pronouns in entities"""
    entity_lower = entity.lower().strip()

    # Check for possessive pronouns at start
    for possessive, replacement in self.POSSESSIVE_PATTERNS.items():
        if entity_lower.startswith(possessive + ' '):
            # "our immune systems" → "human immune systems"
            rest = entity[len(possessive):].strip()
            return f"{replacement} {rest}"

    return None
```

**Expected Impact**: Catches remaining pronoun issues postprocessing can't prevent

### Priority 3: MEDIUM Fixes (List Splitting)

#### Fix 3.1: Context-Aware List Splitting

**File**: `src/knowledge_graph/postprocessing/universal/list_splitter.py`

**Enhancement**:
```python
def should_skip_split(self, target: str, conjunction_pos: int) -> bool:
    """
    V14.3.2.1: Check if we should skip splitting at this conjunction.

    Skip if:
    1. Conjunction is within quotes
    2. Conjunction appears after a colon (likely part of title)
    3. Result would be grammatically broken (starts with "and")
    """

    # Check if within quotes
    before = target[:conjunction_pos]
    quote_count = before.count('"') + before.count("'")
    if quote_count % 2 == 1:  # Odd number of quotes = inside quotes
        return True

    # Check if after colon (title pattern: "Title: Subtitle and More")
    if ':' in before and before.rindex(':') > before.rfind(','):
        return True

    # Check if result would start with "and"
    after = target[conjunction_pos:].strip()
    if after.lower().startswith('and '):
        return True

    return False
```

**Expected Impact**: Eliminates 15 MEDIUM list splitting artifacts

### Priority 4: Prompt Enhancements (Upstream Prevention)

#### Enhancement 4.1: Entity Specificity Requirements

**File**: `kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt`

**Addition**:
```
6. ENTITY SPECIFICITY:
   - Extract CONCRETE, SPECIFIC entities only
   - Avoid vague/abstract terms:
     * ❌ "cognitive performance" → ✅ "human cognitive performance"
     * ❌ "community impact" → ✅ "community engagement" or "community health"
     * ❌ "the process" → ✅ "composting process" or "soil-building process"
   - If an entity feels abstract, add qualifiers to make it specific
   - Generic terms like "performance", "impact", "levels" need context
```

**Expected Impact**: Reduces 12 MEDIUM vague entity issues

#### Enhancement 4.2: Document Structure Awareness

**File**: `kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt`

**Addition**:
```
7. DOCUMENT STRUCTURE AWARENESS:
   - Identify the section type:
     * Front matter (pages 1-15): Foreword, dedication, praise quotes, endorsements
     * Main content: Primary information source
     * Back matter: Appendices, resources, acknowledgments

   - Handle forewords/praise sections specially:
     * Foreword signatures: Extract as "wrote foreword for", NOT "authored"
     * Endorsement quotes: Extract as "endorsed", NOT "authored"
     * Only the actual author (from title page/copyright) should have "authored" relationship

   - Example:
     * "With Love, Jane Doe" at end of foreword → (Jane Doe, wrote foreword for, Book Title)
     * "Author Name, 2018" in introduction → (Author Name, authored, Book Title)
```

**Expected Impact**: Prevents 2 CRITICAL foreword misattribution issues

## Recommended Implementation Sequence

### Phase 1: Quick Wins ✅ **COMPLETE**

1. ✅ **DONE**: Set temperature=0.0 in extraction script
2. ✅ **DONE**: Implement Fix 1.1: PraiseQuoteDetector author whitelist (v1.5.0)
3. ✅ **DONE**: Implement Fix 1.2: FrontMatterDetector module (v1.0.0)
4. **READY**: Test with V14.3.2.1 extraction

### Phase 2: Prompt Enhancements (1-2 hours)

5. **Create v14_3_3 prompts** with:
   - Pronoun resolution instructions (Enhancement 4.1)
   - Entity specificity requirements (Enhancement 4.2)
   - Document structure awareness (Enhancement 4.3)
6. **Test**: Run extraction with new prompts

### Phase 3: Postprocessing Improvements (2-3 hours)

7. **Implement Fix 2.2**: Enhanced PronounResolver
8. **Implement Fix 3.1**: Context-aware ListSplitter
9. **Test**: Full extraction with all fixes

### Phase 4: Validation (30 min)

10. **Run Reflector** on final extraction
11. **Compare** with V14.3.1 and V14.3.2
12. **Verify** A or A+ grade achieved

## Expected Results After All Fixes

| Metric | V14.3.2 | V14.3.3 (Target) | Improvement |
|--------|---------|------------------|-------------|
| Total issues | 53 (11.1%) | 15-20 (3-4%) | -33 to -38 |
| CRITICAL | 2 | 0 | -2 ✅ |
| HIGH | 8 | 0-2 | -6 to -8 ✅ |
| MEDIUM | 15 | 8-10 | -5 to -7 ✅ |
| MILD | 28 | 28 | 0 (user accepts) |
| **Actionable issues** | **25** | **8-12** | **-13 to -17** ✅ |
| **Grade** | **B+** | **A or A+** | **🎯** |

## Key Insights

1. **Temperature matters**: Setting to 0 eliminates variance
2. **Upstream prevention > downstream fixing**: Prompt improvements prevent issues before they occur
3. **Document structure awareness**: Critical for handling front matter correctly
4. **Author whitelisting**: Simple but powerful fix for over-correction
5. **Pronoun resolution**: Must happen early (Pass 1) for best results

## Next Steps

**Recommended**: Start with Phase 1 (Quick Wins) to get immediate improvement, then proceed to prompt enhancements for sustainable quality.
