# Entity Merge Problem - AI Agent Review Request

## Problem Statement

We need an entity deduplication strategy that:
1. ✅ **Merges legitimate abbreviations**: "Dr. Bronner's" ⟷ "Dr Bronners" ⟷ "Bronners"
2. ✅ **Prevents catastrophic merges**: "Moscow" ≠ "moon", "Moscow" ≠ "Soil"
3. ✅ **Handles case variations**: "Organization" = "organization" = "ORGANIZATION"
4. ✅ **Works at scale**: 18,000+ entities, 19,000+ relationships

## Current Solution & Its Limitations

### What We Fixed (ROOT CAUSE ✅)
**Type Normalization** in `scripts/build_unified_graph_hybrid.py`:
- Added `_normalize_type()` method that canonicalizes entity types
- `organization` → `ORGANIZATION`, `person` → `PERSON`, etc.
- **Result**: Reduced from 1,053 unique types → ~150 types
- **Impact**: Entities with same semantic type can now be compared for merging

### Current Merge Logic
Located in `scripts/build_unified_graph_hybrid.py` → `deduplicate_with_validation()`:

```python
# Group entities by normalized type
for entity_type, entity_list in entities_by_type.items():
    for entity_id, entity in entity_list:
        for other_id, other_entity in entity_list:
            # Fuzzy similarity check
            similarity = fuzz.ratio(entity_name, other_name)

            if similarity >= 95:  # THRESHOLD
                # Validator checks
                if self.validator:
                    can_merge, reason = self.validator.can_merge(entity, other_entity)
                    if not can_merge:
                        continue  # Block merge

                # Merge approved
                similar_entities.append((other_id, other_entity))
```

### Validator Logic
Located in `src/knowledge_graph/validators/entity_merge_validator.py`:

**Checks performed:**
1. **Type compatibility**: entity1.type == entity2.type (STRICT)
2. **Length ratio**: min(len1, len2) / max(len1, len2) >= 0.6
3. **Fuzzy threshold**: fuzz.ratio(name1, name2) >= 95
4. **Explicit blocklist**: ("moscow", "soil"), ("earth", "mars"), etc.
5. **Semantic compatibility**: Share at least one token for multi-word names

### Current Performance

**Fuzzy Similarity Analysis:**
```
✅ 96% - "Dr Bronners" vs "Dr Bronner's"         → MERGES
✅ 96% - "Dr. Bronners" vs "Dr. Bronner's"        → MERGES
❌ 84% - "Bronners" vs "Dr Bronners"              → BLOCKED (abbreviation)
❌ 92% - "Dr. Bronner's" vs "Dr. Bronner"         → BLOCKED (apostrophe)
❌ 80% - "Doctor Bronner" vs "Dr. Bronner"        → BLOCKED (abbreviated title)

✅ 100% - "Moscow" vs "Moscow"                     → MERGES
❌ ~70% - "Moscow" vs "moon"                       → BLOCKED (catastrophic)
❌ ~65% - "Moscow" vs "soil"                       → BLOCKED (catastrophic)
```

## The Challenge

**Threshold Tradeoff:**
- **95% threshold**:
  - ✅ Blocks catastrophic merges (Moscow≠moon at ~70%)
  - ✅ Prevents false positives
  - ❌ Blocks legitimate abbreviations ("Bronners" vs "Dr Bronners" at 84%)

- **85% threshold**:
  - ✅ Merges more abbreviations
  - ⚠️ Risk: Could allow "Moscow"+"Moss" (85%), "Leaders"+"Healers" (85%)
  - ❌ Higher false positive rate

## Proposed Solutions (Need Review)

### Solution 1: Token-Based Secondary Validation
**Idea**: For 85-94% similarity, require shared significant tokens

```python
def _has_shared_significant_token(self, name1: str, name2: str) -> bool:
    """Check if names share at least one significant token (length > 3)"""
    # Remove common prefixes/suffixes
    stop_words = {'dr', 'dr.', 'the', 'a', 'an', 'mr', 'mrs', 'ms'}

    tokens1 = {t.lower().strip('.,!?') for t in name1.split() if len(t) > 3}
    tokens2 = {t.lower().strip('.,!?') for t in name2.split() if len(t) > 3}

    tokens1 -= stop_words
    tokens2 -= stop_words

    return len(tokens1.intersection(tokens2)) > 0

# Usage:
if 85 <= similarity < 95:
    if self._has_shared_significant_token(name1, name2):
        # Additional validation passed - allow merge
```

**Analysis:**
- ✅ "Bronners" ∩ "Dr Bronners" = {"bronners"} → ALLOW merge at 84%
- ✅ "Dr. Bronner's" ∩ "Dr. Bronner" = {"bronner"} → ALLOW merge at 92%
- ✅ "Moscow" ∩ "moon" = {} → BLOCK at 70%
- ✅ "Leaders" ∩ "Healers" = {} → BLOCK at 85%
- ⚠️ "Soil Conservation" ∩ "Soil" = {"soil"} → Would ALLOW (is this OK?)

### Solution 2: Abbreviation Normalization Pre-Processing
**Idea**: Normalize abbreviations before fuzzy comparison

```python
ABBREVIATION_MAP = {
    'dr.': 'doctor',
    'dr': 'doctor',
    'mr.': 'mister',
    'mrs.': 'misses',
    'st.': 'saint',
    'co.': 'company',
    'corp.': 'corporation',
    'inc.': 'incorporated',
}

def _normalize_abbreviations(self, name: str) -> str:
    """Expand common abbreviations for better matching"""
    tokens = name.lower().split()
    normalized = []
    for token in tokens:
        token_clean = token.strip('.,!?')
        normalized.append(ABBREVIATION_MAP.get(token_clean, token))
    return ' '.join(normalized)

# Usage:
norm1 = self._normalize_abbreviations(name1)
norm2 = self._normalize_abbreviations(name2)
similarity = fuzz.ratio(norm1, norm2)
```

**Analysis:**
- "Dr. Bronner's" → "doctor bronner's"
- "Bronners" → "bronners"
- Still only ~80% similar (different base form)
- Doesn't fully solve the problem

### Solution 3: Two-Tier Threshold Strategy
**Idea**: Different thresholds for different contexts

```python
def can_merge_with_context(self, entity1, entity2):
    name1, name2 = entity1['name'], entity2['name']
    similarity = fuzz.ratio(name1.lower(), name2.lower())

    # Tier 1: High confidence (95%+)
    if similarity >= 95:
        return self.can_merge(entity1, entity2)  # Full validation

    # Tier 2: Medium confidence (85-94%) - EXTRA checks required
    elif 85 <= similarity < 95:
        # Require: same type AND shared significant token
        if entity1['type'] != entity2['type']:
            return False, "type_mismatch_tier2"

        if not self._has_shared_significant_token(name1, name2):
            return False, "no_shared_token_tier2"

        # Additional check: not on blocklist
        if self._is_blocked_pair(name1, name2):
            return False, "explicit_blocklist"

        return True, f"tier2_approved: similarity={similarity}"

    # Tier 3: Low confidence (<85%) - reject
    else:
        return False, f"low_similarity: {similarity}"
```

**Analysis:**
- ✅ Provides safety net (tier 1) for high confidence
- ✅ Allows abbreviations (tier 2) with extra validation
- ✅ Blocks low similarity completely (tier 3)
- ⚠️ More complex logic - harder to debug

### Solution 4: Machine Learning Approach (Future)
**Idea**: Train a binary classifier on merge pairs

**Features:**
- Fuzzy similarity score
- Jaccard similarity of tokens
- Edit distance
- Shared prefix/suffix length
- Entity type compatibility
- Length ratio
- Has punctuation difference
- Number of tokens

**Training data:**
- Positive examples: Known good merges
- Negative examples: Known bad merges + blocklist

**Not implementing now** - requires significant effort

## Data Context

### Entity Statistics (Current Build)
- **Total entities**: 17,827 (after type normalization)
- **Total relationships**: 19,331
- **Top entity types**:
  - CONCEPT: 8,685
  - PERSON: 4,245
  - ORGANIZATION: 2,696
  - PLACE: 1,582
  - EVENT: 562

### Source Data
- **Episode files**: 41 ACE-postprocessed episode JSON files
  - Location: `data/knowledge_graph_unified/episodes_postprocessed/`
  - Format: Each has `relationships` array with `source`, `target`, `source_type`, `target_type`

- **Book files**: 4 book extraction JSON files
  - Location: `data/knowledge_graph/entities/`
  - Books: VIRIDITAS, Soil Stewardship Handbook, Y on Earth, Our Biggest Deal
  - Format: Each has `entities` array and `relationships` array

### Known Problematic Pairs (MUST NOT MERGE)
From `entity_merge_validator.py` MERGE_BLOCKLIST:
```python
('moscow', 'soil'),
('moscow', 'moon'),
('earth', 'mars'),
('earth', 'paris'),
('leaders', 'healers'),
('leaders', 'readers'),
('organization', 'urbanization'),
('the soil', 'the stove'),
('dia', 'dubai'),
('dia', 'india'),
```

## Questions for AI Review

1. **Which solution (1-3) is most robust?** Consider:
   - Precision (no false positives)
   - Recall (captures legitimate abbreviations)
   - Maintainability
   - Performance at scale

2. **Are there edge cases we're missing?**
   - Acronyms (NASA vs N.A.S.A.)
   - Possessives ("Bronner's" vs "Bronners")
   - Hyphenation ("Y-on-Earth" vs "Y on Earth")
   - Unicode/diacritics

3. **Should we lower the tier-1 threshold from 95% to 92-93%?**
   - Would this catch more legitimate variants?
   - What's the risk of false positives?

4. **Token-based validation concerns:**
   - Is "shared significant token" too permissive?
   - Should we require token overlap ratio (e.g., 50% of tokens must match)?
   - How to handle single-word entities?

5. **Alternative approaches we haven't considered?**

## Implementation Files

If you want to implement a solution, modify these files:

1. **Main builder** (loads & deduplicates):
   - `scripts/build_unified_graph_hybrid.py`
   - Method: `deduplicate_with_validation()`
   - Already has type normalization (✅ fixed)

2. **Validator** (merge logic):
   - `src/knowledge_graph/validators/entity_merge_validator.py`
   - Method: `can_merge(entity1, entity2)`
   - Add new validation logic here

3. **Test the fix**:
   ```bash
   # Rebuild from source with new validation logic
   python3 scripts/build_unified_graph_hybrid.py --similarity-threshold 95

   # Check results
   python3 -c "
   import json
   with open('data/knowledge_graph_unified/unified_normalized.json', 'r') as f:
       data = json.load(f)
   bronners = {k: v for k, v in data['entities'].items() if 'bronner' in k.lower()}
   print(f'Dr Bronner variants: {len(bronners)}')
   for name in sorted(bronners.keys())[:10]:
       print(f'  - {name}')
   "
   ```

## Success Criteria

A successful solution should:
- ✅ Merge "Dr. Bronner's" ⟷ "Dr Bronners" ⟷ "Bronners" (same org)
- ✅ Merge "Dr. Bronner" ⟷ "Doctor Bronner" (same person)
- ✅ Keep Moscow ≠ moon ≠ Soil (catastrophic merges blocked)
- ✅ Keep Leaders ≠ Healers (semantic differences)
- ✅ Handle 18,000+ entities efficiently (<5 min rebuild time)
- ✅ Be maintainable and debuggable

## Your Task

**Please review this problem and provide:**
1. Your assessment of the proposed solutions
2. Any improvements or alternative approaches
3. Specific implementation recommendations
4. Potential risks or edge cases we should test

**Context you have:**
- Full problem statement and constraints
- Current implementation code locations
- Performance data and examples
- Success criteria

**What we need:**
- Your expert opinion on the best path forward
- Concrete code suggestions if you have them
- Validation logic that balances precision and recall
