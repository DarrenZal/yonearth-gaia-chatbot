# V9 Implementation Status

## ✅ Completed

### 1. Enhanced Prompts Created
- ✅ **Pass 1 Extraction Prompt (v9)**: `/kg_extraction_playbook/prompts/pass1_extraction_v9.txt`
  - Added critical extraction rules (avoid pronouns, abstractions, metaphors)
  - Added entity conciseness guidelines
  - Added 5 few-shot examples showing good vs bad extractions
  - Narrowed relationship types to focus on concrete, verifiable claims

- ✅ **Pass 2 Evaluation Prompt (v9)**: `/kg_extraction_playbook/prompts/pass2_evaluation_v9.txt`
  - Added calibrated knowledge plausibility scoring (0.0-0.3 for philosophical, 0.9-1.0 for facts)
  - Added explicit examples for each score range
  - Added conflict detection criteria with 4 case examples
  - Added guidance on entity type assessment

### 2. V9 Script Setup
- ✅ Copied `extract_kg_v8_book.py` to `extract_kg_v9_book.py`
- ✅ Updated header documentation describing V9 fixes
- ✅ Updated log filename to `kg_extraction_book_v9_*.log`
- ✅ Updated OUTPUT_DIR to `output/v9`
- ✅ Updated prompt_version to `v9_reflector_fixes`
- ✅ Updated extractor_version to `2025.10.13_v9`

---

## ⏳ Remaining Code Implementations

### CRITICAL Fix #1: Enhanced Praise Quote Attribution Logic
**Location**: `PraiseQuoteDetector` class (line ~241-309 in v9 script)

**Current Issue**: V8 changes relationship type from "authored" to "endorsed" but doesn't fix the source attribution. The SUBJECT (Perry) is kept as source when the SPEAKER (Adrian Del Caro) should be the source.

**Required Changes**:
1. Add method to extract SPEAKER from attribution marker (`—Name, Title`)
2. Add method to identify if current source is the SUBJECT being praised
3. Swap source from SUBJECT to SPEAKER when detected
4. Handle credential bylines separately (e.g., "—Adrian Del Caro, Author of X")

**Implementation Pseudocode**:
```python
def fix_praise_quote_attribution(self, rel, evidence_text):
    # Extract speaker from attribution marker
    speaker_match = re.search(r'—([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', evidence_text)
    if not speaker_match:
        return rel

    speaker_name = speaker_match.group(1)

    # Check if current source is subject being praised
    subject_patterns = [
        r'(\w+)\s+has given us',
        r'(\w+)\s+provides',
        r'(\w+)\'s?\s+(?:handbook|book|work)',
        r'with (?:his|her)\s+.*?,\s+(\w+)\s+'
    ]

    for pattern in subject_patterns:
        match = re.search(pattern, evidence_text, re.IGNORECASE)
        if match and match.group(1).lower() in rel.source.lower():
            # Swap source
            rel.source = speaker_name
            rel.flags['PRAISE_ATTRIBUTION_FIXED'] = True
            break

    return rel
```

**Expected Impact**: Fixes 3 CRITICAL errors (0.28%)

---

### CRITICAL Fix #2: p_true Threshold Filtering
**Location**: `build_relationship_from_eval` function (line ~1730 in v9 script)

**Required Changes**:
1. Add constant: `MIN_P_TRUE_THRESHOLD = 0.5`
2. After Pass 2 evaluation, before Pass 2.5 postprocessing:
   ```python
   if p_true < MIN_P_TRUE_THRESHOLD:
       logger.debug(f"   Filtered low-confidence relationship (p_true={p_true:.2f}): {eval_rel.source} → {eval_rel.target}")
       stats['low_confidence_filtered'] += 1
       return None  # Skip this relationship
   ```
3. Add `'low_confidence_filtered': 0` to stats initialization
4. Log filtered count in Pass 2.5 stats output

**Expected Impact**: Filters ~8 philosophical/abstract statements (0.73%)

---

### HIGH Fix #3: Enhanced Possessive Pronoun Resolution
**Location**: `PronounResolver` class (line ~500-700 in v9 script)

**Required Changes**:
1. Add possessive patterns to `__init__`:
   ```python
   self.possessive_patterns = [
       (r'\bmy\s+(\w+)', 'author_possessive'),
       (r'\bour\s+(\w+)', 'collective_possessive'),
       (r'\btheir\s+(\w+)', 'third_person_possessive')
   ]
   ```

2. Add `resolve_possessive_pronoun` method:
   ```python
   def resolve_possessive_pronoun(self, entity, context_window):
       for pattern, poss_type in self.possessive_patterns:
           match = re.search(pattern, entity, re.IGNORECASE)
           if not match:
               continue

           possessed_noun = match.group(1)

           if poss_type == 'author_possessive':
               # Look for country/place names
               place_match = re.search(r'\b([A-Z][a-z]+(?:ia|land|stan))\b', context_window)
               if place_match:
                   if possessed_noun in ['people', 'ancestors']:
                       return place_match.group(1) + 'ans'  # Slovenia → Slovenians
                   elif possessed_noun == 'land':
                       return place_match.group(1)

           elif poss_type == 'collective_possessive':
               # Similar logic for "our X"
               place_match = re.search(r'\b([A-Z][a-z]+(?:ia|land|stan))\b', context_window)
               if place_match:
                   if possessed_noun in ['land', 'country', 'region']:
                       return place_match.group(1)
                   elif possessed_noun in ['people', 'tradition', 'connection']:
                       return f"{place_match.group(1)} {possessed_noun}"

       return entity
   ```

3. Call `resolve_possessive_pronoun` in `process_batch` method before existing pronoun resolution logic

**Expected Impact**: Fixes 12 HIGH possessive pronoun errors (1.10%)

---

### Fix #4 & #5: Update Prompt Loading
**Location**: Prompt loading functions (line ~1900+ in v9 script)

**Required Changes**:
Find where prompts are loaded and update version string:
```python
# Current (V8):
pass1_prompt_path = prompts_dir / "pass1_extraction_v8_curator_ace.txt"
pass2_prompt_path = prompts_dir / "pass2_evaluation_v8_curator_ace.txt"

# Update to (V9):
pass1_prompt_path = prompts_dir / "pass1_extraction_v9.txt"
pass2_prompt_path = prompts_dir / "pass2_evaluation_v9.txt"
```

**Expected Impact**: Enables all prompt-based improvements (prevents ~33 errors total from fixes #4 and #5)

---

## Testing & Validation

### Test V9 Extraction
```bash
python3 scripts/extract_kg_v9_book.py
```

**Expected Output**:
- Creates `kg_extraction_playbook/output/v9/soil_stewardship_handbook_v9.json`
- Log file shows V9 version and new stats
- Should process all 46 pages of Soil Stewardship Handbook

### Run Reflector on V9
```bash
python3 scripts/run_reflector_on_v9.py  # Need to create this script
```

**Expected Quality Improvement**:
- V8: 91 issues (8.35%)
- V9 Target: 35-40 issues (3.2-3.7%) ✅ Below 5% production threshold

---

## Next Steps

1. **Implement remaining code fixes** (3 critical/high priority changes)
2. **Update prompt loading** to use v9 prompts
3. **Test V9 extraction** on Soil Stewardship Handbook
4. **Create run_reflector_on_v9.py** script
5. **Run Reflector analysis** on V9 output
6. **Document V9 results** in ACE_CYCLE_1_V9_RESULTS.md

---

## File Locations

- **V9 Script**: `/home/claudeuser/yonearth-gaia-chatbot/scripts/extract_kg_v9_book.py`
- **V9 Pass 1 Prompt**: `/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts/pass1_extraction_v9.txt`
- **V9 Pass 2 Prompt**: `/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts/pass2_evaluation_v9.txt`
- **V9 Output**: `/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/v9/`
- **Implementation Plan**: `/home/claudeuser/yonearth-gaia-chatbot/docs/knowledge_graph/V9_IMPLEMENTATION_PLAN.md`
