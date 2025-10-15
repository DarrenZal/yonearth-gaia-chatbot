# V14.3.9 Implementation Plan

## 🎯 Goal
Fix the 1 CRITICAL reversed authorship error to reach A+ grade.

## 📊 V14.3.8 Results (Baseline)
- **Grade: B**
- **Issue Rate: 11.9%**
- **101 relationships**
- **Issues: 1 CRITICAL, 0 HIGH, 3 MEDIUM, 11 MILD**

## 🐛 Root Cause Analysis

### The Bug
Found in V14.3.8 output:
```json
{
  "source": "Our Biggest Deal",
  "target": "Aaron William Perry",
  "relationship": "authored",
  "source_type": "Person",
  "target_type": "Book",
  "flags": {
    "AUTHORSHIP_REVERSED": True,
    "correction_reason": "bibliographic_citation_detected"
  }
}
```

**Problem:** The module set `AUTHORSHIP_REVERSED=True` but the entities are STILL backwards!

**Root Cause:** BibliographicCitationParser is reversing relationships that are ALREADY CORRECT.

### The Flow
1. **Pass 1** extracts correctly: `"Aaron William Perry" → authored → "Our Biggest Deal"` ✓
2. **BibliographicCitationParser** detects "bibliographic citation" and incorrectly reverses it
3. **Result**: `"Our Biggest Deal" → authored → "Aaron William Perry"` ❌
4. **Flag set**: `AUTHORSHIP_REVERSED=True` (prevents future fixes!)
5. **Types set**: `source_type=Person, target_type=Book` (correct types, wrong entities!)

### Why Detection Failed

The `validate_authorship_direction()` logic (lines 608-626):

```python
source_is_book = self.is_book_title(rel.source)      # "Our Biggest Deal" → False
target_is_person = self.is_person_name(rel.target)   # "Aaron William Perry" → True
source_is_person = self.is_person_name(rel.source)   # "Our Biggest Deal" → True!
target_is_book = self.is_book_title(rel.target)      # "Aaron William Perry" → True!
```

**Problem:** "Our Biggest Deal" (3 capitalized words) passes `is_person_name()` test!

The `is_person_name()` check (lines 162-199) only checks:
- 2-4 words ✓
- No colons/quotes ✓
- All capitalized ✓
- Word length < 15 ✓

It does NOT check for common title words like "Deal", "Book", "Guide", etc.

## ✅ Solution: Improve is_person_name() Detection

### Enhanced is_person_name() Logic

Add check for common title/book keywords that indicate NOT a person:

```python
def is_person_name(self, text: str) -> bool:
    """Enhanced with title keyword detection"""
    if not text:
        return False

    words = text.split()
    if not (2 <= len(words) <= 4):
        return False

    if ':' in text or '"' in text:
        return False

    # NEW: Check for title keywords (indicates book, not person)
    title_keywords = [
        'deal', 'book', 'handbook', 'manual', 'guide', 'introduction',
        'primer', 'companion', 'reference', 'textbook', 'workbook',
        'story', 'tale', 'journey', 'adventure', 'quest'
    ]
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in title_keywords):
        return False

    book_keywords = ['handbook', 'manual', 'guide', 'introduction', 'primer',
                    'companion', 'reference', 'textbook', 'workbook']
    if any(keyword in text_lower for keyword in book_keywords):
        return False

    if not all(w[0].isupper() for w in words if len(w) > 0):
        return False

    if any(len(w) > 15 for w in words):
        return False

    return True
```

### Test Cases

```python
is_person_name("Our Biggest Deal")  # False (has "deal")
is_person_name("Aaron William Perry")  # True
is_person_name("The Great Handbook")  # False (has "handbook")
is_person_name("John Smith")  # True
is_person_name("Regenerative Capitalism")  # True (no title keywords)
```

## 📝 Implementation Steps

1. ✅ **Update `is_person_name()`** in BibliographicCitationParser
   - Add title keyword check
   - File: `src/knowledge_graph/postprocessing/content_specific/books/bibliographic_citation_parser.py`
   - Lines: 162-199

2. ✅ **Bump version** to 1.8.0 (V14.3.9)

3. ✅ **Create V14.3.9 pipeline** in book_pipeline.py
   - Copy `get_book_pipeline_v1438()` → `get_book_pipeline_v1439()`
   - Use updated BibliographicCitationParser

4. ✅ **Create extraction script** for V14.3.9
   - Copy `extract_kg_v14_3_8_incremental.py` → `extract_kg_v14_3_9_incremental.py`
   - Update pipeline import and version references

5. ✅ **Run extraction** and validate

6. ✅ **Run Reflector** to confirm A+ grade

## 🎯 Expected Results

### From V14.3.8 (B grade):
- **1 CRITICAL** (reversed authorship)
- **Issue rate: 11.9%**

### Expected V14.3.9:
- **0 CRITICAL** ✓
- **0-2 HIGH** ✓
- **Issue rate: ≤5%** ✓
- **Grade: A or A+** 🎉

## 📋 Files to Modify/Create

### Modified:
1. `src/knowledge_graph/postprocessing/content_specific/books/bibliographic_citation_parser.py`
   - Update `is_person_name()` method
   - Bump version to 1.8.0

### Created:
1. `src/knowledge_graph/postprocessing/pipelines/book_pipeline.py` (add function)
   - `get_book_pipeline_v1439()`

2. `scripts/extract_kg_v14_3_9_incremental.py`

3. `V14_3_9_IMPLEMENTATION_PLAN.md` (this file)

---

**Status:** Ready for implementation
**Date:** 2025-10-15
**Expected Runtime:** ~4 minutes for extraction + reflector
