# Entity Resolution Guide (Basic Approach)

## Overview

This guide describes the **basic entity resolution approach** using incremental alias learning. This is the recommended starting point for most users.

**📚 For advanced entity resolution** with graph embeddings, multi-signal matching, and automated deduplication, see **[ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md](ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md)**.

---

Entity resolution in knowledge graph extraction follows the **incremental learning approach** described in the master guides. Instead of hardcoding aliases, the system learns entity mappings during human review and stores them in a reusable configuration file.

**When to use this approach:**
- ✅ You want a simple, manual entity resolution workflow
- ✅ You prefer to review and confirm all duplicates yourself
- ✅ Your extraction has <100 entities with duplicates
- ✅ You want full control over alias decisions

**When to use the comprehensive approach:**
- 🔬 You have 1000+ entities to deduplicate
- 🔬 You want automated duplicate detection
- 🔬 You need graph embeddings and multi-signal matching
- 🔬 See: [ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md](ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md)

## How It Works

### 1. Generic Normalization (Always Active)

The `canon()` function provides basic normalization for ALL entity names:

```python
def canon(s: str) -> str:
    """Normalize entity strings for robust matching"""
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)  # Drop punctuation
    s = re.sub(r"\s+", " ", s)       # Normalize whitespace
    return s
```

**This catches:**
- Case variations: "Aaron" → "aaron"
- Punctuation: "25x'25" → "25x 25"
- Unicode: "café" → "cafe"
- Extra spaces: "Y  on Earth" → "y on earth"

### 2. Alias Configuration (Learned During Review)

After extraction, when you find duplicates during review, add them to an alias configuration file:

**Example: `data/knowledge_graph/entity_aliases.json`**

```json
{
  "Aaron": "Aaron William Perry",
  "Aaron Perry": "Aaron William Perry",

  "Y on Earth": "Y on Earth",
  "YonEarth": "Y on Earth",
  "Y on Earth: Get Smarter, Feel Better, Heal the Planet": "Y on Earth",

  "IBI": "International Biochar Initiative",
  "International Biochar Initiative": "International Biochar Initiative"
}
```

### 3. Using Alias Configuration

Pass the alias file to extraction scripts:

**For Books:**
```python
from pathlib import Path

results = extract_knowledge_graph_from_book(
    book_title="My Book",
    pdf_path=Path("/path/to/book.pdf"),
    run_id="run_20251011",
    alias_file="/path/to/entity_aliases.json"  # Optional!
)
```

**For Episodes:**
```python
# Similar pattern - add alias_file parameter
```

### 4. Re-running Extraction

When you add new aliases to the configuration:
1. Update the JSON file with new mappings
2. Re-run extraction with the updated alias file
3. The system will automatically apply all learned aliases

## What Gets Saved

Every relationship preserves the **original surface forms**:

```json
{
  "source": "Aaron William Perry",  // Canonicalized
  "target": "Y on Earth",           // Canonicalized
  "evidence": {
    "source_surface": "Aaron",      // Original from text
    "target_surface": "Y on Earth: Get Smarter, Feel Better, Heal the Planet"
  }
}
```

This allows you to:
- Track how entities were originally mentioned
- Debug extraction issues
- Understand context better

## Building Your Alias File

### Step 1: Extract Without Aliases

First run extracts entities as-is:

```bash
python scripts/extract_kg_v3_2_2_book_improved.py
# No alias file → uses entity names as extracted
```

### Step 2: Find Duplicates

Review the output and identify duplicates:

```bash
# Use the deduplicate_entities.py script to find candidates
python scripts/deduplicate_entities.py --analyze
```

### Step 3: Build Alias Config

Create your alias file from the duplicates found:

```json
{
  "variant1": "Canonical Name",
  "variant2": "Canonical Name"
}
```

### Step 4: Re-extract with Aliases

Run extraction again with the alias file:

```bash
# Modify the script to pass alias_file parameter
# Or create a new script that uses your alias file
```

## Benefits of This Approach

✅ **No Hardcoding**: Code stays generic, works for any domain

✅ **Reusable**: Alias file grows over time, improves future extractions

✅ **Traceable**: Surface forms preserved for debugging

✅ **Incremental**: Learn aliases as you review, no upfront work

✅ **Portable**: Alias file can be version controlled, shared

## Post-Extraction Automated Resolution

For more advanced deduplication after extraction, see:
- `KG_POST_EXTRACTION_REFINEMENT.md` for Splink-based automated entity resolution
- Automated methods can handle 1000s of entities in seconds

## Example Workflow

```
1. Extract → Entities as-is (aaron, Aaron William Perry)
2. Review → Identify duplicates
3. Update → entity_aliases.json
4. Re-extract → Unified entities (Aaron William Perry)
5. Iterate → Keep adding aliases as you find them
```

## Template File

See `data/knowledge_graph/entity_aliases_template.json` for a documented template with examples.

---

## Next Steps

**For advanced entity resolution:**

If you need to deduplicate thousands of entities automatically, see the **[Comprehensive Entity Resolution Guide](ENTITY_RESOLUTION_COMPREHENSIVE_GUIDE.md)** which implements:

- ✨ Graph embeddings (PyKEEN with RotatE)
- ✨ Multi-signal matching (name + type + relationships + embeddings)
- ✨ Automated duplicate detection
- ✨ Active learning (65% reduction in annotation effort)
- ✨ Incremental processing (112× speedup)

**When to graduate to comprehensive approach:**
- You have >100 potential duplicate pairs to review
- Manual review is taking too long
- You want probabilistic matching with confidence scores
- You need to process updates incrementally
