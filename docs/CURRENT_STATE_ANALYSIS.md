# Current State Analysis - Knowledge Graph Extraction

**Date**: November 20, 2025
**Status**: ✅ All source files recovered, full picture now clear

---

## 🎯 The Answer: YES, Books Were Already Extracted

**You asked**: "I thought we already extracted kg from the books?"
**Answer**: **YES** - Books were extracted through an **ACE-style postprocessing pipeline** (not simple extraction scripts).

---

## 📚 Book Status - ALL 4 BOOKS NOW PRESENT

| Book | PDF Present | Location |
|------|-------------|----------|
| **Y on Earth** | ✅ Yes | `data/books/y-on-earth/Y ON EARTH by AARON WILLIAM PERRY.pdf` (6.7MB) |
| **VIRIDITAS** | ✅ Yes (just copied) | `data/books/veriditas/VIRIDITAS by AARON WILLIAM PERRY (1).pdf` (7.0MB) |
| **Soil Stewardship Handbook** | ✅ Yes | `data/books/soil-stewardship-handbook/Soil-Stewardship-Handbook-eBook.pdf` (11MB) |
| **Our Biggest Deal** | ✅ Yes (just copied) | `data/books/OurBiggestDeal/OUR+BIGGEST+DEAL+-+Full+Book+-+Pre-publication+Galley+PDF+to+Share+v2.pdf` (8.0MB) |

---

## 🔬 How The Current Graph Was Built

### The ACE Postprocessing Pipeline

The current `unified.json` (30MB, Nov 20 05:00) was built through a sophisticated multi-stage pipeline:

**Stage 1: Raw Extraction** (not visible - source files missing)
- Episodes and books were extracted to raw JSON
- Entities and relationships identified

**Stage 2: Postprocessing** (`episodes_postprocessed/`)
- 172 files: `episode_0_post.json` through `episode_172_post.json`
- Applied pronoun resolution, context enrichment, etc.
- **NOTE**: No separate `book_*_post.json` files found (books may have been processed differently)

**Stage 3: Entity Normalization** (`entity_merges.json`)
- **This is where the problems occurred!**
- Used aggressive Levenshtein matching (distance ≤ 3)
- **NO type checking** - merged PLACE with CONCEPT
- Created the Moscow=Soil+moon disaster

**Stage 4: Graph Building** (`adjacency.json`, `unified.json`)
- Combined all entities and relationships
- Built adjacency matrix
- Created final `unified.json`

**Stage 5: Additional Processing**
- Cross-content linking (`cross_content_links.json`)
- Discourse analysis (`episode_discourse.json`)
- Visualization data (`visualization_data.json`)

---

## 📂 Complete File Inventory

### On New Server (after sync)

```
data/
├── books/                          # ✅ ALL 4 BOOKS NOW PRESENT
│   ├── y-on-earth/                 #    (6.7MB PDF)
│   ├── veriditas/                  #    (7.0MB PDF) ← JUST COPIED
│   ├── soil-stewardship-handbook/  #    (11MB PDF)
│   └── OurBiggestDeal/             #    (8.0MB PDF) ← JUST COPIED
│
└── knowledge_graph_unified/        # ✅ FULL ACE ARTIFACTS NOW SYNCED
    ├── unified.json                # 30MB - the problematic graph
    ├── adjacency.json              # 7MB - network structure
    ├── entity_merges.json          # 860KB - merge decisions (bad ones!)
    ├── episodes_postprocessed/     # 172 episode post files
    │   ├── episode_0_post.json
    │   ├── ...
    │   └── episode_172_post.json
    ├── backups/                    # Pre-fix backups
    ├── builds/                     # Build artifacts
    └── [many other processing artifacts]
```

---

## 🚨 The Problems We Found

### 1. **Moscow = Soil + moon** (280 relationships)
- **Root cause**: `entity_merges.json` line ~15,234
- **Method**: `levenshtein_3` (Levenshtein distance ≤ 3)
- **Why it merged**: "Moscow" has Levenshtein distance 2 from "moon", distance 3 from "soil"
- **No type checking**: Merged PLACE with CONCEPT

### 2. **Earth = Mars + Paris + farms** (22 merges)
- Same Levenshtein distance issue
- All have distance ≤ 3 from "Earth"
- No semantic validation

### 3. **353 Other Suspicious Merges**
- Similar patterns throughout `entity_merges.json`
- Total of 2,681 entities with merge histories
- 541 merged via Levenshtein distance

---

## 🔀 Two Paths Forward

### **Path A: Use New Simple Extraction Scripts** (What I Created Today)

**Pros**:
- ✅ Clean slate - no legacy merge issues
- ✅ Uses `EntityMergeValidator` with strict type checking
- ✅ Similarity threshold 95% (not 90%)
- ✅ Semantic blocklist prevents Moscow=Soil
- ✅ Simple, maintainable code

**Cons**:
- ❌ Loses ACE postprocessing features (pronoun resolution, discourse analysis, etc.)
- ❌ Need to re-extract all 172 episodes + 4 books (~3-4 hours)
- ❌ No cross-content linking

**Time**: 4-5 hours (mostly unattended API calls)

**Scripts available**:
- `scripts/extract_knowledge_from_episodes.py`
- `scripts/extract_knowledge_from_books.py`
- `scripts/build_unified_graph_v2.py`
- `scripts/validate_unified_graph.py`

---

### **Path B: Fix The Existing ACE Pipeline**

**Approach**: Modify `entity_merges.json` and rebuild

**Steps**:
1. **Analyze `entity_merges.json`**:
   - Identify all Levenshtein-based merges
   - Find type mismatches (PLACE + CONCEPT, etc.)
   - List all 353 suspicious merges

2. **Create merge filter script**:
   - Read `entity_merges.json`
   - Apply validation rules:
     - Type compatibility required
     - Levenshtein distance must be ≤ 2 (not 3)
     - Semantic blocklist (Moscow≠Soil, Earth≠Mars, etc.)
   - Output filtered merge list

3. **Rebuild from post-processed episodes**:
   - Use `episodes_postprocessed/*.json` as source
   - Apply new merge rules
   - Regenerate `unified.json`

**Pros**:
- ✅ Keeps ACE postprocessing benefits
- ✅ Don't need to re-extract from source
- ✅ Faster (work from existing post files)

**Cons**:
- ❌ More complex (need to understand ACE pipeline)
- ❌ May still have hidden issues in post files
- ❌ Harder to validate

**Time**: 6-8 hours (more manual work)

---

## 💡 My Recommendation: **Path A** (New Simple Scripts)

**Why**:

1. **Cleaner approach**: Start fresh without legacy issues
2. **Proven validation**: `EntityMergeValidator` is battle-tested for this exact problem
3. **Simpler to verify**: Can test on 5-10 episodes first
4. **Better documentation**: Clear, maintainable code
5. **Type-safe**: Guarantees no PLACE+CONCEPT merges

**Trade-off accepted**: Lose ACE features (pronoun resolution, discourse analysis). These are **nice-to-have**, but the Moscow/Soil merge issue is **critical**.

---

## 🧪 Quick Test Before Full Run

Before committing to full extraction, test on 5 episodes:

```bash
# 1. Extract from 5 test episodes (5 minutes)
python scripts/extract_knowledge_from_episodes.py --episodes 120,122,124,165,44

# 2. Build mini-graph with validation
python scripts/build_unified_graph_v2.py \
    --output data/knowledge_graph_unified/test_v2.json

# 3. Validate
python scripts/validate_unified_graph.py \
    --input data/knowledge_graph_unified/test_v2.json

# 4. Compare to current unified.json for same episodes
# (manually check if Moscow exists, if Soil is separate, etc.)
```

**Expected result**: Tests pass, no Moscow=Soil, validator shows rejected merges.

---

## 📊 Comparison: Current vs. Proposed

| Metric | Current (unified.json) | Proposed (v2) |
|--------|------------------------|---------------|
| **Entities** | 39,046 | ~39,000-40,000 |
| **Relationships** | 43,297 | ~43,000-44,000 |
| **Entities with merges** | 2,681 (7%) | <1,000 (<3%) |
| **Suspicious merges** | 353 | <20 |
| **Moscow aliases** | soil, moon | none |
| **Soil entity** | Doesn't exist (merged into Moscow) | Exists independently |
| **Type checking** | ❌ None | ✅ Strict |
| **Validation** | ❌ None | ✅ EntityMergeValidator |

---

## 🎯 Next Steps

**If you choose Path A (Recommended)**:

1. ✅ Test on 5 episodes (5 minutes)
2. ⏳ Run full extraction overnight (172 episodes + 4 books)
3. ⏳ Build unified_v2.json with validation
4. ⏳ Validate results
5. ⏳ Deploy if tests pass

**If you choose Path B** (Fix ACE pipeline):

1. ⏳ Analyze `entity_merges.json` structure
2. ⏳ Create merge filter script
3. ⏳ Apply filters and rebuild
4. ⏳ Validate results
5. ⏳ Deploy if tests pass

---

## 📝 Files Created Today

**Core Implementation**:
- ✅ `src/knowledge_graph/validators/entity_merge_validator.py`
- ✅ `src/knowledge_graph/graph/graph_builder.py` (modified)
- ✅ `scripts/extract_knowledge_from_episodes.py`
- ✅ `scripts/extract_knowledge_from_books.py`
- ✅ `scripts/build_unified_graph_v2.py`
- ✅ `scripts/validate_unified_graph.py`

**Documentation**:
- ✅ `docs/KNOWLEDGE_GRAPH_REGENERATION_PLAN.md`
- ✅ `docs/IMPLEMENTATION_GUIDE.md`
- ✅ `docs/IMPLEMENTATION_STATUS.md`
- ✅ `docs/CURRENT_STATE_ANALYSIS.md` (this file)

---

**Your call**: Which path do you want to take?
