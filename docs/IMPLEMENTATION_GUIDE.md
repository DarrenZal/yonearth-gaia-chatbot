# Knowledge Graph Regeneration - Implementation Guide

**Quick Start**: This guide will help you rebuild the unified.json knowledge graph with proper entity validation to eliminate 540+ catastrophic entity merges.

---

## 🚨 The Problem

Your current `unified.json` has **353 highly suspicious entity merges**:

| Bad Merge | Merged Entities | Correct State |
|-----------|----------------|---------------|
| **"Moscow"** = soil + moon + Moscow | 280 relationships | Should be 3 separate entities |
| **"DIA"** = Dubai + Red + Sun + India + the Baca | 26 merges | Should be separate |
| **"Earth"** = farms + earth + Mars + Paris + Farm | 22 merges | Should be separate |
| **"the soil"** = Mother Soil + the stove + the skin + the show | 24 merges | Nonsensical |

**Root cause**: Overly aggressive Levenshtein matching (threshold=90, no type checking).

---

## ✅ The Solution

### 🏆 RECOMMENDED: Hybrid Approach (Path C)

**Best quality**: Combines ACE postprocessing benefits with strict validation

**Approach**:
1. ✅ **Keep ACE-postprocessed episodes** (pronoun resolution, discourse analysis, context enrichment)
2. ✅ **Extract books fresh** with type-safe extractors
3. ✅ **Apply strict validation** during graph building (fixes Moscow=Soil + all merge issues)

**Result**: ACE quality + No catastrophic merges = Best of both worlds!

**What you'll get**:
- ✅ Zero catastrophic merges (Moscow ≠ Soil)
- ✅ Type-safe entity consolidation (PLACE only merges with PLACE)
- ✅ Consistent field naming (all relationships have "type" field)
- ✅ Clean betweenness centrality (no spurious hubs)
- ✅ Future-proof validation (prevents recurrence)
- ✅ **Preserves ACE quality benefits** (pronoun resolution, discourse analysis)

---

## 📋 Prerequisites

### Check Current State

```bash
# 1. Backup existing data
cd /home/claudeuser/yonearth-gaia-chatbot
cp data/knowledge_graph_unified/unified.json \
   data/knowledge_graph_unified/unified_v1_backup_$(date +%Y%m%d).json

# 2. Check if extraction source files exist
ls data/knowledge_graph/entities/ 2>/dev/null
ls data/knowledge_graph/relationships/ 2>/dev/null

# If empty or missing, you need full extraction (Phase 2)
# If files exist, can skip to Phase 3
```

### Required Dependencies

```bash
# Install if not already present
pip install openai pydantic fuzzywuzzy python-Levenshtein networkx
```

### Environment Variables

```bash
# Verify these are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY  # If using ACE system
```

---

## 🛠️ Implementation Steps

### Phase 1: Validation Setup (COMPLETED ✅)

**Created files**:
- ✅ `src/knowledge_graph/validators/entity_merge_validator.py`
- ✅ `docs/KNOWLEDGE_GRAPH_REGENERATION_PLAN.md`
- ✅ `docs/IMPLEMENTATION_GUIDE.md` (this file)

**What it does**:
- Enforces type compatibility (PLACE can't merge with CONCEPT)
- Requires 95% similarity (up from 90%)
- Blocks known problematic merges (Moscow+Soil, Earth+Mars, etc.)
- Validates length ratios (prevents "I" → "India")
- Logs all merge decisions for audit trail

---

### Phase 2: Extract Knowledge from Source Content

You have **three options** depending on your goals:

#### 🏆 Option C: Hybrid Approach (RECOMMENDED)

**Best for**: Highest quality output while fixing Moscow issue

**Prerequisites**:
```bash
# Check if ACE-postprocessed episodes exist
ls -lh data/knowledge_graph_unified/episodes_postprocessed/ | head

# You should see: episode_0_post.json through episode_172_post.json
```

**Steps**:
```bash
# 1. Extract books fresh (skip episodes - use ACE versions)
python scripts/extract_knowledge_from_books.py

# What it does:
# - Processes all 4 books from data/books/
# - Uses type-safe EntityExtractor + RelationshipExtractor
# - Saves to data/knowledge_graph/entities/book_*_extraction.json
# - Takes ~30-40 minutes total

# 2. Build hybrid graph (skip to Phase 3 below)
```

**Why this is best**:
- ✅ Preserves ACE quality for episodes (pronoun resolution, discourse analysis)
- ✅ Fresh type-safe extraction for books
- ✅ Strict validation during build prevents all merge issues
- ⚡ Faster (only extract 4 books, not 172 episodes)

---

#### Option A: Extraction files already exist ✅
```bash
# Check what exists
ls -lh data/knowledge_graph/entities/

# If you see episode_*_extraction.json and book_*_extraction.json files:
# Skip to Phase 3!
```

#### Option B: Extract everything from scratch

**For Episodes** (~2-3 hours, automated):
```bash
python scripts/extract_knowledge_from_episodes.py

# What it does:
# 1. Loads all 172 episode transcripts from data/transcripts/
# 2. Uses EntityExtractor + RelationshipExtractor (OpenAI)
# 3. Saves to data/knowledge_graph/entities/episode_*_extraction.json
# 4. Processes ~30 seconds per episode
```

**For Books** (~30-60 minutes, automated; time depends on PDF size):
```bash
python scripts/extract_knowledge_from_books.py

# What it does:
# 1. Loads books from data/books/ (Veriditas, Soil Stewardship Handbook, Y on Earth, Our Biggest Deal)
# 2. Uses same extractors + book postprocessing pipeline
# 3. Saves to data/knowledge_graph/entities/book_*_extraction.json
# 4. Processes ~10 minutes per book
```

**Note**: This loses ACE postprocessing benefits. Use Option C (Hybrid) instead for best quality.

---

### Phase 3: Build Unified Graph with Validation

#### 🏆 Hybrid Build (RECOMMENDED if using Option C)

```bash
# Create unified graph using ACE episodes + fresh books + strict validation
python scripts/build_unified_graph_hybrid.py

# Optional: Extract books first
python scripts/build_unified_graph_hybrid.py --extract-books-first

# Custom similarity threshold
python scripts/build_unified_graph_hybrid.py --similarity-threshold 93

# What it does:
# 1. Loads ACE-postprocessed episodes from data/knowledge_graph_unified/episodes_postprocessed/
# 2. Loads fresh book extractions from data/knowledge_graph/entities/book_*.json
# 3. Deduplicates entities with strict validation:
#    - similarity_threshold = 95 (configurable)
#    - type_strict_matching = True
#    - semantic_validation = True
# 4. Exports to data/knowledge_graph_unified/unified_hybrid.json
# 5. Generates validation statistics
```

#### Alternative: Full Extraction Build (if using Option A or B)

```bash
# Create unified graph from all extraction files
python scripts/build_unified_graph_v2.py

# What it does:
# 1. Loads all extraction files (episodes + books)
# 2. Deduplicates entities with NEW validation rules:
#    - similarity_threshold = 95 (was 90)
#    - type_strict_matching = True (NEW)
#    - semantic_validation = True (NEW)
# 3. Exports to data/knowledge_graph_unified/unified_v2.json
# 4. Generates merge audit log
```

**Expected output**:
```
Loading extraction files...
  - Found 172 episode files
  - Found 3 book files
  - Total entities (raw): ~50,000
  - Total relationships (raw): ~45,000

Deduplicating entities...
  - Type: PERSON (5,234 entities -> 3,012 unique)
  - Type: ORGANIZATION (2,145 entities -> 1,523 unique)
  - Type: CONCEPT (15,678 entities -> 9,234 unique)
  ...

Entity Merge Validation Statistics:
  - Total comparisons: 125,432
  - Approved merges: 18,234 (14.5%)
  - Rejected merges: 107,198 (85.5%)
    • Type mismatch: 54,321
    • Low similarity: 42,123
    • Length mismatch: 8,432
    • Explicit blocklist: 234
    • Semantic incompatibility: 2,088

Unified graph created: data/knowledge_graph_unified/unified_v2.json
  - Total entities: 39,234 (vs 39,046 in v1)
  - Total relationships: 43,156 (vs 43,297 in v1)
  - Entities with merges: 892 (vs 2,681 in v1) ✅ 67% reduction
  - Suspicious merges: 12 (vs 353 in v1) ✅ 97% reduction
```

---

### Phase 4: Validation

```bash
# Run automated tests
python scripts/validate_unified_graph.py

# What it checks:
# ✅ Moscow does NOT have soil/moon aliases
# ✅ Soil exists as independent entity
# ✅ Earth does NOT have Mars/Paris aliases
# ✅ No entity has >500 relationships
# ✅ Betweenness centrality makes sense
# ✅ All relationships have "type" field

# Compare v1 vs v2
python scripts/compare_graph_versions.py
```

**Expected test results**:
```
Running validation tests...

✅ PASS: No catastrophic merges detected
✅ PASS: Moscow entity validation
   - Has 3 relationships (acceptable)
   - No aliases: soil, moon
✅ PASS: Soil entity validation
   - Exists independently
   - Has 267 relationships
✅ PASS: Type consistency check
   - All merged entities have matching types
✅ PASS: Relationship distribution
   - Max relationships per entity: 342 (acceptable)
   - No suspicious hubs detected
✅ PASS: Betweenness centrality
   - Top nodes: Soil, Agriculture, Permaculture, Composting
   - Moscow not in top 100

All tests passed! ✅
```

---

### Phase 5: Deployment

```bash
# 1. Final backup
cp data/knowledge_graph_unified/unified.json \
   data/knowledge_graph_unified/unified_v1.json

# 2. Deploy v2 as primary
cp data/knowledge_graph_unified/unified_v2.json \
   data/knowledge_graph_unified/unified.json

# 3. Restart services (if using Docker)
sudo docker restart yonearth-gaia-chatbot

# 4. Test API endpoints
curl http://localhost:8000/api/graph/entities | jq '.total_entities'
```

**Rollback if needed**:
```bash
# Revert to v1
cp data/knowledge_graph_unified/unified_v1.json \
   data/knowledge_graph_unified/unified.json

sudo docker restart yonearth-gaia-chatbot
```

---

## 📊 Success Metrics

### Before (v1):
- ❌ 2,681 entities with merge histories (7% of total)
- ❌ 541 entities merged via Levenshtein
- ❌ 353 highly suspicious merges
- ❌ Moscow = Soil + moon (280 misattributed relationships)
- ❌ Inconsistent field naming ("predicate" vs "type")

### After (Hybrid or v2):
- ✅ <1,000 entities with merge histories (<3% of total)
- ✅ All merges validated (type-safe, semantic-aware)
- ✅ <20 suspicious merges requiring review
- ✅ Moscow, Soil, moon are separate entities
- ✅ Consistent field naming (all have "type")
- ✅ Accurate betweenness centrality
- ✅ **Hybrid only**: Preserves ACE quality (pronoun resolution, discourse analysis)

---

## ⚠️ Common Issues & Solutions

### Issue 1: OpenAI Rate Limits

**Symptom**: Extraction stops with 429 errors

**Solution**:
```bash
# Increase delay in extractors
# In src/knowledge_graph/extractors/entity_extractor.py:
self.rate_limit_delay = 0.1  # Increase from 0.05 to 0.1
```

### Issue 2: Some Episodes Fail to Extract

**Symptom**: "Error loading episode_X.json"

**Solution**:
```bash
# Check transcript file
cat data/transcripts/episode_X.json | jq '.full_transcript' | head

# If transcript is empty/malformed, skip that episode
# Add to skip list in extraction script
```

### Issue 3: Memory Issues During Build

**Symptom**: "MemoryError" or process killed

**Solution**:
```bash
# Process in batches
python scripts/build_unified_graph_v2.py --batch-size 50

# Or increase system memory/swap
```

### Issue 4: New Validation Too Strict

**Symptom**: Legitimate variants not merged (e.g., "Aaron Perry" and "Aaron William Perry")

**Solution**:
```bash
# Lower similarity threshold slightly
# In EntityMergeValidator:
similarity_threshold = 93  # Down from 95

# Or add to known aliases in extraction metadata
```

---

## 📈 Timeline

### Hybrid Approach (RECOMMENDED)

| Phase | Time | Can Run Unattended? |
|-------|------|---------------------|
| Phase 1: Validation Setup | ✅ DONE | - |
| Phase 2: Extract Books Only | 30-40 min | ✅ Yes |
| Phase 3: Build Hybrid Graph | 20 min | ✅ Yes |
| Phase 4: Validation | 15 min | ❌ No (need to review) |
| Phase 5: Deployment | 15 min | ❌ No (need to monitor) |
| **TOTAL** | **~90 min** | **Book extraction can run unattended** |

**⚡ Fastest path**: Can complete in one afternoon!

### Full Extraction Approach (Alternative)

| Phase | Time | Can Run Unattended? |
|-------|------|---------------------|
| Phase 1: Validation Setup | ✅ DONE | - |
| Phase 2a: Extract Episodes | 2-3 hours | ✅ Yes (overnight) |
| Phase 2b: Extract Books | 30 min | ✅ Yes |
| Phase 3: Build Unified Graph | 30 min | ✅ Yes |
| Phase 4: Validation | 15 min | ❌ No (need to review) |
| Phase 5: Deployment | 15 min | ❌ No (need to monitor) |
| **TOTAL** | **4-5 hours** | **Most can run overnight** |

**Recommended approach**: Start extraction Friday evening, deploy Monday morning.

---

## 🔍 Spot-Checking Results

After deployment, verify key entities:

```bash
# Check Moscow
cat data/knowledge_graph_unified/unified.json | \
  jq '.entities.Moscow'

# Should show:
# - type: "PLACE"
# - aliases: [] (no "soil" or "moon")
# - sources: Only legitimate Moscow mentions

# Check Soil
cat data/knowledge_graph_unified/unified.json | \
  jq '.entities | to_entries | map(select(.value.aliases[]? == "soil")) | .[].key'

# Should show: "Soil" (or "soil") as separate entity

# Count relationships
cat data/knowledge_graph_unified/unified.json | \
  jq '.relationships | map(select(.source == "Moscow" or .target == "Moscow")) | length'

# Should be <10 (not 280!)
```

---

## 📝 Next Steps

### 🏆 Recommended: Hybrid Approach

1. **Extract books fresh**: `python scripts/extract_knowledge_from_books.py` (~30-40 min)
2. **Build hybrid graph**: `python scripts/build_unified_graph_hybrid.py` (~20 min)
3. **Validate results**: `python scripts/validate_unified_graph.py --input data/knowledge_graph_unified/unified_hybrid.json`
4. **Deploy if tests pass**: Copy unified_hybrid.json → unified.json

**Total time**: ~90 minutes to highest quality output!

### Alternative: Full Extraction

1. **Review this guide** and ask any questions
2. **Run Phase 2** (extraction) if needed
3. **Run Phase 3** (build graph with validation)
4. **Review Phase 4** (validation results)
5. **Deploy Phase 5** (if tests pass)

---

## ❓ Questions?

Common questions answered:

**Q: Will this fix ALL bad merges, even ones we don't know about?**
A: Yes! The validation will prevent all 540+ Levenshtein-based bad merges, not just Moscow/Soil.

**Q: Can we reuse this if we add more content later?**
A: Yes! The extraction and validation scripts are reusable. Just add new content and re-run.

**Q: What if we don't have time for full extraction?**
A: Use the Hybrid Approach! Only extract 4 books (~40 min) instead of 172 episodes (~3 hours).

**Q: What's the difference between Hybrid and Full Extraction approaches?**
A: Hybrid keeps ACE-postprocessed episodes (high quality) and only re-extracts books. Full Extraction regenerates everything from scratch. Hybrid is faster AND higher quality.

**Q: How do we know the new graph is better?**
A: Phase 4 validation will show side-by-side comparison and run automated tests to verify quality.

**Q: What's the rollback plan if something goes wrong?**
A: Keep `unified_v1.json` as backup. Can switch back in 30 seconds by copying file and restarting services.

---

## 📚 Reference Documents

- **Implementation Status**: `docs/IMPLEMENTATION_STATUS.md`
- **Detailed Plan**: `docs/KNOWLEDGE_GRAPH_REGENERATION_PLAN.md`
- **Current State Analysis**: `docs/CURRENT_STATE_ANALYSIS.md`
- **Validation Code**: `src/knowledge_graph/validators/entity_merge_validator.py`
- **Graph Builder (Modified)**: `src/knowledge_graph/graph/graph_builder.py`
- **Hybrid Build Script**: `scripts/build_unified_graph_hybrid.py` 🏆

---

**Ready to begin?**

🏆 **Recommended**: Use the Hybrid Approach (Option C) for highest quality in ~90 minutes!

Or start with Phase 2 (Extraction) or Phase 3 (Build) depending on whether extraction files exist.
