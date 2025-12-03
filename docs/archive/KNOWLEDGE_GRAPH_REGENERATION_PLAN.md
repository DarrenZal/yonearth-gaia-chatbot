> **STATUS: ✅ COMPLETED (December 2025)**
>
> This plan has been fully implemented through a more comprehensive approach than originally specified.
>
> ### Verification Results (2025-12-03):
>
> | Entity | Status | Details |
> |--------|--------|---------|
> | **Moscow** | ✅ CLEAN | Type: PLACE, Aliases: [] (NO soil/moon) |
> | **Soil** | ✅ CLEAN | Type: ECOSYSTEM, Aliases: ['soil'] (NO stove/skin/show) |
> | **Earth** | ✅ CLEAN | Type: ECOSYSTEM, Aliases: [] (NO mars/paris/farm) |
> | **DIA** | ✅ CLEAN | Type: PLACE, Aliases: ['Denver Airport', 'Denver International Airport'] (NO dubai/india) |
>
> ### Root Cause Fixed:
> - `src/knowledge_graph/validators/entity_merge_validator.py` implements:
>   - Explicit blocklist for problematic pairs (moscow/soil, earth/mars, etc.)
>   - Type strict matching (only merge same types)
>   - Higher threshold (95 vs original 90)
>   - Two-tier validation strategy
>   - Semantic compatibility checks
>
> ### Current Data Quality:
> - Max aliases per entity: 9 (Dr. Bronner's - legitimate)
> - No entity has >500 relationships
> - All catastrophic merges eliminated
>
> **This document is archived for historical reference.**

---

# Knowledge Graph Regeneration Plan (Option D)

## Executive Summary
Rebuild unified.json from scratch with strict entity merge validation to eliminate 540+ catastrophic entity merges (e.g., Moscow=Soil, Earth=Mars=Paris=farms).

**Timeline**: 3-4 hours human time + 1-2 days API processing
**Risk**: Low (backup existing data, gradual rollout)
**Impact**: Eliminates all hidden data quality issues + prevents future occurrences

---

## Problem Statement

### Current Issues in unified.json:
- **2,681 entities** (7%) have merge histories
- **541 entities** merged via overly aggressive Levenshtein distance (threshold=90)
- **353 highly suspicious merges** identified:
  - "DIA" = Dubai + Red + Sun + India + the Baca (26 entities!)
  - "the soil" = Mother Soil + the stove + the skin + the show
  - "Earth" = farms + earth + Mars + Paris + Farm (22 entities)
  - "Moscow" = Soil + moon + Moscow (280 relationships misattributed)

### Root Cause:
```python
# In graph_builder.py line 32:
self.similarity_threshold = 90  # Too aggressive!

# In deduplicate_entities() line 132:
if entity_name == other_name or fuzz.ratio(entity_name, other_name) >= self.similarity_threshold:
    # NO TYPE CHECKING - merges PLACE with CONCEPT!
```

---

## Phase 1: Implement Validation (2-3 hours)

### 1.1 Create Entity Merge Validator

**File**: `src/knowledge_graph/validators/entity_merge_validator.py`

**Purpose**: Semantic validation before merging entities

**Key Features**:
- Type compatibility checking (PLACE can't merge with CONCEPT)
- Semantic similarity validation (not just Levenshtein)
- Length ratio checking (prevent "I" from merging with "India")
- Merge logging for audit trail
- Blocklist for known problematic terms

### 1.2 Update GraphBuilder

**File**: `src/knowledge_graph/graph/graph_builder.py`

**Changes**:
```python
# Line 32: Increase threshold
self.similarity_threshold = 95  # Was 90

# Add new parameters:
self.type_strict_matching = True  # NEW: Enforce type compatibility
self.min_length_ratio = 0.6  # NEW: Prevent "I" -> "India" merges
self.semantic_validation = True  # NEW: Check if merge makes sense

# Blocklist for terms that should NEVER merge
self.merge_blocklist = [
    ('moscow', 'soil'), ('moscow', 'moon'),
    ('earth', 'mars'), ('earth', 'paris'),
    ('leaders', 'healers'), ('organization', 'urbanization')
]
```

**Method Updates**:
- `deduplicate_entities()`: Add type checking before merging
- `_can_merge_entities()`: NEW method with validation logic
- `_log_merge_decision()`: NEW method for audit trail

### 1.3 Create Merge Review Script

**File**: `scripts/review_entity_merges.py`

**Purpose**: Human-in-the-loop verification of edge cases

**Features**:
- Lists all proposed merges with similarity scores
- Flags suspicious merges for review
- Interactive approval/rejection
- Saves approved merge list for reproducibility

---

## Phase 2: Extract from Source Content (1-2 days, mostly automated)

### 2.1 Check Current Extraction State

**Action**: Verify if raw extraction files exist

```bash
# Check for existing extractions
ls -lh data/knowledge_graph/entities/ 2>/dev/null
ls -lh data/knowledge_graph/relationships/ 2>/dev/null

# If empty, proceed with full extraction
# If files exist, can skip to Phase 3
```

### 2.2 Extract from Episodes

**Script**: `scripts/extract_knowledge_from_episodes.py`

**Process**:
1. Load episode transcripts from `data/transcripts/episode_*.json`
2. For each episode:
   - Chunk transcript (800 tokens, 100 overlap)
   - Extract entities using `EntityExtractor` (OpenAI gpt-4o-mini)
   - Extract relationships using `RelationshipExtractor`
   - Save to `data/knowledge_graph/entities/episode_N_extraction.json`
3. Estimated time: ~30-60 seconds per episode × 172 = 2-3 hours

**Output**:
```json
{
  "episode_number": 122,
  "entities": [
    {
      "name": "Biochar",
      "type": "CONCEPT",
      "description": "Carbon-rich material produced by pyrolysis...",
      "metadata": {"episode_number": 122, "chunk_id": "chunk_5"}
    }
  ],
  "relationships": [
    {
      "source_entity": "Biochar",
      "relationship_type": "ENHANCES",
      "target_entity": "Soil",
      "description": "Biochar improves soil fertility...",
      "metadata": {"episode_number": 122, "chunk_id": "chunk_5"}
    }
  ]
}
```

### 2.3 Extract from Books

**Script**: `scripts/extract_knowledge_from_books.py`

**Process**:
1. Load book content from `data/books/*/`
2. For each book:
   - Use same EntityExtractor and RelationshipExtractor
   - Apply book-specific postprocessing pipeline (v14.3.8)
   - Save to `data/knowledge_graph/entities/book_NAME_extraction.json`
3. Estimated time: ~5-10 minutes per book × 3 = 15-30 minutes

**Books to Process**:
- Y on Earth (y-on-earth)
- VIRIDITAS: THE GREAT HEALING (veriditas)
- Soil Stewardship Handbook (soil-stewardship-handbook)

---

## Phase 3: Build Unified Graph (30 minutes)

### 3.1 Backup Current Data

```bash
# Backup existing unified.json
cp data/knowledge_graph_unified/unified.json \
   data/knowledge_graph_unified/unified_v1_backup_$(date +%Y%m%d).json

# Backup visualization data
cp data/knowledge_graph/visualization_data.json \
   data/knowledge_graph/visualization_data_v1_backup.json
```

### 3.2 Run Enhanced Graph Builder

**Script**: `scripts/build_unified_graph_v2.py`

**Process**:
```python
from src.knowledge_graph.graph.graph_builder import GraphBuilder
from src.knowledge_graph.validators.entity_merge_validator import EntityMergeValidator

# Initialize with strict parameters
builder = GraphBuilder(
    extraction_dir="data/knowledge_graph/entities",
    similarity_threshold=95,  # Stricter (was 90)
    type_strict_matching=True,  # NEW
    semantic_validation=True  # NEW
)

# Load extractions (episodes + books)
builder.load_extractions()

# Deduplicate with validation
builder.deduplicate_entities()
builder.deduplicate_relationships()

# Export to unified.json v2
builder.export_unified_json("data/knowledge_graph_unified/unified_v2.json")
```

### 3.3 Field Normalization

**Ensure consistency**:
- All relationships have both `type` and `predicate` fields
- Map OpenAI relationship_type to predicate names
- Add source provenance metadata

---

## Phase 4: Validation & Testing (1 hour)

### 4.1 Automated Validation

**Script**: `scripts/validate_unified_graph.py`

**Tests**:

```python
# Test 1: No catastrophic merges
def test_no_catastrophic_merges():
    entities = load_entities()

    # Moscow should NOT have soil/moon aliases
    moscow = entities.get("Moscow")
    assert "soil" not in [a.lower() for a in moscow.get("aliases", [])]
    assert "moon" not in [a.lower() for a in moscow.get("aliases", [])]

    # Soil should exist as separate entity
    assert "Soil" in entities or "soil" in entities

    # Earth should NOT have Mars/Paris aliases
    earth = entities.get("Earth")
    if earth:
        aliases_lower = [a.lower() for a in earth.get("aliases", [])]
        assert "mars" not in aliases_lower
        assert "paris" not in aliases_lower

# Test 2: Type consistency
def test_type_consistency():
    entities = load_entities()

    for name, entity in entities.items():
        provenance = entity.get("provenance", [])
        if not provenance:
            continue

        # All merged entities should have same type
        for merge_record in provenance:
            merged_from = merge_record["merged_from"]
            # Look up original type (would need extraction files)
            # Assert types match

# Test 3: Relationship counts make sense
def test_relationship_distribution():
    rels = load_relationships()

    # Count relationships per entity
    rel_counts = defaultdict(int)
    for rel in rels:
        rel_counts[rel["source"]] += 1
        rel_counts[rel["target"]] += 1

    # No entity should have >500 relationships (indicates bad merge)
    max_rels = max(rel_counts.values())
    assert max_rels < 500, f"Entity with {max_rels} rels suggests bad merge"

    # Soil should have 200-300 relationships
    soil_rels = rel_counts.get("Soil", 0) + rel_counts.get("soil", 0)
    assert 200 <= soil_rels <= 300, f"Soil has {soil_rels} rels (expected 200-300)"

# Test 4: Betweenness centrality
def test_betweenness_centrality():
    graph = build_networkx_graph()
    centrality = nx.betweenness_centrality(graph)

    # Top 20 nodes should be semantically important concepts
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]

    # Moscow should NOT be in top 20
    top_names = [name for name, _ in top_nodes]
    assert "Moscow" not in top_names

    # Soil/Agriculture/Regeneration should be high
    important_concepts = ["Soil", "Agriculture", "Regeneration", "Permaculture"]
    assert any(concept in top_names for concept in important_concepts)
```

### 4.2 Comparison with v1

**Script**: `scripts/compare_graph_versions.py`

**Metrics**:
```python
# Compare statistics
v1_stats = {
    "total_entities": 39046,
    "total_relationships": 43297,
    "entities_with_merges": 2681,
    "suspicious_merges": 353
}

v2_stats = analyze_graph("unified_v2.json")

print(f"Entity count: {v1_stats['total_entities']} -> {v2_stats['total_entities']}")
print(f"Merge count: {v1_stats['entities_with_merges']} -> {v2_stats['entities_with_merges']}")
print(f"Suspicious: {v1_stats['suspicious_merges']} -> {v2_stats['suspicious_merges']}")

# Expected outcomes:
# - Total entities: Similar (39k-40k)
# - Entities with merges: Much lower (500-1000 instead of 2681)
# - Suspicious merges: Near zero (0-10 instead of 353)
```

### 4.3 Spot Check Examples

```python
# Check known problem entities
test_entities = [
    ("Moscow", "PLACE", ["soil", "moon"]),  # Should NOT have these aliases
    ("Soil", "CONCEPT", []),  # Should exist independently
    ("Earth", "PLACE", ["mars", "paris"]),  # Should NOT have these
    ("DIA", "PLACE", ["dubai", "red", "sun"]),  # Should NOT have these
]

for name, expected_type, forbidden_aliases in test_entities:
    entity = get_entity(name)
    assert entity["type"] == expected_type

    actual_aliases = [a.lower() for a in entity.get("aliases", [])]
    for forbidden in forbidden_aliases:
        assert forbidden not in actual_aliases, \
            f"{name} should NOT have alias '{forbidden}'"
```

---

## Phase 5: Deployment (30 minutes)

### 5.1 Gradual Rollout

```bash
# Step 1: Keep v1 as fallback
mv data/knowledge_graph_unified/unified.json \
   data/knowledge_graph_unified/unified_v1.json

# Step 2: Deploy v2 as primary
cp data/knowledge_graph_unified/unified_v2.json \
   data/knowledge_graph_unified/unified.json

# Step 3: Update API endpoints to use new unified.json
# (No code changes needed - same file path)
```

### 5.2 Monitor for Issues

**Watch for**:
- API errors from changed entity names
- Frontend visualization issues
- Missing entities that existed in v1

**Rollback plan**:
```bash
# If issues found, revert immediately
cp data/knowledge_graph_unified/unified_v1.json \
   data/knowledge_graph_unified/unified.json

# Restart services
sudo docker restart yonearth-gaia-chatbot
```

---

## File Structure

```
yonearth-gaia-chatbot/
├── src/knowledge_graph/
│   ├── validators/
│   │   └── entity_merge_validator.py  # NEW: Validation logic
│   ├── graph/
│   │   └── graph_builder.py  # MODIFIED: Add validation
│   └── extractors/
│       ├── entity_extractor.py  # EXISTING: Use as-is
│       └── relationship_extractor.py  # EXISTING: Use as-is
├── scripts/
│   ├── extract_knowledge_from_episodes.py  # NEW
│   ├── extract_knowledge_from_books.py  # NEW
│   ├── build_unified_graph_v2.py  # NEW
│   ├── validate_unified_graph.py  # NEW
│   ├── compare_graph_versions.py  # NEW
│   └── review_entity_merges.py  # NEW: Optional manual review
├── data/knowledge_graph/
│   ├── entities/  # NEW: Raw extraction files
│   │   ├── episode_0_extraction.json
│   │   ├── episode_1_extraction.json
│   │   ├── ...
│   │   ├── book_y-on-earth_extraction.json
│   │   └── book_veriditas_extraction.json
│   └── relationships/  # OPTIONAL: Separate rel files
└── data/knowledge_graph_unified/
    ├── unified_v1_backup_20251120.json  # Backup
    ├── unified_v1.json  # Fallback
    ├── unified_v2.json  # New version
    └── unified.json  # Active (points to v2)
```

---

## Success Criteria

### Must Have:
- ✅ Moscow entity does NOT have "soil" or "moon" aliases
- ✅ Soil entity exists independently with 200-300 relationships
- ✅ Zero catastrophic merges (entity type mismatches)
- ✅ All relationships have "type" field (not just "predicate")
- ✅ Betweenness centrality makes semantic sense

### Nice to Have:
- ✅ <100 entities with merge histories (down from 2,681)
- ✅ <10 suspicious merges requiring review (down from 353)
- ✅ Audit log of all merge decisions
- ✅ Reproducible extraction process

---

## Timeline Estimate

| Phase | Human Time | API Time | Total |
|-------|-----------|----------|-------|
| **Phase 1: Validation Logic** | 2-3 hours | - | 2-3 hours |
| **Phase 2: Extraction** | 30 min setup | 2-3 hours | 3-4 hours |
| **Phase 3: Build Graph** | 15 min | 15 min | 30 min |
| **Phase 4: Validation** | 1 hour | - | 1 hour |
| **Phase 5: Deployment** | 30 min | - | 30 min |
| **TOTAL** | **4-5 hours** | **2-3 hours** | **7-8 hours** |

**Note**: API time is parallelizable (extraction runs unattended overnight)

---

## Risk Mitigation

### Risk 1: New Validation Too Strict
**Impact**: Misses legitimate entity variations
**Mitigation**: Run comparison report showing entities that weren't merged in v2 but were in v1
**Action**: Manual review of <50 edge cases

### Risk 2: API Rate Limits
**Impact**: Extraction takes longer than estimated
**Mitigation**: Use existing rate limiting (0.05s delay between calls = 1,200/min)
**Action**: Run overnight, expect 8-12 hours instead of 2-3

### Risk 3: Missing Entities in v2
**Impact**: Frontend shows "Entity not found" errors
**Mitigation**: Keep v1 as fallback, add alias mapping for renamed entities
**Action**: Create entity name mapping file for backwards compatibility

### Risk 4: Extraction Errors
**Impact**: Some episodes/books fail to process
**Mitigation**: Robust error handling with retry logic (3 attempts)
**Action**: Manual review of failed extractions, reprocess individually

---

## Next Steps

1. **Get approval** for this plan
2. **Create Phase 1 files** (validation logic)
3. **Run small test** on 5 episodes to validate approach
4. **Full extraction** overnight
5. **Deploy and monitor**

---

## Questions?

- Should we also rebuild the Neo4j graph database? (Currently not used in production)
- Should we create visualization comparisons (v1 vs v2 betweenness centrality)?
- Do you want interactive merge review, or trust automated validation?
- Should we preserve v1 for historical analysis?
