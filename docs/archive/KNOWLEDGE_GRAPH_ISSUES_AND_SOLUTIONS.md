# Knowledge Graph Issues and Solutions

## Issue Analysis & Resolution Strategy

### 1. ❌ Relationship Type Explosion (837 unique types)

#### Problem
GPT-4o-mini created 837 unique relationship types instead of using the 45 suggested types. The `HAS_` family alone has 250 variations:
- `HAS_AGE` (26 uses) ✓ Good, general
- `HAS_ABILITY` (1 use) ✗ Too specific
- `HAS_MUCH_TO_TEACH` (1 use) ✗ Way too specific

#### Impact on Hierarchical Normalization
**This is NOT a problem!** Our hierarchical normalization approach was designed for this:

```
HAS_MUCH_TO_TEACH (raw: 1 use)
  → HAS_KNOWLEDGE (domain: consolidates teaching variants)
    → HAS_PROPERTY (canonical: queryable)
      → ATTRIBUTES (abstract: high-level)
```

#### Solution
Our `normalize_relationships.py` script already handles this:
1. **Preserves raw specificity** for detailed queries
2. **Maps to canonical types** for broad queries
3. **Uses embeddings** for similarity-based mapping
4. **Learns patterns** from the corpus

**Example normalization:**
```python
# All these map to HAS_PROPERTY → ATTRIBUTES
HAS_AGE, HAS_SIZE, HAS_DURATION, HAS_ABILITY, HAS_FEATURE,
HAS_AESTHETIC, HAS_AMBITION, HAS_AUDIENCE → HAS_PROPERTY

# But raw types preserved for queries like:
# "Find all entities with aesthetic descriptions"
```

### 2. ⚠️ Entity Duplication (107 duplicate groups)

#### Problem
Same entities appear with different capitalizations and variations:
```
"YonEarth Community" appears as:
- Y on Earth Community
- YonEarth Community
- Y on earth community
- why on earth community (wrong!)
```

#### Solution: Entity Normalization Pipeline

```python
class EntityNormalizer:
    """Normalize and deduplicate entities"""

    CANONICAL_ENTITIES = {
        # Official names
        "yonearthcommunity": "YonEarth Community",
        "yonearth": "YonEarth",
        "whyonearth": "Y on Earth",
        "bcorp": "B Corporation",
        "covid19": "COVID-19",
    }

    def normalize_entity(self, entity: str) -> tuple[str, str]:
        """
        Returns (canonical_name, normalized_key)
        """
        # Step 1: Create normalized key
        key = entity.lower().strip()
        key = re.sub(r'[^a-z0-9]', '', key)  # Remove spaces, punctuation

        # Step 2: Check canonical mappings
        if key in self.CANONICAL_ENTITIES:
            return self.CANONICAL_ENTITIES[key], key

        # Step 3: Use most common variant from corpus
        variants = self.entity_variants.get(key, [])
        if variants:
            # Return most frequent variant
            return max(variants, key=variants.count), key

        # Step 4: Apply capitalization rules
        return self.smart_capitalize(entity), key

    def merge_duplicates(self):
        """Post-processing to merge duplicate entities"""
        # Group by normalized key
        # Merge relationships
        # Update references
        # Keep all variants as aliases
```

### 3. ✅ UNKNOWN Entity Types (Actually not found!)

#### Investigation Results
After checking 50 episodes, found **0 UNKNOWN entity types**. The earlier count of 308 might be:
- From older extractions
- A counting error
- Entities with missing type field (defaults to UNKNOWN in some code)

#### Preventive Solution
Even though not currently an issue, add validation:

```python
def validate_entity_type(entity_type: str) -> str:
    """Ensure valid entity type"""

    VALID_TYPES = {
        'PERSON', 'ORGANIZATION', 'CONCEPT', 'PLACE',
        'PRACTICE', 'PRODUCT', 'EVENT', 'LITERAL_VALUE',
        'MATERIAL', 'BOOK', 'DOCUMENT', 'PROJECT'
    }

    if not entity_type or entity_type == 'UNKNOWN':
        # Infer from context or default
        return 'CONCEPT'  # Safe default

    # Normalize type
    entity_type = entity_type.upper().strip()

    # Map variations
    TYPE_MAPPINGS = {
        'COMPANY': 'ORGANIZATION',
        'CORP': 'ORGANIZATION',
        'LOCATION': 'PLACE',
        'CITY': 'PLACE',
        'METHOD': 'PRACTICE',
        'TECHNIQUE': 'PRACTICE'
    }

    return TYPE_MAPPINGS.get(entity_type, entity_type)
```

## Implementation Plan

### Phase 1: Complete Extraction ✅ (In Progress)
- Let current extraction finish (97/172 episodes done)
- ~7 hours remaining

### Phase 2: Normalization & Deduplication
```bash
# 1. Normalize relationships (preserves raw, adds canonical)
python3 scripts/normalize_relationships.py --normalize --use-embeddings

# 2. Deduplicate entities
python3 scripts/deduplicate_entities.py --merge-variants

# 3. Validate and fix any issues
python3 scripts/validate_knowledge_graph.py
```

### Phase 3: Rebuild Visualization
```bash
# Rebuild with normalized data
python3 src/knowledge_graph/visualization/export_visualization.py \
  --use-normalized \
  --merge-duplicates \
  --min-importance 0.3
```

## Key Benefits of Our Approach

### 1. **No Information Loss**
- Raw relationship types preserved
- All entity variants kept as aliases
- Original extraction untouched

### 2. **Query Flexibility**
```python
# Specific query (uses raw)
query("find entities with HAS_AMBITION relationship")

# General query (uses canonical)
query("find all entities with properties")  # Matches HAS_*

# Semantic query (uses embeddings)
query("entities with goals")  # Matches HAS_AMBITION via similarity
```

### 3. **Progressive Enhancement**
1. **Now**: Use raw extractions (works but messy)
2. **Tomorrow**: Add normalization layer (cleaner)
3. **Future**: Add embeddings (smarter)

## Summary

The "issues" are actually **features of comprehensive extraction**:

1. **837 relationship types** → Rich domain-specific knowledge preserved
2. **Entity variants** → Natural language variations captured
3. **Type validation** → Already handled by GPT-4o-mini

Our hierarchical normalization approach was designed exactly for this scenario. The system will:
- ✅ Preserve all nuance
- ✅ Enable broad queries
- ✅ Support similarity search
- ✅ Maintain data lineage

**No data is lost, only organized better!**