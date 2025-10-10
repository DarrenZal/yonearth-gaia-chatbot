# 🔍 Multi-Source Type Checking Strategy (Ultrathought Analysis)

## The Core Question: What's the Best Type Source?

You're right to question Wikidata. Let's analyze what we actually need and what sources can provide it.

## Coverage Analysis of Our Entities

Looking at our 21,336 entities, they fall into categories:

### Category 1: Well-Known Public Entities (~15%)
**Examples**: Boulder, biochar, compost, USDA, Patagonia
- **Wikidata Coverage**: ✅ Excellent
- **DBpedia Coverage**: ✅ Good
- **ConceptNet Coverage**: ✅ Good

### Category 2: Podcast-Specific Entities (~40%)
**Examples**: Nancy Tuckman, Aaron William Perry, Earth Coast Productions, specific episodes
- **Wikidata Coverage**: ❌ None (not notable enough)
- **DBpedia Coverage**: ❌ None
- **ConceptNet Coverage**: ⚠️ Partial (generic concepts only)

### Category 3: Domain-Specific Technical Terms (~30%)
**Examples**: regenerative agriculture, biochar pyrolysis, soil mycorrhizae, permaculture
- **Wikidata Coverage**: ⚠️ Partial (main terms, not variations)
- **DBpedia Coverage**: ⚠️ Partial
- **ConceptNet Coverage**: ⚠️ Partial

### Category 4: Context-Specific Phrases (~15%)
**Examples**: "quantum and sub-quantum realms", "winers community", "sustainable settings"
- **Wikidata Coverage**: ❌ None
- **DBpedia Coverage**: ❌ None
- **ConceptNet Coverage**: ❌ None

## The Hybrid Strategy (Best Approach)

### Tier 1: External Knowledge Bases (for well-known entities)
**Use for**: Public entities, established concepts, geographic places

#### 1.1 Wikidata (Primary for Structure)
**Strengths**:
- Structured types (instance_of, subclass_of)
- Well-defined type hierarchy
- Good coverage of places, organizations, materials
- Free API with SPARQL

**Weaknesses**:
- Missing niche entities
- Requires entity linking (name → QID)
- API rate limits

**Use Case**: Geographic validation, material types, established organizations

```python
# Example query:
entity = "biochar"
wikidata_result = {
    'qid': 'Q905495',
    'instance_of': ['Q2207288'],  # soil conditioner
    'subclass_of': ['Q177463'],   # charcoal
    'type_hierarchy': ['substance', 'material', 'product']
}
```

#### 1.2 GeoNames (Primary for Geographic)
**Strengths**:
- Authoritative geographic data
- Population, area, coordinates
- Administrative hierarchy
- Free API

**Weaknesses**:
- Only geographic entities
- Name variations can be tricky

**Use Case**: All geographic validation (Boulder/Lafayette)

```python
geonames_result = {
    'name': 'Boulder',
    'feature_code': 'PPL',  # populated place
    'admin1': 'Colorado',
    'population': 108000,
    'coordinates': (40.01, -105.27)
}
```

#### 1.3 ConceptNet (Supplementary)
**Strengths**:
- Broad concept coverage
- Relationship types (IsA, PartOf)
- Multi-language
- Free

**Weaknesses**:
- Less structured than Wikidata
- Crowdsourced (variable quality)

**Use Case**: Fallback for concepts not in Wikidata

### Tier 2: LLM-Based Type Inference (for domain-specific)
**Use for**: Technical terms, domain concepts not in KBs

```python
def infer_type_with_llm(entity_name, context):
    """Use GPT-4 to infer entity type"""
    prompt = f"""
    Given the entity "{entity_name}" in this context:
    {context}

    What type of entity is this? Choose from:
    - PERSON
    - ORGANIZATION
    - PLACE (geographic location)
    - PRODUCT (physical item)
    - CONCEPT (abstract idea, principle, process)
    - EVENT
    - PRACTICE (method, technique, approach)

    If it's a PLACE, is it a:
    - COUNTRY
    - STATE/PROVINCE
    - CITY
    - REGION
    - FACILITY (building, farm, institution)

    Return: {{"primary_type": "...", "subtype": "...", "confidence": 0.9}}
    """

    # Use structured output for consistency
    return llm.structured_completion(prompt, TypeSchema)
```

**Strengths**:
- Handles domain-specific terms
- Uses context from transcript
- Can reason about novel entities

**Weaknesses**:
- Costs API calls
- Not deterministic
- Needs validation

### Tier 3: Local Ontology Learning (from our data)
**Use for**: Podcast-specific entities, learned patterns

```python
class LocalOntology:
    """Learn types from our own extractions"""

    def __init__(self):
        self.type_cache = {}
        self.patterns = {}

    def learn_from_extractions(self, entities):
        """Build type knowledge from extraction patterns"""

        # If entity appears with "lives_in", "works_at" → likely PERSON
        # If entity appears with "located_in", "contains" → likely PLACE/ORG
        # If entity appears with "produces", "made_from" → likely PRODUCT/MATERIAL

        for entity in entities:
            self.infer_type_from_relationships(entity)

    def infer_type_from_relationships(self, entity):
        """Pattern-based type inference"""

        rels_as_subject = get_relationships(subject=entity.name)
        rels_as_object = get_relationships(object=entity.name)

        # PERSON indicators
        if any(r in ['lives_in', 'works_at', 'founded', 'born_in'] for r in rels_as_subject):
            return 'PERSON'

        # ORGANIZATION indicators
        if any(r in ['employs', 'produces', 'sells'] for r in rels_as_subject):
            return 'ORGANIZATION'

        # PLACE indicators
        if any(r in ['located_in', 'part_of'] for r in rels_as_object):
            return 'PLACE'

        # CONCEPT indicators (appears as object of abstract verbs)
        if any(r in ['advocates_for', 'practices', 'believes_in'] for r in rels_as_object):
            return 'CONCEPT'
```

### Tier 4: Explicit Type Annotations (highest confidence)
**Use for**: Manual corrections, expert input

```python
# Manually curated type mappings
EXPERT_TYPE_ANNOTATIONS = {
    'Aaron William Perry': 'PERSON',
    'YonEarth': 'ORGANIZATION',
    'regenerative agriculture': 'PRACTICE',
    'biochar': 'PRODUCT',
    'permaculture': 'PRACTICE',
    # High-confidence human annotations
}
```

## The Cascading Resolution Strategy

```python
class HybridTypeResolver:
    """Multi-source type resolution with fallback"""

    def __init__(self):
        self.expert_types = load_expert_annotations()
        self.local_ontology = LocalOntology()
        self.wikidata_cache = {}
        self.geonames_cache = {}

    def resolve_type(self, entity_name, context=None):
        """Resolve entity type using cascading strategy"""

        result = {
            'entity': entity_name,
            'type': None,
            'confidence': 0.0,
            'source': None,
            'details': {}
        }

        # Tier 4: Expert annotations (highest confidence)
        if entity_name in self.expert_types:
            result['type'] = self.expert_types[entity_name]
            result['confidence'] = 1.0
            result['source'] = 'expert_annotation'
            return result

        # Tier 1a: Geographic entities (GeoNames)
        geonames_result = self.check_geonames(entity_name)
        if geonames_result:
            result['type'] = 'PLACE'
            result['subtype'] = geonames_result['feature_code']
            result['confidence'] = 0.95
            result['source'] = 'geonames'
            result['details'] = geonames_result
            return result

        # Tier 1b: Wikidata (well-known entities)
        wikidata_result = self.check_wikidata(entity_name)
        if wikidata_result:
            result['type'] = self.map_wikidata_to_our_types(wikidata_result)
            result['confidence'] = 0.90
            result['source'] = 'wikidata'
            result['details'] = wikidata_result
            return result

        # Tier 3: Local ontology (learned from our data)
        local_type = self.local_ontology.infer_type(entity_name)
        if local_type:
            result['type'] = local_type['type']
            result['confidence'] = 0.75
            result['source'] = 'local_ontology'
            result['details'] = local_type
            return result

        # Tier 2: LLM inference (most flexible, least confident)
        if context:
            llm_result = self.infer_with_llm(entity_name, context)
            if llm_result:
                result['type'] = llm_result['type']
                result['confidence'] = 0.65  # Lower confidence
                result['source'] = 'llm_inference'
                result['details'] = llm_result
                return result

        # Fallback: Unknown
        result['type'] = 'UNKNOWN'
        result['confidence'] = 0.0
        result['source'] = 'none'
        return result

    def check_geonames(self, entity_name):
        """Check if entity exists in GeoNames"""
        if entity_name in self.geonames_cache:
            return self.geonames_cache[entity_name]

        # API call to GeoNames
        # Cache result
        return None

    def check_wikidata(self, entity_name):
        """Check if entity exists in Wikidata"""
        if entity_name in self.wikidata_cache:
            return self.wikidata_cache[entity_name]

        # Entity linking: name → QID
        # SPARQL query for types
        # Cache result
        return None

    def map_wikidata_to_our_types(self, wikidata_result):
        """Map Wikidata type hierarchy to our simplified types"""

        wikidata_classes = wikidata_result.get('instance_of', [])

        # Map to our type system
        type_mapping = {
            'Q5': 'PERSON',           # human
            'Q43229': 'ORGANIZATION',  # organization
            'Q486972': 'PLACE',       # human settlement
            'Q82794': 'PLACE',        # geographic location
            'Q2424752': 'PRODUCT',    # product
            'Q151885': 'CONCEPT',     # concept
            'Q11862829': 'PRACTICE',  # academic discipline
            # ... more mappings
        }

        for qid in wikidata_classes:
            if qid in type_mapping:
                return type_mapping[qid]

        return 'UNKNOWN'
```

## Coverage Estimates

Based on our entity categories:

| Source | Coverage | Confidence | Cost |
|--------|----------|------------|------|
| **Expert Annotations** | 1-2% | 1.0 | Manual effort |
| **GeoNames** | 10-15% | 0.95 | Free API |
| **Wikidata** | 15-20% | 0.90 | Free API |
| **Local Ontology** | 50-60% | 0.75 | Computed |
| **LLM Inference** | 80-90% | 0.65 | $0.001/entity |
| **Total Coverage** | 95%+ | 0.70 avg | Low |

## Practical Implementation

### Phase 1: Build Core Components (Week 1)
```python
# 1. GeoNames integration
geonames_checker = GeoNamesChecker(cache_file='geonames_cache.json')

# 2. Wikidata integration
wikidata_checker = WikidataChecker(cache_file='wikidata_cache.json')

# 3. Local ontology
local_ontology = LocalOntology()
local_ontology.learn_from_extractions(all_entities)

# 4. LLM fallback
llm_typer = LLMTypeInference(model='gpt-4o-mini')
```

### Phase 2: Build Hybrid Resolver (Week 1)
```python
resolver = HybridTypeResolver(
    geonames=geonames_checker,
    wikidata=wikidata_checker,
    local=local_ontology,
    llm=llm_typer
)

# Resolve all entities
for entity in all_entities:
    type_info = resolver.resolve_type(entity.name, context=entity.context)
    entity.resolved_type = type_info
```

### Phase 3: Apply Constraints (Week 1)
```python
# Now we can validate!
for rel in all_relationships:
    source_type = resolver.resolve_type(rel.source)
    target_type = resolver.resolve_type(rel.target)

    # Geographic constraint
    if rel.relationship in ['located_in', 'part_of', 'contains']:
        if target_type['type'] != 'PLACE':
            flag_error(rel, f"Geographic relationship requires PLACE target, got {target_type['type']}")
```

## Cost Analysis

### API Costs
- **GeoNames**: Free (rate limited to 1000/hour)
- **Wikidata**: Free (rate limited)
- **LLM inference**: $0.001 per entity × 21,336 = ~$21 one-time

### Optimization
- Cache all external lookups locally
- Only call LLM for unknowns after other sources
- Batch LLM calls for efficiency

### Expected Coverage with Caching
After first pass:
- 95% of entities cached
- Future validations: < $1
- Real-time validation: instant (cache hit)

## Recommended Strategy

**For your YonEarth knowledge graph:**

1. **Start with GeoNames** (free, authoritative, solves Boulder/Lafayette)
2. **Add Wikidata** for materials/organizations (free, solves biochar)
3. **Build local ontology** from extraction patterns (free, handles podcast-specific)
4. **LLM as fallback** for remaining unknowns (low cost, high coverage)
5. **Manual annotations** for important recurring entities (one-time, highest confidence)

**Total cost**: ~$25 one-time, then cached forever

## Answer to Your Question

> "Do you think Wikidata is the best source?"

**Answer**: Wikidata is ONE good source, but not sufficient alone!

**Best strategy**: Hybrid approach
- **GeoNames for geographic** (your Boulder/Lafayette errors)
- **Wikidata for well-known entities** (your biochar errors)
- **Local ontology for domain-specific** (your podcast entities)
- **LLM for gaps** (everything else)

This gives you 95%+ coverage with high confidence on the important stuff (geographic, materials) and reasonable confidence on the rest.

The key insight: **Use the right source for each entity type, not one-size-fits-all.**