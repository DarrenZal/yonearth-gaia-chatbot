# Knowledge Graph Architecture: Hybrid Semantic Intelligence System

## Executive Summary

This document describes the architecture of YonEarth's Knowledge Graph system, which combines **text embeddings** and **graph embeddings** to create a powerful semantic intelligence tool that goes beyond visualization to enable complex querying, reasoning, and discovery.

## Core Design Principles

### 1. Hierarchical Semantic Ontology
Relationships are organized in a 4-level hierarchy, with the **Domain layer as the primary semantic intelligence layer**:

- **Raw** (837+ types): Exact relationships as extracted by GPT-4o-mini
  - Examples: `LIGHTED_FIRE_IN`, `SEQUESTERS_CARBON_IN`, `HAS_PROFOUND_IMPACT_ON`
  - Purpose: Preserves all nuance and specificity from extraction

- **Domain** (~150 semantic types): **The Semantic Intelligence Layer**
  - Examples: `MENTORS`, `FUNDS`, `HAS_EXPERTISE`, `TRANSFORMS_INTO`, `ADVOCATES_FOR`
  - Purpose: **Primary layer for natural language query conversion**
  - Rich enough for sophisticated graph queries while maintaining semantic meaning
  - Each type has properties like `implies_seniority`, `financial`, `causal`, etc.

- **Canonical** (45 types): Standard queryable relationships
  - Examples: `EDUCATES`, `FINANCES`, `PRACTICES`, `CREATES`
  - Purpose: Broad queries across semantic families

- **Abstract** (10-15 types): High-level relationship categories
  - Examples: `INFLUENCES`, `EXCHANGES`, `TRANSFORMS`, `ASSOCIATES`
  - Purpose: Most general queries and graph analysis

Example hierarchy:
```
LIGHTED_FIRE_IN (raw: preserves exact phrasing)
  → MENTORS (domain: semantic intelligence - implies personal guidance)
    → EDUCATES (canonical: broader education category)
      → INFLUENCES (abstract: general influence)
```

### 2. Semantic Domain Ontology
The Domain layer is not just a consolidation step but the **semantic brain** of the system:

```python
# Example from our domain ontology
"MENTORS": {
    "raw_patterns": ["MENTOR", "GUIDE", "COACH", "ADVISE", "LIGHTED_FIRE"],
    "canonical": "EDUCATES",
    "abstract": "INFLUENCES",
    "properties": {
        "implies_seniority": True,
        "personal_guidance": True,
        "knowledge_transfer": True,
        "directional": True
    }
},
"FUNDS": {
    "raw_patterns": ["FUND", "FINANCE", "BANKROLL", "INVEST", "GRANT"],
    "canonical": "FINANCES",
    "abstract": "EXCHANGES",
    "properties": {
        "has_amount": True,
        "financial": True,
        "enables": True,
        "resource_transfer": True
    }
}
```

This enables natural language query conversion:
- "Who teaches about biochar?" → Maps to `MENTORS`, `EDUCATES_ABOUT`, `HAS_EXPERTISE`
- "What organizations support regenerative farming?" → Maps to `FUNDS`, `ADVOCATES_FOR`, `PRACTICES`
- "How does composting affect soil?" → Maps to `TRANSFORMS_INTO`, `ENRICHES`, `BENEFITS`

### 3. Multi-Storage Format
Every relationship stores all hierarchy levels for flexible querying:
```json
{
  "source": "Dr. Jane Smith",
  "target": "John Doe",
  "relationship_raw": "LIGHTED_FIRE_IN",        // Exact extraction
  "relationship_domain": "MENTORS",             // Semantic type
  "relationship_canonical": "EDUCATES",         // Broad category
  "relationship_abstract": "INFLUENCES",        // Abstract type
  "properties": {
    "implies_seniority": true,
    "personal_guidance": true
  },
  "description": "Dr. Jane Smith lighted a fire in John Doe",
  "confidence": 0.95
}
```

## Multi-Modal Embedding Architecture

### Text Embeddings (OpenAI)
- **Model**: text-embedding-3-small (1536 dimensions)
- **Content**: Entity names, descriptions, relationships
- **Use cases**: Semantic search, similarity matching

### Graph Embeddings (Node2Vec/GraphSAGE)
- **Dimensions**: 128-256
- **Method**: Random walks capture structural patterns
- **Use cases**: Finding structurally similar entities, community detection

### Relationship Embeddings (TransE/RotatE)
- **Principle**: head + relation ≈ tail in embedding space
- **Dimensions**: 100-200
- **Use cases**: Link prediction, analogical reasoning

### Temporal Embeddings
- **Source**: Episode timestamps, publication dates
- **Use cases**: Temporal evolution analysis, trend detection

## Implementation Architecture

```python
class HybridKnowledgeGraph:
    def __init__(self):
        self.text_embedder = OpenAIEmbedder()      # Semantic meaning
        self.graph_embedder = Node2Vec()           # Structural patterns
        self.relationship_embedder = TransE()      # Relationship semantics
        self.gnn_reasoner = GraphNeuralNetwork()   # Complex reasoning
```

## Natural Language to Graph Query Conversion

The semantic domain ontology enables sophisticated natural language understanding:

### Query Parsing Pipeline
```python
def parse_natural_query(query: str) -> GraphQuery:
    # 1. Extract intent and entities
    intent = extract_intent(query)  # "find", "how", "what", "who"
    entities = extract_entities(query)  # "biochar", "soil", "regenerative"

    # 2. Map to domain ontology relationships
    domain_relationships = map_to_domain_ontology(query)
    # "teaches about" → MENTORS, EDUCATES_ABOUT, HAS_EXPERTISE
    # "improves" → ENHANCES, BENEFITS, ENRICHES
    # "founded by" → FOUNDED_BY, CREATED_BY, ESTABLISHED_BY

    # 3. Build graph query with semantic types
    return build_semantic_query(intent, entities, domain_relationships)
```

### Example Conversions
```python
# Natural: "Who are the experts on biochar?"
# Graph Query:
MATCH (p:PERSON)-[r:HAS_EXPERTISE|RESEARCHES|TEACHES_ABOUT]->(c:CONCEPT {name: "biochar"})
WHERE r.domain_type IN ['HAS_EXPERTISE', 'SPECIALIZES_IN', 'MASTERS']
RETURN p, r.confidence

# Natural: "What practices improve soil health?"
# Graph Query:
MATCH (practice:PRACTICE)-[r]->(soil:CONCEPT {name: "soil health"})
WHERE r.domain_type IN ['ENRICHES', 'REGENERATES', 'BENEFITS', 'ENHANCES']
RETURN practice, r.domain_type, r.properties

# Natural: "Organizations that fund regenerative agriculture"
# Graph Query:
MATCH (org:ORGANIZATION)-[r:FUNDS|SUPPORTS|SPONSORS]->(target)
WHERE r.domain_type IN ['FUNDS', 'INVESTS_IN', 'GRANTS']
  AND target.name CONTAINS 'regenerative'
RETURN org, r.properties.amount, target
```

## Query Capabilities

### 1. Semantic Domain Search
Query using the rich domain ontology:
```python
# Find all mentoring relationships
results = kg.domain_query(
    domain_types=['MENTORS', 'GUIDES', 'COACHES'],
    include_properties={'personal_guidance': True}
)

# Find financial relationships with amounts
financial = kg.domain_query(
    domain_types=['FUNDS', 'INVESTS_IN', 'GRANTS'],
    properties={'has_amount': True},
    min_amount=10000
)
```

### 2. Multi-Level Query
Leverage the full hierarchy:
```python
def multi_level_query(query):
    # Start with specific domain types
    domain_results = search_domain_relationships(query)

    # Fall back to canonical if needed
    if len(domain_results) < threshold:
        canonical_results = search_canonical_relationships(query)

    # Preserve raw relationships for context
    for result in domain_results:
        result.raw_context = get_raw_relationship(result)

    return combine_results(domain_results, canonical_results)
```

### 3. Property-Based Search
Use domain ontology properties:
```python
# Find all causal relationships
causal = kg.property_search(
    properties={'causal': True},
    entities=['climate change', 'carbon']
)

# Find knowledge transfer relationships
education = kg.property_search(
    properties={
        'knowledge_transfer': True,
        'implies_seniority': True
    }
)

### 4. Analogical Reasoning
Using relationship embeddings:
```python
# "Biochar is to soil as compost is to ?"
analogy = kg.complete_analogy(
    source="biochar",
    source_relation="improves",
    source_target="soil",
    query_source="compost"
)
# Returns: "garden beds", "plant health", etc.
```

### 5. Link Prediction
Predict missing relationships:
```python
# Predict likely relationships between entities
predictions = kg.predict_links("Aaron Perry", "regenerative agriculture")
# Returns: [("ADVOCATES_FOR", 0.92), ("TEACHES", 0.87), ...]
```

### 6. Multi-Hop Reasoning
Answer complex questions using GNNs:
```python
question = "What practices do organizations use that were founded by people who graduated from MIT?"
answer = kg.multi_hop_query(question)
# GNN traverses: MIT → graduates → founded → organizations → practices
```

## Graph Neural Network Integration

### Message Passing Architecture
```python
class KnowledgeGNN(nn.Module):
    def forward(self, subgraph, question_embedding):
        # 1. Embed entities and relationships
        node_features = self.embed_nodes(subgraph)
        edge_features = self.embed_edges(subgraph)

        # 2. Message passing layers
        for layer in self.gnn_layers:
            node_features = layer(node_features, edge_features, subgraph)

        # 3. Attention over subgraph guided by question
        attended = self.attention(node_features, question_embedding)

        # 4. Generate answer
        return self.decoder(attended)
```

## Storage Schema

### Entity Storage
```json
{
  "id": "uuid-123",
  "name": "Biochar",
  "type": "CONCEPT",
  "type_canonical": "MATERIAL",
  "embeddings": {
    "text": [0.1, 0.2, ...],      // 1536-dim OpenAI
    "graph": [0.3, 0.4, ...],      // 128-dim Node2Vec
    "context": [0.5, 0.6, ...]     // 256-dim sentence transformer
  },
  "metadata": {
    "first_mentioned": "episode_120",
    "frequency": 47,
    "episodes": [120, 122, 124, 165]
  }
}
```

### Relationship Storage
```json
{
  "id": "rel-456",
  "source_id": "uuid-123",
  "target_id": "uuid-789",
  "types": {
    "raw": "SEQUESTERS_CARBON_IN",
    "canonical": "STORES",
    "abstract": "TRANSFORMS"
  },
  "embedding": [0.7, 0.8, ...],   // TransE embedding
  "metadata": {
    "confidence": 0.95,
    "frequency": 3,
    "temporal": "2024-01-15"
  }
}
```

## Query Examples

### Finding Hidden Connections
```python
# Entities with no text similarity but similar graph positions
dissimilar_text = kg.find_entities_with_different_text_similar_structure(
    min_text_distance=0.8,
    max_graph_distance=0.2
)
# Discovers: "composting" and "biochar" (different processes, similar ecological roles)
```

### Temporal Evolution
```python
# How did understanding of regenerative agriculture evolve?
evolution = kg.track_concept_evolution(
    concept="regenerative agriculture",
    start_episode=1,
    end_episode=172
)
# Shows: Early focus on soil → Integration with climate → Economic models
```

### Community Detection
```python
# Find concept clusters
communities = kg.detect_semantic_communities()
# Returns groups like:
# - Soil health: biochar, composting, mycorrhizae, carbon sequestration
# - Social systems: cooperatives, community gardens, local economy
# - Water systems: rainwater harvesting, greywater, watershed management
```

## Benefits of Hybrid Approach

### 1. Robustness
- If text search fails, graph structure provides alternatives
- Multiple paths to find information

### 2. Discovery
- Find non-obvious connections through graph traversal
- Identify knowledge gaps via link prediction

### 3. Disambiguation
- "Apple" (company) vs "Apple" (fruit) have different graph neighborhoods
- Context from graph structure resolves ambiguity

### 4. Compositional Queries
- "Companies founded by people who practice permaculture"
- Requires both text understanding and graph traversal

### 5. Reasoning
- Answer "why" and "how" questions using path analysis
- Provide evidence chains for answers

## Performance Optimizations

### Embedding Caching
- Pre-compute embeddings for all canonical relationships
- Cache frequently queried entity embeddings
- Use FAISS for fast nearest neighbor search

### Graph Indexing
- Build indices for common traversal patterns
- Pre-compute k-hop neighborhoods for important entities
- Use GraphQL for efficient subgraph queries

### Approximate Methods
- Use LSH for approximate nearest neighbors
- Sample large graphs for GNN processing
- Beam search for multi-hop queries

## Future Enhancements

### 1. Audio Embeddings
- Embed actual audio segments from podcasts
- Enable queries like "find discussions with similar tone/emotion"

### 2. Multimodal Fusion
- Combine text + graph + audio + temporal embeddings
- Learn optimal fusion weights per query type

### 3. Active Learning
- Track which queries fail
- Automatically improve embeddings based on user feedback
- Suggest relationships to verify/add

### 4. Federated Knowledge Graphs
- Connect to other sustainability knowledge bases
- Cross-graph reasoning and discovery
- Maintain provenance and attribution

## Semantic Domain Categories

Our domain ontology includes ~150 semantic relationship types organized into meaningful categories:

### Social & Personal Relationships
- **MENTORS**, **GUIDES**, **COACHES**: Personal guidance and knowledge transfer
- **COLLABORATES_WITH**, **PARTNERS_WITH**: Professional partnerships
- **EMPLOYS**, **WORKS_FOR**, **VOLUNTEERS_AT**: Employment relationships
- **FRIENDS_WITH**, **MARRIED_TO**: Personal connections

### Knowledge & Education
- **HAS_EXPERTISE**, **SPECIALIZES_IN**, **MASTERS**: Domain expertise
- **TEACHES_ABOUT**, **EDUCATES_ABOUT**: Knowledge sharing
- **RESEARCHES**, **STUDIES**, **INVESTIGATES**: Academic pursuits
- **LEARNED_FROM**, **INSPIRED_BY**: Knowledge acquisition

### Financial & Economic
- **FUNDS**, **INVESTS_IN**, **GRANTS**: Financial support
- **COSTS**, **VALUES_AT**: Economic valuation
- **PURCHASES**, **SELLS**, **TRADES**: Transactions
- **OWNS**, **CONTROLS**: Ownership

### Environmental & Ecological
- **SEQUESTERS**, **STORES_CARBON**: Carbon management
- **REGENERATES**, **ENRICHES**, **DEPLETES**: Soil health
- **CONSERVES**, **PRESERVES**: Environmental protection
- **TRANSFORMS_INTO**, **DECOMPOSES_INTO**: Natural processes

### Actions & Effects
- **CREATES**, **PRODUCES**, **MANUFACTURES**: Production
- **BENEFITS**, **HARMS**, **IMPACTS**: Effects and consequences
- **PREVENTS**, **CAUSES**, **ENABLES**: Causation
- **ADVOCATES_FOR**, **OPPOSES**: Advocacy

### Attributes & Properties
- **HAS_CAPACITY**, **HAS_DURATION**, **HAS_SIZE**: Measurements
- **LOCATED_IN**, **ORIGINATES_FROM**: Geography
- **OCCURS_DURING**, **HAPPENS_BEFORE**: Temporal
- **CONTAINS**, **COMPOSED_OF**: Composition

Each domain type includes properties that enable sophisticated queries:
```python
"SEQUESTERS": {
    "properties": {
        "environmental": True,
        "carbon_related": True,
        "quantifiable": True,
        "beneficial": True
    }
}
```

## Implementation Timeline

### Phase 1: Comprehensive Extraction ⏳ (In Progress)
- ✅ Created extraction framework with Pydantic models
- ✅ Designed hierarchical semantic ontology
- ⏳ **Extracting relationships: 22/172 episodes complete** (12.8%)
- 📊 Estimated completion: ~20 hours at current rate
- 💡 Discovering new entities and literal values

### Phase 2: Normalization & Enrichment (Ready)
- ✅ Built semantic normalization script with ~150 domain types
- ✅ Created entity deduplication system
- 🔜 Run normalization on completed extractions
- 🔜 Generate statistics and quality reports

### Phase 3: Graph Embeddings
- Implement Node2Vec for structural embeddings
- Add TransE for relationship embeddings
- Create similarity indices with FAISS

### Phase 4: Query Interface
- Build query API with multiple modes
- Implement GNN for complex reasoning
- Add natural language query processing

### Phase 5: Production Deployment
- Optimize for scale and performance
- Add caching and indexing
- Deploy as queryable API service

## Usage Examples

### Research Assistant
```python
# "What are all the water conservation methods discussed?"
results = kg.query("""
    Find all concepts and practices related to water conservation,
    including who implements them and where
""")
```

### Knowledge Discovery
```python
# Find unexpected connections
surprises = kg.find_surprising_connections(
    max_common_neighbors=2,
    min_embedding_similarity=0.7
)
# Discovers: "mushroom cultivation" connected to "carbon markets"
```

### Fact Verification
```python
# Verify claims with evidence chains
claim = "Biochar can sequester carbon for thousands of years"
evidence = kg.verify_with_sources(claim)
# Returns: Episodes, timestamps, and relationship paths supporting claim
```

## Why Semantic Domain Ontology Matters

The semantic domain layer is the **key innovation** that makes this knowledge graph truly intelligent:

### Natural Language Understanding
```python
# User asks: "Who teaches about soil health?"
# System understands "teaches" maps to multiple domain types:
domain_types = ['MENTORS', 'EDUCATES_ABOUT', 'HAS_EXPERTISE', 'INSTRUCTS']

# This catches relationships that simple keyword matching would miss:
# - "Dr. Smith mentors farmers on soil practices" → MENTORS
# - "Jane has deep expertise in soil biology" → HAS_EXPERTISE
# - "Bob instructs workshops on composting" → INSTRUCTS
```

### Query Intelligence
The domain properties enable sophisticated filtering:
```python
# Find all beneficial environmental practices
results = kg.query(
    properties={
        'environmental': True,
        'beneficial': True
    }
)
# Returns: SEQUESTERS, REGENERATES, CONSERVES, ENRICHES

# Find knowledge transfer with seniority
education = kg.query(
    properties={
        'knowledge_transfer': True,
        'implies_seniority': True
    }
)
# Returns: MENTORS, GUIDES, COACHES (not just "teaches")
```

### Semantic Similarity Without Embeddings
Even without neural embeddings, the domain ontology provides semantic understanding:
```python
# Query: "financial support for regenerative agriculture"
# Domain mapping instantly identifies:
# - FUNDS, INVESTS_IN, GRANTS (financial support)
# - PRACTICES, IMPLEMENTS, ADVOCATES_FOR (regenerative agriculture)

# This works because domain types group semantically similar concepts
```

## Conclusion

This hybrid architecture with **semantic domain ontology at its core** transforms the knowledge graph from a static visualization into a dynamic semantic intelligence system capable of:

1. **Semantic Understanding** - Domain ontology maps natural language to graph concepts
2. **Intelligent Querying** - Properties enable sophisticated filtering and discovery
3. **Preserved Nuance** - All 837+ raw relationship types retained for detailed analysis
4. **Multi-Level Reasoning** - Query at raw, domain, canonical, or abstract levels
5. **Natural Language Conversion** - Domain layer bridges human questions to graph queries

The semantic domain layer is not just an intermediate normalization step—it's the **semantic brain** that enables the system to understand intent, recognize patterns, and answer questions that would be impossible with simple keyword matching or generic relationship types.

By combining:
- **837+ raw relationship types** (nuance)
- **~150 semantic domain types** (intelligence)
- **45 canonical types** (broad queries)
- **15 abstract types** (analysis)

We achieve a system that preserves all the richness of natural language while enabling sophisticated graph queries and reasoning.

---

*Next steps: Complete extraction → Normalize relationships → Implement embeddings → Deploy query interface*