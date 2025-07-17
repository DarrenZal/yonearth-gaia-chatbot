# Semantic Category Implementation Plan

## Current Issues Discovered

1. **Episode 124 Problem**: Categorized with BIOCHAR but doesn't contain the word "biochar" in transcript
2. **Result Diversity Issue**: Category search returns individual chunks, not unique episodes - Episode 120's 31 chunks fill all k=20 slots
3. **Keyword Matching Limitation**: "teach me about soil" won't match BIOCHAR category despite clear relevance

## New Semantic Category Matching Approach

### PHASE 1: Semantic Category Infrastructure

#### 1.1 Create Category Embeddings
```python
# src/rag/semantic_category_matcher.py
class SemanticCategoryMatcher:
    def __init__(self):
        self.categories = self.load_categories()
        self.category_embeddings = self.create_category_embeddings()
        self.category_descriptions = {
            'BIOCHAR': 'carbon sequestration, soil amendment, pyrolysis, charcoal for agriculture',
            'SOIL': 'soil health, regeneration, fertility, microbiome, earth',
            'HERBAL MEDICINE': 'natural healing, plant medicine, herbs, botanical remedies',
            # ... etc
        }
    
    def create_category_embeddings(self):
        """Embed category names + descriptions for semantic matching"""
        embeddings = {}
        for cat, desc in self.category_descriptions.items():
            text = f"{cat} {desc}"
            embeddings[cat] = openai_embed(text)
        return embeddings
```

#### 1.2 Query-Category Matching
```python
def get_semantic_category_matches(self, query: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
    """Find categories semantically related to query"""
    query_embedding = openai_embed(query)
    
    matches = []
    for category, cat_embedding in self.category_embeddings.items():
        similarity = cosine_similarity(query_embedding, cat_embedding)
        if similarity >= threshold:
            matches.append((category, similarity))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)
```

### PHASE 2: Smart Filtering Pipeline

#### 2.1 Update BM25HybridRetriever
```python
def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
    # Step 1: Get semantic category matches
    matched_categories = self.semantic_matcher.get_semantic_category_matches(
        query, 
        threshold=self.category_threshold  # Configurable
    )
    
    if matched_categories:
        # Step 2: Get episodes for matched categories
        episode_ids = set()
        for category, score in matched_categories:
            category_episodes = self.categorizer.get_episodes_by_category(category)
            episode_ids.update(category_episodes)
        
        # Step 3: Ensure result diversity - get best chunks from EACH episode
        results = self.diverse_episode_search(query, episode_ids, k)
    else:
        # No category matches - full search
        results = self.full_hybrid_search(query, k)
    
    return results
```

#### 2.2 Diverse Episode Search
```python
def diverse_episode_search(self, query: str, episode_ids: Set[int], k: int) -> List[Document]:
    """Ensure all matching episodes are represented in results"""
    
    # Get best chunks from each episode
    episode_chunks = {}
    for ep_id in episode_ids:
        # Get all chunks for this episode
        ep_docs = self.get_episode_documents(ep_id)
        
        # Score each chunk against query
        scored_chunks = []
        for doc in ep_docs:
            bm25_score = self.bm25_score(query, doc)
            semantic_score = self.semantic_score(query, doc)
            combined = (self.keyword_weight * bm25_score + 
                       self.semantic_weight * semantic_score)
            scored_chunks.append((doc, combined))
        
        # Keep top 3 chunks per episode
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        episode_chunks[ep_id] = scored_chunks[:3]
    
    # Combine all chunks and sort by score
    all_chunks = []
    for chunks in episode_chunks.values():
        all_chunks.extend(chunks)
    
    all_chunks.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in all_chunks[:k]]
```

### PHASE 3: Category Expansion

#### 3.1 Automatic Synonym Discovery
```python
def build_category_graph(self):
    """Build relationships between categories using embeddings"""
    
    category_relationships = {}
    for cat1 in self.categories:
        related = []
        for cat2 in self.categories:
            if cat1 != cat2:
                similarity = cosine_similarity(
                    self.category_embeddings[cat1],
                    self.category_embeddings[cat2]
                )
                if similarity > 0.6:  # Related threshold
                    related.append((cat2, similarity))
        
        category_relationships[cat1] = sorted(related, key=lambda x: x[1], reverse=True)
    
    return category_relationships
```

#### 3.2 Query Expansion Examples (More Realistic)
- "what makes healthy dirt" → SOIL (0.89), COMPOSTING (0.76), BIOCHAR (0.72), FARMING & FOOD (0.68), PERMACULTURE (0.65)
- "natural ways to heal" → HEALTH & WELLNESS (0.83), HERBAL MEDICINE (0.81), INDIGENOUS WISDOM (0.67), GREEN FAITH (0.64)
- "growing food without chemicals" → FARMING & FOOD (0.86), PERMACULTURE (0.84), BIO-DYNAMICS (0.78), SOIL (0.73), REGENERATIVE (0.71)
- "carbon capture methods" → CLIMATE & SCIENCE (0.88), BIOCHAR (0.85), TECHNOLOGY & MATERIALS (0.72), SUSTAINABILITY (0.68)
- "community resilience building" → COMMUNITY (0.91), SUSTAINABILITY (0.74), REGENERATIVE (0.69), EDUCATION (0.66)

### PHASE 4: Configurable Thresholds

#### 4.1 Add to Web UI
```html
<!-- Semantic Category Matching -->
<div class="category-threshold-selector">
    <label for="categoryMode">Category Matching:</label>
    <select id="categoryMode">
        <option value="strict">Strict (0.8+) - Exact matches only</option>
        <option value="normal" selected>Normal (0.7) - Related categories</option>
        <option value="broad">Broad (0.6) - Explore connections</option>
        <option value="disabled">Disabled - No category filtering</option>
    </select>
</div>
```

#### 4.2 Backend Configuration
```python
CATEGORY_THRESHOLDS = {
    'strict': 0.8,
    'normal': 0.7,
    'broad': 0.6,
    'disabled': 1.1  # Impossible threshold = no matches
}
```

### PHASE 5: Testing & Validation

#### 5.1 Test Cases
1. **"what makes healthy dirt"**
   - Expected matches: SOIL (high), COMPOSTING (high), BIOCHAR (medium), FARMING & FOOD (medium)
   - Should return: Mix of soil health, composting, and biochar episodes including 124
   - Key insight: "dirt" → "soil" semantic connection should be strong

2. **"regenerative farming practices"**
   - Expected matches: REGENERATIVE (very high), FARMING & FOOD (very high), PERMACULTURE (high), SOIL (medium)
   - Should return: Episodes focused on regenerative agriculture
   - Should include episodes that discuss farming even without exact keyword

3. **"how to capture carbon naturally"**
   - Expected matches: BIOCHAR (very high), CLIMATE & SCIENCE (high), SOIL (medium), REGENERATIVE (medium)
   - Should return: All biochar episodes (120, 122, 124, 165) plus other carbon sequestration content
   - Key insight: "capture carbon" → "biochar" connection through semantic understanding

4. **"indigenous knowledge about plants"**
   - Expected matches: INDIGENOUS WISDOM (very high), HERBAL MEDICINE (high), ECOLOGY & NATURE (medium)
   - Should return: Episodes featuring indigenous perspectives on plant medicine and ecology

#### 5.2 Performance Metrics
- Category match accuracy
- Result diversity (unique episodes per query)
- User satisfaction (include in UI feedback)

### PHASE 6: UI Updates

#### 6.1 Show Category Matches
```javascript
// Display which categories matched the query
const categoryMatches = response.category_matches || [];
if (categoryMatches.length > 0) {
    const matchesDiv = document.createElement('div');
    matchesDiv.className = 'category-matches';
    matchesDiv.innerHTML = `
        <small>📂 Matched categories: ${categoryMatches.map(c => c.name).join(', ')}</small>
    `;
    messageDiv.appendChild(matchesDiv);
}
```

#### 6.2 Episode Diversity Indicator
```javascript
// Show how many unique episodes are referenced
const uniqueEpisodes = new Set(citations.map(c => c.episode_number));
const diversityDiv = document.createElement('div');
diversityDiv.className = 'result-diversity';
diversityDiv.innerHTML = `
    <small>📊 Drawing from ${uniqueEpisodes.size} different episodes</small>
`;
```

## Implementation Priority

1. **High Priority** (Week 1)
   - Semantic category matcher
   - Diverse episode search
   - Fix k=20 limitation

2. **Medium Priority** (Week 2)
   - Category expansion/graph
   - UI threshold controls
   - Category match display

3. **Low Priority** (Week 3)
   - Performance optimization
   - Category description editor
   - Analytics dashboard

## Benefits

1. **Solves Episode 124**: Semantic matching ensures "soil" queries find biochar episodes
2. **Better Diversity**: All relevant episodes appear, not just the one with most chunks
3. **User Control**: Adjustable thresholds for different search needs
4. **Transparency**: Users see which categories matched their query
5. **Discoverable**: Helps users explore related topics they didn't know about

## Technical Considerations

1. **Embedding Cache**: Cache category embeddings to avoid repeated API calls
2. **Performance**: Pre-compute category relationships at startup
3. **Fallback**: If semantic matching fails, fall back to keyword matching
4. **Monitoring**: Track which queries match which categories for improvement

This approach gives us true semantic understanding while maintaining the benefits of category-based filtering!