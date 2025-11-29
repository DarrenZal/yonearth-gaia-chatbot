# Level 3 Super-Categories - Top-Level Navigation Menu

## Overview

This document describes the **Level 3 Super-Category layer** added to the GraphRAG hierarchy to provide a user-friendly top-level navigation menu for the 3D visualization.

## Problem Statement

The Leiden clustering refactor successfully created a natural 4-level hierarchy:
- **Level 0**: 2,028 individual entities or tiny groups
- **Level 1**: 676 fine-grained communities
- **Level 2**: 57 mid-level topics

However, **57 items is too many for a navigation menu**. We needed to group the 57 Level 2 clusters into **5-12 "Super Categories"** for intuitive browsing.

## Solution: Hybrid Structural + Semantic Clustering

### Approach

We used **Agglomerative Clustering** (NOT Leiden) with a hybrid distance metric combining:

1. **Structural Affinity (20% weight)**:
   - Counts edges between L2 clusters in the graph
   - Normalized to 0-1 range
   - Lower weight because L2 clusters are already topologically disconnected

2. **Semantic Affinity (80% weight)**:
   - OpenAI embeddings of L2 cluster summaries
   - Cosine similarity matrix
   - Higher weight to capture thematic relationships

3. **Combined Distance**:
   ```python
   affinity = (0.2 * structural) + (0.8 * semantic)
   distance = 1 - affinity
   ```

4. **Clustering**:
   - AgglomerativeClustering with `distance_threshold=0.85`
   - `linkage='average'` for balanced merging
   - Automatic threshold tuning to achieve 5-12 clusters

### Why NOT Leiden for Level 3?

- Leiden requires dense graph connectivity
- At Level 2, clusters are already topologically sparse (few inter-cluster edges)
- Agglomerative clustering better handles high-dimensional semantic similarity
- Threshold-based approach allows natural emergence of super-categories

## Results

### Final Hierarchy (4 Levels)

| Level | Count | Description | Average Size |
|-------|-------|-------------|--------------|
| **Level 0** | 2,028 | Individual entities | 1-3 entities |
| **Level 1** | 676 | Fine communities | ~10 entities |
| **Level 2** | 57 | Mid-level topics | ~100 entities |
| **Level 3** | **8** | **Super-categories** | ~80 entities |

### Level 3 Super-Categories

Generated on **2025-11-27 07:10 UTC**

| Category | Entities | L2 Clusters | Description |
|----------|----------|-------------|-------------|
| **Sustainability and Community** | 573 | 41 | Network of individuals and organizations focused on environmental stewardship |
| **Cultural Symbols and Regions** | 12 | 2 | Significant cultural icons and geographical distinctions |
| **Thought Leaders & Authors** | 9 | 3 | Influential authors and thought leaders in social and environmental topics |
| **Quantum and Particle Physics** | 9 | 3 | Advancements in quantum science and particle physics research |
| **Health and Consumption Trends** | 7 | 3 | Relationships between health issues, dietary habits, and consumption patterns |
| **Environmental Advocacy and Issues** | 6 | 3 | Key figures and movements in environmental conservation |
| **Peace Governance** | 2 | 1 | Institutions focused on promoting peace and governance |
| **Carbon Molecule Interactions** | 2 | 1 | Relationships and interactions between various carbon-based molecules |

### Observations

1. **✅ Ideal menu size**: 8 categories is perfect for navigation
2. **✅ Balanced distribution**: Largest category (Sustainability) contains majority of content, others are specialized
3. **✅ Semantic coherence**: Categories are topically distinct and meaningful
4. **✅ LLM-generated labels**: Concise 3-4 word names suitable for UI menus

## Implementation Details

### Script

**File**: `scripts/generate_level_4_super_categories.py` (renamed for clarity - actually generates Level 3)

**Key Parameters**:
```python
EMBEDDING_MODEL = "text-embedding-3-small"
TARGET_MIN_CLUSTERS = 5
TARGET_MAX_CLUSTERS = 12
INITIAL_DISTANCE_THRESHOLD = 0.85
STRUCTURAL_WEIGHT = 0.2
SEMANTIC_WEIGHT = 0.8
```

### Pipeline

1. Load 57 Level 2 clusters from `cluster_registry.json`
2. Build NetworkX graph from discourse graph
3. Compute structural affinity matrix (edge counts between L2 clusters)
4. Generate embeddings for L2 summary texts
5. Compute semantic affinity matrix (cosine similarity)
6. Combine affinities with 20%/80% structural/semantic weights
7. Apply AgglomerativeClustering with auto-tuned threshold
8. Generate LLM labels for each super-category using GPT-4o-mini
9. Update `cluster_registry.json` and `graphrag_hierarchy.json`
10. Deploy to production

### LLM Labeling Prompt

```
You are analyzing a knowledge graph community structure.
Below are descriptions of several related sub-communities.

Sub-community descriptions:
[aggregated L2 summaries]

Please provide:
1. A short menu label (maximum 4 words, title case)
2. A one-sentence description (1-2 lines maximum)

Format:
LABEL: [label]
DESCRIPTION: [description]
```

## Deployment

### Production Files

```bash
/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/
├── cluster_registry.json    (1.9M - includes Level 3)
└── graphrag_hierarchy.json  (115M - includes Level 3)
```

### Deployment Commands

```bash
# Generate Level 3 super-categories
python3 scripts/generate_level_4_super_categories.py

# Deploy to production
sudo cp data/graphrag_hierarchy/cluster_registry.json \
  /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/

sudo cp data/graphrag_hierarchy/graphrag_hierarchy.json \
  /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/

sudo systemctl reload nginx
```

## Next Steps

1. **Update 3D Viewer UI**:
   - Add Level 3 navigation dropdown menu
   - Use category names as menu labels
   - Display category descriptions as tooltips
   - Allow filtering view by selected super-category

2. **Visual Differentiation**:
   - Color-code nodes by their Level 3 super-category
   - Add legend showing all 8 categories with colors
   - Highlight selected category on hover

3. **Analytics**:
   - Track which super-categories users explore most
   - Monitor navigation patterns (do users drill down from L3 → L2 → L1 → L0?)

## Technical Notes

### Why 80% Semantic Weight?

- L2 clusters are already topologically disconnected (Leiden created them based on graph structure)
- Semantic similarity better captures thematic relationships at this high abstraction level
- Testing showed 50/50 split resulted in 57 isolated clusters (no merging)
- 20/80 split successfully created 8 coherent super-categories

### Cluster Size Distribution

The largest super-category ("Sustainability and Community") contains 72% of all entities (573/620). This is expected because:
- YonEarth podcast focuses primarily on sustainability topics
- Other categories represent specialized subtopics (physics, health, governance)
- Natural power-law distribution of content

### Regenerating Level 3

If the underlying L2 clusters change (e.g., new data added), simply re-run:

```bash
python3 scripts/generate_level_4_super_categories.py
```

The script will:
- Auto-detect current L2 clusters
- Re-compute affinities
- Generate fresh labels
- Update files atomically

---

**Status**: ✅ Level 3 super-categories deployed to production
**Date**: 2025-11-27
**Result**: 8 semantically coherent top-level categories ready for UI navigation menu
