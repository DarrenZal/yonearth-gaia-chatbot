# GraphRAG Leiden Refactor Summary

## Overview

This refactor replaces the **embedding-based KMeans clustering** approach with a **graph-topology-based Hierarchical Leiden** algorithm for generating the GraphRAG hierarchy used in the 3D visualization.

## Problem Statement

### What Was Wrong (Before)

1. **Clustering ignored graph structure**: MiniBatchKMeans clustered entities based only on embedding similarity, not their actual connections in the knowledge graph
2. **Singleton anomalies**: "real estate development" ended up isolated in its own cluster at all hierarchy levels despite being semantically close (0.098 UMAP distance) to "park and open space design"
3. **Name mapping mismatch**: Cluster names came from old Microsoft GraphRAG Leiden output (`community_id_mapping.json`) but clusters came from custom KMeans, creating confusing labels like "Sustainable Agriculture and Ethics" containing only "real estate development"
4. **Fixed cluster counts**: Hard-coded targets (300→30→7) forced artificial boundaries regardless of natural community structure

### Root Cause

The code at lines 299-398 used MiniBatchKMeans on entity embeddings:
- `cluster_level_1()`: Clustered 44k entities into 300 clusters using embedding similarity
- `cluster_parents()`: Clustered L1 centers into L2, then L2 centers into L3
- This ignored the actual graph edges (who is connected to whom)

## Solution Implemented

### New Approach (After)

1. **Graph-based clustering**: Uses `graspologic.partition.hierarchical_leiden()` which respects graph topology
2. **Natural hierarchies**: Leiden algorithm discovers natural community structure based on edge connectivity
3. **Cluster summaries for RAG**: Aggregates entity names/descriptions from top-50 most central nodes per cluster (by degree centrality) into `summary_text` field for future LLM summarization
4. **Proper naming**: Fresh cluster IDs generated from Leiden output, no mixing with old Microsoft data

### Code Changes

**New Dependencies** (lines 33-37):
```python
from graspologic.partition import hierarchical_leiden
```

**New Configuration** (lines 68-70):
```python
MAX_CLUSTER_SIZE = 100  # Controls granularity instead of fixed counts
TOP_ENTITIES_FOR_SUMMARY = 50  # For RAG summaries
```

**Replaced Functions** (lines 296-499):
- ❌ Removed: `cluster_level_1()`, `cluster_parents()`, `build_hierarchy()`
- ✅ Added: `run_hierarchical_leiden()`, `extract_cluster_tree()`, `build_cluster_summary_text()`, `build_hierarchy_from_leiden()`

**New Outputs**:
- `graphrag_hierarchy.json` - Compatible with existing 3D viewer
- `cluster_registry.json` - NEW: Flat registry with `summary_text` for each cluster (ready for LLM summarization)

## Expected Results

### "Real Estate Development" Example

**Before (Wrong)**:
- Isolated in `l1_116` (1 entity), `l2_26` (1 entity), `l3_4` (1 entity)
- Labeled "Sustainable Agriculture and Ethics" (from mismatched mapping)
- 0.098 distance from "park and open space design" but different clusters

**After (Expected)**:
- Grouped with semantically AND topologically related entities
- Cluster name will reflect actual members (e.g., "Urban Planning & Development")
- Natural hierarchy based on graph connectivity, not arbitrary embedding distances

### Benefits

1. **Topology matters**: Entities connected by many edges cluster together
2. **Semantic coherence**: Clusters contain entities that actually reference each other
3. **RAG-ready summaries**: Each cluster has `summary_text` aggregating top entity descriptions
4. **No singleton anomalies**: Leiden merges small isolated nodes into nearby communities
5. **Reproducible**: No mixing of different clustering runs

## How to Run

```bash
# Generate new hierarchy with Leiden clustering
python3 scripts/generate_graphrag_hierarchy.py

# Expected outputs:
# - /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json
# - /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/cluster_registry.json

# Deploy to production (gaiaai.xyz)
sudo cp data/graphrag_hierarchy/graphrag_hierarchy.json /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/
sudo systemctl reload nginx
```

## Next Steps

1. **Run the script** to generate new hierarchy with Leiden clustering
2. **Review cluster_registry.json** to verify `summary_text` quality
3. **Generate LLM summaries** for each cluster using the `summary_text` field (future enhancement)
4. **Test 3D visualization** at https://gaiaai.xyz/YonEarth/graph/ to verify no anomalies
5. **Compare results**: Check if "real estate development" is now properly grouped

## Technical Details

### Hierarchical Leiden Algorithm

- **Input**: NetworkX graph with weighted edges
- **Output**: Nested hierarchy tree (root → communities → sub-communities → leaf clusters)
- **Parameter**: `max_cluster_size=100` controls maximum entities per cluster
- **Method**: Optimizes modularity while respecting graph connectivity

### Cluster Registry Structure

Each cluster in `cluster_registry.json` contains:
```json
{
  "id": "level_1_42",
  "level": 1,
  "type": "community",
  "parent": "level_0_0",
  "children": ["level_2_105", "level_2_106", ...],
  "entities": ["entity_123", "entity_456", ...],
  "entity_count": 847,
  "summary_text": "sustainable agriculture: practices that... | regenerative farming: methods for... | ..."
}
```

The `summary_text` field is ready for LLM summarization like:
```python
summary = openai.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": f"Summarize this community in 2-3 sentences: {cluster['summary_text']}"
    }]
)
```

## Files Modified

- `/home/claudeuser/yonearth-gaia-chatbot/scripts/generate_graphrag_hierarchy.py` - Complete refactor (lines 1-677)

## Dependencies

- ✅ `graspologic` (already installed, v3.4.4)
- ✅ `networkx` (already used)
- ✅ `umap-learn` (unchanged)
- ✅ `openai` (unchanged)

## Backward Compatibility

The output `graphrag_hierarchy.json` maintains the same structure:
```json
{
  "entities": {...},
  "relationships": [...],
  "clusters": {
    "level_0": {...},  // Individual entities
    "level_1": {...},  // Fine clusters (Leiden communities)
    "level_2": {...},  // Medium clusters (Leiden sub-communities)
    "level_3": {...}   // Coarse clusters (Leiden leaf clusters)
  },
  "metadata": {...}
}
```

The 3D viewer code should work without changes. Only the cluster membership and names will differ (correctly this time).

---

**Status**: ✅ Code refactored and ready to run
**Date**: 2025-11-27
**Next Action**: Execute `python3 scripts/generate_graphrag_hierarchy.py` to test
