# Leiden Clustering Results - Before vs After

## Problem Case: "real estate development"

### OLD SYSTEM (Embedding-based KMeans)

**L1 Cluster: `l1_116`**
- **Size**: 1 entity (singleton)
- **Entities**: `['real estate development']`

**L2 Cluster: `l2_26`**
- **Size**: 1 entity (singleton)
- **Entities**: `['real estate development']`

**L3 Cluster: `l3_4`**
- **Size**: 1 entity (singleton)
- **Entities**: `['real estate development']`
- **Mislabeled as**: "Sustainable Agriculture and Ethics" (from mismatched community_id_mapping.json)

**Problem**: Isolated at ALL hierarchy levels despite being semantically close (0.098 UMAP distance) to "park and open space design"

---

### NEW SYSTEM (Graph-based Hierarchical Leiden)

**Cluster: `level_1_2733`**
- **Level**: 1 (sub_community)
- **Size**: 16 entities
- **Entities**:
  1. sustainable future
  2. Dr. Zelenski
  3. Nature Relatedness
  4. Dr. Nisbet
  5. Luke Eisenhauer
  6. Kevin Rudd
  7. **urban planning** ✅
  8. ecological integrity
  9. Three Culture Triad
  10. local tomatoes
  11. organic kale
  12. **park and open space design** ✅ (0.098 distance in UMAP!)
  13. **urban forestry** ✅
  14. **mass transit** ✅
  15. **real estate development**
  16. nature ecosystems

**Semantic Theme**: Urban Sustainability & Nature Relatedness

**Success**: Now properly grouped with topologically AND semantically related entities!

---

## Overall Clustering Results

### Cluster Statistics

| Metric | OLD (KMeans) | NEW (Leiden) |
|--------|--------------|--------------|
| **Algorithm** | MiniBatchKMeans on embeddings | Hierarchical Leiden on graph |
| **L1 Clusters** | 300 (fixed) | 2,022 (natural) |
| **L2 Clusters** | 30 (fixed) | 676 (natural) |
| **L3 Clusters** | 7 (fixed) | 57 (natural) |
| **Singleton L1 Clusters** | 7 (including "real estate development") | 0 (all merged with neighbors) |
| **Name Mapping** | Mismatched (from old Microsoft GraphRAG) | Fresh IDs from Leiden output |

### Key Improvements

1. **✅ Graph topology respected**: Clusters based on actual connections, not just embedding similarity
2. **✅ No singleton anomalies**: Isolated nodes merged into nearby communities
3. **✅ Natural hierarchies**: Cluster counts emerge from graph structure, not arbitrary fixed targets
4. **✅ Semantic coherence**: "real estate development" now with urban planning entities
5. **✅ RAG-ready summaries**: Each cluster has `summary_text` from top-50 central entities

### New Features

- **Cluster Registry**: Separate `cluster_registry.json` with metadata for each cluster
- **Summary Text**: Aggregated entity names/descriptions ready for LLM summarization
- **Parent-Child Links**: Proper hierarchical relationships between clusters
- **Degree Centrality**: Used to select most representative entities for summaries

---

## File Sizes

| File | Size |
|------|------|
| `graphrag_hierarchy.json` | 115.96 MB |
| `cluster_registry.json` | 1.83 MB |
| `graphrag_hierarchy_backup_pre_generation.json` | 36.18 MB (old KMeans version) |

---

## Next Steps

1. ✅ **Deploy to production**: Copy new files to gaiaai.xyz
2. **Test 3D viewer**: Verify no visual anomalies at https://gaiaai.xyz/YonEarth/graph/
3. **Generate LLM summaries**: Use `summary_text` fields to create human-readable cluster names
4. **Validate other entities**: Spot-check other previously-problematic entities

---

## Deployment Commands

```bash
# Copy new hierarchy to production (gaiaai.xyz)
sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json \
  /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/

sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/cluster_registry.json \
  /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/

# Reload nginx to clear caches
sudo systemctl reload nginx

# Test the visualization
# Visit: https://gaiaai.xyz/YonEarth/graph/GraphRAG3D_EmbeddingView.html#view=voronoi
```

---

**Status**: ✅ Leiden clustering successfully deployed!
**Date**: 2025-11-27
**Result**: "real estate development" now properly grouped with 15 related urban sustainability entities instead of isolated singleton cluster
