# Knowledge Graph Data Documentation

**Last Updated**: 2025-12-09
**Status**: Production (GraphRAG v2)

## Overview

The YonEarth Knowledge Graph contains extracted entities and relationships from 172 podcast episodes and 4 books. It powers the GraphRAG search backend and 3D visualization.

**Current Statistics**:
- **Entities**: 26,219
- **Relationships**: 39,118
- **Aliases**: 45,673
- **Fictional Entities**: 3,708 (from VIRIDITAS novel)
- **Communities**: 573 (Level 1) + 73 (Level 2)

## Data Locations

### Development (Source of Truth)

```
/home/claudeuser/yonearth-gaia-chatbot/data/
├── knowledge_graph_unified/           # Unified graph data
│   ├── unified_v2.json               # Current production graph (24 MB)
│   ├── unified.json                  # Symlink to unified_v2.json
│   ├── entities_deduplicated.json    # Deduplicated entities (15 MB)
│   ├── relationships_processed.json  # Processed relationships (12 MB)
│   ├── adjacency.json                # Adjacency list for traversal
│   └── entity_merges.json            # Entity merge history
│
├── graphrag_hierarchy/                # GraphRAG visualization data
│   ├── graphrag_hierarchy.json       # Main hierarchy file (60+ MB)
│   ├── cluster_registry.json         # Cluster metadata
│   ├── cluster_registry_with_relationships.json  # With edge data
│   ├── community_id_mapping.json     # ID → title mapping
│   ├── community_embeddings.json     # Cluster embeddings for search
│   ├── force_layout.json             # Force-directed 3D positions
│   └── constrained_voronoi_layout.json  # Voronoi layout
│
├── knowledge_graph/                   # Raw extraction data
│   ├── entities/                     # Per-episode entity extractions
│   ├── books/                        # Book extraction data
│   └── visualization_data.json       # Legacy visualization
│
└── batch_jobs/                        # Batch extraction artifacts
    ├── parent_chunks.json            # 826 parent chunks (episodes + books)
    ├── child_chunks.json             # Child chunks for vector indexing
    └── results/                      # Batch API results
```

### Production (gaiaai.xyz)

```
/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/
├── graphrag_hierarchy_v6_fixed.json  # PRIMARY hierarchy (loaded first by JS)
├── cluster_registry.json             # Cluster metadata with relationships
├── community_id_mapping.json         # Cluster titles for Voronoi view
├── force_layout.json                 # 3D positions
└── constrained_voronoi_layout.json   # Voronoi layout
```

**Important**: The JS viewer loads files in this order:
1. `graphrag_hierarchy_v6_fixed.json` (tried first)
2. `graphrag_hierarchy_v2.json`
3. `graphrag_hierarchy.json`

## How the Graph Was Built

### Phase 1: Content Extraction (Batch API)

**Script**: `scripts/extract_episodes_batch.py`

1. **Parent-Child Chunking**: Content split into ~3,000 token parent chunks
2. **Batch Submission**: Submitted to OpenAI Batch API with gpt-5.1
3. **Entity Extraction**: Structured outputs with Pydantic schema validation

```bash
# Submit batch job
python scripts/extract_episodes_batch.py --submit

# Poll status
python scripts/extract_episodes_batch.py --poll

# Download results
python scripts/extract_episodes_batch.py --download
```

### Phase 2: Post-Processing Pipeline

**Script**: `scripts/process_batch_results.py`

1. **Entity Quality Filtering**: Remove pronouns, generic nouns, sentence-like entities
2. **Fictional Tagging**: Tag entities from VIRIDITAS/Our Biggest Deal as fictional
3. **List Splitting**: Split "A, B, and C" into separate entities
4. **Ontology Normalization**: Normalize types (COMPANY → FORMAL_ORGANIZATION)

### Phase 3: Entity Resolution & Deduplication

**Script**: `scripts/deduplicate_entities.py`

1. **Canonical Resolution**: Map aliases to canonical forms (yonearth.org → Y on Earth Community)
2. **Fuzzy Matching**: Merge similar entities (85%+ similarity)
3. **Fictional Override**: Non-fictional instances override fictional flags on merge

### Phase 4: Unified Graph Build

**Script**: `scripts/build_unified_graph_v2.py`

1. **Combine**: Merge all entities and relationships
2. **Validate**: Check relationship targets exist
3. **Export**: Write to `unified_v2.json`

### Phase 5: GraphRAG Hierarchy Generation

**Script**: `scripts/generate_graphrag_hierarchy.py`

1. **Community Detection**: Hierarchical Leiden algorithm
2. **UMAP Embeddings**: 3D positions for visualization
3. **Cluster Metadata**: Titles, descriptions, top entities

## Rebuilding the Graph

### Full Rebuild (from scratch)

```bash
# 1. Run batch extraction (if needed)
python scripts/extract_episodes_batch.py --submit
# Wait ~24 hours for batch completion
python scripts/extract_episodes_batch.py --download

# 2. Process results
python scripts/process_batch_results.py

# 3. Deduplicate entities
python scripts/deduplicate_entities.py

# 4. Build unified graph
python scripts/build_unified_graph_v2.py

# 5. Generate GraphRAG hierarchy
python scripts/generate_graphrag_hierarchy.py

# 6. Rebuild graph index for RAG search
python scripts/rebuild_graph_index_from_unified.py

# 7. Add relationships to cluster registry (for viewer)
python scripts/add_relationships_to_cluster_registry.py
```

### Incremental Update (regenerate hierarchy only)

```bash
# Rebuild from existing unified_v2.json
python scripts/rebuild_graph_index_from_unified.py
python scripts/add_relationships_to_cluster_registry.py
./scripts/deploy_graphrag.sh
```

## Deploying to Production

Use the deployment script:

```bash
./scripts/deploy_graphrag.sh
```

This script:
1. Copies hierarchy JSON to production
2. Regenerates `community_id_mapping.json`
3. Updates cache buster
4. Reloads nginx

Manual deployment:

```bash
# Copy hierarchy
sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json \
  /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/graphrag_hierarchy_v6_fixed.json

# Copy cluster registry
sudo cp /home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/cluster_registry_with_relationships.json \
  /var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy/cluster_registry.json

# Reload nginx
sudo systemctl reload nginx
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `unified_v2.json` | Source of truth for entities/relationships |
| `graphrag_hierarchy.json` | Full hierarchy with embeddings, clusters, positions |
| `cluster_registry_with_relationships.json` | Cluster data for 3D viewer |
| `community_id_mapping.json` | Maps cluster IDs to human-readable titles |
| `community_embeddings.json` | Cluster embeddings for community search |

## Configuration

Environment variables in `.env`:

```bash
# Extraction settings
GRAPH_EXTRACTION_MODEL=gpt-5.1
GRAPH_EXTRACTION_MODE=batch
PARENT_CHUNK_SIZE=3000
PARENT_CHUNK_MAX=6000
CHILD_CHUNK_SIZE=600
CHILD_CHUNK_OVERLAP=100

# GraphRAG settings
GRAPHRAG_BACKEND_VERSION=v2
GRAPHRAG_KG_BOOST_FACTOR=1.3
```

## Related Documentation

- [Knowledge Graph Extraction Review](./KNOWLEDGE_GRAPH_EXTRACTION_REVIEW.md) - Full extraction pipeline details
- [GraphRAG Rollout Guide](./GRAPHRAG_ROLLOUT.md) - Deployment and rollback procedures
- [Post-Processing System](../src/knowledge_graph/postprocessing/README.md) - Modular post-processing architecture

## Troubleshooting

### Graph not updating in viewer

1. Check which file JS is loading (v6_fixed.json takes priority)
2. Clear browser cache
3. Verify file was copied to correct production path

### Missing entities in search

1. Check if entity was filtered (pronouns, generic nouns)
2. Check if entity was tagged as fictional
3. Rebuild graph index: `python scripts/rebuild_graph_index_from_unified.py`

### Relationships showing as "related"

The viewer reads `rel.predicate` for relationship type. If edges show as "related", the data may have `rel.type` instead. Fix in `GraphRAG3D_EmbeddingView.js:getEdgeDetails()`.
