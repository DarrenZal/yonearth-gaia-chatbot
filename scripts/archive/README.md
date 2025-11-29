# Archive Directory

This directory contains archived scripts from the development process. These scripts are kept for historical reference but are no longer actively used.

## Directory Structure

### `ace_extraction/`
ACE (Action-Centric Entity) framework knowledge graph extraction scripts:
- `extract_books_ace_full.py` - ACE extraction from books
- `extract_knowledge_from_episodes.py` - ACE extraction from podcast episodes
- `extract_knowledge_from_books.py` - Book knowledge extraction
- `add_classification_flags_to_episodes.py` - Episode classification flags
- `add_classification_flags_to_unified_graph.py` - Graph classification

**Status**: Replaced by `build_unified_graph_hybrid.py`

### `book_processing/`
Early book integration and processing scripts:
- `fix_book_formatting.py` - Book formatting fixes
- `integrate_books_into_unified_graph.py` - Book-graph integration
- `inspect_pinecone_books.py` - Pinecone book inspection
- `postprocess_episodes_podcast.py` - Episode postprocessing

**Status**: Functionality integrated into main ingestion pipeline

### `graphrag_dev/`
GraphRAG development and experimentation scripts:
- `compute_graphrag_umap_embeddings_test.py` - UMAP embedding tests
- `apply_umap_results.py` - UMAP result application
- `build_unified_graph_v2.py` - Earlier graph builder version
- `transform_to_discourse_graph.py` - Discourse graph transformation
- `validate_unified_graph.py` - Graph validation
- `generate_level_3_super_categories.py` - Level 3 category generation
- `generate_hierarchical_voronoi.py` - Hierarchical voronoi layout

**Status**: Superseded by `build_unified_graph_hybrid.py` and `generate_graphrag_hierarchy.py`

### `layout_experiments/`
Various 3D layout algorithm experiments:
- `generate_constrained_voronoi_layout.py` - Constrained voronoi
- `generate_force_layout.py` - Force-directed layout
- `generate_hierarchical_voronoi.py` - Hierarchical voronoi
- `generate_strict_treemap_layout.py` - Strict treemap
- `generate_voronoi2_layout.py` - Voronoi variation 2
- `generate_voronoi4_layout.py` - Voronoi variation 4
- `generate_voronoi5_layout.py` - Voronoi variation 5
- `layout_sanity_check.py` - Layout validation
- `layout_supervised_semantic.py` - Supervised semantic layout
- `fix_level3_positions.py` - Level 3 position fixes
- `fix_level3_children_ids.py` - Level 3 children ID fixes
- `fix_hierarchy_children.py` - Hierarchy children fixes

**Status**: Experimental layouts for 3D graph visualization. Current production layout uses treemap + force-directed hybrid.

### `testing/`
Test scripts for various components:
- `test_api.py` - API testing
- `test_book_api.py` - Book API testing
- `test_improved_validator.py` - Validator testing
- `test_voice_api.py` - Voice API testing
- `test_voice.py` - Voice client testing
- `playwright_stricttreemap_hover_test.py` - Playwright UI testing

**Status**: One-off test scripts. Use `tests/` directory for formal test suite.

### `visualization/`
Legacy 2D visualization scripts:
- Original t-SNE and UMAP podcast map generators
- BERTopic-based clustering visualizations
- Nomic Atlas integration

**Status**: Replaced by 3D GraphRAG visualization system

### `knowledge_graph/`
Earlier knowledge graph extraction experiments:
- Episode and book knowledge extraction
- Entity and relationship extraction prototypes

**Status**: Superseded by ACE framework extraction

### `graphrag/`
GraphRAG-specific archived scripts:
- Community detection experiments
- Graph processing utilities

**Status**: Integrated into main GraphRAG pipeline

## Active Scripts (in `scripts/`)

The main `scripts/` directory contains actively used scripts:
- `start_local.py` - Local development server
- `setup_data.py` - Data initialization
- `view_feedback.py` - Feedback analysis tool
- `add_to_vectorstore.py` - Vector database management
- `build_unified_graph_hybrid.py` - **ACTIVE** graph builder
- `compute_graphrag_umap_embeddings.py` - **ACTIVE** UMAP embedding generation
- `generate_graphrag_hierarchy.py` - **ACTIVE** hierarchical graph generation

## Migration Notes

If you need to reference archived functionality:
1. Check commit history for when the script was active
2. Review the replacement implementation in active codebase
3. Extract specific algorithms or approaches as needed

## Cleanup History

- **2025-11-29**: Major cleanup and reorganization
  - Moved 40+ scripts to archive
  - Organized by functional category
  - Updated documentation
