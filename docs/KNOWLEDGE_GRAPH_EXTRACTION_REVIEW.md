# Knowledge Graph Extraction Review & Improvement Plan

**Date**: December 3, 2025
**Author**: Claude (AI Analysis)
**Status**: Review Complete - Ready for Implementation

---

## Executive Summary

A thorough review of the knowledge graph extraction pipeline revealed several categories of systematic errors affecting data quality. The issues stem from overly permissive extraction prompts, insufficient entity validation, aggressive post-processing merges, and missing entity resolution capabilities.

This document categorizes the errors found, analyzes their root causes, and proposes a comprehensive improvement plan for re-extraction.

---

## Table of Contents

1. [Complete File Inventory](#complete-file-inventory)
2. [Current Data Quality Metrics](#current-data-quality-metrics-baseline)
3. [Error Categories Found](#error-categories-found)
4. [Root Cause Analysis](#root-cause-analysis)
5. [Proposed Improvement Plan](#proposed-improvement-plan)
6. [Implementation Priority](#implementation-priority)
7. [Appendix: Code References](#appendix-code-references)

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-03 | Claude (AI Analysis) | Initial review document |
| 2025-12-03 | Third-party reviewer | Identified additional error categories (6a-6e) |
| 2025-12-03 | Claude (AI Analysis) | Incorporated third-party findings, added Phase 4.3, updated metrics |

---

## Complete File Inventory

This section provides a comprehensive inventory of all files involved in the knowledge graph extraction process.

### Source Data Files

#### Podcast Episode Transcripts
| Path | Description |
|------|-------------|
| `data/transcripts/episode_*.json` | 172 episode transcript files (episodes 0-172, excluding #26) |
| Format | JSON with `full_transcript`, `title`, `guest`, timestamps, metadata |

#### Book Source Files
| Path | Description |
|------|-------------|
| `data/books/veriditas/` | VIRIDITAS: THE GREAT HEALING book |
| `data/books/veriditas/metadata.json` | Book metadata (title, author, chapters) |
| `data/books/y-on-earth/` | Y on Earth: Get Smarter, Feel Better, Heal the Planet |
| `data/books/y-on-earth/metadata.json` | Book metadata |
| `data/books/soil-stewardship-handbook/` | Soil Stewardship Handbook |
| `data/books/soil-stewardship-handbook/metadata.json` | Book metadata |
| `data/books/OurBiggestDeal/` | Our Biggest Deal |
| `data/books/OurBiggestDeal/metadata.json` | Book metadata |

---

### Extraction Scripts

#### Main Extraction Scripts
| Path | Description |
|------|-------------|
| `scripts/archive/ace_extraction/extract_knowledge_from_episodes.py` | Main episode extraction script using ACE framework |
| `scripts/archive/ace_extraction/extract_knowledge_from_books.py` | Main book extraction script |
| `scripts/archive/ace_extraction/extract_books_ace_full.py` | Full book extraction with ACE |
| `scripts/archive/ace_extraction/add_classification_flags_to_episodes.py` | Add classification flags to episode extractions |
| `scripts/archive/ace_extraction/add_classification_flags_to_unified_graph.py` | Add flags to unified graph |

#### Graph Building Scripts
| Path | Description |
|------|-------------|
| `scripts/build_unified_graph_hybrid.py` | **Main graph building script** - combines all extractions into unified graph |
| `scripts/generate_graphrag_hierarchy.py` | Generate GraphRAG hierarchical structure |
| `scripts/compute_graphrag_umap_embeddings.py` | Compute UMAP embeddings for visualization |
| `scripts/archive/graphrag_dev/build_unified_graph_v2.py` | Legacy v2 graph builder |
| `scripts/archive/graphrag_dev/validate_unified_graph.py` | Graph validation script |
| `scripts/archive/graphrag_dev/transform_to_discourse_graph.py` | Transform to discourse graph format |
| `scripts/archive/book_processing/integrate_books_into_unified_graph.py` | Book integration script |

#### GraphRAG Scripts
| Path | Description |
|------|-------------|
| `scripts/archive/graphrag/build_proper_graphrag.py` | Build proper GraphRAG structure |
| `scripts/archive/graphrag/build_microsoft_graphrag.py` | Microsoft GraphRAG format builder |
| `scripts/archive/graphrag/audit_graphrag_hierarchy.py` | Audit hierarchy structure |
| `scripts/archive/layout_experiments/generate_graphsage_layout.py` | Generate GraphSAGE layout |

---

### Core Library Code

#### Extractors (`src/knowledge_graph/extractors/`)
| Path | Description |
|------|-------------|
| `entity_extractor.py` | **Core entity extraction** - LLM prompts, Pydantic schemas |
| `relationship_extractor.py` | **Relationship extraction** between entities |
| `ontology.py` | Entity type definitions and ontology |
| `chunking.py` | Text chunking for extraction |
| `__init__.py` | Module exports |

#### Validators (`src/knowledge_graph/validators/`)
| Path | Description |
|------|-------------|
| `entity_merge_validator.py` | **Entity merge validation** - fuzzy matching, blocklists, type checking |
| `__init__.py` | Module exports |

#### Postprocessing - Universal (`src/knowledge_graph/postprocessing/universal/`)

**Active Modules (per README.md):**
| Path | Priority | Description |
|------|----------|-------------|
| `vague_entity_blocker.py` | 85 | Filters overly vague entities (this, that, it); runs after ContextEnricher |
| `list_splitter.py` | 40 | Splits list targets ("A, B, and C") |
| `context_enricher.py` | 50 | Replaces vague entities with context |
| `pronoun_resolver.py` | 60 | Resolves pronouns to antecedents |
| `predicate_normalizer.py` | 70 | Normalizes verbose predicates |
| `predicate_validator.py` | 80 | Validates predicate logic |

**Additional Files Present (not in active pipeline):**
| Path | Description |
|------|-------------|
| `deduplicator.py` | Basic entity deduplication |
| `semantic_deduplicator.py` | Semantic-aware deduplication |
| `entity_deduplicator.py` | Entity-specific deduplication |
| `entity_resolver.py` | Entity resolution to canonical forms |
| `field_normalizer.py` | Field value normalization |
| `vague_demographic_reclassifier.py` | Reclassify vague demographic entities |
| `type_compatibility_validator.py` | Validate entity type compatibility |
| `generic_isa_filter.py` | Filter generic IS-A relationships |
| `confidence_filter.py` | Filter by confidence scores |
| `claim_classifier.py` | Classify claims/relationships |
| `market_stat_normalizer.py` | Normalize market statistics |
| `__init__.py` | Module exports |

**Note:** Many modules exist in the codebase but are NOT part of the active pipelines defined in `pipelines/`. This is a key gap.

#### Postprocessing - Book-Specific (`src/knowledge_graph/postprocessing/content_specific/books/`)
| Path | Description |
|------|-------------|
| `rhetorical_reclassifier.py` | Reclassify rhetorical entities |
| `title_completeness_validator.py` | Validate book title completeness |
| `statement_conciseness_normalizer.py` | Normalize statement conciseness |
| `subjective_content_filter.py` | Filter subjective content |
| `subtitle_joiner.py` | Join subtitles properly |
| `bibliographic_citation_parser.py` | Parse bibliographic citations |
| `dedication_normalizer.py` | Normalize dedication text |
| `praise_quote_detector.py` | Detect praise quotes |
| `figurative_language_filter.py` | Filter figurative language |
| `front_matter_detector.py` | Detect front matter content |
| `author_placeholder_resolver.py` | Resolve author placeholders |
| `metadata_filter.py` | Filter metadata entities |
| `narrative_filter.py` | Filter narrative-only entities |
| `__init__.py` | Module exports |

#### Postprocessing - Pipelines (`src/knowledge_graph/postprocessing/pipelines/`)
| Path | Description |
|------|-------------|
| `book_pipeline.py` | Complete book postprocessing pipeline |
| `podcast_pipeline.py` | Complete podcast postprocessing pipeline |
| `custom_pipeline.py` | Custom pipeline configuration |
| `__init__.py` | Module exports |

#### Postprocessing - Core (`src/knowledge_graph/postprocessing/`)
| Path | Description |
|------|-------------|
| `base.py` | Base postprocessor class |
| `discourse_assembler.py` | Assemble discourse graph |
| `__init__.py` | Module exports |

#### Graph Building (`src/knowledge_graph/graph/`)
| Path | Description |
|------|-------------|
| `graph_builder.py` | Build graph structure |
| `neo4j_client.py` | Neo4j database client |
| `graph_queries.py` | Graph query utilities |
| `__init__.py` | Module exports |

#### Visualization (`src/knowledge_graph/visualization/`)
| Path | Description |
|------|-------------|
| `export_visualization.py` | Export graph for visualization |
| `__init__.py` | Module exports |

#### Wiki Generation (`src/knowledge_graph/wiki/`)
| Path | Description |
|------|-------------|
| `wiki_builder.py` | Build wiki from knowledge graph |
| `markdown_generator.py` | Generate markdown wiki pages |
| `__init__.py` | Module exports |

#### Core Module Files (`src/knowledge_graph/`)
| Path | Description |
|------|-------------|
| `build_graph.py` | Main graph building entry point |
| `unified_builder.py` | Unified graph builder |
| `ontology.py` | Core ontology definitions |
| `validators.py` | Legacy validators |
| `demo_queries.py` | Demo query examples |
| `__init__.py` | Module exports |

---

### Output Data Files

#### Raw Extraction Outputs (`data/knowledge_graph/entities/`)
| Path | Description |
|------|-------------|
| `book_veriditas_extraction.json` | VIRIDITAS raw entity extraction |
| `book_y_on_earth_extraction.json` | Y on Earth raw entity extraction |
| `book_soil_stewardship_handbook_extraction.json` | Soil Stewardship raw extraction |
| `book_our_biggest_deal_extraction.json` | Our Biggest Deal raw extraction |

#### ACE Book Extractions (`data/knowledge_graph/books/`)
| Path | Description |
|------|-------------|
| `veriditas_ace_v14_3_8.json` | VIRIDITAS ACE extraction v14.3.8 |
| `veriditas_ace_v14_3_8_cleaned.json` | Cleaned version |
| `veriditas_ace_v14_3_8_improved.json` | Improved version |
| `y-on-earth_ace_v14_3_8.json` | Y on Earth ACE extraction |
| `y-on-earth_ace_v14_3_8_cleaned.json` | Cleaned version |
| `soil-stewardship-handbook_ace_v14_3_8.json` | Soil Stewardship ACE extraction |
| `soil-stewardship-handbook_ace_v14_3_8_cleaned.json` | Cleaned version |
| `OurBiggestDeal_ace_v14_3_8.json` | Our Biggest Deal ACE extraction |
| `OurBiggestDeal_ace_v14_3_8_cleaned.json` | Cleaned version |
| `our_biggest_deal_ace_v14_3_10/` | V14.3.10 extraction directory |
| `our_biggest_deal_ace_v14_3_10/chapters/` | Per-chapter extractions |
| `our_biggest_deal_ace_v14_3_10/analysis/` | Chapter analysis/reflections |
| `our_biggest_deal_ace_v14_3_10/manifests/` | Execution manifests |
| `our_biggest_deal_ace_v14_3_10/status.json` | Extraction status |

#### Unified Graph Outputs (`data/knowledge_graph_unified/`)
| Path | Description |
|------|-------------|
| `unified.json` | Original unified graph - **contains legacy bad merges from entity_normalization_v2** |
| `unified_normalized.json` | Normalized/cleaned version - **use this for analysis** (legacy merges largely cleaned; remaining quality issues persist—see baseline metrics) |
| `unified_hybrid.json` | Hybrid version combining multiple approaches |
| `adjacency.json` | Graph adjacency structure |
| `adjacency_normalized.json` | Normalized adjacency |
| `adjacency_with_cross_links.json` | Adjacency with cross-content links |
| `entity_merges.json` | Record of entity merges performed |
| `stats.json` | Graph statistics |
| `stats_normalized.json` | Normalized statistics |
| `visualization_data.json` | Data formatted for visualization |
| `discourse.json` | Discourse graph format |
| `discourse_graph_hybrid.json` | Hybrid discourse graph |
| `episode_discourse.json` | Episode-specific discourse |
| `cross_content_links.json` | Links between different content types |
| `cross_links_stats.json` | Cross-link statistics |
| `pronoun_resolution_log.json` | Log of pronoun resolutions |
| `orphan_triage_report.json` | Report on orphan entities |
| `normalization_verification.json` | Verification of normalization |

#### Postprocessed Episodes (`data/knowledge_graph_unified/episodes_postprocessed/`)
| Path | Description |
|------|-------------|
| `episode_110_post.json` - `episode_150_post.json` | 41 postprocessed episode files |
| Format | ACE-processed with relationships, confidence scores, classification flags |

#### Build History (`data/knowledge_graph_unified/builds/`)
| Path | Description |
|------|-------------|
| `build_20251119_010113_67c2bc1/` | Build snapshot with all outputs |
| Contains | `unified.json`, `adjacency.json`, `stats.json`, `entity_merges.json`, etc. |

#### Backups (`data/knowledge_graph_unified/backups/`)
| Path | Description |
|------|-------------|
| `unified_before_fix_20251119_201056.json` | Pre-fix backup |
| `adjacency_before_fix_20251119_201056.json` | Pre-fix adjacency |
| `entity_merges_before_fix_20251119_201056.json` | Pre-fix merges |
| `unified_normalized_backup_20251121_095259.json` | Normalized backup |
| `unified_before_books_20251121_172146.json` | Before book integration |
| `unified_normalized_backup_20251121_184824.json` | Another normalized backup |

#### Global Backups (`data/backups/`)
| Path | Description |
|------|-------------|
| `kg_backup_20251120_225449/` | Knowledge graph backup |
| `kg_backup_20251120_225451/` | Knowledge graph backup (9 subdirectories) |

---

### GraphRAG Hierarchy Data (`data/graphrag_hierarchy/`)
| Path | Description |
|------|-------------|
| `graphrag_hierarchy.json` | **Main hierarchy file** for 3D visualization |
| `graphrag_hierarchy_microsoft.json` | Microsoft GraphRAG format |
| `graphrag_hierarchy_v4.json` | Version 4 hierarchy |
| `graphrag_hierarchy_fixed.json` | Fixed hierarchy |
| `graphrag_hierarchy_fixed_v2.json` | Fixed version 2 |
| `graphrag_categories.json` | Category definitions |
| `cluster_search_index.json` | Search index for clusters |
| `cluster_registry.json` | Cluster registry |
| `cluster_registry_remapped.json` | Remapped cluster registry |
| `community_id_mapping.json` | Community ID mappings |
| `graphsage_layout.json` | GraphSAGE 2D layout |
| `graphsage_layout_3d.json` | GraphSAGE 3D layout |
| `graphsage_layout_cosine.json` | Cosine-based layout |
| `graphsage_entity_ids.json` | Entity ID mappings |
| `force_layout.json` | Force-directed layout |
| `voronoi_2_layout.json` | Voronoi layout v2 |
| `constrained_voronoi_layout.json` | Constrained Voronoi |
| `voronoi4_hierarchy.json` | Voronoi v4 hierarchy |
| `voronoi5_hierarchy.json` | Voronoi v5 hierarchy |
| `voronoi5_strict_treemap.json` | Voronoi treemap |
| `hierarchical_voronoi.json` | Hierarchical Voronoi |
| Various `*_backup*.json` | Backup files |

---

### Documentation Files

| Path | Description |
|------|-------------|
| `docs/KNOWLEDGE_GRAPH_EXTRACTION_REVIEW.md` | **This document** - comprehensive review |
| `docs/KNOWLEDGE_GRAPH_REGENERATION_PLAN.md` | Regeneration planning document |
| `docs/ACE_FRAMEWORK_DESIGN.md` | ACE framework design documentation |
| `docs/ACE_KG_EXTRACTION_VISION.md` | Vision for ACE-based extraction |
| `docs/ACE_CYCLE_1_COMPLETE.md` | ACE Cycle 1 completion notes |
| `docs/GRAPHRAG_3D_EMBEDDING_VIEW.md` | 3D embedding visualization docs |
| `docs/GRAPHRAG_PRODUCTION_SAFEGUARDS.md` | Production safeguards |
| `docs/GRAPHRAG_2D_ORGANIC_VIEWS_IMPLEMENTATION.md` | 2D organic views implementation |
| `docs/GRAPHRAG_TRI_MODE_IMPLEMENTATION.md` | Tri-mode implementation |
| `docs/IMPLEMENTATION_STATUS.md` | Overall implementation status |
| `docs/IMPLEMENTATION_PLAN.md` | Implementation planning |
| `docs/CONTENT_PROCESSING_PIPELINE.md` | Content processing pipeline docs |

---

### Configuration & Dependencies

| Path | Description |
|------|-------------|
| `.env` | Environment variables (API keys, settings) |
| `requirements.txt` | Python dependencies |
| `src/config/settings.py` | Centralized configuration |

---

### Key Statistics

| Metric | Value |
|--------|-------|
| Episode transcript files | 172 |
| Book source directories | 4 |
| Extraction scripts | 5 |
| Graph building scripts | 10 |
| Core library modules | 55+ |
| Postprocessing modules | 25+ |
| Output JSON files | 50+ |
| GraphRAG hierarchy files | 20+ |
| Documentation files | 12 |
| Postprocessed episode files | 41 |
| ACE book extraction files | 12+ |

---

## Current Data Quality Metrics (Baseline)

**Source:** `data/knowledge_graph_unified/unified_normalized.json` (analyzed December 3, 2025)

### Summary Statistics
| Metric | Value |
|--------|-------|
| Total Entities | 17,827 |
| Total Relationships | 19,331 |
| Episodes Postprocessed | **41 of 172** (24% coverage) |

### Critical Quality Issues

| Issue | Count | Severity | Description |
|-------|-------|----------|-------------|
| Missing relationship targets | 981 | High | Relationships pointing to non-existent entities |
| Missing relationship sources | 32 | Medium | Relationships from non-existent entities |
| Long entity names (>8 tokens) | 161 | Medium | Sentence-like entity names |
| Lowercase ORGANIZATION names | 596 | Medium | Generic orgs like "corporate executives", "humans" |
| PERSON with and/with/comma | 91 | High | Compound names not split (e.g., "Macy, Joanna with Chris Johnstone") |
| PERSON starting with determiner | 66 | High | Generic refs like "the character", "a Catholic mother" |
| URL-like entity names | 77 | Medium | URLs typed as CONCEPT/ORG instead of normalized |

### Additional Issues Identified by Third-Party Review (NEW)

| Issue | Count | Severity | Description |
|-------|-------|----------|-------------|
| **Pronoun hub entities** | 6 entities, 221 relationships | **Critical** | "we" (90), "I" (56), "she" (43), etc. are top hubs |
| **`people` as entity** | 1 entity, 207 relationships | **Critical** | #2 most connected entity in entire graph |
| **Fictional character pollution** | 5+ entities, 800+ relationships | **Critical** | Leo (#1), Sophia (#3), Brigitte (#6) dominate graph |
| **List entities (unsplit)** | 198 | High | "United States, China, France, Brazil" as single entity |
| **"Other X" entities** | 37 | Medium | "other slugs", "other farmers" as entities |
| **Determiner + noun entities** | 111 | Medium | "most jobs", "new concepts", "many more thousands" |
| **Numeric-only entities** | 16 | Medium | "2030", "35", "2094" as entities |
| **Tautological types** | 222 | Medium | Entity name = entity type (e.g., "organization" → ORGANIZATION) |
| **Common noun PERSONs** | 5+ | High | "mom", "friend", "individual", "farmers" typed as PERSON |

### Top Hub Entities (Data Quality Concern)

The most connected entities reveal significant quality issues:

| Rank | Entity | Type | Relationships | Quality Assessment |
|------|--------|------|---------------|-------------------|
| 1 | `Leo` | CHARACTER | 237 | ⚠️ **Fictional** - from "Our Biggest Deal" |
| 2 | `people` | ORGANIZATION | 207 | ❌ **Garbage** - generic common noun |
| 3 | `Sophia` | TECHNOLOGY | 204 | ⚠️ **Fictional** - from "Our Biggest Deal" |
| 4 | `Aaron William Perry` | PERSON | 198 | ✅ Valid - podcast host |
| 5 | `Y on Earth community` | COMMUNITY | 170 | ✅ Valid - core organization |
| 6 | `Brigitte` | PERSON | 157 | ⚠️ **Fictional** - from "Our Biggest Deal" |
| 7 | `Nathan Stuck` | PERSON | 139 | ✅ Valid - real person |
| 8 | `OTTO` | TECHNOLOGY | 117 | ⚠️ **Fictional** - from "Our Biggest Deal" |
| 9 | `Biochar` | PRODUCT | 104 | ✅ Valid - key concept |
| 10 | `community` | CONCEPT | 100 | ⚠️ Borderline - generic concept |
| 11 | `individual` | PERSON | 99 | ❌ **Garbage** - generic noun |
| 12 | `farmers` | PERSON | 99 | ❌ **Garbage** - generic occupation |
| 13 | `we` | ORGANIZATION | 90 | ❌ **Garbage** - pronoun |

**Summary**: Of the top 13 hub entities, only 4 (31%) are clearly valid. 5 are fictional characters (38%), and 4 are garbage entities (31%).

### Coverage Gap

**Critical Issue**: Only **41 of 172 episodes** (24%) have been processed through the ACE postprocessing pipeline.

The episodes in `data/knowledge_graph_unified/episodes_postprocessed/` are episodes 110-150. This means:
- Episodes 0-109: **Not postprocessed**
- Episodes 151-172: **Not postprocessed**
- Episode 26: Does not exist in series

This is a major root cause of data quality issues - most episode data bypassed quality filtering.

---

## Error Categories Found

### 1. Entity Duplication - Organization Name Variants

**Severity**: High
**Frequency**: 50+ variants for Y on Earth alone

**Examples Found**:
```
Y on Earth
Y on earth.org
Y on Earth.org
whyonearth.org
Y on Earth team
Y on Earth Foundation
YonEarth.org-support-page
Y on Earth community
The Y on Earth Community
Y on Earth Communities
Y on Earth Community Stewardship and Sustainability Podcast
Y on Earth Community Stewardship and Sustainability podcast
Y on Earth community stewardship and sustainability podcast series
```

**Impact**: The same organization appears as 50+ separate nodes in the graph, fragmenting relationships and making queries unreliable.

---

### 2. Relationship Endpoint Integrity

**Severity**: High  
**Frequency**: 981 relationships missing targets; 32 missing sources

**Example**:
- `People from poorest countries —stood in line→ for eight to ten hours` (target entity does not exist)

**Impact**: Missing endpoints break downstream graph queries and analytics; indicates insufficient validation before ingest.

---

### 3. Sentence/Phrase Entities (Contextless Extractions)

**Severity**: High
**Frequency**: Hundreds of instances

**Examples Found**:
| Entity Name | Type | Problem |
|-------------|------|---------|
| `People from poorest countries` | PERSON | Describes a group, not a named individual |
| `a Catholic mother and a Jewish father` | PERSON | Descriptive phrase, not a person's name |
| `the character` | PERSON | Generic pronoun reference with many relationships |
| `ebooks and audio book version of Y on Earth` | ENTITY | Product description, not entity name |
| `digital marketing strategy and overall online presence for the Y on Earth brand` | ENTITY | Full sentence fragment |

**Impact**: These non-entities create noise in the graph and generate meaningless relationships like `"the character" → ADVOCATES_FOR → "sustainability"`.

---

### 4. Incorrect Merges - Unrelated People Combined

**Severity**: Critical
**Frequency**: Multiple instances found

**Example**:
```json
"Dr. Ralph Sorenson": {
  "type": "PERSON",
  "description": "Director of Whole Foods Inc. and President Emeritus of Babson College",
  "aliases": [
    "James Featherby",
    "Jandel Allen-Davis",
    "Brad Corrigan"
  ],
  "provenance": [
    {"method": "structural_similarity_1.00", "merged_from": "Brad Corrigan"},
    {"method": "structural_similarity_1.00", "merged_from": "Jandel Allen-Davis"}
  ]
}
```
**Note:** This bad merge appears in `unified.json` but not in `unified_normalized.json` (which has been partially cleaned).

**Impact**: Four completely different people merged into one entity, corrupting all their relationships.

---

### 5. Compound Name Splitting Failures

**Severity**: Medium
**Frequency**: Dozens of instances

**Examples**:
| Original Text | Should Be | What Happened |
|---------------|-----------|---------------|
| `Macy, Joanna with Chris Johnstone` | Two entities: "Joanna Macy", "Chris Johnstone" | Single malformed entity |
| `Adam and Eve` | Two entities (or single CONCEPT) | Merged into "Adam" as PERSON |
| `Adam and Earth Coast Productions` | "Adam" (PERSON) + "Earth Coast Productions" (ORG) | Merged into "Adam" |

**Provenance shows**:
```json
{
  "source": "entity_normalization_v2",
  "merged_from": "Adam and Eve",
  "method": "compound_primary_minor_secondary"
}
```

**Impact**: Relationships incorrectly attributed to one person instead of multiple.

---

### 6. Generic/Collective Terms as PERSON Entities

**Severity**: Medium
**Frequency**: Common pattern

**Examples Found**:
| Entity Name | Assigned Type | Should Be |
|-------------|---------------|-----------|
| `government officials` | PERSON | Should not be an entity (generic reference) |
| `spiritual teachers` | PERSON | Should not be an entity |
| `religious friends` | PERSON | Should not be an entity |
| `environmentally-oriented friends` | PERSON | Should not be an entity |
| `future generations` | CONCEPT | Correct, but merged with "older generations" |

**Impact**: Generic references create entities that can't be meaningfully queried or connected.

---

### 6a. Pronouns as Major Hub Entities (NEW)

**Severity**: Critical
**Frequency**: Top 10 hub nodes in entire graph

**Third-party review revealed that unresolved pronouns have become some of the most connected nodes in the graph:**

| Entity | Type | Relationship Count | Impact |
|--------|------|-------------------|--------|
| `we` | ORGANIZATION | 90 relationships | Aggregates unrelated statements from different speakers |
| `I` | PERSON | 56 relationships | First-person references across all episodes |
| `she` | PERSON | 43 relationships | Relationships like "she → told → the pope" |
| `you` | CONCEPT | 21 relationships | Generic second-person references |
| `he` | PERSON | 7 relationships | Third-person references |
| `they` | CONCEPT | 4 relationships | Plural third-person references |

**Impact**: These pronoun entities create entirely meaningless hub nodes that conflate unrelated speakers and contexts into single entities.

---

### 6b. Common Nouns Masquerading as Named Entities (NEW)

**Severity**: High
**Frequency**: Pervasive pattern - 207+ relationships for top offender

**Third-party review identified lowercase, generic common nouns being extracted as named entities:**

| Entity | Type | Relationship Count | Problem |
|--------|------|-------------------|---------|
| `people` | ORGANIZATION | **207** (#2 Hub) | Generic collective reference |
| `mom` | PERSON | Multiple | Generic familial reference |
| `friend` | PERSON | Multiple | Generic social reference |
| `Woman` | PERSON | Multiple | Generic gender reference |
| `individual` | PERSON | 99 | Generic singular reference |
| `farmers` | PERSON | 99 | Generic occupational reference |

**"Other X" Pattern (37 instances)**:
- `other slugs` (ORGANISM)
- `other farmers`
- `other organizations`
- `other teenagers`

**Determiner + Noun Pattern (111 instances)**:
- `most jobs` (CONCEPT)
- `new concepts` (CONCEPTS)
- `most people`
- `many more thousands worldwide`
- `new generation of farmers`

**Impact**: These generic entities create noise and meaningless relationships that pollute graph queries.

---

### 6c. "List" Entities - Unsplit Clusters (NEW)

**Severity**: High
**Frequency**: 198 instances found

**Third-party review found entire lists extracted as single entities instead of being split:**

| Entity Name | Type | Problem |
|-------------|------|---------|
| `United States, China, France, Brazil` | COUNTRY | Four countries treated as one node |
| `Albert Einstein, Richard Nixon, Eisenhower` | (various) | Three people as single entity |
| `true Bronner, David Bronner, Mike Bronner` | (various) | Three people as single entity |
| `Glasgow, Paris, Copenhagen` | (various) | Three cities as single entity |
| `permaculture classes, events, and workshops` | (various) | List of offerings as single entity |

**Impact**: These list entities create false connections - a relationship to "United States, China, France, Brazil" incorrectly implies all four countries share that relationship.

---

### 6d. Narrative/Fictional Character Pollution (NEW)

**Severity**: Critical
**Frequency**: Top 3 most connected entities

**Third-party review identified that characters from "Our Biggest Deal" (a book) dominate the graph as fictional entities:**

| Entity | Type | Relationship Count | Graph Rank |
|--------|------|-------------------|------------|
| `Leo` | CHARACTER | **237** | #1 Hub |
| `Sophia` | TECHNOLOGY | **204** | #3 Hub |
| `Brigitte` | PERSON | **157** | #6 Hub |
| `OTTO` | TECHNOLOGY | **117** | #8 Hub |
| `MAMA-GAIA` | ECOSYSTEM | **83** | #17 Hub |

**Risk Assessment**:
- If the Knowledge Graph is intended for real-world facts (regenerative agriculture, sustainability), these narrative characters create a massive "fictional cluster"
- RAG responses may hallucinate fictional characters as real people
- Relationships like "Leo → advocates → sustainability" conflate fiction with fact

**Remediation Options**:
1. **Isolation**: Type these as `FICTIONAL_CHARACTER` to prevent merging with real-world concepts
2. **Exclusion**: Remove from unified graph or create separate "narrative graph"
3. **Tagging**: Add `is_fictional: true` flag for downstream filtering

---

### 6e. Numeric & Tautological Garbage (NEW)

**Severity**: Medium
**Frequency**: 16 numeric entities, 222 tautological types

**Numeric-Only Entities**:
| Entity | Type | Problem |
|--------|------|---------|
| `2030` | EVENT | Year as entity |
| `2094` | EVENT | Year as entity |
| `35` | CONCEPT | Number as entity |
| `2021` | CONCEPT | Year as entity |
| `2018`, `2020`, `1956`, `1924` | DATE | Years extracted as entities |

**Tautological Typing** (entity name ≈ entity type):
| Entity | Assigned Type | Problem |
|--------|---------------|---------|
| `organization` | ORGANIZATION | Tautology |
| `places` | PLACE | Tautology |
| `species` | SPECIES | Tautology |
| `chemicals` | CHEMICAL | Tautology |
| `fossil fuels` | FOSSIL FUELS | Custom type matching name |

**Impact**: Numeric entities are meaningless nodes. Tautological typing indicates extraction of generic concepts rather than specific named entities.

---

### 7. Wrong Entity Type Assignments

**Severity**: Medium
**Frequency**: Systemic issue

**Examples**:
| Entity | Original Type | Normalized To | Should Be |
|--------|---------------|---------------|-----------|
| `Hebrew` | LANGUAGE | CONCEPT | LANGUAGE |
| `Huns` | - | PERSON | ETHNIC_GROUP |
| `Goths` | - | PERSON | ETHNIC_GROUP |
| `Carantanians` | - | PERSON | ETHNIC_GROUP |
| `Mohawk people` | - | PERSON | INDIGENOUS_GROUP |
| `Thembu tribe` | - | PERSON | INDIGENOUS_GROUP |

**Root Issue**: The `TYPE_NORMALIZATION` dictionary in `build_unified_graph_hybrid.py` converts everything to a limited set of types, losing important distinctions.

---

### 8. Over-Aggressive Transcription Error Merges

**Severity**: High
**Frequency**: Hundreds of incorrect merges

**Examples from provenance data**:
| Original Entity | Merged Into | Method | Problem |
|-----------------|-------------|--------|---------|
| `mood` | `food` | `transcription_error_0.75` | Different concepts |
| `floods` | `food` | `transcription_error_0.80` | Different concepts |
| `future revelations` | `future generations` | `transcription_error_0.83` | Different concepts |
| `older generations` | `future generations` | `transcription_error_0.74` | Opposite meanings! |
| `Country` | `Community` | `transcription_error_0.75` | Different concepts |
| `commune` | `Community` | `transcription_error_0.75` | Different concepts |

**Impact**: Semantically distinct concepts merged, corrupting the knowledge graph's accuracy.

---

### 9. Very Long Phrase Entities

**Severity**: Low
**Frequency**: Dozens of instances

**Examples**:
```
"the most important thing is to make the benefits of our agricultural preparations
 available to the largest possible areas over the entire earth"

"invoicing, depositing money, building, running reports, reviewing reports,
 budgeting and planning process"

"source identified and regeneratively grown produce, meat, dairy and value
 added food products"

"biodynamically grown CBD hemp infused aromatherapy soaking salts"
```

**Impact**: These are descriptions or lists, not entities. They clutter the graph without adding value.

---

## Root Cause Analysis

### Extraction Phase Issues

#### 1. Overly Permissive Prompt

**Location**: `src/knowledge_graph/extractors/entity_extractor.py:61-106`

The extraction prompt instructs:
> "Extract ALL significant entities from this text"

This encourages over-extraction without clear boundaries on what should NOT be extracted.

**Missing guidance**:
- No instruction to avoid generic group references
- No instruction to split compound names
- No character/word limits on entity names
- No validation that PERSON entities are actual named individuals

#### 2. No Entity Validation Schema

The Pydantic schema (`EntityForExtraction`) only validates structure:
```python
class EntityForExtraction(BaseModel):
    name: str           # No length limit
    type: str           # No enum validation
    description: str    # No constraints
    aliases: List[str]  # No constraints
```

It doesn't validate semantic correctness (e.g., "Is this name a plausible person name?").

#### 3. Per-Chunk Extraction Without Context

Each 800-token chunk is processed independently:
- Same entity extracted with different surface forms in different chunks
- No awareness of previously extracted entities
- No guidance to use consistent naming

#### 4. No Context Window for Pronouns

Entities like "the character" are extracted because within a small chunk, they seem significant. Without document-level context, there's no way to resolve them.

---

### Post-Processing Issues

#### 1. Incomplete Episode Coverage (Critical)

Only **41 of 172 episodes** (24%) have been processed through the ACE postprocessing pipeline. The postprocessed episodes are in `data/knowledge_graph_unified/episodes_postprocessed/` (episodes 110-150 only).

This means 76% of episode data entered the unified graph without:
- Pronoun resolution
- List splitting
- Vague entity blocking
- Predicate normalization
- Context enrichment

#### 2. Legacy Normalization Data (`entity_normalization_v2`)

The `unified.json` file (not `unified_normalized.json`) contains provenance showing merge methods that don't exist in the current codebase:
- `structural_similarity_1.00`
- `transcription_error_0.XX`
- `compound_primary_minor_secondary`
- `possessive_generic`

These were from a previous LLM-based normalization pass that made aggressive (and often incorrect) merge decisions. The bad merges persist in `unified.json` but appear to have been cleaned in `unified_normalized.json`.

**Note:** The "Dr. Ralph Sorenson" merge example exists in `unified.json` but not in `unified_normalized.json`, indicating some cleanup occurred.

#### 3. String Similarity Limitations

The current `EntityMergeValidator` uses fuzzy string matching (fuzzywuzzy):
```python
char_score = fuzz.ratio(norm1, norm2)
if char_score >= 95:
    # Merge approved
```

This cannot understand:
- Semantic equivalence (`whyonearth.org` = `Y on Earth Foundation`)
- That similar strings can be different entities (`mood` ≠ `food`)

#### 3. Type-Based Deduplication Silos

Deduplication groups entities by type first:
```python
for entity_type, entity_list in entities_by_type.items():
    # Only compares within same type
```

This means `Y on Earth` (PRODUCT) won't be compared with `Y on Earth Community` (ORGANIZATION), even though they should potentially be merged or linked.

#### 5. Inactive Postprocessing Modules

Many postprocessing modules exist in `src/knowledge_graph/postprocessing/universal/` but are **NOT included** in the active pipelines:
- `deduplicator.py` - Not in active pipeline
- `semantic_deduplicator.py` - Not in active pipeline
- `entity_resolver.py` - Not in active pipeline
- `confidence_filter.py` - Not in active pipeline
- `field_normalizer.py` - Not in active pipeline
- `type_compatibility_validator.py` - Not in active pipeline

These modules could address many of the quality issues but need to be integrated into the pipelines.

---

### Missing Capabilities

| Capability | Status | Impact |
|------------|--------|--------|
| **Full Episode Coverage** | **76% missing** | Only 41/172 episodes postprocessed |
| Entity Resolution/Linking | Missing | Can't identify that variants refer to same thing |
| Canonical Entity Registry | Missing | No ground truth to merge toward |
| Human Review Queue | Missing | Uncertain merges auto-approved or auto-rejected |
| Cross-Document Entity Tracking | Missing | Same entity extracted differently per source |
| Semantic Similarity Validation | Missing | String similarity is insufficient |
| Active Deduplication Pipeline | Inactive | `deduplicator.py`, `semantic_deduplicator.py` exist but not used |
| URL/Alias Normalization | Missing | 77 URL-like names not normalized to organizations |

---

## Proposed Improvement Plan

### Phase 1: Extraction Prompt Improvements

#### 1.1 Add Explicit Exclusion Rules

Update `src/knowledge_graph/extractors/entity_extractor.py`:

```python
EXTRACTION_PROMPT = """You are an expert at extracting structured entities from podcast transcripts about sustainability, regenerative agriculture, and environmental topics.

Extract significant entities from this text. For each entity, provide:
1. name: The canonical name (prefer full formal names)
2. type: One of {entity_types}
3. description: A concise description (1-2 sentences)
4. aliases: Alternative names or spellings mentioned

=== IMPORTANT: DO NOT EXTRACT ===

❌ Generic group references:
   - "spiritual teachers", "government officials", "our friends"
   - "farmers", "scientists", "activists" (unless naming a specific person)

❌ Pronouns or indefinite references:
   - "the character", "the speaker", "they", "he", "she"
   - "this person", "that organization"

❌ Descriptive phrases or sentence fragments:
   - "people who care about the environment"
   - "the most important thing is..."
   - Anything that reads like a sentence rather than a name

❌ Combined names (extract separately instead):
   - "John and Jane Smith" → Extract as TWO entities: "John Smith" and "Jane Smith"
   - "Joanna Macy with Chris Johnstone" → Extract as TWO entities

❌ Very long names:
   - Entity names should be under 6 words
   - If you need more words, it's probably a description, not an entity

❌ Website URLs as standalone entities:
   - Include URLs as aliases of the organization they belong to
   - "yonearth.org" is an alias for "Y on Earth Community", not a separate entity

=== ENTITY TYPE GUIDANCE ===

PERSON: Only specific named individuals
   ✓ "Aaron William Perry", "Dr. Jane Goodall", "Rowdy Yates"
   ✗ "farmers", "teachers", "friends", "the speaker"

ORGANIZATION: Formal organizations, companies, institutions
   ✓ "Y on Earth Community", "Natural Capitalism Solutions", "EPA"
   ✗ "local organizations", "faith communities" (too generic)

CONCEPT: Ideas, movements, philosophies
   ✓ "regenerative agriculture", "permaculture", "biochar"
   ✗ Full sentences or explanations

PLACE: Specific locations
   ✓ "Colorado", "Boulder", "Amazon Rainforest"
   ✗ "the farm", "around here" (too vague)

=== HANDLING DUPLICATES ===

If the same entity appears with different names, use the most formal/complete version:
- "Y on Earth", "YonEarth", "yonearth.org" → Use "Y on Earth Community"
- "Aaron Perry", "Aaron William Perry" → Use "Aaron William Perry"

Text to analyze:
{text}

Return ONLY the JSON array, no other text."""
```

#### 1.2 Add Entity Name Validation

Add a validation step after extraction:

```python
def validate_entity_name(name: str, entity_type: str) -> Tuple[bool, str]:
    """Validate entity name before inclusion"""

    # Length checks
    if len(name) > 80:
        return False, "name_too_long"
    if len(name.split()) > 8:
        return False, "too_many_words"

    # PERSON-specific validation
    if entity_type == "PERSON":
        # Reject generic references
        generic_patterns = [
            r'^(the |a |an |our |their |my )',
            r'(friends|teachers|officials|people|generations|character)$',
            r'^(who|which|that|those|these) ',
            r'\b(and|with)\b.*\b(and|with)\b',  # Multiple conjunctions
        ]
        for pattern in generic_patterns:
            if re.search(pattern, name.lower()):
                return False, f"generic_person: {pattern}"

        # Should have at least one capitalized word (name-like)
        if not any(word[0].isupper() for word in name.split() if word):
            return False, "no_capitalized_words"

    # Reject sentence-like structures
    sentence_indicators = [
        r'\b(is|are|was|were|has|have|had|will|would|could|should)\b',
        r'\b(the most|in order to|according to)\b',
        r'[.!?]',  # Sentence-ending punctuation
    ]
    for pattern in sentence_indicators:
        if re.search(pattern, name):
            return False, f"sentence_like: {pattern}"

    return True, "valid"
```

---

### Phase 2: Entity Resolution Pipeline

#### 2.1 Create Canonical Entity Registry

Create `data/canonical_entities.json`:

```json
{
  "version": "1.0.0",
  "updated": "2025-12-03",

  "organizations": {
    "y-on-earth": {
      "canonical_name": "Y on Earth Community",
      "type": "ORGANIZATION",
      "description": "Nonprofit community focused on sustainability, stewardship, and regeneration",
      "aliases": [
        "Y on Earth",
        "YonEarth",
        "Y on Earth Foundation",
        "Y on Earth team",
        "The Y on Earth Community",
        "Y on Earth Communities",
        "yonearth.org",
        "whyonearth.org",
        "Y on Earth.org",
        "Y on earth.org"
      ],
      "merge_patterns": [
        "y[\\s\\-]*on[\\s\\-]*earth",
        "why[\\s\\-]*on[\\s\\-]*earth",
        "yonearth"
      ],
      "related_entities": [
        "Y on Earth Community Stewardship and Sustainability Podcast",
        "Earth Water Press"
      ]
    },

    "earth-water-press": {
      "canonical_name": "Earth Water Press",
      "type": "ORGANIZATION",
      "description": "Publishing company focused on sustainability literature",
      "aliases": [
        "Earthwater Press",
        "earth water press"
      ]
    }
  },

  "people": {
    "aaron-perry": {
      "canonical_name": "Aaron William Perry",
      "type": "PERSON",
      "description": "Founder of Y on Earth Community, author, podcast host",
      "aliases": [
        "Aaron Perry",
        "Aaron W. Perry",
        "Aaron"
      ],
      "roles": ["host", "author", "founder"]
    },

    "joanna-macy": {
      "canonical_name": "Joanna Macy",
      "type": "PERSON",
      "description": "Environmental activist, author, scholar of Buddhism",
      "aliases": [
        "Joanna Rogers Macy",
        "Dr. Joanna Macy"
      ],
      "note": "Often mentioned with Chris Johnstone - these are TWO separate people"
    }
  },

  "products": {
    "y-on-earth-podcast": {
      "canonical_name": "Y on Earth Community Stewardship and Sustainability Podcast",
      "type": "PRODUCT",
      "description": "Podcast series featuring conversations on sustainability",
      "aliases": [
        "Y on Earth Podcast",
        "Y on Earth Community Podcast",
        "Y on Earth community podcast",
        "Y on Earth Communities Stewardship and Sustainability Podcast"
      ],
      "parent_organization": "y-on-earth"
    }
  }
}
```

#### 2.2 Implement Entity Resolver

Create `src/knowledge_graph/resolvers/entity_resolver.py`:

```python
"""
Entity Resolver - Maps extracted entities to canonical forms

This module resolves variant entity names to their canonical forms
using a combination of:
1. Exact alias matching
2. Regex pattern matching
3. Fuzzy string matching
4. Embedding similarity (for uncertain cases)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz


class EntityResolver:
    """Resolves entity names to canonical forms"""

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("data/canonical_entities.json")
        self.registry = self._load_registry()
        self._build_lookup_indices()

    def _load_registry(self) -> Dict:
        """Load canonical entity registry"""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"organizations": {}, "people": {}, "products": {}}

    def _build_lookup_indices(self):
        """Build fast lookup indices for resolution"""
        self.alias_to_canonical: Dict[str, str] = {}
        self.patterns: List[Tuple[re.Pattern, str]] = []

        for category in ["organizations", "people", "products"]:
            for entity_id, entity in self.registry.get(category, {}).items():
                canonical = entity["canonical_name"]

                # Index aliases (case-insensitive)
                for alias in entity.get("aliases", []):
                    self.alias_to_canonical[alias.lower()] = canonical
                self.alias_to_canonical[canonical.lower()] = canonical

                # Compile regex patterns
                for pattern in entity.get("merge_patterns", []):
                    try:
                        compiled = re.compile(pattern, re.IGNORECASE)
                        self.patterns.append((compiled, canonical))
                    except re.error:
                        pass

    def resolve(
        self,
        entity_name: str,
        entity_type: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """
        Resolve entity name to canonical form.

        Returns:
            Tuple of (resolved_name, confidence, method)
            - confidence: 0.0 to 1.0
            - method: "exact_alias", "pattern", "fuzzy", "unresolved"
        """
        if not entity_name:
            return entity_name, 0.0, "empty"

        name_lower = entity_name.lower().strip()

        # 1. Exact alias match
        if name_lower in self.alias_to_canonical:
            return self.alias_to_canonical[name_lower], 1.0, "exact_alias"

        # 2. Regex pattern match
        for pattern, canonical in self.patterns:
            if pattern.fullmatch(name_lower):
                return canonical, 0.95, "pattern_match"

        # 3. Fuzzy match against all aliases
        best_match = None
        best_score = 0

        for alias, canonical in self.alias_to_canonical.items():
            score = fuzz.ratio(name_lower, alias)
            if score > best_score and score >= 85:
                best_score = score
                best_match = canonical

        if best_match:
            return best_match, best_score / 100, "fuzzy_match"

        # 4. Unresolved - return original
        return entity_name, 0.0, "unresolved"

    def resolve_batch(
        self,
        entities: List[Dict]
    ) -> List[Dict]:
        """Resolve a batch of entities, updating names in place"""
        for entity in entities:
            original_name = entity.get("name", "")
            resolved_name, confidence, method = self.resolve(
                original_name,
                entity.get("type")
            )

            if resolved_name != original_name:
                # Keep original as alias
                aliases = entity.get("aliases", [])
                if original_name not in aliases:
                    aliases.append(original_name)
                entity["aliases"] = aliases

                # Update to canonical name
                entity["name"] = resolved_name
                entity["resolution"] = {
                    "original": original_name,
                    "method": method,
                    "confidence": confidence
                }

        return entities
```

---

### Phase 3: Multi-Pass Extraction with Verification

#### 3.1 Add Verification Pass

After initial extraction, run a verification pass to filter bad entities:

```python
def verify_entities_batch(
    entities: List[Dict],
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Verify extracted entities using LLM judgment.

    Asks the model to confirm each entity is a valid, specific entity
    rather than a generic reference or phrase.
    """

    VERIFICATION_PROMPT = """Review these extracted entities and mark each as VALID or INVALID.

An entity is VALID if:
- It's a specific, named thing (person, organization, place, concept)
- Someone could look it up or verify it exists
- It's not a generic group reference or description

An entity is INVALID if:
- It's a generic reference ("teachers", "friends", "the speaker")
- It's a sentence fragment or description
- It's too vague to be useful
- It contains multiple entities that should be separate

Entities to verify:
{entities_json}

Return JSON array with each entity and a "valid" boolean field:
[{{"name": "...", "type": "...", "valid": true/false, "reason": "..."}}]
"""

    # ... implementation ...
```

#### 3.2 Cross-Chunk Entity Awareness

Modify extraction to provide context about previously extracted entities:

```python
def extract_with_context(
    text: str,
    prior_entities: List[str],
    episode_context: str
) -> List[Dict]:
    """Extract entities with awareness of prior extractions"""

    context_prompt = f"""
You are extracting entities from a podcast transcript.

Episode: {episode_context}

Previously extracted entities from this episode:
{chr(10).join(f'- {e}' for e in prior_entities[:50])}

If you extract an entity that matches one above, use the SAME name.
For example, if "Aaron William Perry" was already extracted, don't extract "Aaron Perry" as a separate entity.

Text to analyze:
{text}
"""
    # ... continue with extraction ...
```

---

### Immediate Safeguards (Pre-Re-extraction)

1. **Harden `_should_filter_name()` in `scripts/build_unified_graph_hybrid.py`**
   - Add explicit patterns for clause-like/groupy names (e.g., `^people from`, `^the character`, `^a catholic mother`, `^the speaker`, `^our .*`, `^their .*`).
   - Keep the existing length/verb heuristics but fail closed for these prefixes to stop obvious noise before re-extraction.

2. **Mine `data/knowledge_graph_unified/entity_merges.json` for Blocklist Seeds**
   - Programmatically extract merged pairs with low semantic plausibility (e.g., Levenshtein-only merges across type or concept drift such as `mood`→`food`, `Adam and Eve`→`Adam`, `people`→`People from poorest countries`).
   - Use this to auto-populate/append the `SEMANTIC_BLOCKLIST` in `EntityMergeValidator` and to seed canonical alias maps, preserving type information where available.

3. **Intelligent Reuse of Existing Dedup Modules**
   - Before writing new code, audit and selectively integrate existing modules in `src/knowledge_graph/postprocessing/universal/` (e.g., `semantic_deduplicator.py`, `deduplicator.py`, `entity_resolver.py`) into the active pipelines.
   - Do this with guardrails: add tests, inspect module assumptions, and ensure type compatibility so we don’t reintroduce legacy aggressive merges.

---

### Phase 4: Post-Processing Validation

#### 4.1 Enhanced Entity Quality Filter

Add to `src/knowledge_graph/validators/entity_quality_filter.py`:

```python
"""
Entity Quality Filter

Filters out low-quality entities before graph building.
"""

import re
from typing import Dict, List, Tuple


class EntityQualityFilter:
    """Filter entities based on quality heuristics"""

    # Patterns that indicate generic/bad entities
    GENERIC_PERSON_PATTERNS = [
        r'^(the |a |an |our |their |my |your )',
        r'(friends|teachers|officials|people|generations|character|speaker)s?$',
        r'^(who|which|that|those|these|some|many|few|all) ',
        r'^(someone|anyone|everyone|nobody|somebody) ',
    ]

    SENTENCE_PATTERNS = [
        r'\b(is|are|was|were|has|have|had|will|would|could|should|can|may|might)\b',
        r'\b(the most|in order to|according to|in terms of|as well as)\b',
        r'[.!?;]',  # Sentence punctuation
        r',.*,.*,',  # Multiple commas (likely a list)
    ]

    # Maximum lengths
    MAX_NAME_LENGTH = 80
    MAX_WORD_COUNT = 8

    def __init__(self):
        self.stats = {
            "total_checked": 0,
            "filtered_out": 0,
            "reasons": {}
        }

    def filter_entity(self, entity: Dict) -> Tuple[bool, str]:
        """
        Check if entity passes quality filter.

        Returns:
            (passes_filter, reason)
        """
        self.stats["total_checked"] += 1

        name = entity.get("name", "")
        entity_type = entity.get("type", "")

        # Empty name
        if not name or not name.strip():
            return self._reject("empty_name")

        name = name.strip()

        # Length checks
        if len(name) > self.MAX_NAME_LENGTH:
            return self._reject("name_too_long")

        word_count = len(name.split())
        if word_count > self.MAX_WORD_COUNT:
            return self._reject("too_many_words")

        # PERSON-specific checks
        if entity_type == "PERSON":
            for pattern in self.GENERIC_PERSON_PATTERNS:
                if re.search(pattern, name.lower()):
                    return self._reject(f"generic_person")

            # Should have name-like structure (capitalized words)
            words = name.split()
            capitalized = sum(1 for w in words if w and w[0].isupper())
            if capitalized == 0:
                return self._reject("no_proper_nouns")

        # Sentence-like structure (any type)
        for pattern in self.SENTENCE_PATTERNS:
            if re.search(pattern, name):
                return self._reject("sentence_like")

        return True, "passed"

    def _reject(self, reason: str) -> Tuple[bool, str]:
        """Record rejection and return result"""
        self.stats["filtered_out"] += 1
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
        return False, reason

    def filter_batch(self, entities: List[Dict]) -> List[Dict]:
        """Filter a batch of entities, returning only those that pass"""
        return [e for e in entities if self.filter_entity(e)[0]]

    def get_stats(self) -> Dict:
        """Get filtering statistics"""
        return {
            **self.stats,
            "pass_rate": (
                (self.stats["total_checked"] - self.stats["filtered_out"])
                / self.stats["total_checked"]
                if self.stats["total_checked"] > 0 else 0
            )
        }
```

#### 4.2 Compound Name Splitter

Add compound name detection and splitting:

```python
def split_compound_entities(entities: List[Dict]) -> List[Dict]:
    """
    Split entities that contain multiple people/things.

    "Joanna Macy with Chris Johnstone" → Two entities
    "John and Jane Smith" → Two entities
    """

    COMPOUND_PATTERNS = [
        # "X with Y"
        (r'^(.+?)\s+with\s+(.+)$', ["with"]),
        # "X and Y" (but not "X and Y Organization")
        (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$', ["and"]),
        # "X, Y" for names
        (r'^([A-Z][a-z]+),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$', ["comma"]),
    ]

    result = []

    for entity in entities:
        if entity.get("type") != "PERSON":
            result.append(entity)
            continue

        name = entity.get("name", "")
        split_made = False

        for pattern, _ in COMPOUND_PATTERNS:
            match = re.match(pattern, name)
            if match:
                # Create two entities
                for i, group in enumerate(match.groups(), 1):
                    new_entity = entity.copy()
                    new_entity["name"] = group.strip()
                    new_entity["split_from"] = name
                    result.append(new_entity)
                split_made = True
                break

        if not split_made:
            result.append(entity)

    return result
```

---

### Phase 4.3: Heuristic Filtering Refinements (NEW)

**Added based on third-party review findings.**

This phase addresses specific garbage patterns identified in the current graph that were not explicitly caught by existing filters.

#### 4.3.1 Stop-Word Entity Blocker

Add explicit blocking for pronoun and common word entities:

```python
# Add to EntityQualityFilter or create new StopWordEntityBlocker

STOP_WORD_ENTITIES = {
    # Pronouns (should never be entities)
    'we', 'she', 'he', 'they', 'it', 'i', 'you',

    # Generic collective nouns
    'people', 'person', 'individual', 'individuals',
    'everyone', 'someone', 'anyone', 'nobody',

    # Generic familial/social references
    'mom', 'dad', 'mother', 'father', 'friend', 'friends',
    'guy', 'woman', 'man', 'kid', 'kids',

    # Generic occupational references (singular lowercase)
    'farmer', 'teacher', 'scientist', 'activist',
}

def is_stop_word_entity(name: str) -> bool:
    """Check if entity name is a stop word that should be blocked"""
    normalized = name.lower().strip()
    return normalized in STOP_WORD_ENTITIES
```

#### 4.3.2 Numeric-Only Entity Filter

Block entities that are purely numeric:

```python
import re

def is_numeric_entity(name: str) -> bool:
    """Check if entity name is purely numeric (years, numbers)"""
    # Match pure numbers: "2030", "35", "1956"
    if re.match(r'^\d+$', name.strip()):
        return True
    return False

# Usage: Filter out entities where is_numeric_entity(name) == True
```

#### 4.3.3 Lowercase Single-Word PERSON Filter

Block generic lowercase single-word PERSONs unless whitelisted:

```python
# Whitelist for legitimate single lowercase names (rare)
LOWERCASE_PERSON_WHITELIST = {
    # Add any legitimate single lowercase names here
}

def is_invalid_lowercase_person(name: str, entity_type: str) -> bool:
    """
    Check if entity is an invalid lowercase single-word PERSON.

    Returns True for entities like:
    - "mom" (PERSON)
    - "friend" (PERSON)
    - "guy" (PERSON)

    Returns False for:
    - "Aaron" (PERSON) - capitalized
    - "farmers" (PERSON) - already caught by stop words
    - "John Smith" (PERSON) - multi-word
    """
    if entity_type != "PERSON":
        return False

    words = name.split()
    if len(words) != 1:
        return False  # Multi-word, different rules apply

    word = words[0]

    # If it's all lowercase and not whitelisted, reject
    if word.islower() and word.lower() not in LOWERCASE_PERSON_WHITELIST:
        return True

    return False
```

#### 4.3.4 Narrative/Fictional Character Isolation

Isolate characters from "Our Biggest Deal" to prevent contamination:

```python
# Known fictional characters from "Our Biggest Deal"
OUR_BIGGEST_DEAL_CHARACTERS = {
    'Leo', 'Sophia', 'Brigitte', 'OTTO', 'MAMA-GAIA',
    'Leo von Übergarten', 'Lily Sophia von Übergarten',
    'Brigitte Sophia', 'Brigitte Sophia Miklavc von Übergarten',
    # Add other known characters
}

def tag_fictional_character(entity: Dict, source: str) -> Dict:
    """
    Tag entities from narrative sources as fictional.

    Options:
    1. Change type to FICTIONAL_CHARACTER
    2. Add is_fictional: true flag
    3. Add source_type: "narrative" flag
    """
    name = entity.get('name', '')

    # Check if from Our Biggest Deal
    if source == 'OurBiggestDeal' or name in OUR_BIGGEST_DEAL_CHARACTERS:
        entity['is_fictional'] = True
        entity['source_type'] = 'narrative'

        # Option: Change type
        if entity.get('type') in ['CHARACTER', 'PERSON']:
            entity['original_type'] = entity['type']
            entity['type'] = 'FICTIONAL_CHARACTER'

    return entity
```

#### 4.3.5 List Entity Detector

Detect and flag unsplit list entities:

```python
def is_list_entity(name: str) -> bool:
    """
    Detect entities that appear to be unsplit lists.

    Examples that should be split:
    - "United States, China, France, Brazil"
    - "Albert Einstein, Richard Nixon, Eisenhower"
    - "Glasgow, Paris, Copenhagen"
    """
    # Count commas
    comma_count = name.count(',')

    # If 2+ commas, likely a list
    if comma_count >= 2:
        return True

    # Check for "X, Y, and Z" pattern
    if re.match(r'.+,\s*.+,?\s*and\s+.+', name, re.IGNORECASE):
        return True

    return False

def split_list_entity(entity: Dict) -> List[Dict]:
    """
    Split a list entity into individual entities.

    "United States, China, France, Brazil" →
    [{"name": "United States"}, {"name": "China"},
     {"name": "France"}, {"name": "Brazil"}]
    """
    name = entity.get('name', '')

    # Split on commas and "and"
    # Handle "A, B, and C" and "A, B, C" patterns
    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', name)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        return [entity]  # Not a list, return as-is

    result = []
    for part in parts:
        new_entity = entity.copy()
        new_entity['name'] = part
        new_entity['split_from_list'] = name
        result.append(new_entity)

    return result
```

#### 4.3.6 Tautological Type Detector

Flag entities where the name matches the type:

```python
def is_tautological_entity(name: str, entity_type: str) -> bool:
    """
    Detect tautological entities where name ≈ type.

    Examples:
    - "organization" with type ORGANIZATION
    - "places" with type PLACE
    - "chemicals" with type CHEMICAL
    """
    name_normalized = name.lower().rstrip('s')  # Remove trailing 's'
    type_normalized = entity_type.lower().rstrip('s')

    # Direct match
    if name_normalized == type_normalized:
        return True

    # Check if name is just the type with minor variations
    if name_normalized.replace(' ', '_') == type_normalized.replace(' ', '_'):
        return True

    return False

# Usage: Filter out entities where is_tautological_entity(name, type) == True
```

#### 4.3.7 Integration into Pipeline

Add these filters to the existing pipeline in `build_unified_graph_hybrid.py`:

```python
def enhanced_entity_filter(entity: Dict, source: str = None) -> Tuple[bool, str]:
    """
    Enhanced entity filtering with all new heuristics.

    Returns:
        (should_include, rejection_reason)
    """
    name = entity.get('name', '')
    entity_type = entity.get('type', '')

    # 1. Stop word check
    if is_stop_word_entity(name):
        return False, "stop_word_entity"

    # 2. Numeric check
    if is_numeric_entity(name):
        return False, "numeric_only_entity"

    # 3. Lowercase single-word PERSON
    if is_invalid_lowercase_person(name, entity_type):
        return False, "invalid_lowercase_person"

    # 4. Tautological check
    if is_tautological_entity(name, entity_type):
        return False, "tautological_entity"

    # 5. List detection (split instead of filter)
    if is_list_entity(name):
        return False, "needs_list_split"  # Handle separately

    return True, "passed"
```

---

### Phase 5: Merge Validation Improvements

#### 5.1 Add Semantic Blocklist

Extend the blocklist in `EntityMergeValidator`:

```python
# Add to MERGE_BLOCKLIST
SEMANTIC_BLOCKLIST = [
    # Concepts that sound similar but are different
    ('mood', 'food'),
    ('floods', 'food'),
    ('future revelations', 'future generations'),
    ('older generations', 'future generations'),
    ('country', 'community'),
    ('commune', 'community'),

    # People who should not be merged
    ('joanna macy', 'chris johnstone'),

    # Organizations
    ('y on earth', 'earth water press'),  # Related but distinct
]
```

#### 5.2 Add Embedding-Based Validation

For merges in the uncertainty zone (60-90% string similarity), use embeddings:

```python
def validate_merge_semantic(
    entity1: Dict,
    entity2: Dict,
    embedding_model: str = "text-embedding-3-small"
) -> Tuple[bool, float]:
    """
    Validate merge using embedding similarity.

    Returns (should_merge, similarity_score)
    """
    from openai import OpenAI
    import numpy as np

    client = OpenAI()

    # Create descriptive text for each entity
    text1 = f"{entity1['name']}: {entity1.get('description', '')}"
    text2 = f"{entity2['name']}: {entity2.get('description', '')}"

    # Get embeddings
    response = client.embeddings.create(
        model=embedding_model,
        input=[text1, text2]
    )

    emb1 = np.array(response.data[0].embedding)
    emb2 = np.array(response.data[1].embedding)

    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Threshold for merge approval
    return similarity > 0.85, float(similarity)
```

---

### Phase 6: Re-Extraction Process

#### 6.1 Complete Re-Extraction Script

Create `scripts/reextract_knowledge_graph.py`:

```python
#!/usr/bin/env python3
"""
Complete Knowledge Graph Re-Extraction

This script performs a clean re-extraction of all knowledge graph data
with improved prompts, validation, and entity resolution.

Usage:
    python scripts/reextract_knowledge_graph.py --all
    python scripts/reextract_knowledge_graph.py --episodes-only
    python scripts/reextract_knowledge_graph.py --books-only
    python scripts/reextract_knowledge_graph.py --validate-only
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--episodes-only', action='store_true')
    parser.add_argument('--books-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Backup current data
    logger.info("=" * 60)
    logger.info("STEP 1: Backing up current knowledge graph")
    logger.info("=" * 60)
    backup_dir = Path(f"data/backups/kg_backup_{timestamp}")
    # ... backup implementation ...

    # Step 2: Extract episodes
    if args.all or args.episodes_only:
        logger.info("=" * 60)
        logger.info("STEP 2: Extracting from episodes")
        logger.info("=" * 60)
        # ... extraction with new prompts ...

    # Step 3: Extract books
    if args.all or args.books_only:
        logger.info("=" * 60)
        logger.info("STEP 3: Extracting from books")
        logger.info("=" * 60)
        # ... extraction ...

    # Step 4: Entity resolution
    logger.info("=" * 60)
    logger.info("STEP 4: Resolving entities to canonical forms")
    logger.info("=" * 60)
    # ... resolution ...

    # Step 5: Quality filtering
    logger.info("=" * 60)
    logger.info("STEP 5: Filtering low-quality entities")
    logger.info("=" * 60)
    # ... filtering ...

    # Step 6: Build unified graph
    logger.info("=" * 60)
    logger.info("STEP 6: Building unified graph")
    logger.info("=" * 60)
    # ... graph building ...

    # Step 7: Generate quality report
    logger.info("=" * 60)
    logger.info("STEP 7: Generating quality report")
    logger.info("=" * 60)
    # ... report generation ...

if __name__ == "__main__":
    main()
```

#### 6.2 Quality Report Generator

Create validation report after extraction:

```python
def generate_quality_report(unified_path: Path) -> Dict:
    """Generate comprehensive quality report for knowledge graph"""

    with open(unified_path) as f:
        data = json.load(f)

    entities = data.get("entities", {})
    relationships = data.get("relationships", [])

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
        },
        "entity_analysis": {
            "by_type": {},
            "long_names": [],  # > 50 chars
            "potential_duplicates": [],
            "generic_persons": [],
            "orphan_entities": [],  # No relationships
        },
        "relationship_analysis": {
            "by_type": {},
            "missing_source": [],
            "missing_target": [],
        },
        "recommendations": []
    }

    # ... analysis implementation ...

    return report
```

---

## Implementation Priority

### Immediate Priority (Before Re-Extraction)

| Task | Effort | Impact | File |
|------|--------|--------|------|
| Update extraction prompt with exclusions | Low | High | `entity_extractor.py` |
| Create canonical entity registry | Medium | High | `data/canonical_entities.json` |
| Add entity quality filter | Medium | High | New file |
| Add compound name splitter | Low | Medium | New file |
| **Add stop-word entity blocker (NEW)** | Low | **Critical** | `entity_quality_filter.py` |
| **Add numeric entity filter (NEW)** | Low | Medium | `entity_quality_filter.py` |
| **Add list entity splitter (NEW)** | Low | High | `entity_quality_filter.py` |

### Short-Term (During Re-Extraction)

| Task | Effort | Impact | File |
|------|--------|--------|------|
| Implement entity resolver | Medium | High | New file |
| Add semantic blocklist to validator | Low | Medium | `entity_merge_validator.py` |
| Create re-extraction script | Medium | High | New script |
| Create quality report generator | Medium | Medium | New script |
| **Tag/isolate fictional characters (NEW)** | Medium | **Critical** | `build_unified_graph_hybrid.py` |
| **Add tautological type detector (NEW)** | Low | Medium | `entity_quality_filter.py` |

### Medium-Term (After Re-Extraction)

| Task | Effort | Impact | File |
|------|--------|--------|------|
| Add verification pass | High | High | New module |
| Add embedding-based merge validation | Medium | Medium | `entity_merge_validator.py` |
| Cross-chunk entity awareness | High | Medium | `entity_extractor.py` |
| Human review queue | High | Medium | New system |
| **Decide narrative content strategy (NEW)** | Medium | High | Architecture decision |

---

## Appendix: Code References

### Current Extraction Pipeline

| Step | File | Function/Class |
|------|------|----------------|
| Episode extraction | `scripts/archive/ace_extraction/extract_knowledge_from_episodes.py` | `extract_from_episode()` |
| Book extraction | `scripts/archive/ace_extraction/extract_knowledge_from_books.py` | `extract_book()` |
| Entity extraction | `src/knowledge_graph/extractors/entity_extractor.py` | `EntityExtractor.extract_entities()` |
| Relationship extraction | `src/knowledge_graph/extractors/relationship_extractor.py` | `RelationshipExtractor.extract_relationships()` |
| Text chunking | `src/knowledge_graph/extractors/chunking.py` | `chunk_text()` |
| Graph building | `scripts/build_unified_graph_hybrid.py` | `HybridGraphBuilder` |
| Merge validation | `src/knowledge_graph/validators/entity_merge_validator.py` | `EntityMergeValidator.can_merge()` |

### Postprocessing Pipeline Order (Actual Active Pipeline)

**Per `src/knowledge_graph/postprocessing/README.md`:**

| Priority | Module | Purpose |
|----------|--------|---------|
| 10-20 | Book-specific modules | Praise quotes, citations (books only) |
| 30 | `VagueEntityBlocker` | Filter vague entities (this, that, it) |
| 40 | `ListSplitter` | Split "A, B, and C" lists |
| 50 | `ContextEnricher` | Replace vague entities with context |
| 60 | `PronounResolver` | Resolve pronouns to antecedents |
| 70 | `PredicateNormalizer` | Normalize verbose predicates |
| 80 | `PredicateValidator` | Validate predicate logic |
| 90-100 | Final validation | Title validation, figurative language |

**Note:** Many modules in `universal/` (deduplicator, semantic_deduplicator, entity_resolver, confidence_filter, etc.) are **not included** in the active pipelines. This is a significant gap in the current processing.

### Key Data Files

| File | Description | Size/Count |
|------|-------------|------------|
| `data/knowledge_graph_unified/unified.json` | Main unified knowledge graph | Primary output |
| `data/knowledge_graph/entities/*.json` | Raw book extraction outputs | 4 files |
| `data/knowledge_graph_unified/episodes_postprocessed/*.json` | ACE-processed episode data | 41 files |
| `data/transcripts/episode_*.json` | Source episode transcripts | 172 files |
| `data/books/*/metadata.json` | Book metadata | 4 files |
| `data/graphrag_hierarchy/graphrag_hierarchy.json` | 3D visualization hierarchy | Main viz file |

### Key Configuration Settings

| Setting | Current Value | Location | Description |
|---------|---------------|----------|-------------|
| Chunk size | 800 tokens | `extract_knowledge_from_episodes.py:92` | Text chunk size for extraction |
| Chunk overlap | 100 tokens | `extract_knowledge_from_episodes.py:92` | Overlap between chunks |
| Similarity threshold | 95% | `build_unified_graph_hybrid.py:113` | Fuzzy match threshold for Tier 1 merges |
| Tier 2 threshold | 85-94% | `entity_merge_validator.py:517` | Medium confidence merge range |
| Min length ratio | 0.6 | `entity_merge_validator.py:159` | Minimum name length ratio for merge |
| Max entity name length | 80 chars | Proposed | Currently no limit |
| Max entity word count | 8 words | Proposed | Currently no limit |

### API Keys Required

| Key | Purpose | Used By |
|-----|---------|---------|
| `OPENAI_API_KEY` | LLM extraction, embeddings | Entity/relationship extraction |
| `PINECONE_API_KEY` | Vector storage | RAG system (not KG extraction) |
| `NEO4J_URI` (optional) | Graph database | `neo4j_client.py` |
| `NEO4J_USER` (optional) | Graph database auth | `neo4j_client.py` |
| `NEO4J_PASSWORD` (optional) | Graph database auth | `neo4j_client.py` |

### File Format References

#### Episode Transcript Format (`data/transcripts/episode_*.json`)
```json
{
  "episode_number": 120,
  "title": "Episode Title",
  "guest": "Guest Name",
  "full_transcript": "...",
  "segments": [...],
  "audio_url": "...",
  "youtube_url": "..."
}
```

#### Book Metadata Format (`data/books/*/metadata.json`)
```json
{
  "title": "Book Title",
  "author": "Author Name",
  "slug": "book-slug",
  "chapters": [
    {"number": 1, "title": "Chapter 1 Title", "page": 1}
  ]
}
```

#### Entity Extraction Output Format
```json
{
  "name": "Entity Name",
  "type": "PERSON|ORGANIZATION|CONCEPT|PLACE|...",
  "description": "Brief description",
  "aliases": ["alias1", "alias2"],
  "metadata": {
    "episode_number": 120,
    "chunk_id": "episode_120_chunk_5"
  }
}
```

#### Relationship Extraction Output Format
```json
{
  "source": "Source Entity Name",
  "relationship": "predicate",
  "target": "Target Entity Name",
  "source_type": "PERSON",
  "target_type": "ORGANIZATION",
  "evidence_text": "Supporting text from source",
  "p_true": 0.89,
  "classification_flags": ["factual"]
}
```

#### Unified Graph Format (`unified.json`)
```json
{
  "entities": {
    "Entity Name": {
      "type": "PERSON",
      "description": "...",
      "sources": ["episode_120", "y-on-earth"],
      "aliases": [...],
      "provenance": [...]
    }
  },
  "relationships": [
    {
      "source": "...",
      "predicate": "...",
      "target": "...",
      "evidence": "...",
      "sources": [...]
    }
  ]
}
```

---

## Next Steps

1. **Review this document** with stakeholders
2. **Approve priority order** for implementation
3. **Create canonical entity registry** for known entities
4. **Update extraction prompts** before next extraction run
5. **Run test extraction** on small subset to validate improvements
6. **Full re-extraction** with new pipeline
7. **Quality review** of results

---

*This document should be updated as improvements are implemented and new issues are discovered.*
