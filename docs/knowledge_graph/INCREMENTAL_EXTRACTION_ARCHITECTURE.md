# Incremental Chapter-by-Chapter Extraction Architecture

## Problem Statement

**Context**: We need to extract a knowledge graph from "OUR BIGGEST DEAL" (480 pages, 26 chapters + essays + case studies).

**Challenges**:
1. **Scale**: 480 pages is too large for single-pass extraction (memory, token limits, quality control)
2. **Quality Assurance**: Need to achieve A+ grade (1-2% issue rate) for each section
3. **Whole-Book Context**: Some processes need cross-chapter context:
   - **Deduplication**: Same relationship mentioned in multiple chapters
   - **Entity Resolution**: "Aaron Perry" vs "Aaron William Perry" vs "Perry" across chapters
   - **Semantic Deduplication**: Similar concepts expressed differently
   - **Cross-references**: "As discussed in Chapter 2..." needs context
4. **No Re-extraction**: Once a chapter achieves A+ grade, we should NOT re-extract it
5. **Provenance**: Need complete version history to replicate any extraction

**Question**: How do we progress chapter-by-chapter while handling whole-book context processes?

---

## Proposed Solution: Hybrid Incremental with Periodic Consolidation

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Per-Chapter Extraction (Independent)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Front Matter (pages 1-30) → Extract → Iterate → A+ grade ✅       │
│   Pipeline: Pass 1 → Pass 2 → Pass 2.5 (within-chapter only)      │
│   Output: kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/  │
│           front_matter_v14_3_3_20251015_143052.json                         │
│   Script: extract_kg_v14_3_3_incremental.py                        │
│                                                                     │
│ Chapter 1 (pages 31-50) → Extract → Iterate → A+ grade ✅         │
│   Same pipeline                                                     │
│   Output: chapter_01_v14_3_3_20251015_150234.json                  │
│                                                                     │
│ Chapter 2 (pages 51-70) → Extract → Iterate → A+ grade ✅         │
│   Same pipeline                                                     │
│   Output: chapter_02_v14_3_3_20251015_152109.json                  │
│                                                                     │
│ ... continue for all chapters/sections ...                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Periodic Consolidation (Every Part or ~5 Chapters)        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Consolidation Pass 1: Part I (Chapters 1-13)                       │
│   Input: front_matter.json + chapter_01.json + ... + chapter_13    │
│   Process:                                                          │
│     1. Merge all chapter JSONs                                      │
│     2. Cross-chapter Deduplicator                                   │
│     3. Cross-chapter EntityResolver                                 │
│     4. Cross-chapter SemanticDeduplicator (optional)               │
│   Output: kg_extraction_playbook/output/our_biggest_deal/v14_3_3/consolidations/  │
│           part_1_consolidated_v14_3_3_20251015_160045.json                        │
│   Script: consolidate_chapters.py --part part_1                     │
│                                                                     │
│ Consolidation Pass 2: Part II (Guest Essays 1-26)                  │
│   Same process                                                      │
│   Output: part_2_consolidated_v14_3_3_20251015_165023.json         │
│                                                                     │
│ Consolidation Pass 3: Part III (Case Studies)                      │
│   Same process                                                      │
│   Output: part_3_consolidated_v14_3_3_20251015_171234.json         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Final Whole-Book Consolidation                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Final Pass: Merge ALL Parts                                        │
│   Input: part_1.json + part_2.json + part_3.json + part_4.json    │
│   Process:                                                          │
│     1. Merge all part consolidations                                │
│     2. Final cross-book Deduplicator                                │
│     3. Final cross-book EntityResolver                              │
│     4. Final SemanticDeduplicator                                   │
│   Output: kg_extraction_playbook/output/our_biggest_deal/v14_3_3/final/       │
│           our_biggest_deal_final_v14_3_3_20251015_180156.json                 │
│   Script: consolidate_chapters.py --final                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Reasoning and Benefits

### Why This Approach Works

#### 1. Chapters Are Mostly Self-Contained

**Observation**: Book chapters focus on specific topics, authors, or themes.

**Implication**:
- Most entity mentions are localized to their chapter context
- Cross-chapter references are explicit ("As discussed in Chapter 2...")
- Within-chapter deduplication catches 80-90% of duplicate relationships

**Evidence from Book Structure**:
- Part I: Conceptual chapters (each with distinct focus)
- Part II: Guest essays (each by different author, self-contained)
- Part III: Case studies (each about different organization)
- Part IV: Aphorisms (self-contained wisdom)

**Conclusion**: We can safely extract each chapter independently without losing quality.

---

#### 2. Periodic Consolidation is Efficient

**Problem**: Cross-chapter duplicates/entities need resolution, but we don't want to re-extract.

**Solution**: Lightweight merge operations that run ONLY on completed chapters.

**How It Works**:
```python
# Consolidation is NOT re-extraction
# It's just merge + dedup + entity resolution

# CONCEPTUAL PSEUDOCODE (see full implementation with PipelineOrchestrator at line 312)
def consolidate_chapters(chapter_files):
    # 1. Load all chapter extractions (already A+ grade)
    all_relationships = []
    for file in chapter_files:
        relationships = load_json(file)
        all_relationships.extend(relationships)

    # 2. Run cross-chapter deduplication (via PipelineOrchestrator)
    deduped = run_deduplicator(all_relationships)

    # 3. Run cross-chapter entity resolution (via PipelineOrchestrator)
    resolved = run_entity_resolver(deduped)

    # 4. Save consolidated result
    save_json(consolidated_output, resolved)

    # NOTE: Original chapter files remain UNCHANGED
```

**Benefits**:
- Fast (no LLM calls, just Python operations)
- Cheap (no API costs)
- Safe (original extractions preserved)
- Validatable (can compare before/after consolidation)

**When to Consolidate**:
- After each Part (natural book divisions)
- Every 5 chapters (prevents accumulation of duplicates)
- At the end (final whole-book consolidation)

---

#### 3. No Re-extraction Needed

**Key Principle**: Once a chapter extraction achieves A+ grade, it is FROZEN.

**Workflow**:
```
Chapter 3 Iteration 1: Extract → Reflector → B+ grade
                      ↓
                  Analyze issues, improve prompts
                      ↓
Chapter 3 Iteration 2: Extract → Reflector → A grade
                      ↓
                  Fine-tune edge cases
                      ↓
Chapter 3 Iteration 3: Extract → Reflector → A+ grade ✅
                      ↓
                  FROZEN - Save final extraction
                      ↓
                  Move to Chapter 4

IMPORTANT: Chapter 3 is NEVER re-extracted
           Only merged during consolidation passes
```

**Benefits**:
- Quality locked in (no risk of regression)
- Cost-effective (no redundant API calls)
- Clear progress tracking (each chapter is a milestone)
- Parallelizable (can extract multiple chapters concurrently if needed)

---

#### 4. Perfect Provenance

**Directory Structure**:
```
kg_extraction_playbook/output/our_biggest_deal/
├── v14_3_3/
│   ├── chapters/
│   │   ├── front_matter_v14_3_3_20251015_143052.json
│   │   ├── chapter_01_v14_3_3_20251015_150234.json
│   │   ├── chapter_01_v14_3_3_20251015_151045.json  (iteration 2)
│   │   ├── chapter_01_v14_3_3_20251015_152109.json  (iteration 3, FINAL ✅)
│   │   ├── chapter_02_v14_3_3_20251016_090512.json  (FINAL ✅)
│   │   └── ...
│   ├── consolidations/
│   │   ├── part_1_consolidated_v14_3_3_20251016_160045.json
│   │   ├── part_1_entity_aliases_20251016_160045.json  (alias map for Part 1)
│   │   ├── part_1_stats_20251016_160045.json  (dedup/merge statistics)
│   │   ├── part_2_consolidated_v14_3_3_20251017_101234.json
│   │   ├── part_2_entity_aliases_20251017_101234.json
│   │   ├── part_2_stats_20251017_101234.json
│   │   └── ...
│   ├── final/
│   │   ├── our_biggest_deal_final_v14_3_3_20251017_180156.json
│   │   ├── our_biggest_deal_entity_aliases_20251017_180156.json  (final alias map)
│   │   └── our_biggest_deal_stats_20251017_180156.json  (final statistics)
│   ├── analysis/
│   │   ├── front_matter_reflection_v14_3_3_20251015_143500.json
│   │   ├── front_matter_summary_20251015_143500.json  (machine-readable A+ gate)
│   │   ├── chapter_01_reflection_v14_3_3_20251015_152200.json
│   │   ├── chapter_01_summary_20251015_152200.json
│   │   └── ...
│   ├── manifests/
│   │   ├── front_matter_execution_20251015_143052.json  (git hash, env, packages)
│   │   ├── chapter_01_execution_20251015_152109.json
│   │   └── ...
│   ├── scripts_used/
│   │   ├── extract_kg_v14_3_3_incremental_20251015.py  (timestamped copy)
│   │   ├── consolidate_chapters_20251016.py  (timestamped copy)
│   │   ├── pass1_extraction_v14_3_3_20251015.txt  (prompt snapshot)
│   │   └── pass2_evaluation_v14_3_3_20251015.txt  (prompt snapshot)
│   └── status.json  (freeze tracking, progress coordination)
└── v14_3_4/  (future iteration if needed)
    └── ...
```

**Provenance Tracking**:
- Every extraction has timestamp
- Scripts are copied with timestamp at execution
- Prompts are snapshotted at extraction time
- Analysis results linked to extractions
- Complete audit trail from raw PDF to final KG

**Replication**:
```bash
# To replicate Chapter 3 extraction from 2025-10-15:
python3 kg_extraction_playbook/output/our_biggest_deal/v14_3_3/scripts_used/extract_kg_v14_3_3_incremental_20251015.py \
  --section chapter_03 \
  --pages 51-70 \
  --prompts kg_extraction_playbook/output/our_biggest_deal/v14_3_3/scripts_used/

# Result will match: chapter_03_v14_3_3_20251015_152109.json
# Execution manifest confirms: git hash, Python version, package versions
```

---

## Modified Pipeline Architecture

### Within-Chapter Pipeline (Pass 2.5)

**Runs ONLY on current chapter relationships**:

```python
# Pass 2.5: Chapter-level postprocessing
modules = [
    # Book-specific (run first)
    PraiseQuoteDetector(v1.5.0),  # Author whitelist
    MetadataFilter,
    FrontMatterDetector(v1.0.0),  # Foreword detection
    SubjectiveContentFilter,
    BibliographicCitationParser,

    # Universal processing
    ContextEnricher,  # Resolve vague entities
    ListSplitter,
    PronounResolver,
    PredicateNormalizer,
    PredicateValidator,
    VagueEntityBlocker,  # Block unresolved vague entities

    # Book-specific validation
    TitleCompletenessValidator,
    FigurativeLanguageFilter,

    # Chapter-level deduplication
    ClaimClassifier,
    Deduplicator,  # ← Within-chapter ONLY
]

# NOTE: SemanticDeduplicator NOT run here (too expensive for per-chapter)
```

**Why This Works**:
- Each module processes only the current chapter's relationships
- Deduplicator catches within-chapter duplicates (most common case)
- No cross-chapter knowledge needed for these modules
- Fast, efficient, complete quality assurance per chapter

---

### Cross-Chapter Consolidation Pipeline (Pass 3)

**Runs on merged chapter relationships**:

```python
from src.knowledge_graph.postprocessing.base import PipelineOrchestrator, ProcessingContext
from src.knowledge_graph.postprocessing.universal import (
    FieldNormalizer, Deduplicator, EntityResolver, SemanticDeduplicator
)

def consolidate_chapters(chapter_files, document_metadata=None, use_semantic_dedup=False):
    """
    Consolidate multiple chapter extractions into a unified knowledge graph.

    Args:
        chapter_files: List of paths to chapter extraction JSON files
        document_metadata: Dict with known_entities (allowlist) and other metadata
        use_semantic_dedup: Whether to run expensive SemanticDeduplicator

    Returns:
        (consolidated_relationships, consolidation_stats)
    """

    # Step 1: Load and merge all chapter relationships
    all_relationships = []
    for file in chapter_files:
        chapter_rels = load_json(file)
        all_relationships.extend(chapter_rels)

    logger.info(f"Loaded {len(all_relationships)} relationships from {len(chapter_files)} chapters")

    # Step 2: Create processing context with allowlist
    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata or {
            'known_entities': ['Aaron William Perry', 'John Perkins'],  # Example allowlist
            'title': 'OUR BIGGEST DEAL'
        }
    )

    # Step 3: Create consolidation pipeline
    modules = [
        FieldNormalizer(),       # Priority 5 - normalize field names first
        Deduplicator(),          # Priority 110 - remove exact duplicates
        EntityResolver(),        # Priority 112 - resolve entity variations
    ]

    # Optional: Add semantic deduplication (expensive, uses embeddings)
    if use_semantic_dedup:
        modules.append(
            SemanticDeduplicator(config={
                'similarity_threshold': 0.85,  # Default: 0.85
                'model_name': 'all-MiniLM-L6-v2'  # From requirements.txt (sentence-transformers)
            })
        )

    # Step 4: Run pipeline using orchestrator
    orchestrator = PipelineOrchestrator(modules, config={
        'halt_on_error': False  # Continue processing even if a module fails
    })

    consolidated, pipeline_stats = orchestrator.run(all_relationships, context)

    logger.info(f"Consolidation complete: {len(all_relationships)} → {len(consolidated)} relationships")

    return consolidated, pipeline_stats

# Input: All chapter relationships already postprocessed to A+
# Process: Lightweight merge operations (no LLM calls for Deduplicator/EntityResolver)
# Optional: SemanticDeduplicator if budget allows (uses embeddings, ~$0.01 per 1000 rels)
# Output: Consolidated relationships with cross-chapter duplicates removed
```

**ProcessingContext Construction**:
```python
# Minimal context for consolidation
context = ProcessingContext(
    content_type='book',
    document_metadata={
        'known_entities': ['Author Name'],  # Allowlist for EntityResolver
        'title': 'Book Title'
    }
)

# Full context with additional metadata
context = ProcessingContext(
    content_type='book',
    document_metadata={
        'author': 'Aaron William Perry',
        'title': 'OUR BIGGEST DEAL',
        'known_entities': ['Aaron William Perry', 'Aaron Perry'],  # Variants
        'publication_year': 2024,
        'isbn': '978-1234567890'
    },
    config={
        'entity_resolution_threshold': 0.8,
        'semantic_dedup_enabled': False  # Expensive, opt-in
    }
)
```

**EntityResolver Module (NEW)**:

**File Path**: `src/knowledge_graph/postprocessing/universal/entity_resolver.py`

```python
class EntityResolver(PostProcessingModule):
    """
    Resolves entity name variations across chapters with deterministic tie-breaking.

    Priority: 112 (after Deduplicator at 110, before SemanticDeduplicator at 115)
    Content Types: All

    Example:
    - Chapter 1: "Aaron Perry" (appears 5 times)
    - Chapter 10: "Aaron William Perry" (appears 15 times)
    - Chapter 15: "Perry" (appears 2 times)

    Resolution: All → "Aaron William Perry" (canonical form)
    Alias Map: {"aaron perry": "Aaron William Perry", "perry": "Aaron William Perry"}

    Deterministic Tie-Breaking Rules (in order):
    1. Longest name wins (more specific)
    2. Most frequent occurrence across all chapters
    3. Earliest occurrence (chapter/page order)
    4. Explicit allowlist (known authors from metadata)
    """

    name = "EntityResolver"
    description = "Resolves entity name variations with deterministic canonicalization"
    priority = 112
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.alias_map = {}  # Persisted alongside consolidated output
        self.canonical_entities = {}  # variant_key -> canonical_form

    def process_batch(self, relationships, context):
        # Step 1: Build entity mention index
        entity_variants = self._find_variants(relationships)

        # Step 2: Apply deterministic tie-breaking to select canonical forms
        self.canonical_entities = self._select_canonical_forms(entity_variants, context)

        # Step 3: Build alias map for persistence
        self.alias_map = self._build_alias_map(self.canonical_entities)

        # Step 4: Update all relationships with canonical entities
        resolved = self._apply_canonical_forms(relationships, self.canonical_entities)

        # Step 5: Persist alias map (called by consolidation script)
        # self._save_alias_map(output_path)

        return resolved

    def _select_canonical_forms(self, entity_variants, context):
        """
        Deterministic tie-breaking to select canonical entity name.

        Returns: {variant_key: canonical_name}
        """
        canonical = {}

        for variant_key, variants in entity_variants.items():
            # variants = [("Aaron Perry", count=5, first_page=12),
            #             ("Aaron William Perry", count=15, first_page=145), ...]

            # Rule 1: Longest name (most specific)
            longest = max(variants, key=lambda v: len(v[0]))

            # Rule 2: If tie in length, use most frequent
            same_length = [v for v in variants if len(v[0]) == len(longest[0])]
            most_frequent = max(same_length, key=lambda v: v[1])  # v[1] = count

            # Rule 3: If tie in frequency, use earliest occurrence
            same_freq = [v for v in same_length if v[1] == most_frequent[1]]
            earliest = min(same_freq, key=lambda v: v[2])  # v[2] = first_page

            # Rule 4: If in allowlist, override (known authors take precedence)
            allowlist = context.document_metadata.get('known_entities', [])
            for variant_name, count, page in variants:
                if variant_name.lower() in [a.lower() for a in allowlist]:
                    canonical[variant_key] = variant_name
                    break
            else:
                canonical[variant_key] = earliest[0]

        return canonical

    def _build_alias_map(self, canonical_entities):
        """
        Build alias map for persistence.

        Returns: {variant: canonical_form}
        """
        alias_map = {}
        for variant_key, canonical_name in canonical_entities.items():
            # variant_key might be normalized form (lowercase, no spaces)
            # Store mapping from all variants to canonical
            alias_map[variant_key] = canonical_name

        return alias_map

    def save_alias_map(self, output_path):
        """
        Persist alias map for reproducibility and stability across runs.

        File format: {variant: canonical, ...}
        Example: {
            "aaron perry": "Aaron William Perry",
            "perry": "Aaron William Perry",
            "john smith": "John P. Smith",
            ...
        }
        """
        import json
        with open(output_path, 'w') as f:
            json.dump(self.alias_map, f, indent=2, sort_keys=True)  # sort_keys for determinism

        logger.info(f"   EntityResolver: Saved {len(self.alias_map)} aliases to {output_path}")
```

---

### FieldNormalizer Module (Adapter)

**Problem**: Historical relationship field inconsistency across pipeline versions.

**Context**:
- Some pipelines use `predicate` field (e.g., "authored", "practices")
- Most modules expect `relationship` field (Deduplicator, PredicateNormalizer, PredicateValidator)
- Consolidation must handle both formats seamlessly

**Solution**: FieldNormalizer adapter normalizes at pipeline ingress, making `relationship` canonical.

**File Path**: `src/knowledge_graph/postprocessing/universal/field_normalizer.py`

```python
class FieldNormalizer(PostProcessingModule):
    """
    Adapter module to normalize field naming inconsistencies across pipeline versions.

    Purpose: Ensures consolidated data uses consistent field names regardless of source.
    Priority: 5 (earliest - runs before all other modules)
    Content Types: All

    Normalizations:
    1. predicate <-> relationship (ensures 'relationship' is canonical)
    2. source_entity <-> source (ensures 'source' is canonical)
    3. target_entity <-> target (ensures 'target' is canonical)

    Notes:
    - Both 'relationship' and 'predicate' fields are set to the same value for compatibility
    - Most modules expect 'relationship', but some may read 'predicate'
    """

    name = "FieldNormalizer"
    description = "Normalizes field naming across pipeline versions"
    priority = 5
    version = "1.0.0"

    FIELD_MAPPINGS = {
        # Legacy field -> Canonical field
        'predicate': 'relationship',  # relationship is canonical
        'source_entity': 'source',
        'target_entity': 'target',
    }

    def process_batch(self, relationships, context):
        """
        Normalize field names in all relationships.

        Args:
            relationships: List of relationship dicts or objects
            context: ProcessingContext (unused for this module)

        Returns:
            List of relationships with normalized field names
        """
        normalized = []

        for rel in relationships:
            rel_dict = rel if isinstance(rel, dict) else rel.__dict__

            # Create normalized copy
            normalized_rel = {}

            for key, value in rel_dict.items():
                # Check if key needs normalization
                canonical_key = self.FIELD_MAPPINGS.get(key, key)
                normalized_rel[canonical_key] = value

            # Ensure canonical field exists (backfill if only legacy exists)
            for legacy, canonical in self.FIELD_MAPPINGS.items():
                if legacy in rel_dict and canonical not in normalized_rel:
                    normalized_rel[canonical] = rel_dict[legacy]

            normalized.append(normalized_rel)

        self.stats['normalized'] = len(normalized)
        return normalized

    def validate_config(self) -> bool:
        """No configuration needed for FieldNormalizer."""
        return True
```

**Usage in Consolidation**:
```python
from src.knowledge_graph.postprocessing.base import PipelineOrchestrator, ProcessingContext
from src.knowledge_graph.postprocessing.universal import (
    FieldNormalizer, Deduplicator, EntityResolver
)

def consolidate_chapters(chapter_files, document_metadata=None):
    # Step 1: Load and merge all chapter relationships
    all_relationships = []
    for file in chapter_files:
        chapter_rels = load_json(file)
        all_relationships.extend(chapter_rels)

    # Step 2: Create processing context
    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata or {}
    )

    # Step 3: Create consolidation pipeline with FieldNormalizer first
    modules = [
        FieldNormalizer(),      # Priority 5 - normalize field names first
        Deduplicator(),         # Priority 110 - remove exact duplicates
        EntityResolver(),       # Priority 112 - resolve entity variations
    ]

    # Step 4: Run pipeline using orchestrator
    orchestrator = PipelineOrchestrator(modules)
    consolidated, stats = orchestrator.run(all_relationships, context)

    return consolidated, stats
```

---

## Schema Definitions

### Execution Manifest Schema

**Purpose**: Capture complete execution environment for reproducibility.

**File**: `manifests/{section}_execution_{timestamp}.json`

**Schema**:
```json
{
  "section": "chapter_03",
  "timestamp": "20251015_152109",
  "version": "v14_3_3",
  "git": {
    "commit_hash": "a1b2c3d4e5f6...",
    "branch": "main",
    "is_dirty": false,
    "uncommitted_files": []
  },
  "environment": {
    "python_version": "3.11.5",
    "platform": "Linux-5.15.0-143-generic-x86_64",
    "hostname": "claudeuser-yonearth"
  },
  "packages": {
    "openai": "1.12.0",
    "anthropic": "0.18.1",
    "pydantic": "2.6.1",
    "python-dateutil": "2.8.2"
  },
  "script": {
    "path": "scripts/extract_kg_v14_3_3_incremental.py",
    "checksum": "sha256:abc123...",
    "args": {
      "book": "our_biggest_deal",
      "section": "chapter_03",
      "pages": "51-70",
      "output": "kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/"
    }
  },
  "prompts": {
    "pass1_extraction": {
      "path": "kg_extraction_playbook/prompts/pass1_extraction_v14_3_3.txt",
      "checksum": "sha256:def456..."
    },
    "pass2_evaluation": {
      "path": "kg_extraction_playbook/prompts/pass2_evaluation_v14_3_3.txt",
      "checksum": "sha256:ghi789..."
    }
  },
  "model_config": {
    "pass1_model": "gpt-4o-2024-08-06",
    "pass2_model": "gpt-4o-2024-08-06",
    "temperature": 0.0,
    "max_tokens": 16384
  },
  "duration_seconds": 1247,
  "output_file": "kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/chapter_03_v14_3_3_20251015_152109.json"
}
```

**Generation**:
```python
def generate_execution_manifest(section, script_args):
    import subprocess
    import hashlib
    import sys
    import platform
    import pkg_resources

    # Get git info
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
    git_dirty = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD']) != 0

    # Get package versions
    packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    manifest = {
        'section': section,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'git': {
            'commit_hash': git_hash,
            'branch': git_branch,
            'is_dirty': git_dirty,
        },
        'environment': {
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
        },
        'packages': packages,
        # ... rest of manifest
    }

    return manifest
```

---

### Status Tracking Schema

**Purpose**: Prevent overwrites, enable parallel coordination, track freeze status.

**File**: `status.json` (at version root)

**Schema**:
```json
{
  "version": "v14_3_3",
  "book": "our_biggest_deal",
  "last_updated": "2025-10-15T15:21:09Z",
  "sections": {
    "front_matter": {
      "status": "frozen",
      "grade": "A+",
      "issue_rate": 1.2,
      "final_extraction": "front_matter_v14_3_3_20251015_143052.json",
      "iterations": 2,
      "frozen_at": "2025-10-15T14:30:52Z"
    },
    "chapter_01": {
      "status": "frozen",
      "grade": "A+",
      "issue_rate": 1.8,
      "final_extraction": "chapter_01_v14_3_3_20251015_152109.json",
      "iterations": 3,
      "frozen_at": "2025-10-15T15:21:09Z"
    },
    "chapter_02": {
      "status": "in_progress",
      "grade": "B+",
      "issue_rate": 8.5,
      "current_extraction": "chapter_02_v14_3_3_20251016_083421.json",
      "iterations": 1,
      "last_updated": "2025-10-16T08:34:21Z"
    },
    "chapter_03": {
      "status": "pending"
    }
  },
  "consolidations": {
    "part_1": {
      "status": "pending",
      "chapters": ["front_matter", "chapter_01", "chapter_02", "...", "chapter_13"],
      "ready_count": 2,
      "total_count": 14
    }
  },
  "final": {
    "status": "pending"
  }
}
```

**Status Values**:
- `pending`: Not started
- `in_progress`: Currently being extracted/processed
- `frozen`: A+ grade achieved, locked from re-extraction
- `failed`: Extraction encountered critical error

**Usage**:
```python
def check_section_status(section):
    """Check if section can be extracted or is frozen."""
    status = load_json('kg_extraction_playbook/output/our_biggest_deal/v14_3_3/status.json')

    section_status = status['sections'].get(section, {}).get('status', 'pending')

    if section_status == 'frozen':
        raise ValueError(f"Section {section} is frozen (A+ grade). Cannot re-extract.")

    if section_status == 'in_progress':
        logger.warning(f"Section {section} is already in progress. Check for parallel process.")

    return section_status

def freeze_section(section, extraction_file, grade, issue_rate):
    """Mark section as frozen after achieving A+ grade."""
    status = load_json('status.json')

    status['sections'][section] = {
        'status': 'frozen',
        'grade': grade,
        'issue_rate': issue_rate,
        'final_extraction': extraction_file,
        'frozen_at': datetime.now().isoformat(),
    }

    save_json('status.json', status)
    logger.info(f"✅ Section {section} FROZEN at {grade} grade")
```

---

### Reflector Summary Schema (Machine-Readable A+ Gate)

**Purpose**: Enable automated quality gates without parsing natural language.

**File**: `analysis/{section}_summary_{timestamp}.json`

**Schema**:
```json
{
  "section": "chapter_03",
  "timestamp": "20251015_152200",
  "version": "v14_3_3",
  "extraction_file": "chapter_03_v14_3_3_20251015_152109.json",
  "quality_metrics": {
    "overall_grade": "A+",
    "issue_rate": 1.8,
    "total_relationships": 478,
    "total_issues": 9
  },
  "issue_breakdown": {
    "CRITICAL": 0,
    "HIGH": 0,
    "MEDIUM": 2,
    "MILD": 7
  },
  "quality_gate": {
    "passes_a_plus_gate": true,
    "criteria": {
      "critical_issues_zero": true,
      "high_issues_lte_2": true,
      "issue_rate_lte_2_percent": true
    }
  },
  "actionable_issues": [
    {
      "severity": "MEDIUM",
      "category": "list_splitting",
      "description": "Entity split on 'and' within compound noun",
      "example": "soil health and fertility",
      "relationship_id": "rel_234"
    },
    {
      "severity": "MEDIUM",
      "category": "vague_entity",
      "description": "Abstract entity without qualifier",
      "example": "community impact",
      "relationship_id": "rel_456"
    }
  ],
  "recommendations": [
    "Consider adding context-aware splitting to ListSplitter",
    "Entity 'community impact' could be more specific: 'community health impact'"
  ]
}
```

**Automated Quality Gate Check**:
```python
def check_quality_gate(summary_file):
    """
    Check if section passes A+ quality gate.

    Returns: (passes: bool, grade: str, reason: str)
    """
    summary = load_json(summary_file)

    gate = summary.get('quality_gate', {})
    passes = gate.get('passes_a_plus_gate', False)
    grade = summary['quality_metrics']['overall_grade']

    if not passes:
        failed_criteria = [k for k, v in gate['criteria'].items() if not v]
        reason = f"Failed criteria: {', '.join(failed_criteria)}"
        return False, grade, reason

    return True, grade, "All A+ criteria met"

# Usage in extraction pipeline
summary_file = "analysis/chapter_03_summary_20251015_152200.json"
passes, grade, reason = check_quality_gate(summary_file)

if passes:
    freeze_section('chapter_03', extraction_file, grade, issue_rate)
    logger.info(f"✅ Chapter 3 achieves {grade} grade - FROZEN")
else:
    logger.warning(f"❌ Chapter 3 grade {grade} - {reason}")
    logger.info("Analyzing issues for next iteration...")
```

---

### Consolidation Statistics Schema

**Purpose**: Track deduplication and entity resolution metrics for reproducibility.

**File**: `consolidations/{part}_stats_{timestamp}.json`

**Schema**:
```json
{
  "consolidation": "part_1",
  "timestamp": "20251016_160045",
  "version": "v14_3_3",
  "input_chapters": [
    "front_matter_v14_3_3_20251015_143052.json",
    "chapter_01_v14_3_3_20251015_152109.json",
    "chapter_02_v14_3_3_20251016_090512.json",
    "..."
  ],
  "processing_steps": [
    {
      "module": "FieldNormalizer",
      "priority": 5,
      "before_count": 4567,
      "after_count": 4567,
      "changes": {
        "field_normalizations": 1234,
        "details": "Normalized 'predicate' -> 'relationship' in 1234 rels"
      }
    },
    {
      "module": "Deduplicator",
      "priority": 110,
      "before_count": 4567,
      "after_count": 3921,
      "changes": {
        "removed_duplicates": 646,
        "duplicate_groups": 234,
        "details": "Removed 646 exact duplicates across chapters"
      }
    },
    {
      "module": "EntityResolver",
      "priority": 115,
      "before_count": 3921,
      "after_count": 3921,
      "changes": {
        "entity_merges": 89,
        "aliases_created": 178,
        "details": "Resolved 89 entity variations (e.g., 'Aaron Perry' -> 'Aaron William Perry')"
      },
      "alias_map_file": "part_1_entity_aliases_20251016_160045.json"
    }
  ],
  "summary": {
    "total_input_relationships": 4567,
    "total_output_relationships": 3921,
    "reduction_count": 646,
    "reduction_percentage": 14.1,
    "processing_time_seconds": 347
  },
  "output_file": "part_1_consolidated_v14_3_3_20251016_160045.json"
}
```

**Idempotency Verification**:
```python
def verify_consolidation_idempotency(chapters):
    """
    Verify that running consolidation twice produces identical results.

    Purpose: Ensures deterministic behavior, catches non-deterministic bugs.
    """
    # Run consolidation twice
    result1, stats1 = consolidate_chapters(chapters, run_id='test_run_1')
    result2, stats2 = consolidate_chapters(chapters, run_id='test_run_2')

    # Compare outputs
    if result1 == result2:
        logger.info("✅ Consolidation is idempotent (identical results)")
    else:
        logger.error("❌ Consolidation is NOT idempotent (results differ)")
        diff = compare_results(result1, result2)
        logger.error(f"   Differences: {diff}")
        raise ValueError("Consolidation must be idempotent")

    # Compare stats
    assert stats1['summary']['reduction_count'] == stats2['summary']['reduction_count']
    assert stats1['processing_steps'][1]['changes']['entity_merges'] == \
           stats2['processing_steps'][1]['changes']['entity_merges']

    logger.info("✅ Consolidation statistics are deterministic")
```

---

## Optional Dependencies and Fallback Behavior

### spaCy Dependency for ListSplitter

**Context**: `ListSplitter` module uses spaCy for noun phrase detection to avoid splitting within compound nouns.

**Requirement**: Optional (graceful degradation if not available)

**Installation**:
```bash
# Install spaCy and English model
pip install spacy
python -m spacy download en_core_web_sm
```

**Module Behavior**:
```python
class ListSplitter(PostProcessingModule):
    """
    Splits list-type entities on conjunctions (and, or) with context awareness.

    Optional Dependency: spaCy (en_core_web_sm)
    Fallback: Simple rule-based splitting if spaCy unavailable
    """

    def __init__(self, config=None):
        super().__init__(config)

        # Try to load spaCy
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("   ListSplitter: Using spaCy for noun phrase detection")
        except (ImportError, OSError):
            logger.warning("   ListSplitter: spaCy not available, using fallback rules")
            logger.warning("   Install: pip install spacy && python -m spacy download en_core_web_sm")

    def should_skip_split(self, target, conjunction_pos):
        """
        Check if we should skip splitting at this conjunction.

        With spaCy: Uses noun phrase boundaries
        Without spaCy: Uses simple heuristics (quotes, colons)
        """
        if self.nlp:
            # Use spaCy noun phrase detection
            doc = self.nlp(target)
            for chunk in doc.noun_chunks:
                if chunk.start_char <= conjunction_pos <= chunk.end_char:
                    return True  # Conjunction is within noun phrase, don't split
        else:
            # Fallback: Simple rule-based checks
            before = target[:conjunction_pos]

            # Check if within quotes
            quote_count = before.count('"') + before.count("'")
            if quote_count % 2 == 1:
                return True

            # Check if after colon (title pattern)
            if ':' in before and before.rindex(':') > before.rfind(','):
                return True

        return False
```

**Documented in Installation Guide**:
```markdown
## Optional Dependencies

### spaCy (for Enhanced List Splitting)

**Purpose**: Improves `ListSplitter` accuracy by detecting noun phrase boundaries.

**Without spaCy**: ListSplitter falls back to simpler heuristics (still functional, slightly lower accuracy).

**Installation**:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Benefit**: Prevents splitting within compound nouns like "soil health and fertility" → keeps as single entity.

**Requirements File Note**:
- spaCy is intentionally **NOT** in `requirements.txt`
- This is an optional dependency for enhanced accuracy
- System degrades gracefully if spaCy is absent (fallback to simple heuristics)
- Installation: `pip install spacy && python -m spacy download en_core_web_sm`
```

---

## Freeze Enforcement

### Purpose
Prevents accidental re-extraction of chapters that have achieved A+ grade.

### Implementation

**Status Tracking** (`status.json`):
```json
{
  "sections": {
    "chapter_03": {
      "status": "frozen",  // Cannot be overwritten
      "grade": "A+",
      "final_extraction": "chapter_03_v14_3_3_20251015_152109.json"
    }
  }
}
```

**Extraction Script Behavior**:
```python
def check_freeze_status(section):
    """Check if section is frozen before extraction."""
    status = load_json('status.json')
    section_info = status['sections'].get(section, {})

    if section_info.get('status') == 'frozen':
        logger.error(f"❌ ERROR: Section '{section}' is FROZEN (A+ grade achieved)")
        logger.error(f"   Final extraction: {section_info['final_extraction']}")
        logger.error(f"   Grade: {section_info['grade']} ({section_info['issue_rate']}% issue rate)")
        logger.error(f"   To re-extract, manually remove freeze status from status.json")
        sys.exit(1)  # Exit with error code

    return section_info

# Usage at start of extraction
section_info = check_freeze_status(args.section)
logger.info(f"✅ Section '{args.section}' is not frozen, proceeding with extraction")
```

**Freeze Command**:
```python
def freeze_section(section, extraction_file, grade, issue_rate):
    """Mark section as frozen after achieving A+ grade."""
    status = load_json('status.json')

    status['sections'][section] = {
        'status': 'frozen',
        'grade': grade,
        'issue_rate': issue_rate,
        'final_extraction': extraction_file,
        'iterations': status['sections'][section].get('iterations', 0) + 1,
        'frozen_at': datetime.now().isoformat()
    }

    save_json('status.json', status)
    logger.info(f"🔒 Section '{section}' FROZEN at {grade} grade ({issue_rate}% issues)")
    logger.info(f"   This section will not be re-extracted unless manually unfrozen")
```

**Manual Unfreeze** (if needed):
```bash
# Edit status.json manually
# Change "status": "frozen" → "status": "pending"
# Or remove the section entry entirely
```

**Benefits**:
- Prevents accidental overwrites of high-quality extractions
- Clear error messages if freeze is violated
- Manual override available for legitimate re-extraction needs
- Audit trail preserved (frozen_at timestamp, iterations count)

---

## Idempotency Requirements

### Consolidation Idempotency Guarantees

**Requirement**: Running consolidation multiple times on the same inputs MUST produce identical outputs.

**Why It Matters**:
- Reproducibility: Results must be replicable across runs
- Debugging: Non-determinism hides bugs
- Coordination: Parallel processes must converge to same result
- Trust: Users need confidence in output stability

**Implementation Requirements**:

1. **Deterministic Tie-Breaking** (EntityResolver):
   ```python
   # REQUIRED: All tie-breaking must be deterministic
   # Example: EntityResolver canonical selection

   # ✅ GOOD: Deterministic (longest → most frequent → earliest → allowlist)
   canonical = max(variants, key=lambda v: (len(v[0]), v[1], -v[2]))

   # ❌ BAD: Non-deterministic (random tie-breaking)
   canonical = random.choice(variants)

   # ❌ BAD: Non-deterministic (dict iteration order pre-3.7)
   canonical = list(entity_dict.values())[0]
   ```

2. **Stable Sorting**:
   ```python
   # REQUIRED: Sort operations must have stable, explicit keys

   # ✅ GOOD: Explicit sort key
   relationships.sort(key=lambda r: (r.source, r.target, r.relationship))

   # ✅ GOOD: JSON output with sorted keys
   json.dump(alias_map, f, indent=2, sort_keys=True)

   # ❌ BAD: Unstable sort (insertion order dependent)
   relationships = list(set(relationships))  # Set order is non-deterministic
   ```

3. **Deterministic Hashing** (for deduplication):
   ```python
   # REQUIRED: Hash functions must be deterministic across runs

   # ✅ GOOD: Explicit field order
   def relationship_hash(rel):
       return hash((rel.source, rel.relationship, rel.target, rel.evidence_text))

   # ❌ BAD: Hash includes object id (non-deterministic)
   def relationship_hash(rel):
       return hash(id(rel))
   ```

4. **Alias Map Stability**:
   ```python
   # REQUIRED: Alias map must be sorted for deterministic output

   def save_alias_map(self, output_path):
       with open(output_path, 'w') as f:
           json.dump(
               self.alias_map,
               f,
               indent=2,
               sort_keys=True,  # REQUIRED for determinism
               ensure_ascii=False
           )
   ```

5. **Timestamp Exclusion from Hashing**:
   ```python
   # REQUIRED: Don't include timestamps in deduplication logic

   # ✅ GOOD: Hash only content fields
   def dedup_key(rel):
       return (rel.source, rel.relationship, rel.target, rel.evidence_text)

   # ❌ BAD: Hash includes timestamp (always unique)
   def dedup_key(rel):
       return (rel.source, rel.relationship, rel.target, rel.timestamp)
   ```

**Testing Idempotency**:
```python
def test_consolidation_idempotency():
    """
    Unit test to verify consolidation idempotency.

    Runs consolidation twice and compares outputs.
    """
    chapters = [
        'chapter_01_v14_3_3_20251015_152109.json',
        'chapter_02_v14_3_3_20251016_090512.json',
        'chapter_03_v14_3_3_20251016_101234.json',
    ]

    # Run twice
    result1 = consolidate_chapters(chapters, output_suffix='_test1')
    result2 = consolidate_chapters(chapters, output_suffix='_test2')

    # Load outputs
    consolidated1 = load_json(result1)
    consolidated2 = load_json(result2)

    # Compare
    assert consolidated1 == consolidated2, "Consolidation is not idempotent!"

    # Also check alias maps
    alias1 = load_json(result1.replace('.json', '_aliases.json'))
    alias2 = load_json(result2.replace('.json', '_aliases.json'))
    assert alias1 == alias2, "Alias maps are not deterministic!"

    logger.info("✅ Consolidation idempotency verified")
```

**Continuous Validation**:
- Add idempotency test to CI/CD pipeline
- Run on representative sample (3-5 chapters)
- Fail build if non-determinism detected
- Alert on unexpected differences

---

## Practical Workflow

### Book Structure (OUR BIGGEST DEAL)

**Total**: 480 pages organized as:
- **Front Matter** (pages 1-30): Accolades, Foreword, Introduction
- **Part I** (pages ~31-106): Chapters 1-13 (Structures and Strategies)
- **Part II** (pages ~107-304): Guest Essays 1-26 (VIP Contributors)
- **Part III** (pages ~305-354): Case Studies 1-34 (Organizations)
- **Part IV** (pages ~355-416): Aphoristic Amusings
- **Part V** (pages ~417-436): Frameworks and Checklists
- **Back Matter** (pages ~437-480): Acknowledgments, References

---

### Phase 1: Per-Chapter Extraction

**Step 1**: Extract Front Matter
```bash
python3 scripts/extract_kg_v14_3_3_incremental.py \
  --book our_biggest_deal \
  --section front_matter \
  --pages 1-30 \
  --output kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/

# Monitor extraction
tail -f kg_extraction_front_matter.log

# Run Reflector
python3 scripts/run_reflector_incremental.py \
  --input kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/front_matter_v14_3_3_*.json \
  --output kg_extraction_playbook/output/our_biggest_deal/v14_3_3/analysis/

# Check grade
# If not A+, analyze issues, improve prompts, re-extract
# If A+ ✅, move to Chapter 1
```

**Step 2**: Extract Chapter 1
```bash
python3 scripts/extract_kg_v14_3_3_incremental.py \
  --book our_biggest_deal \
  --section chapter_01 \
  --pages 31-50 \
  --output kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/

# Iterate until A+ ✅
```

**Step 3-15**: Continue through Part I (Chapters 2-13)
```bash
# Extract each chapter, iterate to A+
# Chapters 1-13 cover pages ~31-106
```

**Step 16**: Extract Guest Essay 1
```bash
python3 scripts/extract_kg_v14_3_3_incremental.py \
  --book our_biggest_deal \
  --section essay_01_john_perkins \
  --pages 121-125 \
  --output kg_extraction_playbook/output/our_biggest_deal/v14_3_3/chapters/

# Iterate until A+ ✅
```

**Continue** through all essays, case studies, and remaining sections...

---

### Phase 2: Periodic Consolidation

**Consolidation Point 1**: After Part I (Chapters 1-13)
```bash
python3 scripts/consolidate_chapters.py \
  --book our_biggest_deal \
  --version v14_3_3 \
  --chapters front_matter,chapter_01,chapter_02,...,chapter_13 \
  --output kg_extraction_playbook/output/our_biggest_deal/v14_3_3/consolidations/part_1_consolidated.json

# Process:
# 1. Load all Part I chapter JSONs (already A+ grade)
# 2. Merge relationships
# 3. Run cross-chapter Deduplicator
# 4. Run EntityResolver
# 5. Save consolidated Part I

# Time: ~5-10 minutes (no LLM calls)
```

**Consolidation Point 2**: After Part II (Guest Essays 1-26)
```bash
python3 scripts/consolidate_chapters.py \
  --book our_biggest_deal \
  --version v14_3_3 \
  --chapters essay_01,essay_02,...,essay_26 \
  --output kg_extraction_playbook/output/our_biggest_deal/v14_3_3/consolidations/part_2_consolidated.json
```

**Consolidation Point 3**: After Part III (Case Studies)
```bash
# Same process for Part III
```

---

### Phase 3: Final Whole-Book Consolidation

**Final Pass**: Merge all Parts
```bash
python3 scripts/consolidate_chapters.py \
  --book our_biggest_deal \
  --version v14_3_3 \
  --final \
  --parts part_1,part_2,part_3,part_4,part_5 \
  --output kg_extraction_playbook/output/our_biggest_deal/v14_3_3/final/our_biggest_deal_final.json

# Process:
# 1. Load all part consolidations
# 2. Merge relationships
# 3. Final cross-book Deduplicator
# 4. Final EntityResolver
# 5. Optional: SemanticDeduplicator (if budget allows)
# 6. Save final consolidated KG

# Time: ~10-15 minutes (no LLM calls unless using SemanticDeduplicator)
```

---

## Implementation Plan

### Scripts to Create

**1. extract_kg_v14_3_3_incremental.py**
- Based on extract_kg_v14_3_2_book.py
- Add command-line arguments for section/page ranges
- Use v14_3_3 prompts (with Phase 2 enhancements)
- Output to section-specific files with timestamps

**2. consolidate_chapters.py**
- Load multiple chapter JSON files
- Merge relationships
- Run cross-chapter Deduplicator
- Run EntityResolver
- Optional SemanticDeduplicator
- Save consolidated output

**3. run_reflector_incremental.py**
- Modified version of run_reflector_on_v14_3_2.py
- Accept chapter/section-specific inputs
- Generate section-specific analysis

**4. create_entity_resolver.py**
- New module: EntityResolver
- Canonical name resolution
- Variant detection and merging

---

## Benefits Summary

| Benefit | Description | Impact |
|---------|-------------|--------|
| **No Re-extraction** | Each chapter extracted once to A+ | Cost savings, time efficiency |
| **Iterative Quality** | Can refine prompts between chapters | Continuous improvement |
| **Clean Provenance** | Every state saved with timestamp | Full audit trail, reproducibility |
| **Efficient Dedup** | Only runs when needed (not per-chapter) | Saves time, maintains quality |
| **Scalable** | Works for 480-page book without memory issues | Handles large documents |
| **Flexible** | Can pause/resume at any chapter | Manageable workflow |
| **Parallel-Ready** | Chapters can be extracted concurrently | Future optimization potential |
| **Quality-Locked** | A+ chapters never degrade | Guaranteed quality floor |

---

## Risks and Mitigations

### Risk 1: Cross-Chapter References

**Risk**: "As mentioned in Chapter 2..." might not resolve correctly.

**Mitigation**:
- EntityResolver can track cross-references
- Chapter context preserved in evidence field
- Consolidation pass can flag unresolved references for manual review

### Risk 2: Entity Ambiguity

**Risk**: "John Smith" in Chapter 3 vs Chapter 10 might be different people.

**Mitigation**:
- EntityResolver uses context (title, organization, topic)
- Disambiguation rules based on semantic similarity
- Flag ambiguous cases for human review

### Risk 3: Semantic Drift

**Risk**: Prompt improvements between chapters might create inconsistency.

**Mitigation**:
- Track prompt versions with each extraction
- Final consolidation pass can normalize
- Can re-extract specific chapters if major prompt changes occur

### Risk 4: Consolidation Errors

**Risk**: Merge operations might introduce errors.

**Mitigation**:
- Keep original chapter files unchanged
- Consolidation creates NEW files (never overwrites)
- Can always rebuild from chapter sources
- Run Reflector on consolidated outputs to verify quality

---

## Success Criteria

**Per-Chapter Success**:
- ✅ A+ grade from Reflector (≤2% issue rate)
- ✅ All CRITICAL issues resolved
- ✅ HIGH issues ≤2
- ✅ Extraction time <30 minutes per chapter

**Consolidation Success**:
- ✅ No new issues introduced by merging
- ✅ Deduplication reduces total relationships by 5-15%
- ✅ Entity resolution reduces entity count by 10-20%
- ✅ Consolidation time <15 minutes per pass

**Whole-Book Success**:
- ✅ Final KG maintains A+ grade
- ✅ Complete provenance chain
- ✅ All sections represented
- ✅ Cross-chapter relationships correctly linked

---

## Questions for Review

1. **Architecture**: Does the 3-phase approach (chapter → consolidation → final) make sense?

2. **Deduplication**: Should we run lightweight Deduplicator per-chapter, or only during consolidation?

3. **Entity Resolution**: Should this be a new module, or enhancement to existing modules?

4. **Consolidation Frequency**: Every 5 chapters, every Part, or only at the end?

5. **SemanticDeduplicator**: Run per-consolidation (expensive) or only final pass?

6. **Parallel Extraction**: Should we support parallel chapter extraction from the start?

7. **Quality Gates**: Should consolidation be blocked if any chapter is not A+ grade?

8. **Prompt Evolution**: How to handle prompt improvements mid-book? Re-extract previous chapters?

---

## Next Steps (If Approved)

1. ✅ Create incremental extraction script
2. ✅ Create consolidation script
3. ✅ Create EntityResolver module
4. ✅ Set up directory structure
5. ✅ Extract Front Matter (first test)
6. ✅ Validate approach on 1-2 chapters
7. ✅ Scale to full book

---

## Conclusion

This hybrid incremental approach balances:
- **Quality**: A+ grade per chapter before moving forward
- **Efficiency**: No re-extraction of completed chapters
- **Context**: Cross-chapter processes handled in consolidation
- **Provenance**: Complete version history maintained
- **Scalability**: Works for any size book

The key insight is that **most processing is chapter-local**, and **cross-chapter context can be handled efficiently in lightweight merge operations** that don't require re-extraction.

---

**Ready for review and feedback.**
