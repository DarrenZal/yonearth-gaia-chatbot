# Modular Post-Processing System

**Version:** 1.0.0
**Status:** Production-ready
**Created:** 2025-10-13 (ACE Cycle 1, V8 Modularization)

## Overview

A composable "lego block" architecture for post-processing knowledge graph relationships. Modules can be mixed-and-matched for different content types (books, podcasts, academic papers, news, etc.).

## Architecture

```
src/knowledge_graph/postprocessing/
├── base.py                           # Core abstractions
├── universal/                        # Universal modules (all content types)
│   ├── pronoun_resolver.py         # Pronoun → antecedent resolution
│   ├── list_splitter.py             # Split "A, B, and C" lists
│   ├── context_enricher.py          # Replace vague entities
│   ├── predicate_normalizer.py      # Normalize verbose predicates
│   ├── predicate_validator.py       # Validate predicate logic
│   └── vague_entity_blocker.py      # Filter overly vague entities
├── content_specific/
│   └── books/                       # Book-specific modules
│       ├── praise_quote_detector.py # Detect praise quotes in front matter
│       ├── bibliographic_citation_parser.py # Parse citations
│       ├── title_completeness_validator.py # Validate titles
│       └── figurative_language_filter.py # Normalize metaphors
└── pipelines/                       # Pre-configured pipelines
    ├── book_pipeline.py            # Full pipeline for books
    ├── podcast_pipeline.py         # Pipeline for podcasts
    └── custom_pipeline.py          # Builder for custom pipelines
```

## Quick Start

### Option 1: Use Pre-Configured Pipeline

```python
from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline

# Get pre-configured book pipeline
pipeline = get_book_pipeline()

# Create context
context = ProcessingContext(
    content_type='book',
    document_metadata={'author': 'John Doe', 'title': 'Example Book'},
    pages_with_text=[(1, "Page 1 text..."), (2, "Page 2 text...")]
)

# Run pipeline
processed_relationships, stats = pipeline.run(relationships, context)

# Access stats
print(f"Processed: {stats['processed']}")
print(f"Modified: {stats['modified']}")
for module_name, module_stats in stats['modules'].items():
    print(f"  {module_name}: {module_stats}")
```

### Option 2: Build Custom Pipeline

```python
from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import CustomPipeline
from src.knowledge_graph.postprocessing.universal import (
    PronounResolver,
    ListSplitter,
    ContextEnricher
)
from src.knowledge_graph.postprocessing.content_specific.books import (
    PraiseQuoteDetector
)

# Build custom pipeline
pipeline = (
    CustomPipeline()
    .add_module(PraiseQuoteDetector())  # Book-specific
    .add_module(PronounResolver())       # Universal
    .add_module(ListSplitter())          # Universal
    .add_module(ContextEnricher())       # Universal
    .build()
)

# Create context
context = ProcessingContext(content_type='book')

# Run pipeline
processed, stats = pipeline.run(relationships, context)
```

## Available Modules

### Universal Modules (All Content Types)

| Module | Priority | Description | Version |
|--------|----------|-------------|---------|
| **VagueEntityBlocker** | 30 | Filters overly vague entities (this, that, it) | 1.0.0 (V7) |
| **ListSplitter** | 40 | Splits list targets ("A, B, and C") | 1.1.0 (V8) |
| **ContextEnricher** | 50 | Replaces vague entities with context | 1.1.0 (V8) |
| **PronounResolver** | 60 | Resolves pronouns to antecedents | 1.2.0 (V8) |
| **PredicateNormalizer** | 70 | Normalizes verbose predicates | 1.1.0 (V8) |
| **PredicateValidator** | 80 | Validates predicate logic | 1.0.0 (V6) |

### Book-Specific Modules

| Module | Priority | Description | Version |
|--------|----------|-------------|---------|
| **PraiseQuoteDetector** | 10 | Detects praise quotes in front matter | 1.0.0 (V8) |
| **BibliographicCitationParser** | 20 | Parses bibliographic citations | 1.2.0 (V8) |
| **TitleCompletenessValidator** | 90 | Validates book title completeness | 1.0.0 (V6) |
| **FigurativeLanguageFilter** | 100 | Normalizes metaphors to literal forms | 1.1.0 (V8) |

## Module Features

### PronounResolver (V8 Enhanced)
- Generic pronoun handling ("we humans" → "humans")
- Anaphoric resolution with expanding context windows
- Possessive pronoun resolution ("my people" → "Slovenians")
- Multi-pass resolution (same sentence → previous → paragraph)
- Author context awareness

### ListSplitter (V8 Enhanced)
- POS tagging to distinguish adjective series from noun lists
- 'and' conjunction pattern handling (new in V8)
- Preserves adjective series ("physical, mental, spiritual growth")
- Splits true lists ("families, communities and planet")

### ContextEnricher (V8 Enhanced)
- Context-aware replacement using keyword matching
- Document entity mapping ("this handbook" → "Soil Stewardship Handbook")
- Evidence-based enrichment

### PredicateNormalizer (V8 Enhanced)
- Verbose predicate normalization ("flourish with" → "experience")
- Semantic validation (Books can "guide" but not "heal")
- Entity type detection and constraint checking

### PraiseQuoteDetector (V8 New)
- Front matter page detection (pages 1-15)
- Endorsement language pattern matching
- Attribution marker detection (—Name, Title)
- Converts misattributed authorship to endorsement

### BibliographicCitationParser (V8 Enhanced)
- Citation pattern detection
- Authorship reversal (Title authored Author → Author authored Title)
- Endorsement detection (praise quotes, forewords)
- **NEW in V8:** Dedication detection

### FigurativeLanguageFilter (V8 Enhanced)
- Metaphorical term detection (sacred, magical, spiritual)
- Abstract noun detection (compass, journey, gateway)
- **NEW in V8:** Metaphor normalization ("is a compass" → "provides direction")

## Priority System

Modules execute in priority order (lower number = earlier execution):

1. **10-20:** Book-specific early processing (praise quotes, citations)
2. **30-40:** Early universal processing (vague entity blocking, list splitting)
3. **50-60:** Mid-pipeline processing (enrichment, pronouns)
4. **70-80:** Validation and normalization
5. **90-100:** Final validation (titles, figurative language)

Dependencies are validated automatically - modules requiring other modules will only run if dependencies are present.

## Configuration

Each module accepts an optional configuration dictionary:

```python
config = {
    'pronoun_resolver': {
        'resolution_window': 1000,  # characters
        'context_window': 5,        # sentences
        'generic_pronouns': {
            'we humans': 'humans',
            'you': 'readers',
        }
    },
    'list_splitter': {
        'min_list_length': 15,
        'use_pos_tagging': True,
    },
    'context_enricher': {
        'doc_entities': {
            'this book': 'My Custom Book Title',
        }
    }
}

pipeline = get_book_pipeline(config)
```

## Processing Context

The `ProcessingContext` dataclass provides shared context for all modules:

```python
@dataclass
class ProcessingContext:
    content_type: str  # 'book', 'podcast', 'academic', 'news'
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    pages_with_text: Optional[List[Tuple[int, str]]] = None
    audio_timestamps: Optional[List[Dict[str, Any]]] = None
    sections: Optional[List[Dict[str, Any]]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    extraction_version: Optional[str] = None
```

## Pre-Configured Pipelines

### Book Pipeline

All V8 modules enabled:
- Praise quote detection
- Bibliographic citation parsing
- All universal modules
- Title validation
- Figurative language normalization

```python
from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline
pipeline = get_book_pipeline()
```

### Podcast Pipeline

Universal modules only (no book-specific):
- Vague entity blocking
- List splitting
- Context enrichment
- Pronoun resolution
- Predicate normalization
- Predicate validation

```python
from src.knowledge_graph.postprocessing.pipelines import get_podcast_pipeline
pipeline = get_podcast_pipeline()
```

## Custom Pipeline Builder

```python
from src.knowledge_graph.postprocessing.pipelines import CustomPipeline

pipeline = (
    CustomPipeline()
    .add_module(Module1())
    .add_module(Module2())
    .remove_module('ModuleName')  # Remove by name
    .add_modules([Module3(), Module4()])  # Add multiple
    .build()
)
```

## Statistics

Each module tracks:
- `processed_count`: Total relationships processed
- `modified_count`: Relationships modified by module
- Module-specific stats (e.g., `resolved`, `blocked`, `normalized`)

Pipeline-level stats:
- `total_processed`: Total input relationships
- `total_output`: Total output relationships
- `total_modified`: Total modifications across all modules
- `modules`: Dict of per-module statistics

## Development

### Creating a New Module

```python
from typing import List, Dict, Any, Optional
from ..base import PostProcessingModule, ProcessingContext

class MyNewModule(PostProcessingModule):
    # Metadata
    name = "MyNewModule"
    description = "Does something useful"
    content_types = ["all"]  # or ["book", "podcast"]
    priority = 50
    dependencies = []  # or ["OtherModuleName"]
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Initialize module-specific attributes
        self.my_setting = self.config.get('my_setting', 'default')

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        for rel in relationships:
            # Process relationship
            # ...
            if modified:
                self.stats['modified_count'] += 1
            processed.append(rel)

        # Log results
        logger.info(f"   {self.name}: {self.stats['modified_count']} modified")

        return processed
```

### Adding to a Pipeline

```python
# Add to existing pipeline configuration
from .my_new_module import MyNewModule

def get_book_pipeline(config):
    modules = [
        # ... existing modules ...
        MyNewModule(config.get('my_new_module', {})),
    ]
    return PipelineOrchestrator(modules)
```

## Version History

- **1.0.0 (2025-10-13):** Initial modular system extracted from V8 monolithic extractor
  - 6 universal modules
  - 4 book-specific modules
  - 2 pre-configured pipelines (book, podcast)
  - CustomPipeline builder
  - Full V8 feature parity

## Future Enhancements

Potential additions:
- **Podcast-specific modules:** Speaker attribution, filler word removal, timestamp handling
- **Academic-specific modules:** Citation parsing, equation handling, reference validation
- **News-specific modules:** Dateline parsing, byline extraction, quote attribution
- **Additional pipelines:** Academic, news, social media
- **Module marketplace:** Community-contributed modules

## Related Documentation

- [ACE Framework Design](../../../docs/ACE_FRAMEWORK_DESIGN.md)
- [V8 Extraction Results](../../../docs/knowledge_graph/ACE_CYCLE_1_COMPLETE.md)
- [Knowledge Graph README](../../../docs/knowledge_graph/README.md)
