"""
Book Pipeline Configuration

Pre-configured pipeline for book extraction with all V11.2 enhancements.

Modules included (in priority order):
1. VagueEntityBlocker (30) - Filter overly vague entities early
2. ListSplitter (40) - Split list targets early (V11.2: enhanced "and" support)
3. ContextEnricher (50) - Enrich vague entities with context
4. PronounResolver (60) - Resolve pronouns to antecedents
5. PredicateNormalizer (70) - Normalize verbose predicates
6. PredicateValidator (80) - Validate predicate logic
7. TitleCompletenessValidator (90) - Validate book titles
8. FigurativeLanguageFilter (100) - Normalize metaphors
9. ClaimClassifier (105) - Classify relationships (V11.2: NEW)
10. Deduplicator (110) - Remove duplicates (V11.2: NEW)

Book-specific modules:
- PraiseQuoteDetector (10) - Detect praise quotes in front matter
- BibliographicCitationParser (20) - Parse bibliographic citations (V11.2: fixed dedication logic)
"""

from typing import Optional, Dict, Any

from ..base import PipelineOrchestrator
from ..universal import (
    VagueEntityBlocker,
    ListSplitter,
    ContextEnricher,
    PronounResolver,
    PredicateNormalizer,
    PredicateValidator,
    ClaimClassifier,
    Deduplicator,
)
from ..content_specific.books import (
    PraiseQuoteDetector,
    BibliographicCitationParser,
    TitleCompletenessValidator,
    FigurativeLanguageFilter,
)


def get_book_pipeline(config: Optional[Dict[str, Any]] = None) -> PipelineOrchestrator:
    """
    Get a pre-configured pipeline for book extraction.

    This pipeline includes all V8 enhancements and is optimized for
    extracting knowledge graphs from books.

    Args:
        config: Optional configuration dictionary for modules

    Returns:
        PipelineOrchestrator with book-optimized modules
    """
    if config is None:
        config = {}

    # Create all modules
    modules = [
        # Book-specific (run first)
        PraiseQuoteDetector(config.get('praise_quote_detector', {})),
        BibliographicCitationParser(config.get('bibliographic_parser', {})),

        # Universal processing
        VagueEntityBlocker(config.get('vague_entity_blocker', {})),
        ListSplitter(config.get('list_splitter', {})),
        ContextEnricher(config.get('context_enricher', {})),
        PronounResolver(config.get('pronoun_resolver', {})),
        PredicateNormalizer(config.get('predicate_normalizer', {})),
        PredicateValidator(config.get('predicate_validator', {})),

        # Book-specific validation
        TitleCompletenessValidator(config.get('title_validator', {})),
        FigurativeLanguageFilter(config.get('figurative_filter', {})),

        # V11.2 NEW: Classification and deduplication (run last)
        ClaimClassifier(config.get('claim_classifier', {})),
        Deduplicator(config.get('deduplicator', {})),
    ]

    # Create and return orchestrator
    return PipelineOrchestrator(modules)
