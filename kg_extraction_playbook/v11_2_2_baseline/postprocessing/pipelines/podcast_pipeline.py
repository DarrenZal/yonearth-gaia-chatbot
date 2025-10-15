"""
Podcast Pipeline Configuration

Pre-configured pipeline for podcast transcript extraction.

Modules included (in priority order):
1. VagueEntityBlocker (30) - Filter overly vague entities
2. ListSplitter (40) - Split list targets
3. ContextEnricher (50) - Enrich vague entities
4. PronounResolver (60) - Resolve pronouns
5. PredicateNormalizer (70) - Normalize predicates
6. PredicateValidator (80) - Validate predicate logic

Note: Only universal modules - no book-specific modules like
praise quotes or bibliographic citations.
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
)


def get_podcast_pipeline(config: Optional[Dict[str, Any]] = None) -> PipelineOrchestrator:
    """
    Get a pre-configured pipeline for podcast transcript extraction.

    This pipeline uses only universal modules, excluding book-specific
    modules like praise quote detection or bibliographic citations.

    Args:
        config: Optional configuration dictionary for modules

    Returns:
        PipelineOrchestrator with podcast-optimized modules
    """
    if config is None:
        config = {}

    # Create universal modules only
    modules = [
        VagueEntityBlocker(config.get('vague_entity_blocker', {})),
        ListSplitter(config.get('list_splitter', {})),
        ContextEnricher(config.get('context_enricher', {})),
        PronounResolver(config.get('pronoun_resolver', {})),
        PredicateNormalizer(config.get('predicate_normalizer', {})),
        PredicateValidator(config.get('predicate_validator', {})),
    ]

    # Create and return orchestrator
    return PipelineOrchestrator(modules)
