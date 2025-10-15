"""
Universal Post-Processing Modules

These modules work for ANY content type (books, podcasts, papers, etc.).
They handle common knowledge graph quality issues like pronouns, lists,
vague entities, and invalid predicates.

Available Modules:
-----------------
- PronounResolver: Resolve pronouns to their antecedents
- ListSplitter: Split list targets into individual relationships
- ContextEnricher: Replace vague entities with specific ones from context
- PredicateNormalizer: Normalize verbose predicates to standard forms
- PredicateValidator: Validate predicates for logical consistency
- VagueEntityBlocker: Filter out relationships with overly vague entities
- ClaimClassifier: Classify relationships (factual, philosophical, opinion, recommendation)
- Deduplicator: Remove duplicate relationships
"""

from .pronoun_resolver import PronounResolver
from .list_splitter import ListSplitter
from .context_enricher import ContextEnricher
from .predicate_normalizer import PredicateNormalizer
from .predicate_validator import PredicateValidator
from .vague_entity_blocker import VagueEntityBlocker
from .claim_classifier import ClaimClassifier
from .deduplicator import Deduplicator

__all__ = [
    "PronounResolver",
    "ListSplitter",
    "ContextEnricher",
    "PredicateNormalizer",
    "PredicateValidator",
    "VagueEntityBlocker",
    "ClaimClassifier",
    "Deduplicator",
]
