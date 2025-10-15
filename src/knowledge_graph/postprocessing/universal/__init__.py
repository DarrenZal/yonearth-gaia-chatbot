"""
Universal Post-Processing Modules

These modules work for ANY content type (books, podcasts, papers, etc.).
They handle common knowledge graph quality issues like pronouns, lists,
vague entities, and invalid predicates.

Available Modules:
-----------------
- FieldNormalizer: Normalize field naming inconsistencies (relationship/predicate) (V14.3.3)
- PronounResolver: Resolve pronouns to their antecedents
- ListSplitter: Split list targets into individual relationships
- ContextEnricher: Replace vague entities with specific ones from context
- PredicateNormalizer: Normalize verbose predicates to standard forms
- PredicateValidator: Validate predicates for logical consistency
- TypeCompatibilityValidator: Validate and auto-fix entity type mismatches (V14.3.7)
- VagueEntityBlocker: Filter out relationships with overly vague entities
- ClaimClassifier: Classify relationships (factual, philosophical, opinion, recommendation)
- Deduplicator: Remove duplicate relationships
- EntityResolver: Resolve entity name variations with deterministic canonicalization (V14.3.3)
- ConfidenceFilter: Filter relationships by p_true thresholds (V14.0)
- SemanticDeduplicator: Remove semantic duplicates using embeddings (V14.1)
"""

from .field_normalizer import FieldNormalizer
from .pronoun_resolver import PronounResolver
from .list_splitter import ListSplitter
from .context_enricher import ContextEnricher
from .predicate_normalizer import PredicateNormalizer
from .predicate_validator import PredicateValidator
from .type_compatibility_validator import TypeCompatibilityValidator
from .vague_entity_blocker import VagueEntityBlocker
from .claim_classifier import ClaimClassifier
from .deduplicator import Deduplicator
from .entity_resolver import EntityResolver
from .confidence_filter import ConfidenceFilter
from .semantic_deduplicator import SemanticDeduplicator

__all__ = [
    "FieldNormalizer",
    "PronounResolver",
    "ListSplitter",
    "ContextEnricher",
    "PredicateNormalizer",
    "PredicateValidator",
    "TypeCompatibilityValidator",
    "VagueEntityBlocker",
    "ClaimClassifier",
    "Deduplicator",
    "EntityResolver",
    "ConfidenceFilter",
    "SemanticDeduplicator",
]
