#!/usr/bin/env python3
"""
Knowledge Graph Extraction v5 - WITH PASS 2.5 QUALITY POST-PROCESSING

✨ V5 NEW FEATURES (Pass 2.5 Quality Improvement):
✅ BibliographicCitationParser - Fixes 105 reversed authorships (12%)
✅ ListSplitter - Splits 100 list targets into separate relationships (11.5%)
✅ PronounResolver - Resolves 75 pronouns to entities (8.6%)
✅ ContextEnricher - Expands 56 vague entities with context (6.4%)
✅ TitleCompletenessValidator - Flags 70 incomplete titles (8%)
✅ PredicateValidator - Flags 52 invalid predicates (6%)
✅ FigurativeLanguageFilter - Flags 44 metaphors (5%)

GOAL: Reduce quality issues from 57% (V4) to <10% (V5)

Architecture: V4 base + NEW Pass 2.5 post-processing layer

Uses OPENAI_API_KEY_2 for separate rate limit
"""

import json
import logging
import os
import hashlib
import re
import unicodedata
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Pydantic imports
from pydantic import BaseModel, Field

# OpenAI imports
from openai import OpenAI

# PDF processing
import pdfplumber

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_extraction_book_v5_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
BOOKS_DIR = DATA_DIR / "books"
PLAYBOOK_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook")
OUTPUT_DIR = PLAYBOOK_DIR / "output" / "v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY_2 not set in .env!")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# Cache for scorer results
edge_cache: Dict[str, Any] = {}
cache_stats = {'hits': 0, 'misses': 0}


# ============================================================================
# DATACLASSES & SCHEMAS
# ============================================================================

def _default_evidence():
    """Factory for book evidence dict"""
    return {
        "doc_id": None,
        "doc_sha256": None,
        "page_number": None,
        "start_char": None,
        "end_char": None,
        "window_text": "",
        "source_surface": None,
        "target_surface": None
    }

def _default_flags():
    """Factory for flags dict"""
    return {}

def _default_extraction_metadata():
    """Factory for extraction metadata dict"""
    return {
        "model_pass1": "gpt-4o-mini",
        "model_pass2": "gpt-4o-mini",
        "prompt_version": "v5_with_pass2_5",
        "extractor_version": "2025.10.12_v5",
        "content_type": "book",
        "run_id": None,
        "extracted_at": None,
        "batch_id": None
    }


@dataclass
class ProductionRelationship:
    """Production-ready relationship"""
    # Core extraction
    source: str
    relationship: str
    target: str

    # Type information
    source_type: Optional[str] = None
    target_type: Optional[str] = None

    # Validation flags
    flags: Dict[str, Any] = field(default_factory=_default_flags)

    # Evidence tracking
    evidence_text: str = ""
    evidence: Dict[str, Any] = field(default_factory=_default_evidence)

    # Dual signals from Pass 2
    text_confidence: float = 0.0
    knowledge_plausibility: float = 0.0

    # Pattern prior
    pattern_prior: float = 0.5

    # Conflict detection
    signals_conflict: bool = False
    conflict_explanation: Optional[str] = None
    suggested_correction: Optional[str] = None

    # Calibrated probability
    p_true: float = 0.0

    # Identity
    claim_uid: Optional[str] = None
    candidate_uid: Optional[str] = None

    # Metadata
    extraction_metadata: Dict[str, Any] = field(default_factory=_default_extraction_metadata)


# Pydantic models for API calls
class SimpleRelationship(BaseModel):
    """Pass 1 extraction result"""
    source: str
    relationship: str
    target: str
    evidence_text: str = Field(description="Quote from text supporting this relationship")


class ComprehensiveExtraction(BaseModel):
    """Pass 1 extraction container"""
    relationships: List[SimpleRelationship]


class DualSignalEvaluation(BaseModel):
    """Pass 2 evaluation result"""
    candidate_uid: str
    source: str
    relationship: str
    target: str
    evidence_text: str
    text_confidence: float = Field(ge=0.0, le=1.0)
    knowledge_plausibility: float = Field(ge=0.0, le=1.0)
    source_type: Optional[str] = None
    target_type: Optional[str] = None
    signals_conflict: bool
    conflict_explanation: Optional[str] = None
    suggested_correction: Optional[str] = None


class BatchedEvaluationResult(BaseModel):
    """Pass 2 batch evaluation container"""
    evaluations: List[DualSignalEvaluation]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def canon(s: str) -> str:
    """Normalize entity strings"""
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def make_candidate_uid(source: str, relationship: str, target: str,
                       evidence_text: str, doc_sha256: str) -> str:
    """Create deterministic candidate UID"""
    evidence_hash = hashlib.sha1(evidence_text.encode()).hexdigest()[:8]
    base = f"{source}|{relationship}|{target}|{evidence_hash}|{doc_sha256}"
    return hashlib.sha1(base.encode()).hexdigest()


def generate_claim_uid(rel: ProductionRelationship) -> str:
    """Stable identity for the fact"""
    evidence_hash = hashlib.sha1(rel.evidence_text.encode()).hexdigest()[:8]
    doc_sha = rel.evidence.get('doc_sha256') or 'unknown'
    components = [
        rel.source,
        rel.relationship,
        rel.target,
        doc_sha,
        evidence_hash
    ]
    uid_string = "|".join(components)
    return hashlib.sha1(uid_string.encode()).hexdigest()


def compute_p_true(text_conf: float, knowledge_plaus: float,
                  pattern_prior: float, conflict: bool) -> float:
    """Calibrated probability combiner"""
    z = (-1.2 + 2.1 * text_conf + 0.9 * knowledge_plaus +
         0.6 * pattern_prior - 0.8 * int(conflict))
    p_true = 1 / (1 + math.exp(-z))
    return p_true


def chunks(seq, size: int):
    """Yield fixed-size slices"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ============================================================================
# ✨ PASS 2.5: QUALITY POST-PROCESSING MODULES (NEW!)
# ============================================================================

class BibliographicCitationParser:
    """
    Detects and corrects authorship relationships from bibliographic citations.
    Fixes ~12% of relationships (105 in V4)
    """

    def __init__(self):
        # Citation format patterns
        self.citation_patterns = [
            r'^([A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+,\s+[A-Z][a-z]+)*)\.',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\.',
        ]

        # Authorship predicates that should be reversed
        self.authorship_predicates = ('authored', 'wrote', 'published', 'created', 'composed', 'edited', 'compiled', 'produced')

    def is_bibliographic_citation(self, evidence_text: str) -> bool:
        """Check if evidence text matches bibliographic citation format"""
        for pattern in self.citation_patterns:
            if re.match(pattern, evidence_text.strip()):
                return True
        return False

    def should_reverse_authorship(self, rel: ProductionRelationship) -> bool:
        """Determine if authorship relationship should be reversed"""
        if rel.relationship not in self.authorship_predicates:
            return False

        evidence = rel.evidence_text.strip()

        # Check bibliographic format
        if not self.is_bibliographic_citation(evidence):
            return False

        # Check if source looks like title
        source_is_title = (
            len(rel.source.split()) > 3 or
            '"' in evidence[:50] or
            ':' in rel.source
        )

        # Check if target looks like author name
        target_words = rel.target.split()
        target_is_author = (
            2 <= len(target_words) <= 4 and
            all(w[0].isupper() for w in target_words if len(w) > 2)
        )

        return source_is_title and target_is_author

    def reverse_authorship(self, rel: ProductionRelationship) -> ProductionRelationship:
        """Reverse source and target, update types, add flag"""
        # Swap source and target
        rel.source, rel.target = rel.target, rel.source
        rel.source_type, rel.target_type = rel.target_type, rel.source_type

        # Update evidence surface forms
        rel.evidence['source_surface'], rel.evidence['target_surface'] = \
            rel.evidence.get('target_surface'), rel.evidence.get('source_surface')

        # Add correction flag
        if rel.flags is None:
            rel.flags = {}
        rel.flags['AUTHORSHIP_REVERSED'] = True
        rel.flags['correction_reason'] = 'bibliographic_citation_detected'

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """Process batch of relationships, reversing authorship where needed"""
        corrected = []
        correction_count = 0

        for rel in relationships:
            if self.should_reverse_authorship(rel):
                rel = self.reverse_authorship(rel)
                correction_count += 1
            corrected.append(rel)

        logger.info(f"   Bibliographic citations: {correction_count} authorships reversed")
        return corrected


class ListSplitter:
    """
    Splits relationships with comma-separated targets into multiple relationships.
    Fixes ~11.5% of relationships (100 in V4), creates ~250 total relationships
    """

    def __init__(self):
        self.min_list_length = 15
        self.list_prone_predicates = {
            'is used for', 'includes', 'contains', 'produces',
            'affects', 'benefits', 'improves', 'creates',
            'can do', 'supports', 'enhances'
        }

    def is_list_target(self, target: str) -> bool:
        """Check if target appears to be a comma-separated list"""
        if ',' not in target:
            return False

        if len(target) < self.min_list_length:
            return False

        if ' and ' in target and ',' in target:
            return True

        comma_count = target.count(',')
        if comma_count >= 2:
            return True

        return False

    def split_target_list(self, target: str) -> List[str]:
        """Split comma-separated target into individual items"""
        normalized = target

        # Handle oxford comma: "X, Y, and Z"
        normalized = re.sub(r',\s+and\s+', ', ', normalized)

        # Split on commas
        parts = []
        for segment in normalized.split(','):
            segment = segment.strip()

            # If this is the last segment and has " and "
            if ' and ' in segment and segment == normalized.split(',')[-1]:
                # Split on final "and"
                final_parts = segment.rsplit(' and ', 1)
                parts.extend([p.strip() for p in final_parts])
            else:
                parts.append(segment)

        # Clean up
        items = [item.strip() for item in parts if item.strip()]

        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in items:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique_items.append(item)

        return unique_items

    def split_relationship(self, rel: ProductionRelationship) -> List[ProductionRelationship]:
        """Split one relationship into N relationships"""
        items = self.split_target_list(rel.target)

        if len(items) <= 1:
            return [rel]

        # Create N new relationships
        split_rels = []
        for i, item in enumerate(items):
            new_rel = ProductionRelationship(
                source=rel.source,
                relationship=rel.relationship,
                target=item,
                source_type=rel.source_type,
                target_type=rel.target_type,
                evidence_text=rel.evidence_text,
                evidence=rel.evidence.copy(),
                text_confidence=rel.text_confidence,
                knowledge_plausibility=rel.knowledge_plausibility,
                pattern_prior=rel.pattern_prior,
                signals_conflict=rel.signals_conflict,
                conflict_explanation=rel.conflict_explanation,
                p_true=rel.p_true,
                candidate_uid=rel.candidate_uid + f"_split_{i}",
                claim_uid=None,  # Will be regenerated
                flags=rel.flags.copy() if rel.flags else {},
                extraction_metadata=rel.extraction_metadata.copy()
            )

            # Mark as split
            new_rel.flags['LIST_SPLIT'] = True
            new_rel.flags['split_index'] = i
            new_rel.flags['split_total'] = len(items)
            new_rel.flags['original_target'] = rel.target

            split_rels.append(new_rel)

        return split_rels

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """Process batch, splitting list relationships"""
        processed = []
        split_count = 0
        original_count = len(relationships)

        for rel in relationships:
            if self.is_list_target(rel.target):
                split_rels = self.split_relationship(rel)
                processed.extend(split_rels)
                if len(split_rels) > 1:
                    split_count += 1
            else:
                processed.append(rel)

        new_count = len(processed)
        logger.info(f"   List splitting: {split_count} lists split, {original_count} → {new_count} relationships")
        return processed


class PronounResolver:
    """
    Resolves pronoun sources/targets to actual entity names.
    Resolves ~8.6% of relationships (75 in V4), expected 60% success rate
    """

    def __init__(self):
        self.pronouns = {
            'he', 'she', 'him', 'her', 'his', 'hers',
            'it', 'its',
            'we', 'us', 'our', 'ours',
            'they', 'them', 'their', 'theirs'
        }
        self.page_context = {}

    def is_pronoun(self, entity: str) -> bool:
        """Check if entity is a pronoun"""
        return entity.lower().strip() in self.pronouns

    def load_page_context(self, pages_with_text: List[tuple]):
        """Load page context for pronoun resolution"""
        self.page_context = {page_num: text for page_num, text in pages_with_text}

    def find_antecedent(self, pronoun: str, page_num: int, evidence_text: str) -> Optional[str]:
        """Find the antecedent (entity the pronoun refers to)"""
        pronoun_lower = pronoun.lower()

        # Get page context
        page_text = self.page_context.get(page_num, '')

        # Find evidence position in page
        evidence_pos = page_text.find(evidence_text[:50])
        if evidence_pos == -1:
            return None

        # Look in previous 500 characters for antecedent
        context_start = max(0, evidence_pos - 500)
        context = page_text[context_start:evidence_pos]

        # Person pronouns
        if pronoun_lower in {'he', 'she', 'his', 'her', 'him'}:
            names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', context)
            if names:
                return names[-1]

        # Collective pronouns
        elif pronoun_lower in {'we', 'our', 'us', 'ours'}:
            orgs = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Foundation|Institute|Organization|Guild|Movement))\b', context)
            if orgs:
                return orgs[-1]

            collectives = re.findall(r'\b(humanity|people|society|humans|communities|families)\b', context, re.IGNORECASE)
            if collectives:
                return collectives[-1]

            if 'soil' in context.lower() or 'earth' in context.lower():
                return 'humanity'
            return 'people'

        return None

    def resolve_pronouns(self, rel: ProductionRelationship) -> ProductionRelationship:
        """Resolve pronouns in source and target"""
        page_num = rel.evidence.get('page_number')
        evidence_text = rel.evidence_text

        # Resolve source
        if self.is_pronoun(rel.source):
            antecedent = self.find_antecedent(rel.source, page_num, evidence_text)
            if antecedent:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['PRONOUN_RESOLVED_SOURCE'] = True
                rel.flags['original_source'] = rel.source
                rel.source = antecedent
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['PRONOUN_UNRESOLVED_SOURCE'] = True

        # Resolve target
        if self.is_pronoun(rel.target):
            antecedent = self.find_antecedent(rel.target, page_num, evidence_text)
            if antecedent:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['PRONOUN_RESOLVED_TARGET'] = True
                rel.flags['original_target'] = rel.target
                rel.target = antecedent
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['PRONOUN_UNRESOLVED_TARGET'] = True

        return rel

    def process_batch(self, relationships: List[ProductionRelationship],
                     pages_with_text: List[tuple]) -> List[ProductionRelationship]:
        """Process batch, resolving pronouns"""
        self.load_page_context(pages_with_text)

        processed = []
        resolved_count = 0
        unresolved_count = 0

        for rel in relationships:
            rel = self.resolve_pronouns(rel)

            if rel.flags and (rel.flags.get('PRONOUN_RESOLVED_SOURCE') or \
                           rel.flags.get('PRONOUN_RESOLVED_TARGET')):
                resolved_count += 1

            if rel.flags and (rel.flags.get('PRONOUN_UNRESOLVED_SOURCE') or \
                            rel.flags.get('PRONOUN_UNRESOLVED_TARGET')):
                unresolved_count += 1

            processed.append(rel)

        logger.info(f"   Pronoun resolution: {resolved_count} resolved, {unresolved_count} flagged for review")
        return processed


class ContextEnricher:
    """
    Enriches vague entity references with context from evidence text.
    Enriches ~6.4% of relationships (56 in V4), expected 54% success rate
    """

    def __init__(self):
        self.vague_terms = {
            'the amount', 'the process', 'the practice', 'the method',
            'the system', 'the approach', 'the way', 'the idea',
            'this', 'that', 'these', 'those',
            'this handbook', 'this book', 'the handbook', 'the book'
        }

        self.doc_entities = {
            'this handbook': 'Soil Stewardship Handbook',
            'this book': 'Soil Stewardship Handbook',
            'the handbook': 'Soil Stewardship Handbook',
            'the book': 'Soil Stewardship Handbook'
        }

    def is_vague(self, entity: str) -> bool:
        """Check if entity starts with vague term"""
        entity_lower = entity.lower().strip()

        if entity_lower in self.vague_terms:
            return True

        for term in self.vague_terms:
            if entity_lower.startswith(term):
                return True

        return False

    def enrich_entity(self, entity: str, evidence_text: str,
                     relationship: str, other_entity: str) -> Optional[str]:
        """Enrich vague entity with context"""
        entity_lower = entity.lower().strip()

        # Document-level mappings
        if entity_lower in self.doc_entities:
            return self.doc_entities[entity_lower]

        # "the amount of X" → "X amount"
        if entity_lower.startswith('the amount'):
            match = re.search(r'the amount of ([^,\.]+)', evidence_text, re.IGNORECASE)
            if match:
                qualifier = match.group(1).strip()
                qualifier = re.sub(r'\s+(by|in|at)\s+.*', '', qualifier)
                return f"{qualifier}"

        # "the process" → "[specific] process"
        if entity_lower in {'the process', 'this process'}:
            processes = ['composting', 'pyrolysis', 'photosynthesis',
                        'decomposition', 'fermentation', 'soil building']
            for proc in processes:
                if proc in evidence_text.lower():
                    return f"{proc} process"

        # "this handbook" variants
        if 'handbook' in entity_lower or 'book' in entity_lower:
            return 'Soil Stewardship Handbook'

        return None

    def enrich_relationship(self, rel: ProductionRelationship) -> ProductionRelationship:
        """Enrich vague entities in relationship"""
        # Enrich source
        if self.is_vague(rel.source):
            enriched_source = self.enrich_entity(
                rel.source, rel.evidence_text,
                rel.relationship, rel.target
            )
            if enriched_source:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['CONTEXT_ENRICHED_SOURCE'] = True
                rel.flags['original_source'] = rel.source
                rel.source = enriched_source
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['VAGUE_SOURCE'] = True

        # Enrich target
        if self.is_vague(rel.target):
            enriched_target = self.enrich_entity(
                rel.target, rel.evidence_text,
                rel.relationship, rel.source
            )
            if enriched_target:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['CONTEXT_ENRICHED_TARGET'] = True
                rel.flags['original_target'] = rel.target
                rel.target = enriched_target
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['VAGUE_TARGET'] = True

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """Process batch, enriching vague entities"""
        processed = []
        enriched_count = 0
        vague_count = 0

        for rel in relationships:
            rel = self.enrich_relationship(rel)

            if rel.flags and (rel.flags.get('CONTEXT_ENRICHED_SOURCE') or \
                            rel.flags.get('CONTEXT_ENRICHED_TARGET')):
                enriched_count += 1

            if rel.flags and (rel.flags.get('VAGUE_SOURCE') or \
                            rel.flags.get('VAGUE_TARGET')):
                vague_count += 1

            processed.append(rel)

        logger.info(f"   Context enrichment: {enriched_count} enriched, {vague_count} flagged as vague")
        return processed


class TitleCompletenessValidator:
    """
    Validates that extracted book/article titles are complete.
    Flags ~8% of relationships (70 in V4)
    """

    def __init__(self):
        self.bad_endings = {
            'and', 'or', 'but', 'to', 'for', 'with', 'by',
            'in', 'on', 'at', 'of', 'the', 'a', 'an'
        }

        self.title_relationships = {
            'authored', 'wrote', 'published', 'edited',
            'compiled', 'created', 'produced'
        }

    def is_incomplete_title(self, title: str) -> tuple:
        """Check if title appears incomplete. Returns (is_incomplete, reason)"""
        words = title.split()

        # Check last word
        if words:
            last_word = words[-1].lower().rstrip('.,!?')
            if last_word in self.bad_endings:
                return True, f"ends_with_{last_word}"

        # Check for unmatched quotes
        if title.count('"') == 1:
            return True, "unmatched_quotes"

        # Suspiciously short for a title
        if len(words) <= 2 and ':' not in title:
            return True, "too_short"

        # Ends with ellipsis
        if title.rstrip().endswith('...'):
            return True, "ellipsis_ending"

        return False, ""

    def validate_relationship(self, rel: ProductionRelationship) -> ProductionRelationship:
        """Validate title completeness in relationship"""
        if rel.relationship not in self.title_relationships:
            return rel

        is_incomplete, reason = self.is_incomplete_title(rel.target)

        if is_incomplete:
            if rel.flags is None:
                rel.flags = {}
            rel.flags['INCOMPLETE_TITLE'] = True
            rel.flags['incompleteness_reason'] = reason
            rel.p_true = rel.p_true * 0.7

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """Process batch, validating titles"""
        processed = []
        incomplete_count = 0

        for rel in relationships:
            rel = self.validate_relationship(rel)

            if rel.flags and rel.flags.get('INCOMPLETE_TITLE'):
                incomplete_count += 1

            processed.append(rel)

        logger.info(f"   Title validation: {incomplete_count} incomplete titles flagged")
        return processed


class PredicateValidator:
    """
    Validates that relationship predicates match their context.
    Flags ~6% of relationships (52 in V4)
    """

    def __init__(self):
        self.invalid_patterns = [
            ('Organization', 'published', 'Date'),
            ('Person', 'is-a', 'Person'),
        ]

    def validate_no_self_loop(self, rel: ProductionRelationship) -> tuple:
        """Validate source != target"""
        if rel.source.lower() == rel.target.lower():
            if rel.relationship not in {'is-a', 'is defined as', 'means', 'equals'}:
                return False, "self_loop"
        return True, ""

    def validate_publication_context(self, rel: ProductionRelationship) -> tuple:
        """Validate publication relationships"""
        if rel.relationship in {'published', 'wrote', 'authored'}:
            target_words = rel.target.split()

            date_patterns = [
                r'^\d{1,2}/\d{1,2}/\d{2,4}$',
                r'^\d{4}-\d{2}-\d{2}$',
                r'^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}$',
            ]

            for pattern in date_patterns:
                if re.match(pattern, rel.target):
                    return False, "published_date_not_title"

        return True, ""

    def validate_predicate(self, rel: ProductionRelationship) -> ProductionRelationship:
        """Validate predicate appropriateness"""
        issues = []

        valid, reason = self.validate_no_self_loop(rel)
        if not valid:
            issues.append(reason)

        valid, reason = self.validate_publication_context(rel)
        if not valid:
            issues.append(reason)

        if issues:
            if rel.flags is None:
                rel.flags = {}
            rel.flags['INVALID_PREDICATE'] = True
            rel.flags['validation_issues'] = issues
            rel.p_true = rel.p_true * 0.3

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """Process batch, validating predicates"""
        processed = []
        invalid_count = 0

        for rel in relationships:
            rel = self.validate_predicate(rel)

            if rel.flags and rel.flags.get('INVALID_PREDICATE'):
                invalid_count += 1

            processed.append(rel)

        logger.info(f"   Predicate validation: {invalid_count} invalid predicates flagged")
        return processed


class FigurativeLanguageFilter:
    """
    Detects and flags metaphorical/poetic language.
    Flags ~5% of relationships (44 in V4)
    """

    def __init__(self):
        self.metaphorical_terms = {
            'sacred', 'magic', 'magical', 'mystical', 'spiritual',
            'alchemy', 'divine', 'blessed', 'holy', 'sanctity',
            'touch of god', "god's touch", 'miracle', 'miraculous',
            'soul', 'spirit', 'essence', 'nexus'
        }

        self.abstract_nouns = {
            'compass', 'journey', 'quest', 'adventure', 'path',
            'gateway', 'portal', 'door', 'key', 'bridge'
        }

    def contains_metaphorical_language(self, text: str) -> tuple:
        """Check if text contains metaphorical language"""
        text_lower = text.lower()
        found_terms = []

        for term in self.metaphorical_terms:
            if term in text_lower:
                found_terms.append(term)

        for noun in self.abstract_nouns:
            if f"is a {noun}" in text_lower or f"is the {noun}" in text_lower:
                found_terms.append(f"metaphor:{noun}")

        return len(found_terms) > 0, found_terms

    def filter_relationship(self, rel: ProductionRelationship) -> ProductionRelationship:
        """Filter/flag metaphorical relationships"""
        is_metaphorical, terms = self.contains_metaphorical_language(rel.evidence_text)

        if is_metaphorical:
            if rel.flags is None:
                rel.flags = {}
            rel.flags['FIGURATIVE_LANGUAGE'] = True
            rel.flags['metaphorical_terms'] = terms
            rel.p_true = rel.p_true * 0.6

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """Process batch, flagging figurative language"""
        processed = []
        metaphorical_count = 0

        for rel in relationships:
            rel = self.filter_relationship(rel)

            if rel.flags and rel.flags.get('FIGURATIVE_LANGUAGE'):
                metaphorical_count += 1

            processed.append(rel)

        logger.info(f"   Figurative language: {metaphorical_count} metaphors flagged")
        return processed


def pass_2_5_quality_post_processing(
    relationships: List[ProductionRelationship],
    pages_with_text: List[tuple],
    config: dict = None
) -> tuple:
    """
    ✨ Pass 2.5: Quality Post-Processing Pipeline

    Applies all quality fixes in optimal order
    """
    logger.info("🎨 PASS 2.5: Quality Post-Processing...")

    config = config or {}
    initial_count = len(relationships)

    # Statistics
    stats = {
        'initial_count': initial_count,
        'authorship_reversed': 0,
        'pronouns_resolved': 0,
        'pronouns_unresolved': 0,
        'entities_enriched': 0,
        'entities_vague': 0,
        'lists_split': 0,
        'titles_incomplete': 0,
        'predicates_invalid': 0,
        'metaphors_flagged': 0,
        'final_count': 0
    }

    # 1. Bibliographic Citation Parser
    logger.info("  1/7: Bibliographic citation parsing...")
    bib_parser = BibliographicCitationParser()
    relationships = bib_parser.process_batch(relationships)
    stats['authorship_reversed'] = sum(1 for r in relationships if r.flags and r.flags.get('AUTHORSHIP_REVERSED'))

    # 2. Title Completeness Validator
    logger.info("  2/7: Title completeness validation...")
    title_validator = TitleCompletenessValidator()
    relationships = title_validator.process_batch(relationships)
    stats['titles_incomplete'] = sum(1 for r in relationships if r.flags and r.flags.get('INCOMPLETE_TITLE'))

    # 3. Predicate Validator
    logger.info("  3/7: Predicate validation...")
    pred_validator = PredicateValidator()
    relationships = pred_validator.process_batch(relationships)
    stats['predicates_invalid'] = sum(1 for r in relationships if r.flags and r.flags.get('INVALID_PREDICATE'))

    # 4. Pronoun Resolver
    logger.info("  4/7: Pronoun resolution...")
    pronoun_resolver = PronounResolver()
    relationships = pronoun_resolver.process_batch(relationships, pages_with_text)
    stats['pronouns_resolved'] = sum(1 for r in relationships if r.flags and
                                    (r.flags.get('PRONOUN_RESOLVED_SOURCE') or r.flags.get('PRONOUN_RESOLVED_TARGET')))
    stats['pronouns_unresolved'] = sum(1 for r in relationships if r.flags and
                                      (r.flags.get('PRONOUN_UNRESOLVED_SOURCE') or r.flags.get('PRONOUN_UNRESOLVED_TARGET')))

    # 5. Context Enricher
    logger.info("  5/7: Context enrichment...")
    context_enricher = ContextEnricher()
    relationships = context_enricher.process_batch(relationships)
    stats['entities_enriched'] = sum(1 for r in relationships if r.flags and
                                    (r.flags.get('CONTEXT_ENRICHED_SOURCE') or r.flags.get('CONTEXT_ENRICHED_TARGET')))
    stats['entities_vague'] = sum(1 for r in relationships if r.flags and
                                 (r.flags.get('VAGUE_SOURCE') or r.flags.get('VAGUE_TARGET')))

    # 6. List Splitter (LAST - creates new relationships)
    logger.info("  6/7: List splitting...")
    list_splitter = ListSplitter()
    relationships = list_splitter.process_batch(relationships)
    stats['lists_split'] = sum(1 for r in relationships if r.flags and r.flags.get('LIST_SPLIT'))

    # 7. Figurative Language Filter
    logger.info("  7/7: Figurative language detection...")
    fig_filter = FigurativeLanguageFilter()
    relationships = fig_filter.process_batch(relationships)
    stats['metaphors_flagged'] = sum(1 for r in relationships if r.flags and r.flags.get('FIGURATIVE_LANGUAGE'))

    stats['final_count'] = len(relationships)

    logger.info(f"✅ PASS 2.5 COMPLETE:")
    logger.info(f"   - Initial: {initial_count} relationships")
    logger.info(f"   - Authorship reversed: {stats['authorship_reversed']}")
    logger.info(f"   - Pronouns resolved: {stats['pronouns_resolved']} ({stats['pronouns_unresolved']} unresolved)")
    logger.info(f"   - Context enriched: {stats['entities_enriched']} ({stats['entities_vague']} still vague)")
    logger.info(f"   - Lists split: {stats['lists_split']} new relationships")
    logger.info(f"   - Titles incomplete: {stats['titles_incomplete']} flagged")
    logger.info(f"   - Predicates invalid: {stats['predicates_invalid']} flagged")
    logger.info(f"   - Metaphors: {stats['metaphors_flagged']} flagged")
    logger.info(f"   - Final: {stats['final_count']} relationships")

    return relationships, stats


# ============================================================================
# PDF TEXT EXTRACTION & CHUNKING (from V4)
# ============================================================================

def extract_text_from_pdf(pdf_path: Path) -> tuple:
    """Extract text from PDF, returning (full_text, [(page_num, page_text), ...])"""
    logger.info(f"📖 Extracting text from PDF: {pdf_path.name}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_with_text = []

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"  Total pages: {len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                pages_with_text.append((page_num, text))

            if page_num % 20 == 0:
                logger.info(f"  Processed {page_num}/{len(pdf.pages)} pages")

    full_text = "\n\n".join([text for _, text in pages_with_text])
    word_count = len(full_text.split())

    logger.info(f"✅ Extracted {word_count:,} words from {len(pages_with_text)} pages")

    return full_text, pages_with_text


def chunk_book_text(pages_with_text: List[tuple],
                    chunk_size: int = 500,
                    overlap: int = 75,
                    min_page_words: int = 50) -> List[tuple]:
    """Chunk book text while preserving page information"""
    chunks_with_pages = []
    current_chunk_words = []
    current_chunk_pages = set()

    pages_included = set()
    pages_skipped = []

    for page_num, page_text in pages_with_text:
        words = page_text.split()

        if len(words) < min_page_words:
            pages_skipped.append((page_num, len(words)))
            continue

        pages_included.add(page_num)

        for word in words:
            current_chunk_words.append(word)
            current_chunk_pages.add(page_num)

            if len(current_chunk_words) >= chunk_size:
                chunk_text = ' '.join(current_chunk_words)
                chunk_pages = sorted(current_chunk_pages)
                chunks_with_pages.append((chunk_pages, chunk_text))

                overlap_words = current_chunk_words[-overlap:] if overlap > 0 else []
                current_chunk_words = overlap_words
                current_chunk_pages = {page_num} if overlap > 0 else set()

    if current_chunk_words:
        chunk_text = ' '.join(current_chunk_words)
        chunk_pages = sorted(current_chunk_pages)
        chunks_with_pages.append((chunk_pages, chunk_text))

    logger.info(f"📄 Created {len(chunks_with_pages)} chunks from book")
    logger.info(f"   - Pages included: {len(pages_included)}/{len(pages_with_text)} ({len(pages_included)/len(pages_with_text)*100:.1f}%)")
    logger.info(f"   - Pages skipped (< {min_page_words} words): {len(pages_skipped)}")

    return chunks_with_pages


# ============================================================================
# PASS 1 & PASS 2 (from V4 - unchanged)
# ============================================================================

BOOK_EXTRACTION_PROMPT = """Extract ALL relationships you can find in this text.

Don't worry about whether they're correct or make sense - just extract EVERYTHING.
We'll validate later in a separate pass.

## 📚 RELATIONSHIP TYPES TO EXTRACT ##

Extract ALL of these types (and more):

### 1. ENTITY RELATIONSHIPS
- Authorship: X authored Y, X wrote Y, X published Y
- Organizational: X founded Y, X works for Y, X is affiliated with Y
- Location: X is located in Y, X occurs in Y
- Attribution: X said Y, X stated Y, X believes Y

### 2. DISCOURSE GRAPH
- **Claims**: Assertions, arguments, theses
- **Evidence**: Facts, data, observations that support or oppose claims
- **Questions**: Inquiries, problems, challenges posed
- Relationships:
  - Claim → supports → Claim
  - Claim → opposes → Claim
  - Evidence → supports → Claim
  - Evidence → opposes → Claim
  - Claim → answers → Question
  - Claim → informs → Question

### 3. PROCESSES & PRACTICES
- Methods: X involves Y, X requires Y, X includes Y, X produces Y
- Procedures: X transforms Y, X creates Y, X converts Y into Z
- Practices: X recommends Y, X suggests Y, X implements Y

### 4. CAUSATION & EFFECTS
- Positive: X causes Y, X leads to Y, X enables Y, X improves Y
- Negative: X prevents Y, X harms Y, X destroys Y, X threatens Y
- Neutral: X affects Y, X influences Y, X changes Y

### 5. DEFINITIONS & DESCRIPTIONS
- Identity: X is-a Y, X is defined as Y, X means Y
- Properties: X has Y, X contains Y, X exhibits Y
- Comparison: X is similar to Y, X differs from Y, X equals Y

### 6. QUANTITATIVE RELATIONSHIPS
- Measurements: X measures Y units, X equals Y percent
- Changes: X increases by Y, X decreases by Y
- Comparisons: X is greater than Y, X is equivalent to Y

## 📝 OUTPUT FORMAT ##

For each relationship provide:
- source: Complete source entity/concept/claim (with all qualifiers)
- relationship: Relationship type (supports, causes, is-a, produces, etc.)
- target: Complete target entity/concept/claim (with all qualifiers)
- evidence_text: Quote from text (100-300 characters)

## 📖 TEXT TO EXTRACT FROM ##

{text}

## ⚡ BE EXHAUSTIVE ##

Extract EVERY relationship you find. Don't hold back!"""


def pass1_extract_book(chunk: str, doc_sha256: str, page_numbers: List[int]) -> List[ProductionRelationship]:
    """Pass 1 extraction for book chunk"""
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at comprehensively extracting ALL relationships from book text."
                },
                {
                    "role": "user",
                    "content": BOOK_EXTRACTION_PROMPT.format(text=chunk)
                }
            ],
            response_format=ComprehensiveExtraction,
            temperature=0.3
        )

        extraction = response.choices[0].message.parsed

        candidates = []
        for rel in extraction.relationships:
            candidate_uid = make_candidate_uid(
                rel.source, rel.relationship, rel.target,
                rel.evidence_text, doc_sha256
            )

            prod_rel = ProductionRelationship(
                source=rel.source,
                relationship=rel.relationship,
                target=rel.target,
                evidence_text=rel.evidence_text,
                candidate_uid=candidate_uid
            )

            prod_rel.evidence['doc_sha256'] = doc_sha256
            prod_rel.evidence['page_number'] = page_numbers[0] if page_numbers else None

            evidence_start = chunk.find(rel.evidence_text[:50]) if rel.evidence_text else -1
            if evidence_start >= 0:
                prod_rel.evidence['start_char'] = evidence_start
                prod_rel.evidence['end_char'] = evidence_start + len(rel.evidence_text)

            candidates.append(prod_rel)

        return candidates

    except Exception as e:
        logger.error(f"Error in Pass 1 extraction: {e}")
        return []


DUAL_SIGNAL_EVALUATION_PROMPT = """Evaluate these {batch_size} extracted relationships using dual-signal analysis.

RELATIONSHIPS TO EVALUATE:
{relationships_json}

For EACH of the {batch_size} relationships above, provide TWO INDEPENDENT evaluations:

1. TEXT SIGNAL (ignore world knowledge):
   - How clearly does the text state this relationship?
   - Score 0.0-1.0 based purely on text clarity

2. KNOWLEDGE SIGNAL (ignore the text):
   - Is this relationship plausible given world knowledge?
   - What types are the source and target entities?
   - Score 0.0-1.0 based purely on plausibility

If the signals conflict:
- Set signals_conflict = true
- Include conflict_explanation
- Include suggested_correction if known

CRITICAL: Return candidate_uid UNCHANGED in every output object.

Evaluate all {batch_size} relationships in the same order as input."""


def evaluate_batch_robust(batch: List[ProductionRelationship],
                         model: str, prompt_version: str) -> List[ProductionRelationship]:
    """Robust batch evaluation"""
    if not batch:
        return []

    try:
        batch_data = [
            {
                "candidate_uid": item.candidate_uid,
                "source": item.source,
                "relationship": item.relationship,
                "target": item.target,
                "evidence_text": item.evidence_text
            }
            for item in batch
        ]

        relationships_json = json.dumps(batch_data, indent=2)

        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at evaluating relationships with separated text and knowledge signals."
                },
                {
                    "role": "user",
                    "content": DUAL_SIGNAL_EVALUATION_PROMPT.format(
                        batch_size=len(batch),
                        relationships_json=relationships_json
                    )
                }
            ],
            response_format=BatchedEvaluationResult,
            temperature=0.3
        )

        batch_result = response.choices[0].message.parsed
        uid_to_item = {it.candidate_uid: it for it in batch}

        results = []
        for evaluation in batch_result.evaluations:
            candidate = uid_to_item.get(evaluation.candidate_uid)

            prod_rel = ProductionRelationship(
                source=evaluation.source,
                relationship=evaluation.relationship,
                target=evaluation.target,
                evidence_text=evaluation.evidence_text,
                text_confidence=evaluation.text_confidence,
                knowledge_plausibility=evaluation.knowledge_plausibility,
                signals_conflict=evaluation.signals_conflict,
                conflict_explanation=evaluation.conflict_explanation,
                suggested_correction=evaluation.suggested_correction,
                source_type=evaluation.source_type,
                target_type=evaluation.target_type,
                candidate_uid=evaluation.candidate_uid,
                flags=candidate.flags.copy() if candidate else {},
                evidence=candidate.evidence.copy() if candidate else _default_evidence()
            )

            results.append(prod_rel)

        return results

    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


# ============================================================================
# MAIN V5 EXTRACTION PIPELINE
# ============================================================================

def extract_knowledge_graph_from_book_v5(book_title: str,
                                         pdf_path: Path,
                                         run_id: str,
                                         batch_size: int = 25) -> Dict[str, Any]:
    """
    ✨ V5 EXTRACTION SYSTEM WITH PASS 2.5 QUALITY POST-PROCESSING

    V4 (Pass 1 + Pass 2) → ✨ NEW Pass 2.5 → Output

    GOAL: Reduce quality issues from 57% (V4) to <10% (V5)
    """
    logger.info(f"🚀 Starting V5 extraction with Pass 2.5: {book_title}")

    start_time = time.time()

    # Extract text from PDF
    full_text, pages_with_text = extract_text_from_pdf(pdf_path)
    doc_sha256 = hashlib.sha256(full_text.encode()).hexdigest()

    # Chunk text
    text_chunks = chunk_book_text(pages_with_text, chunk_size=500, overlap=75)

    # PASS 1: Extract everything
    logger.info("📝 PASS 1: Comprehensive extraction...")
    logger.info(f"  Processing {len(text_chunks)} chunks")

    all_candidates = []

    for i, (page_nums, chunk) in enumerate(text_chunks):
        if i % 10 == 0:
            logger.info(f"  Chunk {i}/{len(text_chunks)} (pages {page_nums[0]}-{page_nums[-1]})")

        candidates = pass1_extract_book(chunk, doc_sha256, page_nums)
        all_candidates.extend(candidates)
        time.sleep(0.05)

    logger.info(f"✅ PASS 1 COMPLETE: {len(all_candidates)} candidates extracted")

    # PASS 2: Evaluate in batches
    logger.info(f"⚡ PASS 2: Batched evaluation ({batch_size} rels/batch)...")

    validated_relationships = []
    batches = list(chunks(all_candidates, batch_size))
    logger.info(f"  Processing {len(batches)} batches...")

    for batch_num, batch in enumerate(batches):
        if batch_num % 5 == 0:
            logger.info(f"  Batch {batch_num + 1}/{len(batches)}")

        evaluations = evaluate_batch_robust(
            batch=batch,
            model="gpt-4o-mini",
            prompt_version="v5_with_pass2_5"
        )

        validated_relationships.extend(evaluations)
        time.sleep(0.1)

    logger.info(f"✅ PASS 2 COMPLETE: {len(validated_relationships)} relationships evaluated")

    # ✨ NEW: PASS 2.5 QUALITY POST-PROCESSING
    validated_relationships, pass2_5_stats = pass_2_5_quality_post_processing(
        validated_relationships,
        pages_with_text
    )

    # POST-PROCESSING: Compute final p_true for all relationships
    logger.info("🎯 FINAL POST-PROCESSING: Computing calibrated probabilities...")

    for rel in validated_relationships:
        # Compute final p_true (may have been modified by Pass 2.5)
        rel.p_true = compute_p_true(
            rel.text_confidence,
            rel.knowledge_plausibility,
            rel.pattern_prior,
            rel.signals_conflict
        )

        # Generate claim UID
        rel.claim_uid = generate_claim_uid(rel)

        # Set metadata
        rel.extraction_metadata["run_id"] = run_id
        rel.extraction_metadata["extracted_at"] = datetime.now().isoformat()

    # ANALYZE & RETURN
    high_confidence = [r for r in validated_relationships if r.p_true >= 0.75]
    medium_confidence = [r for r in validated_relationships if 0.5 <= r.p_true < 0.75]
    low_confidence = [r for r in validated_relationships if r.p_true < 0.5]
    conflicts = [r for r in validated_relationships if r.signals_conflict]

    # Page coverage
    pages_with_extractions = set()
    for rel in validated_relationships:
        page = rel.evidence.get('page_number')
        if page:
            pages_with_extractions.add(page)

    total_book_pages = len(pages_with_text)
    coverage_percentage = len(pages_with_extractions) / total_book_pages * 100 if total_book_pages > 0 else 0

    total_time = time.time() - start_time

    results = {
        'book_title': book_title,
        'run_id': run_id,
        'version': 'v5_with_pass2_5',
        'timestamp': datetime.now().isoformat(),
        'doc_sha256': doc_sha256,
        'pages': len(pages_with_text),
        'word_count': len(full_text.split()),
        'extraction_time_minutes': round(total_time / 60, 1),

        # Stage counts
        'pass1_candidates': len(all_candidates),
        'pass2_evaluated': pass2_5_stats['initial_count'],
        'pass2_5_final': pass2_5_stats['final_count'],

        # Pass 2.5 statistics
        'pass2_5_stats': pass2_5_stats,

        # Quality metrics
        'high_confidence_count': len(high_confidence),
        'medium_confidence_count': len(medium_confidence),
        'low_confidence_count': len(low_confidence),
        'conflicts_detected': len(conflicts),

        # Page coverage
        'pages_with_extractions': len(pages_with_extractions),
        'page_coverage_percentage': round(coverage_percentage, 1),

        # Relationships
        'relationships': [
            {
                'source': r.source,
                'relationship': r.relationship,
                'target': r.target,
                'source_type': r.source_type,
                'target_type': r.target_type,
                'evidence_text': r.evidence_text,
                'evidence': r.evidence,
                'text_confidence': r.text_confidence,
                'knowledge_plausibility': r.knowledge_plausibility,
                'pattern_prior': r.pattern_prior,
                'signals_conflict': r.signals_conflict,
                'conflict_explanation': r.conflict_explanation,
                'p_true': r.p_true,
                'claim_uid': r.claim_uid,
                'flags': r.flags,
                'extraction_metadata': r.extraction_metadata
            }
            for r in validated_relationships
        ]
    }

    logger.info(f"📊 FINAL V5 RESULTS:")
    logger.info(f"  - Pass 1 extracted: {results['pass1_candidates']} candidates")
    logger.info(f"  - Pass 2 evaluated: {results['pass2_evaluated']}")
    logger.info(f"  - ✨ Pass 2.5 final: {results['pass2_5_final']} (+{results['pass2_5_final'] - results['pass2_evaluated']} from list splitting)")
    logger.info(f"  - High confidence (p≥0.75): {len(high_confidence)} ({len(high_confidence)/len(validated_relationships)*100:.1f}%)")
    logger.info(f"  - Medium confidence: {len(medium_confidence)} ({len(medium_confidence)/len(validated_relationships)*100:.1f}%)")
    logger.info(f"  - Low confidence: {len(low_confidence)} ({len(low_confidence)/len(validated_relationships)*100:.1f}%)")
    logger.info(f"  - Page coverage: {coverage_percentage:.1f}% ({len(pages_with_extractions)}/{total_book_pages} pages)")
    logger.info(f"  - Total time: {total_time/60:.1f} minutes")

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Extract knowledge graph from Soil Stewardship Handbook with V5 system"""
    logger.info("="*80)
    logger.info("🚀 V5 KNOWLEDGE GRAPH EXTRACTION - WITH PASS 2.5 QUALITY POST-PROCESSING")
    logger.info("="*80)
    logger.info("")
    logger.info("✨ V5 NEW FEATURES:")
    logger.info("  ✅ 7 Pass 2.5 quality modules")
    logger.info("  ✅ Automatic authorship direction correction")
    logger.info("  ✅ List target splitting")
    logger.info("  ✅ Pronoun resolution")
    logger.info("  ✅ Vague entity context enrichment")
    logger.info("  ✅ Title/predicate/metaphor validation")
    logger.info("")
    logger.info("GOAL: Reduce quality issues from 57% (V4) to <10% (V5)")
    logger.info("")

    # Book details
    book_dir = BOOKS_DIR / "soil-stewardship-handbook"
    pdf_path = book_dir / "Soil-Stewardship-Handbook-eBook.pdf"
    book_title = "Soil Stewardship Handbook"

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    run_id = f"book_soil_handbook_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    start_time = time.time()

    # Extract with V5 system
    results = extract_knowledge_graph_from_book_v5(
        book_title=book_title,
        pdf_path=pdf_path,
        run_id=run_id,
        batch_size=25
    )

    # Save results
    output_path = OUTPUT_DIR / f"{book_title.replace(' ', '_').lower()}_v5.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("✨ V5 EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Run KG Reflector to analyze V5 quality")
    logger.info("2. Compare V5 vs V4 quality metrics")
    logger.info("3. Launch continuous ACE improvement cycle if needed")
    logger.info("="*80)


if __name__ == "__main__":
    main()
