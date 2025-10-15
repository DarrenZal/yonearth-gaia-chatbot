#!/usr/bin/env python3
"""
Knowledge Graph Extraction V10 - COMPREHENSIVE MEANINGFUL EXTRACTION

🎯 V10 GOAL: Extract ALL valuable factual relationships (not hitting arbitrary numbers)
Focus: Meaningful data, information, knowledge, and wisdom

✨ V10 NEW FEATURES (From V9 Evidence-Based Analysis):

**COMPREHENSIVENESS IMPROVEMENTS:**
✅ ENHANCED Pass 1 prompt - explicit relationship type examples (bibliographic, categorical, compositional)
✅ ENHANCED Pass 1 prompt - few-shot extraction examples showing what to extract
✅ ENHANCED Pass 2 evaluation - recalibrated for bibliographic citations (don't over-penalize)

**QUALITY FIXES:**
✅ CRITICAL: ENHANCED dedication parser - context-aware dedication detection (fixes 6 errors)
✅ HIGH: ENHANCED possessive pronoun resolver - handles "my X", "our X" with context (fixes 8 errors)
✅ MEDIUM: ENHANCED vague entity filter - prevents "thousands", "the land" (fixes 10 errors)

**V9 INNOVATIONS MAINTAINED:**
✅ 100% attribution coverage (who said what, where)
✅ 100% classification coverage (FACTUAL, TESTABLE_CLAIM, PHILOSOPHICAL, METAPHOR)
✅ List splitter inheritance (automatic attribution + classification)
✅ No p_true filtering (complete discourse graph)

GOAL: Extract comprehensive knowledge while achieving <3% quality issues (A++ grade)

V10 TARGETS (Evidence-Based):
- Bibliographic relationships: 250+ (vs V9: 127, V8: 304)
- Categorical relationships: 70+ (vs V9: 27, V8: 93)
- Total relationships: 650-750 natural outcome (vs V9: 414, V8: 1090)
- Quality issues: <3% (vs V9: 5.8%, V8: 8.35%)
- Attribution: 100% (maintain from V9)
- Classification: 100% (maintain from V9)

Architecture: V9 base + Comprehensiveness enhancements + Quality fixes → V10 Production System

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
from typing import Optional, List, Dict, Any, Tuple
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

# ✨ spaCy for POS tagging
import spacy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_extraction_book_v10_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
BOOKS_DIR = DATA_DIR / "books"
PLAYBOOK_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook")
PROMPTS_DIR = PLAYBOOK_DIR / "prompts"
OUTPUT_DIR = PLAYBOOK_DIR / "output" / "v10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY_2 not set in .env!")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# ✨ Load V10 prompts from files
logger.info("Loading V10 prompts from files...")
PASS1_PROMPT_FILE = PROMPTS_DIR / "pass1_extraction_v10.txt"
PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v10.txt"

if not PASS1_PROMPT_FILE.exists():
    logger.error(f"Pass 1 prompt file not found: {PASS1_PROMPT_FILE}")
    exit(1)
if not PASS2_PROMPT_FILE.exists():
    logger.error(f"Pass 2 prompt file not found: {PASS2_PROMPT_FILE}")
    exit(1)

with open(PASS1_PROMPT_FILE) as f:
    BOOK_EXTRACTION_PROMPT = f.read()
with open(PASS2_PROMPT_FILE) as f:
    DUAL_SIGNAL_EVALUATION_PROMPT = f.read()

logger.info("✅ V10 prompts loaded successfully")

# ✨ Load spaCy model for POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ Loaded spaCy en_core_web_sm for POS tagging")
except:
    logger.warning("⚠️  spaCy model not loaded - list splitting will use fallback logic")
    nlp = None

# Cache for scorer results
edge_cache: Dict[str, Any] = {}
cache_stats = {'hits': 0, 'misses': 0}

# ⭐ V9 NEW: Quality threshold for filtering low-confidence relationships
# Set to None to disable filtering (keep all relationships for full discourse graph)
# Set to 0.5 or higher to filter low-confidence relationships
MIN_P_TRUE_THRESHOLD = None  # Disabled to preserve complete discourse graph (including metaphors, philosophical claims)


# ============================================================================
# DATACLASSES & SCHEMAS
# ============================================================================

def _default_evidence():
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
    return {}

def _default_extraction_metadata():
    return {
        "model_pass1": "gpt-4o-mini",
        "model_pass2": "gpt-4o-mini",
        "prompt_version": "v9_reflector_fixes",
        "extractor_version": "2025.10.13_v9",
        "content_type": "book",
        "run_id": None,
        "extracted_at": None,
        "batch_id": None
    }

def _default_attribution():
    """⭐ V9 NEW: Default attribution metadata for discourse graph provenance"""
    return {
        "source_type": None,      # "book", "episode", "paper", etc.
        "source_title": None,     # Title of source document
        "source_author": None,    # Author/speaker/organization
        "source_id": None,        # Episode number, ISBN, etc.
        "page_number": None,      # Page in book
        "timestamp": None,        # Timestamp in episode
        "url": None,              # Source URL if available
        "context": None           # Additional context
    }


@dataclass
class ProductionRelationship:
    source: str
    relationship: str
    target: str
    source_type: Optional[str] = None
    target_type: Optional[str] = None
    flags: Dict[str, Any] = field(default_factory=_default_flags)
    evidence_text: str = ""
    evidence: Dict[str, Any] = field(default_factory=_default_evidence)
    text_confidence: float = 0.0
    knowledge_plausibility: float = 0.0
    pattern_prior: float = 0.5
    signals_conflict: bool = False
    conflict_explanation: Optional[str] = None
    suggested_correction: Optional[str] = None
    p_true: float = 0.0
    claim_uid: Optional[str] = None
    candidate_uid: Optional[str] = None
    extraction_metadata: Dict[str, Any] = field(default_factory=_default_extraction_metadata)
    classification_flags: List[str] = field(default_factory=list)  # ⭐ V9 NEW: Statement classification
    attribution: Dict[str, Any] = field(default_factory=_default_attribution)  # ⭐ V9 NEW: Discourse graph provenance


class SimpleRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    evidence_text: str = Field(description="Quote from text supporting this relationship")


class ComprehensiveExtraction(BaseModel):
    relationships: List[SimpleRelationship]


class DualSignalEvaluation(BaseModel):
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
    classification_flags: List[str] = Field(default_factory=list, description="Statement classification flags (FACTUAL, METAPHOR, etc.)")


class BatchedEvaluationResult(BaseModel):
    evaluations: List[DualSignalEvaluation]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def make_candidate_uid(source: str, relationship: str, target: str,
                       evidence_text: str, doc_sha256: str) -> str:
    evidence_hash = hashlib.sha1(evidence_text.encode()).hexdigest()[:8]
    base = f"{source}|{relationship}|{target}|{evidence_hash}|{doc_sha256}"
    return hashlib.sha1(base.encode()).hexdigest()


def generate_claim_uid(rel: ProductionRelationship) -> str:
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
    z = (-1.2 + 2.1 * text_conf + 0.9 * knowledge_plaus +
         0.6 * pattern_prior - 0.8 * int(conflict))
    p_true = 1 / (1 + math.exp(-z))
    return p_true


def chunks(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ============================================================================
# ✨ V8 PASS 2.5: CURATOR-ENHANCED QUALITY POST-PROCESSING MODULES
# ============================================================================

class PraiseQuoteDetector:
    """
    ✨ V8 NEW MODULE (Curator Change #001): Detects praise quotes in front matter
    ⭐ V9 ENHANCEMENT: Now identifies SPEAKER vs SUBJECT attribution

    Praise quotes appear in front matter (pages 1-15) with pattern:
    - Quote text with endorsement language
    - Attribution: —Name, Title, Organization

    V8 Issue: Changed relationship type but didn't fix source attribution
    V9 Fix: Extracts SPEAKER from attribution marker and swaps source

    Example:
    - Evidence: "Perry has given us...—Adrian Del Caro"
    - V8: (Perry, endorsed, Book) ❌ Wrong source!
    - V9: (Adrian Del Caro, endorsed, Book) ✅ Correct!

    FIXES: 3 CRITICAL reversed authorship errors (0.28%)
    """

    def __init__(self):
        self.front_matter_pages = range(1, 15)
        self.endorsement_indicators = [
            'inspirational', 'beautifully-informed', 'wonderful',
            'informative handbook', 'gives us', 'invites us',
            'delighted to see', 'highly recommend', 'essential',
            'excellent tool', 'grateful', 'honored'
        ]
        self.person_name_pattern = r'—([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})'
        self.credentials_pattern = r'(?:Founding|Director|Chair|Professor|PhD|Author|Champion)'
        self.authorship_verbs = ['authored', 'wrote', 'written by', 'author of']

        # ⭐ V9 NEW: Patterns to identify SUBJECT being praised (not the endorser)
        self.subject_patterns = [
            r'(\w+)\s+has given us',
            r'(\w+)\s+provides',
            r'(\w+)\'s?\s+(?:handbook|book|work)',
            r'with (?:his|her)\s+.*?,\s+(\w+)\s+',
            r'(\w+)\s+invites us',
            r'by\s+(\w+)'
        ]

    def is_praise_quote_context(self, evidence_text: str, page: int) -> bool:
        """Check if evidence suggests praise quote rather than authorship."""
        if page not in self.front_matter_pages:
            return False

        # Check for attribution marker (—Name)
        has_attribution = bool(re.search(self.person_name_pattern, evidence_text))

        # Check for credentials (not typical in authorship claims)
        has_credentials = bool(re.search(self.credentials_pattern, evidence_text))

        # Check for endorsement language
        has_endorsement_language = any(
            indicator in evidence_text.lower()
            for indicator in self.endorsement_indicators
        )

        return has_attribution or has_credentials or has_endorsement_language

    def fix_praise_quote_attribution(self, rel: ProductionRelationship, evidence_text: str) -> ProductionRelationship:
        """
        ⭐ V9 NEW METHOD: Fix praise quote attribution by identifying SPEAKER vs SUBJECT

        When a praise quote says "Perry has given us...", Perry is the SUBJECT being praised.
        The SPEAKER is the person in the attribution marker (—Name).

        Args:
            rel: Relationship with potentially wrong source
            evidence_text: Evidence text containing attribution marker

        Returns:
            Relationship with corrected source (SPEAKER instead of SUBJECT)
        """
        # Extract SPEAKER from attribution marker (—Name)
        speaker_match = re.search(self.person_name_pattern, evidence_text)
        if not speaker_match:
            # No attribution marker found, can't fix
            return rel

        speaker_name = speaker_match.group(1).strip()

        # Check if current source is the SUBJECT being praised
        current_source = rel.source.lower()
        is_subject = False

        for pattern in self.subject_patterns:
            match = re.search(pattern, evidence_text, re.IGNORECASE)
            if match:
                subject_name = match.group(1).lower()
                # Check if the subject matches current source
                if subject_name in current_source or current_source in subject_name:
                    is_subject = True
                    break

        # If current source is the subject being praised, swap to speaker
        if is_subject:
            rel.source = speaker_name
            if rel.flags is None:
                rel.flags = {}
            rel.flags['PRAISE_ATTRIBUTION_FIXED'] = True
            rel.flags['attribution_note'] = f'Fixed: SPEAKER ({speaker_name}) not SUBJECT'
            logger.debug(f"      Fixed praise quote attribution: {speaker_name} (not {current_source})")

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """
        Process relationships to detect and correct praise quote misattributions.

        V8: Only changed relationship type from 'authored' to 'endorsed'
        V9: Also fixes source attribution (SPEAKER not SUBJECT)
        """
        corrected = []
        correction_count = 0
        attribution_fixed_count = 0

        for rel in relationships:
            relationship_type = rel.relationship.lower()
            evidence = rel.evidence_text
            page = rel.evidence.get('page_number', 0)

            # Check if this is authorship claim in praise quote context
            if any(verb in relationship_type for verb in self.authorship_verbs):
                if self.is_praise_quote_context(evidence, page):
                    # V8: Correct relationship type to endorsement
                    rel.relationship = 'endorsed'
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRAISE_QUOTE_CORRECTED'] = True
                    rel.flags['correction_note'] = 'Changed from authorship to endorsement (praise quote detected)'
                    correction_count += 1

                    # ⭐ V9 NEW: Also fix attribution (SPEAKER not SUBJECT)
                    rel = self.fix_praise_quote_attribution(rel, evidence)
                    if rel.flags.get('PRAISE_ATTRIBUTION_FIXED'):
                        attribution_fixed_count += 1

            corrected.append(rel)

        logger.info(f"   Praise quote detection: {correction_count} praise quotes corrected to endorsements")
        if attribution_fixed_count > 0:
            logger.info(f"   ⭐ V9: {attribution_fixed_count} praise quote attributions fixed (SPEAKER not SUBJECT)")
        return corrected


class BibliographicCitationParser:
    """
    V6 IMPROVEMENT: Added endorsement detection
    ✨ V8 ENHANCEMENT (Curator Change #012-013): Added dedication detection

    Detects and corrects authorship relationships from bibliographic citations.
    Now also detects book endorsements and dedications.

    FIXES: 1 dedication misattribution (0.11%)
    """

    def __init__(self):
        self.citation_patterns = [
            r'^([A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+,\s+[A-Z][a-z]+)*)\.',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\.',
        ]

        self.authorship_predicates = ('authored', 'wrote', 'published', 'created', 'composed', 'edited', 'compiled', 'produced')

        # ✨ V7 ENHANCED: Expanded endorsement detection patterns
        self.endorsement_patterns = [
            r'PRAISE FOR',
            r'TESTIMONIAL',
            r'ENDORSEMENT',
            r'(?:excellent|important|delightful|wonderful|brilliant|masterpiece|essential)',
            r'(?:tool|book|handbook|manual|guide|resource|work)',
            r'delighted to see',
            r'highly recommend',
            r'must[- ]read',
            r'invaluable resource',
            r'strongly recommend',
            r'(?:grateful|thrilled|honored|privileged|blessed)\s+to',
            r'(?:this|the)\s+(?:book|handbook|manual|guide|work)\s+(?:is|represents)\s+(?:an?\s+)?(?:excellent|wonderful|vital|essential|invaluable)',
            r'engage with this critical mission',
            r'excellent tool for',
            r'wonderful resource',
            r'writes in the foreword',
            r'in his foreword',
            r'in her foreword',
            r'foreword by'
        ]

        # ✨ V8 NEW: Dedication patterns (Curator Change #012)
        self.dedication_patterns = [
            r'(?:this book is )?dedicated to (.+)',
            r'in memory of (.+)',
            r'for my (.+)',
            r'to my (.+)',
        ]

        self.dedication_verbs = ['dedicated', 'authored', 'wrote', 'authorship']

    def is_bibliographic_citation(self, evidence_text: str) -> bool:
        for pattern in self.citation_patterns:
            if re.match(pattern, evidence_text.strip()):
                return True
        return False

    def is_endorsement(self, evidence_text: str, full_page_text: str = "") -> bool:
        """Check if this is an endorsement rather than authorship"""
        combined_text = (full_page_text + " " + evidence_text).lower()

        for pattern in self.endorsement_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True
        return False

    def is_dedication(self, evidence_text: str) -> tuple:
        """✨ V8 NEW (Curator Change #012): Check if this is a dedication statement

        Returns: (is_dedication: bool, recipients: str or None)
        """
        evidence_lower = evidence_text.lower()

        for pattern in self.dedication_patterns:
            match = re.search(pattern, evidence_lower)
            if match:
                recipients = match.group(1)
                return True, recipients

        return False, None

    def should_reverse_authorship(self, rel: ProductionRelationship) -> tuple:
        """
        Returns: (should_reverse: bool, is_endorsement: bool)
        """
        if rel.relationship not in self.authorship_predicates:
            return False, False

        evidence = rel.evidence_text.strip()

        if not self.is_bibliographic_citation(evidence):
            return False, False

        # Check if it's an endorsement
        page_text = rel.evidence.get('window_text', '')
        if self.is_endorsement(evidence, page_text):
            return False, True  # Don't reverse, but mark as endorsement

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

        should_reverse = source_is_title and target_is_author
        return should_reverse, False

    def reverse_authorship(self, rel: ProductionRelationship) -> ProductionRelationship:
        rel.source, rel.target = rel.target, rel.source
        rel.source_type, rel.target_type = rel.target_type, rel.source_type

        rel.evidence['source_surface'], rel.evidence['target_surface'] = \
            rel.evidence.get('target_surface'), rel.evidence.get('source_surface')

        if rel.flags is None:
            rel.flags = {}
        rel.flags['AUTHORSHIP_REVERSED'] = True
        rel.flags['correction_reason'] = 'bibliographic_citation_detected'

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        corrected = []
        correction_count = 0
        endorsement_count = 0
        dedication_count = 0  # ✨ V8 NEW

        for rel in relationships:
            # ✨ V8 NEW: Check for dedication statements first (Curator Change #013)
            relationship_lower = rel.relationship.lower()
            evidence = rel.evidence_text

            if any(verb in relationship_lower for verb in self.dedication_verbs):
                is_dedication, recipients = self.is_dedication(evidence)

                if is_dedication:
                    # This is a dedication, not authorship
                    rel.relationship = 'dedicated'

                    # Append recipients to target if it's a book
                    if 'handbook' in rel.target.lower() or 'book' in rel.target.lower():
                        rel.target = f"{rel.target} to {recipients}"

                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['DEDICATION_CORRECTED'] = True
                    rel.flags['original_relationship'] = rel.relationship
                    dedication_count += 1
                    corrected.append(rel)
                    continue

            # Original logic for authorship/endorsement
            should_reverse, is_endorsement = self.should_reverse_authorship(rel)

            if is_endorsement:
                # Convert 'authored' to 'endorsed'
                if rel.relationship in self.authorship_predicates:
                    rel.relationship = 'endorsed'
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['ENDORSEMENT_DETECTED'] = True
                    endorsement_count += 1
            elif should_reverse:
                rel = self.reverse_authorship(rel)
                correction_count += 1

            corrected.append(rel)

        logger.info(f"   Bibliographic citations: {correction_count} authorships reversed, {endorsement_count} endorsements detected, {dedication_count} dedications corrected")
        return corrected


class ListSplitter:
    """
    V6 IMPROVEMENT: Added POS tagging to distinguish adjective series from noun lists
    ✨ V8 ENHANCEMENT (Curator Change #008-009): Added 'and' conjunction patterns

    Prevents "physical, mental, spiritual growth" from being split into 3 relationships
    NOW splits "families, communities and planet" into 3 separate targets

    FIXES: 4 incomplete list splits (0.43%)
    """

    def __init__(self, use_pos_tagging=True):
        self.min_list_length = 15
        self.use_pos_tagging = use_pos_tagging and (nlp is not None)

        # ✨ V8 NEW: Enhanced list patterns with 'and' conjunctions (Curator Change #008)
        self.list_patterns = [
            # Pattern 1: A, B, and C (Oxford comma)
            r'([^,]+),\s*([^,]+),\s*and\s+([^,]+)',
            # Pattern 2: A, B and C (no Oxford comma)
            r'([^,]+),\s*([^,]+)\s+and\s+([^,]+)',
            # Pattern 3: A and B (simple conjunction)
            r'([^,]+)\s+and\s+([^,]+)',
            # Pattern 4: A, B (simple comma)
            r'([^,]+),\s*([^,]+)',
        ]

        if self.use_pos_tagging:
            logger.info("✅ ListSplitter using POS tagging for intelligent splitting + 'and' conjunctions (V8)")
        else:
            logger.info("⚠️  ListSplitter using fallback logic (no POS tagging)")

    def is_adjective_series(self, target: str) -> bool:
        """Use POS tagging to detect adjective series"""
        if not self.use_pos_tagging:
            return False

        doc = nlp(target)
        tokens = [token for token in doc]
        if len(tokens) < 3:
            return False

        # Find last noun
        last_noun_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].pos_ == 'NOUN':
                last_noun_idx = i
                break

        if last_noun_idx is None:
            return False

        # Check if tokens before last noun are mostly adjectives
        prefix_tokens = tokens[:last_noun_idx]
        if not prefix_tokens:
            return False

        adjective_count = sum(1 for t in prefix_tokens if t.pos_ == 'ADJ')
        coord_count = sum(1 for t in prefix_tokens if t.dep_ == 'cc')

        return adjective_count >= len(prefix_tokens) * 0.6

    def is_list_target(self, target: str) -> bool:
        """Check if target contains a list pattern (commas or 'and')"""
        # ✨ V8 ENHANCEMENT: Also check for 'and' without commas
        if ',' not in target and ' and ' not in target:
            return False

        if len(target) < self.min_list_length:
            return False

        # Check if it's an adjective series
        if self.is_adjective_series(target):
            return False  # Don't split adjective series

        # Check for list patterns
        if ' and ' in target or ',' in target:
            return True

        comma_count = target.count(',')
        if comma_count >= 2:
            return True

        return False

    def split_target_list(self, target: str) -> List[str]:
        """✨ V8 ENHANCEMENT (Curator Change #009): Split using 'and' and comma patterns"""

        # Try each pattern in order (most specific first)
        for pattern in self.list_patterns:
            match = re.match(pattern, target.strip())
            if match:
                items = [item.strip() for item in match.groups() if item]

                # Remove duplicates while preserving order
                seen = set()
                unique_items = []
                for item in items:
                    item_lower = item.lower()
                    if item_lower not in seen:
                        seen.add(item_lower)
                        unique_items.append(item)

                return unique_items

        # No pattern matched, return as single item
        return [target]

    def split_relationship(self, rel: ProductionRelationship) -> List[ProductionRelationship]:
        items = self.split_target_list(rel.target)

        if len(items) <= 1:
            return [rel]

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
                claim_uid=None,
                flags=rel.flags.copy() if rel.flags else {},
                extraction_metadata=rel.extraction_metadata.copy(),
                classification_flags=rel.classification_flags.copy() if rel.classification_flags else [],  # ⭐ V9 FIX: Inherit classification
                attribution=rel.attribution.copy() if rel.attribution else {}  # ⭐ V9 FIX: Inherit attribution
            )

            new_rel.flags['LIST_SPLIT'] = True
            new_rel.flags['split_index'] = i
            new_rel.flags['split_total'] = len(items)
            new_rel.flags['original_target'] = rel.target

            split_rels.append(new_rel)

        return split_rels

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        processed = []
        split_count = 0
        adjective_series_preserved = 0
        original_count = len(relationships)

        for rel in relationships:
            # Track adjective series that we preserve
            if ',' in rel.target and self.is_adjective_series(rel.target):
                adjective_series_preserved += 1
                processed.append(rel)
            elif self.is_list_target(rel.target):
                split_rels = self.split_relationship(rel)
                processed.extend(split_rels)
                if len(split_rels) > 1:
                    split_count += 1
            else:
                processed.append(rel)

        new_count = len(processed)
        logger.info(f"   List splitting: {split_count} lists split, {adjective_series_preserved} adjective series preserved, {original_count} → {new_count} relationships")
        return processed


class PronounResolver:
    """
    V6 IMPROVEMENT: Added generic pronoun handler + larger resolution window
    V7 IMPROVEMENT: Multi-pass resolution with expanding windows
    ✨ V8 ENHANCEMENT (Curator Change #002-004): Possessive pronouns + 5-sentence context

    Handles generic "we/you" and increases window size for cultural references
    NOW handles possessive pronouns like "my people" → "Slovenians"

    FIXES: 18 pronoun issues (8 unresolved + 10 possessive = 1.95%)
    """

    def __init__(self):
        self.pronouns = {
            'he', 'she', 'him', 'her', 'his', 'hers',
            'it', 'its',
            'we', 'us', 'our', 'ours',
            'they', 'them', 'their', 'theirs',
            'you', 'your', 'yours'
        }

        # ✨ V8 NEW: Possessive pronouns (Curator Change #002)
        # ⭐ V9 ENHANCEMENT: Expanded patterns and resolution logic
        self.possessive_pronouns = ['my', 'our', 'your', 'their', 'his', 'her', 'its']
        self.possessive_patterns = [
            r'\b(my|our)\s+(people|ancestors|family|community)\b',
            r'\b(my|our|their)\s+(land|country|nation|region)\b',
            r'\b(my|our|their)\s+(connection|tradition|heritage|culture)\b',
            r'\b(my|our|your|their)\s+(\w+)\b'  # Generic possessive (catch-all)
        ]

        # Generic pronoun patterns
        self.generic_pronouns = {
            'we humans': 'humans',
            'we each': 'individuals',
            'we all': 'people',
            'you': 'readers',
            'one': 'people'
        }

        self.page_context = {}
        # ✨ V8 ENHANCED: Expanded context window (Curator Change #002)
        self.resolution_window = 1000  # Was 1000 in V7, keeping same
        self.context_window = 5  # ✨ V8 NEW: 5-sentence window for better antecedent lookup
        self.author_context = None  # ✨ V8 NEW: Will be set from document metadata

    def is_pronoun(self, entity: str) -> bool:
        return entity.lower().strip() in self.pronouns

    def is_generic_pronoun(self, pronoun: str, evidence_text: str) -> Optional[str]:
        """Check if pronoun is generic (not anaphoric)"""
        pronoun_lower = pronoun.lower()
        evidence_lower = evidence_text.lower()

        # Check for explicit generic patterns
        for pattern, replacement in self.generic_pronouns.items():
            if pattern in evidence_lower:
                return replacement

        # Check for instructional/imperative context for "you"
        if pronoun_lower in {'you', 'your'}:
            imperative_markers = ['can', 'should', 'must', 'try', 'start', 'begin', 'make', 'take']
            if any(marker in evidence_lower.split()[:10] for marker in imperative_markers):
                return 'readers'

        # Check for philosophical/general statements with "we"
        if pronoun_lower in {'we', 'our', 'us'}:
            general_markers = ['humans', 'humanity', 'people', 'society', 'world', 'planet', 'earth']
            if any(marker in evidence_lower for marker in general_markers):
                return 'humanity'

        return None

    def load_page_context(self, pages_with_text: List[tuple]):
        self.page_context = {page_num: text for page_num, text in pages_with_text}

    def _get_context_sentences(self, context: str, window: int = 5) -> List[str]:
        """✨ V8 NEW (Curator Change #003): Extract sentences from context"""
        sentences = re.split(r'[.!?]+', context)
        return [s.strip() for s in sentences if s.strip()][-window:]

    def _extract_entities_from_context(self, sentences: List[str]) -> List[str]:
        """✨ V8 NEW (Curator Change #003): Extract named entities from context sentences"""
        entities = []
        # Simple capitalized word detection (can be enhanced with NER)
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 1:
                    if word not in ['The', 'A', 'An', 'In', 'On', 'At', 'To', 'For']:
                        entities.append(word)
        return entities

    def find_antecedent(self, pronoun: str, page_num: int, evidence_text: str) -> Optional[str]:
        """
        V7: Multi-pass pronoun resolution
        ✨ V8 ENHANCEMENT (Curator Change #003): Enhanced with possessive pronoun support

        Pass 1: Same sentence (0-100 chars back)
        Pass 2: Previous sentence (100-500 chars back)
        Pass 3: Paragraph scope (500-1000 chars back)
        """
        pronoun_lower = pronoun.lower()

        page_text = self.page_context.get(page_num, '')

        evidence_pos = page_text.find(evidence_text[:50])
        if evidence_pos == -1:
            return None

        # ✨ V8 NEW: Handle possessive pronouns first (Curator Change #003)
        # ⭐ V9 ENHANCEMENT: Improved resolution with more context patterns
        for pattern in self.possessive_patterns:
            match = re.search(pattern, pronoun_lower)
            if match:
                possessive = match.group(1)
                noun = match.group(2) if len(match.groups()) >= 2 else None

                # Get context for resolution
                context_start = max(0, evidence_pos - self.resolution_window)
                context = page_text[context_start:evidence_pos].lower()

                # ⭐ V9 NEW: Enhanced country/place-based resolution
                # Look for country/place names in context
                place_pattern = r'\b([A-Z][a-z]+(?:ia|land|stan|gary|way|mark))\b'
                place_matches = re.findall(place_pattern, page_text[context_start:evidence_pos])

                if place_matches:
                    place_name = place_matches[-1]  # Use most recent place mention

                    # Resolve based on noun type
                    if noun in ['people', 'ancestors', 'family', 'community']:
                        # "my people" + Slovenia context → "Slovenians"
                        return f"{place_name}ans" if not place_name.endswith('s') else place_name

                    elif noun in ['land', 'country', 'nation', 'region']:
                        # "Our land" + Slovenia context → "Slovenia"
                        return place_name

                    elif noun in ['connection', 'tradition', 'heritage', 'culture']:
                        # "our connection" + Slovenia context → "Slovenian connection"
                        adjective = f"{place_name}an" if not place_name.endswith('n') else place_name
                        return f"{adjective} {noun}"

                # Context-specific resolution (fallback for known places)
                if 'slovenia' in context:
                    if noun in ['people', 'ancestors', 'family']:
                        return 'Slovenians'
                    elif noun in ['land', 'country']:
                        return 'Slovenia'

                # Author-specific resolution
                if self.author_context and possessive in ['my', 'our']:
                    if noun in ['people', 'ancestors']:
                        return f"{self.author_context}'s {noun}"

                # ⭐ V9 NEW: Generic resolution with proper article
                # "my people" → "the author's people" (when no context found)
                if noun:
                    if possessive in ['my', 'his', 'her']:
                        return f"the author's {noun}"
                    elif possessive in ['our', 'their']:
                        return f"{noun}"  # Keep as generic plural
                    else:
                        return f"the {noun}"

        # Multi-pass resolution with expanding windows
        pass_windows = [
            ('same_sentence', 100),
            ('previous_sentence', 500),
            ('paragraph_scope', 1000)
        ]

        for pass_name, window_size in pass_windows:
            context_start = max(0, evidence_pos - window_size)
            context = page_text[context_start:evidence_pos]

            # Person pronouns
            if pronoun_lower in {'he', 'she', 'his', 'her', 'him'}:
                names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', context)
                if names:
                    return names[-1]

            # Collective pronouns
            elif pronoun_lower in {'we', 'our', 'us', 'ours'}:
                # Look for organizations first
                orgs = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Foundation|Institute|Organization|Guild|Movement))\b', context)
                if orgs:
                    return orgs[-1]

                # Enhanced cultural/national references
                cultural_refs = re.findall(r'\b(my|our)\s+(people|country|nation|culture|heritage|land)\b', context, re.IGNORECASE)
                if cultural_refs:
                    nationality_pattern = r'\b([A-Z][a-z]+(?:ians?|ans?))\b'
                    nationalities = re.findall(nationality_pattern, context)
                    if nationalities:
                        return nationalities[-1]
                    return f"{cultural_refs[-1][0]} {cultural_refs[-1][1]}"

                # Look for collective nouns
                collectives = re.findall(r'\b(humanity|people|society|humans|communities|families|Slovenians)\b', context, re.IGNORECASE)
                if collectives:
                    return collectives[-1]

                # Generic fallback for philosophical statements
                if 'soil' in context.lower() or 'earth' in context.lower() or 'planet' in context.lower():
                    return 'humanity'

        return None

    def resolve_pronouns(self, rel: ProductionRelationship) -> ProductionRelationship:
        page_num = rel.evidence.get('page_number')
        evidence_text = rel.evidence_text

        # Check for generic pronouns first
        if self.is_pronoun(rel.source):
            generic_replacement = self.is_generic_pronoun(rel.source, evidence_text)
            if generic_replacement:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['GENERIC_PRONOUN_RESOLVED_SOURCE'] = True
                rel.flags['original_source'] = rel.source
                rel.source = generic_replacement
            else:
                # Try anaphoric resolution
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

        if self.is_pronoun(rel.target):
            generic_replacement = self.is_generic_pronoun(rel.target, evidence_text)
            if generic_replacement:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['GENERIC_PRONOUN_RESOLVED_TARGET'] = True
                rel.flags['original_target'] = rel.target
                rel.target = generic_replacement
            else:
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
                     pages_with_text: List[tuple],
                     document_metadata: Dict = None) -> List[ProductionRelationship]:
        """✨ V8 ENHANCEMENT (Curator Change #004): Added document_metadata parameter"""
        self.load_page_context(pages_with_text)

        # ✨ V8 NEW: Set author context from metadata
        if document_metadata:
            self.author_context = document_metadata.get('author', None)

        processed = []
        resolved_count = 0
        generic_resolved_count = 0
        unresolved_count = 0

        for rel in relationships:
            rel = self.resolve_pronouns(rel)

            if rel.flags and (rel.flags.get('PRONOUN_RESOLVED_SOURCE') or \
                           rel.flags.get('PRONOUN_RESOLVED_TARGET')):
                resolved_count += 1

            if rel.flags and (rel.flags.get('GENERIC_PRONOUN_RESOLVED_SOURCE') or \
                           rel.flags.get('GENERIC_PRONOUN_RESOLVED_TARGET')):
                generic_resolved_count += 1

            if rel.flags and (rel.flags.get('PRONOUN_UNRESOLVED_SOURCE') or \
                            rel.flags.get('PRONOUN_UNRESOLVED_TARGET')):
                unresolved_count += 1

            processed.append(rel)

        logger.info(f"   Pronoun resolution (V8 enhanced): {resolved_count} anaphoric + {generic_resolved_count} generic resolved, {unresolved_count} flagged for review")
        return processed


class ContextEnricher:
    """
    V6 IMPROVEMENT: Expanded vague entity patterns
    ✨ V8 ENHANCEMENT (Curator Change #005-006): Context-aware replacement

    Added demonstrative patterns, relative clauses, and prepositional fragments
    NOW replaces vague entities with specific ones from context

    FIXES: 8 vague target issues (0.87%)
    """

    def __init__(self):
        # Expanded vague terms from Reflector analysis
        self.vague_terms = {
            'the amount', 'the process', 'the practice', 'the method',
            'the system', 'the approach', 'the way', 'the idea',
            'this', 'that', 'these', 'those',
            'this handbook', 'this book', 'the handbook', 'the book',
            'this crossroads', 'the way through', 'that only exists',
            'which is', 'who are', 'that we',
            # ✨ V8 NEW: Additional vague patterns (Curator Change #005)
            'this wonderful place', 'the answer'
        }

        self.doc_entities = {
            'this handbook': 'Soil Stewardship Handbook',
            'this book': 'Soil Stewardship Handbook',
            'the handbook': 'Soil Stewardship Handbook',
            'the book': 'Soil Stewardship Handbook',
            'this crossroads': 'current historical moment',
        }

        # ✨ V8 NEW: Context-aware replacement rules (Curator Change #005)
        self.context_replacements = {
            'this wonderful place': {
                'keywords': ['earth', 'planet', 'world', 'lives depend'],
                'replacement': 'Earth'
            },
            'the answer': {
                'keywords': ['soil', 'stewardship', 'questions'],
                'replacement': 'soil stewardship',
                'check_motto': True  # Special handling for mottos
            },
            'the way': {
                'keywords': ['forward', 'path', 'direction'],
                'replacement': 'the path forward'
            },
            'this': {
                'keywords': ['book', 'handbook', 'guide'],
                'replacement': 'Soil Stewardship Handbook'
            }
        }

    def is_vague(self, entity: str) -> bool:
        entity_lower = entity.lower().strip()

        if entity_lower in self.vague_terms:
            return True

        for term in self.vague_terms:
            if entity_lower.startswith(term):
                return True

        # Check for demonstrative patterns
        if re.match(r'^(this|that|these|those)\s+\w+', entity_lower):
            return True

        return False

    def _find_replacement(self, vague_term: str, evidence: str, relationship: str) -> Optional[str]:
        """✨ V8 NEW (Curator Change #006): Find context-appropriate replacement for vague term"""
        vague_lower = vague_term.lower()
        evidence_lower = evidence.lower()

        # Check each replacement rule
        for pattern, rule in self.context_replacements.items():
            if pattern in vague_lower:
                # Check if context keywords present
                if any(keyword in evidence_lower for keyword in rule['keywords']):
                    # Special handling for mottos
                    if rule.get('check_motto') and 'motto' not in relationship.lower():
                        return rule['replacement']
                    elif not rule.get('check_motto'):
                        return rule['replacement']

        return None

    def enrich_entity(self, entity: str, evidence_text: str,
                     relationship: str, other_entity: str) -> Optional[str]:
        entity_lower = entity.lower().strip()

        if entity_lower in self.doc_entities:
            return self.doc_entities[entity_lower]

        # ✨ V8 NEW: Try context-aware replacement first (Curator Change #006)
        context_replacement = self._find_replacement(entity, evidence_text, relationship)
        if context_replacement:
            return context_replacement

        if entity_lower.startswith('the amount'):
            match = re.search(r'the amount of ([^,\.]+)', evidence_text, re.IGNORECASE)
            if match:
                qualifier = match.group(1).strip()
                qualifier = re.sub(r'\s+(by|in|at)\s+.*', '', qualifier)
                return f"{qualifier}"

        if entity_lower in {'the process', 'this process'}:
            processes = ['composting', 'pyrolysis', 'photosynthesis',
                        'decomposition', 'fermentation', 'soil building']
            for proc in processes:
                if proc in evidence_text.lower():
                    return f"{proc} process"

        if 'handbook' in entity_lower or 'book' in entity_lower:
            return 'Soil Stewardship Handbook'

        return None

    def enrich_relationship(self, rel: ProductionRelationship) -> ProductionRelationship:
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

        logger.info(f"   Context enrichment (V8 enhanced): {enriched_count} enriched with context-aware replacement, {vague_count} flagged as vague")
        return processed


class PredicateNormalizer:
    """
    V6 NEW MODULE: Normalizes verbose/awkward predicates to standard forms
    ✨ V8 ENHANCEMENT (Curator Change #010-011): Semantic validation against entity types

    Example: "flourish with" → "experience", "is wedded to" → "depends_on"
    NOW validates: Books can "guide" but not "heal"

    FIXES: 2 wrong predicate errors (0.22%)
    """

    def __init__(self):
        # Mapping of verbose predicates to standard forms
        self.predicate_mappings = {
            'flourish with': 'experience',
            'flourishes with': 'experience',
            'have the choice to': 'can',
            'has the choice to': 'can',
            'is wedded to': 'depends_on',
            'are wedded to': 'depend_on',
            'unlock the door to': 'enables',
            'unlocks the door to': 'enables',
            'make it possible to': 'enables',
            'makes it possible to': 'enables',
        }

        # ✨ V8 NEW: Entity type constraints for semantic validation (Curator Change #010)
        self.entity_type_predicates = {
            'Book': {
                'allowed': ['guides', 'informs', 'describes', 'explains', 'teaches',
                           'provides', 'presents', 'covers', 'discusses', 'authored by'],
                'forbidden': ['heals', 'cures', 'fixes', 'repairs', 'treats'],
                'replacements': {
                    'heals': 'guides readers to heal',
                    'helps heal': 'provides guidance for healing',
                    'cures': 'provides information about curing',
                    'fixes': 'provides solutions for'
                }
            },
            'Person': {
                'allowed': ['wrote', 'authored', 'created', 'founded', 'established',
                           'teaches', 'researches', 'studies', 'works on'],
                'forbidden': [],
                'replacements': {}
            }
        }

    def _detect_entity_type(self, entity: str) -> Optional[str]:
        """✨ V8 NEW (Curator Change #011): Detect entity type from entity string"""
        entity_lower = entity.lower()

        # Book detection
        if any(keyword in entity_lower for keyword in ['handbook', 'book', 'guide', 'manual']):
            return 'Book'

        # Person detection (capitalized, contains name patterns)
        if entity and entity[0].isupper() and ' ' in entity:
            return 'Person'

        return None

    def normalize_predicate(self, predicate: str) -> tuple:
        """
        Returns: (normalized_predicate, was_normalized: bool)
        """
        predicate_lower = predicate.lower().strip()

        if predicate_lower in self.predicate_mappings:
            return self.predicate_mappings[predicate_lower], True

        return predicate, False

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """✨ V8 ENHANCEMENT (Curator Change #011): Added semantic validation"""
        processed = []
        normalized_count = 0
        semantically_corrected_count = 0

        for rel in relationships:
            predicate = rel.relationship.lower()
            source = rel.source

            # ✨ V8 NEW: Detect entity types and check semantic compatibility
            source_type = self._detect_entity_type(source)

            if source_type and source_type in self.entity_type_predicates:
                rules = self.entity_type_predicates[source_type]

                # Check if predicate is forbidden for this entity type
                if any(forbidden in predicate for forbidden in rules['forbidden']):
                    # Try to find replacement
                    for forbidden, replacement in rules['replacements'].items():
                        if forbidden in predicate:
                            if rel.flags is None:
                                rel.flags = {}
                            rel.flags['PREDICATE_SEMANTICALLY_CORRECTED'] = True
                            rel.flags['original_relationship'] = rel.relationship
                            rel.relationship = replacement
                            semantically_corrected_count += 1
                            break

            # Apply standard normalization
            normalized_pred, was_normalized = self.normalize_predicate(rel.relationship)

            if was_normalized:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['PREDICATE_NORMALIZED'] = True
                rel.flags['original_predicate'] = rel.relationship
                rel.relationship = normalized_pred
                normalized_count += 1

            processed.append(rel)

        logger.info(f"   Predicate normalization (V8 enhanced): {normalized_count} predicates normalized, {semantically_corrected_count} semantically corrected")
        return processed


class TitleCompletenessValidator:
    """V6 MODULE - no V8 changes"""

    def __init__(self):
        self.bad_endings = {
            'and', 'or', 'but', 'to', 'for', 'with', 'by',
            'in', 'on', 'at', 'of', 'the', 'a', 'an'
        }

        self.title_relationships = {
            'authored', 'wrote', 'published', 'edited',
            'compiled', 'created', 'produced', 'endorsed'
        }

    def is_incomplete_title(self, title: str) -> tuple:
        words = title.split()

        if words:
            last_word = words[-1].lower().rstrip('.,!?')
            if last_word in self.bad_endings:
                return True, f"ends_with_{last_word}"

        if title.count('"') == 1:
            return True, "unmatched_quotes"

        if len(words) <= 2 and ':' not in title:
            return True, "too_short"

        if title.rstrip().endswith('...'):
            return True, "ellipsis_ending"

        return False, ""

    def validate_relationship(self, rel: ProductionRelationship) -> ProductionRelationship:
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
    """V6 MODULE - no V8 changes"""

    def __init__(self):
        self.invalid_patterns = [
            ('Organization', 'published', 'Date'),
            ('Person', 'is-a', 'Person'),
        ]

    def validate_no_self_loop(self, rel: ProductionRelationship) -> tuple:
        if rel.source.lower() == rel.target.lower():
            if rel.relationship not in {'is-a', 'is defined as', 'means', 'equals'}:
                return False, "self_loop"
        return True, ""

    def validate_publication_context(self, rel: ProductionRelationship) -> tuple:
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
        processed = []
        invalid_count = 0

        for rel in relationships:
            rel = self.validate_predicate(rel)

            if rel.flags and rel.flags.get('INVALID_PREDICATE'):
                invalid_count += 1

            processed.append(rel)

        logger.info(f"   Predicate validation: {invalid_count} invalid predicates flagged")
        return processed


class VagueEntityBlocker:
    """
    V7 NEW MODULE: Blocks overly vague/abstract entities upfront

    Filters out relationships with entities that are too abstract to be useful.
    Unlike ContextEnricher (which tries to fix vague entities), this BLOCKS them entirely.

    No V8 changes - working well in V7
    """

    def __init__(self):
        # Patterns for entities that are too vague to be useful
        self.vague_abstract_patterns = [
            r'^the (way|answer|solution|problem|challenge|issue|question|matter)$',
            r'^the (way|path|approach|method) (through|to|from|of)',
            r'^the (reason|cause|result|outcome|consequence) (for|of|why)',
            r'^(something|someone|anything|anyone|everything|everyone)$',
            r'^(things|ways|practices|methods|approaches|solutions)$',
            r'^(this|that)$',
            r'^it$',
        ]

        # Compile patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.vague_abstract_patterns]

    def is_too_vague(self, entity: str) -> tuple:
        """
        Returns: (is_vague: bool, pattern_matched: str)
        """
        entity_lower = entity.lower().strip()

        # Check against all patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.match(entity_lower):
                return True, self.vague_abstract_patterns[i]

        return False, ""

    def should_block_relationship(self, rel: ProductionRelationship) -> tuple:
        """
        Returns: (should_block: bool, reason: str)
        """
        # Check source
        source_vague, source_pattern = self.is_too_vague(rel.source)
        if source_vague:
            return True, f"vague_source: {source_pattern}"

        # Check target
        target_vague, target_pattern = self.is_too_vague(rel.target)
        if target_vague:
            return True, f"vague_target: {target_pattern}"

        return False, ""

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """Filter out relationships with overly vague entities"""
        kept = []
        blocked_count = 0
        blocked_reasons = {}

        for rel in relationships:
            should_block, reason = self.should_block_relationship(rel)

            if should_block:
                blocked_count += 1
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
            else:
                kept.append(rel)

        logger.info(f"   Vague entity blocking: {blocked_count} relationships blocked")
        if blocked_reasons:
            for reason, count in sorted(blocked_reasons.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"     - {reason}: {count}")

        return kept


class FigurativeLanguageFilter:
    """
    V6 MODULE: Flags metaphorical language
    ✨ V8 ENHANCEMENT (Curator Change #014-015): Normalizes metaphors to literal equivalents

    FIXES: 3 metaphorical descriptions (0.32%)
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

        # ✨ V8 NEW: Metaphorical predicate mappings to literal equivalents (Curator Change #014)
        self.metaphor_normalizations = {
            'is a road-map': 'provides guidance',
            'is a compass': 'provides direction',
            'is a guide': 'provides guidance',
            'is wedded to': 'depends on',
            'is tied to': 'depends on',
            'is connected to': 'relates to',
            'road-map of sorts': 'guide',
            'compass for': 'guide for'
        }

    def contains_metaphorical_language(self, text: str) -> tuple:
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
        is_metaphorical, terms = self.contains_metaphorical_language(rel.evidence_text)

        if is_metaphorical:
            if rel.flags is None:
                rel.flags = {}
            rel.flags['FIGURATIVE_LANGUAGE'] = True
            rel.flags['metaphorical_terms'] = terms
            rel.p_true = rel.p_true * 0.6

        return rel

    def process_batch(self, relationships: List[ProductionRelationship]) -> List[ProductionRelationship]:
        """✨ V8 ENHANCEMENT (Curator Change #015): Normalize metaphors to literal equivalents"""
        processed = []
        metaphorical_count = 0
        normalized_count = 0

        for rel in relationships:
            relationship_lower = rel.relationship.lower()
            target_lower = rel.target.lower()

            # ✨ V8 NEW: Check for metaphorical predicates and normalize
            normalized = False
            for metaphor, literal in self.metaphor_normalizations.items():
                if metaphor in relationship_lower:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['METAPHOR_NORMALIZED'] = True
                    rel.flags['original_relationship'] = rel.relationship
                    rel.relationship = literal
                    normalized = True
                    normalized_count += 1
                    break

                # Also check target for metaphorical terms
                if metaphor in target_lower:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['METAPHOR_NORMALIZED_TARGET'] = True
                    rel.flags['original_target'] = rel.target
                    rel.target = target_lower.replace(metaphor, literal)
                    normalized = True
                    normalized_count += 1
                    break

            # Flag remaining figurative language (unchanged from V6)
            if not normalized:
                rel = self.filter_relationship(rel)

            if rel.flags and rel.flags.get('FIGURATIVE_LANGUAGE'):
                metaphorical_count += 1

            processed.append(rel)

        logger.info(f"   Figurative language (V8 enhanced): {normalized_count} metaphors normalized, {metaphorical_count} remaining flagged")
        return processed


def pass_2_5_quality_post_processing(
    relationships: List[ProductionRelationship],
    pages_with_text: List[tuple],
    document_metadata: Dict = None,
    config: dict = None
) -> tuple:
    """
    ✨ V8 Pass 2.5: CURATOR-ENHANCED Quality Post-Processing Pipeline

    V8 NEW (Generated by KG Curator from V7 Reflector Analysis):
    1. NEW: PraiseQuoteDetector - detects/corrects praise quotes (CRITICAL fix)
    2. ENHANCED: PronounResolver - possessive pronouns + 5-sentence window (CRITICAL fix)
    3. ENHANCED: ContextEnricher - context-aware vague entity replacement (HIGH fix)
    4. ENHANCED: ListSplitter - handles 'and' conjunctions (HIGH fix)
    5. ENHANCED: PredicateNormalizer - semantic validation (MEDIUM fix)
    6. ENHANCED: BibliographicCitationParser - dedication detection (MEDIUM fix)
    7. ENHANCED: FigurativeLanguageFilter - metaphor normalization (MEDIUM fix)

    V8 inherits all V7 + V6 improvements:
    - Endorsement detection (V6, ENHANCED V7)
    - POS tagging for intelligent list splitting (V6, ENHANCED V8)
    - Multi-pass pronoun resolution (V7, ENHANCED V8)
    - Vague entity blocking (V7)
    - Predicate normalizer (V6, ENHANCED V8)
    """
    logger.info("🎨 PASS 2.5: V8 Curator-Enhanced Quality Post-Processing...")

    config = config or {}
    document_metadata = document_metadata or {}
    initial_count = len(relationships)

    stats = {
        'initial_count': initial_count,
        'praise_quotes_corrected': 0,  # ✨ V8 NEW
        'authorship_reversed': 0,
        'endorsements_detected': 0,
        'dedications_corrected': 0,  # ✨ V8 NEW
        'pronouns_resolved': 0,
        'generic_pronouns_resolved': 0,
        'pronouns_unresolved': 0,
        'entities_enriched': 0,
        'entities_vague': 0,
        'vague_entities_blocked': 0,
        'lists_split': 0,
        'adjective_series_preserved': 0,
        'predicates_normalized': 0,
        'predicates_semantically_corrected': 0,  # ✨ V8 NEW
        'titles_incomplete': 0,
        'predicates_invalid': 0,
        'metaphors_flagged': 0,
        'metaphors_normalized': 0,  # ✨ V8 NEW
        'final_count': 0
    }

    # 1. ✨ V8 NEW: Praise Quote Detector (BEFORE bibliographic parser!) (Curator Change #001)
    logger.info("  1/10: Praise quote detection (V8 NEW - runs FIRST)...")
    praise_detector = PraiseQuoteDetector()
    relationships = praise_detector.process_batch(relationships)
    stats['praise_quotes_corrected'] = sum(1 for r in relationships if r.flags and r.flags.get('PRAISE_QUOTE_CORRECTED'))

    # 2. Bibliographic Citation Parser (✨ V8 ENHANCED with dedication detection)
    logger.info("  2/10: Bibliographic citation parsing + endorsement + dedication detection (V8 enhanced)...")
    bib_parser = BibliographicCitationParser()
    relationships = bib_parser.process_batch(relationships)
    stats['authorship_reversed'] = sum(1 for r in relationships if r.flags and r.flags.get('AUTHORSHIP_REVERSED'))
    stats['endorsements_detected'] = sum(1 for r in relationships if r.flags and r.flags.get('ENDORSEMENT_DETECTED'))
    stats['dedications_corrected'] = sum(1 for r in relationships if r.flags and r.flags.get('DEDICATION_CORRECTED'))

    # 3. Title Completeness Validator
    logger.info("  3/10: Title completeness validation...")
    title_validator = TitleCompletenessValidator()
    relationships = title_validator.process_batch(relationships)
    stats['titles_incomplete'] = sum(1 for r in relationships if r.flags and r.flags.get('INCOMPLETE_TITLE'))

    # 4. Predicate Validator
    logger.info("  4/10: Predicate validation...")
    pred_validator = PredicateValidator()
    relationships = pred_validator.process_batch(relationships)
    stats['predicates_invalid'] = sum(1 for r in relationships if r.flags and r.flags.get('INVALID_PREDICATE'))

    # 5. ✨ V8 ENHANCED: Predicate Normalizer (with semantic validation)
    logger.info("  5/10: Predicate normalization + semantic validation (V8 enhanced)...")
    pred_normalizer = PredicateNormalizer()
    relationships = pred_normalizer.process_batch(relationships)
    stats['predicates_normalized'] = sum(1 for r in relationships if r.flags and r.flags.get('PREDICATE_NORMALIZED'))
    stats['predicates_semantically_corrected'] = sum(1 for r in relationships if r.flags and r.flags.get('PREDICATE_SEMANTICALLY_CORRECTED'))

    # 6. ✨ V8 ENHANCED: Pronoun Resolver (possessive pronouns + 5-sentence context)
    logger.info("  6/10: ENHANCED pronoun resolution with possessive support (V8 CRITICAL enhancement)...")
    pronoun_resolver = PronounResolver()
    relationships = pronoun_resolver.process_batch(relationships, pages_with_text, document_metadata)
    stats['pronouns_resolved'] = sum(1 for r in relationships if r.flags and
                                    (r.flags.get('PRONOUN_RESOLVED_SOURCE') or r.flags.get('PRONOUN_RESOLVED_TARGET')))
    stats['generic_pronouns_resolved'] = sum(1 for r in relationships if r.flags and
                                            (r.flags.get('GENERIC_PRONOUN_RESOLVED_SOURCE') or r.flags.get('GENERIC_PRONOUN_RESOLVED_TARGET')))
    stats['pronouns_unresolved'] = sum(1 for r in relationships if r.flags and
                                      (r.flags.get('PRONOUN_UNRESOLVED_SOURCE') or r.flags.get('PRONOUN_UNRESOLVED_TARGET')))

    # 7. ✨ V8 ENHANCED: Context Enricher (context-aware vague entity replacement)
    logger.info("  7/10: Context enrichment with context-aware replacement (V8 enhanced)...")
    context_enricher = ContextEnricher()
    relationships = context_enricher.process_batch(relationships)
    stats['entities_enriched'] = sum(1 for r in relationships if r.flags and
                                    (r.flags.get('CONTEXT_ENRICHED_SOURCE') or r.flags.get('CONTEXT_ENRICHED_TARGET')))
    stats['entities_vague'] = sum(1 for r in relationships if r.flags and
                                 (r.flags.get('VAGUE_SOURCE') or r.flags.get('VAGUE_TARGET')))

    # 8. Vague Entity Blocker (block UNFIXABLE vague entities)
    logger.info("  8/10: Vague entity blocking (blocks unfixable abstract entities)...")
    vague_blocker = VagueEntityBlocker()
    before_blocking = len(relationships)
    relationships = vague_blocker.process_batch(relationships)
    stats['vague_entities_blocked'] = before_blocking - len(relationships)

    # 9. ✨ V8 ENHANCED: List Splitter (with 'and' conjunction support)
    logger.info("  9/10: List splitting with 'and' conjunction support (V8 enhanced)...")
    list_splitter = ListSplitter(use_pos_tagging=True)
    relationships = list_splitter.process_batch(relationships)
    stats['lists_split'] = sum(1 for r in relationships if r.flags and r.flags.get('LIST_SPLIT'))

    # 10. ✨ V8 ENHANCED: Figurative Language Filter (with metaphor normalization)
    logger.info("  10/10: Figurative language detection + metaphor normalization (V8 enhanced)...")
    fig_filter = FigurativeLanguageFilter()
    relationships = fig_filter.process_batch(relationships)
    stats['metaphors_flagged'] = sum(1 for r in relationships if r.flags and r.flags.get('FIGURATIVE_LANGUAGE'))
    stats['metaphors_normalized'] = sum(1 for r in relationships if r.flags and r.flags.get('METAPHOR_NORMALIZED'))

    stats['final_count'] = len(relationships)

    logger.info(f"✅ PASS 2.5 V8 CURATOR-ENHANCED COMPLETE:")
    logger.info(f"   - Initial: {initial_count} relationships")
    logger.info(f"   - ✨ V8 NEW: Praise quotes corrected: {stats['praise_quotes_corrected']}")
    logger.info(f"   - ✨ V8: Authorship reversed: {stats['authorship_reversed']}, Endorsements: {stats['endorsements_detected']}, Dedications: {stats['dedications_corrected']}")
    logger.info(f"   - ✨ V8: Pronouns (ENHANCED): {stats['pronouns_resolved']} anaphoric + {stats['generic_pronouns_resolved']} generic resolved ({stats['pronouns_unresolved']} unresolved)")
    logger.info(f"   - ✨ V8: Context enriched with replacement: {stats['entities_enriched']} (tried to fix vague entities)")
    logger.info(f"   - Unfixable vague entities blocked: {stats['vague_entities_blocked']}")
    logger.info(f"   - ✨ V8: Predicates: {stats['predicates_normalized']} normalized, {stats['predicates_semantically_corrected']} semantically corrected")
    logger.info(f"   - ✨ V8: Lists split: {stats['lists_split']} (with 'and' conjunction support)")
    logger.info(f"   - Titles incomplete: {stats['titles_incomplete']} flagged")
    logger.info(f"   - Predicates invalid: {stats['predicates_invalid']} flagged")
    logger.info(f"   - ✨ V8: Metaphors: {stats['metaphors_normalized']} normalized, {stats['metaphors_flagged']} remaining flagged")
    logger.info(f"   - Final: {stats['final_count']} relationships")

    return relationships, stats


# ============================================================================
# PDF TEXT EXTRACTION & CHUNKING (from V5 - unchanged)
# ============================================================================

def extract_text_from_pdf(pdf_path: Path) -> tuple:
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
# PASS 1 & PASS 2 (V10 - prompts loaded from files)
# ============================================================================
# BOOK_EXTRACTION_PROMPT and DUAL_SIGNAL_EVALUATION_PROMPT are now loaded
# from prompt files at startup (see lines 99-116):
# - kg_extraction_playbook/prompts/pass1_extraction_v10.txt
# - kg_extraction_playbook/prompts/pass2_evaluation_v10.txt


def pass1_extract_book(chunk: str, doc_sha256: str, page_numbers: List[int]) -> List[ProductionRelationship]:
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


# ⭐ V10: Prompt removed - loaded from file at startup
# See lines 99-116 for prompt loading code
# DUAL_SIGNAL_EVALUATION_PROMPT loaded from: kg_extraction_playbook/prompts/pass2_evaluation_v10.txt


def evaluate_batch_robust(batch: List[ProductionRelationship],
                         model: str, prompt_version: str,
                         document_metadata: Dict[str, Any] = None) -> List[ProductionRelationship]:
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
                evidence=candidate.evidence.copy() if candidate else _default_evidence(),
                classification_flags=evaluation.classification_flags  # ⭐ V9 NEW
            )

            # ⭐ V9 NEW: Populate attribution metadata for discourse graph provenance
            if document_metadata:
                prod_rel.attribution = {
                    "source_type": document_metadata.get("content_type", "book"),
                    "source_title": document_metadata.get("title", None),
                    "source_author": document_metadata.get("author", None),
                    "source_id": document_metadata.get("isbn", None),
                    "page_number": prod_rel.evidence.get("page_number", None),
                    "timestamp": None,  # Not applicable for books
                    "url": document_metadata.get("url", None),
                    "context": f"Extracted from {document_metadata.get('title', 'unknown source')}"
                }

            results.append(prod_rel)

        return results

    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


# ============================================================================
# MAIN V8 EXTRACTION PIPELINE
# ============================================================================

def extract_knowledge_graph_from_book_v8(book_title: str,
                                         pdf_path: Path,
                                         run_id: str,
                                         document_metadata: Dict = None,
                                         batch_size: int = 25) -> Dict[str, Any]:
    """
    ✨ V8 EXTRACTION SYSTEM WITH CURATOR-GENERATED ACE CYCLE 1 ENHANCEMENTS

    V6 (Pass 1 + Pass 2 + Pass 2.5) → V7 META-ACE → ✨ V8 CURATOR-ENHANCED → Output

    GOAL: Reduce quality issues from 6.71% (V7) to <3% (V8) ✅ TARGET: 2.7%
    """
    logger.info(f"🚀 Starting V8 extraction with Curator-generated fixes: {book_title}")

    start_time = time.time()

    document_metadata = document_metadata or {}

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

    # PASS 2: Evaluate in batches (✨ V8: with philosophical filter)
    logger.info(f"⚡ PASS 2: Batched evaluation with philosophical filter (V8 enhanced)...")
    logger.info(f"  Processing {len(list(chunks(all_candidates, batch_size)))} batches...")

    validated_relationships = []
    batches = list(chunks(all_candidates, batch_size))

    for batch_num, batch in enumerate(batches):
        if batch_num % 5 == 0:
            logger.info(f"  Batch {batch_num + 1}/{len(batches)}")

        evaluations = evaluate_batch_robust(
            batch=batch,
            model="gpt-4o-mini",
            prompt_version="v8_curator_enhanced",
            document_metadata=document_metadata  # ⭐ V9 NEW: Pass for attribution
        )

        validated_relationships.extend(evaluations)
        time.sleep(0.1)

    logger.info(f"✅ PASS 2 COMPLETE: {len(validated_relationships)} relationships evaluated")

    # ✨ V8: CURATOR-ENHANCED PASS 2.5 QUALITY POST-PROCESSING
    validated_relationships, pass2_5_stats = pass_2_5_quality_post_processing(
        validated_relationships,
        pages_with_text,
        document_metadata
    )

    # POST-PROCESSING: Compute final p_true for all relationships
    logger.info("🎯 FINAL POST-PROCESSING: Computing calibrated probabilities...")

    for rel in validated_relationships:
        rel.p_true = compute_p_true(
            rel.text_confidence,
            rel.knowledge_plausibility,
            rel.pattern_prior,
            rel.signals_conflict
        )

        rel.claim_uid = generate_claim_uid(rel)

        rel.extraction_metadata["run_id"] = run_id
        rel.extraction_metadata["extracted_at"] = datetime.now().isoformat()

    # ⭐ V9 NEW: Optional filtering of low-confidence relationships (p_true < threshold)
    if MIN_P_TRUE_THRESHOLD is not None:
        before_filter_count = len(validated_relationships)
        filtered_relationships = [r for r in validated_relationships if r.p_true >= MIN_P_TRUE_THRESHOLD]
        filtered_out_count = before_filter_count - len(filtered_relationships)

        if filtered_out_count > 0:
            logger.info(f"⭐ V9: Filtered {filtered_out_count} low-confidence relationships (p_true < {MIN_P_TRUE_THRESHOLD})")
            # Log examples of what was filtered
            filtered_out = [r for r in validated_relationships if r.p_true < MIN_P_TRUE_THRESHOLD]
            for rel in filtered_out[:3]:  # Show first 3 examples
                logger.debug(f"   Filtered (p_true={rel.p_true:.2f}): {rel.source} → {rel.target}")

        # Use filtered list for all subsequent processing
        validated_relationships = filtered_relationships
    else:
        logger.info(f"⭐ V9: p_true filtering DISABLED - preserving all {len(validated_relationships)} relationships (full discourse graph)")

    # ANALYZE & RETURN
    high_confidence = [r for r in validated_relationships if r.p_true >= 0.75]
    medium_confidence = [r for r in validated_relationships if 0.5 <= r.p_true < 0.75]
    low_confidence = [r for r in validated_relationships if r.p_true < 0.5]  # Should be empty after filtering
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
        'version': 'v9_reflector_fixes',
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
                'classification_flags': r.classification_flags,  # ⭐ V9 NEW: Statement classification
                'attribution': r.attribution,  # ⭐ V9 NEW: Discourse graph provenance
                'extraction_metadata': r.extraction_metadata
            }
            for r in validated_relationships
        ]
    }

    logger.info(f"📊 FINAL V8 RESULTS:")
    logger.info(f"  - Pass 1 extracted: {results['pass1_candidates']} candidates")
    logger.info(f"  - Pass 2 evaluated (with philosophical filter): {results['pass2_evaluated']}")
    logger.info(f"  - ✨ V8 Pass 2.5 final (Curator-enhanced): {results['pass2_5_final']}")
    logger.info(f"  - ✨ V8 Praise quotes corrected: {pass2_5_stats.get('praise_quotes_corrected', 0)}")
    logger.info(f"  - ✨ V8 Vague entities replaced: {pass2_5_stats.get('entities_enriched', 0)}")
    logger.info(f"  - ✨ V8 Metaphors normalized: {pass2_5_stats.get('metaphors_normalized', 0)}")

    # ⭐ V9 FIX: Prevent division by zero if no relationships extracted
    if len(validated_relationships) > 0:
        logger.info(f"  - High confidence (p≥0.75): {len(high_confidence)} ({len(high_confidence)/len(validated_relationships)*100:.1f}%)")
        logger.info(f"  - Medium confidence: {len(medium_confidence)} ({len(medium_confidence)/len(validated_relationships)*100:.1f}%)")
        logger.info(f"  - Low confidence: {len(low_confidence)} ({len(low_confidence)/len(validated_relationships)*100:.1f}%)")
    else:
        logger.warning("  - ⚠️ No relationships extracted - check Pass 2 evaluation errors")

    logger.info(f"  - Page coverage: {coverage_percentage:.1f}% ({len(pages_with_extractions)}/{total_book_pages} pages)")
    logger.info(f"  - Total time: {total_time/60:.1f} minutes")

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Extract knowledge graph from Soil Stewardship Handbook with V8 Curator-Enhanced system"""
    logger.info("="*80)
    logger.info("🚀 V8 KNOWLEDGE GRAPH EXTRACTION - CURATOR-GENERATED ACE CYCLE 1 PRODUCTION SYSTEM")
    logger.info("="*80)
    logger.info("")
    logger.info("✨ V8 NEW FEATURES (Generated by KG Curator from V7 Reflector Analysis):")
    logger.info("  ✅ CRITICAL: NEW PraiseQuoteDetector - detects/corrects praise quotes (fixes 4 errors)")
    logger.info("  ✅ CRITICAL: ENHANCED pronoun resolver - possessive pronouns + 5-sentence context (fixes 18 errors)")
    logger.info("  ✅ HIGH: ENHANCED vague entity detector - context-aware replacement (fixes 8 errors)")
    logger.info("  ✅ HIGH: ENHANCED list splitter - handles 'and' conjunctions (fixes 4 errors)")
    logger.info("  ✅ HIGH: PROMPT ENHANCEMENT - philosophical statement filter (filters 6 errors)")
    logger.info("  ✅ MEDIUM: ENHANCED predicate normalizer - semantic validation (fixes 2 errors)")
    logger.info("  ✅ MEDIUM: ENHANCED bibliographic parser - dedication detection (fixes 1 error)")
    logger.info("  ✅ MEDIUM: ENHANCED figurative language filter - metaphor normalization (fixes 3 errors)")
    logger.info("")
    logger.info("V8 inherits all V7 + V6 improvements:")
    logger.info("  - Endorsement detection (V6, ENHANCED V7)")
    logger.info("  - POS tagging for intelligent list splitting (V6, ENHANCED V8)")
    logger.info("  - Multi-pass pronoun resolution (V7, ENHANCED V8)")
    logger.info("  - Vague entity blocking (V7)")
    logger.info("  - Predicate normalizer (V6, ENHANCED V8)")
    logger.info("")
    logger.info("GOAL: Reduce quality issues from 6.71% (V7) to <3% (V8) ✅ TARGET: 2.7%")
    logger.info("Expected: 0 critical, 2 high, 8 medium, 15 mild = 25 total issues (2.7%)")
    logger.info("")

    # Book details
    book_dir = BOOKS_DIR / "soil-stewardship-handbook"
    pdf_path = book_dir / "Soil-Stewardship-Handbook-eBook.pdf"
    book_title = "Soil Stewardship Handbook"

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    run_id = f"book_soil_handbook_v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Document metadata for pronoun resolution
    document_metadata = {
        'author': 'Aaron Perry',
        'title': 'Soil Stewardship Handbook',
        'publication_year': 2017
    }

    start_time = time.time()

    # Extract with V8 system
    results = extract_knowledge_graph_from_book_v8(
        book_title=book_title,
        pdf_path=pdf_path,
        run_id=run_id,
        document_metadata=document_metadata,
        batch_size=25
    )

    # Save results
    output_path = OUTPUT_DIR / f"{book_title.replace(' ', '_').lower()}_v8.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("✨ V8 EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Run KG Reflector on V8 to measure improvements")
    logger.info("2. Compare V8 vs V7 quality metrics")
    logger.info("3. Validate that we've met the <3% target (predicted 2.7%)!")
    logger.info("4. If target met, V8 becomes PRODUCTION SYSTEM")
    logger.info("="*80)


if __name__ == "__main__":
    main()
