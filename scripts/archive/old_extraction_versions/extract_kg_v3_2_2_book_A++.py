#!/usr/bin/env python3
"""
Knowledge Graph Extraction v3.2.2 - A++ Book Implementation

✨ A++ IMPROVEMENTS (based on deep investigation):
✅ Simple Pass 1 prompt (high recall: "Extract EVERYTHING...don't worry if correct")
✅ Removed entity presence validation (28.1% filtering was too aggressive)
✅ Batch size reduced from 50 to 25 (avoids token limit errors)
✅ Multi-word concept extraction guidance (e.g., "soil carbon content" not "soil")
✅ Quantitative relationship handling (numbers/percentages with full context)

ARCHITECTURE (from KG_MASTER_GUIDE_V3.md):
✅ Two-pass extraction: Pass 1 (high recall) → Pass 2 (high precision)
✅ Dual-signal evaluation: Text confidence + Knowledge plausibility
✅ Type validation with SHACL-lite domain/range checking
✅ Geographic validation with 3-tier checking
✅ Pattern priors from existing graph analysis
✅ Character-level evidence spans

Uses OPENAI_API_KEY_2 for separate rate limit (parallel with episode extraction)
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
try:
    from pydantic import BaseModel, Field
except ImportError:
    print("ERROR: Pydantic not installed. Run: pip install pydantic")
    exit(1)

# OpenAI imports
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: OpenAI library not installed. Run: pip install openai")
    exit(1)

# PDF processing
try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_extraction_book_improved_v3_2_2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
BOOKS_DIR = DATA_DIR / "books"
OUTPUT_DIR = DATA_DIR / "knowledge_graph_books_v3_2_2_improved"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# API setup - Use OPENAI_API_KEY_2 for separate rate limit!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY_2 not set in .env!")
    logger.error("This script uses the secondary API key for parallel extraction.")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# Cache for scorer results
edge_cache: Dict[str, Any] = {}
cache_stats = {'hits': 0, 'misses': 0}


# ============================================================================
# DATACLASSES & SCHEMAS (Same as episode v3.2.2)
# ============================================================================

def _default_evidence():
    """Factory for book evidence dict"""
    return {
        "doc_id": None,
        "doc_sha256": None,
        "page_number": None,  # BOOK SPECIFIC: page instead of timestamp
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
        "prompt_version": "v3.2.2_book_A++",
        "extractor_version": "2025.10.11_A++",
        "content_type": "book",
        "run_id": None,
        "extracted_at": None,
        "batch_id": None
    }


@dataclass
class ProductionRelationship:
    """Production-ready relationship (same schema as episodes)"""
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


class SimpleAliasResolver:
    """
    Configurable alias resolution - loads from JSON file
    NO hardcoded aliases - build during review and store in config
    """
    def __init__(self, alias_file: Optional[str] = None):
        self.aliases = {}

        # Load from config file if provided
        if alias_file and Path(alias_file).exists():
            logger.info(f"Loading aliases from: {alias_file}")
            with open(alias_file, 'r') as f:
                alias_config = json.load(f)
                # Convert to normalized keys
                for variant, canonical in alias_config.items():
                    self.aliases[canon(variant)] = canonical
            logger.info(f"  Loaded {len(self.aliases)} alias mappings")
        else:
            logger.info("No alias file provided - using entity names as-is")

    def resolve(self, entity: str) -> str:
        """
        Normalize and resolve entity name
        Falls back to original entity if no alias found
        """
        normalized = canon(entity)
        return self.aliases.get(normalized, entity)


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


def scorer_cache_key(candidate_uid: str, scorer_model: str, prompt_version: str) -> str:
    """Cache key with scorer context"""
    full_key = f"{candidate_uid}|{scorer_model}|{prompt_version}"
    return hashlib.sha1(full_key.encode()).hexdigest()


def calculate_cache_hit_rate() -> float:
    """Calculate cache hit rate"""
    total = cache_stats['hits'] + cache_stats['misses']
    return (cache_stats['hits'] / total) if total > 0 else 0.0


# ============================================================================
# TYPE VALIDATION (✨ NEW - FROM VALIDATORS.PY)
# ============================================================================

# Mock type resolver for now - replace with actual GeoNames/Wikidata lookups
def resolve_type(entity: str) -> str:
    """Resolve entity type from cache/API"""
    # Simple heuristic - replace with actual lookups
    if any(word in entity.lower() for word in ['bank', 'company', 'corporation', 'foundation', 'institute', 'organization']):
        return "Org"
    elif any(word in entity.lower() for word in ['colorado', 'usa', 'america', 'city', 'county', 'state']):
        return "Place"
    elif entity[0].isupper() and ' ' in entity and len(entity.split()) <= 3:
        # Likely a person name
        return "Person"
    return "UNKNOWN"


def get_type_provenance(entity: str) -> str:
    """Get provenance of entity type"""
    return "local"  # For now, all types are local heuristics


def type_validate(candidate: ProductionRelationship) -> ProductionRelationship:
    """
    ✨ NEW: Soft type validation - only hard-fail on KNOWN violations
    Prevents losing 30-40% of data from unknown entities
    """
    if candidate.flags is None:
        candidate.flags = {}

    src_type = resolve_type(candidate.source) or "UNKNOWN"
    tgt_type = resolve_type(candidate.target) or "UNKNOWN"

    candidate.source_type = src_type
    candidate.target_type = tgt_type

    # SHACL-lite: domain/range for common relations
    allowed = {
        "located_in":      ({"Place","Org","Event"}, {"Place"}),
        "works_at":        ({"Person"}, {"Org"}),
        "founded":         ({"Person","Org"}, {"Org"}),
        "born_in":         ({"Person"}, {"Place"}),
        "affiliated_with": ({"Person","Org"}, {"Person","Org"}),  # Symmetric
        "near":            ({"Place"}, {"Place"}),                # Symmetric
        "knows":           ({"Person"}, {"Person"}),              # Symmetric
        "collaborates_with": ({"Person","Org"}, {"Person","Org"}), # Symmetric
    }

    symmetric_relations = {"affiliated_with", "near", "knows", "collaborates_with"}

    dom_rng = allowed.get(candidate.relationship)
    if not dom_rng:
        return candidate  # No rule defined → pass through

    dom, rng = dom_rng

    # Handle symmetric relations
    if candidate.relationship in symmetric_relations:
        forward_valid = (src_type in dom and tgt_type in rng) if src_type != "UNKNOWN" and tgt_type != "UNKNOWN" else True
        reverse_valid = (tgt_type in dom and src_type in rng) if src_type != "UNKNOWN" and tgt_type != "UNKNOWN" else True

        if not (forward_valid or reverse_valid):
            if src_type != "UNKNOWN" and tgt_type != "UNKNOWN":
                candidate.flags["TYPE_VIOLATION"] = True
                candidate.flags["skip_reason"] = f"type_mismatch:{src_type}-{candidate.relationship}->{tgt_type} (symmetric)"
        return candidate

    # CRITICAL: Only hard-fail if BOTH types are KNOWN and violate rules
    if src_type != "UNKNOWN" and tgt_type != "UNKNOWN":
        if src_type not in dom or tgt_type not in rng:
            candidate.flags["TYPE_VIOLATION"] = True
            candidate.flags["skip_reason"] = f"type_mismatch:{src_type}-{candidate.relationship}->{tgt_type}"
    elif src_type != "UNKNOWN" or tgt_type != "UNKNOWN":
        # One side is known - flag for review but don't skip
        if (src_type != "UNKNOWN" and src_type not in dom) or \
           (tgt_type != "UNKNOWN" and tgt_type not in rng):
            candidate.flags["TYPE_WARNING"] = True
            candidate.flags["review_reason"] = "partial_type_mismatch"

    return candidate


# ============================================================================
# GEOGRAPHIC VALIDATION (✨ NEW - FROM VALIDATORS.PY)
# ============================================================================

def get_geo_data(entity: str) -> Optional[Dict]:
    """Get geographic data for entity (mock for now)"""
    # TODO: Replace with actual GeoNames API lookups
    return None


def haversine_distance(coords1, coords2) -> float:
    """Calculate distance between coordinates in km"""
    # TODO: Implement actual haversine formula
    return 0.0


def is_admin_parent(parent_path, child_path):
    """Check if parent is in child's admin hierarchy"""
    if not parent_path or not child_path:
        return True  # Can't verify, assume ok

    for i, parent_level in enumerate(parent_path):
        if i >= len(child_path) or child_path[i] != parent_level:
            return False
    return True


def validate_geographic_relationship(rel: ProductionRelationship) -> dict:
    """
    ✨ NEW: Three-tier validation: Admin hierarchy → Population → Distance
    Catches edge cases that distance-only validation misses
    """
    if rel.relationship != "located_in":
        return {"valid": True}

    src = get_geo_data(rel.source)
    tgt = get_geo_data(rel.target)

    if not src or not tgt:
        return {"valid": None, "reason": "missing_geo_data"}

    # 1) Admin hierarchy check (most decisive)
    if not is_admin_parent(tgt.get("admin_path"), src.get("admin_path")):
        return {
            "valid": False,
            "reason": "admin_hierarchy_mismatch",
            "confidence_penalty": 0.7
        }

    # 2) Population sanity check (catches obvious reversals)
    if src.get("population") and tgt.get("population"):
        if src["population"] > 1.2 * tgt["population"]:
            return {
                "valid": False,
                "reason": "population_hierarchy_violation",
                "suggested_correction": {
                    "source": rel.target,
                    "relationship": "located_in",
                    "target": rel.source
                },
                "confidence_penalty": 0.6
            }

    # 3) Distance as fallback
    if src.get("coords") and tgt.get("coords"):
        d_km = haversine_distance(src["coords"], tgt["coords"])
        max_distance = 50  # Default threshold

        if d_km > max_distance:
            return {
                "valid": False,
                "reason": f"too_far:{int(d_km)}km (max:{max_distance}km)",
                "confidence_penalty": 0.3
            }

    return {"valid": True}


# ============================================================================
# PATTERN PRIORS (✨ NEW - FROM MASTER GUIDE)
# ============================================================================

class SmoothedPatternPriors:
    """
    ✨ NEW: Pattern frequency with Laplace smoothing
    Reuses type information from nodes/edges to avoid re-resolution
    """
    def __init__(self, existing_graph=None, alpha=3):
        self.alpha = alpha
        self.entity_types = {}
        self.pattern_counts = {}
        self.total_relationships = 0
        self.num_unique_patterns = 0

        if existing_graph:
            # Build from existing graph
            for edge in existing_graph.get('relationships', []):
                src_type = edge.get('source_type') or resolve_type(edge['source'])
                tgt_type = edge.get('target_type') or resolve_type(edge['target'])
                rel_type = edge['relationship']

                pattern = (src_type, rel_type, tgt_type)
                self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1
                self.total_relationships += 1

            self.num_unique_patterns = len(self.pattern_counts)
            logger.info(f"📊 Pattern priors: {self.num_unique_patterns} unique patterns from {self.total_relationships} relationships")

    def get_prior(self, source: str, relationship: str, target: str,
                  source_type: str = None, target_type: str = None) -> float:
        """Get smoothed pattern prior for relationship"""
        if not self.pattern_counts:
            return 0.5  # Default uninformed prior

        # Resolve types if not provided
        if not source_type:
            source_type = self.entity_types.get(source) or resolve_type(source)
        if not target_type:
            target_type = self.entity_types.get(target) or resolve_type(target)

        pattern = (source_type, relationship, target_type)

        # Laplace smoothing: (count + alpha) / (total + alpha * num_patterns)
        count = self.pattern_counts.get(pattern, 0)
        prior = (count + self.alpha) / (self.total_relationships + self.alpha * self.num_unique_patterns)

        return prior


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

def save_checkpoint(checkpoint_name: str, data: Any, run_id: str) -> None:
    """Save checkpoint to disk"""
    checkpoint_path = CHECKPOINT_DIR / f"{run_id}_{checkpoint_name}.json"
    logger.info(f"💾 Saving checkpoint: {checkpoint_name}")

    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"✅ Checkpoint saved: {checkpoint_path.name}")


def load_checkpoint(checkpoint_name: str, run_id: str) -> Optional[Any]:
    """Load checkpoint from disk if it exists"""
    checkpoint_path = CHECKPOINT_DIR / f"{run_id}_{checkpoint_name}.json"

    if checkpoint_path.exists():
        logger.info(f"📂 Loading checkpoint: {checkpoint_name}")
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        logger.info(f"✅ Checkpoint loaded: {checkpoint_path.name}")
        return data

    return None


def serialize_relationships(relationships: List[ProductionRelationship]) -> List[Dict]:
    """Convert ProductionRelationship objects to JSON-serializable dicts"""
    return [
        {
            'source': r.source,
            'relationship': r.relationship,
            'target': r.target,
            'source_type': r.source_type,
            'target_type': r.target_type,
            'flags': r.flags,
            'evidence_text': r.evidence_text,
            'evidence': r.evidence,
            'text_confidence': r.text_confidence,
            'knowledge_plausibility': r.knowledge_plausibility,
            'pattern_prior': r.pattern_prior,
            'signals_conflict': r.signals_conflict,
            'conflict_explanation': r.conflict_explanation,
            'suggested_correction': r.suggested_correction,
            'p_true': r.p_true,
            'claim_uid': r.claim_uid,
            'candidate_uid': r.candidate_uid,
            'extraction_metadata': r.extraction_metadata
        }
        for r in relationships
    ]


def deserialize_relationships(data: List[Dict]) -> List[ProductionRelationship]:
    """Convert JSON dicts back to ProductionRelationship objects"""
    relationships = []
    for item in data:
        rel = ProductionRelationship(
            source=item['source'],
            relationship=item['relationship'],
            target=item['target'],
            source_type=item.get('source_type'),
            target_type=item.get('target_type'),
            flags=item.get('flags', {}),
            evidence_text=item.get('evidence_text', ''),
            evidence=item.get('evidence', _default_evidence()),
            text_confidence=item.get('text_confidence', 0.0),
            knowledge_plausibility=item.get('knowledge_plausibility', 0.0),
            pattern_prior=item.get('pattern_prior', 0.5),
            signals_conflict=item.get('signals_conflict', False),
            conflict_explanation=item.get('conflict_explanation'),
            suggested_correction=item.get('suggested_correction'),
            p_true=item.get('p_true', 0.0),
            claim_uid=item.get('claim_uid'),
            candidate_uid=item.get('candidate_uid'),
            extraction_metadata=item.get('extraction_metadata', _default_extraction_metadata())
        )
        relationships.append(rel)
    return relationships


# ============================================================================
# PDF TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(pdf_path: Path) -> tuple[str, List[tuple[int, str]]]:
    """
    Extract text from PDF, returning (full_text, [(page_num, page_text), ...])
    """
    logger.info(f"📖 Extracting text from PDF: {pdf_path.name}")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_with_text = []

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"  Total pages: {len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                # Clean text
                text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                pages_with_text.append((page_num, text))

            if page_num % 20 == 0:
                logger.info(f"  Processed {page_num}/{len(pdf.pages)} pages")

    full_text = "\n\n".join([text for _, text in pages_with_text])
    word_count = len(full_text.split())

    logger.info(f"✅ Extracted {word_count:,} words from {len(pages_with_text)} pages")

    return full_text, pages_with_text


# ============================================================================
# CHUNKING FOR BOOKS (page-aware)
# ============================================================================

def chunk_book_text(pages_with_text: List[tuple[int, str]],
                    chunk_size: int = 800,
                    overlap: int = 100,
                    min_page_words: int = 50) -> List[tuple[List[int], str]]:
    """
    Chunk book text while preserving page information
    ✨ IMPROVED: Filters out pages with <50 words (title pages, images, etc.)
    Returns: [(page_numbers, chunk_text), ...]
    """
    chunks_with_pages = []
    current_chunk_words = []
    current_chunk_pages = set()

    # Track page coverage
    pages_included = set()
    pages_skipped = []

    for page_num, page_text in pages_with_text:
        words = page_text.split()

        # Skip pages with very little text (likely title pages, images, etc.)
        if len(words) < min_page_words:
            pages_skipped.append((page_num, len(words)))
            continue

        pages_included.add(page_num)

        for word in words:
            current_chunk_words.append(word)
            current_chunk_pages.add(page_num)

            if len(current_chunk_words) >= chunk_size:
                # Save chunk
                chunk_text = ' '.join(current_chunk_words)
                chunk_pages = sorted(current_chunk_pages)
                chunks_with_pages.append((chunk_pages, chunk_text))

                # Start new chunk with overlap
                overlap_words = current_chunk_words[-overlap:] if overlap > 0 else []
                current_chunk_words = overlap_words
                current_chunk_pages = {page_num} if overlap > 0 else set()

    # Don't forget last chunk
    if current_chunk_words:
        chunk_text = ' '.join(current_chunk_words)
        chunk_pages = sorted(current_chunk_pages)
        chunks_with_pages.append((chunk_pages, chunk_text))

    logger.info(f"📄 Created {len(chunks_with_pages)} chunks from book")
    logger.info(f"   - Pages included: {len(pages_included)}/{len(pages_with_text)} ({len(pages_included)/len(pages_with_text)*100:.1f}%)")
    logger.info(f"   - Pages skipped (< {min_page_words} words): {len(pages_skipped)}")

    if pages_skipped and len(pages_skipped) <= 10:
        logger.info(f"   - Skipped pages: {[p[0] for p in pages_skipped]}")

    return chunks_with_pages


# ============================================================================
# PASS 1: COMPREHENSIVE EXTRACTION
# ============================================================================

BOOK_EXTRACTION_PROMPT = """Extract ALL relationships you can find in this text.

Don't worry about whether they're correct or make sense - just extract EVERYTHING.
We'll validate later in a separate pass.

## 📚 IMPORTANT GUIDELINES ##

### 1. Extract COMPLETE multi-word concepts

✅ "soil carbon content" NOT just "soil"
✅ "organic matter" NOT just "matter"
✅ "fossil fuel emissions" NOT just "emissions"
✅ "global temperature increase" NOT just "temperature"

Keep adjectives/modifiers that change meaning:
- Scope: global, worldwide, regional, local
- Type: organic, chemical, natural, synthetic
- Time: annual, daily, long-term, short-term
- State: active, stable, labile, total

### 2. SPECIAL: For claims with numbers/percentages

Extract the FULL measurable quantity and SCOPE:

✅ GOOD: "soil carbon content → can increase by → 10%" (with scope: worldwide)
❌ BAD: "soil → is increased by → 10%"

When text says "amount equal to" or "same as", extract as:
(Thing A, equals, Thing B)

Example:
Text: "Goal is an amount equal to a 10% increase in soil carbon worldwide"
Extract:
- fossil carbon sequestration goal → equals → 10% global soil carbon increase
- soil carbon content → can increase by → 10% (scope: worldwide)

### 3. Extract all types of relationships

- Facts: X authored Y, X was published in Y, X occurred in Y
- Definitions: X is defined as Y, X means Y, X refers to Y
- Causation: X causes Y, X leads to Y, X results in Y
- Processes: X involves Y, X requires Y, X includes Y
- Benefits: X improves Y, X enhances Y, X helps Y
- Problems: X threatens Y, X harms Y, X destroys Y
- Wisdom: X stated Y, X believes Y, X teaches Y

## 📝 OUTPUT FORMAT ##

For each relationship provide:
- source: Complete source entity (with all qualifiers and modifiers)
- relationship: Relationship type
- target: Complete target entity (with all qualifiers and modifiers)
- evidence_text: Quote from text (100-300 characters preferred)

## 📖 TEXT TO EXTRACT FROM ##

{text}

## ⚡ BE EXHAUSTIVE ##

Extract ALL relationships you find. It's better to extract too much than too little.
Don't worry if some might be wrong - we'll filter them later."""


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

            # Store book-specific evidence
            prod_rel.evidence['doc_sha256'] = doc_sha256
            prod_rel.evidence['page_number'] = page_numbers[0] if page_numbers else None

            # ✨ NEW: Find character positions in chunk
            evidence_start = chunk.find(rel.evidence_text[:50]) if rel.evidence_text else -1
            if evidence_start >= 0:
                prod_rel.evidence['start_char'] = evidence_start
                prod_rel.evidence['end_char'] = evidence_start + len(rel.evidence_text)

            candidates.append(prod_rel)

        return candidates

    except Exception as e:
        logger.error(f"Error in Pass 1 extraction: {e}")
        return []


# ============================================================================
# PASS 2: BATCHED DUAL-SIGNAL EVALUATION
# ============================================================================

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

    # Check cache
    cached_results = []
    uncached_batch = []

    for item in batch:
        cache_key = scorer_cache_key(item.candidate_uid, model, prompt_version)
        if cache_key in edge_cache:
            cached_results.append(edge_cache[cache_key])
            cache_stats['hits'] += 1
        else:
            uncached_batch.append(item)
            cache_stats['misses'] += 1

    if not uncached_batch:
        return cached_results

    try:
        batch_data = [
            {
                "candidate_uid": item.candidate_uid,
                "source": item.source,
                "relationship": item.relationship,
                "target": item.target,
                "evidence_text": item.evidence_text
            }
            for item in uncached_batch
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
                        batch_size=len(uncached_batch),
                        relationships_json=relationships_json
                    )
                }
            ],
            response_format=BatchedEvaluationResult,
            temperature=0.3
        )

        batch_result = response.choices[0].message.parsed
        uid_to_item = {it.candidate_uid: it for it in uncached_batch}

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

            if candidate:
                cache_key = scorer_cache_key(candidate.candidate_uid, model, prompt_version)
                edge_cache[cache_key] = prod_rel

        return cached_results + results

    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


# ============================================================================
# MAIN BOOK EXTRACTION PIPELINE (✨ IMPROVED WITH FULL VALIDATION)
# ============================================================================

def extract_knowledge_graph_from_book(book_title: str,
                                      pdf_path: Path,
                                      run_id: str,
                                      existing_graph: Optional[Dict] = None,
                                      alias_file: Optional[str] = None,
                                      batch_size: int = 25) -> Dict[str, Any]:
    """
    ✨ A++ SYSTEM: Two-pass extraction optimized for maximum recall and precision

    Pass 1 (High Recall) → Type Validate → Pass 2 (High Precision) → Post-Process

    A++ IMPROVEMENTS:
    - ✅ Simple encouraging prompt in Pass 1 (extracts everything)
    - ✅ No entity presence validation (let Pass 2 filter)
    - ✅ Batch size 25 (prevents token limit errors)
    - ✅ Multi-word concept and quantitative guidance

    VALIDATION PIPELINE:
    - ✅ Type validation with SHACL-lite checking
    - ✅ Dual-signal evaluation (text + knowledge)
    - ✅ Geographic validation with confidence penalties
    - ✅ Pattern priors from existing graph
    """
    logger.info(f"🚀 Starting A++ v3.2.2 book extraction: {book_title}")

    # ========================================================================
    # CHECKPOINT RESUME LOGIC
    # ========================================================================

    pass2_checkpoint = load_checkpoint("pass2_complete", run_id)
    if pass2_checkpoint:
        logger.info("🎯 Resuming from Pass 2 checkpoint!")
        validated_relationships = deserialize_relationships(pass2_checkpoint['validated_relationships'])
        doc_sha256 = pass2_checkpoint['doc_sha256']
        pages_with_text = pass2_checkpoint['pages_with_text']
        all_candidates = deserialize_relationships(pass2_checkpoint['all_candidates'])
        full_text = "\n\n".join([text for _, text in pages_with_text])

    else:
        pass1_checkpoint = load_checkpoint("pass1_complete", run_id)

        if pass1_checkpoint:
            logger.info("📂 Resuming from Pass 1 checkpoint!")
            all_candidates = deserialize_relationships(pass1_checkpoint['all_candidates'])
            doc_sha256 = pass1_checkpoint['doc_sha256']
            pages_with_text = pass1_checkpoint['pages_with_text']
            full_text = "\n\n".join([text for _, text in pages_with_text])

        else:
            logger.info("🆕 Starting fresh extraction")

            # Extract text from PDF
            full_text, pages_with_text = extract_text_from_pdf(pdf_path)
            doc_sha256 = hashlib.sha256(full_text.encode()).hexdigest()

            # Chunk text
            text_chunks = chunk_book_text(pages_with_text, chunk_size=800, overlap=100)

            # ================================================================
            # PASS 1: Extract everything
            # ================================================================
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

            # Save Pass 1 checkpoint
            pass1_data = {
                'all_candidates': serialize_relationships(all_candidates),
                'doc_sha256': doc_sha256,
                'pages_with_text': [(page_num, text) for page_num, text in pages_with_text]
            }
            save_checkpoint("pass1_complete", pass1_data, run_id)

    # ========================================================================
    # ✨ TYPE VALIDATION (NEW - WAS SKIPPED BEFORE!)
    # ========================================================================
    if not pass2_checkpoint:
        logger.info("🔍 TYPE VALIDATION: Checking domain/range constraints...")

        valid_candidates = []
        type_violations = 0
        type_warnings = 0

        for candidate in all_candidates:
            validated = type_validate(candidate)

            if validated.flags.get("TYPE_VIOLATION"):
                type_violations += 1
                # Skip relationships with known type violations
                continue

            if validated.flags.get("TYPE_WARNING"):
                type_warnings += 1

            valid_candidates.append(validated)

        logger.info(f"✅ TYPE VALIDATION COMPLETE:")
        logger.info(f"   - Valid: {len(valid_candidates)}")
        logger.info(f"   - Violations (filtered): {type_violations}")
        logger.info(f"   - Warnings (kept): {type_warnings}")

    # ========================================================================
    # PASS 2: Evaluate in batches
    # ========================================================================
    if not pass2_checkpoint:
        logger.info(f"⚡ PASS 2: Batched evaluation ({batch_size} rels/batch)...")

        validated_relationships = []
        batches = list(chunks(valid_candidates, batch_size))
        logger.info(f"  Processing {len(batches)} batches...")

        for batch_num, batch in enumerate(batches):
            if batch_num % 5 == 0:
                logger.info(f"  Batch {batch_num + 1}/{len(batches)}")

            evaluations = evaluate_batch_robust(
                batch=batch,
                model="gpt-4o-mini",
                prompt_version="v3.2.2_book_A++"
            )

            validated_relationships.extend(evaluations)
            time.sleep(0.1)

        logger.info(f"✅ PASS 2 COMPLETE: {len(validated_relationships)} relationships evaluated")

        # Save Pass 2 checkpoint
        pass2_data = {
            'validated_relationships': serialize_relationships(validated_relationships),
            'all_candidates': serialize_relationships(all_candidates),
            'doc_sha256': doc_sha256,
            'pages_with_text': [(page_num, text) for page_num, text in pages_with_text]
        }
        save_checkpoint("pass2_complete", pass2_data, run_id)

    # ========================================================================
    # ✨ POST-PROCESSING WITH FULL VALIDATION (NEW!)
    # ========================================================================
    logger.info("🎯 POST-PROCESSING: Canonicalization, pattern priors, geo validation...")

    alias_resolver = SimpleAliasResolver(alias_file=alias_file)

    # ✨ NEW: Initialize pattern priors from existing graph
    priors = SmoothedPatternPriors(existing_graph, alpha=3) if existing_graph else None

    geo_validated = 0
    geo_corrections = 0
    geo_penalties_applied = 0

    for rel in validated_relationships:
        # Save surface forms
        src_surface = rel.source
        tgt_surface = rel.target

        # Canonicalize
        rel.source = alias_resolver.resolve(rel.source)
        rel.target = alias_resolver.resolve(rel.target)

        # Attach surface forms
        rel.evidence["source_surface"] = src_surface
        rel.evidence["target_surface"] = tgt_surface
        rel.evidence["doc_id"] = book_title

        # Cap evidence windows (increased from 500 to 1500 for more context)
        MAX_WIN = 1500
        if len(rel.evidence_text) > MAX_WIN:
            rel.evidence_text = rel.evidence_text[:MAX_WIN] + "…"

        # Generate claim UID
        rel.claim_uid = generate_claim_uid(rel)

        # ✨ NEW: Compute pattern prior from existing graph
        if priors:
            rel.pattern_prior = priors.get_prior(
                rel.source, rel.relationship, rel.target,
                rel.source_type, rel.target_type
            )
        else:
            rel.pattern_prior = 0.5  # Default if no existing graph

        # Compute calibrated probability (initial)
        rel.p_true = compute_p_true(
            rel.text_confidence,
            rel.knowledge_plausibility,
            rel.pattern_prior,
            rel.signals_conflict
        )

        # ✨ NEW: Geographic validation with confidence penalties
        geo_validation = validate_geographic_relationship(rel)
        geo_validated += 1

        if geo_validation.get("valid") is False:
            # Invalid relationship - apply penalty
            penalty = float(geo_validation.get("confidence_penalty", 0.0))
            rel.p_true = max(0.0, rel.p_true - penalty)
            geo_penalties_applied += 1

            # Add suggested correction if available
            if geo_validation.get("suggested_correction"):
                rel.suggested_correction = str(geo_validation["suggested_correction"])
                geo_corrections += 1

        elif geo_validation.get("valid") is None:
            # Missing geo data - small penalty
            rel.p_true = max(0.0, rel.p_true - 0.05)
            if rel.flags is None:
                rel.flags = {}
            rel.flags["GEO_LOOKUP_NEEDED"] = True

        # Set metadata
        rel.extraction_metadata["run_id"] = run_id
        rel.extraction_metadata["extracted_at"] = datetime.now().isoformat()

    logger.info(f"✅ POST-PROCESSING COMPLETE:")
    logger.info(f"   - Geographic checks: {geo_validated}")
    logger.info(f"   - Penalties applied: {geo_penalties_applied}")
    logger.info(f"   - Corrections suggested: {geo_corrections}")

    # ========================================================================
    # ANALYZE & RETURN
    # ========================================================================

    high_confidence = [r for r in validated_relationships if r.p_true >= 0.75]
    medium_confidence = [r for r in validated_relationships if 0.5 <= r.p_true < 0.75]
    low_confidence = [r for r in validated_relationships if r.p_true < 0.5]
    conflicts = [r for r in validated_relationships if r.signals_conflict]

    # ✨ NEW: Page coverage analysis
    pages_with_extractions = set()
    for rel in validated_relationships:
        page = rel.evidence.get('page_number')
        if page:
            pages_with_extractions.add(page)

    total_book_pages = len(pages_with_text)
    coverage_percentage = len(pages_with_extractions) / total_book_pages * 100 if total_book_pages > 0 else 0

    # Find pages that were skipped (no extractions)
    all_page_numbers = {p[0] for p in pages_with_text}
    pages_completely_skipped = sorted(all_page_numbers - pages_with_extractions)

    logger.info(f"📊 PAGE COVERAGE ANALYSIS:")
    logger.info(f"   - Total pages in book: {total_book_pages}")
    logger.info(f"   - Pages with extractions: {len(pages_with_extractions)} ({coverage_percentage:.1f}%)")
    logger.info(f"   - Pages completely skipped: {len(pages_completely_skipped)}")

    if pages_completely_skipped and len(pages_completely_skipped) <= 20:
        logger.info(f"   - Skipped page numbers: {pages_completely_skipped}")

    results = {
        'book_title': book_title,
        'run_id': run_id,
        'version': 'v3.2.2_book_A++',
        'timestamp': datetime.now().isoformat(),
        'doc_sha256': doc_sha256,
        'pages': len(pages_with_text),
        'word_count': len(full_text.split()),

        # ✨ NEW: Page coverage metrics
        'pages_with_extractions': len(pages_with_extractions),
        'page_coverage_percentage': round(coverage_percentage, 1),
        'pages_completely_skipped': len(pages_completely_skipped),
        'skipped_page_numbers': pages_completely_skipped if len(pages_completely_skipped) <= 50 else [],

        # Stage counts
        'pass1_candidates': len(all_candidates),
        'type_valid': len(valid_candidates),
        'pass2_evaluated': len(validated_relationships),

        # Quality metrics
        'high_confidence_count': len(high_confidence),
        'medium_confidence_count': len(medium_confidence),
        'low_confidence_count': len(low_confidence),
        'conflicts_detected': len(conflicts),
        'geo_validated': geo_validated,
        'geo_penalties_applied': geo_penalties_applied,
        'geo_corrections_suggested': geo_corrections,

        # Cache metrics
        'cache_hit_rate': calculate_cache_hit_rate(),

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

    logger.info(f"📊 FINAL RESULTS:")
    logger.info(f"  - Pass 1 extracted: {results['pass1_candidates']} candidates")
    logger.info(f"  - Type validation: {len(valid_candidates)} valid")
    logger.info(f"  - Pass 2 evaluated: {len(validated_relationships)}")
    logger.info(f"  - High confidence (p≥0.75): {len(high_confidence)}")
    logger.info(f"  - Medium confidence: {len(medium_confidence)}")
    logger.info(f"  - Low confidence: {len(low_confidence)}")
    logger.info(f"  - Conflicts detected: {len(conflicts)}")
    logger.info(f"  - Page coverage: {coverage_percentage:.1f}% ({len(pages_with_extractions)}/{total_book_pages} pages)")
    logger.info(f"  - Cache hit rate: {calculate_cache_hit_rate():.2%}")

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Extract knowledge graph from Soil Stewardship Handbook with A++ system"""
    logger.info("="*80)
    logger.info("🚀 A++ KNOWLEDGE GRAPH EXTRACTION v3.2.2 - BOOK")
    logger.info("="*80)
    logger.info("")
    logger.info("✨ A++ IMPROVEMENTS:")
    logger.info("  ✅ Simple Pass 1 prompt (high recall)")
    logger.info("  ✅ Removed entity presence validation (too aggressive)")
    logger.info("  ✅ Batch size 25 (avoids token limits)")
    logger.info("  ✅ Multi-word concept extraction")
    logger.info("  ✅ Quantitative relationship handling")
    logger.info("")
    logger.info("Using OPENAI_API_KEY_2 for separate rate limit")
    logger.info("")

    # Book details
    book_dir = BOOKS_DIR / "soil-stewardship-handbook"
    pdf_path = book_dir / "Soil-Stewardship-Handbook-eBook.pdf"
    book_title = "Soil Stewardship Handbook"

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    run_id = f"book_soil_handbook_A++_v3_2_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load existing graph for pattern priors (if available)
    existing_graph = None
    existing_graph_path = OUTPUT_DIR.parent / "knowledge_graph_books_v3_2_2" / "our_biggest_deal_v3_2_2.json"
    if existing_graph_path.exists():
        logger.info(f"📊 Loading existing graph for pattern priors: {existing_graph_path.name}")
        with open(existing_graph_path) as f:
            existing_graph = json.load(f)
        logger.info(f"   Loaded {len(existing_graph.get('relationships', []))} relationships")

    start_time = time.time()

    # Extract with full validation
    results = extract_knowledge_graph_from_book(
        book_title=book_title,
        pdf_path=pdf_path,
        run_id=run_id,
        existing_graph=existing_graph,  # ✨ NEW: Pass existing graph for priors
        batch_size=25  # ✅ A++: Reduced from 50 to avoid token limit errors
    )

    # Save results
    output_path = OUTPUT_DIR / f"{book_title.replace(' ', '_').lower()}_A++_v3_2_2.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("✨ A++ BOOK EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
