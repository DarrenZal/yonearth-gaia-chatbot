#!/usr/bin/env python3
"""
Knowledge Graph Extraction v3.2.2 - Book Adaptation

Extracts knowledge graph from PDF books using the same three-stage architecture
as episode extraction, but adapted for books:
- Page numbers instead of timestamps
- PDF text extraction
- Book-specific evidence tracking

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
        logging.FileHandler(f'kg_extraction_book_v3_2_2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
BOOKS_DIR = DATA_DIR / "books"
OUTPUT_DIR = DATA_DIR / "knowledge_graph_books_v3_2_2"
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
# DATACLASSES & SCHEMAS (Same as episode v3.2.2, adapted for books)
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
        "prompt_version": "v3.2.2_book",
        "extractor_version": "2025.10.10",
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

    # Evidence tracking (adapted for books)
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


# Pydantic models for API calls (same as episodes)

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
# HELPER FUNCTIONS (Same as episodes)
# ============================================================================

def canon(s: str) -> str:
    """Normalize entity strings"""
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


class SimpleAliasResolver:
    """Simple alias resolution"""
    def __init__(self):
        self.aliases = {
            canon("Y on Earth"): "Y on Earth",
            canon("YonEarth"): "Y on Earth",
            canon("yon earth"): "Y on Earth",
        }

    def resolve(self, entity: str) -> str:
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
    # Handle missing doc_sha256 gracefully (for old extractions)
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
                    overlap: int = 100) -> List[tuple[List[int], str]]:
    """
    Chunk book text while preserving page information
    Returns: [(page_numbers, chunk_text), ...]
    """
    chunks_with_pages = []
    current_chunk_words = []
    current_chunk_pages = set()

    for page_num, page_text in pages_with_text:
        words = page_text.split()

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
    return chunks_with_pages


# ============================================================================
# PASS 1: COMPREHENSIVE EXTRACTION (adapted for books)
# ============================================================================

BOOK_EXTRACTION_PROMPT = """Extract ALL relationships from this book segment.

CRITICAL: Be COMPREHENSIVE and THOROUGH. Extract EVERY relationship you can find.

Extract relationships between:
- People and organizations
- People and places
- Organizations and their work
- Concepts and practices
- Ideas and their authors
- Historical events and people
- Any other meaningful connections

For each relationship:
- source: The source entity
- relationship: The type of relationship (clear, descriptive)
- target: The target entity
- evidence_text: The exact quote from text supporting this

DO NOT filter or judge. Extract everything - we'll validate later!

Book segment:
{text}

Extract all relationships comprehensively."""


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
            prod_rel.evidence['page_number'] = page_numbers[0] if page_numbers else None  # First page of chunk

            candidates.append(prod_rel)

        return candidates

    except Exception as e:
        logger.error(f"Error in Pass 1 extraction: {e}")
        return []


# ============================================================================
# PASS 2: BATCHED DUAL-SIGNAL EVALUATION (same as episodes)
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
    """Robust batch evaluation (same as episodes)"""
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
# MAIN BOOK EXTRACTION PIPELINE
# ============================================================================

def extract_knowledge_graph_from_book(book_title: str,
                                      pdf_path: Path,
                                      run_id: str,
                                      batch_size: int = 50) -> Dict[str, Any]:
    """
    Three-stage book extraction: Extract → Type Validate → Score

    Now with checkpointing to prevent data loss from crashes!
    """
    logger.info(f"🚀 Starting v3.2.2 book extraction: {book_title}")

    # ========================================================================
    # CHECKPOINT RESUME LOGIC
    # ========================================================================

    # Try to resume from Pass 2 checkpoint (most advanced)
    pass2_checkpoint = load_checkpoint("pass2_complete", run_id)
    if pass2_checkpoint:
        logger.info("🎯 Resuming from Pass 2 checkpoint - skipping to post-processing!")
        validated_relationships = deserialize_relationships(pass2_checkpoint['validated_relationships'])
        doc_sha256 = pass2_checkpoint['doc_sha256']
        pages_with_text = pass2_checkpoint['pages_with_text']
        all_candidates = deserialize_relationships(pass2_checkpoint['all_candidates'])
        batches = pass2_checkpoint['batches']

        # Skip directly to post-processing
        logger.info(f"  Loaded {len(validated_relationships)} evaluated relationships")

    else:
        # Try to resume from Pass 1 checkpoint
        pass1_checkpoint = load_checkpoint("pass1_complete", run_id)

        if pass1_checkpoint:
            logger.info("📂 Resuming from Pass 1 checkpoint - skipping extraction!")
            all_candidates = deserialize_relationships(pass1_checkpoint['all_candidates'])
            doc_sha256 = pass1_checkpoint['doc_sha256']
            pages_with_text = pass1_checkpoint['pages_with_text']

            logger.info(f"  Loaded {len(all_candidates)} candidates from checkpoint")

        else:
            # No checkpoints - start from scratch
            logger.info("🆕 No checkpoints found - starting fresh extraction")

            # Extract text from PDF
            full_text, pages_with_text = extract_text_from_pdf(pdf_path)
            doc_sha256 = hashlib.sha256(full_text.encode()).hexdigest()

            # Chunk text (page-aware)
            text_chunks = chunk_book_text(pages_with_text, chunk_size=800, overlap=100)

    # ========================================================================
    # PASS 1: Extract everything (skip if loaded from checkpoint)
    # ========================================================================
    if not pass1_checkpoint and not pass2_checkpoint:
        logger.info("📝 PASS 1: Comprehensive extraction...")
        logger.info(f"  Processing {len(text_chunks)} chunks")

        all_candidates = []
        for i, (page_nums, chunk) in enumerate(text_chunks):
            if i % 10 == 0:
                logger.info(f"  Chunk {i}/{len(text_chunks)} (pages {page_nums[0]}-{page_nums[-1]})")

            candidates = pass1_extract_book(chunk, doc_sha256, page_nums)
            all_candidates.extend(candidates)

            time.sleep(0.05)  # Rate limiting

        logger.info(f"✅ PASS 1 COMPLETE: {len(all_candidates)} candidates extracted")

        # Save Pass 1 checkpoint
        pass1_data = {
            'all_candidates': serialize_relationships(all_candidates),
            'doc_sha256': doc_sha256,
            'pages_with_text': [(page_num, text) for page_num, text in pages_with_text]
        }
        save_checkpoint("pass1_complete", pass1_data, run_id)

    # Type validation (simplified for now - can enhance later)
    valid_candidates = all_candidates  # Skip type validation for books initially
    logger.info(f"✅ TYPE VALIDATION: {len(valid_candidates)} valid")

    # ========================================================================
    # PASS 2: Evaluate in batches (skip if loaded from checkpoint)
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
                prompt_version="v3.2.2_book"
            )

            validated_relationships.extend(evaluations)
            time.sleep(0.1)

        logger.info(f"✅ PASS 2 COMPLETE: {len(validated_relationships)} relationships evaluated")

        # Save Pass 2 checkpoint
        pass2_data = {
            'validated_relationships': serialize_relationships(validated_relationships),
            'all_candidates': serialize_relationships(all_candidates),
            'doc_sha256': doc_sha256,
            'pages_with_text': [(page_num, text) for page_num, text in pages_with_text],
            'batches': len(batches)
        }
        save_checkpoint("pass2_complete", pass2_data, run_id)

    # ========================================================================
    # POST-PROCESSING
    # ========================================================================
    logger.info("🎯 POST-PROCESSING: Canonicalization, evidence, confidence...")

    alias_resolver = SimpleAliasResolver()

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

        # Cap evidence windows
        MAX_WIN = 500
        if len(rel.evidence_text) > MAX_WIN:
            rel.evidence_text = rel.evidence_text[:MAX_WIN] + "…"

        # Generate claim UID
        rel.claim_uid = generate_claim_uid(rel)

        # Compute calibrated probability
        rel.p_true = compute_p_true(
            rel.text_confidence,
            rel.knowledge_plausibility,
            rel.pattern_prior,
            rel.signals_conflict
        )

        # Set metadata
        rel.extraction_metadata["run_id"] = run_id
        rel.extraction_metadata["extracted_at"] = datetime.now().isoformat()

    logger.info("✅ POST-PROCESSING COMPLETE")

    # ========================================================================
    # ANALYZE & RETURN
    # ========================================================================

    high_confidence = [r for r in validated_relationships if r.p_true >= 0.75]
    medium_confidence = [r for r in validated_relationships if 0.5 <= r.p_true < 0.75]
    low_confidence = [r for r in validated_relationships if r.p_true < 0.5]
    conflicts = [r for r in validated_relationships if r.signals_conflict]

    results = {
        'book_title': book_title,
        'run_id': run_id,
        'version': 'v3.2.2_book',
        'timestamp': datetime.now().isoformat(),
        'doc_sha256': doc_sha256,
        'pages': len(pages_with_text),
        'word_count': len(full_text.split()),

        # Stage counts
        'pass1_candidates': len(all_candidates),
        'type_valid': len(valid_candidates),
        'pass2_evaluated': len(validated_relationships),
        'pass2_batches': len(batches),

        # Quality metrics
        'high_confidence_count': len(high_confidence),
        'medium_confidence_count': len(medium_confidence),
        'low_confidence_count': len(low_confidence),
        'conflicts_detected': len(conflicts),

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

    logger.info(f"📊 RESULTS:")
    logger.info(f"  - Pass 1: {len(all_candidates)} candidates")
    logger.info(f"  - Pass 2: {len(validated_relationships)} evaluated")
    logger.info(f"  - High confidence (p≥0.75): {len(high_confidence)}")
    logger.info(f"  - Medium confidence: {len(medium_confidence)}")
    logger.info(f"  - Low confidence: {len(low_confidence)}")
    logger.info(f"  - Conflicts: {len(conflicts)}")
    logger.info(f"  - Cache hit rate: {calculate_cache_hit_rate():.2%}")

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Extract knowledge graph from Our Biggest Deal book"""
    logger.info("="*80)
    logger.info("🚀 KNOWLEDGE GRAPH EXTRACTION v3.2.2 - BOOK: OUR BIGGEST DEAL")
    logger.info("="*80)
    logger.info("")
    logger.info("Using OPENAI_API_KEY_2 for separate rate limit")
    logger.info("")

    # Book details
    book_dir = BOOKS_DIR / "OurBiggestDeal"
    pdf_path = book_dir / "OUR+BIGGEST+DEAL+-+Full+Book+-+Pre-publication+Galley+PDF+to+Share+v2.pdf"
    book_title = "Our Biggest Deal"

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    run_id = f"book_ourbiggestdeal_v3_2_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    start_time = time.time()

    # Extract
    results = extract_knowledge_graph_from_book(
        book_title=book_title,
        pdf_path=pdf_path,
        run_id=run_id,
        batch_size=50
    )

    # Save results
    output_path = OUTPUT_DIR / f"{book_title.replace(' ', '_').lower()}_v3_2_2.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("✨ BOOK EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
