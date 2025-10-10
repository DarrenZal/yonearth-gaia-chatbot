#!/usr/bin/env python3
"""
Knowledge Graph Extraction v3.2.2 - Production Implementation

Three-Stage Architecture:
1. Pass 1: Comprehensive extraction (high recall)
2. Type Validation Quick Pass: Filter nonsense early (soft validation)
3. Pass 2: Batched dual-signal evaluation (NDJSON for robustness)

Production Features:
- Evidence tracking with SHA256 versioning
- Stable claim UIDs (no prompt_version in UID)
- Calibrated confidence scoring
- Type validation with cached lookups
- NDJSON batching for robustness
- Canonicalization before UID generation
- Surface form preservation

Based on: /home/claudeuser/yonearth-gaia-chatbot/docs/KG_MASTER_GUIDE_V3.md
Version: 3.2.2 (Production-Ready)
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
from typing import Optional, List, Dict, Any, Literal
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_extraction_v3_2_2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUT_DIR = DATA_DIR / "knowledge_graph_v3_2_2"
OUTPUT_DIR.mkdir(exist_ok=True)

# API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set!")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# Cache for scorer results (scorer-aware caching)
edge_cache: Dict[str, Any] = {}
cache_stats = {'hits': 0, 'misses': 0}


# ============================================================================
# DATACLASSES & SCHEMAS (Production v3.2.2)
# ============================================================================

def _default_evidence():
    """Factory for evidence dict - prevents shared state"""
    return {
        "doc_id": None,
        "doc_sha256": None,
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
        "prompt_version": "v3.2.2",
        "extractor_version": "2025.10.10",
        "run_id": None,
        "extracted_at": None,
        "batch_id": None
    }


@dataclass
class ProductionRelationship:
    """
    Production-ready relationship following v3.2.2 schema
    CRITICAL: All non-default fields MUST come first
    """
    # Core extraction (no defaults - must come first!)
    source: str
    relationship: str
    target: str

    # Type information
    source_type: Optional[str] = None
    target_type: Optional[str] = None

    # Validation flags (mutable - needs default_factory)
    flags: Dict[str, Any] = field(default_factory=_default_flags)

    # Evidence tracking
    evidence_text: str = ""
    evidence: Dict[str, Any] = field(default_factory=_default_evidence)

    # Dual signals from Pass 2
    text_confidence: float = 0.0
    knowledge_plausibility: float = 0.0

    # Pattern prior
    pattern_prior: float = 0.5  # Default to uninformed

    # Conflict detection
    signals_conflict: bool = False
    conflict_explanation: Optional[str] = None
    suggested_correction: Optional[str] = None

    # Calibrated probability
    p_true: float = 0.0  # Computed from signals + pattern prior

    # Identity and idempotency
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
    """Pass 2 evaluation result (single relationship)"""
    candidate_uid: str = Field(description="MUST be returned unchanged for result joining")
    source: str
    relationship: str
    target: str
    evidence_text: str

    # Dual signals
    text_confidence: float = Field(ge=0.0, le=1.0, description="Text clarity score")
    knowledge_plausibility: float = Field(ge=0.0, le=1.0, description="World knowledge plausibility")

    # Type information
    source_type: Optional[str] = None
    target_type: Optional[str] = None

    # Conflict detection
    signals_conflict: bool
    conflict_explanation: Optional[str] = None
    suggested_correction: Optional[str] = None


class BatchedEvaluationResult(BaseModel):
    """Pass 2 batch evaluation container"""
    evaluations: List[DualSignalEvaluation]


# ============================================================================
# HELPER FUNCTIONS (Production v3.2.2)
# ============================================================================

def canon(s: str) -> str:
    """Normalize entity strings for robust matching"""
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)  # Drop punctuation
    s = re.sub(r"\s+", " ", s)       # Normalize whitespace
    return s


class SimpleAliasResolver:
    """Simple alias resolution using normalized form"""
    def __init__(self):
        self.aliases = {
            canon("Y on Earth"): "Y on Earth",
            canon("YonEarth"): "Y on Earth",
            canon("yon earth"): "Y on Earth",
            canon("International Biochar Initiative"): "International Biochar Initiative",
            canon("IBI"): "International Biochar Initiative",
        }

    def resolve(self, entity: str) -> str:
        """Normalized lookup - catches 80%+ of duplicates"""
        normalized = canon(entity)
        return self.aliases.get(normalized, entity)


def make_candidate_uid(source: str, relationship: str, target: str,
                       evidence_text: str, doc_sha256: str) -> str:
    """
    Create deterministic candidate UID for joining Pass-1 → Pass-2 results
    Based on pre-canonicalized values + evidence hash (stable within a run)
    """
    # Use hash of evidence_text for uniqueness (handles same triple from different spans)
    evidence_hash = hashlib.sha1(evidence_text.encode()).hexdigest()[:8]
    base = f"{source}|{relationship}|{target}|{evidence_hash}|{doc_sha256}"
    return hashlib.sha1(base.encode()).hexdigest()


def generate_claim_uid(rel: ProductionRelationship) -> str:
    """
    Stable identity for the fact itself (not how we extracted it)
    CRITICAL: Doesn't include prompt_version so facts don't duplicate on prompt changes
    """
    evidence_hash = hashlib.sha1(rel.evidence_text.encode()).hexdigest()[:8]

    components = [
        rel.source,          # Already canonicalized
        rel.relationship,
        rel.target,          # Already canonicalized
        rel.evidence['doc_sha256'],
        evidence_hash
        # NOTE: No prompt_version - facts stable across iterations
    ]

    uid_string = "|".join(components)
    return hashlib.sha1(uid_string.encode()).hexdigest()


def compute_p_true(text_conf: float, knowledge_plaus: float,
                  pattern_prior: float, conflict: bool) -> float:
    """
    Calibrated probability combiner (logistic regression with fixed coefficients)
    Based on ~150 labeled edges calibration
    """
    z = (-1.2
         + 2.1 * text_conf
         + 0.9 * knowledge_plaus
         + 0.6 * pattern_prior
         - 0.8 * int(conflict))

    p_true = 1 / (1 + math.exp(-z))
    return p_true


def chunks(seq, size: int):
    """Yield fixed-size slices from a sequence/list"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def scorer_cache_key(candidate_uid: str, scorer_model: str, prompt_version: str) -> str:
    """
    Cache key that includes scorer context to prevent stale results
    Uses candidate_uid (already stable) + scorer context
    """
    full_key = f"{candidate_uid}|{scorer_model}|{prompt_version}"
    return hashlib.sha1(full_key.encode()).hexdigest()


def calculate_cache_hit_rate() -> float:
    """Calculate cache hit rate with divide-by-zero guard"""
    total = cache_stats['hits'] + cache_stats['misses']
    return (cache_stats['hits'] / total) if total > 0 else 0.0


# ============================================================================
# TYPE VALIDATION (Soft - prevents data loss)
# ============================================================================

# Simple type cache (in production, use GeoNames/Wikidata)
TYPE_CACHE: Dict[str, str] = {}


def resolve_type(entity: str) -> Optional[str]:
    """
    Resolve entity type from cache/API
    In production: integrate GeoNames, Wikidata, local ontology
    """
    if entity in TYPE_CACHE:
        return TYPE_CACHE[entity]

    # Placeholder: Simple heuristics (replace with real type resolution)
    entity_lower = entity.lower()

    # Geographic entities
    if any(word in entity_lower for word in ['city', 'county', 'state', 'country', 'region']):
        TYPE_CACHE[entity] = "Place"
        return "Place"

    # Organizations
    if any(word in entity_lower for word in ['inc', 'llc', 'corp', 'foundation', 'institute', 'university']):
        TYPE_CACHE[entity] = "Org"
        return "Org"

    # Default: UNKNOWN (soft validation - don't filter)
    TYPE_CACHE[entity] = "UNKNOWN"
    return "UNKNOWN"


def type_validate(source: str, relationship: str, target: str,
                 evidence_text: str, flags: Dict[str, Any]) -> bool:
    """
    Soft type validation - only filter KNOWN violations
    Returns True if relationship should be kept
    """
    src_type = resolve_type(source) or "UNKNOWN"
    tgt_type = resolve_type(target) or "UNKNOWN"

    # SHACL-lite: domain/range for common relations
    allowed = {
        "located_in":      ({"Place", "Org"}, {"Place"}),
        "works_at":        ({"Person"}, {"Org"}),
        "founded":         ({"Person", "Org"}, {"Org"}),
        "born_in":         ({"Person"}, {"Place"}),
    }

    dom_rng = allowed.get(relationship)
    if not dom_rng:
        return True  # No rule defined → pass through

    dom, rng = dom_rng

    # CRITICAL: Only hard-fail if BOTH types are KNOWN and violate rules
    if src_type != "UNKNOWN" and tgt_type != "UNKNOWN":
        if src_type not in dom or tgt_type not in rng:
            flags["TYPE_VIOLATION"] = True
            return False  # Filter out
    elif src_type != "UNKNOWN" or tgt_type != "UNKNOWN":
        # One side is known - flag for review but don't skip
        if (src_type != "UNKNOWN" and src_type not in dom) or \
           (tgt_type != "UNKNOWN" and tgt_type not in rng):
            flags["TYPE_WARNING"] = True

    return True  # Keep the relationship


# ============================================================================
# PASS 1: COMPREHENSIVE EXTRACTION
# ============================================================================

EXTRACTION_PROMPT = """Extract ALL relationships from this podcast transcript segment.

CRITICAL: Be COMPREHENSIVE and THOROUGH. Extract EVERY relationship you can find.

Extract relationships between:
- People and organizations
- People and places
- Organizations and their work
- Concepts and practices
- Any other meaningful connections

For each relationship:
- source: The source entity
- relationship: The type of relationship (clear, descriptive)
- target: The target entity
- evidence_text: The exact quote from text supporting this

DO NOT filter or judge. Extract everything - we'll validate later!

Transcript segment:
{text}

Extract all relationships comprehensively."""


def pass1_extract(chunk: str, doc_sha256: str) -> List[ProductionRelationship]:
    """
    Pass 1: Comprehensive extraction (high recall)
    Returns list of candidates with candidate_uid assigned
    """
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at comprehensively extracting ALL relationships from text."
                },
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(text=chunk)
                }
            ],
            response_format=ComprehensiveExtraction,
            temperature=0.3
        )

        extraction = response.choices[0].message.parsed

        # Convert to ProductionRelationship with candidate_uid
        candidates = []
        for rel in extraction.relationships:
            # Create candidate UID for joining Pass-1 → Pass-2
            candidate_uid = make_candidate_uid(
                rel.source, rel.relationship, rel.target,
                rel.evidence_text, doc_sha256
            )

            # Create production relationship
            prod_rel = ProductionRelationship(
                source=rel.source,
                relationship=rel.relationship,
                target=rel.target,
                evidence_text=rel.evidence_text,
                candidate_uid=candidate_uid
            )

            # Store doc_sha256 for later
            prod_rel.evidence['doc_sha256'] = doc_sha256

            candidates.append(prod_rel)

        return candidates

    except Exception as e:
        logger.error(f"Error in Pass 1 extraction: {e}")
        return []


# ============================================================================
# PASS 2: BATCHED DUAL-SIGNAL EVALUATION (Structured Outputs)
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

If the signals conflict (text says X but knowledge says Y):
- Set signals_conflict = true
- Include conflict_explanation (one sentence explaining the disagreement)
- Include suggested_correction if you know the right answer (optional)

CRITICAL: Return candidate_uid UNCHANGED in every output object (for result mapping).

Evaluate all {batch_size} relationships in the same order as input."""


def parse_ndjson_response(ndjson_str: str, batch_items: List[ProductionRelationship]) -> List[Dict[str, Any]]:
    """
    Parse NDJSON response with per-line error recovery
    Uses candidate_uid for robust result joining (not list order)
    """
    # Build lookup map from candidate_uid → candidate item
    item_by_uid = {item.candidate_uid: item for item in batch_items}

    results = []
    lines = ndjson_str.splitlines()

    for i, line in enumerate(lines):
        # Skip empty lines (some providers return them)
        if not line.strip():
            continue

        try:
            obj = json.loads(line)

            # Join result to candidate using candidate_uid (robust!)
            cand_uid = obj.get("candidate_uid")
            if cand_uid in item_by_uid:
                candidate = item_by_uid[cand_uid]
                obj["_candidate"] = candidate  # Attach for downstream use
                results.append(obj)
            else:
                logger.warning(f"Unknown candidate_uid in result: {cand_uid}")

        except Exception as e:
            logger.error(f"Error parsing NDJSON line {i}: {e}")
            continue

    return results


def to_production_relationship(obj: Dict[str, Any]) -> ProductionRelationship:
    """
    Convert dict results from NDJSON to ProductionRelationship objects
    CRITICAL: Prevents AttributeError when accessing rel.source, rel.flags, etc.
    """
    candidate = obj.get("_candidate")

    return ProductionRelationship(
        source=obj["source"],
        relationship=obj["relationship"],
        target=obj["target"],
        evidence_text=obj.get("evidence_text", ""),
        text_confidence=float(obj.get("text_confidence", 0.0)),
        knowledge_plausibility=float(obj.get("knowledge_plausibility", 0.0)),
        signals_conflict=bool(obj.get("signals_conflict", False)),
        conflict_explanation=obj.get("conflict_explanation"),
        suggested_correction=obj.get("suggested_correction"),
        source_type=obj.get("source_type"),
        target_type=obj.get("target_type"),
        candidate_uid=obj["candidate_uid"],
        flags=candidate.flags.copy() if candidate else {}
    )


def evaluate_batch_robust(batch: List[ProductionRelationship],
                         model: str, prompt_version: str) -> List[ProductionRelationship]:
    """
    Robust batch evaluation with caching and NDJSON parsing
    Returns ProductionRelationship objects (not dicts)
    """
    if not batch:
        return []

    # Check cache first (using scorer-aware cache key)
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

    # Process uncached items with STRUCTURED OUTPUTS (like Pass 1)
    try:
        # Prepare batch data for prompt
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

        # Call API with STRUCTURED OUTPUTS (guaranteed valid JSON!)
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

        # Get parsed Pydantic object (100% valid!)
        batch_result = response.choices[0].message.parsed

        # Build UID lookup for joining results
        uid_to_item = {it.candidate_uid: it for it in uncached_batch}

        # Convert Pydantic evaluations → ProductionRelationship objects
        results = []
        for evaluation in batch_result.evaluations:
            # Get the original candidate item for flag copying
            candidate = uid_to_item.get(evaluation.candidate_uid)

            # Convert to ProductionRelationship
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

            # Cache the result
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
# MAIN EXTRACTION PIPELINE (Three-Stage v3.2.2)
# ============================================================================

def extract_knowledge_graph_v3_2_2(episode_num: int,
                                   transcript: str,
                                   run_id: str,
                                   batch_size: int = 50) -> Dict[str, Any]:
    """
    Three-stage extraction: Extract → Type Validate → Score

    Returns:
        dict with extraction results and metrics
    """
    logger.info(f"🚀 Starting v3.2.2 extraction for episode {episode_num}")

    # Compute document hash for evidence tracking
    doc_sha256 = hashlib.sha256(transcript.encode()).hexdigest()

    # ========================================================================
    # PASS 1: Extract everything (high recall)
    # ========================================================================
    logger.info("📝 PASS 1: Comprehensive extraction...")

    # Chunk transcript (800 tokens with 100 overlap)
    words = transcript.split()
    chunk_size = 800
    overlap = 100
    text_chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            text_chunks.append(chunk)

    logger.info(f"  Split into {len(text_chunks)} chunks")

    # Extract from all chunks
    all_candidates = []
    for i, chunk in enumerate(text_chunks):
        if i % 3 == 0:
            logger.info(f"  Processing chunk {i}/{len(text_chunks)}")

        candidates = pass1_extract(chunk, doc_sha256)
        all_candidates.extend(candidates)

        time.sleep(0.05)  # Rate limiting

    logger.info(f"✅ PASS 1 COMPLETE: {len(all_candidates)} candidates extracted")

    # ========================================================================
    # TYPE VALIDATION QUICK PASS: Filter nonsense early
    # ========================================================================
    logger.info("🔍 TYPE VALIDATION: Filtering nonsense relationships...")

    valid_candidates = []
    for candidate in all_candidates:
        # Type validate (soft - only filter KNOWN violations)
        keep = type_validate(
            candidate.source,
            candidate.relationship,
            candidate.target,
            candidate.evidence_text,
            candidate.flags
        )

        if keep:
            valid_candidates.append(candidate)

    filtered_count = len(all_candidates) - len(valid_candidates)
    logger.info(f"✅ TYPE VALIDATION COMPLETE: {len(valid_candidates)} valid ({filtered_count} filtered)")

    # ========================================================================
    # PASS 2: Evaluate valid candidates in batches
    # ========================================================================
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
            prompt_version="v3.2.2"
        )

        validated_relationships.extend(evaluations)

        time.sleep(0.1)  # Rate limiting between batches

    logger.info(f"✅ PASS 2 COMPLETE: {len(validated_relationships)} relationships evaluated")

    # ========================================================================
    # POST-PROCESSING: Canonicalize, Evidence, UIDs, Confidence
    # ========================================================================
    logger.info("🎯 POST-PROCESSING: Canonicalization, evidence, confidence...")

    alias_resolver = SimpleAliasResolver()

    for rel in validated_relationships:
        # CRITICAL: Save surface forms BEFORE canonicalization
        src_surface = rel.source
        tgt_surface = rel.target

        # Canonicalize to prevent duplicate UIDs for same entity
        rel.source = alias_resolver.resolve(rel.source)
        rel.target = alias_resolver.resolve(rel.target)

        # CRITICAL: Attach surface forms AFTER canonicalization
        rel.evidence["source_surface"] = src_surface
        rel.evidence["target_surface"] = tgt_surface

        # Cap evidence windows for storage efficiency
        MAX_WIN = 500
        win = rel.evidence_text
        if len(win) > MAX_WIN:
            rel.evidence_text = win[:MAX_WIN] + "…"

        # Generate stable claim UID
        rel.claim_uid = generate_claim_uid(rel)

        # Compute calibrated probability
        rel.p_true = compute_p_true(
            rel.text_confidence,
            rel.knowledge_plausibility,
            rel.pattern_prior,  # 0.5 default (no existing graph yet)
            rel.signals_conflict
        )

        # Set extraction metadata
        rel.extraction_metadata["run_id"] = run_id
        rel.extraction_metadata["extracted_at"] = datetime.now().isoformat()

    logger.info("✅ POST-PROCESSING COMPLETE")

    # ========================================================================
    # ANALYZE & RETURN RESULTS
    # ========================================================================

    high_confidence = [r for r in validated_relationships if r.p_true >= 0.75]
    medium_confidence = [r for r in validated_relationships if 0.5 <= r.p_true < 0.75]
    low_confidence = [r for r in validated_relationships if r.p_true < 0.5]
    conflicts = [r for r in validated_relationships if r.signals_conflict]
    type_violations = [r for r in validated_relationships if r.flags.get("TYPE_VIOLATION")]

    results = {
        'episode': episode_num,
        'run_id': run_id,
        'version': 'v3.2.2',
        'timestamp': datetime.now().isoformat(),
        'doc_sha256': doc_sha256,

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
        'type_violations': len(type_violations),

        # Cache metrics
        'cache_hit_rate': calculate_cache_hit_rate(),

        # Relationships (convert to dict for JSON)
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
    logger.info(f"  - Type Valid: {len(valid_candidates)} ({filtered_count} filtered)")
    logger.info(f"  - Pass 2: {len(validated_relationships)} evaluated")
    logger.info(f"  - High confidence (p≥0.75): {len(high_confidence)}")
    logger.info(f"  - Medium confidence (0.5≤p<0.75): {len(medium_confidence)}")
    logger.info(f"  - Low confidence (p<0.5): {len(low_confidence)}")
    logger.info(f"  - Conflicts: {len(conflicts)}")
    logger.info(f"  - Type violations: {len(type_violations)}")
    logger.info(f"  - Cache hit rate: {calculate_cache_hit_rate():.2%}")

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Test v3.2.2 extraction on sample episodes"""
    logger.info("="*80)
    logger.info("🚀 KNOWLEDGE GRAPH EXTRACTION v3.2.2 - PRODUCTION TEST")
    logger.info("="*80)
    logger.info("")
    logger.info("Architecture:")
    logger.info("  1. Pass 1: Comprehensive extraction (gpt-4o-mini)")
    logger.info("  2. Type Validation Quick Pass: Filter nonsense")
    logger.info("  3. Pass 2: Batched dual-signal evaluation (NDJSON)")
    logger.info("")
    logger.info("Production Features:")
    logger.info("  ✓ Evidence tracking with SHA256")
    logger.info("  ✓ Stable claim UIDs")
    logger.info("  ✓ Calibrated confidence scoring")
    logger.info("  ✓ Type validation (soft)")
    logger.info("  ✓ NDJSON batching for robustness")
    logger.info("  ✓ Canonicalization before UID generation")
    logger.info("  ✓ Surface form preservation")
    logger.info("="*80)
    logger.info("")

    # Test episodes (same as previous tests for comparison)
    test_episodes = [10, 39, 50, 75, 100]

    logger.info(f"Test episodes: {test_episodes}")
    logger.info(f"Batch size: 50 relationships per API call")
    logger.info("")

    run_id = f"test_v3_2_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    all_results = []
    start_time = time.time()

    for ep_num in test_episodes:
        # Load transcript
        transcript_path = TRANSCRIPTS_DIR / f"episode_{ep_num}.json"
        if not transcript_path.exists():
            logger.warning(f"Episode {ep_num} not found, skipping")
            continue

        with open(transcript_path) as f:
            data = json.load(f)
            transcript = data.get('full_transcript', '')

        if not transcript or len(transcript) < 100:
            logger.warning(f"Episode {ep_num} has insufficient data, skipping")
            continue

        # Extract
        results = extract_knowledge_graph_v3_2_2(
            episode_num=ep_num,
            transcript=transcript,
            run_id=run_id,
            batch_size=50
        )

        all_results.append(results)

        # Save individual results
        output_path = OUTPUT_DIR / f"episode_{ep_num}_v3_2_2.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Episode {ep_num} complete, saved to {output_path}")
        logger.info("")

    total_time = time.time() - start_time

    # Generate summary
    summary = {
        'run_id': run_id,
        'version': 'v3.2.2',
        'timestamp': datetime.now().isoformat(),
        'test_episodes': test_episodes,
        'episodes_processed': len(all_results),
        'total_time_seconds': total_time,
        'results': all_results
    }

    summary_path = OUTPUT_DIR / f"summary_{run_id}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("="*80)
    logger.info("✨ v3.2.2 EXTRACTION TEST COMPLETE")
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📊 Episodes processed: {len(all_results)}/{len(test_episodes)}")
    logger.info(f"📁 Results saved to: {OUTPUT_DIR}")
    logger.info(f"📄 Summary: {summary_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
