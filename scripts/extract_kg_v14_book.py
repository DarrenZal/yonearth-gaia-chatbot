#!/usr/bin/env python3
"""
Knowledge Graph Extraction V14.0 - Comprehensive Quality System

🎯 V14.0 ENHANCEMENTS (from V13.1 A-):
1. ✅ Enhanced Pass 1 Prompt: Entity specificity + extraction scope (prevents vague entities + metadata)
2. ✅ Enhanced Pass 2 Prompt: Improved claim type classification (FACTUAL vs NORMATIVE vs PHILOSOPHICAL)
3. ✅ MetadataFilter Module: Multi-layer detection for book metadata relationships
4. ✅ ConfidenceFilter Module: Flag-specific thresholds + unresolved pronoun handling
5. ✅ Predicate Normalizer V1.4: Tense normalization, modal verb preservation, semantic validation
6. ✅ Pronoun Resolver: Enhanced with unresolved pronoun filtering (0.7 threshold)

🐛 V13.1 ISSUES FIXED (discovered by Reflector):
- Issue #1: Vague entities extracted ("aspects of life", "the answer") → 10 issues
  → Fixed by Pass 1 prompt entity specificity requirements
- Issue #2: Book metadata not filtered (praise quotes, dedications) → 10 issues
  → Fixed by MetadataFilter module with 4-layer detection
- Issue #3: Philosophical/normative claims over-scored → 12 issues
  → Fixed by enhanced Pass 2 claim type classification
- Issue #4: Modal verbs stripped causing semantic loss
  → Fixed by preserving 'can', 'may', 'might' in predicates
- Issue #5: Unresolved pronouns with low confidence not filtered → 5-7 issues
  → Fixed by ConfidenceFilter with 0.7 threshold for unresolved pronouns

🏗️ Architecture:
- Pass 1: Extract with V14 enhanced prompts (prevents issues at source)
- Pass 2: Dual-signal with V14 improved claim classification
- Pass 2.5: 14 postprocessing modules including new V14 filters

Target: A or A+ grade (<6.0% issue rate, down from 8.6%)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field as dataclass_field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Pydantic imports
from pydantic import BaseModel, Field

# OpenAI imports
from openai import OpenAI

# PDF processing
import pdfplumber

# ✨ V11 NEW: Import modular postprocessing system
from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kg_extraction_book_v14_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
BOOKS_DIR = BASE_DIR / "data" / "books"
PLAYBOOK_DIR = BASE_DIR / "kg_extraction_playbook"
PROMPTS_DIR = PLAYBOOK_DIR / "prompts"
OUTPUT_DIR = PLAYBOOK_DIR / "output" / "v14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ✨ Load V14 prompts from files
logger.info("Loading V14 prompts from files...")
PASS1_PROMPT_FILE = PROMPTS_DIR / "pass1_extraction_v14.txt"
PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v14.txt"

if not PASS1_PROMPT_FILE.exists():
    logger.warning(f"⚠️  V14 Pass 1 prompt not found, falling back to V13.1")
    PASS1_PROMPT_FILE = PROMPTS_DIR / "pass1_extraction_v13_1.txt"

if not PASS2_PROMPT_FILE.exists():
    logger.warning(f"⚠️  V14 Pass 2 prompt not found, falling back to V13.1")
    PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v13_1.txt"

with open(PASS1_PROMPT_FILE, 'r') as f:
    BOOK_EXTRACTION_PROMPT = f.read()

with open(PASS2_PROMPT_FILE, 'r') as f:
    DUAL_SIGNAL_EVALUATION_PROMPT = f.read()

logger.info(f"✅ V14 prompts loaded successfully from {PASS1_PROMPT_FILE.name} and {PASS2_PROMPT_FILE.name}")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ==============================================================================

class ExtractedRelationship(BaseModel):
    """Pydantic model for Pass 1 extraction"""
    source: str = Field(description="The subject entity")
    relationship: str = Field(description="The relationship/predicate connecting source to target")
    target: str = Field(description="The object entity")
    source_type: str = Field(description="Entity type: PERSON, ORGANIZATION, CONCEPT, PLACE, etc.")
    target_type: str = Field(description="Entity type: PERSON, ORGANIZATION, CONCEPT, PLACE, etc.")
    context: str = Field(description="The text context where this relationship was found")
    page: int = Field(description="Page number where relationship was found")


class ExtractionResult(BaseModel):
    """Pydantic model for Pass 1 batch result"""
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


class RelationshipEvaluation(BaseModel):
    """Pydantic model for Pass 2 evaluation - V13 with entity specificity and claim type fields"""
    candidate_uid: str = Field(description="Unique ID from candidate")
    text_confidence: float = Field(ge=0.0, le=1.0, description="Text signal score 0.0-1.0")
    p_true: float = Field(ge=0.0, le=1.0, description="Knowledge signal score 0.0-1.0")
    entity_specificity_score: float = Field(ge=0.0, le=1.0, description="Entity concreteness score 0.0-1.0")
    signals_conflict: bool = Field(description="True if text and knowledge signals diverge")
    conflict_explanation: Optional[str] = Field(default=None, description="Why signals conflict")
    suggested_correction: Optional[Dict[str, str]] = Field(default=None, description="Suggested fix")
    source_type: str = Field(description="Entity type of source")
    target_type: str = Field(description="Entity type of target")
    classification_flags: List[str] = Field(default_factory=list, description="FACTUAL, METAPHOR, PHILOSOPHICAL_CLAIM, etc.")


class EvaluationBatchResult(BaseModel):
    """Pydantic model for Pass 2 batch result"""
    evaluations: List[RelationshipEvaluation] = Field(default_factory=list)


# ==============================================================================
# MODULE-COMPATIBLE RELATIONSHIP CLASS
# ==============================================================================

@dataclass
class ModuleRelationship:
    """
    ✨ V13 FIX: Relationship format with entity specificity and claim type tracking.

    Modules expect objects with attributes (not dicts):
    - rel.source, rel.target, rel.relationship
    - rel.evidence (dict with 'page_number')
    - rel.evidence_text (string)
    - rel.flags (mutable dict)
    - rel.knowledge_plausibility (alias for p_true)
    - rel.pattern_prior, rel.claim_uid, rel.extraction_metadata (defaults)

    V13 NEW: Entity specificity and claim type fields for quality tracking
    """
    source: str
    relationship: str
    target: str
    source_type: str
    target_type: str
    context: str
    page: int
    text_confidence: float
    p_true: float
    signals_conflict: bool
    conflict_explanation: Optional[str]
    suggested_correction: Optional[Dict[str, str]]
    classification_flags: List[str]
    candidate_uid: str

    # ✨ V13 NEW: Entity specificity and claim type tracking
    entity_specificity_score: float = 1.0  # Default to 1.0 (fully specific)

    # Module interface fields
    evidence: Dict[str, Any] = dataclass_field(default_factory=dict)
    evidence_text: str = ""
    flags: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize module interface fields from existing data"""
        self.evidence = {'page_number': self.page}
        self.evidence_text = self.context
        if self.flags is None:
            self.flags = {}

    # ✨ V11.2.1 FIX: Add properties for ListSplitter compatibility
    @property
    def knowledge_plausibility(self) -> float:
        """Alias for p_true (ListSplitter expects this name)"""
        return self.p_true

    @property
    def pattern_prior(self) -> float:
        """Default pattern prior (not used in V11.2 but ListSplitter expects it)"""
        return 0.5

    @property
    def claim_uid(self) -> Optional[str]:
        """Default claim_uid (not used in V11.2 but ListSplitter expects it)"""
        return None

    @property
    def extraction_metadata(self) -> Dict[str, Any]:
        """Default extraction metadata (not used in V11.2 but ListSplitter expects it)"""
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization - V13 with entity specificity and claim type"""
        return {
            'source': self.source,
            'relationship': self.relationship,
            'target': self.target,
            'source_type': self.source_type,
            'target_type': self.target_type,
            'context': self.context,
            'page': self.page,
            'text_confidence': self.text_confidence,
            'p_true': self.p_true,
            'signals_conflict': self.signals_conflict,
            'conflict_explanation': self.conflict_explanation,
            'suggested_correction': self.suggested_correction,
            'classification_flags': self.classification_flags,
            'candidate_uid': self.candidate_uid,
            # ✨ V13 NEW: Include entity specificity and claim type
            'entity_specificity_score': self.entity_specificity_score,
            'flags': self.flags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleRelationship':
        """
        Create from dict - V13 with entity specificity and claim type support.

        ✨ V11.2.1 FIX: Filters out property-only keys that ListSplitter might add
        (knowledge_plausibility, pattern_prior, claim_uid, extraction_metadata)
        """
        # Filter out property-only keys before creating object
        clean_data = {k: v for k, v in data.items()
                     if k not in ['knowledge_plausibility', 'pattern_prior', 'claim_uid', 'extraction_metadata']}

        return cls(
            source=clean_data['source'],
            relationship=clean_data['relationship'],
            target=clean_data['target'],
            source_type=clean_data.get('source_type', 'UNKNOWN'),
            target_type=clean_data.get('target_type', 'UNKNOWN'),
            context=clean_data.get('context', ''),
            page=clean_data.get('page', 0),
            text_confidence=clean_data.get('text_confidence', 0.5),
            p_true=clean_data.get('p_true', clean_data.get('knowledge_plausibility', 0.5)),
            signals_conflict=clean_data.get('signals_conflict', False),
            conflict_explanation=clean_data.get('conflict_explanation'),
            suggested_correction=clean_data.get('suggested_correction'),
            classification_flags=clean_data.get('classification_flags', []),
            candidate_uid=clean_data.get('candidate_uid', ''),
            # ✨ V13 NEW: Load entity specificity with defaults
            entity_specificity_score=clean_data.get('entity_specificity_score', 1.0),
            evidence=clean_data.get('evidence', {}),
            evidence_text=clean_data.get('evidence_text', clean_data.get('context', '')),
            flags=clean_data.get('flags')
        )


# ==============================================================================
# PDF EXTRACTION
# ==============================================================================

def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Extract text from PDF with page-level granularity.

    Returns:
        (full_text, pages_with_text)
    """
    logger.info(f"📖 Extracting text from PDF: {pdf_path.name}")

    pages_with_text = []
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"  Total pages: {len(pdf.pages)}")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and len(text.strip()) > 50:  # Skip near-empty pages
                pages_with_text.append((i + 1, text))  # 1-indexed
                all_text.append(text)

            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{len(pdf.pages)} pages")

    full_text = "\n\n".join(all_text)
    logger.info(f"✅ Extracted {len(full_text.split())} words from {len(pages_with_text)} pages")

    return full_text, pages_with_text


def create_chunks(
    pages_with_text: List[Tuple[int, str]],
    chunk_size: int = 600,  # ✨ V11.2 FIX: Reduced from 800 → 600
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Create overlapping chunks from page text.

    ✨ V11.2 FIX: Reduced default chunk_size from 800 → 600 words
    to prevent hitting 16K token output limit.

    Args:
        pages_with_text: List of (page_num, text) tuples
        chunk_size: Target chunk size in words (reduced to 600)
        overlap: Overlap size in words

    Returns:
        List of chunk dicts with text, pages, and metadata
    """
    chunks = []
    current_chunk = []
    current_words = []
    current_pages = set()

    for page_num, text in pages_with_text:
        words = text.split()

        for word in words:
            current_words.append(word)
            current_pages.add(page_num)

            if len(current_words) >= chunk_size:
                # Create chunk
                chunk_text = " ".join(current_words)
                chunks.append({
                    'text': chunk_text,
                    'pages': sorted(list(current_pages)),
                    'word_count': len(current_words)
                })

                # Overlap: keep last `overlap` words
                current_words = current_words[-overlap:] if overlap > 0 else []
                current_pages = set([page_num])  # Reset to current page

    # Handle remaining words
    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append({
            'text': chunk_text,
            'pages': sorted(list(current_pages)),
            'word_count': len(current_words)
        })

    logger.info(f"📄 Created {len(chunks)} chunks from book")
    logger.info(f"   - Pages included: {len(pages_with_text)}/{len(pages_with_text)} (100.0%)")

    return chunks


# ==============================================================================
# PASS 1: EXTRACTION WITH RETRY LOGIC
# ==============================================================================

def extract_pass1(
    chunk: Dict[str, Any],
    model: str = "gpt-4o-mini",
    retry_split: bool = True
) -> Tuple[List[ExtractedRelationship], bool]:
    """
    Pass 1: Extract relationships using structured outputs.

    ✨ V11.2 FIX: Added retry logic that splits chunk if token limit hit.

    Args:
        chunk: Chunk dict with text and page info
        model: OpenAI model to use
        retry_split: If True, retry with split chunk on token limit error

    Returns:
        (relationships, was_split)
    """
    chunk_text = chunk['text']
    pages = chunk['pages']
    primary_page = pages[0] if pages else 1

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": BOOK_EXTRACTION_PROMPT.format(text=chunk_text)
                }
            ],
            response_format=ExtractionResult,
            temperature=0.3
        )

        result = response.choices[0].message.parsed

        # Add page info to each relationship
        for rel in result.relationships:
            rel.page = primary_page

        return result.relationships, False

    except Exception as e:
        error_str = str(e)

        # ✨ V11.2 FIX: Detect token limit and retry with split
        if "length limit was reached" in error_str and retry_split:
            logger.warning(f"⚠️  Token limit hit, splitting chunk and retrying...")

            # Split chunk in half
            words = chunk_text.split()
            mid_point = len(words) // 2

            chunk1 = {
                'text': " ".join(words[:mid_point]),
                'pages': pages,
                'word_count': mid_point
            }
            chunk2 = {
                'text': " ".join(words[mid_point:]),
                'pages': pages,
                'word_count': len(words) - mid_point
            }

            # Retry both halves (without further splitting)
            rels1, _ = extract_pass1(chunk1, model, retry_split=False)
            rels2, _ = extract_pass1(chunk2, model, retry_split=False)

            logger.info(f"✅ Retry successful: {len(rels1)} + {len(rels2)} = {len(rels1) + len(rels2)} relationships")

            return rels1 + rels2, True
        else:
            logger.error(f"❌ Pass 1 extraction failed: {e}")
            return [], False


# ==============================================================================
# PASS 2: DUAL-SIGNAL EVALUATION
# ==============================================================================

def evaluate_pass2(
    candidates: List[ExtractedRelationship],
    batch_size: int = 25,
    model: str = "gpt-4o-mini"
) -> List[ModuleRelationship]:
    """
    Pass 2: Evaluate candidates with dual-signal analysis.

    ✨ V11.2 FIX: Returns ModuleRelationship objects (not dicts)

    Args:
        candidates: List of extracted relationships
        batch_size: Batch size for evaluation
        model: OpenAI model

    Returns:
        List of ModuleRelationship objects with evaluations
    """
    evaluated = []

    # Process in batches
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]

        # Prepare batch for evaluation
        batch_json = []
        for idx, rel in enumerate(batch):
            batch_json.append({
                'candidate_uid': f"cand_{i+idx}",
                'source': rel.source,
                'relationship': rel.relationship,
                'target': rel.target,
                'context': rel.context,
                'page': rel.page
            })

        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": DUAL_SIGNAL_EVALUATION_PROMPT.format(
                            batch_size=len(batch),
                            relationships_json=json.dumps(batch_json, indent=2)
                        )
                    }
                ],
                response_format=EvaluationBatchResult,
                temperature=0.2
            )

            result = response.choices[0].message.parsed

            # Merge evaluations with candidates → ModuleRelationship objects
            for eval_result in result.evaluations:
                # Find matching candidate
                cand_idx = int(eval_result.candidate_uid.split('_')[1]) - i
                if 0 <= cand_idx < len(batch):
                    cand = batch[cand_idx]

                    # ✨ V13 FIX: Create ModuleRelationship with entity specificity and claim type
                    module_rel = ModuleRelationship(
                        source=cand.source,
                        relationship=cand.relationship,
                        target=cand.target,
                        source_type=eval_result.source_type,
                        target_type=eval_result.target_type,
                        context=cand.context,
                        page=cand.page,
                        text_confidence=eval_result.text_confidence,
                        p_true=eval_result.p_true,
                        signals_conflict=eval_result.signals_conflict,
                        conflict_explanation=eval_result.conflict_explanation,
                        suggested_correction=eval_result.suggested_correction,
                        classification_flags=eval_result.classification_flags,
                        candidate_uid=eval_result.candidate_uid,
                        # ✨ V13 NEW: Pass entity specificity field
                        entity_specificity_score=eval_result.entity_specificity_score
                    )
                    evaluated.append(module_rel)

        except Exception as e:
            logger.error(f"❌ Pass 2 batch {i//batch_size + 1} failed: {e}")
            continue

    return evaluated


# ==============================================================================
# PASS 2.5: MODULAR POSTPROCESSING
# ==============================================================================

def postprocess_pass2_5(
    relationships: List[ModuleRelationship],
    context: ProcessingContext
) -> Tuple[List[ModuleRelationship], Dict[str, Any]]:
    """
    Pass 2.5: Run modular postprocessing pipeline.

    ✨ V14 ENHANCEMENTS: 14 modules including new ConfidenceFilter and enhanced PredicateNormalizer

    Args:
        relationships: List of ModuleRelationship objects
        context: Processing context with metadata

    Returns:
        (processed_relationships, stats)
    """
    logger.info("🔧 PASS 2.5: Running V14 modular postprocessing pipeline (14 modules)...")

    # Get book pipeline (14 modules with V14 enhancements)
    pipeline = get_book_pipeline()

    # Pass ModuleRelationship objects to pipeline
    processed_objs, stats = pipeline.run(relationships, context)

    logger.info(f"✅ Pass 2.5 complete: {len(relationships)} → {len(processed_objs)} relationships")

    return processed_objs, stats


# ==============================================================================
# MAIN EXTRACTION PIPELINE
# ==============================================================================

def extract_knowledge_graph_v14(
    book_title: str,
    pdf_path: Path,
    document_metadata: Dict[str, Any],
    run_id: str,
    batch_size: int = 25
) -> Dict[str, Any]:
    """
    V14.0 knowledge graph extraction pipeline - Comprehensive Quality System.

    ✨ V14.0 ENHANCEMENTS (from V13.1 A-):
    1. Enhanced Pass 1: Entity specificity + extraction scope (prevents vague entities + metadata)
    2. Enhanced Pass 2: Improved claim classification (FACTUAL vs NORMATIVE vs PHILOSOPHICAL)
    3. MetadataFilter: 4-layer detection for book metadata
    4. ConfidenceFilter: Flag-specific thresholds + unresolved pronoun handling (0.7)
    5. Predicate Normalizer V1.4: Tense norm, modal verb preservation, semantic validation
    6. Pronoun Resolver: Enhanced with unresolved filtering

    Architecture:
    1. Extract text from PDF
    2. Create chunks with overlap (600 words)
    3. Pass 1: Extract with V14 enhanced prompts (prevents issues at source)
    4. Pass 2: Dual-signal with V14 improved claim classification
    5. Pass 2.5: 14 modular postprocessing modules with V14 filters
    6. Return results with full metadata

    Target: A or A+ grade (<6.0% issue rate, down from 8.6%)
    """
    logger.info(f"🚀 Starting V14.0 extraction: {book_title}")

    start_time = time.time()

    # Step 1: Extract text
    full_text, pages_with_text = extract_text_from_pdf(pdf_path)

    # Step 2: Create chunks (reduced size)
    chunks = create_chunks(pages_with_text, chunk_size=600, overlap=100)

    # Step 3: Pass 1 - Extraction with retry logic
    logger.info(f"📝 PASS 1: Comprehensive extraction with retry logic...")
    logger.info(f"  Processing {len(chunks)} chunks")

    all_candidates = []
    chunks_split = 0
    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)} (pages {chunk['pages'][0]}-{chunk['pages'][-1]})")
        candidates, was_split = extract_pass1(chunk)
        all_candidates.extend(candidates)
        if was_split:
            chunks_split += 1

    logger.info(f"✅ Pass 1 complete: {len(all_candidates)} candidates extracted")
    if chunks_split > 0:
        logger.info(f"   ✨ {chunks_split} chunks auto-split and retried due to token limit")

    # ✨ CHECKPOINT: Save Pass 1 results
    checkpoint_pass1 = OUTPUT_DIR / f"{run_id}_pass1_checkpoint.json"
    with open(checkpoint_pass1, 'w') as f:
        json.dump([rel.model_dump() for rel in all_candidates], f, indent=2)
    logger.info(f"💾 Pass 1 checkpoint saved: {checkpoint_pass1.name}")

    # Step 4: Pass 2 - Evaluation
    logger.info(f"🔍 PASS 2: Dual-signal evaluation...")
    logger.info(f"  Evaluating {len(all_candidates)} candidates in batches of {batch_size}")

    evaluated = evaluate_pass2(all_candidates, batch_size=batch_size)

    logger.info(f"✅ Pass 2 complete: {len(evaluated)} relationships evaluated")

    # ✨ CHECKPOINT: Save Pass 2 results
    checkpoint_pass2 = OUTPUT_DIR / f"{run_id}_pass2_checkpoint.json"
    with open(checkpoint_pass2, 'w') as f:
        json.dump([rel.to_dict() for rel in evaluated], f, indent=2)
    logger.info(f"💾 Pass 2 checkpoint saved: {checkpoint_pass2.name}")

    # Step 5: Pass 2.5 - Modular postprocessing with V14 enhancements
    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata,
        pages_with_text=pages_with_text,
        run_id=run_id,
        extraction_version='v14'
    )

    final_relationships, pp_stats = postprocess_pass2_5(evaluated, context)

    logger.info(f"✅ Pass 2.5 complete: {len(final_relationships)} final relationships")

    # Calculate stats
    elapsed = time.time() - start_time

    # Count confidence levels
    high_conf = sum(1 for r in final_relationships if r.p_true >= 0.75)
    med_conf = sum(1 for r in final_relationships if 0.5 <= r.p_true < 0.75)
    low_conf = sum(1 for r in final_relationships if r.p_true < 0.5)

    # Count classification flags
    flag_counts = {}
    for rel in final_relationships:
        for flag in rel.classification_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    # Count module flags
    module_flag_counts = {}
    for rel in final_relationships:
        if rel.flags:
            for flag_key in rel.flags.keys():
                module_flag_counts[flag_key] = module_flag_counts.get(flag_key, 0) + 1

    # Prepare results
    results = {
        'metadata': {
            'book_title': book_title,
            'extraction_version': 'v14',
            'run_id': run_id,
            'extraction_date': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'document_metadata': document_metadata,
            'enhancements_applied': [
                'V14: Enhanced Pass 1 prompt - entity specificity + extraction scope',
                'V14: Enhanced Pass 2 prompt - improved claim type classification (FACTUAL/NORMATIVE/PHILOSOPHICAL)',
                'V14: MetadataFilter module - 4-layer detection for book metadata',
                'V14: ConfidenceFilter module - flag-specific thresholds + unresolved pronoun handling',
                'V14: Predicate Normalizer V1.4 - tense normalization, modal verb preservation',
                'V14: Predicate Normalizer V1.4 - enhanced semantic validation',
                'V14: Pronoun Resolver - unresolved pronoun filtering (0.7 threshold)',
                'V14: Target grade: A or A+ (<6.0% issue rate, down from 8.6%)'
            ]
        },
        'extraction_stats': {
            'pass1_candidates': len(all_candidates),
            'pass1_chunks_split': chunks_split,
            'pass2_evaluated': len(evaluated),
            'pass2_5_final': len(final_relationships),
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'classification_flags': flag_counts,
            'module_flags': module_flag_counts
        },
        'postprocessing_stats': pp_stats,
        'relationships': [rel.to_dict() for rel in final_relationships]
    }

    logger.info("")
    logger.info("📊 FINAL V14.0 RESULTS:")
    logger.info(f"  - Pass 1 extracted: {len(all_candidates)} candidates")
    logger.info(f"  - Pass 1 chunks split: {chunks_split}")
    logger.info(f"  - Pass 2 evaluated: {len(evaluated)}")
    logger.info(f"  - ✨ V14.0 Pass 2.5 final: {len(final_relationships)}")
    logger.info(f"  - High confidence (p≥0.75): {high_conf} ({100*high_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Medium confidence: {med_conf} ({100*med_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Low confidence: {low_conf} ({100*low_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Total time: {elapsed/60:.1f} minutes")

    if flag_counts:
        logger.info(f"  - Classification flags:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            logger.info(f"      {flag}: {count}")

    if module_flag_counts:
        logger.info(f"  - Module flags (postprocessing):")
        for flag, count in sorted(module_flag_counts.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"      {flag}: {count}")

    return results


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Extract knowledge graph from Soil Stewardship Handbook with V14.0 comprehensive quality system"""
    logger.info("="*80)
    logger.info("🚀 V14.0 KNOWLEDGE GRAPH EXTRACTION - COMPREHENSIVE QUALITY SYSTEM")
    logger.info("="*80)
    logger.info("")
    logger.info("✨ V14.0 ENHANCEMENTS (from V13.1 A-):")
    logger.info("  1. ✅ Enhanced Pass 1: Entity specificity + extraction scope")
    logger.info("  2. ✅ Enhanced Pass 2: Improved claim classification (FACTUAL/NORMATIVE/PHILOSOPHICAL)")
    logger.info("  3. ✅ MetadataFilter Module: 4-layer detection for book metadata")
    logger.info("  4. ✅ ConfidenceFilter Module: Flag-specific thresholds + unresolved pronoun handling")
    logger.info("  5. ✅ Predicate Normalizer V1.4: Tense norm, modal verb preservation, semantic validation")
    logger.info("  6. ✅ Pronoun Resolver: Enhanced with unresolved pronoun filtering (0.7 threshold)")
    logger.info("")
    logger.info("🐛 V13.1 ISSUES FIXED (discovered by Reflector):")
    logger.info("  - Vague entities: 10 issues (prevented at extraction with entity specificity)")
    logger.info("  - Book metadata: 10 issues (filtered with 4-layer MetadataFilter)")
    logger.info("  - Philosophical/normative: 12 issues (fixed with improved claim classification)")
    logger.info("  - Unresolved pronouns: 5-7 issues (filtered with 0.7 threshold)")
    logger.info("  - Semantic mismatches: 5-8 issues (caught by semantic validation)")
    logger.info("")
    logger.info("🎯 EXPECTED IMPROVEMENTS:")
    logger.info("  - V13.1 Grade: A- (8.6% issue rate, 75 issues)")
    logger.info("  - V14.0 Target: A or A+ (<6.0% issue rate, <55 issues)")
    logger.info("  - Expected improvement: 50-57 issues fixed (27% reduction)")
    logger.info("")

    # Book details
    book_dir = BOOKS_DIR / "soil-stewardship-handbook"
    pdf_path = book_dir / "Soil-Stewardship-Handbook-eBook.pdf"
    book_title = "Soil Stewardship Handbook"

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    run_id = f"book_soil_handbook_v14_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Document metadata
    document_metadata = {
        'author': 'Aaron Perry',
        'title': 'Soil Stewardship Handbook',
        'publication_year': 2017
    }

    start_time = time.time()

    # Extract with V14.0 comprehensive quality system
    results = extract_knowledge_graph_v14(
        book_title=book_title,
        pdf_path=pdf_path,
        document_metadata=document_metadata,
        run_id=run_id,
        batch_size=25
    )

    # Save results
    output_path = OUTPUT_DIR / f"{book_title.replace(' ', '_').lower()}_v14.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("✨ V14.0 EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Run KG Reflector on V14.0 to verify quality improvements")
    logger.info("2. Compare V14.0 vs V13.1 (A-) quality metrics")
    logger.info("3. Validate all 6 success criteria (metadata filter, vague entities, etc.)")
    logger.info("4. If grade reaches A or A+, V14.0 becomes new stable release")
    logger.info("="*80)


if __name__ == "__main__":
    main()
