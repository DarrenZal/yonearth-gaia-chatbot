#!/usr/bin/env python3
"""
Knowledge Graph Extraction V14.3.1 - Prompt Enhancement (Phase 1.5)

🎯 V14.3.1 STRATEGY: REDUCE SUBJECTIVE CONTENT EXTRACTION

**Goal**: Achieve A- grade (5-8% issue rate) through enhanced Pass 1 prompt and filtering.

**Configuration**:
- ✅ V14.3.1 Pass 1 Prompt: FACTUAL-ONLY extraction (no aspirational/philosophical/metaphorical)
- ✅ V13.1 Pass 2 Prompt: Dual-signal evaluation (same as V14.3)
- ✅ V14.3.1 Pass 2.5: 13-module pipeline (V13 + SubjectiveContentFilter)

**Why V14.3.1**:
- V14.3 got B grade (16.2%) instead of expected A- due to LLM variance
- Fresh extraction produced 14% fewer candidates (738 vs V13.1's 861 checkpoint)
- V14.3 extracted too much philosophical/aspirational content (67 issues, 9.3%)

**Three High-Impact Enhancements**:

1. **Enhanced Pass 1 Prompt** (-11.4% expected impact):
   - Constraint 1: Extract only FACTUAL relationships (no aspirational "we can/should")
   - Constraint 2: Entities must be SPECIFIC ("living soil" not "soil")
   - Constraint 3: Resolve PRONOUNS before extraction ("humanity" not "we")

2. **SubjectiveContentFilter Module** (-9.8% expected impact):
   - Filters p_true < 0.5 AND (PHILOSOPHICAL_CLAIM OR FIGURATIVE_LANGUAGE)
   - Filters all OPINION and SEMANTIC_INCOMPATIBILITY flags
   - Priority 15 (early filtering)

3. **Enhanced PronounResolver** (-1.7% expected impact):
   - Fallback rules: 'we' → 'humanity', 'us' → 'humanity', 'our' → 'human'
   - Aspirational pattern detection (V14.3.1 NEW)

**Expected Result**: 16.2% → 5-8% issue rate (A- to A grade)

🏗️ Architecture:
- Pass 1: V14.3.1 prompt (factual-only, specific entities, resolved pronouns)
- Pass 2: V13.1 prompt (dual-signal evaluation)
- Pass 2.5: 13-module pipeline (V13's 12 + SubjectiveContentFilter)
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
        logging.FileHandler(f'kg_extraction_book_v14_3_1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
BOOKS_DIR = BASE_DIR / "data" / "books"
PLAYBOOK_DIR = BASE_DIR / "kg_extraction_playbook"
PROMPTS_DIR = PLAYBOOK_DIR / "prompts"
OUTPUT_DIR = PLAYBOOK_DIR / "output" / "v14_3_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ✨ Load V14.3.1 prompts from files
logger.info("Loading V14.3.1 prompts (enhanced factual-only extraction)...")
PASS1_PROMPT_FILE = PROMPTS_DIR / "pass1_extraction_v14_3_1.txt"  # ✅ V14.3.1 Pass 1: factual-only
PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v13_1.txt"  # ✅ V13.1 Pass 2: dual-signal

if not PASS1_PROMPT_FILE.exists():
    raise FileNotFoundError(f"V14.3.1 Pass 1 prompt not found: {PASS1_PROMPT_FILE}")

if not PASS2_PROMPT_FILE.exists():
    raise FileNotFoundError(f"V13.1 Pass 2 prompt not found: {PASS2_PROMPT_FILE}")

with open(PASS1_PROMPT_FILE, 'r') as f:
    BOOK_EXTRACTION_PROMPT = f.read()

with open(PASS2_PROMPT_FILE, 'r') as f:
    DUAL_SIGNAL_EVALUATION_PROMPT = f.read()

logger.info(f"✅ V14.3.1 prompts loaded successfully from {PASS1_PROMPT_FILE.name} and {PASS2_PROMPT_FILE.name}")

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

    ✨ V14.3.1: Uses V14.3.1's 13-module pipeline (version='v14_3_1')

    Args:
        relationships: List of ModuleRelationship objects
        context: Processing context with metadata

    Returns:
        (processed_relationships, stats)
    """
    logger.info("🔧 PASS 2.5: Running V14.3.1 modular postprocessing pipeline (13 modules)...")

    # ✨ V14.3.1: Explicitly use V14.3.1's 13-module pipeline
    pipeline = get_book_pipeline(version='v14_3_1')

    # Pass objects directly (not dicts) - modules expect objects with attributes
    processed_objs, stats = pipeline.run(relationships, context)

    logger.info(f"✅ Pass 2.5 complete: {len(relationships)} → {len(processed_objs)} relationships")

    return processed_objs, stats


# ==============================================================================
# MAIN EXTRACTION PIPELINE
# ==============================================================================

def extract_knowledge_graph_v14_3_1(
    book_title: str,
    pdf_path: Path,
    document_metadata: Dict[str, Any],
    run_id: str,
    batch_size: int = 25
) -> Dict[str, Any]:
    """
    V14.3 knowledge graph extraction - Full V13.1 baseline restoration.

    ✨ V14.3 CONFIGURATION (EXACT V13.1 REPLICA):
    1. V12 Pass 1: Comprehensive extraction, simple predicates
    2. V13.1 Pass 2: Dual-signal evaluation
    3. V13.1 Pass 2.5: 12-module pipeline (proven A- grade)

    Architecture:
    1. Extract text from PDF
    2. Create chunks with overlap (600 words)
    3. Pass 1: Extract using V12 prompt (simple predicates)
    4. Pass 2: Evaluate using V13.1 prompt (dual-signal)
    5. Pass 2.5: Postprocess using V13.1's 12 modules
    6. Return results with full metadata
    """
    logger.info(f"🚀 Starting V14.3 extraction: {book_title}")

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

    # Step 5: Pass 2.5 - Modular postprocessing with V13.1's 12 modules
    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata,
        pages_with_text=pages_with_text,
        run_id=run_id,
        extraction_version='v14_3_1'
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
            'extraction_version': 'v14_3_1',
            'run_id': run_id,
            'extraction_date': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'document_metadata': document_metadata,
            'configuration': {
                'pass1_prompt': 'pass1_extraction_v14_3_1.txt',
                'pass2_prompt': 'pass2_evaluation_v13_1.txt',
                'pass2_5_pipeline': 'v14_3_1 (13 modules: V13 + SubjectiveContentFilter)',
                'strategy': 'Factual-only extraction with prompt enhancement'
            },
            'fixes_applied': [
                'V14.3: EXACT V13.1 configuration (A- baseline)',
                'V12 Pass 1: Comprehensive extraction with simple predicates',
                'V13.1 Pass 2: Dual-signal evaluation',
                'V13.1 Pass 2.5: 12-module pipeline (proven A- grade)',
                'Establishes stable baseline for incremental improvements to A+'
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
    logger.info("📊 FINAL V14.3 RESULTS:")
    logger.info(f"  - Pass 1 extracted: {len(all_candidates)} candidates")
    logger.info(f"  - Pass 1 chunks split: {chunks_split}")
    logger.info(f"  - Pass 2 evaluated: {len(evaluated)}")
    logger.info(f"  - ✨ V14.3 Pass 2.5 final: {len(final_relationships)}")
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
    """Extract knowledge graph from Soil Stewardship Handbook using V14.3 (V13.1 baseline)"""
    logger.info("="*80)
    logger.info("🚀 V14.3 KNOWLEDGE GRAPH EXTRACTION - V13.1 BASELINE RESTORATION")
    logger.info("="*80)
    logger.info("")
    logger.info("✨ V14.3 CONFIGURATION (EXACT V13.1 REPLICA):")
    logger.info("  1. ✅ V12 Pass 1: Comprehensive extraction, simple predicates")
    logger.info("  2. ✅ V13.1 Pass 2: Dual-signal evaluation")
    logger.info("  3. ✅ V13.1 Pass 2.5: 12-module pipeline (proven A- grade)")
    logger.info("")
    logger.info("🔍 ROOT CAUSE OF V14.0-V14.2 FAILURES:")
    logger.info("  - V14.0: B+ (10.78%) - V14 prompt + 14 modules")
    logger.info("  - V14.1: C+ (16.7%) - V12 prompt extracts low-quality content")
    logger.info("  - V14.2: C+ (27.3%) - V14 prompt + V13.1 modules = 100 predicate issues!")
    logger.info("  - Root Cause: V14 prompt extracts verbose predicates V13.1 can't normalize")
    logger.info("")
    logger.info("🎯 EXPECTED V14.3 RESULTS:")
    logger.info("  - V13.1 Baseline: A- grade (8.6% issue rate, 873 relationships)")
    logger.info("  - V14.3 Target: A- grade (match V13.1 exactly)")
    logger.info("  - Foundation for incremental improvements to A+ in V14.4-V15")
    logger.info("")

    # Book details
    book_dir = BOOKS_DIR / "soil-stewardship-handbook"
    pdf_path = book_dir / "Soil-Stewardship-Handbook-eBook.pdf"
    book_title = "Soil Stewardship Handbook"

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    run_id = f"book_soil_handbook_v14_3_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Document metadata
    document_metadata = {
        'author': 'Aaron Perry',
        'title': 'Soil Stewardship Handbook',
        'publication_year': 2017
    }

    start_time = time.time()

    # Extract with V14.3 (V13.1 baseline)
    results = extract_knowledge_graph_v14_3_1(
        book_title=book_title,
        pdf_path=pdf_path,
        document_metadata=document_metadata,
        run_id=run_id,
        batch_size=25
    )

    # Save results
    output_path = OUTPUT_DIR / f"{book_title.replace(' ', '_').lower()}_v14_3_1.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("✨ V14.3 EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Run KG Reflector on V14.3 to verify A- grade restoration")
    logger.info("2. Compare V14.3 vs V13.1 (should match A- baseline)")
    logger.info("3. If A- grade achieved, proceed to Phase 2 (V14.4 PredicateNormalizer)")
    logger.info("4. Incremental path to A+: V14.4 → V14.5 → V14.6 → V15")
    logger.info("="*80)


if __name__ == "__main__":
    main()
