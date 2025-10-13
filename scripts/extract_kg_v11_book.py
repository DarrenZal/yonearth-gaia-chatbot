#!/usr/bin/env python3
"""
Knowledge Graph Extraction V11 - Clean Modular Architecture

🎯 V11 IMPROVEMENTS:
- Uses production-ready modular postprocessing system from src/knowledge_graph/postprocessing/
- Corrected prompts: "extract and LABEL" not "block"
- Clean separation: extraction logic vs postprocessing modules
- Properly integrates book_pipeline with all 10 modules

🏗️ Architecture:
- Pass 1: Extract relationships using OpenAI structured outputs
- Pass 2: Dual-signal evaluation (text confidence + knowledge plausibility)
- Pass 2.5: Modular postprocessing pipeline with proper book modules
  - Modules handle: pronouns, metaphors, philosophical claims, dedications, etc.
  - Modules LABEL issues, not block them

Based on V10 but completely refactored to use proper module system.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

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
        logging.FileHandler(f'kg_extraction_book_v11_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
BOOKS_DIR = BASE_DIR / "data" / "books"
PLAYBOOK_DIR = BASE_DIR / "kg_extraction_playbook"
PROMPTS_DIR = PLAYBOOK_DIR / "prompts"
OUTPUT_DIR = PLAYBOOK_DIR / "output" / "v11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ✨ Load V11 prompts from files
logger.info("Loading V11 prompts from files...")
PASS1_PROMPT_FILE = PROMPTS_DIR / "pass1_extraction_v11.txt"
PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v11.txt"

if not PASS1_PROMPT_FILE.exists():
    logger.warning(f"⚠️  V11 Pass 1 prompt not found, falling back to V10")
    PASS1_PROMPT_FILE = PROMPTS_DIR / "pass1_extraction_v10.txt"

if not PASS2_PROMPT_FILE.exists():
    logger.warning(f"⚠️  V11 Pass 2 prompt not found, falling back to V10")
    PASS2_PROMPT_FILE = PROMPTS_DIR / "pass2_evaluation_v10.txt"

with open(PASS1_PROMPT_FILE, 'r') as f:
    BOOK_EXTRACTION_PROMPT = f.read()

with open(PASS2_PROMPT_FILE, 'r') as f:
    DUAL_SIGNAL_EVALUATION_PROMPT = f.read()

logger.info("✅ V11 prompts loaded successfully")

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
    """Pydantic model for Pass 2 evaluation"""
    candidate_uid: str = Field(description="Unique ID from candidate")
    text_confidence: float = Field(ge=0.0, le=1.0, description="Text signal score 0.0-1.0")
    p_true: float = Field(ge=0.0, le=1.0, description="Knowledge signal score 0.0-1.0")
    signals_conflict: bool = Field(description="True if text and knowledge signals diverge")
    conflict_explanation: Optional[str] = Field(default=None, description="Why signals conflict")
    suggested_correction: Optional[Dict[str, str]] = Field(default=None, description="Suggested fix")
    source_type: str = Field(description="Entity type of source")
    target_type: str = Field(description="Entity type of target")
    classification_flags: List[str] = Field(default_factory=list, description="FACTUAL, METAPHOR, PHILOSOPHICAL_CLAIM, etc.")


class EvaluationBatchResult(BaseModel):
    """Pydantic model for Pass 2 batch result"""
    evaluations: List[RelationshipEvaluation] = Field(default_factory=list)


@dataclass
class V11Relationship:
    """Relationship with full metadata for V11 system"""
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

    def to_dict(self) -> Dict[str, Any]:
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
            'candidate_uid': self.candidate_uid
        }


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

    full_text = "\\n\\n".join(all_text)
    logger.info(f"✅ Extracted {len(full_text.split())} words from {len(pages_with_text)} pages")

    return full_text, pages_with_text


def create_chunks(
    pages_with_text: List[Tuple[int, str]],
    chunk_size: int = 800,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Create overlapping chunks from page text.

    Args:
        pages_with_text: List of (page_num, text) tuples
        chunk_size: Target chunk size in words
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
# PASS 1: EXTRACTION
# ==============================================================================

def extract_pass1(chunk: Dict[str, Any], model: str = "gpt-4o-mini") -> List[ExtractedRelationship]:
    """
    Pass 1: Extract relationships using structured outputs.

    Args:
        chunk: Chunk dict with text and page info
        model: OpenAI model to use

    Returns:
        List of extracted relationships
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

        return result.relationships

    except Exception as e:
        logger.error(f"❌ Pass 1 extraction failed: {e}")
        return []


# ==============================================================================
# PASS 2: DUAL-SIGNAL EVALUATION
# ==============================================================================

def evaluate_pass2(
    candidates: List[ExtractedRelationship],
    batch_size: int = 25,
    model: str = "gpt-4o-mini"
) -> List[V11Relationship]:
    """
    Pass 2: Evaluate candidates with dual-signal analysis.

    Args:
        candidates: List of extracted relationships
        batch_size: Batch size for evaluation
        model: OpenAI model

    Returns:
        List of V11Relationship objects with evaluations
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

            # Merge evaluations with candidates
            for eval_result in result.evaluations:
                # Find matching candidate
                cand_idx = int(eval_result.candidate_uid.split('_')[1]) - i
                if 0 <= cand_idx < len(batch):
                    cand = batch[cand_idx]

                    v11_rel = V11Relationship(
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
                        candidate_uid=eval_result.candidate_uid
                    )
                    evaluated.append(v11_rel)

        except Exception as e:
            logger.error(f"❌ Pass 2 batch {i//batch_size + 1} failed: {e}")
            continue

    return evaluated


# ==============================================================================
# PASS 2.5: MODULAR POSTPROCESSING
# ==============================================================================

def postprocess_pass2_5(
    relationships: List[V11Relationship],
    context: ProcessingContext
) -> Tuple[List[V11Relationship], Dict[str, Any]]:
    """
    Pass 2.5: Run modular postprocessing pipeline.

    ✨ V11 NEW: Uses production-ready modules from src/knowledge_graph/postprocessing/

    Args:
        relationships: List of evaluated relationships
        context: Processing context with metadata

    Returns:
        (processed_relationships, stats)
    """
    logger.info("🔧 PASS 2.5: Running modular postprocessing pipeline...")

    # Get book pipeline (all 10 modules)
    pipeline = get_book_pipeline()

    # Convert V11Relationship to dict format expected by modules
    rel_dicts = [rel.to_dict() for rel in relationships]

    # Run pipeline
    processed_dicts, stats = pipeline.run(rel_dicts, context)

    # Convert back to V11Relationship objects
    processed = []
    for rel_dict in processed_dicts:
        processed.append(V11Relationship(
            source=rel_dict['source'],
            relationship=rel_dict['relationship'],
            target=rel_dict['target'],
            source_type=rel_dict.get('source_type', 'UNKNOWN'),
            target_type=rel_dict.get('target_type', 'UNKNOWN'),
            context=rel_dict.get('context', ''),
            page=rel_dict.get('page', 0),
            text_confidence=rel_dict.get('text_confidence', 0.5),
            p_true=rel_dict.get('p_true', 0.5),
            signals_conflict=rel_dict.get('signals_conflict', False),
            conflict_explanation=rel_dict.get('conflict_explanation'),
            suggested_correction=rel_dict.get('suggested_correction'),
            classification_flags=rel_dict.get('classification_flags', []),
            candidate_uid=rel_dict.get('candidate_uid', '')
        ))

    logger.info(f"✅ Pass 2.5 complete: {len(relationships)} → {len(processed)} relationships")

    return processed, stats


# ==============================================================================
# MAIN EXTRACTION PIPELINE
# ==============================================================================

def extract_knowledge_graph_v11(
    book_title: str,
    pdf_path: Path,
    document_metadata: Dict[str, Any],
    run_id: str,
    batch_size: int = 25
) -> Dict[str, Any]:
    """
    V11 knowledge graph extraction pipeline.

    Architecture:
    1. Extract text from PDF
    2. Create chunks with overlap
    3. Pass 1: Extract relationships (structured outputs)
    4. Pass 2: Dual-signal evaluation (structured outputs)
    5. Pass 2.5: Modular postprocessing (production modules)
    6. Return results with full metadata
    """
    logger.info(f"🚀 Starting V11 extraction: {book_title}")

    start_time = time.time()

    # Step 1: Extract text
    full_text, pages_with_text = extract_text_from_pdf(pdf_path)

    # Step 2: Create chunks
    chunks = create_chunks(pages_with_text, chunk_size=800, overlap=100)

    # Step 3: Pass 1 - Extraction
    logger.info(f"📝 PASS 1: Comprehensive extraction...")
    logger.info(f"  Processing {len(chunks)} chunks")

    all_candidates = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i}/{len(chunks)} (pages {chunk['pages'][0]}-{chunk['pages'][-1]})")
        candidates = extract_pass1(chunk)
        all_candidates.extend(candidates)

    logger.info(f"✅ Pass 1 complete: {len(all_candidates)} candidates extracted")

    # Step 4: Pass 2 - Evaluation
    logger.info(f"🔍 PASS 2: Dual-signal evaluation...")
    logger.info(f"  Evaluating {len(all_candidates)} candidates in batches of {batch_size}")

    evaluated = evaluate_pass2(all_candidates, batch_size=batch_size)

    logger.info(f"✅ Pass 2 complete: {len(evaluated)} relationships evaluated")

    # Step 5: Pass 2.5 - Modular postprocessing
    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata,
        pages_with_text=pages_with_text,
        run_id=run_id,
        extraction_version='v11'
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

    # Prepare results
    results = {
        'metadata': {
            'book_title': book_title,
            'extraction_version': 'v11',
            'run_id': run_id,
            'extraction_date': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'document_metadata': document_metadata
        },
        'extraction_stats': {
            'pass1_candidates': len(all_candidates),
            'pass2_evaluated': len(evaluated),
            'pass2_5_final': len(final_relationships),
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'classification_flags': flag_counts
        },
        'postprocessing_stats': pp_stats,
        'relationships': [rel.to_dict() for rel in final_relationships]
    }

    logger.info("")
    logger.info("📊 FINAL V11 RESULTS:")
    logger.info(f"  - Pass 1 extracted: {len(all_candidates)} candidates")
    logger.info(f"  - Pass 2 evaluated: {len(evaluated)}")
    logger.info(f"  - ✨ V11 Pass 2.5 final (modular): {len(final_relationships)}")
    logger.info(f"  - High confidence (p≥0.75): {high_conf} ({100*high_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Medium confidence: {med_conf} ({100*med_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Low confidence: {low_conf} ({100*low_conf/len(final_relationships) if final_relationships else 0:.1f}%)")
    logger.info(f"  - Total time: {elapsed/60:.1f} minutes")

    if flag_counts:
        logger.info(f"  - Classification flags:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            logger.info(f"      {flag}: {count}")

    return results


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Extract knowledge graph from Soil Stewardship Handbook with V11 modular system"""
    logger.info("="*80)
    logger.info("🚀 V11 KNOWLEDGE GRAPH EXTRACTION - CLEAN MODULAR ARCHITECTURE")
    logger.info("="*80)
    logger.info("")
    logger.info("✨ V11 NEW FEATURES:")
    logger.info("  ✅ Uses production-ready modular postprocessing system")
    logger.info("  ✅ All 10 modules: PraiseQuoteDetector, PronounResolver, ListSplitter, etc.")
    logger.info("  ✅ Modules LABEL issues (metaphors, philosophical claims) - don't block them")
    logger.info("  ✅ Proper integration with book_pipeline")
    logger.info("  ✅ Clean separation: extraction vs postprocessing")
    logger.info("")
    logger.info("GOAL: Fix V10 issues using proper modular architecture")
    logger.info("Expected: Properly labeled metaphors, fixed dedications, resolved pronouns")
    logger.info("")

    # Book details
    book_dir = BOOKS_DIR / "soil-stewardship-handbook"
    pdf_path = book_dir / "Soil-Stewardship-Handbook-eBook.pdf"
    book_title = "Soil Stewardship Handbook"

    if not pdf_path.exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    run_id = f"book_soil_handbook_v11_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Document metadata
    document_metadata = {
        'author': 'Aaron Perry',
        'title': 'Soil Stewardship Handbook',
        'publication_year': 2017
    }

    start_time = time.time()

    # Extract with V11 system
    results = extract_knowledge_graph_v11(
        book_title=book_title,
        pdf_path=pdf_path,
        document_metadata=document_metadata,
        run_id=run_id,
        batch_size=25
    )

    # Save results
    output_path = OUTPUT_DIR / f"{book_title.replace(' ', '_').lower()}_v11.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    logger.info("")
    logger.info("="*80)
    logger.info("✨ V11 EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Run KG Reflector on V11 to measure improvements")
    logger.info("2. Compare V11 vs V10 quality metrics")
    logger.info("3. Validate that modules properly labeled issues")
    logger.info("4. If successful, V11 becomes baseline for next ACE cycle")
    logger.info("="*80)


if __name__ == "__main__":
    main()
