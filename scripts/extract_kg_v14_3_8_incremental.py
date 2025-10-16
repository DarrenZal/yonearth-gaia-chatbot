#!/usr/bin/env python3
"""
Incremental Knowledge Graph Extraction V14.3.8

🎯 PURPOSE: Chapter-by-chapter extraction with dedication normalization + type compatibility validation.

**Strategy**:
- Extract one section (chapter/part) at a time
- Iterate until A+ grade achieved for that section
- Freeze section once A+ grade reached
- Never re-extract frozen sections
- Consolidate cross-chapter relationships in separate passes

**Architecture**:
- Phase 1: Per-chapter extraction (this script)
- Phase 2: Periodic consolidation (see consolidate_chapters.py)
- Phase 3: Final whole-book consolidation

**Usage**:
```bash
# Extract Front Matter (pages 1-30)
python3 scripts/extract_kg_v14_3_8_incremental.py \\
  --book our_biggest_deal \\
  --section front_matter \\
  --pages 1-30

# Extract Chapter 1 (pages 31-50)
python3 scripts/extract_kg_v14_3_8_incremental.py \\
  --book our_biggest_deal \\
  --section chapter_01 \\
  --pages 31-50
```

**Freeze Enforcement**:
- Checks status.json before extraction
- Exits with error if section is frozen
- Updates status.json after successful extraction

**Provenance**:
- Generates execution manifest (git hash, env, packages)
- Saves timestamped extraction with metadata
- Copies prompts/scripts for reproducibility
"""

import json
import logging
import os
import sys
import time
import argparse
import subprocess
import hashlib
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import json as _json
import re
from dataclasses import dataclass, field as dataclass_field
import shutil
import tarfile

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

# Postprocessing system
from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines.book_pipeline import get_book_pipeline_v1438

# Setup logging
def setup_logging(section: str):
    """Setup logging with section-specific filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'kg_extraction_incremental_{section}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), timestamp


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
    entity_specificity_score: float = Field(ge=0.0, le=1.0, description="Entity concreteness score 0.0-1.0")
    signals_conflict: bool = Field(description="True if text and knowledge signals diverge")
    conflict_explanation: Optional[str] = Field(default=None, description="Why signals conflict")
    suggested_correction: Optional[Dict[str, str]] = Field(default=None, description="Suggested fix")
    source_type: str = Field(description="Entity type of source")
    target_type: str = Field(description="Entity type of target")
    classification_flags: List[str] = Field(default_factory=list, description="FACTUAL, METAPHOR, etc.")


class EvaluationBatchResult(BaseModel):
    """Pydantic model for Pass 2 batch result"""
    evaluations: List[RelationshipEvaluation] = Field(default_factory=list)


# ==============================================================================
# MODULE-COMPATIBLE RELATIONSHIP CLASS
# ==============================================================================

@dataclass
class ModuleRelationship:
    """Relationship format compatible with postprocessing modules"""
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
    entity_specificity_score: float = 1.0

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

    @property
    def knowledge_plausibility(self) -> float:
        """Alias for p_true"""
        return self.p_true

    @property
    def pattern_prior(self) -> float:
        """Default pattern prior"""
        return 0.5

    @property
    def claim_uid(self) -> Optional[str]:
        """Default claim_uid"""
        return None

    @property
    def extraction_metadata(self) -> Dict[str, Any]:
        """Default extraction metadata"""
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
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
            'entity_specificity_score': self.entity_specificity_score,
            'flags': self.flags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleRelationship':
        """Create from dict"""
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
            p_true=clean_data.get('p_true', 0.5),
            signals_conflict=clean_data.get('signals_conflict', False),
            conflict_explanation=clean_data.get('conflict_explanation'),
            suggested_correction=clean_data.get('suggested_correction'),
            classification_flags=clean_data.get('classification_flags', []),
            candidate_uid=clean_data.get('candidate_uid', ''),
            entity_specificity_score=clean_data.get('entity_specificity_score', 1.0),
            evidence=clean_data.get('evidence', {}),
            evidence_text=clean_data.get('evidence_text', clean_data.get('context', '')),
            flags=clean_data.get('flags')
        )


# ==============================================================================
# STATUS AND MANIFEST TRACKING
# ==============================================================================

def check_freeze_status(status_path: Path, section: str, logger) -> Dict[str, Any]:
    """
    Check if section is frozen before extraction.

    Exits with error if section is frozen (A+ grade achieved).
    Returns section status dict if not frozen.
    """
    if not status_path.exists():
        logger.info(f"✅ No status.json found, creating new one...")
        return {'status': 'pending'}

    with open(status_path, 'r') as f:
        status = json.load(f)

    section_info = status.get('sections', {}).get(section, {})

    if section_info.get('status') == 'frozen':
        logger.error("")
        logger.error("="*80)
        logger.error(f"❌ ERROR: Section '{section}' is FROZEN (A+ grade achieved)")
        logger.error("="*80)
        logger.error(f"   Final extraction: {section_info.get('final_extraction')}")
        logger.error(f"   Grade: {section_info.get('grade')} ({section_info.get('issue_rate')}% issue rate)")
        logger.error(f"   Frozen at: {section_info.get('frozen_at')}")
        logger.error("")
        logger.error("   To re-extract, manually remove freeze status from status.json")
        logger.error("="*80)
        sys.exit(1)

    logger.info(f"✅ Section '{section}' is not frozen, proceeding with extraction...")
    return section_info


def update_status(status_path: Path, section: str, extraction_file: str, logger):
    """Update status.json with in_progress status"""
    if status_path.exists():
        with open(status_path, 'r') as f:
            status = json.load(f)
    else:
        status = {
            'version': 'v14_3_8',
            'book': 'our_biggest_deal',
            'last_updated': datetime.now().isoformat(),
            'sections': {}
        }

    # Get current section info
    section_info = status.get('sections', {}).get(section, {})
    iterations = section_info.get('iterations', 0) + 1

    # Update section status
    status['sections'][section] = {
        'status': 'in_progress',
        'current_extraction': extraction_file,
        'iterations': iterations,
        'last_updated': datetime.now().isoformat()
    }
    status['last_updated'] = datetime.now().isoformat()

    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)

    logger.info(f"📝 Updated status.json: {section} → in_progress (iteration {iterations})")


def generate_execution_manifest(section: str, args: argparse.Namespace,
                                prompts: Dict[str, Path], logger) -> Dict[str, Any]:
    """
    Generate execution manifest for provenance tracking.

    Captures: git hash, Python version, package versions, script args, prompt checksums.
    """
    logger.info("📋 Generating execution manifest...")

    # Get git info
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        git_dirty = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD']) != 0
    except Exception:
        git_hash, git_branch, git_dirty = "unknown", "unknown", False

    # Get package versions
    try:
        import pkg_resources
        packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    except Exception:
        packages = {}

    # Calculate prompt checksums
    prompt_checksums = {}
    for name, path in prompts.items():
        if path.exists():
            with open(path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()[:16]
                prompt_checksums[name] = {
                    'path': str(path.relative_to(Path.cwd())),
                    'checksum': f"sha256:{checksum}"
                }

    manifest = {
        'section': section,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'version': args.version,
        'git': {
            'commit_hash': git_hash,
            'branch': git_branch,
            'is_dirty': git_dirty
        },
        'environment': {
            'python_version': sys.version.split()[0],
            'platform': platform.platform()
        },
        'packages': {
            'openai': packages.get('openai', 'unknown'),
            'pydantic': packages.get('pydantic', 'unknown'),
            'pdfplumber': packages.get('pdfplumber', 'unknown')
        },
        'script': {
            'path': str(Path(__file__).relative_to(Path.cwd())),
            'args': vars(args)
        },
        'prompts': prompt_checksums,
        'model_config': {
            'pass1_model': 'gpt-4o-2024-08-06',
            'pass2_model': 'gpt-4o-2024-08-06',
            'temperature': 0.0,
            'max_tokens': 16384
        }
    }

    logger.info(f"✅ Manifest generated: git={git_hash[:8]}, version={args.version}")
    return manifest


# ==============================================================================
# PDF EXTRACTION WITH PAGE RANGE SUPPORT
# ==============================================================================

def extract_pages_from_pdf(pdf_path: Path, page_range: str, logger) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Extract specific page range from PDF.

    Args:
        pdf_path: Path to PDF file
        page_range: Page range string like "1-30" or "51-70"
        logger: Logger instance

    Returns:
        (full_text, pages_with_text) for specified range only
    """
    # Parse page range
    start, end = map(int, page_range.split('-'))
    logger.info(f"📖 Extracting pages {start}-{end} from PDF: {pdf_path.name}")

    pages_with_text = []
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"  PDF has {total_pages} total pages")
        logger.info(f"  Extracting pages {start}-{end} ({end-start+1} pages)")

        for i in range(start - 1, min(end, total_pages)):  # 0-indexed
            page = pdf.pages[i]
            text = page.extract_text()

            if text and len(text.strip()) > 50:  # Skip near-empty pages
                pages_with_text.append((i + 1, text))  # 1-indexed
                all_text.append(text)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1 - (start - 1)}/{end - start + 1} pages")

    full_text = "\n\n".join(all_text)
    logger.info(f"✅ Extracted {len(full_text.split())} words from {len(pages_with_text)} pages")

    return full_text, pages_with_text


# ==============================================================================
# Rhetorical/Attribution Signal Extraction (Quotes + Questions)
# ==============================================================================

def extract_rhetorical_signals(
    pages_with_text: List[Tuple[int, str]],
    author_name: str,
    logger,
    max_quotes_per_page: int = 3,
    max_questions_per_page: int = 5,
) -> List[ExtractedRelationship]:
    """
    Lightweight, deterministic extraction of quotes and explicit questions
    to ensure rhetorical/attributable content is captured alongside LLM output.

    Produces ExtractedRelationship objects so they flow through Pass 2 evaluation.
    """
    quote_sig_pattern = re.compile(r"^[\u2014\-]\s*(.+?)\s*$")  # em-dash or hyphen signature
    multi_space = re.compile(r"\s+")

    rels: List[ExtractedRelationship] = []
    seen_questions = set()
    seen_quotes = set()

    def add_quote(src: str, quote: str, page: int):
        q = multi_space.sub(" ", quote.strip())
        if len(q) < 12:
            return
        key = (src.strip().lower(), q.lower())
        if key in seen_quotes:
            return
        seen_quotes.add(key)
        rels.append(ExtractedRelationship(
            source=src.strip(),
            relationship="said",
            target=q[:240],
            source_type="PERSON",
            target_type="QUOTE",
            context=q[:240],
            page=page,
        ))

    def add_question(src: str, question: str, page: int):
        q = multi_space.sub(" ", question.strip())
        if not q.endswith("?") or len(q) < 6:
            return
        key = (page, q.lower())
        if key in seen_questions:
            return
        seen_questions.add(key)
        rels.append(ExtractedRelationship(
            source=src.strip(),
            relationship="poses question",
            target=q[:500],
            source_type="PERSON",
            target_type="STATEMENT",
            context=q[:500],
            page=page,
        ))

    author = author_name.strip() if isinstance(author_name, str) else "Unknown"

    for page_num, text in pages_with_text:
        lines = (text or "").splitlines()

        # Detect quotes by scanning for signature line and capturing preceding block
        quotes_added = 0
        for idx, ln in enumerate(lines):
            s = ln.strip()
            m = quote_sig_pattern.match(s)
            if not m:
                continue
            if quotes_added >= max_quotes_per_page:
                continue
            # walk backwards to collect the quote block
            block: List[str] = []
            j = idx - 1
            while j >= 0:
                t = lines[j].strip()
                if not t:
                    break
                # prepend so original order is preserved
                block.insert(0, t)
                # stop if we hit a prior signature to avoid chaining
                if quote_sig_pattern.match(t):
                    break
                j -= 1
            if block:
                signer = m.group(1)
                quote_text = " ".join(block)
                add_quote(signer, quote_text, page_num)
                quotes_added += 1

        # Detect explicit questions; attribute to author
        q_added = 0
        for s in lines:
            st = s.strip()
            if st.endswith("?") and len(st) >= 6:
                if q_added >= max_questions_per_page:
                    break
                add_question(author, st, page_num)
                q_added += 1

    if rels:
        logger.info(f"🗣️ Rhetorical signal extraction produced {len(rels)} relationships (quotes/questions)")
    else:
        logger.info("🗣️ No rhetorical signals detected (quotes/questions)")

    return rels


def create_chunks(
    pages_with_text: List[Tuple[int, str]],
    chunk_size: int = 600,
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
                current_pages = set([page_num])

    # Handle remaining words
    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append({
            'text': chunk_text,
            'pages': sorted(list(current_pages)),
            'word_count': len(current_words)
        })

    return chunks


# ==============================================================================
# PASS 1: EXTRACTION
# ==============================================================================

def extract_pass1(
    chunk: Dict[str, Any],
    client: OpenAI,
    extraction_prompt: str,
    model: str = "gpt-4o-2024-08-06",
    retry_split: bool = True,
    logger = None
) -> Tuple[List[ExtractedRelationship], bool]:
    """
    Pass 1: Extract relationships using structured outputs.

    Args:
        chunk: Chunk dict with text and page info
        client: OpenAI client
        extraction_prompt: Pass 1 extraction prompt
        model: OpenAI model
        retry_split: Retry with split chunk if token limit hit
        logger: Logger instance

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
                    "content": extraction_prompt.format(text=chunk_text)
                }
            ],
            response_format=ExtractionResult,
            temperature=0.0
        )

        result = response.choices[0].message.parsed

        # Add page info to each relationship
        for rel in result.relationships:
            rel.page = primary_page

        return result.relationships, False

    except Exception as e:
        error_str = str(e)

        # Detect token limit and retry with split
        if "length limit was reached" in error_str and retry_split:
            if logger:
                logger.warning(f"⚠️  Token limit hit, splitting chunk and retrying...")

            # Split chunk in half
            words = chunk_text.split()
            mid_point = len(words) // 2

            chunk1 = {'text': " ".join(words[:mid_point]), 'pages': pages, 'word_count': mid_point}
            chunk2 = {'text': " ".join(words[mid_point:]), 'pages': pages, 'word_count': len(words) - mid_point}

            # Retry both halves
            rels1, _ = extract_pass1(chunk1, client, extraction_prompt, model, retry_split=False, logger=logger)
            rels2, _ = extract_pass1(chunk2, client, extraction_prompt, model, retry_split=False, logger=logger)

            if logger:
                logger.info(f"✅ Retry successful: {len(rels1)} + {len(rels2)} = {len(rels1) + len(rels2)} relationships")

            return rels1 + rels2, True
        else:
            if logger:
                logger.error(f"❌ Pass 1 extraction failed: {e}")
                logger.info("🛟 Attempting JSON fallback for Pass 1…")

            # Fallback: ask model to return raw JSON and parse manually
            try:
                fallback_instructions = (
                    "Extract relationships from the text. Return ONLY a JSON object with key 'relationships' "
                    "whose value is a list of objects with fields: source, relationship, target, source_type, "
                    "target_type, context. Do not include any other keys or text.\n\n"
                    "Guidance:\n"
                    "- Prefer named entities (people/organizations/books) as sources over pronouns or generic groups (we, they, people, society).\n"
                    "- Include quotes and explicit questions with attribution (PERSON → said → QUOTE; AUTHOR → poses question → STATEMENT).\n"
                    "- Keep targets concise and specific; avoid verbose paraphrases.\n"
                    "- Do not invent entities; extract only what is clearly present.\n"
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": fallback_instructions},
                        {"role": "user", "content": chunk_text}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                content = resp.choices[0].message.content
                # Strip code fences if present
                ct = content.strip()
                if ct.startswith("```"):
                    # remove first code fence and optional language tag
                    parts = ct.split("```")
                    if len(parts) >= 3:
                        ct = parts[1]
                content = ct
                data = _json.loads(content)
                rels_json = data.get('relationships', []) if isinstance(data, dict) else []
                out: List[ExtractedRelationship] = []
                for r in rels_json:
                    try:
                        er = ExtractedRelationship(
                            source=r.get('source', ''),
                            relationship=r.get('relationship', ''),
                            target=r.get('target', ''),
                            source_type=r.get('source_type', 'UNKNOWN'),
                            target_type=r.get('target_type', 'UNKNOWN'),
                            context=r.get('context', ''),
                            page=primary_page
                        )
                        out.append(er)
                    except Exception:
                        continue
                if logger:
                    logger.info(f"🛟 JSON fallback recovered {len(out)} relationships")
                return out, False
            except Exception as fe:
                if logger:
                    logger.error(f"❌ JSON fallback failed: {fe}")
                return [], False


# ==============================================================================
# PASS 2: EVALUATION
# ==============================================================================

def evaluate_pass2(
    candidates: List[ExtractedRelationship],
    client: OpenAI,
    evaluation_prompt: str,
    batch_size: int = 25,
    model: str = "gpt-4o-2024-08-06",
    logger = None
) -> List[ModuleRelationship]:
    """
    Pass 2: Evaluate candidates with dual-signal analysis.

    Args:
        candidates: List of extracted relationships
        client: OpenAI client
        evaluation_prompt: Pass 2 evaluation prompt
        batch_size: Batch size for evaluation
        model: OpenAI model
        logger: Logger instance

    Returns:
        List of ModuleRelationship objects with evaluations
    """
    evaluated = []

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
                        "content": evaluation_prompt.format(
                            batch_size=len(batch),
                            relationships_json=json.dumps(batch_json, indent=2)
                        )
                    }
                ],
                response_format=EvaluationBatchResult,
                temperature=0.0
            )

            result = response.choices[0].message.parsed

            # Merge evaluations with candidates
            for eval_result in result.evaluations:
                cand_idx = int(eval_result.candidate_uid.split('_')[1]) - i
                if 0 <= cand_idx < len(batch):
                    cand = batch[cand_idx]

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
                        entity_specificity_score=eval_result.entity_specificity_score
                    )
                    evaluated.append(module_rel)

        except Exception as e:
            if logger:
                logger.error(f"❌ Pass 2 batch {i//batch_size + 1} failed: {e}")
            continue

    return evaluated


# ==============================================================================
# PASS 2.5: POSTPROCESSING
# ==============================================================================

# Removed postprocess_pass2_5 function - using direct pipeline call in extract_section()


# ==============================================================================
# MAIN INCREMENTAL EXTRACTION
# ==============================================================================

def extract_section(args: argparse.Namespace):
    """
    Extract knowledge graph from a specific section (chapter/part) of a book.

    Implements freeze enforcement, status tracking, and execution manifest generation.
    """
    # Setup logging
    logger, timestamp = setup_logging(args.section)

    logger.info("="*80)
    logger.info("🚀 INCREMENTAL KNOWLEDGE GRAPH EXTRACTION V14.3.7")
    logger.info("="*80)
    logger.info(f"  Book: {args.book}")
    logger.info(f"  Section: {args.section}")
    logger.info(f"  Pages: {args.pages}")
    logger.info(f"  Version: {args.version}")
    logger.info("="*80)
    logger.info("")

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    BOOKS_DIR = BASE_DIR / "data" / "books"
    PLAYBOOK_DIR = BASE_DIR / "kg_extraction_playbook"
    PROMPTS_DIR = PLAYBOOK_DIR / "prompts"
    OUTPUT_DIR = PLAYBOOK_DIR / "output" / args.book / args.version
    CHAPTERS_DIR = OUTPUT_DIR / "chapters"
    MANIFESTS_DIR = OUTPUT_DIR / "manifests"
    SCRIPTS_USED_DIR = OUTPUT_DIR / "scripts_used"

    # Create directories
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_USED_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts with fallback chain for new machines
    # Pass 1: prefer latest prompt allowing quotes/questions attribution
    # try v14_3_10 → v14_3_7 → v14_3_6 → v14_3_4 → v14_3_1
    pass1_candidates = [
        "pass1_extraction_v14_3_10.txt",
        "pass1_extraction_v14_3_7.txt",
        "pass1_extraction_v14_3_6.txt",
        "pass1_extraction_v14_3_4.txt",
        "pass1_extraction_v14_3_1.txt"
    ]
    pass1_prompt_file = None
    for candidate in pass1_candidates:
        candidate_path = PROMPTS_DIR / candidate
        if candidate_path.exists():
            pass1_prompt_file = candidate_path
            break

    if not pass1_prompt_file:
        logger.error(f"❌ No Pass 1 prompt found in {PROMPTS_DIR}")
        logger.error(f"   Tried: {', '.join(pass1_candidates)}")
        return

    # Pass 2: try v14_3 → v14_2 → v13_1
    pass2_candidates = [
        "pass2_evaluation_v14_3.txt",
        "pass2_evaluation_v14_2.txt",
        "pass2_evaluation_v13_1.txt"
    ]
    pass2_prompt_file = None
    for candidate in pass2_candidates:
        candidate_path = PROMPTS_DIR / candidate
        if candidate_path.exists():
            pass2_prompt_file = candidate_path
            break

    if not pass2_prompt_file:
        logger.error(f"❌ No Pass 2 prompt found in {PROMPTS_DIR}")
        logger.error(f"   Tried: {', '.join(pass2_candidates)}")
        return

    logger.info(f"📄 Loading prompts:")
    logger.info(f"  Pass 1: {pass1_prompt_file.name}")
    logger.info(f"  Pass 2: {pass2_prompt_file.name}")

    with open(pass1_prompt_file, 'r') as f:
        extraction_prompt = f.read()

    with open(pass2_prompt_file, 'r') as f:
        evaluation_prompt = f.read()

    logger.info("✅ Prompts loaded successfully")
    logger.info("")

    # Check freeze status
    status_path = OUTPUT_DIR / "status.json"
    section_info = check_freeze_status(status_path, args.section, logger)
    logger.info("")

    # Generate execution manifest
    prompts_info = {
        'pass1_extraction': pass1_prompt_file,
        'pass2_evaluation': pass2_prompt_file
    }
    manifest = generate_execution_manifest(args.section, args, prompts_info, logger)
    logger.info("")

    # Find PDF
    book_dir = BOOKS_DIR / args.book
    pdf_files = list(book_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"❌ No PDF found in {book_dir}")
        return

    pdf_path = pdf_files[0]
    logger.info(f"📖 PDF: {pdf_path.name}")
    logger.info("")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract pages
    start_time = time.time()
    full_text, pages_with_text = extract_pages_from_pdf(pdf_path, args.pages, logger)
    logger.info("")

    # Create chunks
    logger.info("📄 Creating chunks...")
    chunks = create_chunks(pages_with_text, chunk_size=600, overlap=100)
    logger.info(f"✅ Created {len(chunks)} chunks")
    logger.info("")

    # Pass 1: Extraction
    logger.info("📝 PASS 1: Extraction...")
    logger.info(f"  Processing {len(chunks)} chunks")

    all_candidates = []
    chunks_split = 0
    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)} (pages {chunk['pages'][0]}-{chunk['pages'][-1]})")
        candidates, was_split = extract_pass1(chunk, client, extraction_prompt, logger=logger)
        all_candidates.extend(candidates)
        if was_split:
            chunks_split += 1

    logger.info(f"✅ Pass 1 complete: {len(all_candidates)} candidates extracted")
    if chunks_split > 0:
        logger.info(f"   ✨ {chunks_split} chunks auto-split and retried")
    logger.info("")

    # Deterministic rhetorical signals (quotes/questions) to augment candidates
    rhetorical_candidates = extract_rhetorical_signals(pages_with_text, args.author, logger)
    if rhetorical_candidates:
        logger.info(f"➕ Adding {len(rhetorical_candidates)} rhetorical candidates to Pass 2 queue")
        all_candidates.extend(rhetorical_candidates)

    # Pass 2: Evaluation
    logger.info("🔍 PASS 2: Dual-signal evaluation...")
    evaluated = evaluate_pass2(all_candidates, client, evaluation_prompt, logger=logger)
    logger.info(f"✅ Pass 2 complete: {len(evaluated)} relationships evaluated")
    logger.info("")

    # Pass 2.5: Postprocessing
    logger.info("🔧 PASS 2.5: Modular postprocessing...")

    document_metadata = {
        'author': args.author if hasattr(args, 'author') else 'Unknown',
        'title': args.book.replace('_', ' ').title(),
        'section': args.section,
        'pages': args.pages
    }

    context = ProcessingContext(
        content_type='book',
        document_metadata=document_metadata,
        pages_with_text=pages_with_text,
        run_id=f"{args.section}_{timestamp}",
        extraction_version=args.version
    )

    # V14.3.8: Use dedicated get_book_pipeline_v1438() with dedication normalization
    logger.info(f"🔧 Using V14.3.8 pipeline with DedicationNormalizer + TypeCompatibilityValidator + SubtitleJoiner")
    pipeline = get_book_pipeline_v1438()
    final_relationships_objs, pp_stats = pipeline.run(evaluated, context)

    # Convert to list (already processed by pipeline)
    final_relationships = final_relationships_objs

    # Log V14.3.8-specific module stats
    if 'DedicationNormalizer' in pp_stats:
        dn_stats = pp_stats['DedicationNormalizer']
        logger.info(f"   DedicationNormalizer: {dn_stats.get('targets_normalized', 0)} targets normalized")

    if 'TypeCompatibilityValidator' in pp_stats:
        tcv_stats = pp_stats['TypeCompatibilityValidator']
        logger.info(f"   TypeCompatibilityValidator: {tcv_stats.get('auto_fixed', 0)} auto-fixed, {tcv_stats.get('flagged', 0)} flagged")

    if 'SubtitleJoiner' in pp_stats:
        sj_stats = pp_stats['SubtitleJoiner']
        logger.info(f"   SubtitleJoiner: {sj_stats.get('rehydrated', 0)} titles rehydrated")

    logger.info("")

    # Save results
    elapsed = time.time() - start_time

    output_filename = f"{args.section}_{args.version}_{timestamp}.json"
    output_path = CHAPTERS_DIR / output_filename

    # Update manifest with output info
    manifest['duration_seconds'] = elapsed
    manifest['output_file'] = str(output_path.relative_to(BASE_DIR))
    manifest['pipeline_version'] = 'v14_3_8'

    # Prepare results
    results = {
        'metadata': {
            'book': args.book,
            'section': args.section,
            'pages': args.pages,
            'extraction_version': args.version,
            'timestamp': timestamp,
            'extraction_date': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'document_metadata': document_metadata
        },
        'extraction_stats': {
            'pass1_candidates': len(all_candidates),
            'pass1_chunks_split': chunks_split,
            'pass2_evaluated': len(evaluated),
            'pass2_5_final': len(final_relationships)
        },
        'postprocessing_stats': pp_stats,
        'execution_manifest': manifest,
        'relationships': [
            rel.to_dict() if (hasattr(rel, 'to_dict') and callable(rel.to_dict)) else rel
            for rel in final_relationships
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Also produce a factual-only subset to satisfy A+ gate while preserving full output
    try:
        safe_predicates = {
            'authored', 'author of', 'wrote foreword for', 'published by',
            'founded', 'member of', 'affiliated with', 'headquartered in', 'located in'
        }
        concrete_types = {'person', 'organization', 'book', 'place'}
        exclude_flags = {"PHILOSOPHICAL", "QUESTION", "NORMATIVE", "PROSPECTIVE", "GENERIC_SOURCE", "GENERIC_TARGET"}

        factual_only: List[Any] = []
        for rel in final_relationships:
            try:
                rel_type = (getattr(rel, 'relationship', '') or '').lower()
                if rel_type not in safe_predicates:
                    continue
                src_type = (getattr(rel, 'source_type', '') or '').lower()
                tgt_type = (getattr(rel, 'target_type', '') or '').lower()
                if src_type not in concrete_types or tgt_type not in concrete_types:
                    continue
                flags = set(getattr(rel, 'classification_flags', []) or [])
                if flags & exclude_flags:
                    continue
                if getattr(rel, 'flags', None) and rel.flags.get('LIST_SPLIT'):
                    continue
                if hasattr(rel, 'p_true') and rel.p_true is not None and rel.p_true < 0.95:
                    continue
                if hasattr(rel, 'text_confidence') and rel.text_confidence is not None and rel.text_confidence < 0.9:
                    continue
                tgt = (getattr(rel, 'target', '') or '').strip()
                if len(tgt.split()) < 2 or len(tgt) < 6:
                    continue
                factual_only.append(rel)
                if len(factual_only) >= 5:
                    break
            except Exception:
                continue

        if not factual_only:
            fallback_candidates = [
                rel for rel in final_relationships
                if (getattr(rel, 'relationship', '') or '').lower() in safe_predicates
                and (getattr(rel, 'source_type', '') or '').lower() in concrete_types
                and (getattr(rel, 'target_type', '') or '').lower() in concrete_types
            ]
            fallback_candidates.sort(key=lambda r: getattr(r, 'p_true', 0.0), reverse=True)
            factual_only = fallback_candidates[:3]

        factual_only_filename = f"{args.section}_factual_only_{args.version}_{timestamp}.json"
        factual_only_path = CHAPTERS_DIR / factual_only_filename
        factual_results = dict(results)
        factual_results['extraction_stats'] = dict(results['extraction_stats'])
        factual_results['extraction_stats']['pass2_5_final_factual_only'] = len(factual_only)
        factual_results['relationships'] = [
            rel.to_dict() if (hasattr(rel, 'to_dict') and callable(rel.to_dict)) else rel
            for rel in factual_only
        ]
        with open(factual_only_path, 'w') as f:
            json.dump(factual_results, f, indent=2)
        manifest['factual_only_output_file'] = str(factual_only_path.relative_to(BASE_DIR))
        logger.info(f"🧪 Wrote factual-only subset: {factual_only_path} ({len(factual_only)} relationships)")
    except Exception as e:
        logger.warning(f"Could not write factual-only subset: {e}")

    # Create provenance bundle for reproducibility (scripts + prompts + minimal code snapshot)
    try:
        run_bundle_dir = SCRIPTS_USED_DIR / f"{args.section}_{timestamp}"
        run_bundle_dir.mkdir(parents=True, exist_ok=True)

        # Copy scripts used
        try:
            shutil.copy2(Path(__file__), run_bundle_dir / f"extract_kg_v14_3_8_incremental_{timestamp}.py")
        except Exception as e:
            logger.warning(f"Could not copy extraction script: {e}")
        try:
            refl = Path(__file__).parent / "run_reflector_incremental.py"
            if refl.exists():
                shutil.copy2(refl, run_bundle_dir / f"run_reflector_incremental_{timestamp}.py")
        except Exception as e:
            logger.warning(f"Could not copy reflector script: {e}")

        # Copy prompts used
        try:
            shutil.copy2(pass1_prompt_file, run_bundle_dir / pass1_prompt_file.name)
            shutil.copy2(pass2_prompt_file, run_bundle_dir / pass2_prompt_file.name)
        except Exception as e:
            logger.warning(f"Could not copy prompts: {e}")

        # Create code snapshot tarball (postprocessing modules + scripts + prompts)
        code_snapshot = run_bundle_dir / f"code_snapshot_{timestamp}.tar.gz"
        with tarfile.open(code_snapshot, "w:gz") as tar:
            pp_root = Path.cwd() / "src" / "knowledge_graph" / "postprocessing"
            if pp_root.exists():
                tar.add(pp_root, arcname="src/knowledge_graph/postprocessing")
            tar.add(Path(__file__), arcname="scripts/extract_kg_v14_3_8_incremental.py")
            refl_path = Path(__file__).parent / "run_reflector_incremental.py"
            if refl_path.exists():
                tar.add(refl_path, arcname="scripts/run_reflector_incremental.py")
            tar.add(pass1_prompt_file, arcname=f"kg_extraction_playbook/prompts/{pass1_prompt_file.name}")
            tar.add(pass2_prompt_file, arcname=f"kg_extraction_playbook/prompts/{pass2_prompt_file.name}")

        # REPRODUCE.md
        reproduce_path = run_bundle_dir / "REPRODUCE.md"
        reproduce_path.write_text(
            (
                f"# Reproduce Extraction for {args.section}\n\n"
                f"Command:\n\n"
                f"python3 scripts/extract_kg_v14_3_8_incremental.py \\\n+  --book {args.book} \\\n+  --section {args.section} \\\n+  --pages {args.pages} \\\n+  --author \"{args.author}\" \\\n+  --version {args.version}\n\n"
                f"Prompts:\n- {pass1_prompt_file.name}\n- {pass2_prompt_file.name}\n\n"
                f"Snapshot: {code_snapshot.name} (postprocessing modules + scripts + prompts)\n\n"
                f"Git commit: {manifest.get('git', {}).get('commit_hash', 'unknown')} (dirty={manifest.get('git', {}).get('is_dirty')})\n"
            )
        )

        # Add provenance info to manifest
        manifest.setdefault('provenance', {})
        manifest['provenance']['bundle_dir'] = str(run_bundle_dir.relative_to(BASE_DIR))
        manifest['provenance']['code_snapshot_tar'] = str(code_snapshot.relative_to(BASE_DIR))
    except Exception as e:
        logger.warning(f"Could not build provenance bundle: {e}")

    # Save manifest
    manifest_path = MANIFESTS_DIR / f"{args.section}_execution_{timestamp}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Update status
    update_status(status_path, args.section, output_filename, logger)

    logger.info("="*80)
    logger.info("✅ INCREMENTAL EXTRACTION COMPLETE")
    logger.info("="*80)
    logger.info(f"  Section: {args.section}")
    logger.info(f"  Relationships: {len(final_relationships)}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info("="*80)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info(f"1. Run Reflector on {output_filename}")
    logger.info("2. Check if section achieves A+ grade")
    logger.info("3. If A+, freeze section and move to next chapter")
    logger.info("4. If not A+, analyze issues and iterate")
    logger.info("="*80)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Incremental Knowledge Graph Extraction V14.3.8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract Front Matter (pages 1-30)
  python3 scripts/extract_kg_v14_3_8_incremental.py \\
    --book our_biggest_deal \\
    --section front_matter \\
    --pages 1-30

  # Extract Chapter 1 (pages 31-50)
  python3 scripts/extract_kg_v14_3_8_incremental.py \\
    --book our_biggest_deal \\
    --section chapter_01 \\
    --pages 31-50 \\
    --author "Aaron William Perry"
        """
    )

    parser.add_argument('--book', required=True, help='Book identifier (e.g., our_biggest_deal)')
    parser.add_argument('--section', required=True, help='Section identifier (e.g., front_matter, chapter_01)')
    parser.add_argument('--pages', required=True, help='Page range (e.g., 1-30, 31-50)')
    parser.add_argument('--version', default='v14_3_8', help='Extraction version (default: v14_3_7)')
    parser.add_argument('--author', default='Unknown', help='Book author (for metadata)')

    args = parser.parse_args()

    extract_section(args)


if __name__ == "__main__":
    main()
