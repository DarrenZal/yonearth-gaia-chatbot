#!/usr/bin/env python3
"""
Incremental Knowledge Graph Extraction V14.3.4

Purpose: Chapter-by-chapter extraction with freeze enforcement, tuned to fix
front matter issues (authorship reversal and list over-fragmentation).

Highlights in v14.3.4:
- Pipeline adds FieldNormalizer early to canonicalize fields (relationship/predicate)
- ListSplitter is quote-aware and conservative in front matter (subtitles)
- BibliographicCitationParser handles "Last, First. Title" author-title format

Usage:
  python3 scripts/extract_kg_v14_3_4_incremental.py \
    --book our_biggest_deal \
    --section front_matter \
    --pages 1-30 \
    --author "Aaron William Perry"
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
from dataclasses import dataclass, field as dataclass_field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from openai import OpenAI
import pdfplumber

from src.knowledge_graph.postprocessing import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline


def setup_logging(section: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'kg_extraction_incremental_{section}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__), timestamp


class ExtractedRelationship(BaseModel):
    source: str
    relationship: str
    target: str
    source_type: str
    target_type: str
    context: str
    page: int


class ExtractionResult(BaseModel):
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


class RelationshipEvaluation(BaseModel):
    candidate_uid: str
    text_confidence: float = Field(ge=0.0, le=1.0)
    p_true: float = Field(ge=0.0, le=1.0)
    entity_specificity_score: float = Field(ge=0.0, le=1.0)
    signals_conflict: bool
    conflict_explanation: Optional[str] = None
    suggested_correction: Optional[Dict[str, str]] = None
    source_type: str
    target_type: str
    classification_flags: List[str] = Field(default_factory=list)


class EvaluationBatchResult(BaseModel):
    evaluations: List[RelationshipEvaluation] = Field(default_factory=list)


@dataclass
class ModuleRelationship:
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

    evidence: Dict[str, Any] = dataclass_field(default_factory=dict)
    evidence_text: str = ""
    flags: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.evidence = {'page_number': self.page}
        self.evidence_text = self.context
        if self.flags is None:
            self.flags = {}

    @property
    def knowledge_plausibility(self) -> float:
        return self.p_true

    @property
    def pattern_prior(self) -> float:
        return 0.5

    @property
    def claim_uid(self) -> Optional[str]:
        return None

    @property
    def extraction_metadata(self) -> Dict[str, Any]:
        return {}

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
            'candidate_uid': self.candidate_uid,
            'entity_specificity_score': self.entity_specificity_score,
            'flags': self.flags
        }


def check_freeze_status(status_path: Path, section: str, logger) -> Dict[str, Any]:
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
        logger.error("To re-extract, remove freeze status from status.json")
        logger.error("="*80)
        sys.exit(1)
    logger.info(f"✅ Section '{section}' is not frozen, proceeding...")
    return section_info


def update_status(status_path: Path, section: str, extraction_file: str, logger):
    if status_path.exists():
        with open(status_path, 'r') as f:
            status = json.load(f)
    else:
        status = {
            'version': 'v14_3_4',
            'book': 'our_biggest_deal',
            'last_updated': datetime.now().isoformat(),
            'sections': {}
        }
    section_info = status.get('sections', {}).get(section, {})
    iterations = section_info.get('iterations', 0) + 1
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


def generate_execution_manifest(section: str, args: argparse.Namespace, prompts: Dict[str, Path], logger) -> Dict[str, Any]:
    logger.info("📋 Generating execution manifest...")
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        git_dirty = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD']) != 0
    except Exception:
        git_hash, git_branch, git_dirty = "unknown", "unknown", False

    try:
        import pkg_resources
        packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    except Exception:
        packages = {}

    prompt_checksums = {}
    for name, path in prompts.items():
        if path.exists():
            with open(path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()[:16]
                prompt_checksums[name] = {'path': str(path.relative_to(Path.cwd())), 'checksum': f"sha256:{checksum}"}

    manifest = {
        'section': section,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'version': args.version,
        'git': {'commit_hash': git_hash, 'branch': git_branch, 'is_dirty': git_dirty},
        'environment': {'python_version': sys.version.split()[0], 'platform': platform.platform()},
        'packages': {
            'openai': packages.get('openai', 'unknown'),
            'pydantic': packages.get('pydantic', 'unknown'),
            'pdfplumber': packages.get('pdfplumber', 'unknown')
        },
        'script': {'path': str(Path(__file__).relative_to(Path.cwd())), 'args': vars(args)},
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


def extract_pages_from_pdf(pdf_path: Path, page_range: str, logger) -> Tuple[str, List[Tuple[int, str]]]:
    start, end = map(int, page_range.split('-'))
    logger.info(f"📖 Extracting pages {start}-{end} from PDF: {pdf_path.name}")
    pages_with_text = []
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"  PDF has {total_pages} total pages")
        logger.info(f"  Extracting pages {start}-{end} ({end-start+1} pages)")
        for i in range(start - 1, min(end, total_pages)):
            page = pdf.pages[i]
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                pages_with_text.append((i + 1, text))
                all_text.append(text)
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1 - (start - 1)}/{end - start + 1} pages")
    full_text = "\n\n".join(all_text)
    logger.info(f"✅ Extracted {len(full_text.split())} words from {len(pages_with_text)} pages")
    return full_text, pages_with_text


def create_chunks(pages_with_text: List[Tuple[int, str]], chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
    chunks = []
    current_words: List[str] = []
    current_pages = set()
    for page_num, text in pages_with_text:
        words = text.split()
        for word in words:
            current_words.append(word)
            current_pages.add(page_num)
            if len(current_words) >= chunk_size:
                chunk_text = " ".join(current_words)
                chunks.append({'text': chunk_text, 'pages': sorted(list(current_pages)), 'word_count': len(current_words)})
                current_words = current_words[-overlap:] if overlap > 0 else []
                current_pages = set([page_num])
    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append({'text': chunk_text, 'pages': sorted(list(current_pages)), 'word_count': len(current_words)})
    return chunks


def extract_pass1(chunk: Dict[str, Any], client: OpenAI, extraction_prompt: str, model: str = "gpt-4o-2024-08-06", retry_split: bool = True, logger=None) -> Tuple[List[ExtractedRelationship], bool]:
    chunk_text = chunk['text']
    pages = chunk['pages']
    primary_page = pages[0] if pages else 1
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": extraction_prompt.format(text=chunk_text)}],
            response_format=ExtractionResult,
            temperature=0.0
        )
        result = response.choices[0].message.parsed
        for rel in result.relationships:
            rel.page = primary_page
        return result.relationships, False
    except Exception as e:
        error_str = str(e)
        if "length limit was reached" in error_str and retry_split:
            if logger:
                logger.warning("⚠️  Token limit hit, splitting chunk and retrying...")
            words = chunk_text.split()
            mid = len(words) // 2
            chunk1 = {'text': " ".join(words[:mid]), 'pages': pages, 'word_count': mid}
            chunk2 = {'text': " ".join(words[mid:]), 'pages': pages, 'word_count': len(words) - mid}
            rels1, _ = extract_pass1(chunk1, client, extraction_prompt, model, retry_split=False, logger=logger)
            rels2, _ = extract_pass1(chunk2, client, extraction_prompt, model, retry_split=False, logger=logger)
            if logger:
                logger.info(f"✅ Retry successful: {len(rels1)} + {len(rels2)} = {len(rels1)+len(rels2)} relationships")
            return rels1 + rels2, True
        else:
            if logger:
                logger.error(f"❌ Pass 1 extraction failed: {e}")
            return [], False


def evaluate_pass2(candidates: List[ExtractedRelationship], client: OpenAI, evaluation_prompt: str, batch_size: int = 25, model: str = "gpt-4o-2024-08-06", logger=None) -> List[ModuleRelationship]:
    evaluated: List[ModuleRelationship] = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
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
                messages=[{"role": "user", "content": evaluation_prompt.format(batch_size=len(batch), relationships_json=json.dumps(batch_json, indent=2))}],
                response_format=EvaluationBatchResult,
                temperature=0.0
            )
            result = response.choices[0].message.parsed
            for eval_result in result.evaluations:
                cand_idx = int(eval_result.candidate_uid.split('_')[1]) - i
                if 0 <= cand_idx < len(batch):
                    cand = batch[cand_idx]
                    evaluated.append(ModuleRelationship(
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
                    ))
        except Exception as e:
            if logger:
                logger.error(f"❌ Pass 2 batch {i//batch_size + 1} failed: {e}")
            continue
    return evaluated


def postprocess_pass2_5(relationships: List[ModuleRelationship], context: ProcessingContext, version: str, logger) -> Tuple[List[ModuleRelationship], Dict[str, Any]]:
    logger.info(f"🔧 PASS 2.5: Running {version} modular postprocessing pipeline...")
    pipeline = get_book_pipeline(version=version)
    processed_objs, stats = pipeline.run(relationships, context)
    logger.info(f"✅ Pass 2.5 complete: {len(relationships)} → {len(processed_objs)} relationships")
    return processed_objs, stats


def extract_section(args: argparse.Namespace):
    logger, timestamp = setup_logging(args.section)
    logger.info("="*80)
    logger.info("🚀 INCREMENTAL KNOWLEDGE GRAPH EXTRACTION V14.3.4")
    logger.info("="*80)
    logger.info(f"  Book: {args.book}")
    logger.info(f"  Section: {args.section}")
    logger.info(f"  Pages: {args.pages}")
    logger.info(f"  Version: {args.version}")
    logger.info("="*80)

    BASE_DIR = Path(__file__).parent.parent
    BOOKS_DIR = BASE_DIR / "data" / "books"
    PLAYBOOK_DIR = BASE_DIR / "kg_extraction_playbook"
    PROMPTS_DIR = PLAYBOOK_DIR / "prompts"
    OUTPUT_DIR = PLAYBOOK_DIR / "output" / args.book / args.version
    CHAPTERS_DIR = OUTPUT_DIR / "chapters"
    MANIFESTS_DIR = OUTPUT_DIR / "manifests"
    CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts (fallbacks included)
    pass1_prompt_candidates = [
        PROMPTS_DIR / "pass1_extraction_v14_3_4.txt",
        PROMPTS_DIR / "pass1_extraction_v14_3_3.txt",
        PROMPTS_DIR / "pass1_extraction_v14_3_2.txt",
        PROMPTS_DIR / "pass1_extraction_v14_3_1.txt",
        PROMPTS_DIR / "pass1_extraction_v14_2.txt",
    ]
    pass2_prompt_candidates = [
        PROMPTS_DIR / "pass2_evaluation_v14_3_4.txt",
        PROMPTS_DIR / "pass2_evaluation_v14_3_3.txt",
        PROMPTS_DIR / "pass2_evaluation_v14_3.txt",
        PROMPTS_DIR / "pass2_evaluation_v14_2.txt",
        PROMPTS_DIR / "pass2_evaluation_v13_1.txt",
    ]

    pass1_prompt_file = next((p for p in pass1_prompt_candidates if p.exists()), PROMPTS_DIR / "pass1_extraction_v14_3_1.txt")
    pass2_prompt_file = next((p for p in pass2_prompt_candidates if p.exists()), PROMPTS_DIR / "pass2_evaluation_v13_1.txt")

    logger.info("📄 Loading prompts:")
    logger.info(f"  Pass 1: {pass1_prompt_file.name}")
    logger.info(f"  Pass 2: {pass2_prompt_file.name}")
    with open(pass1_prompt_file, 'r') as f:
        extraction_prompt = f.read()
    with open(pass2_prompt_file, 'r') as f:
        evaluation_prompt = f.read()
    logger.info("✅ Prompts loaded successfully")
    logger.info("")

    # Freeze check
    status_path = OUTPUT_DIR / "status.json"
    _ = check_freeze_status(status_path, args.section, logger)

    # Manifest
    prompts_info = {'pass1_extraction': pass1_prompt_file, 'pass2_evaluation': pass2_prompt_file}
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

    # Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract and chunk
    start_time = time.time()
    full_text, pages_with_text = extract_pages_from_pdf(pdf_path, args.pages, logger)
    logger.info("")
    logger.info("📄 Creating chunks...")
    chunks = create_chunks(pages_with_text, chunk_size=600, overlap=100)
    logger.info(f"✅ Created {len(chunks)} chunks")
    logger.info("")

    # Pass 1
    logger.info("📝 PASS 1: Extraction...")
    all_candidates: List[ExtractedRelationship] = []
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

    # Pass 2
    logger.info("🔍 PASS 2: Dual-signal evaluation...")
    evaluated = evaluate_pass2(all_candidates, client, evaluation_prompt, logger=logger)
    logger.info(f"✅ Pass 2 complete: {len(evaluated)} relationships evaluated")
    logger.info("")

    # Pass 2.5
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
    pipeline_version = 'v14_3_4' if args.version.startswith('v14_3_4') else ('v14_3_3' if args.version.startswith('v14_3_3') else 'v14_3_2')
    final_relationships, pp_stats = postprocess_pass2_5(evaluated, context, pipeline_version, logger)
    logger.info("")

    # Save
    elapsed = time.time() - start_time
    output_filename = f"{args.section}_{args.version}_{timestamp}.json"
    output_path = CHAPTERS_DIR / output_filename
    manifest['duration_seconds'] = elapsed
    manifest['output_file'] = str(output_path.relative_to(BASE_DIR))
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
        'relationships': [rel.to_dict() for rel in final_relationships]
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    manifest_path = MANIFESTS_DIR / f"{args.section}_execution_{timestamp}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
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


def main():
    parser = argparse.ArgumentParser(
        description="Incremental Knowledge Graph Extraction V14.3.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/extract_kg_v14_3_4_incremental.py \
    --book our_biggest_deal \
    --section front_matter \
    --pages 1-30 \
    --author "Aaron William Perry"
        """
    )
    parser.add_argument('--book', required=True, help='Book identifier (e.g., our_biggest_deal)')
    parser.add_argument('--section', required=True, help='Section identifier (e.g., front_matter, chapter_01)')
    parser.add_argument('--pages', required=True, help='Page range (e.g., 1-30, 31-50)')
    parser.add_argument('--version', default='v14_3_4', help='Extraction version (default: v14_3_4)')
    parser.add_argument('--author', default='Unknown', help='Book author (for metadata)')
    args = parser.parse_args()
    extract_section(args)


if __name__ == "__main__":
    main()
