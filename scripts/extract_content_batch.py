#!/usr/bin/env python3
"""
Unified Content Extraction Script for YonEarth Knowledge Graph.

This script manages the full batch extraction lifecycle for BOTH episodes AND books:
1. Submit: Load all episodes + books, create parent/child chunks, submit batch job
2. Poll: Check status of batch jobs
3. Download: Retrieve and process completed results
4. Retry-failed: Resubmit only failed chunks
5. Merge: Combine results from multiple batches into unified output

Usage:
    python scripts/extract_content_batch.py --submit           # Submit all content
    python scripts/extract_content_batch.py --submit-books     # Submit only books
    python scripts/extract_content_batch.py --poll             # Check status
    python scripts/extract_content_batch.py --download         # Download results
    python scripts/extract_content_batch.py --retry-failed     # Retry failed chunks
    python scripts/extract_content_batch.py --status           # Detailed status report

Multi-Modal Extraction Profiles:
- Episodes: Standard entity/relationship extraction
- Technical books (Soil Stewardship): Process/Tool/Instruction focus
- Fiction books (VIRIDITAS): Character/Location/Event focus with fictional tagging
- Rhetorical books (Y on Earth, Our Biggest Deal): Concept/Argument focus

Production Reliability Features:
- Automatic failure capture and tracking in failed_chunks.json
- Retry mechanism for transient API errors
- Merge capability to combine original + retry results
- Content profile metadata for downstream filtering
- Header/footer stripping for clean PDF extraction
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.knowledge_graph.chunking import ParentChildChunker, ParentChunk, ChildChunk
from src.knowledge_graph.extractors.batch_collector import BatchCollector
from src.knowledge_graph.extractors.entity_extractor import EntityExtractor

# Try to import pdfplumber for book processing
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    pdfplumber = None
    PDF_SUPPORT = False


# =============================================================================
# CONTENT PROFILES - Customized extraction for different content types
# =============================================================================

@dataclass
class ContentProfile:
    """Configuration for content-specific extraction"""
    content_type: str           # "episode", "technical", "fiction", "rhetorical"
    reality_tag: str            # "factual", "fictional", "conceptual"
    system_prompt_focus: str    # Additional instructions for LLM
    chunking_strategy: str      # "speaker", "hierarchical", "narrative", "semantic"


# Book-specific extraction profiles
BOOK_PROFILES: Dict[str, ContentProfile] = {
    "Soil Stewardship Handbook": ContentProfile(
        content_type="hybrid_technical",
        reality_tag="factual_stewardship",
        system_prompt_focus="""Extract TWO layers of information:
1. **Rhetorical Framework:** Capture the core arguments (e.g., 'War on Soil'), historical claims, and the ethical 'call to action'.
2. **Technical Stewardship:** Extract specific Practices, Tools, and Biological Processes that solve the problems identified in layer 1.
Link the 'Philosophical Why' to the 'Technical How'.""",
        chunking_strategy="hierarchical"
    ),
    "VIRIDITAS": ContentProfile(
        content_type="fiction",
        reality_tag="fictional",
        system_prompt_focus="""This is a NARRATIVE FICTION text. Focus on extracting:
- CHARACTERS (mark all as fictional)
- LOCATIONS and SETTINGS (mark as fictional unless real-world reference)
- NARRATIVE EVENTS and PLOT POINTS
- THEMES and SYMBOLS
IMPORTANT: Mark ALL entities from this source with is_fictional=true.
Do NOT extract scientific claims as factual - they are part of the narrative.""",
        chunking_strategy="narrative"
    ),
    "Y on Earth": ContentProfile(
        content_type="rhetorical",
        reality_tag="conceptual",
        system_prompt_focus="""This is a philosophical/rhetorical text. Focus on extracting:
- CORE CONCEPTS and DEFINITIONS
- PHILOSOPHICAL ARGUMENTS and THESES
- VALUES and PRINCIPLES advocated
- PRACTICES connected to concepts
- REAL-WORLD EXAMPLES and CASE STUDIES
Connect abstract concepts to concrete practices where mentioned.""",
        chunking_strategy="semantic"
    ),
    "Our Biggest Deal": ContentProfile(
        content_type="rhetorical",
        reality_tag="conceptual",
        system_prompt_focus="""This is a rhetorical/argumentative text. Focus on extracting:
- CENTRAL THESES and ARGUMENTS
- HISTORICAL REFERENCES and PRECEDENTS
- PROPOSED SOLUTIONS and FRAMEWORKS
- KEY STAKEHOLDERS mentioned
- CAUSE-EFFECT RELATIONSHIPS argued
Distinguish between factual claims and rhetorical positions.""",
        chunking_strategy="semantic"
    ),
}

# Default profile for episodes
EPISODE_PROFILE = ContentProfile(
    content_type="episode",
    reality_tag="factual",
    system_prompt_focus="""Extract entities and relationships from this podcast episode transcript.
Focus on:
- PEOPLE mentioned (guests, experts, historical figures)
- ORGANIZATIONS (companies, nonprofits, institutions)
- PLACES (locations, regions, ecosystems)
- CONCEPTS (practices, technologies, movements)
- PRODUCTS (books, tools, services)
Preserve speaker context where relevant.""",
    chunking_strategy="speaker"
)


# =============================================================================
# PDF TEXT EXTRACTION WITH HEADER/FOOTER STRIPPING
# =============================================================================

class EnhancedPDFExtractor:
    """Extract text from PDFs with intelligent header/footer removal."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.header_pattern = None
        self.footer_pattern = None

    def extract_text(self) -> Tuple[str, int]:
        """
        Extract text from PDF, stripping headers/footers.

        Returns:
            Tuple of (full_text, page_count)
        """
        if not PDF_SUPPORT:
            raise ImportError("pdfplumber required. Install with: pip install pdfplumber")

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        pages_text = []

        with pdfplumber.open(self.pdf_path) as pdf:
            # First pass: detect repeating headers/footers
            self._detect_headers_footers(pdf)

            # Second pass: extract with filtering
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                cleaned = self._clean_page_text(text, page_num)
                if cleaned:
                    pages_text.append(cleaned)

        full_text = "\n\n".join(pages_text)
        return full_text, len(pages_text)

    def _detect_headers_footers(self, pdf) -> None:
        """Detect repeating text patterns across pages (likely headers/footers)."""
        if len(pdf.pages) < 5:
            return  # Not enough pages to detect patterns

        # Sample first and last lines from multiple pages
        first_lines = []
        last_lines = []

        sample_pages = list(range(min(20, len(pdf.pages))))
        for i in sample_pages:
            text = pdf.pages[i].extract_text() or ""
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if len(lines) >= 2:
                first_lines.append(lines[0])
                last_lines.append(lines[-1])

        # Find repeating patterns (appear in >50% of sampled pages)
        threshold = len(sample_pages) * 0.5

        # Check first lines for header pattern
        first_line_counts = {}
        for line in first_lines:
            # Normalize: remove page numbers
            normalized = re.sub(r'\d+', 'N', line)
            first_line_counts[normalized] = first_line_counts.get(normalized, 0) + 1

        for pattern, count in first_line_counts.items():
            if count >= threshold and len(pattern) > 5:
                # Convert back to regex
                self.header_pattern = re.escape(pattern).replace('N', r'\d+')
                break

        # Check last lines for footer pattern
        last_line_counts = {}
        for line in last_lines:
            normalized = re.sub(r'\d+', 'N', line)
            last_line_counts[normalized] = last_line_counts.get(normalized, 0) + 1

        for pattern, count in last_line_counts.items():
            if count >= threshold and len(pattern) > 3:
                self.footer_pattern = re.escape(pattern).replace('N', r'\d+')
                break

    def _clean_page_text(self, text: str, page_num: int) -> str:
        """Clean a single page of text."""
        lines = text.split('\n')
        cleaned_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip standalone page numbers
            if re.match(r'^\d+$', line):
                continue

            # Skip very short lines (likely artifacts)
            if len(line) < 3:
                continue

            # Skip detected header pattern (first few lines)
            if i < 3 and self.header_pattern:
                if re.match(self.header_pattern, line, re.IGNORECASE):
                    continue

            # Skip detected footer pattern (last few lines)
            if i >= len(lines) - 3 and self.footer_pattern:
                if re.match(self.footer_pattern, line, re.IGNORECASE):
                    continue

            cleaned_lines.append(line)

        # Normalize whitespace
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()


# =============================================================================
# CONTENT LOADING FUNCTIONS
# =============================================================================

def load_episodes(episode_range: str = "0-172") -> List[Dict[str, Any]]:
    """
    Load episodes from transcripts directory.

    Args:
        episode_range: Range like "0-172" or comma-separated "1,5,10"

    Returns:
        List of episode dictionaries with transcript data
    """
    episodes = []
    transcripts_dir = Path("data/transcripts")

    # Parse range
    if "-" in episode_range:
        start, end = map(int, episode_range.split("-"))
        episode_nums = range(start, end + 1)
    else:
        episode_nums = [int(x.strip()) for x in episode_range.split(",")]

    for num in episode_nums:
        if num == 26:  # Episode 26 doesn't exist in the series
            continue

        path = transcripts_dir / f"episode_{num}.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)

                # Check for transcript content
                transcript = data.get("full_transcript", "")
                if transcript and len(transcript) > 100:
                    episodes.append({
                        "episode_number": num,
                        "title": data.get("title", f"Episode {num}"),
                        "transcript": transcript,
                        "guest_name": data.get("about_sections", {}).get("about_guest", ""),
                        "profile": EPISODE_PROFILE
                    })
            except Exception as e:
                print(f"Error loading episode {num}: {e}")
        else:
            print(f"Episode file not found: {path}")

    return episodes


def load_books() -> List[Dict[str, Any]]:
    """
    Load books from PDFs with content profiles.

    Returns:
        List of book dictionaries with extracted text and profiles
    """
    if not PDF_SUPPORT:
        print("WARNING: pdfplumber not installed. Cannot process books.")
        print("Install with: pip install pdfplumber")
        return []

    books = []
    books_dir = Path("data/books")

    if not books_dir.exists():
        print(f"Books directory not found: {books_dir}")
        return books

    for book_dir in sorted(books_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        metadata_path = book_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            title = metadata.get("title", book_dir.name)
            pdf_filename = metadata.get("file_path", "")

            # Find PDF file
            pdf_path = book_dir / pdf_filename
            if not pdf_path.exists():
                # Try to find any PDF in the directory
                pdfs = list(book_dir.glob("*.pdf"))
                if pdfs:
                    pdf_path = pdfs[0]
                else:
                    print(f"No PDF found for book: {title}")
                    continue

            # Extract text with header/footer removal
            print(f"Extracting text from: {title}")
            extractor = EnhancedPDFExtractor(pdf_path)
            content, page_count = extractor.extract_text()

            if not content or len(content) < 100:
                print(f"  No content extracted from: {title}")
                continue

            # Get profile for this book
            profile = None
            for profile_title, prof in BOOK_PROFILES.items():
                if profile_title.lower() in title.lower():
                    profile = prof
                    break

            if not profile:
                # Default to rhetorical for unknown books
                print(f"  No profile found for '{title}', using rhetorical default")
                profile = ContentProfile(
                    content_type="rhetorical",
                    reality_tag="conceptual",
                    system_prompt_focus="Extract key concepts, arguments, and relationships.",
                    chunking_strategy="semantic"
                )

            books.append({
                "slug": book_dir.name,
                "title": title,
                "author": metadata.get("author", "Unknown"),
                "content": content,
                "page_count": page_count,
                "word_count": len(content.split()),
                "profile": profile
            })

            print(f"  Loaded: {title} ({page_count} pages, {len(content.split()):,} words)")
            print(f"  Profile: {profile.content_type} / {profile.reality_tag}")

        except Exception as e:
            print(f"Error loading book {book_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    return books


# =============================================================================
# BATCH SUBMISSION WITH CONTENT PROFILES
# =============================================================================

def create_profiled_system_prompt(profile: ContentProfile, base_prompt: str) -> str:
    """
    Create a customized system prompt based on content profile.

    Args:
        profile: The content profile to use
        base_prompt: The base entity extraction prompt

    Returns:
        Customized system prompt
    """
    additions = f"""

=== CONTENT PROFILE: {profile.content_type.upper()} ===
Reality Tag: {profile.reality_tag}

{profile.system_prompt_focus}

IMPORTANT: Tag all extracted entities with:
- content_type: "{profile.content_type}"
- reality_tag: "{profile.reality_tag}"
"""
    return base_prompt + additions


def submit_batch(
    episode_range: str = "0-172",
    include_episodes: bool = True,
    include_books: bool = True,
    dry_run: bool = False
):
    """
    Submit batch extraction job for episodes and/or books.

    Args:
        episode_range: Which episodes to process
        include_episodes: Whether to include episodes
        include_books: Whether to include books
        dry_run: If True, create JSONL files but don't submit to API
    """
    print("=" * 60)
    print("Submitting Batch Extraction Job")
    print("=" * 60)
    print(f"Include episodes: {include_episodes}")
    print(f"Include books: {include_books}")
    print()

    # Ensure output directory exists
    batch_dir = Path("data/batch_jobs")
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    chunker = ParentChildChunker()
    collector = BatchCollector(output_dir=batch_dir)
    extractor = EntityExtractor()

    all_parents: List[ParentChunk] = []
    all_children: List[ChildChunk] = []
    content_metadata: Dict[str, Dict] = {}  # Track profile metadata per chunk

    episodes_processed = 0
    books_processed = 0

    # Process episodes
    if include_episodes:
        print("Loading episodes...")
        episodes = load_episodes(episode_range)
        print(f"Loaded {len(episodes)} episodes")

        for i, episode in enumerate(episodes):
            parents, children = chunker.process_content(
                text=episode["transcript"],
                source_type="episode",
                source_id=str(episode["episode_number"])
            )

            # Track profile metadata for each parent chunk
            for p in parents:
                content_metadata[p.id] = {
                    "content_type": episode["profile"].content_type,
                    "reality_tag": episode["profile"].reality_tag,
                    "source_title": episode["title"]
                }

            all_parents.extend(parents)
            all_children.extend(children)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(episodes)} episodes...")

        episodes_processed = len(episodes)
        print(f"  Episode chunks: {len([p for p in all_parents if p.source_type == 'episode'])} parents")

    # Process books
    if include_books:
        print("\nLoading books...")
        books = load_books()
        print(f"Loaded {len(books)} books")

        for book in books:
            parents, children = chunker.process_content(
                text=book["content"],
                source_type="book",
                source_id=book["slug"]
            )

            # Track profile metadata for each parent chunk
            for p in parents:
                content_metadata[p.id] = {
                    "content_type": book["profile"].content_type,
                    "reality_tag": book["profile"].reality_tag,
                    "source_title": book["title"],
                    "system_prompt_focus": book["profile"].system_prompt_focus
                }

            all_parents.extend(parents)
            all_children.extend(children)
            books_processed += 1
            print(f"  {book['title']}: {len(parents)} parent chunks")

    if not all_parents:
        print("\nNo content to process!")
        return

    # Save content metadata for later use
    metadata_path = batch_dir / "content_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(content_metadata, f, indent=2)
    print(f"\nContent metadata saved to: {metadata_path}")

    # Add all parent chunks to batch collector with profile-aware prompts
    print(f"\nPreparing batch requests...")

    # Group chunks by profile for customized prompts
    chunks_by_profile: Dict[str, List[ParentChunk]] = {}
    for p in all_parents:
        meta = content_metadata.get(p.id, {})
        profile_key = f"{meta.get('content_type', 'episode')}_{meta.get('reality_tag', 'factual')}"
        if profile_key not in chunks_by_profile:
            chunks_by_profile[profile_key] = []
        chunks_by_profile[profile_key].append(p)

    # Add requests with appropriate prompts
    for profile_key, chunks in chunks_by_profile.items():
        print(f"  Adding {len(chunks)} chunks with profile: {profile_key}")

        # Get a sample chunk's metadata for the prompt
        sample_meta = content_metadata.get(chunks[0].id, {})

        # Create profile object from metadata
        profile = ContentProfile(
            content_type=sample_meta.get("content_type", "episode"),
            reality_tag=sample_meta.get("reality_tag", "factual"),
            system_prompt_focus=sample_meta.get("system_prompt_focus", ""),
            chunking_strategy="semantic"
        )

        # Add chunks with profiled extraction
        extractor.extract_entities_batch(
            chunks,
            collector,
            content_profile=profile
        )

    # Summary
    print(f"\n{'=' * 40}")
    print("Chunking Summary")
    print(f"{'=' * 40}")
    print(f"Total parent chunks: {len(all_parents)}")
    print(f"Total child chunks: {len(all_children)}")

    episode_parents = len([p for p in all_parents if p.source_type == "episode"])
    book_parents = len([p for p in all_parents if p.source_type == "book"])
    print(f"  - Episode parents: {episode_parents}")
    print(f"  - Book parents: {book_parents}")

    total_tokens = sum(p.token_count for p in all_parents)
    print(f"Total tokens: {total_tokens:,}")

    # Finalize JSONL files
    jsonl_files = collector.finalize()
    print(f"\nCreated {len(jsonl_files)} JSONL file(s):")
    for f in jsonl_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    # Save child chunks for later vector indexing
    children_path = batch_dir / "child_chunks.json"
    with open(children_path, 'w') as f:
        json.dump([{
            "id": c.id,
            "parent_id": c.parent_id,
            "content": c.content,
            "start_offset": c.start_offset,
            "end_offset": c.end_offset,
            "metadata": c.metadata
        } for c in all_children], f)
    print(f"\nChild chunks saved to: {children_path}")

    # Save parent chunks for reference (with full content for retries)
    parents_path = batch_dir / "parent_chunks.json"
    with open(parents_path, 'w') as f:
        json.dump([{
            "id": p.id,
            "source_type": p.source_type,
            "source_id": p.source_id,
            "token_count": p.token_count,
            "content": p.content,  # Full content for retries
            "content_preview": p.content[:200] + "..." if len(p.content) > 200 else p.content,
            "metadata": p.metadata
        } for p in all_parents], f, indent=2)
    print(f"Parent chunks saved to: {parents_path}")

    if dry_run:
        print("\n[DRY RUN] JSONL files created but not submitted to API")
        print("To submit, run without --dry-run flag")
        return

    # Submit batch jobs
    print("\nSubmitting batch jobs to OpenAI...")
    batch_ids = collector.submit_all_batches()

    # Save state for later polling/downloading
    state = {
        "timestamp": datetime.now().isoformat(),
        "episode_range": episode_range,
        "batch_ids": batch_ids,
        "parent_count": len(all_parents),
        "child_count": len(all_children),
        "total_tokens": total_tokens,
        "jsonl_files": [str(f) for f in jsonl_files],
        "episodes_processed": episodes_processed,
        "books_processed": books_processed,
        "include_episodes": include_episodes,
        "include_books": include_books
    }
    state_path = batch_dir / "batch_state.json"
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n{'=' * 40}")
    print("Submission Complete")
    print(f"{'=' * 40}")
    print(f"Submitted {len(batch_ids)} batch job(s):")
    for bid in batch_ids:
        print(f"  - {bid}")
    print(f"\nState saved to: {state_path}")
    print("\nUse --poll to check status")


def poll_status():
    """Check batch job status."""
    state_path = Path("data/batch_jobs/batch_state.json")
    if not state_path.exists():
        print("No batch state found. Run with --submit first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    collector = BatchCollector(output_dir=Path("data/batch_jobs"))
    collector.batch_ids = state["batch_ids"]

    print("=" * 60)
    print("Batch Status")
    print("=" * 60)
    print(f"Submitted: {state['timestamp']}")
    print(f"Episodes: {state.get('episodes_processed', 'N/A')}")
    print(f"Books: {state.get('books_processed', 'N/A')}")
    print(f"Parent chunks: {state['parent_count']}")
    print(f"Child chunks: {state['child_count']}")
    print()

    all_done = True
    all_completed = True

    for batch_id in state["batch_ids"]:
        status = collector.poll_batch(batch_id)
        print(f"Batch: {batch_id}")
        print(f"  Status: {status.status}")
        print(f"  Progress: {status.completed}/{status.total} ({status.progress_percent:.1f}%)")
        if status.failed > 0:
            print(f"  Failed: {status.failed}")
        print()

        if not status.is_done:
            all_done = False
        if status.status != "completed":
            all_completed = False

    if all_completed:
        print("✅ All batches completed successfully!")
        print("Use --download to retrieve results.")
    elif all_done:
        print("⚠️ All batches finished but some may have failures.")
        print("Use --download to retrieve available results.")
    else:
        print("⏳ Batches still processing. Check again later.")


def download_results():
    """Download and process batch results, capturing failures for retry."""
    state_path = Path("data/batch_jobs/batch_state.json")
    if not state_path.exists():
        print("No batch state found. Run with --submit first.")
        return

    with open(state_path) as f:
        state = json.load(f)

    collector = BatchCollector(output_dir=Path("data/batch_jobs"))
    collector.batch_ids = state["batch_ids"]
    extractor = EntityExtractor()

    print("=" * 60)
    print("Downloading Results")
    print("=" * 60)
    print()

    # Check status first
    incomplete = []
    for batch_id in state["batch_ids"]:
        status = collector.poll_batch(batch_id)
        if status.status != "completed":
            incomplete.append((batch_id, status.status))

    if incomplete:
        print("⚠️ Some batches not yet completed:")
        for bid, status in incomplete:
            print(f"  - {bid}: {status}")
        print("\nDownloading available completed batches...")

    # Download all completed results AND capture failures
    all_results = collector.download_all_results()
    failed_chunks = _capture_failures(state["batch_ids"], collector)

    # Load content metadata to enrich results
    metadata_path = Path("data/batch_jobs/content_metadata.json")
    content_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            content_metadata = json.load(f)

    # Process results
    all_extractions = {}
    total_entities = 0
    total_relationships = 0

    for batch_id, results in all_results.items():
        print(f"Processing batch {batch_id}: {len(results)} results")
        extractions = extractor.process_batch_results(results)

        # Enrich with content metadata
        for chunk_id, extraction in extractions.items():
            meta = content_metadata.get(chunk_id, {})
            # Add profile info to each entity
            for entity in extraction.entities:
                entity.metadata["content_type"] = meta.get("content_type", "episode")
                entity.metadata["reality_tag"] = meta.get("reality_tag", "factual")

        all_extractions.update(extractions)

        for extraction in extractions.values():
            total_entities += len(extraction.entities)
            total_relationships += len(extraction.relationships)

    print(f"\n{'=' * 40}")
    print("Extraction Summary")
    print(f"{'=' * 40}")
    print(f"Total chunks processed: {len(all_extractions)}")
    print(f"Total entities extracted: {total_entities}")
    print(f"Total relationships extracted: {total_relationships}")

    # Save raw results
    results_dir = Path("data/batch_jobs/results")
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / "extraction_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            chunk_id: {
                "entities": [e.model_dump() for e in result.entities],
                "relationships": [r.model_dump() for r in result.relationships],
                "chunk_id": result.chunk_id,
                "episode_number": result.episode_number
            }
            for chunk_id, result in all_extractions.items()
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save entity and relationship counts per source
    source_stats: Dict[str, Dict[str, int]] = {}
    for chunk_id, result in all_extractions.items():
        # Parse source from chunk ID
        parts = chunk_id.split("_")
        if parts[0] == "episode":
            source_key = f"episode_{parts[1]}"
        elif parts[0] == "book":
            source_key = f"book_{parts[1]}"
        else:
            source_key = "unknown"

        if source_key not in source_stats:
            source_stats[source_key] = {"entities": 0, "relationships": 0, "chunks": 0}

        source_stats[source_key]["entities"] += len(result.entities)
        source_stats[source_key]["relationships"] += len(result.relationships)
        source_stats[source_key]["chunks"] += 1

    stats_path = results_dir / "source_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(source_stats, f, indent=2)
    print(f"Source statistics saved to: {stats_path}")

    # Report on failures
    if failed_chunks:
        print(f"\n{'=' * 40}")
        print(f"⚠️  FAILURES DETECTED: {len(failed_chunks)} chunks")
        print(f"{'=' * 40}")
        for chunk in failed_chunks:
            print(f"  - {chunk['chunk_id']}: {chunk['error'][:60]}...")
        print(f"\nFailed chunks saved to: data/batch_jobs/failed_chunks.json")
        print("Run with --retry-failed to resubmit these chunks.")
    else:
        print(f"\n✅ All chunks processed successfully!")

    print(f"\n{'=' * 40}")
    print("Next Steps")
    print(f"{'=' * 40}")
    if failed_chunks:
        print("1. Run --retry-failed to reprocess failed chunks")
        print("2. After retry completes, run --download again to merge")
        print("3. Then proceed to Phase 8 pipeline")
    else:
        print("1. Run quality filters (Phases 1-6) on results")
        print("2. Run entity resolver for deduplication")
        print("3. Build unified knowledge graph")
        print("\nExample:")
        print("  python scripts/run_phase8_pipeline.py --skip-download")

    # Auto-merge if retry exists and is complete
    retry_state_path = Path("data/batch_jobs/retry/retry_state.json")
    if retry_state_path.exists():
        print("\n" + "=" * 60)
        print("Checking for retry results to merge...")
        download_and_merge()


def _capture_failures(batch_ids: List[str], collector: BatchCollector) -> List[Dict]:
    """
    Capture failed chunks from batch error files.

    Args:
        batch_ids: List of batch IDs to check
        collector: BatchCollector instance

    Returns:
        List of failed chunk info dicts with chunk_id and error
    """
    failed_chunks = []
    batch_dir = Path("data/batch_jobs")

    for batch_id in batch_ids:
        try:
            batch = collector.client.batches.retrieve(batch_id)

            if batch.error_file_id:
                content = collector.client.files.content(batch.error_file_id)

                for line in content.text.strip().split('\n'):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk_id = data.get('custom_id', 'unknown')
                        error_msg = data.get('response', {}).get('body', {}).get('error', {}).get('message', 'Unknown error')

                        failed_chunks.append({
                            "chunk_id": chunk_id,
                            "batch_id": batch_id,
                            "error": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"Warning: Could not check errors for batch {batch_id}: {e}")

    # Save failed chunks for retry
    if failed_chunks:
        failed_path = batch_dir / "failed_chunks.json"
        with open(failed_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "count": len(failed_chunks),
                "chunks": failed_chunks
            }, f, indent=2)

    return failed_chunks


def retry_failed():
    """Retry only the failed chunks from a previous batch run."""
    batch_dir = Path("data/batch_jobs")
    failed_path = batch_dir / "failed_chunks.json"

    if not failed_path.exists():
        print("No failed chunks found. Nothing to retry.")
        print("Run --download first to capture any failures.")
        return

    with open(failed_path) as f:
        failed_data = json.load(f)

    failed_chunks = failed_data.get("chunks", [])
    if not failed_chunks:
        print("No failed chunks to retry.")
        return

    print("=" * 60)
    print("Retrying Failed Chunks")
    print("=" * 60)
    print(f"Found {len(failed_chunks)} failed chunks to retry:")
    for chunk in failed_chunks:
        print(f"  - {chunk['chunk_id']}")
    print()

    # Load parent chunks with full content
    parents_path = batch_dir / "parent_chunks.json"
    if not parents_path.exists():
        print("ERROR: parent_chunks.json not found. Cannot retry.")
        return

    with open(parents_path) as f:
        parent_data = json.load(f)

    # Build lookup by ID
    parent_lookup = {p["id"]: p for p in parent_data}

    # Load content metadata for profiles
    metadata_path = batch_dir / "content_metadata.json"
    content_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            content_metadata = json.load(f)

    # Reconstruct parent chunks
    failed_chunk_ids = {c["chunk_id"] for c in failed_chunks}
    retry_parents: List[ParentChunk] = []

    for chunk_id in failed_chunk_ids:
        parent_info = parent_lookup.get(chunk_id)

        # Check if we have cached content
        content = parent_info.get("content") if parent_info else None

        if content and len(content) >= 100:
            # Use cached content
            retry_parents.append(ParentChunk(
                id=chunk_id,
                content=content,
                source_type=parent_info["source_type"],
                source_id=parent_info["source_id"],
                token_count=parent_info["token_count"],
                metadata=parent_info.get("metadata", {})
            ))
            print(f"  Using cached content for {chunk_id}")
        else:
            # Need to reload from source - parse chunk_id to get source info
            # Format: "episode_24_parent_0" or "book_viriditas_parent_0"
            parts = chunk_id.split("_")
            source_type = parts[0]  # "episode" or "book"

            if source_type == "episode":
                source_id = parts[1]  # episode number
                transcript_path = Path(f"data/transcripts/episode_{source_id}.json")
                if transcript_path.exists():
                    with open(transcript_path) as f:
                        episode_data = json.load(f)
                    chunker = ParentChildChunker()
                    parents, _ = chunker.process_content(
                        text=episode_data.get("full_transcript", ""),
                        source_type="episode",
                        source_id=source_id
                    )
                    for p in parents:
                        if p.id == chunk_id:
                            retry_parents.append(p)
                            print(f"  Reloaded from transcript: {chunk_id}")
                            break
                    else:
                        print(f"  Warning: Chunk {chunk_id} not found in transcript")
                else:
                    print(f"  Warning: Transcript not found for episode {source_id}")

            elif source_type == "book":
                # Book slug is between "book_" and "_parent_"
                source_id = "_".join(parts[1:-2])  # Handle slugs with underscores
                book_dir = Path(f"data/books/{source_id}")
                pdfs = list(book_dir.glob("*.pdf"))
                if pdfs:
                    pdf_extractor = EnhancedPDFExtractor(pdfs[0])
                    text, _ = pdf_extractor.extract_text()
                    chunker = ParentChildChunker()
                    parents, _ = chunker.process_content(
                        text=text,
                        source_type="book",
                        source_id=source_id
                    )
                    for p in parents:
                        if p.id == chunk_id:
                            retry_parents.append(p)
                            print(f"  Reloaded from PDF: {chunk_id}")
                            break
                    else:
                        print(f"  Warning: Chunk {chunk_id} not found in book")
                else:
                    print(f"  Warning: PDF not found for book {source_id}")
            else:
                print(f"  Warning: Unknown source type in {chunk_id}")

    if not retry_parents:
        print("ERROR: Could not reconstruct any parent chunks for retry.")
        return

    print(f"Reconstructed {len(retry_parents)} parent chunks for retry")

    # Create retry batch
    retry_dir = batch_dir / "retry"
    retry_dir.mkdir(exist_ok=True)

    collector = BatchCollector(output_dir=retry_dir)
    extractor = EntityExtractor()

    # Add with appropriate profiles
    for p in retry_parents:
        meta = content_metadata.get(p.id, {})
        profile = ContentProfile(
            content_type=meta.get("content_type", "episode"),
            reality_tag=meta.get("reality_tag", "factual"),
            system_prompt_focus=meta.get("system_prompt_focus", ""),
            chunking_strategy="semantic"
        )
        extractor.extract_entities_batch([p], collector, content_profile=profile)

    jsonl_files = collector.finalize()
    print(f"\nCreated retry JSONL: {jsonl_files[0].name}")

    # Submit retry batch
    print("\nSubmitting retry batch to OpenAI...")
    batch_ids = collector.submit_all_batches()

    # Save retry state
    retry_state = {
        "timestamp": datetime.now().isoformat(),
        "original_failed_count": len(failed_chunks),
        "retry_count": len(retry_parents),
        "batch_ids": batch_ids,
        "chunk_ids": [p.id for p in retry_parents]
    }
    retry_state_path = retry_dir / "retry_state.json"
    with open(retry_state_path, 'w') as f:
        json.dump(retry_state, f, indent=2)

    print(f"\n{'=' * 40}")
    print("Retry Submitted")
    print(f"{'=' * 40}")
    print(f"Batch ID: {batch_ids[0]}")
    print(f"Chunks: {len(retry_parents)}")
    print(f"\nState saved to: {retry_state_path}")
    print("\nUse --poll-retry to check status")
    print("When complete, use --download to merge results")


def poll_retry():
    """Check status of retry batch."""
    retry_state_path = Path("data/batch_jobs/retry/retry_state.json")

    if not retry_state_path.exists():
        print("No retry batch found. Run --retry-failed first.")
        return

    with open(retry_state_path) as f:
        retry_state = json.load(f)

    collector = BatchCollector(output_dir=Path("data/batch_jobs/retry"))
    collector.batch_ids = retry_state["batch_ids"]

    print("=" * 60)
    print("Retry Batch Status")
    print("=" * 60)
    print(f"Submitted: {retry_state['timestamp']}")
    print(f"Chunks to retry: {retry_state['retry_count']}")
    print()

    for batch_id in retry_state["batch_ids"]:
        status = collector.poll_batch(batch_id)
        print(f"Batch: {batch_id}")
        print(f"  Status: {status.status}")
        print(f"  Progress: {status.completed}/{status.total} ({status.progress_percent:.1f}%)")
        if status.failed > 0:
            print(f"  Failed: {status.failed}")

        if status.status == "completed":
            print("\n✅ Retry batch complete!")
            print("Run --download to merge results into main extraction.")
        elif status.is_done:
            print(f"\n⚠️ Retry batch finished with status: {status.status}")
        else:
            print("\n⏳ Still processing...")


def download_and_merge():
    """Download retry results and merge with main extraction."""
    retry_state_path = Path("data/batch_jobs/retry/retry_state.json")
    results_path = Path("data/batch_jobs/results/extraction_results.json")

    if not retry_state_path.exists():
        print("No retry batch to merge.")
        return False

    with open(retry_state_path) as f:
        retry_state = json.load(f)

    collector = BatchCollector(output_dir=Path("data/batch_jobs/retry"))
    collector.batch_ids = retry_state["batch_ids"]
    extractor = EntityExtractor()

    # Check if retry is complete
    for batch_id in retry_state["batch_ids"]:
        status = collector.poll_batch(batch_id)
        if not status.is_done:
            print(f"Retry batch {batch_id} not yet complete (status: {status.status})")
            return False

    print("=" * 60)
    print("Merging Retry Results")
    print("=" * 60)

    # Download retry results
    all_results = collector.download_all_results()

    retry_extractions = {}
    for batch_id, results in all_results.items():
        extractions = extractor.process_batch_results(results)
        retry_extractions.update(extractions)

    print(f"Downloaded {len(retry_extractions)} retry results")

    # Load existing results
    if results_path.exists():
        with open(results_path) as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing results")
    else:
        existing_results = {}

    # Merge: retry results overwrite any existing (in case of re-retries)
    for chunk_id, result in retry_extractions.items():
        existing_results[chunk_id] = {
            "entities": [e.model_dump() for e in result.entities],
            "relationships": [r.model_dump() for r in result.relationships],
            "chunk_id": result.chunk_id,
            "episode_number": result.episode_number,
            "from_retry": True
        }

    # Save merged results
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=2)

    print(f"\n✅ Merged! Total chunks: {len(existing_results)}")
    print(f"Results saved to: {results_path}")

    # Clear failed chunks file since we've recovered
    failed_path = Path("data/batch_jobs/failed_chunks.json")
    if failed_path.exists():
        # Check if any new failures in retry
        new_failures = _capture_failures(retry_state["batch_ids"], collector)
        if not new_failures:
            failed_path.unlink()
            print("Cleared failed_chunks.json - all chunks recovered!")
        else:
            print(f"⚠️ {len(new_failures)} chunks still failed after retry")

    return True


def show_status():
    """Show detailed status of extraction including any gaps."""
    batch_dir = Path("data/batch_jobs")
    state_path = batch_dir / "batch_state.json"
    results_path = batch_dir / "results" / "extraction_results.json"
    failed_path = batch_dir / "failed_chunks.json"
    metadata_path = batch_dir / "content_metadata.json"

    print("=" * 60)
    print("Extraction Status Report")
    print("=" * 60)

    # Check batch state
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        print(f"\nBatch Job:")
        print(f"  Submitted: {state['timestamp']}")
        print(f"  Expected chunks: {state['parent_count']}")
        print(f"  Episodes: {state.get('episodes_processed', 'N/A')}")
        print(f"  Books: {state.get('books_processed', 'N/A')}")
    else:
        print("\n❌ No batch state found. Run --submit first.")
        return

    # Check content profiles
    if metadata_path.exists():
        with open(metadata_path) as f:
            content_metadata = json.load(f)

        profile_counts: Dict[str, int] = {}
        for chunk_id, meta in content_metadata.items():
            profile_key = f"{meta.get('content_type', 'unknown')}/{meta.get('reality_tag', 'unknown')}"
            profile_counts[profile_key] = profile_counts.get(profile_key, 0) + 1

        print(f"\nContent Profiles:")
        for profile, count in sorted(profile_counts.items()):
            print(f"  {profile}: {count} chunks")

    # Check results
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        total_entities = sum(len(r.get("entities", [])) for r in results.values())
        total_rels = sum(len(r.get("relationships", [])) for r in results.values())

        print(f"\nResults:")
        print(f"  Chunks extracted: {len(results)}")
        print(f"  Total entities: {total_entities}")
        print(f"  Total relationships: {total_rels}")

        # Check coverage
        expected = state['parent_count']
        actual = len(results)
        coverage = (actual / expected) * 100 if expected > 0 else 0
        print(f"  Coverage: {actual}/{expected} ({coverage:.1f}%)")

        if coverage < 100:
            print(f"  ⚠️ Missing {expected - actual} chunks")
    else:
        print("\n❌ No results yet. Run --download after batch completes.")

    # Check failures
    if failed_path.exists():
        with open(failed_path) as f:
            failed_data = json.load(f)
        failed_chunks = failed_data.get("chunks", [])

        if failed_chunks:
            print(f"\n⚠️ Failed Chunks ({len(failed_chunks)}):")

            # Group by source
            by_source: Dict[str, List[str]] = {}
            for chunk in failed_chunks:
                chunk_id = chunk["chunk_id"]
                parts = chunk_id.split("_")
                if parts[0] == "episode":
                    key = f"Episode {parts[1]}"
                elif parts[0] == "book":
                    key = f"Book: {parts[1]}"
                else:
                    key = "Unknown"

                if key not in by_source:
                    by_source[key] = []
                by_source[key].append(chunk_id)

            for source, chunks in sorted(by_source.items()):
                print(f"  {source}: {len(chunks)} chunk(s)")
                for c in chunks:
                    print(f"    - {c}")

            print(f"\nRun --retry-failed to resubmit these chunks.")

    # Check retry status
    retry_state_path = batch_dir / "retry" / "retry_state.json"
    if retry_state_path.exists():
        with open(retry_state_path) as f:
            retry_state = json.load(f)
        print(f"\nRetry Batch:")
        print(f"  Submitted: {retry_state['timestamp']}")
        print(f"  Chunks: {retry_state['retry_count']}")
        print("  Run --poll-retry to check status")


def main():
    parser = argparse.ArgumentParser(
        description="Unified content extraction for YonEarth Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit batch job for all content (episodes + books)
  python scripts/extract_content_batch.py --submit

  # Submit only books (with content profiles)
  python scripts/extract_content_batch.py --submit-books

  # Submit for specific episode range
  python scripts/extract_content_batch.py --submit --episodes 0-50

  # Dry run (create files without submitting)
  python scripts/extract_content_batch.py --submit --dry-run

  # Check batch status
  python scripts/extract_content_batch.py --poll

  # Download results when complete
  python scripts/extract_content_batch.py --download

  # View detailed status report
  python scripts/extract_content_batch.py --status

  # Retry failed chunks
  python scripts/extract_content_batch.py --retry-failed

  # Check retry batch status
  python scripts/extract_content_batch.py --poll-retry

Content Profiles:
  - Episodes: Standard entity extraction (factual)
  - Soil Stewardship Handbook: Technical/How-To focus (factual)
  - VIRIDITAS: Fictional narrative focus (fictional)
  - Y on Earth / Our Biggest Deal: Rhetorical/conceptual focus (conceptual)
        """
    )

    parser.add_argument("--submit", action="store_true", help="Submit batch job for all content")
    parser.add_argument("--submit-books", action="store_true", help="Submit batch job for books only")
    parser.add_argument("--submit-episodes", action="store_true", help="Submit batch job for episodes only")
    parser.add_argument("--poll", action="store_true", help="Check batch status")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--status", action="store_true", help="Show detailed status report")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed chunks")
    parser.add_argument("--poll-retry", action="store_true", help="Check retry batch status")
    parser.add_argument("--merge", action="store_true", help="Merge retry results into main extraction")
    parser.add_argument("--episodes", default="0-172",
                        help="Episode range to process (default: 0-172)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create JSONL files but don't submit to API")

    args = parser.parse_args()

    if args.submit:
        submit_batch(episode_range=args.episodes, include_episodes=True, include_books=True, dry_run=args.dry_run)
    elif args.submit_books:
        submit_batch(episode_range=args.episodes, include_episodes=False, include_books=True, dry_run=args.dry_run)
    elif args.submit_episodes:
        submit_batch(episode_range=args.episodes, include_episodes=True, include_books=False, dry_run=args.dry_run)
    elif args.poll:
        poll_status()
    elif args.download:
        download_results()
    elif args.status:
        show_status()
    elif args.retry_failed:
        retry_failed()
    elif args.poll_retry:
        poll_retry()
    elif args.merge:
        download_and_merge()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
