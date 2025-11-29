#!/usr/bin/env python3
"""
Build MemoRAG Memory Index for "Our Biggest Deal"

This script extracts text from the book PDF and builds a MemoRAG memory index
for efficient long-context question answering.

The text extraction uses pdfplumber (same as ACE pipeline) to ensure clean text.
"""

import argparse
import json
import pickle
import sys
import os
from pathlib import Path
from typing import List, Dict

import pdfplumber
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_book_text(pdf_path: Path) -> str:
    """
    Extract text from PDF using pdfplumber (same method as ACE pipeline).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Full book text as string
    """
    print(f"📖 Extracting text from: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n\n".join(
            page.extract_text() or ""
            for page in pdf.pages
        )
        total_pages = len(pdf.pages)

    print(f"   ✅ Extracted {len(full_text):,} characters from {total_pages} pages")
    return full_text


def chunk_text_by_chapters(text: str) -> List[Dict[str, str]]:
    """
    Attempt to split text into chapters while preserving structure.

    This helps MemoRAG understand the book's organization, even though
    it can handle long context natively.

    Args:
        text: Full book text

    Returns:
        List of chapter dicts with 'chapter_num', 'title', 'text'
    """
    # Simple chapter detection - look for "Chapter N" or "CHAPTER N" patterns
    import re

    # Pattern to match chapter headers
    chapter_pattern = r'(?:Chapter|CHAPTER)\s+(\d+|[IVX]+)(?:\s*[:\-]\s*(.+?))?(?=\n)'

    matches = list(re.finditer(chapter_pattern, text))

    if not matches:
        print("   ⚠️  No chapter markers found, treating as single document")
        return [{'chapter_num': 1, 'title': 'Full Book', 'text': text}]

    chunks = []
    for i, match in enumerate(matches):
        chapter_num = match.group(1)
        chapter_title = match.group(2) or ""

        # Get text from this chapter to next chapter (or end)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = text[start:end].strip()

        chunks.append({
            'chapter_num': chapter_num,
            'title': chapter_title.strip(),
            'text': chapter_text
        })

    print(f"   ✅ Split into {len(chunks)} chapters")
    return chunks


def build_memorag_index(
    text: str,
    model_name: str = "memorag-qwen2-7b-inst",
    use_openai_generation: bool = True,
    cache_dir: str = None
):
    """
    Build MemoRAG memory index from book text with hybrid architecture.

    HYBRID ARCHITECTURE:
    - Local retrieval: Uses CPU for memory/context retrieval
    - Cloud generation: Uses OpenAI API (GPT-4o-mini) for answer generation
    This avoids 30s+ per-query latency on CPU-only servers.

    Args:
        text: Full book text (or concatenated chapters)
        model_name: MemoRAG memory model to use for retrieval
        use_openai_generation: If True, use OpenAI for generation (recommended)
        cache_dir: Directory to cache models

    Returns:
        MemoRAG pipeline object with memorized content
    """
    try:
        from memorag import MemoRAG, Agent
    except ImportError:
        print("❌ MemoRAG not installed. Installing now...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "memorag", "torch", "transformers"
        ])
        from memorag import MemoRAG, Agent

    print(f"\n🧠 Initializing MemoRAG (Hybrid Architecture)")
    print(f"   Memory Model (Local CPU): {model_name}")

    # Configure generation model
    customized_gen_model = None
    if use_openai_generation:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("   ⚠️  Warning: OPENAI_API_KEY not found in environment")
            print("   ⚠️  Falling back to local generation (will be slow)")
        else:
            print(f"   Generation Model (Cloud): gpt-4o-mini (OpenAI)")
            customized_gen_model = Agent(
                model="gpt-4o-mini",
                source="openai",
                api_dict={"api_key": openai_key}
            )

    cache_path = cache_dir or str(Path(__file__).parent.parent / "indices" / "model_cache")

    # Initialize pipeline with hybrid setup
    #  MemoRAG requires both memory and retrieval models
    # Use same model for both memory and retrieval for simplicity
    pipe = MemoRAG(
        mem_model_name_or_path=model_name,
        ret_model_name_or_path=model_name,  # Use same model for retrieval
        cache_dir=cache_path,
        customized_gen_model=customized_gen_model,
        enable_flash_attn=False  # Disable flash attention for CPU
    )

    print(f"   ✅ Pipeline initialized")
    print(f"\n📝 Memorizing book content ({len(text):,} characters)...")
    print(f"   This may take several minutes depending on hardware...")

    # Memorize the content
    pipe.memorize(text)

    print(f"   ✅ Memory index built successfully!")

    return pipe


def save_pipeline(pipe, output_path: Path):
    """Save MemoRAG pipeline to disk."""
    print(f"\n💾 Saving pipeline to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(pipe, f)

    print(f"   ✅ Pipeline saved!")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Build MemoRAG memory index for Our Biggest Deal"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "data" / "books" / "OurBiggestDeal" / "our_biggest_deal.pdf",
        help="Path to PDF file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "indices" / "memorag_pipeline.pkl",
        help="Output path for pickled pipeline"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-1.5B-Instruct",  # Using smaller public model for CPU
        help="MemoRAG memory/retrieval model name (use smaller models for CPU)"
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Disable OpenAI generation (use local model, will be slow)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Model cache directory"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 MemoRAG Memory Builder - Our Biggest Deal")
    print("=" * 60)

    # Check if PDF exists
    if not args.pdf.exists():
        print(f"❌ PDF not found: {args.pdf}")
        print(f"   Please ensure the book PDF is at the correct location.")
        sys.exit(1)

    # Extract text
    text = extract_book_text(args.pdf)

    # Optional: Split by chapters for better context
    chapters = chunk_text_by_chapters(text)

    # Concatenate chapters with clear markers
    formatted_text = "\n\n".join(
        f"### Chapter {ch['chapter_num']}: {ch['title']}\n\n{ch['text']}"
        for ch in chapters
    )

    # Build MemoRAG index
    try:
        pipe = build_memorag_index(
            text=formatted_text,
            model_name=args.model,
            use_openai_generation=not args.no_openai,
            cache_dir=str(args.cache_dir) if args.cache_dir else None
        )

        # Save pipeline
        save_pipeline(pipe, args.output)

        print("\n" + "=" * 60)
        print("✅ SUCCESS! MemoRAG memory index is ready.")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Test the index with query_memory.py")
        print(f"  2. Example query:")
        print(f'     python experiments/memorag/scripts/query_memory.py \\')
        print(f'       "What is the narrative arc of Planetary Prosperity?"')

    except Exception as e:
        print(f"\n❌ Error building MemoRAG index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
