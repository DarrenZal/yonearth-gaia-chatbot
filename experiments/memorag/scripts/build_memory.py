#!/usr/bin/env python3
"""
Build MemoRAG Memory Index for "Our Biggest Deal" - SHARDED ARCHITECTURE

This script builds memory shards for each chunk of the book, enabling
MemoRAG's full reasoning capabilities across the entire book content.

Each chunk gets its own memory shard in indices/shard_{i}/ containing:
- memory.bin: MemoRAG memory embeddings
- index.bin: FAISS retrieval index
- chunks.json: Text chunks for retrieval
"""

import argparse
import json
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

# Configuration
MAX_CHUNK_CHARS = 80000  # ~20K tokens for Qwen2's 32K context window


def extract_book_text(pdf_path: Path) -> str:
    """
    Extract text from PDF using pdfplumber.
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


def split_into_chunks(text: str, max_chunk_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split text into chunks that fit within the model's context window.
    Breaks at paragraph boundaries for coherence.
    """
    chunks = []
    start = 0
    total_chars = len(text)

    while start < total_chars:
        end = min(start + max_chunk_chars, total_chars)

        # Try to break at paragraph boundaries
        if end < total_chars:
            # Look for paragraph break within last 5000 chars
            break_point = text.rfind('\n\n', end - 5000, end)
            if break_point > start:
                end = break_point + 2  # Include the newlines

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def create_shard_metadata(chunks: List[str], save_dir: Path) -> Dict:
    """
    Create metadata file describing all shards.
    """
    metadata = {
        "num_shards": len(chunks),
        "total_chars": sum(len(c) for c in chunks),
        "shards": []
    }

    for i, chunk in enumerate(chunks):
        # Extract a preview (first 200 chars)
        preview = chunk[:200].replace('\n', ' ')

        # Extract key terms for routing (simple word frequency)
        words = chunk.lower().split()
        word_freq = {}
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1

        top_terms = sorted(word_freq.items(), key=lambda x: -x[1])[:20]

        metadata["shards"].append({
            "index": i,
            "path": f"shard_{i}",
            "char_count": len(chunk),
            "preview": preview,
            "top_terms": [term for term, count in top_terms]
        })

    # Save metadata
    metadata_path = save_dir / "shard_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   📋 Saved shard metadata to: {metadata_path}")
    return metadata


def build_memorag_shards(
    chunks: List[str],
    model_name: str = "Qwen/Qwen2-1.5B-Instruct",
    use_openai_generation: bool = True,
    cache_dir: str = None,
    indices_dir: str = None
):
    """
    Build MemoRAG memory shards - one per chunk.

    SHARDING ARCHITECTURE:
    - Each chunk gets its own memory shard in indices/shard_{i}/
    - Avoids the overwrite bug in memorize()
    - Enables full book coverage with MemoRAG's reasoning

    Args:
        chunks: List of text chunks to memorize
        model_name: MemoRAG memory model
        use_openai_generation: Use OpenAI for generation (faster)
        cache_dir: Model cache directory
        indices_dir: Directory to save shard subdirectories
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

    indices_path = Path(indices_dir)
    indices_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🧠 Building MemoRAG Shards (Sharded Architecture)")
    print(f"   Memory Model: {model_name}")
    print(f"   Number of shards: {len(chunks)}")

    # Configure generation model
    customized_gen_model = None
    if use_openai_generation:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("   ⚠️  Warning: OPENAI_API_KEY not found")
            print("   ⚠️  Falling back to local generation (will be slow)")
        else:
            print(f"   Generation Model: gpt-4.1-mini (OpenAI)")
            customized_gen_model = Agent(
                model="gpt-4.1-mini",
                source="openai",
                api_dict={"api_key": openai_key}
            )

    cache_path = cache_dir or str(indices_path.parent / "model_cache")

    # Initialize pipeline ONCE (model loading is slow)
    print(f"\n   🔄 Loading Qwen model (this takes a few minutes)...")
    pipe = MemoRAG(
        mem_model_name_or_path=model_name,
        ret_model_name_or_path=model_name,
        cache_dir=cache_path,
        customized_gen_model=customized_gen_model,
        enable_flash_attn=False,
        load_in_4bit=False
    )
    print(f"   ✅ Pipeline initialized")

    # Create shard metadata
    create_shard_metadata(chunks, indices_path)

    # Build each shard
    print(f"\n📝 Building {len(chunks)} memory shards...")

    for i, chunk in enumerate(chunks):
        shard_dir = indices_path / f"shard_{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n   🧩 Shard {i}/{len(chunks)-1} ({len(chunk):,} chars)")
        print(f"      Preview: {chunk[:80].replace(chr(10), ' ')}...")

        try:
            # Memorize this chunk to its own shard directory
            pipe.memorize(chunk, save_dir=str(shard_dir), print_stats=True)

            # Verify files were created
            expected_files = ["memory.bin", "index.bin", "chunks.json"]
            missing = [f for f in expected_files if not (shard_dir / f).exists()]

            if missing:
                print(f"      ⚠️  Missing files: {missing}")
            else:
                # Check file sizes
                sizes = {f: (shard_dir / f).stat().st_size for f in expected_files}
                print(f"      ✅ Saved: memory.bin={sizes['memory.bin']//1024}KB, "
                      f"index.bin={sizes['index.bin']//1024}KB, "
                      f"chunks.json={sizes['chunks.json']//1024}KB")

        except Exception as e:
            print(f"      ❌ Error: {e}")
            raise

    print(f"\n✅ Built {len(chunks)} memory shards successfully!")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Build MemoRAG memory shards for Our Biggest Deal"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "data" / "books" / "OurBiggestDeal" / "our_biggest_deal.pdf",
        help="Path to PDF file"
    )
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=Path(__file__).parent.parent / "indices",
        help="Directory to save shard subdirectories"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-1.5B-Instruct",
        help="MemoRAG memory model name"
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Disable OpenAI generation (use local model, slow)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Model cache directory"
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=MAX_CHUNK_CHARS,
        help="Maximum characters per chunk (default: 80000)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 MemoRAG Sharded Memory Builder - Our Biggest Deal")
    print("=" * 60)

    # Check if PDF exists
    if not args.pdf.exists():
        print(f"❌ PDF not found: {args.pdf}")
        sys.exit(1)

    # Extract text
    text = extract_book_text(args.pdf)

    # Split into chunks
    chunks = split_into_chunks(text, args.max_chunk_chars)
    print(f"\n📚 Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: {len(chunk):,} chars")

    # Build shards
    try:
        num_shards = build_memorag_shards(
            chunks=chunks,
            model_name=args.model,
            use_openai_generation=not args.no_openai,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            indices_dir=str(args.indices_dir)
        )

        print("\n" + "=" * 60)
        print("✅ SUCCESS! MemoRAG shards are ready.")
        print("=" * 60)
        print(f"\nShards saved to: {args.indices_dir}")
        print(f"Total shards: {num_shards}")
        print(f"\nNext steps:")
        print(f"  1. Test with query_memory.py")
        print(f"  2. Example query:")
        print(f'     python experiments/memorag/scripts/query_memory.py \\')
        print(f'       "What is Bernard Lietaer\'s vision for the future of money?"')

    except Exception as e:
        print(f"\n❌ Error building MemoRAG shards: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
