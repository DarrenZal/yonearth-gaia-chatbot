#!/usr/bin/env python3
"""
Query MemoRAG Memory Index

This script loads a pre-built MemoRAG memory index and runs queries against it.
Uses native MemoRAG load for robust memory loading.

Usage:
    python query_memory.py "Your question here"
    python query_memory.py --interactive  # Interactive mode
    python query_memory.py --memory-dir path/to/indices  # Custom memory location
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()


def load_pipeline(memory_dir: Path, model_name: str = "Qwen/Qwen2-1.5B-Instruct"):
    """Load saved MemoRAG pipeline from disk using native load.

    Args:
        memory_dir: Directory containing memory index files (memory.bin, etc.)
        model_name: Model name to use for the pipeline

    Returns:
        MemoRAG pipeline object
    """
    if not memory_dir.exists():
        print(f"❌ Memory directory not found: {memory_dir}")
        print(f"\n   Please run build_memory.py first to create the index:")
        print(f"   python experiments/memorag/scripts/build_memory.py")
        sys.exit(1)

    # Check for expected files
    expected_files = ["memory.bin"]  # MemoRAG creates these
    found_files = list(memory_dir.glob("*.bin")) + list(memory_dir.glob("*.json"))

    if not found_files:
        print(f"❌ No memory files found in: {memory_dir}")
        print(f"   Expected files like memory.bin")
        print(f"\n   Please run build_memory.py first to create the index:")
        print(f"   python experiments/memorag/scripts/build_memory.py")
        sys.exit(1)

    print(f"📥 Loading MemoRAG memory from: {memory_dir}")
    print(f"   Found files: {[f.name for f in found_files]}")

    # Import MemoRAG and configure hybrid generation
    from memorag import MemoRAG, Agent

    # Configure OpenAI generation if available
    openai_key = os.getenv("OPENAI_API_KEY")
    customized_gen_model = None
    if openai_key:
        print(f"   🌐 Hybrid Mode: Local retrieval + OpenAI generation (fast)")
        customized_gen_model = Agent(
            model="gpt-4.1-mini",
            source="openai",
            api_dict={"api_key": openai_key}
        )
    else:
        print(f"   💻 Local Mode: CPU-only generation (will be slow)")

    # Initialize pipeline and load memory
    cache_path = str(memory_dir.parent / "model_cache")
    pipe = MemoRAG(
        mem_model_name_or_path=model_name,
        ret_model_name_or_path=model_name,
        cache_dir=cache_path,
        customized_gen_model=customized_gen_model,
        enable_flash_attn=False,
        load_in_4bit=False
    )

    # Load the memory from saved directory
    pipe.load(str(memory_dir))

    print(f"   ✅ Pipeline loaded successfully!")

    return pipe


def query_memory(pipe, question: str, context_length: int = 2000, top_k: int = 5):
    """
    Query the MemoRAG memory.

    Args:
        pipe: MemoRAG pipeline
        question: User question
        context_length: Maximum context length for retrieval
        top_k: Number of relevant passages to retrieve

    Returns:
        Answer string
    """
    print(f"\n🔍 Query: {question}")
    print(f"   Retrieving relevant context (top_k={top_k})...")

    # Query the memorized content
    answer = pipe.query(
        question,
        context_length=context_length,
        top_k=top_k
    )

    return answer


def interactive_mode(pipe):
    """Run interactive query session."""
    print("\n" + "=" * 60)
    print("🧠 MemoRAG Interactive Mode - Our Biggest Deal")
    print("=" * 60)
    print("\nType your questions below. Type 'quit' or 'exit' to stop.")
    print("Commands:")
    print("  /context <n>  - Set context length (default: 2000)")
    print("  /topk <n>     - Set top_k passages (default: 5)")
    print("=" * 60)

    context_length = 2000
    top_k = 5

    while True:
        try:
            question = input("\n💭 Question: ").strip()

            if not question:
                continue

            if question.lower() in ('quit', 'exit', 'q'):
                print("\n👋 Goodbye!")
                break

            # Handle commands
            if question.startswith('/context '):
                try:
                    context_length = int(question.split()[1])
                    print(f"   ✅ Context length set to: {context_length}")
                    continue
                except (ValueError, IndexError):
                    print("   ❌ Invalid context length. Usage: /context 2000")
                    continue

            if question.startswith('/topk '):
                try:
                    top_k = int(question.split()[1])
                    print(f"   ✅ Top-k set to: {top_k}")
                    continue
                except (ValueError, IndexError):
                    print("   ❌ Invalid top-k. Usage: /topk 5")
                    continue

            # Query
            answer = query_memory(pipe, question, context_length, top_k)

            print(f"\n💡 Answer:\n")
            print(answer)
            print("\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Query MemoRAG memory index for Our Biggest Deal"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask (optional if using --interactive)"
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path(__file__).parent.parent / "indices",
        help="Directory containing MemoRAG memory files"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-1.5B-Instruct",
        help="MemoRAG memory/retrieval model name"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2000,
        help="Maximum context length for retrieval (default: 2000)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant passages to retrieve (default: 5)"
    )

    args = parser.parse_args()

    # Load pipeline with native load
    pipe = load_pipeline(args.memory_dir, args.model)

    # Interactive mode
    if args.interactive:
        interactive_mode(pipe)
        return

    # Single query mode
    if not args.question:
        print("❌ No question provided.")
        print("\nUsage:")
        print('  python query_memory.py "Your question here"')
        print('  python query_memory.py --interactive')
        sys.exit(1)

    question = " ".join(args.question)

    # Query
    answer = query_memory(pipe, question, args.context_length, args.top_k)

    # Display answer
    print(f"\n💡 Answer:\n")
    print(answer)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
