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


def query_memory(pipe, question: str, max_new_tokens: int = 512):
    """
    Query the MemoRAG memory.

    Args:
        pipe: MemoRAG pipeline
        question: User question
        max_new_tokens: Maximum tokens in generated answer

    Returns:
        Answer string
    """
    print(f"\n🔍 Query: {question}")
    print(f"   Generating answer from memory...")

    # Query the memorized content using mem_model.answer()
    answer = pipe.mem_model.answer(question, max_new_tokens=max_new_tokens)

    return answer


def interactive_mode(pipe):
    """Run interactive query session."""
    print("\n" + "=" * 60)
    print("🧠 MemoRAG Interactive Mode - Our Biggest Deal")
    print("=" * 60)
    print("\nType your questions below. Type 'quit' or 'exit' to stop.")
    print("Commands:")
    print("  /tokens <n>   - Set max tokens in answer (default: 512)")
    print("=" * 60)

    max_new_tokens = 512

    while True:
        try:
            question = input("\n💭 Question: ").strip()

            if not question:
                continue

            if question.lower() in ('quit', 'exit', 'q'):
                print("\n👋 Goodbye!")
                break

            # Handle commands
            if question.startswith('/tokens '):
                try:
                    max_new_tokens = int(question.split()[1])
                    print(f"   ✅ Max tokens set to: {max_new_tokens}")
                    continue
                except (ValueError, IndexError):
                    print("   ❌ Invalid token count. Usage: /tokens 512")
                    continue

            # Query
            answer = query_memory(pipe, question, max_new_tokens)

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
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens in generated answer (default: 512)"
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
    answer = query_memory(pipe, question, args.max_tokens)

    # Display answer
    print(f"\n💡 Answer:\n")
    print(answer)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
