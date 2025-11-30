#!/usr/bin/env python3
"""
Query MemoRAG Sharded Memory Index - BM25 Shard Router

Uses a two-stage approach:
1. ROUTE: BM25 pre-selection to identify most relevant shards
2. QUERY: Load selected shards and query MemoRAG for reasoning

This preserves MemoRAG's memory/reasoning capabilities while enabling
full book coverage through sharding.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
INDICES_DIR = Path(__file__).parent.parent / "indices"


class BM25ShardRouter:
    """
    BM25-based router to select most relevant shards for a query.
    Uses shard metadata (top_terms) for fast pre-selection.
    """

    def __init__(self, metadata_path: Path):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.num_shards = self.metadata["num_shards"]
        self.shards = self.metadata["shards"]

        # Build inverted index from shard terms
        self.term_to_shards: Dict[str, List[int]] = {}
        self.shard_term_counts: Dict[int, Counter] = {}

        for shard in self.shards:
            idx = shard["index"]
            terms = shard["top_terms"]
            self.shard_term_counts[idx] = Counter(terms)

            for term in terms:
                if term not in self.term_to_shards:
                    self.term_to_shards[term] = []
                self.term_to_shards[term].append(idx)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        words = text.lower().split()
        return [''.join(c for c in word if c.isalnum()) for word in words if len(word) > 2]

    def _bm25_score(self, query_terms: List[str], shard_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calculate BM25 score for a shard given query terms.
        Simplified version using term overlap.
        """
        shard_terms = set(self.shard_term_counts[shard_idx].keys())
        avg_terms = sum(len(s["top_terms"]) for s in self.shards) / len(self.shards)

        score = 0.0
        for term in query_terms:
            if term in shard_terms:
                # IDF component
                n_containing = len(self.term_to_shards.get(term, []))
                idf = math.log((self.num_shards - n_containing + 0.5) / (n_containing + 0.5) + 1)

                # TF component (simplified)
                tf = 1.0  # Term appears in top terms
                doc_len = len(shard_terms)
                tf_normalized = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_terms)))

                score += idf * tf_normalized

        return score

    def route(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Route query to most relevant shards using BM25.

        Returns:
            List of (shard_idx, score) tuples, sorted by score descending
        """
        query_terms = self._tokenize(query)

        scores = []
        for shard in self.shards:
            idx = shard["index"]
            score = self._bm25_score(query_terms, idx)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])

        # Return top-k shards (at least 1, even if score is 0)
        selected = scores[:top_k]

        # If all scores are 0, include all shards (fallback)
        if all(s[1] == 0 for s in selected):
            print("   ⚠️  No term matches, querying all shards...")
            return [(s["index"], 0.0) for s in self.shards]

        return selected


class ShardedMemoRAGQuerier:
    """
    Query handler for sharded MemoRAG indices.
    """

    def __init__(
        self,
        indices_dir: Path,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        use_openai_generation: bool = True
    ):
        self.indices_dir = Path(indices_dir)
        self.model_name = model_name
        self.use_openai_generation = use_openai_generation

        # Load router
        metadata_path = self.indices_dir / "shard_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Shard metadata not found: {metadata_path}")

        self.router = BM25ShardRouter(metadata_path)
        self.pipe = None  # Lazy load

    def _init_pipeline(self):
        """Initialize MemoRAG pipeline (lazy loading)."""
        if self.pipe is not None:
            return

        from memorag import MemoRAG, Agent

        print(f"   🔄 Loading Qwen model...")

        # Configure generation
        customized_gen_model = None
        if self.use_openai_generation:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                customized_gen_model = Agent(
                    model="gpt-4.1-mini",
                    source="openai",
                    api_dict={"api_key": openai_key}
                )

        cache_path = str(self.indices_dir.parent / "model_cache")

        self.pipe = MemoRAG(
            mem_model_name_or_path=self.model_name,
            ret_model_name_or_path=self.model_name,
            cache_dir=cache_path,
            customized_gen_model=customized_gen_model,
            enable_flash_attn=False,
            load_in_4bit=False
        )
        print(f"   ✅ Pipeline ready")

    def query(
        self,
        question: str,
        max_tokens: int = 512,
        top_shards: int = 3
    ) -> Dict:
        """
        Query the sharded MemoRAG index.

        Strategy:
        1. Route: Use BM25 to select top shards
        2. Query: Load each shard and get MemoRAG's answer
        3. Aggregate: Combine answers (best score wins)

        Args:
            question: User's question
            max_tokens: Max tokens in response
            top_shards: Number of shards to query

        Returns:
            Dict with answer, shard info, and timing
        """
        start_time = time.time()

        # Initialize pipeline if needed
        self._init_pipeline()

        # Route to relevant shards
        print(f"\n🔍 Routing query to shards...")
        selected_shards = self.router.route(question, top_k=top_shards)

        print(f"   Selected shards: {[s[0] for s in selected_shards]}")
        for idx, score in selected_shards:
            preview = self.router.shards[idx]["preview"][:60]
            print(f"   Shard {idx} (score={score:.2f}): {preview}...")

        # Query each shard
        answers = []
        for shard_idx, route_score in selected_shards:
            shard_path = self.indices_dir / f"shard_{shard_idx}"

            if not shard_path.exists():
                print(f"   ⚠️  Shard {shard_idx} not found, skipping")
                continue

            print(f"\n   📖 Querying shard {shard_idx}...")
            shard_start = time.time()

            try:
                # Load shard
                self.pipe.load(str(shard_path))

                # Query MemoRAG's memory model
                answer = self.pipe.mem_model.answer(
                    question,
                    max_new_tokens=max_tokens
                )

                shard_time = time.time() - shard_start

                answers.append({
                    "shard_idx": shard_idx,
                    "route_score": route_score,
                    "answer": answer,
                    "query_time_ms": int(shard_time * 1000)
                })

                print(f"      ✅ Got answer in {shard_time:.1f}s")

            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue

        total_time = time.time() - start_time

        # Select best answer (highest route score with non-empty answer)
        best_answer = None
        for ans in answers:
            if ans["answer"] and not ans["answer"].lower().startswith("the context does not"):
                if best_answer is None or ans["route_score"] > best_answer["route_score"]:
                    best_answer = ans

        # Fallback to first answer if all were "no context"
        if best_answer is None and answers:
            best_answer = answers[0]

        return {
            "answer": best_answer["answer"] if best_answer else "No relevant information found.",
            "shard_answers": answers,
            "selected_shards": [s[0] for s in selected_shards],
            "total_time_ms": int(total_time * 1000),
            "source": "Our Biggest Deal"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Query MemoRAG sharded memory index"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask about the book"
    )
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=INDICES_DIR,
        help="Directory containing shard subdirectories"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-1.5B-Instruct",
        help="MemoRAG memory model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens in response"
    )
    parser.add_argument(
        "--top-shards",
        type=int,
        default=3,
        help="Number of shards to query"
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Use local generation (slow)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🧠 MemoRAG Sharded Query - Our Biggest Deal")
    print("=" * 60)

    # Initialize querier
    try:
        querier = ShardedMemoRAGQuerier(
            indices_dir=args.indices_dir,
            model_name=args.model,
            use_openai_generation=not args.no_openai
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run build_memory.py first to create shards.")
        sys.exit(1)

    print(f"\n📚 Loaded {querier.router.num_shards} shards from {args.indices_dir}")

    if args.interactive:
        print("\n💬 Interactive mode. Type 'quit' to exit.\n")

        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue

                result = querier.query(
                    query,
                    max_tokens=args.max_tokens,
                    top_shards=args.top_shards
                )

                print(f"\n🌍 Gaia: {result['answer']}")
                print(f"   [Shards: {result['selected_shards']}, Time: {result['total_time_ms']}ms]\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

    elif args.query:
        result = querier.query(
            args.query,
            max_tokens=args.max_tokens,
            top_shards=args.top_shards
        )

        print(f"\n" + "=" * 60)
        print("📝 ANSWER:")
        print("=" * 60)
        print(result["answer"])
        print(f"\n🔍 Shards queried: {result['selected_shards']}")
        print(f"⏱️  Total time: {result['total_time_ms']}ms")

        # Show shard details if verbose
        if len(result["shard_answers"]) > 1:
            print("\n📊 Shard answers:")
            for ans in result["shard_answers"]:
                preview = ans["answer"][:100] if ans["answer"] else "(empty)"
                print(f"   Shard {ans['shard_idx']}: {preview}...")

    else:
        print("\n⚠️  No query provided. Use --interactive or pass a query.")
        print("   Example: python query_memory.py \"What is Quintenary Economics?\"")


if __name__ == "__main__":
    main()
