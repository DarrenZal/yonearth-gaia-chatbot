#!/usr/bin/env python3
"""
Collect QA answer pairs from two Hybrid QA endpoints for LLM-as-judge evaluation.

This script calls /api/qa/hybrid on two different base URLs (e.g. baseline vs
candidate deployment) for the same set of queries and writes a JSON file in the
format expected by scripts/llm_judge_qa_pairs.py:

[
  {
    "query": "...",
    "answer_a": "...",  # from URL A
    "answer_b": "..."   # from URL B
  },
  ...
]

Typical usage:

  # Baseline at :8000 (e.g. GRAPH_MODE=off), candidate at :8001 (GRAPH_MODE=full)
  python scripts/collect_qa_pairs.py \
      --prompts data/search_eval_prompts.json \
      --url-a http://localhost:8000/api/qa/hybrid \
      --url-b http://localhost:8001/api/qa/hybrid \
      --output data/eval/qa_pairs.json \
      --limit 20
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import requests


def load_queries(path: Path) -> List[str]:
    """Load queries from the shared prompts JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    queries = [p.get("query") for p in prompts if p.get("query")]
    return queries


def call_hybrid(url: str, query: str, k: int = 10, timeout: float = 30.0) -> str:
    """Call the Hybrid QA endpoint and return the answer text (or empty string on error)."""
    try:
        resp = requests.post(
            url,
            json={"query": query, "k": k},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        # HybridQAResponse has "answer" field which may be None
        return data.get("answer") or ""
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/search_eval_prompts.json",
        help="Path to search_eval_prompts.json",
    )
    parser.add_argument(
        "--url-a",
        type=str,
        required=True,
        help="Hybrid QA endpoint URL for answer A (e.g. http://localhost:8000/api/qa/hybrid)",
    )
    parser.add_argument(
        "--url-b",
        type=str,
        required=True,
        help="Hybrid QA endpoint URL for answer B (e.g. http://localhost:8001/api/qa/hybrid)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write qa_pairs.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of queries to evaluate",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="k parameter for Hybrid QA retrieval",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_path = Path(args.output)

    queries = load_queries(prompts_path)
    if args.limit is not None:
        queries = queries[: args.limit]

    print(f"Loaded {len(queries)} queries from {prompts_path}")
    print(f"URL A: {args.url_a}")
    print(f"URL B: {args.url_b}")

    pairs: List[Dict[str, Any]] = []

    for i, q in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] Query: {q[:80]}...")
        answer_a = call_hybrid(args.url_a, q, k=args.k)
        answer_b = call_hybrid(args.url_b, q, k=args.k)
        pairs.append(
            {
                "query": q,
                "answer_a": answer_a,
                "answer_b": answer_b,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(pairs)} QA pairs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

