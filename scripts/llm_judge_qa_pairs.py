#!/usr/bin/env python3
"""
LLM-as-judge evaluation for QA answer pairs.

This script expects a JSON file containing a list of items, each with:

[
  {
    "query": "original user question",
    "answer_a": "baseline answer (e.g., GRAPH_MODE=off)",
    "answer_b": "candidate answer (e.g., GRAPH_MODE=full)"
  },
  ...
]

For each item, it asks an OpenAI model to judge which answer is better
("A", "B", or "tie") and records the verdict plus a short explanation.

Usage:

  # Ensure OPENAI_API_KEY is set (e.g., via .env) and venv is active.
  python scripts/llm_judge_qa_pairs.py \
      --input data/eval/qa_pairs.json \
      --output data/eval/qa_judge_results.json \
      --model gpt-4o-mini
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI  # type: ignore[import]


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of {query, answer_a, answer_b} objects")
    return data


def judge_pair(
    client: OpenAI,
    model: str,
    query: str,
    answer_a: str,
    answer_b: str,
) -> Dict[str, Any]:
    """Ask the LLM to judge which answer is better."""
    system_msg = (
        "You are an impartial evaluator for question-answer quality. "
        "Given a user question and two candidate answers, decide which answer is "
        "more helpful, accurate, specific, and grounded in the question, or if "
        "they are roughly tied. Respond with a JSON object only, no commentary."
    )
    user_msg = (
        "Question:\n"
        f"{query}\n\n"
        "Answer A:\n"
        f"{answer_a}\n\n"
        "Answer B:\n"
        f"{answer_b}\n\n"
        "Return a JSON object with:\n"
        '{ "winner": "A" | "B" | "tie", '
        '"explanation": "<short explanation>" }'
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
        winner = parsed.get("winner")
        explanation = parsed.get("explanation")
        if winner not in ("A", "B", "tie"):
            raise ValueError(f"Invalid winner: {winner}")
        return {"winner": winner, "explanation": explanation}
    except Exception as exc:
        # Fallback: treat as tie with raw content for debugging
        return {
            "winner": "tie",
            "explanation": f"Failed to parse judge output ({exc}); raw: {content}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file with [{query, answer_a, answer_b}, ...].",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write JSON results with judge verdicts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use as judge (default: gpt-4o-mini).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    pairs = load_pairs(input_path)
    client = OpenAI()

    results: List[Dict[str, Any]] = []
    for item in pairs:
        query = item.get("query", "")
        answer_a = item.get("answer_a", "")
        answer_b = item.get("answer_b", "")
        verdict = judge_pair(client, args.model, query, answer_a, answer_b)
        results.append(
            {
                "query": query,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "winner": verdict["winner"],
                "explanation": verdict["explanation"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print a tiny summary
    counts = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        counts[r["winner"]] = counts.get(r["winner"], 0) + 1
    print(
        f"Judged {len(results)} pairs. "
        f"A wins: {counts['A']}, B wins: {counts['B']}, ties: {counts['tie']}."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

