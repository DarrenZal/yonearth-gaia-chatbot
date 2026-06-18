"""
Retrieval-quality eval harness for Aaron's three issues with the YOE AI Guide.

Runs golden queries through `BM25RAGChain.chat()` (the same path /api/bm25/chat
hits in production) and asserts:
  - definitional summaries come from YOE content (not LLM training);
  - theme queries surface theme-tagged episodes;
  - books surface as citations for book-targeted queries.

Two run modes:
    local  — instantiate BM25RAGChain in-process (needs Pinecone + OpenAI creds in env)
    http   — POST to a deployed /api/bm25/chat endpoint (needs only network)

The HTTP mode is what we use for the baseline against the live earthdo.me/guide/
since local OpenAI quota may be exhausted, and it directly reproduces what users
see. The local mode is for testing patched code before a deploy when quota is
available.

Usage:
    # Baseline against production (HTTP)
    python -m tests.eval.retrieval_quality --mode http \\
        --base-url https://earthdo.me --tag baseline-prod

    # Patched code in-process (local)
    python -m tests.eval.retrieval_quality --mode local --tag patched

    # Single query by id-prefix
    python -m tests.eval.retrieval_quality --mode http --query finance

    # Diff a result file vs the latest run
    python -m tests.eval.retrieval_quality --diff baseline-prod-2026-05-04T...json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rag.episode_categorizer import EpisodeCategorizer  # noqa: E402

logger = logging.getLogger("retrieval_eval")

GOLDEN_PATH = Path(__file__).parent / "golden_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_golden() -> List[Dict[str, Any]]:
    return json.loads(GOLDEN_PATH.read_text())["queries"]


# Phrases the grounded LLM should produce when asked something outside the
# YOE archive. The grounding contract instructs it to say "I don't have that
# in the YonEarth archive yet — would you like me to look more broadly within
# the YonEarth community?" — assertions match flexibly against the *signal*
# rather than the exact wording.
# Phrases that genuinely signal an out-of-archive *decline*. NOTE (2026-06-18):
# the bare corpus-name phrases "yonearth archive" / "yonearth community" /
# "yonearth library" were REMOVED — they are the name of the corpus and appear
# constantly in *positive, grounded* answers ("Episodes within the YonEarth
# archive explore biochar…"), producing false positives on
# response_must_not_signal_out_of_archive items (e.g. #15 "What is biochar?").
# The genuine decline fallback always contains "don't have that" AND "look more
# broadly", so removing the bare names does not weaken detection of real
# declines (verified against #23/#24/#25, which still trip on those phrases).
OUT_OF_ARCHIVE_SIGNAL_PHRASES = (
    "don't have that",
    "do not have that",
    # Honest partial-coverage disclaimer (added 2026-06-18): the Guide often
    # declines the *specific* requested item while pointing to related YOE
    # content ("I don't have a specific chocolate cake recipe in the YonEarth
    # archive, however Episode 21 features Chef Maria Cooper…"). That genuinely
    # signals out-of-archive scope; it just isn't the canned fallback string.
    # Safe to recognize because the separate response_must_not_contain_any
    # fabrication guard (e.g. "cup of flour", "preheat the oven") independently
    # blocks any answer that actually fabricates the requested recipe/specs.
    "don't have a specific",
    "do not have a specific",
    "haven't covered",
    "have not covered",
    "look more broadly",
    "isn't in the",
    "is not in the",
    "outside the yonearth",
    "outside of the yonearth",
    "not specifically discussed",
    "not specifically covered",
    "no specific information",
    "no information specifically",
    "isn't currently in",
    "not currently in",
)


def _signals_out_of_archive(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in OUT_OF_ARCHIVE_SIGNAL_PHRASES)


def _episodes_in_category(categorizer: EpisodeCategorizer, category: str) -> set[int]:
    return set(categorizer.get_episodes_by_category(category))


def _check_assertions(
    query_def: Dict[str, Any],
    chat_result: Dict[str, Any],
    categorizer: EpisodeCategorizer,
) -> Tuple[bool, List[str]]:
    """Evaluate assertions for a single query. Returns (passed, list-of-failures)."""
    failures: List[str] = []
    a = query_def.get("assertions", {})
    response_text = (chat_result.get("response") or "").lower()
    sources = chat_result.get("sources") or []

    # response_must_not_contain_any: any hit fails
    for forbidden in a.get("response_must_not_contain_any", []):
        if forbidden.lower() in response_text:
            failures.append(f"response contains forbidden substring '{forbidden}'")

    # response_must_contain_any: at least one must hit
    must_contain = a.get("response_must_contain_any")
    if must_contain and not any(s.lower() in response_text for s in must_contain):
        failures.append(f"response missing all of expected substrings {must_contain}")

    # min_citations
    min_citations = a.get("min_citations")
    if min_citations is not None and len(sources) < min_citations:
        failures.append(f"got {len(sources)} citations, need >= {min_citations}")

    # citations_must_include_category_any: at least one cited episode must be tagged with one of the categories
    cat_any = a.get("citations_must_include_category_any") or []
    if cat_any:
        cited_ep_ids = {
            int(s["episode_number"])
            for s in sources
            if s.get("content_type") == "episode"
            and str(s.get("episode_number", "")).strip().isdigit()
        }
        cat_episodes_union: set[int] = set()
        for cat in cat_any:
            cat_episodes_union |= _episodes_in_category(categorizer, cat)
        if not (cited_ep_ids & cat_episodes_union):
            failures.append(
                f"no cited episode is tagged with any of {cat_any} "
                f"(cited episodes: {sorted(cited_ep_ids)})"
            )

    # citations_must_include_category_all: every category must contribute at least one cited episode
    cat_all = a.get("citations_must_include_category_all") or []
    cited_ep_ids_all = {
        int(s["episode_number"])
        for s in sources
        if s.get("content_type") == "episode"
        and str(s.get("episode_number", "")).strip().isdigit()
    }
    for cat in cat_all:
        if not (cited_ep_ids_all & _episodes_in_category(categorizer, cat)):
            failures.append(f"no cited episode tagged with category {cat}")

    # min_distinct_impact_investing_episodes
    min_ii = a.get("min_distinct_impact_investing_episodes")
    if min_ii is not None:
        ii_eps = _episodes_in_category(categorizer, "IMPACT INVESTING")
        cited_ii = cited_ep_ids_all & ii_eps
        if len(cited_ii) < min_ii:
            failures.append(
                f"got {len(cited_ii)} IMPACT INVESTING-tagged citations, need >= {min_ii}"
            )

    # min_distinct_episodes_in_category — generalized form: {"category": "X", "min": N}
    spec = a.get("min_distinct_episodes_in_category")
    if isinstance(spec, dict):
        cat = spec.get("category")
        min_n = int(spec.get("min", 1))
        cat_eps = _episodes_in_category(categorizer, cat) if cat else set()
        cited_in_cat = cited_ep_ids_all & cat_eps
        if len(cited_in_cat) < min_n:
            failures.append(
                f"got {len(cited_in_cat)} {cat}-tagged citations, need >= {min_n} "
                f"(cited episodes: {sorted(cited_ep_ids_all)})"
            )

    # response_should_signal_out_of_archive: grounding contract should kick in
    if a.get("response_should_signal_out_of_archive"):
        if not _signals_out_of_archive(response_text):
            failures.append(
                "response did not signal out-of-archive scope (grounding contract not triggering)"
            )

    # response_must_not_signal_out_of_archive: inverse — grounding contract
    # should NOT fire because the Context does cover the topic. Catches false-
    # refusal regressions where over-strict prompting makes the LLM bail on
    # legitimate partial-coverage queries.
    if a.get("response_must_not_signal_out_of_archive"):
        if _signals_out_of_archive(response_text):
            failures.append(
                "response falsely signaled out-of-archive scope despite "
                "Context containing on-topic material"
            )

    # citations_must_include_book_any: at least one cited book title must match
    book_any = a.get("citations_must_include_book_any") or []
    if book_any:
        cited_books = {
            (s.get("book_title") or "").strip()
            for s in sources
            if s.get("content_type") == "book"
        }
        if not any(b in cited_books for b in book_any):
            failures.append(
                f"no cited book matches {book_any} (cited books: {sorted(cited_books)})"
            )

    return (len(failures) == 0, failures)


def _summarize_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact representation for the report."""
    out = []
    for s in sources:
        if s.get("content_type") == "book":
            out.append(
                {
                    "type": "book",
                    "book_title": s.get("book_title"),
                    "chapter_number": s.get("chapter_number"),
                    "chapter_title": s.get("chapter_title"),
                }
            )
        else:
            out.append(
                {
                    "type": "episode",
                    "episode_number": s.get("episode_number"),
                    "title": s.get("title"),
                    "guest_name": s.get("guest_name"),
                }
            )
    return out


def _markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Retrieval Quality Eval — {report['tag']} ({report['timestamp']})")
    lines.append("")
    lines.append(
        f"**Summary:** {report['summary']['passed']} / {report['summary']['total']} passed"
    )
    lines.append("")
    for r in report["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"## [{status}] `{r['id']}` — {r['query']!r}")
        lines.append(f"_{r['intent']}_")
        lines.append("")
        if r["failures"]:
            lines.append("**Failures:**")
            for f in r["failures"]:
                lines.append(f"- {f}")
            lines.append("")
        lines.append(
            f"**Search method:** {r['search_method']} • "
            f"**Docs retrieved:** {r['documents_retrieved']} • "
            f"**Citations:** {len(r['sources'])}"
        )
        lines.append("")
        lines.append("**Cited:**")
        for s in r["sources"]:
            if s["type"] == "book":
                lines.append(
                    f"- BOOK: {s['book_title']} — Ch. {s['chapter_number']} ({s['chapter_title']})"
                )
            else:
                lines.append(
                    f"- EP {s['episode_number']}: {s['title']} (guest: {s['guest_name']})"
                )
        lines.append("")
        lines.append("**Response (first 800 chars):**")
        lines.append("")
        lines.append("> " + (r["response_excerpt"].replace("\n", "\n> ") or "(empty)"))
        lines.append("")
    return "\n".join(lines)


def _chat_http(base_url: str, message: str, k: int, timeout_s: float) -> Dict[str, Any]:
    """POST to /api/bm25/chat on a deployed instance and normalize the response shape."""
    url = base_url.rstrip("/") + "/api/bm25/chat"
    payload = json.dumps(
        {
            "message": message,
            "search_method": "auto",
            "k": k,
            "include_sources": True,
            "gaia_personality": "aaron_guide",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read())
    return data


def run_eval(
    query_id_prefix: Optional[str] = None,
    tag: str = "untagged",
    k: int = 5,
    mode: str = "local",
    base_url: str = "https://earthdo.me",
    http_timeout: float = 90.0,
) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    golden = _load_golden()
    if query_id_prefix:
        golden = [q for q in golden if q["id"].startswith(query_id_prefix)]
        if not golden:
            raise SystemExit(f"No queries match prefix {query_id_prefix!r}")

    chain = None
    if mode == "local":
        from src.rag.bm25_chain import BM25RAGChain  # local-only import

        logger.info("Initializing BM25RAGChain (this loads Pinecone + categorizer)...")
        chain = BM25RAGChain(initialize_data=True)
    elif mode == "http":
        logger.info(f"HTTP mode against {base_url}/api/bm25/chat")
    else:
        raise SystemExit(f"Unknown mode {mode!r}")

    # Categorizer is needed for assertion lookups in both modes (loads from local CSV).
    categorizer = EpisodeCategorizer()

    results: List[Dict[str, Any]] = []
    for q in golden:
        qid = q["id"]
        query = q["query"]
        logger.info(f"Running [{qid}]: {query!r}")

        try:
            if mode == "local":
                chat_result = chain.chat(
                    message=query,
                    search_method="auto",
                    k=k,
                    include_sources=True,
                    personality_variant="aaron_guide",
                )
            else:
                chat_result = _chat_http(base_url, query, k, http_timeout)
            error_note = None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            chat_result = {"response": "", "sources": [], "error": str(e)}
            error_note = str(e)
            logger.error(f"  HTTP error: {e}")

        if mode == "http":
            # Throttle — don't hammer the deployed endpoint
            time.sleep(1.5)

        passed, failures = _check_assertions(q, chat_result, categorizer)
        if error_note:
            failures.insert(0, f"transport error: {error_note}")
            passed = False
        results.append(
            {
                "id": qid,
                "query": query,
                "intent": q.get("intent", ""),
                "passed": passed,
                "failures": failures,
                "search_method": chat_result.get("search_method_used", "?"),
                "documents_retrieved": chat_result.get("documents_retrieved", 0),
                "sources": _summarize_sources(chat_result.get("sources") or []),
                "response_excerpt": (chat_result.get("response") or "")[:800],
            }
        )
        logger.info(f"  -> {'PASS' if passed else 'FAIL'} ({len(failures)} failures)")

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    report = {
        "tag": tag,
        "timestamp": timestamp,
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{tag}-{timestamp}.json"
    md_path = RESULTS_DIR / f"{tag}-{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"\n{'='*60}")
    print(f"Eval complete: {summary['passed']}/{summary['total']} passed")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print(f"{'='*60}")
    return report


def diff_reports(baseline_path: Path, current_path: Path) -> str:
    baseline = json.loads(baseline_path.read_text())
    current = json.loads(current_path.read_text())
    by_id_b = {r["id"]: r for r in baseline["results"]}
    by_id_c = {r["id"]: r for r in current["results"]}

    lines = [
        f"# Diff — {baseline['tag']} vs {current['tag']}",
        "",
        f"Baseline ({baseline['timestamp']}): {baseline['summary']['passed']}/{baseline['summary']['total']}",
        f"Current  ({current['timestamp']}): {current['summary']['passed']}/{current['summary']['total']}",
        "",
        "| Query | Baseline | Current | Change |",
        "|---|---|---|---|",
    ]
    for qid in sorted(set(by_id_b) | set(by_id_c)):
        b = by_id_b.get(qid)
        c = by_id_c.get(qid)
        b_status = "PASS" if b and b["passed"] else "FAIL" if b else "—"
        c_status = "PASS" if c and c["passed"] else "FAIL" if c else "—"
        change = ""
        if b_status == "FAIL" and c_status == "PASS":
            change = "FIXED"
        elif b_status == "PASS" and c_status == "FAIL":
            change = "REGRESSED"
        lines.append(f"| `{qid}` | {b_status} | {c_status} | {change} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Retrieval quality eval harness")
    parser.add_argument("--tag", default="untagged", help="Tag for the result filename (e.g. baseline, patched)")
    parser.add_argument("--query", default=None, help="Run only queries whose id starts with this prefix")
    parser.add_argument("--k", type=int, default=5, help="Number of docs to retrieve per query")
    parser.add_argument(
        "--mode",
        choices=["local", "http"],
        default="local",
        help="local = in-process BM25RAGChain; http = POST to a deployed /api/bm25/chat",
    )
    parser.add_argument(
        "--base-url",
        default="https://earthdo.me",
        help="Base URL for HTTP mode (default: https://earthdo.me)",
    )
    parser.add_argument("--http-timeout", type=float, default=90.0, help="Per-request timeout in HTTP mode")
    parser.add_argument("--diff", default=None, help="Diff this baseline JSON file against the latest run")
    args = parser.parse_args()

    if args.diff:
        baseline_path = Path(args.diff)
        if not baseline_path.is_absolute():
            baseline_path = RESULTS_DIR / baseline_path
        # find newest report file other than baseline
        candidates = sorted(
            (p for p in RESULTS_DIR.glob("*.json") if p != baseline_path),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit("No other result file to diff against. Run an eval first.")
        current_path = candidates[0]
        print(diff_reports(baseline_path, current_path))
        return

    run_eval(
        query_id_prefix=args.query,
        tag=args.tag,
        k=args.k,
        mode=args.mode,
        base_url=args.base_url,
        http_timeout=args.http_timeout,
    )


if __name__ == "__main__":
    main()
