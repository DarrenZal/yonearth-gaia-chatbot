#!/usr/bin/env python3
"""
Hybrid eval (Pinecone-backed): Compare BM25+Vector vs BM25+Vector+Graph

Outputs JSON to data/eval/hybrid_eval_pinecone.json with per-query episode overlap stats
and simple latency measurements. Handles network errors gracefully and records them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import sys
from pathlib import Path as _Path

# Ensure project root on sys.path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.config import settings


def get_episode_id(meta: Dict[str, Any]) -> str:
    """Extract a human-friendly episode identifier from metadata.
    Returns episode number when present, else attempts to parse from chunk_id, else a book label.
    """
    ep = meta.get("episode_number") or meta.get("episode_id")
    if ep and str(ep).strip():
        return str(ep)
    # Try parse from chunk_id like ep120:ck000
    cid = meta.get("chunk_id") or ""
    if isinstance(cid, str) and cid.startswith("ep") and ":" in cid:
        return cid.split(":", 1)[0].replace("ep", "")
    # Tag as book content
    bt = meta.get("book_title")
    if bt:
        return f"book:{bt[:24]}"  # short label
    return "unknown"


def run_eval(queries: List[str]) -> Dict[str, Any]:
    from src.rag.vectorstore import create_vectorstore
    from src.rag.bm25_hybrid_retriever import BM25HybridRetriever

    out: Dict[str, Any] = {"results": [], "errors": []}

    try:
        # Build retriever WITHOUT graph
        settings.enable_graph_retrieval = False
        vs1 = create_vectorstore()
        r_no_graph = BM25HybridRetriever(vs1, use_reranker=False)
    except Exception as e:
        out["errors"].append({"stage": "init_no_graph", "error": str(e)})
        return out

    try:
        # Build retriever WITH graph
        settings.enable_graph_retrieval = True
        vs2 = create_vectorstore()
        r_graph = BM25HybridRetriever(vs2, use_reranker=False)
    except Exception as e:
        out["errors"].append({"stage": "init_graph", "error": str(e)})
        return out

    for q in queries:
        rec: Dict[str, Any] = {"query": q}

        # No-graph run
        t0 = time.time()
        try:
            docs_no = r_no_graph.hybrid_search(q, k=15)
            dur_no = time.time() - t0
            eps_no = []
            seen = set()
            for d in docs_no:
                eid = get_episode_id(getattr(d, "metadata", {}))
                if eid not in seen:
                    seen.add(eid)
                    eps_no.append(eid)
            rec["no_graph_eps"] = eps_no
            rec["no_graph_ms"] = int(dur_no * 1000)
        except Exception as e:
            rec["no_graph_error"] = str(e)
            rec["no_graph_ms"] = None

        # Graph run
        t1 = time.time()
        try:
            docs_g = r_graph.hybrid_search(q, k=15)
            dur_g = time.time() - t1
            eps_g = []
            seen = set()
            for d in docs_g:
                eid = get_episode_id(getattr(d, "metadata", {}))
                if eid not in seen:
                    seen.add(eid)
                    eps_g.append(eid)
            rec["graph_eps"] = eps_g
            rec["graph_ms"] = int(dur_g * 1000)
        except Exception as e:
            rec["graph_error"] = str(e)
            rec["graph_ms"] = None

        # Overlap stats
        try:
            s_no = set(rec.get("no_graph_eps", []))
            s_g = set(rec.get("graph_eps", []))
            rec["overlap"] = sorted(list(s_no & s_g))
            rec["only_graph"] = sorted(list(s_g - s_no))
            rec["only_no_graph"] = sorted(list(s_no - s_g))
        except Exception:
            pass

        out["results"].append(rec)

    return out


def main():
    queries = [
        "Rowdy Yeatts biochar",
        "Ecosia Ruby Au interview",
        "Engineers Without Borders Bernard Amadei",
        "Regenerative Organic Alliance Elizabeth Whitlow",
        "Ann Armbrecht Business of Botanicals",
        "Michael Bronner Magic Chocolate",
        "Organic India Miguel Gil",
        "Brigitte Mars cannabis tree of life",
        "Permaculture design certification Stephen Brooks",
        "David Laird biochar science",
    ]

    result = run_eval(queries)
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hybrid_eval_pinecone.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved Pinecone hybrid eval to {out_path}")


if __name__ == "__main__":
    main()
