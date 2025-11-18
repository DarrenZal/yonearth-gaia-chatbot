#!/usr/bin/env python3
"""
Local Hybrid Eval (Keyword + Graph)

Builds a simple keyword index from local transcripts and compares it to GraphRetriever
for a small set of queries. Reports unique episodes found by each and overlap.

This avoids remote vectorstores and runs entirely on local data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Set

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.rag.graph_retriever import GraphRetriever
from src.rag.keyword_indexer import KeywordIndexer


def build_keyword_index() -> KeywordIndexer:
    ki = KeywordIndexer(use_stemming=True, remove_stopwords=True)
    ki.build_index_from_episodes_dir(settings.episodes_dir)
    return ki


def search_keyword_episodes(ki: KeywordIndexer, query: str, top_k: int = 10) -> List[str]:
    # Very simple scoring: sum frequencies for query terms, return top episodes
    terms = ki._preprocess_text(query)
    scores: Dict[str, int] = {}
    for t in terms:
        eps = ki.word_episode_mapping.get(t, set())
        for ep in eps:
            scores[ep] = scores.get(ep, 0) + ki.episode_word_frequencies.get(ep, {}).get(t, 0)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Convert episode_id like 'episode_120' → '120'
    eps: List[str] = []
    for eid, _ in ranked[:top_k]:
        try:
            num = eid.split('_')[1]
            eps.append(num)
        except Exception:
            pass
    return eps


def search_graph_episodes(gr: GraphRetriever, query: str, top_k: int = 10) -> List[str]:
    res = gr.retrieve(query, k=top_k)
    eps: List[str] = []
    for d in res.chunks:
        en = d.metadata.get('episode_number')
        if en is not None:
            eps.append(str(en))
    # De-dup preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for e in eps:
        if e not in seen and e.isdigit():
            seen.add(e)
            out.append(e)
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

    print("Building local keyword index...")
    ki = build_keyword_index()
    gr = GraphRetriever()

    results: List[Dict[str, Any]] = []
    for q in queries:
        k_eps = search_keyword_episodes(ki, q, top_k=10)
        g_eps = search_graph_episodes(gr, q, top_k=10)
        k_set, g_set = set(k_eps), set(g_eps)
        overlap = sorted(k_set & g_set, key=lambda x: k_eps.index(x) if x in k_eps else 999)
        only_k = sorted(k_set - g_set, key=lambda x: k_eps.index(x))
        only_g = sorted(g_set - k_set, key=lambda x: g_eps.index(x))
        results.append({
            'query': q,
            'keyword_eps': k_eps,
            'graph_eps': g_eps,
            'overlap': overlap,
            'only_keyword': only_k,
            'only_graph': only_g,
        })

    # Print concise report
    print("\nHYBRID EVAL (Local Keyword vs Graph)")
    for r in results:
        print(f"\nQ: {r['query']}")
        print(f"  keyword eps: {r['keyword_eps']}")
        print(f"  graph eps:   {r['graph_eps']}")
        print(f"  overlap:     {r['overlap']}")
        print(f"  only_graph:  {r['only_graph']}")
        print(f"  only_kw:     {r['only_keyword']}")

    # Optionally save JSON summary
    try:
        out_dir = settings.data_dir / 'eval'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'hybrid_eval_local.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'results': results}, f, indent=2)
        print(f"\nSaved eval summary to {out_path}")
    except Exception as e:
        print(f"\nWarning: could not save eval summary: {e}")


if __name__ == "__main__":
    main()
