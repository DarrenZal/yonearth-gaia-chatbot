#!/usr/bin/env python3
"""
Postprocess Episode KG files with the podcast pipeline (speaker resolver + contact info filter + universal fixes).

Inputs: data/knowledge_graph_v3_2_2/episode_*_v3_2_2.json
Outputs: data/knowledge_graph_unified/episodes_postprocessed/episode_*.json

Usage:
  python3 scripts/postprocess_episodes_podcast.py --episodes 120,121,122
  python3 scripts/postprocess_episodes_podcast.py --range 110-130
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.config import settings
from src.knowledge_graph.postprocessing.base import ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import get_podcast_pipeline

logger = logging.getLogger(__name__)


def read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@dataclass
class SimpleRel:
    source: str
    relationship: str
    target: str
    source_type: Optional[str] = None
    target_type: Optional[str] = None
    evidence_text: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None
    flags: Optional[Dict[str, Any]] = None
    # Optional context/page fields used by some modules
    context: Optional[str] = None
    page: int = 0
    # Optional fields referenced by some modules (provide safe defaults)
    text_confidence: float = 1.0
    p_true: float = 1.0
    signals_conflict: bool = False
    conflict_explanation: Optional[str] = None
    suggested_correction: Optional[str] = None
    classification_flags: Optional[List[str]] = None
    candidate_uid: str = "cand"


def adapt_in(rel: Dict[str, Any]) -> SimpleRel:
    return SimpleRel(
        source=rel.get("source", ""),
        relationship=rel.get("relationship", rel.get("predicate", "")),
        target=rel.get("target", ""),
        source_type=rel.get("source_type"),
        target_type=rel.get("target_type"),
        evidence_text=rel.get("evidence_text"),
        evidence=rel.get("evidence"),
        flags=rel.get("flags", {}),
        text_confidence=float(rel.get("text_confidence", 1.0) or 1.0),
        p_true=float(rel.get("p_true", 1.0) or 1.0),
        signals_conflict=bool(rel.get("signals_conflict", False)),
        conflict_explanation=rel.get("conflict_explanation"),
        suggested_correction=rel.get("suggested_correction"),
        classification_flags=rel.get("classification_flags") or [],
        candidate_uid=str(rel.get("candidate_uid", "cand")),
    )


def adapt_out(rel: SimpleRel) -> Dict[str, Any]:
    return {
        "source": rel.source,
        "relationship": rel.relationship,
        "target": rel.target,
        "source_type": rel.source_type,
        "target_type": rel.target_type,
        "evidence_text": rel.evidence_text,
        "evidence": rel.evidence,
        "flags": rel.flags or {},
        "context": rel.context,
        "page": rel.page,
        "text_confidence": rel.text_confidence,
        "p_true": rel.p_true,
        "signals_conflict": rel.signals_conflict,
        "conflict_explanation": rel.conflict_explanation,
        "suggested_correction": rel.suggested_correction,
        "classification_flags": rel.classification_flags or [],
        "candidate_uid": rel.candidate_uid,
    }


def load_title(ep_num: int) -> str:
    try:
        tpath = settings.episodes_dir / f"episode_{ep_num}.json"
        tdata = read_json(tpath)
        return tdata.get("title", "")
    except Exception:
        return ""


def process_episode_file(ep_num: int, pipeline) -> Optional[Path]:
    src_dir = settings.data_dir / "knowledge_graph_v3_2_2"
    out_dir = settings.data_dir / "knowledge_graph_unified" / "episodes_postprocessed"
    in_path = src_dir / f"episode_{ep_num}_v3_2_2.json"
    if not in_path.exists():
        logger.warning(f"Missing input for episode {ep_num}: {in_path}")
        return None
    data = read_json(in_path)
    rels_raw = data.get("relationships", [])
    rels = [adapt_in(r) for r in rels_raw]

    # Build processing context
    ctx = ProcessingContext(
        content_type="episode",
        document_metadata={
            "episode_number": ep_num,
            "title": load_title(ep_num)
        },
        config={}
    )

    processed, stats = pipeline.run(rels, ctx)
    out_rels = [adapt_out(r) for r in processed]
    out = {
        **{k: v for k, v in data.items() if k != "relationships"},
        "relationships": out_rels,
        "postprocess": {
            "modules": stats.get("modules_run", []),
        }
    }
    out_path = out_dir / f"episode_{ep_num}_post.json"
    write_json(out_path, out)
    return out_path


def parse_episodes_arg(episodes: Optional[str], range_arg: Optional[str]) -> List[int]:
    eps: List[int] = []
    if episodes:
        eps.extend([int(x.strip()) for x in episodes.split(",") if x.strip()])
    if range_arg:
        a, b = range_arg.split("-")
        eps.extend(list(range(int(a), int(b) + 1)))
    return sorted(set(eps))


def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Postprocess episodes with podcast pipeline")
    ap.add_argument("--episodes", type=str, default=None, help="Comma-separated episode numbers")
    ap.add_argument("--range", type=str, default=None, help="Range A-B of episode numbers")
    args = ap.parse_args()

    eps = parse_episodes_arg(args.episodes, args.range)
    if not eps:
        print("Provide --episodes or --range")
        return

    pipeline = get_podcast_pipeline()
    outputs = []
    for ep in eps:
        try:
            p = process_episode_file(ep, pipeline)
            if p:
                logger.info(f"Wrote {p}")
                outputs.append(str(p))
        except Exception as e:
            logger.warning(f"Postprocess failed for episode {ep}: {e}")

    print(json.dumps({"processed": eps, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
