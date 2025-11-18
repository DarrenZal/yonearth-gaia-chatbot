"""
PodcastSpeakerResolver

Resolves placeholders like "the speaker", "the host", "guest" to concrete names
using episode metadata (title, known host list) to improve graph quality.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class PodcastSpeakerResolver(PostProcessingModule):
    name = "PodcastSpeakerResolver"
    description = "Resolve 'the speaker/host/guest' placeholders to actual names"
    content_types = ["episode"]
    priority = 55  # Run before PronounResolver (60) and VagueEntityBlocker (85)
    dependencies: List[str] = []
    version = "0.1.0"

    PLACEHOLDER_TERMS = {"the speaker", "speaker", "the host", "host", "guest"}
    DEFAULT_HOSTS = ["Aaron William Perry"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.known_hosts = self.config.get("known_hosts", self.DEFAULT_HOSTS)

    def _resolve_from_title(self, title: str) -> Optional[str]:
        # Heuristic: titles like "Episode 120 – Rowdy Yeatts, ..."
        if not title:
            return None
        for sep in ["–", "-", "—", ":"]:
            if sep in title:
                right = title.split(sep, 1)[-1].strip()
                # First comma- or dash-separated segment likely guest name
                guest = right.split(",")[0].split(" – ")[0].strip()
                if 2 <= len(guest.split()) <= 5:
                    return guest
        return None

    def _maybe_resolve(self, text: str, episode_title: str) -> Optional[str]:
        if not text:
            return None
        tl = text.strip().lower()
        if tl in self.PLACEHOLDER_TERMS:
            # Prefer guest from title; fallback to known host
            guest = self._resolve_from_title(episode_title)
            return guest or (self.known_hosts[0] if self.known_hosts else None)
        return None

    def process_batch(self, relationships: List[Any], context: ProcessingContext) -> List[Any]:
        self.stats["processed_count"] = len(relationships)
        replaced = 0
        title = context.document_metadata.get("title") or context.document_metadata.get("episode_title") or ""

        kept: List[Any] = []
        for rel in relationships:
            changed = False
            new_src = self._maybe_resolve(getattr(rel, "source", ""), title)
            if new_src:
                rel.source = new_src
                changed = True
            new_tgt = self._maybe_resolve(getattr(rel, "target", ""), title)
            if new_tgt:
                rel.target = new_tgt
                changed = True
            if changed:
                replaced += 1
            kept.append(rel)

        self.stats["modified_count"] = replaced
        self.stats["resolved"] = replaced
        logger.info(f"   {self.name}: resolved {replaced} placeholder speakers")
        return kept

