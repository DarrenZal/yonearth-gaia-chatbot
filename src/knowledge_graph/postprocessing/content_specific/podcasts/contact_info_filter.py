"""
ContactInfoFilter

Filters relationships that encode PII-like contact information from podcast content,
e.g., CONTACT_INFORMATION → email address / phone / social.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, List, Dict, Any

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class ContactInfoFilter(PostProcessingModule):
    name = "ContactInfoFilter"
    description = "Filter CONTACT_INFORMATION/PII-like edges from podcasts"
    content_types = ["episode"]
    priority = 12  # Early, after basic parsing
    dependencies: List[str] = []
    version = "0.1.0"

    PREDICATE_BLOCKLIST = {"contact_information", "contact-info", "contact", "email", "phone"}
    TARGET_PATTERNS = [
        re.compile(r"email", re.IGNORECASE),
        re.compile(r"@\w+"),  # social handle
        re.compile(r"https?://", re.IGNORECASE),
        re.compile(r"phone|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", re.IGNORECASE),
    ]

    def process_batch(self, relationships: List[Any], context: ProcessingContext) -> List[Any]:
        self.stats["processed_count"] = len(relationships)
        filtered = 0
        kept: List[Any] = []
        for rel in relationships:
            pred = (getattr(rel, "relationship", "") or getattr(rel, "predicate", "") or "").strip().lower()
            tgt = getattr(rel, "target", "") or getattr(rel, "target_entity", "") or ""

            block = False
            if pred in self.PREDICATE_BLOCKLIST:
                block = True
            else:
                for pat in self.TARGET_PATTERNS:
                    if pat.search(tgt):
                        block = True
                        break
            if block:
                filtered += 1
                continue
            kept.append(rel)

        self.stats["modified_count"] = filtered
        self.stats["filtered"] = filtered
        logger.info(f"   {self.name}: filtered {filtered} contact-info relationships")
        return kept

