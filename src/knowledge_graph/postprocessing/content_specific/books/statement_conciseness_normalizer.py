"""
StatementConcisenessNormalizer Module

Trims overly long statement and quote targets to concise spans to reduce
verbosity issues while preserving meaning.

Rules:
- For target_type == 'QUOTE' or relationship 'said':
  - Trim to max_chars (default 240), preserve sentence boundary when possible.
- For predicate 'poses question' or target_type == 'STATEMENT':
  - Trim to max_chars (default 240), keep trailing '?'.
- Normalize whitespace: collapse multiple spaces and newlines

Version: 1.0.0 (V14.3.10)
"""

import re
from typing import Any, Dict, List, Optional

from ...base import PostProcessingModule, ProcessingContext


class StatementConcisenessNormalizer(PostProcessingModule):
    name = "StatementConcisenessNormalizer"
    description = "Trims long statement/quote targets and normalizes whitespace"
    content_types = ["book"]
    priority = 113  # After ClaimClassifier (~110) and before RhetoricalReclassifier (~114)
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_chars = int(self.config.get('max_chars', 240))

    def _squash_ws(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s or "").strip()
        return s

    def _trim_to_boundary(self, s: str) -> str:
        if len(s) <= self.max_chars:
            return s
        cut = s[: self.max_chars]
        # Try to cut at last sentence end within the limit
        m = re.search(r"[\.\!\?]\s+[^\.\!\?]*$", cut)
        if m:
            idx = m.start() + 1
            return cut[:idx].strip()
        # Fallback to last space
        idx = cut.rfind(' ')
        return cut[: idx if idx > 0 else self.max_chars].strip()

    def process_batch(self, relationships: List[Any], context: ProcessingContext) -> List[Any]:
        out: List[Any] = []
        self.stats['processed_count'] = len(relationships)
        changed = 0

        for rel in relationships:
            try:
                rel_type = (getattr(rel, 'relationship', '') or '').lower()
                tgt_type = (getattr(rel, 'target_type', '') or '').lower()
                tgt = getattr(rel, 'target', '')
                if not isinstance(tgt, str) or not tgt:
                    out.append(rel)
                    continue

                original = tgt
                tgt_norm = self._squash_ws(tgt)

                if tgt_type == 'quote' or rel_type == 'said':
                    tgt_new = self._trim_to_boundary(tgt_norm)
                elif rel_type == 'poses question' or tgt_type == 'statement' or tgt_norm.endswith('?'):
                    tgt_new = self._trim_to_boundary(tgt_norm)
                    if not tgt_new.endswith('?') and tgt_norm.endswith('?'):
                        tgt_new = (tgt_new + '?').strip()
                else:
                    tgt_new = tgt_norm

                if tgt_new != original:
                    rel.target = tgt_new
                    # reflect trimmed context too if present
                    if hasattr(rel, 'evidence_text'):
                        rel.evidence_text = tgt_new
                    changed += 1
            except Exception:
                pass
            out.append(rel)

        self.stats['modified_count'] = changed
        return out

