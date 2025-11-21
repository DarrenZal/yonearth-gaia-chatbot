"""
RhetoricalReclassifier Module

Ensures rhetorical questions and philosophical/abstract statements are marked
with appropriate classification flags and conservative knowledge scores so they
are not treated as factual errors downstream.

Rules:
- If relationship == 'poses question' OR target ends with '?' OR target_type == 'STATEMENT':
  - Add flags: QUESTION and/or PHILOSOPHICAL (if not already present)
  - Cap p_true at 0.4 (do not raise existing lower values)
  - Ensure source_type remains unchanged; do not drop content

Version: 1.0.0 (V14.3.10)
"""

from typing import Any, Dict, List, Optional

from ...base import PostProcessingModule, ProcessingContext


class RhetoricalReclassifier(PostProcessingModule):
    name = "RhetoricalReclassifier"
    description = "Marks rhetorical questions/abstract statements with conservative scoring"
    content_types = ["book"]
    priority = 114  # After ClaimClassifier (~110) and before Deduplicator (~118)
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_p_true = float(self.config.get('max_p_true', 0.4))
        self.max_p_true_normative = float(self.config.get('max_p_true_normative', 0.3))

    def process_batch(self, relationships: List[Any], context: ProcessingContext) -> List[Any]:
        self.stats['processed_count'] = len(relationships)
        modified = 0
        out: List[Any] = []

        for rel in relationships:
            changed = False
            try:
                rel_type = (getattr(rel, 'relationship', '') or '').lower()
                tgt = getattr(rel, 'target', '') or ''
                tgt_type = (getattr(rel, 'target_type', '') or '').lower()

                is_question = rel_type == 'poses question' or (isinstance(tgt, str) and tgt.strip().endswith('?'))
                is_statement = (tgt_type == 'statement')

                # Detect normative/future modality in statements
                tgn = (tgt or '').strip().lower()
                normative = any(w in tgn.split() for w in ['should', 'must', 'ought'])
                future = (' will ' in f' {tgn} ') or tgn.startswith('will ')

                if is_question or is_statement or normative or future:
                    # Ensure flags include QUESTION/PHILOSOPHICAL
                    flags = list(getattr(rel, 'classification_flags', []) or [])
                    if is_question and 'QUESTION' not in flags:
                        flags.append('QUESTION')
                    if 'PHILOSOPHICAL' not in flags:
                        flags.append('PHILOSOPHICAL')
                    if normative and 'NORMATIVE' not in flags:
                        flags.append('NORMATIVE')
                    if future and 'PROSPECTIVE' not in flags:
                        flags.append('PROSPECTIVE')
                    rel.classification_flags = flags

                    # Cap knowledge score
                    if hasattr(rel, 'p_true') and rel.p_true is not None:
                        cap = self.max_p_true_normative if (normative or future) else self.max_p_true
                        if rel.p_true > cap:
                            rel.p_true = cap
                            changed = True
                    changed = True
            except Exception:
                pass

            if changed:
                modified += 1
            out.append(rel)

        self.stats['modified_count'] = modified
        return out
