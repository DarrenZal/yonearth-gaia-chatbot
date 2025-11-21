"""
VagueDemographicReclassifier Module

Flags and down-scores relationships whose source or target entities are
overly generic demographic references (e.g., "we", "people", "society",
"the company", "this body") instead of concrete named entities.

Behavior:
- Does not drop relationships; adds classification flags and caps p_true.
- Works after ClaimClassifier; before RhetoricalReclassifier/Deduplicator.

Version: 1.0.0 (V14.3.10)
"""

from typing import Any, Dict, List, Optional

from ..base import PostProcessingModule, ProcessingContext


class VagueDemographicReclassifier(PostProcessingModule):
    name = "VagueDemographicReclassifier"
    description = "Flags generic demographic entities and caps knowledge scores"
    content_types = ["all"]
    priority = 112  # After ClaimClassifier (~110), before Statement/Rhetorical normalizers (~113-114)
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_p_true = float(self.config.get('max_p_true', 0.4))
        self.generic_terms = set(
            term.lower() for term in self.config.get(
                'generic_terms', [
                    'we', 'people', 'society', 'humanity', 'community', 'communities',
                    'the company', 'the organization', 'this body', 'the body',
                    'they', 'them', 'our', 'us'
                ]
            )
        )
        self.generic_heads = set(
            head.lower() for head in self.config.get(
                'generic_heads', [
                    'company', 'organization', 'group', 'body', 'committee', 'board',
                    'government', 'team', 'community', 'movement', 'institution',
                    'agency', 'council', 'department', 'church', 'party'
                ]
            )
        )

    def is_generic(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        s = text.strip().lower()
        if not s:
            return False
        # exact match or startswith articles + term
        if s in self.generic_terms:
            return True
        if s.startswith('the ') and s[4:] in self.generic_terms:
            return True
        if s.startswith('this ') and s[5:] in self.generic_terms:
            return True
        # Head-noun heuristic: 'the company', 'the organization', etc.
        if s.startswith('the ') or s.startswith('this '):
            tokens = s.split()
            if len(tokens) >= 2:
                head = tokens[1].rstrip('.,;:!?')
                if head in self.generic_heads or (head.endswith('s') and head[:-1] in self.generic_heads):
                    return True
        return False

    def process_batch(self, relationships: List[Any], context: ProcessingContext) -> List[Any]:
        out: List[Any] = []
        self.stats['processed_count'] = len(relationships)
        modified = 0

        for rel in relationships:
            try:
                src = getattr(rel, 'source', '')
                tgt = getattr(rel, 'target', '')
                flagged = False
                flags = list(getattr(rel, 'classification_flags', []) or [])

                if self.is_generic(src):
                    if 'GENERIC_SOURCE' not in flags:
                        flags.append('GENERIC_SOURCE')
                        flagged = True
                if self.is_generic(tgt):
                    if 'GENERIC_TARGET' not in flags:
                        flags.append('GENERIC_TARGET')
                        flagged = True

                if flagged:
                    rel.classification_flags = flags
                    if hasattr(rel, 'p_true') and rel.p_true is not None and rel.p_true > self.max_p_true:
                        rel.p_true = self.max_p_true
                    modified += 1
            except Exception:
                pass
            out.append(rel)

        self.stats['modified_count'] = modified
        return out
