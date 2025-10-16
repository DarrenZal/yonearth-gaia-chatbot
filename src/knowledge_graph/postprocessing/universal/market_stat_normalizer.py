from __future__ import annotations

from typing import List, Dict, Any


class MarketStatNormalizer:
    """
    Normalize simple market-stat patterns and social-post predicates.

    - Map relationship 'posted' -> 'published' for social posts.
    - Normalize LOHAS-like stats:
        * "half a trillion dollars ... annual sales worldwide" -> "$500 billion annual sales worldwide"
        * "one out of five" -> "about 20% of people in the United States" (when US is implied)
    - Ensure article-like targets use target_type 'Article' when predicate is 'published' and context mentions 'post'/'LinkedIn'.
    """

    name = "MarketStatNormalizer"
    version = "1.0.0"

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def process_batch(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rel in relationships:
            r = dict(rel)
            pred = (r.get('relationship') or '').strip().lower()
            ctx = (r.get('context') or '').lower()
            tgt = (r.get('target') or '')
            src = (r.get('source') or '')

            # posted -> published for social content
            if pred == 'posted':
                r['relationship'] = 'published'
                pred = 'published'

            # article typing for social posts
            if pred == 'published' and ('linkedin' in ctx or 'post' in ctx or 'tweet' in ctx):
                r['target_type'] = r.get('target_type') or 'Article'

            # LOHAS sales normalization
            low = (tgt + ' ' + (r.get('context') or '')).lower()
            if 'half a trillion' in low and 'annual sales' in low and 'worldwide' in low:
                r['relationship'] = 'has'
                r['target'] = 'approximately $500 billion annual sales worldwide'

            # one out of five -> about 20%
            if 'one out of five' in low and ('us' in low or 'united states' in low):
                r['relationship'] = 'includes'
                r['target'] = 'about 20% of people in the United States'

            out.append(r)

        return out

