"""
AuthorPlaceholderResolver Module

Resolves generic author placeholders (e.g., "AUTHOR", "the author") to the
actual author name derived from document metadata. This improves entity
specificity and prevents vague placeholders from being blocked later.

Behavior:
- If document_metadata.author is present, replace occurrences of common
  placeholders when they appear as the full source or target entity.
- Normalize source/target types to "Person" when replacement occurs.
- Record basic statistics about replacements performed.

Version History:
- 1.0.0 (V14.3.10): Initial implementation for OBD chapter processing
"""

import re
from typing import Any, Dict, List, Optional

from ...base import PostProcessingModule, ProcessingContext


class AuthorPlaceholderResolver(PostProcessingModule):
    """Replace generic author placeholders with the actual author name."""

    name = "AuthorPlaceholderResolver"
    description = "Resolves 'AUTHOR'/'the author' placeholders to actual author"
    content_types = ["book"]
    priority = 35  # After ContextEnricher (30), before ListSplitter (40)
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Strict placeholder forms (full-string matches, case-insensitive)
        self.placeholders = {
            "author",
            "the author",
            "book author",
            "this book's author",
            "the book's author",
            "the book’s author",
            "author of this book",
        }

        # Optional: allow uppercased AUTHOR
        self.upper_placeholder = "AUTHOR"

        # Minimum length of resolved name to accept
        self.min_author_len = int(self.config.get("min_author_len", 3))

    def is_placeholder(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        s = text.strip()
        if not s:
            return False
        if s.upper() == self.upper_placeholder:
            return True
        return s.lower() in self.placeholders

    def resolve_name(self, name: str, context: ProcessingContext) -> Optional[str]:
        try:
            author = (getattr(context, 'document_metadata', {}) or {}).get('author')
        except Exception:
            author = None
        if isinstance(author, str) and len(author.strip()) >= self.min_author_len:
            return author.strip()
        return None

    def process_batch(self, relationships: List[Any], context: ProcessingContext) -> List[Any]:
        processed: List[Any] = []
        self.stats['processed_count'] = len(relationships)
        replaced = 0

        for rel in relationships:
            try:
                changed = False

                # Resolve source placeholder
                src = getattr(rel, 'source', None)
                if self.is_placeholder(src):
                    resolved = self.resolve_name(src, context)
                    if resolved:
                        rel.source = resolved
                        if hasattr(rel, 'source_type'):
                            rel.source_type = "Person"
                        changed = True

                # Resolve target placeholder
                tgt = getattr(rel, 'target', None)
                if self.is_placeholder(tgt):
                    resolved = self.resolve_name(tgt, context)
                    if resolved:
                        rel.target = resolved
                        if hasattr(rel, 'target_type'):
                            rel.target_type = "Person"
                        changed = True

                if changed:
                    if not hasattr(rel, 'flags') or rel.flags is None:
                        rel.flags = {}
                    rel.flags['AUTHOR_PLACEHOLDER_RESOLVED'] = True
                    replaced += 1
            except Exception:
                # Preserve original on any failure
                pass

            processed.append(rel)

        self.stats['modified_count'] = replaced
        self.stats['resolved'] = replaced
        return processed

