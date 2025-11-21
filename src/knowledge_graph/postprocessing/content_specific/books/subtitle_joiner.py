"""
Subtitle Joiner Module

Rehydrates incomplete book titles by extracting full "Title: Subtitle" from evidence.

Features:
- Detects INCOMPLETE_TITLE flags from TitleCompletenessValidator
- Extracts full titles from evidence_text using patterns
- Preserves quotes and capitalization
- Validates minimum length and capitalization requirements

Version History:
- v1.0.0 (V14.3.7): Initial implementation
"""

import re
import logging
from typing import Optional, List, Dict, Any

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class SubtitleJoiner(PostProcessingModule):
    """
    Rehydrates incomplete book titles from evidence text.

    Content Types: Books only
    Priority: 19 (just before BibliographicCitationParser)
    """

    name = "SubtitleJoiner"
    description = "Rehydrates incomplete book titles from evidence"
    content_types = ["book"]
    priority = 19
    dependencies = ["FrontMatterDetector"]
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.min_title_chars = self.config.get('min_title_chars', 15)
        self.min_capitalized_words = self.config.get('min_capitalized_words', 2)

        # Title extraction patterns (ordered by priority)
        self.title_patterns = [
            # Pattern 1: "Author. Title: Subtitle..." (bibliographic format)
            r'(?:[A-Z][a-z]+(?:,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)?)\.\s+([A-Z][^.:;]+(?::\s*[^.:;]+)?)',
            # Pattern 2: Quoted title with optional subtitle
            r'"([A-Z][^"]+(?::\s*[^"]+)?)"',
            r'"([A-Z][^"]+(?::\s*[^"]+)?)"',  # Curly quotes
            # Pattern 3: Title: Subtitle (standalone)
            r'([A-Z][A-Za-z\s,]+:\s*[A-Z][A-Za-z\s,]+)',
            # Pattern 4: Title — Subtitle or Title – Subtitle
            r'([A-Z][A-Za-z\s,]+[—–-]\s*[A-Z][A-Za-z\s,]+)',
            # Pattern 4: All-caps title with subtitle
            r'([A-Z][A-Z\s]+:\s*[A-Z][A-Za-z\s,]+)',
        ]

    def is_incomplete_title(self, rel: Any) -> bool:
        """Check if relationship has INCOMPLETE_TITLE flag"""
        if not hasattr(rel, 'flags') or not rel.flags:
            return False

        return rel.flags.get('INCOMPLETE_TITLE', False)

    def extract_full_title(self, evidence_text: str, current_target: str) -> Optional[str]:
        """
        Extract full title from evidence text.

        Args:
            evidence_text: The evidence text to search
            current_target: The current (incomplete) target

        Returns:
            Full title string, or None if cannot extract
        """
        for pattern in self.title_patterns:
            matches = re.findall(pattern, evidence_text)
            for match in matches:
                candidate = match.strip()

                # Must contain current target as substring (case-insensitive)
                if current_target.lower() not in candidate.lower():
                    continue

                # Validate minimum requirements
                if len(candidate) < self.min_title_chars:
                    continue

                # Count capitalized words
                words = candidate.split()
                cap_words = sum(1 for w in words if w and w[0].isupper())
                if cap_words < self.min_capitalized_words:
                    continue

                # Found valid candidate
                return candidate

        return None

    def rehydrate_title(self, rel: Any) -> Any:
        """Rehydrate incomplete title from evidence"""
        if not self.is_incomplete_title(rel):
            return rel

        # Only process Book targets
        if rel.target_type not in ['Book', 'Work', 'Essay']:
            return rel

        # Try to extract full title
        evidence_text = getattr(rel, 'evidence_text', '') or getattr(rel, 'context', '')
        if not evidence_text:
            return rel

        # Normalize newlines to improve pattern matching across breaks
        evidence_text = evidence_text.replace('\n', ' ')

        full_title = self.extract_full_title(evidence_text, rel.target)

        if full_title and full_title != rel.target:
            # Create new relationship with full title
            import copy
            new_rel = copy.deepcopy(rel)
            new_rel.target = full_title

            # Update flags
            if new_rel.flags is None:
                new_rel.flags = {}
            new_rel.flags['TITLE_REHYDRATED'] = True
            new_rel.flags['original_target'] = rel.target
            new_rel.flags['rehydration_source'] = 'evidence_text'

            # Remove INCOMPLETE_TITLE flag
            if 'INCOMPLETE_TITLE' in new_rel.flags:
                del new_rel.flags['INCOMPLETE_TITLE']

            logger.debug(f"Rehydrated title: '{rel.target}' → '{full_title}'")
            self.stats['rehydrated'] += 1
            return new_rel

        return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to rehydrate titles"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0
        self.stats['rehydrated'] = 0

        processed = []
        for rel in relationships:
            rehydrated_rel = self.rehydrate_title(rel)

            if rehydrated_rel != rel:
                self.stats['modified_count'] += 1

            processed.append(rehydrated_rel)

        logger.info(
            f"   {self.name}: {self.stats['rehydrated']} titles rehydrated"
        )

        return processed
