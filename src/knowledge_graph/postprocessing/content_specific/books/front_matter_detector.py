"""
Front Matter Detector Module (V14.3.2.1)

Detects and corrects relationships extracted from front matter sections
(foreword, preface, introduction, praise quotes).

Purpose:
Prevents misattribution of foreword signatures as book authorship. For example:
- "With Love, Lily Sophia" at end of foreword → "wrote foreword for", NOT "authored"
- Actual author signatures → keep as "authored" (protected by author whitelist)

Detection Approach:
1. Page-based: Front matter typically on pages 1-15
2. Keyword-based: Foreword, preface, introduction, dedication keywords
3. Signature pattern-based: "With Love", "Gratefully", etc.

Version History:
- v1.0.0 (V14.3.2.1): Initial implementation for foreword signature correction
"""

import re
import logging
from typing import Optional, List, Dict, Any
import copy

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class FrontMatterDetector(PostProcessingModule):
    """
    Detects relationships extracted from front matter and corrects authorship.

    Content Types: Books only
    Priority: 12 (after MetadataFilter, before BibliographicCitationParser)
    Dependencies: None (but benefits from PraiseQuoteDetector's author whitelist)
    """

    name = "FrontMatterDetector"
    description = "Detects/corrects foreword and front matter signatures"
    content_types = ["book"]
    priority = 12  # After MetadataFilter (11), before BibliographicCitationParser (20)
    dependencies = []
    version = "1.0.0"  # V14.3.2.1

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Configuration
        self.front_matter_page_limit = self.config.get('front_matter_page_limit', 15)

        # Front matter keywords
        self.front_matter_keywords = set(self.config.get('front_matter_keywords', [
            'foreword',
            'preface',
            'introduction',
            'dedication',
            'with love and hope',
            'endorsement',
            'praise',
            'what people are saying',
            'advance praise',
            'testimonials',
            'acknowledgments',
            'acknowledgements'
        ]))

        # Signature patterns (lowercase for matching)
        self.signature_patterns = self.config.get('signature_patterns', [
            r'with love',
            r'gratefully',
            r'sincerely',
            r'in service',
            r'with gratitude',
            r'with appreciation',
            r'respectfully',
            r'humbly',
            r'with joy',
            r'in celebration'
        ])

        # Known authors (populated from document metadata)
        self.known_authors = set()

    def _populate_author_whitelist(self, context: ProcessingContext) -> None:
        """
        Extract known authors from document metadata.

        Creates multiple variants of author names to catch different spellings:
        - Full name: "Aaron William Perry"
        - First + Last: "Aaron Perry"
        - Lowercase variants
        """
        if not context.document_metadata:
            return

        author = context.document_metadata.get('author')
        if not author:
            return

        # Add full name
        self.known_authors.add(author.lower().strip())

        # Add first + last name variant
        parts = author.split()
        if len(parts) >= 2:
            # "Aaron William Perry" → "Aaron Perry"
            first_last = f"{parts[0]} {parts[-1]}"
            self.known_authors.add(first_last.lower().strip())

        logger.debug(f"Author whitelist populated: {self.known_authors}")

    def is_front_matter_signature(self, evidence_text: str, page: int) -> bool:
        """
        Detect if evidence is from front matter signature.

        Args:
            evidence_text: The evidence text to check
            page: The page number

        Returns:
            True if evidence suggests front matter signature
        """
        # Check page number first (front matter usually < page 15)
        if page > self.front_matter_page_limit:
            return False

        evidence_lower = evidence_text.lower()

        # Check for front matter keywords
        for keyword in self.front_matter_keywords:
            if keyword in evidence_lower:
                logger.debug(f"Front matter keyword detected: '{keyword}' on page {page}")
                return True

        # Check for signature patterns
        for pattern in self.signature_patterns:
            if re.search(pattern, evidence_lower):
                logger.debug(f"Signature pattern detected: '{pattern}' on page {page}")
                return True

        return False

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process relationships to detect and correct front matter authorship"""

        # Populate author whitelist from document metadata
        self._populate_author_whitelist(context)

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        corrected = []
        correction_count = 0

        for rel in relationships:
            relationship_type = rel.relationship.lower().strip()
            evidence = rel.evidence_text
            page = rel.evidence.get('page_number', 0)

            # Only process authorship relationships
            if relationship_type not in ['authored', 'wrote', 'author of']:
                corrected.append(rel)
                continue

            # Check if source is known author (should keep as "authored")
            source_is_author = rel.source.lower().strip() in self.known_authors

            if source_is_author:
                # Keep relationship as-is (actual author)
                corrected.append(rel)
                logger.debug(f"Preserved author '{rel.source}' on page {page}")
                continue

            # Check if this is a front matter signature
            if self.is_front_matter_signature(evidence, page):
                # Convert authorship to "wrote foreword for" and ensure correct DIRECTION: Person → Book
                new_rel = copy.deepcopy(rel)
                new_rel.relationship = 'wrote foreword for'

                # Direction fix: if source is a Book and target is a Person, swap
                src_type = (getattr(new_rel, 'source_type', '') or '').lower()
                tgt_type = (getattr(new_rel, 'target_type', '') or '').lower()
                if src_type == 'book' and tgt_type == 'person':
                    new_rel.source, new_rel.target = new_rel.target, new_rel.source
                    new_rel.source_type, new_rel.target_type = new_rel.target_type, new_rel.source_type
                    # Swap evidence surfaces if present
                    if isinstance(new_rel.evidence, dict):
                        if 'source_surface' in new_rel.evidence and 'target_surface' in new_rel.evidence:
                            new_rel.evidence['source_surface'], new_rel.evidence['target_surface'] = \
                                new_rel.evidence.get('target_surface'), new_rel.evidence.get('source_surface')

                if new_rel.flags is None:
                    new_rel.flags = {}
                new_rel.flags['FRONT_MATTER_CORRECTED'] = True
                new_rel.flags['correction_note'] = 'Changed from authorship to foreword (front matter signature detected)'
                new_rel.flags['original_relationship'] = relationship_type

                correction_count += 1
                self.stats['modified_count'] += 1

                corrected.append(new_rel)
                logger.debug(
                    f"Corrected front matter: '{rel.source}' authored → wrote foreword for "
                    f"'{getattr(rel, 'target', '')}' (page {page})"
                )
            else:
                # Not in front matter, keep as-is
                corrected.append(rel)

        # Update stats
        self.stats['corrected'] = correction_count

        logger.info(
            f"   {self.name}: {correction_count} front matter signatures corrected "
            f"to 'wrote foreword for'"
        )

        return corrected
