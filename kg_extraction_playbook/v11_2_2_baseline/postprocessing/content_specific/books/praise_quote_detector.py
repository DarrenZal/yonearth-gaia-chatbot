"""
Praise Quote Detector Module

Detects praise quotes in book front matter and corrects authorship attribution.

Features:
- Front matter page detection (pages 1-15)
- Endorsement language pattern matching
- Attribution marker detection (—Name, Title)
- Converts misattributed authorship to endorsement

Version History:
- v1.0.0 (V8): Initial Curator-generated implementation
"""

import re
import logging
from typing import Optional, List, Dict, Any

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class PraiseQuoteDetector(PostProcessingModule):
    """
    Detects praise quotes in front matter and corrects authorship.

    Content Types: Books only
    Priority: 10 (runs very early, before other processing)
    """

    name = "PraiseQuoteDetector"
    description = "Detects/corrects praise quotes in front matter"
    content_types = ["book"]
    priority = 10
    dependencies = []
    version = "1.0.0"  # V8 new

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.front_matter_pages = self.config.get('front_matter_pages', range(1, 15))

        self.endorsement_indicators = self.config.get('endorsement_indicators', [
            'inspirational', 'beautifully-informed', 'wonderful',
            'informative handbook', 'gives us', 'invites us',
            'delighted to see', 'highly recommend', 'essential',
            'excellent tool', 'grateful', 'honored'
        ])

        self.person_name_pattern = self.config.get(
            'person_name_pattern',
            r'—([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        )

        self.credentials_pattern = self.config.get(
            'credentials_pattern',
            r'(?:Founding|Director|Chair|Professor|PhD|Author|Champion)'
        )

        self.authorship_verbs = self.config.get('authorship_verbs', [
            'authored', 'wrote', 'written by', 'author of'
        ])

    def is_praise_quote_context(self, evidence_text: str, page: int) -> bool:
        """Check if evidence suggests praise quote rather than authorship"""
        if page not in self.front_matter_pages:
            return False

        # Check for attribution marker (—Name)
        has_attribution = bool(re.search(self.person_name_pattern, evidence_text))

        # Check for credentials (not typical in authorship claims)
        has_credentials = bool(re.search(self.credentials_pattern, evidence_text))

        # Check for endorsement language
        has_endorsement_language = any(
            indicator in evidence_text.lower()
            for indicator in self.endorsement_indicators
        )

        return has_attribution or has_credentials or has_endorsement_language

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process relationships to detect and correct praise quote misattributions"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        corrected = []
        correction_count = 0

        for rel in relationships:
            relationship_type = rel.relationship.lower()
            evidence = rel.evidence_text
            page = rel.evidence.get('page_number', 0)

            # Check if this is authorship claim in praise quote context
            if any(verb in relationship_type for verb in self.authorship_verbs):
                if self.is_praise_quote_context(evidence, page):
                    # Correct to endorsement
                    rel.relationship = 'endorsed'
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRAISE_QUOTE_CORRECTED'] = True
                    rel.flags['correction_note'] = 'Changed from authorship to endorsement (praise quote detected)'
                    correction_count += 1
                    self.stats['modified_count'] += 1

            corrected.append(rel)

        # Update stats
        self.stats['corrected'] = correction_count

        logger.info(f"   {self.name}: {correction_count} praise quotes corrected to endorsements")

        return corrected
