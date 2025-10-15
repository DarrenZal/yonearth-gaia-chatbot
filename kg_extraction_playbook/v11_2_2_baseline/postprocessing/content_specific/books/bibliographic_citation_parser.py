"""
Bibliographic Citation Parser Module

Detects and corrects authorship relationships from bibliographic citations.

Features:
- Citation pattern detection (Author, Name. Title...)
- Authorship reversal (Title authored Author → Author authored Title)
- Endorsement detection (praise quotes, forewords)
- Dedication detection (V8)

Version History:
- v1.0.0 (V6): Basic citation parsing and reversal
- v1.1.0 (V7): Enhanced endorsement detection
- v1.2.0 (V8): Dedication detection
- v1.3.0 (V11.2): Fixed dedication parsing logic, proper recipient splitting
"""

import re
import logging
import copy
from typing import Optional, List, Dict, Any, Tuple

from ...base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class BibliographicCitationParser(PostProcessingModule):
    """
    Parses bibliographic citations and corrects authorship attribution.

    Content Types: Books only
    Priority: 20 (runs early, after praise quote detection)
    """

    name = "BibliographicCitationParser"
    description = "Bibliographic citation parsing with fixed dedication detection"
    content_types = ["book"]
    priority = 20
    dependencies = ["PraiseQuoteDetector"]
    version = "1.3.0"  # V11.2 enhanced

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.citation_patterns = self.config.get('citation_patterns', [
            r'^([A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+,\s+[A-Z][a-z]+)*)\.',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\.',
        ])

        self.authorship_predicates = self.config.get('authorship_predicates', (
            'authored', 'wrote', 'published', 'created', 'composed',
            'edited', 'compiled', 'produced'
        ))

        # V7 ENHANCED: Expanded endorsement detection patterns
        self.endorsement_patterns = self.config.get('endorsement_patterns', [
            r'PRAISE FOR',
            r'TESTIMONIAL',
            r'ENDORSEMENT',
            r'(?:excellent|important|delightful|wonderful|brilliant|masterpiece|essential)',
            r'(?:tool|book|handbook|manual|guide|resource|work)',
            r'delighted to see',
            r'highly recommend',
            r'must[- ]read',
            r'invaluable resource',
            r'strongly recommend',
            r'(?:grateful|thrilled|honored|privileged|blessed)\s+to',
            r'(?:this|the)\s+(?:book|handbook|manual|guide|work)\s+(?:is|represents)\s+(?:an?\s+)?(?:excellent|wonderful|vital|essential|invaluable)',
            r'engage with this critical mission',
            r'excellent tool for',
            r'wonderful resource',
            r'writes in the foreword',
            r'in his foreword',
            r'in her foreword',
            r'foreword by'
        ])

        # V8 NEW: Dedication patterns
        self.dedication_patterns = self.config.get('dedication_patterns', [
            r'(?:this book is )?dedicated to (.+)',
            r'in memory of (.+)',
            r'for my (.+)',
            r'to my (.+)',
        ])

        self.dedication_verbs = self.config.get('dedication_verbs', [
            'dedicated', 'authored', 'wrote', 'authorship'
        ])

    def is_bibliographic_citation(self, evidence_text: str) -> bool:
        """Check if text matches bibliographic citation patterns"""
        for pattern in self.citation_patterns:
            if re.match(pattern, evidence_text.strip()):
                return True
        return False

    def is_endorsement(self, evidence_text: str, full_page_text: str = "") -> bool:
        """Check if this is an endorsement rather than authorship"""
        combined_text = (full_page_text + " " + evidence_text).lower()

        for pattern in self.endorsement_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True
        return False

    def is_dedication(self, evidence_text: str) -> Tuple[bool, Optional[str]]:
        """
        V8 NEW: Check if this is a dedication statement

        Returns:
            (is_dedication: bool, recipients: str or None)
        """
        evidence_lower = evidence_text.lower()

        for pattern in self.dedication_patterns:
            match = re.search(pattern, evidence_lower)
            if match:
                recipients = match.group(1)
                return True, recipients

        return False, None

    def extract_dedication_recipients(self, recipients_text: str) -> List[str]:
        """
        V11.2.2 FIX: Extract ONLY proper names from dedication text.

        Filters out:
        - Descriptive clauses (after "whose", "which", "that")
        - Abstract qualities ("brilliance", "courage")
        - Sentence fragments (ending with periods, containing verbs)
        - Relationship words ("children", "friends")

        Examples:
        - "my two children, Osha and Hunter, whose brilliance..." → ["Osha", "Hunter"]
        - "the Y on Earth Community" → ["Y on Earth Community"]
        - "community impact ambassadors who are informing..." → ["Community Impact Ambassadors"]
        """
        # Step 1: Stop at descriptive clause markers (preserve case!)
        desc_markers = ['whose', 'which', 'that', 'who are', 'who is', 'with a message']
        for marker in desc_markers:
            if marker in recipients_text.lower():
                # Only keep text before the marker (case-insensitive search, but preserve case)
                marker_pos = recipients_text.lower().index(marker)
                recipients_text = recipients_text[:marker_pos]
                break

        # Step 2: Split on commas and "and" to get potential recipients
        items = re.split(r',\s*|\s+and\s+', recipients_text, flags=re.IGNORECASE)

        # Step 3: Clean and filter each item
        valid_recipients = []

        for item in items:
            item = item.strip()
            if not item or len(item) < 2:
                continue

            # Remove possessives, articles, and "to"
            item = re.sub(r'^(my|our|the|to)\s+', '', item, flags=re.IGNORECASE).strip()

            # V11.2.2 FIX: Only remove relationship words if they're not part of an organization name
            # Organization keywords that indicate this is an entity name, not a relationship
            org_keywords = ['community', 'guild', 'association', 'foundation', 'institute',
                           'society', 'organization', 'group', 'network', 'alliance', 'impact']

            has_org_keyword = any(kw in item.lower() for kw in org_keywords)

            # Remove number words and relationship indicators ONLY if not an organization
            if not has_org_keyword:
                remove_patterns = [
                    r'\b(two|three|four|five|six|seven|eight|nine|ten)\b',
                    r'\b(children|child|kids|kid)\b',
                    r'\b(friends|friend)\b',
                    r'\b(colleagues|colleague)\b',
                    r'\b(members|member)\b',
                    r'\b(ambassadors|ambassador)\b'
                ]
                for pattern in remove_patterns:
                    item = re.sub(pattern, '', item, flags=re.IGNORECASE).strip()
            else:
                # For organizations, only remove number words
                remove_patterns = [
                    r'\b(two|three|four|five|six|seven|eight|nine|ten)\b',
                ]
                for pattern in remove_patterns:
                    item = re.sub(pattern, '', item, flags=re.IGNORECASE).strip()

            # Clean up extra whitespace
            item = ' '.join(item.split())

            if not item or len(item) < 2:
                continue

            # Step 4: Filter out invalid items

            # Skip if it's a sentence fragment (ends with punctuation)
            if item.endswith(('.', '!', '?', ',', ';')):
                continue

            # Skip if it contains verb indicators (sentence fragments)
            verb_words = ['give', 'gives', 'giving', 'are', 'is', 'was', 'were', 'has', 'have',
                         'will', 'can', 'could', 'would', 'should', 'do', 'does']
            if any(verb in item.lower().split() for verb in verb_words):
                continue

            # Skip if it's an abstract quality (not a proper noun)
            abstract_qualities = ['brilliance', 'courage', 'determination', 'compassion',
                                 'celebration', 'gratitude', 'joy', 'inspiration',
                                 'message', 'action', 'hope', 'future']
            if item.lower() in abstract_qualities:
                continue

            # Step 5: Keep if it looks like a proper name or organization
            words = item.split()

            # Check if it has capitalized words (proper nouns) or organization keywords
            has_capitals = any(word[0].isupper() for word in words if word)
            org_keywords = ['community', 'guild', 'association', 'foundation', 'institute',
                           'society', 'organization', 'group', 'network', 'alliance']
            has_org_keyword = any(kw in item.lower() for kw in org_keywords)

            if has_capitals or has_org_keyword:
                # Capitalize first letter of each word for consistency
                item = ' '.join(word.capitalize() for word in words)
                valid_recipients.append(item)

        return valid_recipients

    def should_reverse_authorship(self, rel: Any) -> Tuple[bool, bool]:
        """
        Determine if authorship should be reversed.

        Returns:
            (should_reverse: bool, is_endorsement: bool)
        """
        if rel.relationship not in self.authorship_predicates:
            return False, False

        evidence = rel.evidence_text.strip()

        if not self.is_bibliographic_citation(evidence):
            return False, False

        # Check if it's an endorsement
        page_text = rel.evidence.get('window_text', '')
        if self.is_endorsement(evidence, page_text):
            return False, True  # Don't reverse, but mark as endorsement

        # Check if source looks like title
        source_is_title = (
            len(rel.source.split()) > 3 or
            '"' in evidence[:50] or
            ':' in rel.source
        )

        # Check if target looks like author name
        target_words = rel.target.split()
        target_is_author = (
            2 <= len(target_words) <= 4 and
            all(w[0].isupper() for w in target_words if len(w) > 2)
        )

        should_reverse = source_is_title and target_is_author
        return should_reverse, False

    def reverse_authorship(self, rel: Any) -> Any:
        """Reverse source and target for misattributed authorship"""
        rel.source, rel.target = rel.target, rel.source
        rel.source_type, rel.target_type = rel.target_type, rel.source_type

        rel.evidence['source_surface'], rel.evidence['target_surface'] = \
            rel.evidence.get('target_surface'), rel.evidence.get('source_surface')

        if rel.flags is None:
            rel.flags = {}
        rel.flags['AUTHORSHIP_REVERSED'] = True
        rel.flags['correction_reason'] = 'bibliographic_citation_detected'

        return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to correct bibliographic citations"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        corrected = []
        correction_count = 0
        endorsement_count = 0
        dedication_count = 0

        for rel in relationships:
            # V8 NEW: Check for dedication statements first
            relationship_lower = rel.relationship.lower()
            evidence = rel.evidence_text

            if any(verb in relationship_lower for verb in self.dedication_verbs):
                is_dedication_stmt, recipients = self.is_dedication(evidence)

                if is_dedication_stmt and recipients:
                    # V11.2.2 FIX: Extract ONLY proper names, filter out descriptive text
                    cleaned_recipients = self.extract_dedication_recipients(recipients)

                    if not cleaned_recipients:
                        # No valid recipients, keep original
                        corrected.append(rel)
                        continue

                    # Create one relationship per recipient
                    for recipient in cleaned_recipients:
                        new_rel = copy.deepcopy(rel)
                        new_rel.relationship = 'dedicated'
                        new_rel.target = recipient

                        if new_rel.flags is None:
                            new_rel.flags = {}
                        new_rel.flags['DEDICATION_CORRECTED'] = True
                        new_rel.flags['original_relationship'] = rel.relationship
                        new_rel.flags['original_target'] = rel.target

                        corrected.append(new_rel)
                        dedication_count += 1

                    self.stats['modified_count'] += 1
                    continue  # Don't add original relationship

            # Original logic for authorship/endorsement
            should_reverse, is_endorsement_rel = self.should_reverse_authorship(rel)

            if is_endorsement_rel:
                # Convert 'authored' to 'endorsed'
                if rel.relationship in self.authorship_predicates:
                    rel.relationship = 'endorsed'
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['ENDORSEMENT_DETECTED'] = True
                    endorsement_count += 1
                    self.stats['modified_count'] += 1
            elif should_reverse:
                rel = self.reverse_authorship(rel)
                correction_count += 1
                self.stats['modified_count'] += 1

            corrected.append(rel)

        # Update stats
        self.stats['reversed'] = correction_count
        self.stats['endorsements'] = endorsement_count
        self.stats['dedications'] = dedication_count

        logger.info(
            f"   {self.name}: {correction_count} authorships reversed, "
            f"{endorsement_count} endorsements detected, {dedication_count} dedications corrected"
        )

        return corrected
