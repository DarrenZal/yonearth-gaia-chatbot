"""
Praise Quote Detector Module

Detects praise quotes in book front matter and corrects authorship attribution.

Features:
- Front matter page detection (pages 1-15)
- Endorsement language pattern matching
- Attribution marker detection (—Name, Title)
- Converts misattributed authorship to endorsement
- Copyright/authorship statement exclusion
- Author whitelist to prevent over-correction (V14.3.2.1)

Version History:
- v1.0.0 (V8): Initial Curator-generated implementation
- v1.1.0 (V8): Expanded detection patterns and strict filtering policy
- v1.2.0 (V8): Further expanded patterns and enforced strict filtering
- v1.3.0 (V8): Comprehensive pattern expansion and universal non-endorsement filtering
- v1.4.0 (V8): Added copyright/authorship exclusion patterns
- v1.5.0 (V14.3.2.1): Added author whitelist to prevent over-correction
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
    version = "1.5.0"  # V14.3.2.1: Added author whitelist

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # V14.3.2.1: Author whitelist to prevent over-correction
        self.known_authors = set()  # Populated from document metadata

        self.front_matter_pages = self.config.get('front_matter_pages', range(1, 15))

        self.copyright_patterns = self.config.get('copyright_patterns', [
            '©',
            'copyright',
            'all rights reserved',
            'authored by',
            'written by'
        ])

        self.endorsement_indicators = self.config.get('endorsement_indicators', [
            # Original indicators
            'inspirational', 'beautifully-informed', 'wonderful',
            'informative handbook', 'gives us', 'invites us',
            'delighted to see', 'highly recommend', 'essential',
            'excellent tool', 'grateful', 'honored',
            # Expanded action verbs
            'provides', 'guides', 'helps', 'contains', 'nourishes',
            'offers', 'delivers', 'presents', 'brings', 'shares',
            'explores', 'examines', 'reveals', 'illuminates', 'demonstrates',
            'teaches', 'shows', 'explains', 'clarifies', 'unpacks',
            'weaves', 'combines', 'integrates', 'blends', 'merges',
            # Praise adjectives
            'powerful', 'compelling', 'insightful', 'thoughtful', 'profound',
            'brilliant', 'remarkable', 'outstanding', 'exceptional', 'invaluable',
            'timely', 'important', 'necessary', 'vital', 'crucial',
            'comprehensive', 'thorough', 'detailed', 'rich', 'deep',
            'accessible', 'readable', 'engaging', 'captivating', 'moving',
            # Recommendation language
            'recommend', 'endorse', 'praise', 'commend', 'applaud',
            'must-read', 'should read', 'will enjoy', 'will benefit',
            'perfect for', 'ideal for', 'great for', 'excellent for',
            # Gratitude/appreciation
            'thank', 'appreciate', 'grateful for', 'indebted to',
            'fortunate', 'lucky', 'pleased', 'happy to see',
            # Impact language
            'changed my', 'transformed', 'influenced', 'inspired me',
            'opened my eyes', 'made me think', 'challenged me',
            # Additional action verbs
            'introduces', 'invites', 'encourages', 'empowers', 'enables',
            'supports', 'assists', 'facilitates', 'promotes', 'advances',
            'enriches', 'enhances', 'improves', 'strengthens', 'deepens',
            'broadens', 'expands', 'extends', 'develops', 'cultivates',
            # Additional praise language
            'masterful', 'superb', 'excellent', 'magnificent', 'splendid',
            'wonderful', 'marvelous', 'fantastic', 'terrific', 'fabulous',
            'impressive', 'noteworthy', 'significant', 'valuable', 'precious',
            'indispensable', 'essential', 'critical', 'key', 'fundamental',
            # Reader benefit language
            'readers will', 'you will', 'anyone who', 'those who',
            'helps readers', 'allows readers', 'enables readers',
            # Book quality descriptors
            'well-written', 'well-researched', 'well-crafted', 'carefully',
            'meticulously', 'thoughtfully', 'skillfully', 'expertly',
            # Endorsement phrases
            'a must', 'highly valuable', 'strongly recommend', 'cannot recommend enough',
            'do yourself a favor', 'don\'t miss', 'be sure to read',
            # Additional comprehensive patterns
            'sheds light', 'breaks down', 'makes clear', 'lays out', 'sets forth',
            'draws on', 'builds on', 'takes us', 'leads us', 'walks us through',
            'reminds us', 'calls us', 'challenges us', 'urges us', 'asks us',
            'celebrates', 'honors', 'highlights', 'showcases', 'features',
            'captures', 'conveys', 'expresses', 'articulates', 'communicates',
            'addresses', 'tackles', 'confronts', 'grapples with', 'deals with',
            'covers', 'spans', 'encompasses', 'includes', 'incorporates',
            'synthesizes', 'distills', 'summarizes', 'outlines', 'maps',
            'traces', 'chronicles', 'documents', 'records', 'catalogs',
            'analyzes', 'dissects', 'deconstructs', 'investigates', 'probes',
            'questions', 'interrogates', 'critiques', 'evaluates', 'assesses',
            'argues', 'contends', 'proposes', 'suggests', 'advocates',
            'champions', 'defends', 'supports', 'promotes', 'advances',
            # Emotional/experiential language
            'moved by', 'touched by', 'struck by', 'impressed by', 'amazed by',
            'surprised by', 'delighted by', 'pleased by', 'excited by', 'thrilled by',
            'fascinated by', 'intrigued by', 'captivated by', 'enchanted by', 'charmed by',
            # Comparative/superlative language
            'best', 'finest', 'greatest', 'most important', 'most valuable',
            'most comprehensive', 'most thorough', 'most accessible', 'most readable',
            'unlike any other', 'stands out', 'sets apart', 'distinguishes itself',
            # Necessity/urgency language
            'need to read', 'must have', 'should not miss', 'cannot afford to miss',
            'required reading', 'essential reading', 'necessary reading', 'vital reading',
            # Personal testimony
            'i found', 'i discovered', 'i learned', 'i appreciated', 'i enjoyed',
            'i was struck', 'i was moved', 'i was impressed', 'i was inspired',
            # Universal appeal language
            'everyone should', 'anyone interested', 'all who', 'every reader',
            'for anyone', 'for everyone', 'for all', 'for those',
            # Achievement/accomplishment language
            'succeeds in', 'manages to', 'accomplishes', 'achieves', 'attains',
            'pulls off', 'delivers on', 'lives up to', 'fulfills', 'realizes',
            # Contribution language
            'contributes', 'adds to', 'builds upon', 'extends', 'expands',
            'furthers', 'advances', 'moves forward', 'pushes forward', 'takes forward',
            # Clarity/understanding language
            'clear', 'lucid', 'understandable', 'comprehensible', 'intelligible',
            'makes sense of', 'helps understand', 'aids understanding', 'facilitates understanding',
            # Wisdom/insight language
            'wise', 'sage', 'astute', 'perceptive', 'discerning',
            'keen', 'sharp', 'penetrating', 'incisive', 'acute',
            # Originality/innovation language
            'original', 'innovative', 'groundbreaking', 'pioneering', 'trailblazing',
            'fresh', 'novel', 'new', 'unique', 'distinctive',
            # Completeness/thoroughness language
            'complete', 'full', 'entire', 'whole', 'total',
            'exhaustive', 'extensive', 'wide-ranging', 'far-reaching', 'all-encompassing'
        ])

        self.person_name_pattern = self.config.get(
            'person_name_pattern',
            r'—([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        )

        self.credentials_pattern = self.config.get(
            'credentials_pattern',
            r'(?:Founding|Director|Chair|Professor|PhD|Author|Champion|Executive|President|CEO|Founder|Editor|Researcher|Scholar|Expert|Consultant|Advisor|Leader|Advocate)'
        )

        self.authorship_verbs = self.config.get('authorship_verbs', [
            'authored', 'wrote', 'written by', 'author of'
        ])

    def _populate_author_whitelist(self, context: ProcessingContext) -> None:
        """
        V14.3.2.1: Extract known authors from document metadata.

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

    def is_praise_quote_context(self, evidence_text: str, page: int) -> bool:
        """Check if evidence suggests praise quote rather than authorship"""
        if page not in self.front_matter_pages:
            return False

        # Check for copyright/authorship exclusion patterns FIRST
        evidence_lower = evidence_text.lower()
        for pattern in self.copyright_patterns:
            if pattern in evidence_lower:
                logger.debug(f"Skipped praise detection - copyright statement detected: '{pattern}'")
                return False

        # Check for attribution marker (—Name)
        has_attribution = bool(re.search(self.person_name_pattern, evidence_text))

        # Check for credentials (not typical in authorship claims)
        has_credentials = bool(re.search(self.credentials_pattern, evidence_text))

        # Check for endorsement language
        has_endorsement_language = any(
            indicator in evidence_lower
            for indicator in self.endorsement_indicators
        )

        return has_attribution or has_credentials or has_endorsement_language

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process relationships to detect and correct praise quote misattributions"""

        # V14.3.2.1: Populate author whitelist from document metadata
        self._populate_author_whitelist(context)

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        corrected = []
        correction_count = 0
        filtered_count = 0
        skipped_known_authors = 0

        for rel in relationships:
            relationship_type = rel.relationship.lower()
            evidence = rel.evidence_text
            page = rel.evidence.get('page_number', 0)

            # V14.3.2.1: Check if source is known author FIRST (before praise quote detection)
            source_is_author = rel.source.lower().strip() in self.known_authors

            # Check if this is in a praise quote context
            if self.is_praise_quote_context(evidence, page):
                # V14.3.2.1 FIX: Don't flag actual author as praise quote
                if source_is_author:
                    # Keep relationship as-is (actual author, not praise quote)
                    corrected.append(rel)
                    skipped_known_authors += 1
                    logger.debug(f"Skipped known author '{rel.source}' - not a praise quote")
                # Strict filtering: only allow 'endorsed' relationships from praise contexts
                elif 'endorse' in relationship_type or relationship_type == 'endorsed':
                    # Already correct, keep as-is
                    corrected.append(rel)
                elif any(verb in relationship_type for verb in self.authorship_verbs):
                    # Correct authorship to appropriate praise relation
                    evidence_low = evidence.lower()
                    # If a foreword is mentioned, map to 'wrote foreword for'
                    rel.relationship = 'wrote foreword for' if 'foreword' in evidence_low else 'endorsed'

                    # Direction correction: ensure Person → (endorsed|wrote foreword for) → Book
                    src_type = (getattr(rel, 'source_type', '') or '').lower()
                    tgt_type = (getattr(rel, 'target_type', '') or '').lower()
                    if src_type == 'book' and tgt_type == 'person':
                        rel.source, rel.target = rel.target, rel.source
                        rel.source_type, rel.target_type = rel.target_type, rel.source_type
                        if isinstance(rel.evidence, dict) and 'source_surface' in rel.evidence and 'target_surface' in rel.evidence:
                            rel.evidence['source_surface'], rel.evidence['target_surface'] = \
                                rel.evidence.get('target_surface'), rel.evidence.get('source_surface')

                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRAISE_QUOTE_CORRECTED'] = True
                    rel.flags['correction_note'] = 'Changed from authorship to endorsement (praise quote detected)'
                    rel.flags['original_relationship'] = relationship_type
                    correction_count += 1
                    self.stats['modified_count'] += 1
                    corrected.append(rel)
                else:
                    # Filter out ALL other relationship types from praise contexts
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRAISE_QUOTE_FILTERED'] = True
                    rel.flags['filter_reason'] = f'Non-endorsement relationship in praise context: {relationship_type}'
                    rel.flags['original_relationship'] = relationship_type
                    filtered_count += 1
                    self.stats['modified_count'] += 1
                    # Don't add to corrected list (filtered out)
                    logger.debug(f"Filtered {relationship_type} from praise context on page {page}")
            else:
                # Not in praise context, keep as-is
                corrected.append(rel)

        # Update stats
        self.stats['corrected'] = correction_count
        self.stats['filtered'] = filtered_count
        self.stats['skipped_known_authors'] = skipped_known_authors

        logger.info(f"   {self.name}: {correction_count} praise quotes corrected to endorsements, {filtered_count} non-endorsement relationships filtered, {skipped_known_authors} known authors preserved")

        return corrected
