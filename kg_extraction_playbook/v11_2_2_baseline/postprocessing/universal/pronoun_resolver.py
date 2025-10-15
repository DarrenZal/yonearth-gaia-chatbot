"""
Pronoun Resolver Module

Resolves pronouns to their antecedents using context-aware resolution.

Features:
- Generic pronoun handling (we humans → humans, you → readers)
- Anaphoric resolution with expanding context windows
- Possessive pronoun resolution (my people → Slovenians)
- Multi-pass resolution (same sentence → previous sentence → paragraph)
- Author context awareness

Version History:
- v1.0.0 (V6): Basic resolution with generic pronouns
- v1.1.0 (V7): Multi-pass resolution with expanding windows
- v1.2.0 (V8): Possessive pronoun support + 5-sentence context
"""

import re
from typing import Optional, List, Dict, Any, Tuple
import logging

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class PronounResolver(PostProcessingModule):
    """
    Resolves pronouns to their antecedents using context-aware resolution.

    Content Types: Universal (works for all content types)
    Priority: 60 (mid-pipeline, after validation but before enrichment)
    """

    name = "PronounResolver"
    description = "Resolves pronouns to their antecedents with possessive support"
    content_types = ["all"]
    priority = 60
    dependencies = []  # No dependencies
    version = "1.2.0"  # V8 enhanced

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Subject pronouns
        self.pronouns = {
            'he', 'she', 'him', 'her', 'his', 'hers',
            'it', 'its',
            'we', 'us', 'our', 'ours',
            'they', 'them', 'their', 'theirs',
            'you', 'your', 'yours'
        }

        # Possessive pronouns (V8 NEW)
        self.possessive_pronouns = self.config.get('possessive_pronouns', [
            'my', 'our', 'your', 'their', 'his', 'her', 'its'
        ])

        # Possessive patterns for resolution
        self.possessive_patterns = [
            r'\b(my|our|your|their)\s+(people|ancestors|family|community)\b',
            r'\b(my|our|your|their)\s+(\w+)\b'  # Generic possessive
        ]

        # Generic pronoun mappings
        self.generic_pronouns = self.config.get('generic_pronouns', {
            'we humans': 'humans',
            'we each': 'individuals',
            'we all': 'people',
            'you': 'readers',
            'one': 'people'
        })

        # Resolution parameters
        self.resolution_window = self.config.get('resolution_window', 1000)  # characters
        self.context_window = self.config.get('context_window', 5)  # sentences

        # Author context (set from document metadata)
        self.author_context = None
        self.page_context = {}

    def is_pronoun(self, entity: str) -> bool:
        """Check if entity is a pronoun"""
        return entity.lower().strip() in self.pronouns

    def is_generic_pronoun(self, pronoun: str, evidence_text: str) -> Optional[str]:
        """
        Check if pronoun is generic (not anaphoric).

        Returns:
            Generic replacement if found, None otherwise
        """
        pronoun_lower = pronoun.lower()
        evidence_lower = evidence_text.lower()

        # Check for explicit generic patterns
        for pattern, replacement in self.generic_pronouns.items():
            if pattern in evidence_lower:
                return replacement

        # Check for instructional/imperative context for "you"
        if pronoun_lower in {'you', 'your'}:
            imperative_markers = ['can', 'should', 'must', 'try', 'start', 'begin', 'make', 'take']
            if any(marker in evidence_lower.split()[:10] for marker in imperative_markers):
                return 'readers'

        # Check for philosophical/general statements with "we"
        if pronoun_lower in {'we', 'our', 'us'}:
            general_markers = ['humans', 'humanity', 'people', 'society', 'world', 'planet', 'earth']
            if any(marker in evidence_lower for marker in general_markers):
                return 'humanity'

        return None

    def load_page_context(self, pages_with_text: List[Tuple[int, str]]):
        """Load page context for resolution"""
        self.page_context = {page_num: text for page_num, text in pages_with_text}

    def _get_context_sentences(self, context: str, window: int = 5) -> List[str]:
        """Extract sentences from context (V8 NEW)"""
        sentences = re.split(r'[.!?]+', context)
        return [s.strip() for s in sentences if s.strip()][-window:]

    def _extract_entities_from_context(self, sentences: List[str]) -> List[str]:
        """Extract named entities from context sentences (V8 NEW)"""
        entities = []
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 1:
                    if word not in ['The', 'A', 'An', 'In', 'On', 'At', 'To', 'For']:
                        entities.append(word)
        return entities

    def find_antecedent(
        self,
        pronoun: str,
        page_num: int,
        evidence_text: str
    ) -> Optional[str]:
        """
        Find antecedent for pronoun using multi-pass resolution.

        V7/V8 Enhancement: Multi-pass with expanding windows
        - Pass 1: Same sentence (0-100 chars back)
        - Pass 2: Previous sentence (100-500 chars back)
        - Pass 3: Paragraph scope (500-1000 chars back)
        """
        pronoun_lower = pronoun.lower()

        page_text = self.page_context.get(page_num, '')
        if not page_text:
            return None

        evidence_pos = page_text.find(evidence_text[:50])
        if evidence_pos == -1:
            return None

        # V8 NEW: Handle possessive pronouns first
        for pattern in self.possessive_patterns:
            match = re.search(pattern, pronoun_lower)
            if match:
                possessive = match.group(1)
                noun = match.group(2) if len(match.groups()) >= 2 else None

                # Get context for resolution
                context_start = max(0, evidence_pos - self.resolution_window)
                context = page_text[context_start:evidence_pos]

                # Context-specific resolution
                if 'slovenia' in context.lower() or 'slovenian' in context.lower():
                    if noun in ['people', 'ancestors', 'family']:
                        return 'Slovenians'

                # Author-specific resolution
                if self.author_context and possessive in ['my', 'our']:
                    if noun == 'people':
                        return f"{self.author_context}'s people"

                # Generic possessive resolution
                if noun:
                    return f"the {noun}"

        # Multi-pass resolution with expanding windows
        pass_windows = [
            ('same_sentence', 100),
            ('previous_sentence', 500),
            ('paragraph_scope', 1000)
        ]

        for pass_name, window_size in pass_windows:
            context_start = max(0, evidence_pos - window_size)
            context = page_text[context_start:evidence_pos]

            # Person pronouns
            if pronoun_lower in {'he', 'she', 'his', 'her', 'him'}:
                names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', context)
                if names:
                    return names[-1]  # Most recent name

            # Collective pronouns
            elif pronoun_lower in {'we', 'our', 'us', 'ours'}:
                # Look for organizations first
                orgs = re.findall(
                    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Foundation|Institute|Organization|Guild|Movement))\b',
                    context
                )
                if orgs:
                    return orgs[-1]

                # Cultural/national references
                cultural_refs = re.findall(
                    r'\b(my|our)\s+(people|country|nation|culture|heritage|land)\b',
                    context,
                    re.IGNORECASE
                )
                if cultural_refs:
                    nationality_pattern = r'\b([A-Z][a-z]+(?:ians?|ans?))\b'
                    nationalities = re.findall(nationality_pattern, context)
                    if nationalities:
                        return nationalities[-1]
                    return f"{cultural_refs[-1][0]} {cultural_refs[-1][1]}"

                # Look for collective nouns
                collectives = re.findall(
                    r'\b(humanity|people|society|humans|communities|families|Slovenians)\b',
                    context,
                    re.IGNORECASE
                )
                if collectives:
                    return collectives[-1]

                # Generic fallback for philosophical statements
                if any(term in context.lower() for term in ['soil', 'earth', 'planet']):
                    return 'humanity'

        return None

    def resolve_pronouns(self, rel: Any) -> Any:
        """Resolve pronouns in a single relationship"""
        page_num = rel.evidence.get('page_number', 0)
        evidence_text = rel.evidence_text

        # Resolve source
        if self.is_pronoun(rel.source):
            generic_replacement = self.is_generic_pronoun(rel.source, evidence_text)
            if generic_replacement:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['GENERIC_PRONOUN_RESOLVED_SOURCE'] = True
                rel.flags['original_source'] = rel.source
                rel.source = generic_replacement
                self.stats['modified_count'] += 1
            else:
                # Try anaphoric resolution
                antecedent = self.find_antecedent(rel.source, page_num, evidence_text)
                if antecedent:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRONOUN_RESOLVED_SOURCE'] = True
                    rel.flags['original_source'] = rel.source
                    rel.source = antecedent
                    self.stats['modified_count'] += 1
                else:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRONOUN_UNRESOLVED_SOURCE'] = True

        # Resolve target
        if self.is_pronoun(rel.target):
            generic_replacement = self.is_generic_pronoun(rel.target, evidence_text)
            if generic_replacement:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['GENERIC_PRONOUN_RESOLVED_TARGET'] = True
                rel.flags['original_target'] = rel.target
                rel.target = generic_replacement
                self.stats['modified_count'] += 1
            else:
                antecedent = self.find_antecedent(rel.target, page_num, evidence_text)
                if antecedent:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRONOUN_RESOLVED_TARGET'] = True
                    rel.flags['original_target'] = rel.target
                    rel.target = antecedent
                    self.stats['modified_count'] += 1
                else:
                    if rel.flags is None:
                        rel.flags = {}
                    rel.flags['PRONOUN_UNRESOLVED_TARGET'] = True

        return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to resolve pronouns"""

        # Set author context from metadata (V8 NEW)
        if context.document_metadata:
            self.author_context = context.document_metadata.get('author', None)

        # Load page context if available
        if context.pages_with_text:
            self.load_page_context(context.pages_with_text)

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0
        resolved_count = 0
        generic_resolved_count = 0
        unresolved_count = 0

        processed = []
        for rel in relationships:
            rel = self.resolve_pronouns(rel)

            # Count resolution types
            if rel.flags:
                if rel.flags.get('PRONOUN_RESOLVED_SOURCE') or rel.flags.get('PRONOUN_RESOLVED_TARGET'):
                    resolved_count += 1
                if rel.flags.get('GENERIC_PRONOUN_RESOLVED_SOURCE') or rel.flags.get('GENERIC_PRONOUN_RESOLVED_TARGET'):
                    generic_resolved_count += 1
                if rel.flags.get('PRONOUN_UNRESOLVED_SOURCE') or rel.flags.get('PRONOUN_UNRESOLVED_TARGET'):
                    unresolved_count += 1

            processed.append(rel)

        # Update stats
        self.stats['anaphoric_resolved'] = resolved_count
        self.stats['generic_resolved'] = generic_resolved_count
        self.stats['unresolved'] = unresolved_count

        logger.info(
            f"   {self.name}: {resolved_count} anaphoric + {generic_resolved_count} generic resolved, "
            f"{unresolved_count} flagged for review"
        )

        return processed
