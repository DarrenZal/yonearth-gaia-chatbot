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
- v1.3.0 (V9): Enhanced possessive pronoun resolution with entity linking
- v1.4.0 (V10): Full possessive pronoun resolution implementation
- v1.5.0 (V11): Batch processing optimization for possessive pronouns
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
    version = "1.6.0"  # V14.3.4: Entity-type-aware pronoun detection

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

        # V14.3.4 NEW: Entity types that should skip pronoun resolution
        # These are proper nouns where "our", "my", etc. are part of the name
        self.skip_pronoun_types = self.config.get('skip_pronoun_types', [
            'Book', 'Organization', 'Location', 'Event', 'Product', 'Work', 'Title'
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
        self.author_heritage = None
        self.author_nationality = None
        self.page_context = {}
        
        # V11: Cache for resolved possessive pronouns
        self.possessive_cache = {}

    def is_pronoun(self, entity: str) -> bool:
        """Check if entity is a pronoun"""
        return entity.lower().strip() in self.pronouns

    def is_possessive_pronoun(self, entity: str) -> bool:
        """Check if entity contains a possessive pronoun (V9 NEW)"""
        entity_lower = entity.lower().strip()
        for possessive in self.possessive_pronouns:
            if entity_lower.startswith(possessive + ' '):
                return True
        return False

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

        # V14.3.1: Enhanced fallback for aspirational/general statements
        # These patterns indicate abstract/general usage rather than specific antecedents
        if pronoun_lower in {'we', 'us'}:
            aspirational_markers = [
                'we can', 'we will', 'we should', 'we must', 'we have the opportunity',
                'we have the choice', 'let us', 'we are embarking', 'we get to choose',
                'if we', 'when we', 'as we', 'we need to', 'we ought to'
            ]
            if any(marker in evidence_lower for marker in aspirational_markers):
                return 'humanity'

        if pronoun_lower == 'our':
            aspirational_markers = [
                'our planet', 'our world', 'our future', 'our communities', 'our liberty',
                'our health', 'our well-being', 'our intelligence'
            ]
            if any(marker in evidence_lower for marker in aspirational_markers):
                return 'human'

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

    def resolve_possessive_pronouns(
        self,
        entity: str,
        page_num: int,
        evidence_text: str
    ) -> Optional[str]:
        """
        Resolve possessive pronouns using context-aware entity linking (V10 FULL IMPLEMENTATION).
        
        Args:
            entity: Entity containing possessive pronoun (e.g., 'my people')
            page_num: Page number for context lookup
            evidence_text: Evidence text for context
            
        Returns:
            Resolved entity or None if unresolvable
        """
        entity_lower = entity.lower().strip()
        
        # V11: Check cache first
        cache_key = f"{entity_lower}:{page_num}"
        if cache_key in self.possessive_cache:
            return self.possessive_cache[cache_key]
        
        # Extract possessive pronoun and noun
        possessive_match = None
        noun = None
        
        for pattern in self.possessive_patterns:
            match = re.match(pattern, entity_lower)
            if match:
                possessive_match = match.group(1)
                if len(match.groups()) >= 2:
                    noun = match.group(2)
                break
        
        if not possessive_match or not noun:
            return None
        
        # Get page context for resolution
        page_text = self.page_context.get(page_num, '')
        evidence_pos = page_text.find(evidence_text[:50]) if page_text else -1
        
        # Build context window
        context = ''
        if evidence_pos != -1:
            context_start = max(0, evidence_pos - self.resolution_window)
            context = page_text[context_start:evidence_pos]
        
        resolved = None
        
        # Strategy 1: Author heritage/nationality metadata (highest priority)
        if possessive_match in ['my', 'our']:
            if self.author_heritage:
                # Map heritage to nationality/ethnicity
                heritage_mappings = {
                    'slovenian': 'Slovenians',
                    'slovenia': 'Slovenians',
                    'italian': 'Italians',
                    'italy': 'Italians',
                    'german': 'Germans',
                    'germany': 'Germans',
                    'french': 'French',
                    'france': 'French',
                    'spanish': 'Spanish',
                    'spain': 'Spanish',
                    'irish': 'Irish',
                    'ireland': 'Irish',
                    'scottish': 'Scots',
                    'scotland': 'Scots',
                    'english': 'English',
                    'england': 'English',
                    'american': 'Americans',
                    'america': 'Americans',
                }
                
                heritage_lower = self.author_heritage.lower()
                for key, value in heritage_mappings.items():
                    if key in heritage_lower:
                        if noun in ['people', 'ancestors', 'family', 'community', 'heritage', 'culture']:
                            resolved = value
                            break
            
            if not resolved and self.author_nationality:
                nationality_lower = self.author_nationality.lower()
                if noun in ['people', 'ancestors', 'family', 'community', 'heritage', 'culture']:
                    # Capitalize and pluralize if needed
                    if nationality_lower.endswith('s'):
                        resolved = self.author_nationality.capitalize()
                    else:
                        resolved = self.author_nationality.capitalize() + 's'
        
        # Strategy 2: Context-based entity linking
        if not resolved and context:
            # Look for nationality/ethnicity mentions in context
            nationality_patterns = [
                r'\b([A-Z][a-z]+(?:ian|an|ese|ish|i)s?)\b',  # Slovenian, American, Chinese, Irish, Israeli
                r'\b(Scots|Welsh|French|Dutch|Swiss|Czech)\b',  # Irregular forms
            ]
            
            nationalities = []
            for pattern in nationality_patterns:
                matches = re.findall(pattern, context)
                nationalities.extend(matches)
            
            if nationalities:
                # Use most recent nationality mention
                nationality = nationalities[-1]
                if noun in ['people', 'ancestors', 'family', 'community', 'heritage', 'culture', 'land', 'country']:
                    # Ensure plural form
                    if not nationality.endswith('s'):
                        nationality = nationality + 's'
                    resolved = nationality
            
            # Look for country names
            if not resolved:
                country_pattern = r'\b(Slovenia|Italy|Germany|France|Spain|Ireland|Scotland|England|America|USA|China|Japan|India)\b'
                countries = re.findall(country_pattern, context, re.IGNORECASE)
                
                if countries:
                    country_to_nationality = {
                        'slovenia': 'Slovenians',
                        'italy': 'Italians',
                        'germany': 'Germans',
                        'france': 'French',
                        'spain': 'Spanish',
                        'ireland': 'Irish',
                        'scotland': 'Scots',
                        'england': 'English',
                        'america': 'Americans',
                        'usa': 'Americans',
                        'china': 'Chinese',
                        'japan': 'Japanese',
                        'india': 'Indians',
                    }
                    
                    country_lower = countries[-1].lower()
                    if country_lower in country_to_nationality:
                        if noun in ['people', 'ancestors', 'family', 'community', 'heritage', 'culture', 'land', 'country']:
                            resolved = country_to_nationality[country_lower]
            
            # Look for cultural/ethnic group mentions
            if not resolved:
                cultural_pattern = r'\b([A-Z][a-z]+)\s+(?:culture|heritage|tradition|community)\b'
                cultural_matches = re.findall(cultural_pattern, context)
                
                if cultural_matches:
                    cultural_group = cultural_matches[-1]
                    if noun in ['people', 'ancestors', 'family', 'community', 'heritage', 'culture']:
                        # Pluralize if needed
                        if not cultural_group.endswith('s'):
                            cultural_group = cultural_group + 's'
                        resolved = cultural_group
        
        # Strategy 3: Evidence text analysis
        if not resolved and evidence_text:
            evidence_lower = evidence_text.lower()
            
            # Check for nationality/ethnicity in evidence
            nationality_patterns = [
                r'\b([A-Z][a-z]+(?:ian|an|ese|ish|i)s?)\b',
                r'\b(Scots|Welsh|French|Dutch|Swiss|Czech)\b',
            ]
            
            for pattern in nationality_patterns:
                matches = re.findall(pattern, evidence_text)
                if matches:
                    nationality = matches[-1]
                    if noun in ['people', 'ancestors', 'family', 'community', 'heritage', 'culture']:
                        if not nationality.endswith('s'):
                            nationality = nationality + 's'
                        resolved = nationality
                        break
        
        # Strategy 4: Generic fallback based on noun type
        if not resolved:
            if noun in ['people', 'ancestors', 'family', 'community']:
                # If we have author context but no specific heritage, use author's name
                if self.author_context and possessive_match in ['my', 'our']:
                    resolved = f"{self.author_context}'s {noun}"
                else:
                    # Generic fallback
                    resolved = f"the {noun}"
        
        # V11: Cache the result
        if resolved:
            self.possessive_cache[cache_key] = resolved
        
        return resolved

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

        # V14.3.1: Final fallback for unresolved collective pronouns
        # If we/us/our cannot be resolved to a specific entity, use 'humanity' as generic fallback
        if pronoun_lower in {'we', 'us'}:
            return 'humanity'
        elif pronoun_lower in {'our', 'ours'}:
            return 'human'

        return None

    def resolve_pronouns(self, rel: Any) -> Any:
        """Resolve pronouns in a single relationship"""
        page_num = rel.evidence.get('page_number', 0)
        evidence_text = rel.evidence_text

        # Resolve source
        # V14.3.4 NEW: Skip pronoun resolution for proper noun types
        source_type = getattr(rel, 'source_type', None)
        should_skip_source = source_type in self.skip_pronoun_types if source_type else False

        # V10: Check for possessive pronouns first
        if self.is_possessive_pronoun(rel.source) and not should_skip_source:
            resolved = self.resolve_possessive_pronouns(rel.source, page_num, evidence_text)
            if resolved:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['POSSESSIVE_PRONOUN_RESOLVED_SOURCE'] = True
                rel.flags['original_source'] = rel.source
                rel.source = resolved
                self.stats['modified_count'] += 1
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['POSSESSIVE_PRONOUN_UNRESOLVED_SOURCE'] = True
        elif self.is_pronoun(rel.source) and not should_skip_source:
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
        # V14.3.4 NEW: Skip pronoun resolution for proper noun types
        target_type = getattr(rel, 'target_type', None)
        should_skip_target = target_type in self.skip_pronoun_types if target_type else False

        # V10: Check for possessive pronouns first
        if self.is_possessive_pronoun(rel.target) and not should_skip_target:
            resolved = self.resolve_possessive_pronouns(rel.target, page_num, evidence_text)
            if resolved:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['POSSESSIVE_PRONOUN_RESOLVED_TARGET'] = True
                rel.flags['original_target'] = rel.target
                rel.target = resolved
                self.stats['modified_count'] += 1
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['POSSESSIVE_PRONOUN_UNRESOLVED_TARGET'] = True
        elif self.is_pronoun(rel.target) and not should_skip_target:
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
        """Process batch of relationships to resolve pronouns (V11 OPTIMIZED)"""

        # V11: Clear cache at start of batch
        self.possessive_cache.clear()

        # Set author context from metadata (V10 ENHANCED)
        if context.document_metadata:
            self.author_context = context.document_metadata.get('author', None)
            self.author_heritage = context.document_metadata.get('heritage', None)
            self.author_nationality = context.document_metadata.get('nationality', None)
            
            # Also check for alternative metadata keys
            if not self.author_heritage:
                self.author_heritage = context.document_metadata.get('ethnicity', None)
            if not self.author_heritage:
                self.author_heritage = context.document_metadata.get('cultural_background', None)

        # Load page context if available
        if context.pages_with_text:
            self.load_page_context(context.pages_with_text)

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0
        resolved_count = 0
        generic_resolved_count = 0
        possessive_resolved_count = 0
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
                if rel.flags.get('POSSESSIVE_PRONOUN_RESOLVED_SOURCE') or rel.flags.get('POSSESSIVE_PRONOUN_RESOLVED_TARGET'):
                    possessive_resolved_count += 1
                if rel.flags.get('PRONOUN_UNRESOLVED_SOURCE') or rel.flags.get('PRONOUN_UNRESOLVED_TARGET'):
                    unresolved_count += 1
                if rel.flags.get('POSSESSIVE_PRONOUN_UNRESOLVED_SOURCE') or rel.flags.get('POSSESSIVE_PRONOUN_UNRESOLVED_TARGET'):
                    unresolved_count += 1

            processed.append(rel)

        # Update stats
        self.stats['anaphoric_resolved'] = resolved_count
        self.stats['generic_resolved'] = generic_resolved_count
        self.stats['possessive_resolved'] = possessive_resolved_count
        self.stats['unresolved'] = unresolved_count

        logger.info(
            f"   {self.name}: {resolved_count} anaphoric + {generic_resolved_count} generic + "
            f"{possessive_resolved_count} possessive resolved, {unresolved_count} flagged for review"
        )

        return processed