"""
Context Enricher Module

Replaces vague entities with specific ones from context.

Features:
- Expanded vague entity patterns (demonstratives, relative clauses, prepositional fragments)
- Context-aware replacement (V8)
- Document entity mapping (e.g., "this handbook" → "Soil Stewardship Handbook")
- Evidence-based enrichment using context keywords

Version History:
- v1.0.0 (V6): Basic vague entity detection and replacement
- v1.1.0 (V8): Context-aware replacement with keyword matching
"""

import re
import logging
from typing import Optional, List, Dict, Any

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class ContextEnricher(PostProcessingModule):
    """
    Enriches vague entities by replacing them with specific ones from context.

    Content Types: Universal (works for all content types)
    Priority: 50 (mid-pipeline, after list splitting)
    """

    name = "ContextEnricher"
    description = "Context-aware replacement of vague entities"
    content_types = ["all"]
    priority = 30  # V14.3.2: Moved from 50 to 30 to run BEFORE VagueEntityBlocker
    dependencies = []
    version = "1.2.0"  # V14.3.2: Priority reordering for resolve-then-block workflow

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Expanded vague terms from Reflector analysis
        self.vague_terms = {
            'the amount', 'the process', 'the practice', 'the method',
            'the system', 'the approach', 'the way', 'the idea',
            'this', 'that', 'these', 'those',
            'this handbook', 'this book', 'the handbook', 'the book',
            'this crossroads', 'the way through', 'that only exists',
            'which is', 'who are', 'that we',
            # V8: Additional vague patterns
            'this wonderful place', 'the answer',
            # V14.3.2 NEW: Unknown entities
            'unknown', 'community activities', 'personal life-hacks',
            'activities', 'life-hacks', 'practices'
        }

        # Document-specific entity mappings
        self.doc_entities = self.config.get('doc_entities', {
            'this handbook': 'Soil Stewardship Handbook',
            'this book': 'Soil Stewardship Handbook',
            'the handbook': 'Soil Stewardship Handbook',
            'the book': 'Soil Stewardship Handbook',
            'this crossroads': 'current historical moment',
        })

        # V8 NEW: Context-aware replacement rules
        self.context_replacements = self.config.get('context_replacements', {
            'this wonderful place': {
                'keywords': ['earth', 'planet', 'world', 'lives depend'],
                'replacement': 'Earth'
            },
            'the answer': {
                'keywords': ['soil', 'stewardship', 'questions'],
                'replacement': 'soil stewardship',
                'check_motto': True  # Special handling for mottos
            },
            'the way': {
                'keywords': ['forward', 'path', 'direction'],
                'replacement': 'the path forward'
            },
            'this': {
                'keywords': ['book', 'handbook', 'guide'],
                'replacement': 'Soil Stewardship Handbook'
            }
        })

    def is_vague(self, entity: str) -> bool:
        """Check if entity is vague"""
        entity_lower = entity.lower().strip()

        if entity_lower in self.vague_terms:
            return True

        for term in self.vague_terms:
            if entity_lower.startswith(term):
                return True

        # Check for demonstrative patterns
        if re.match(r'^(this|that|these|those)\s+\w+', entity_lower):
            return True

        return False

    def _find_replacement(
        self,
        vague_term: str,
        evidence: str,
        relationship: str
    ) -> Optional[str]:
        """V8 NEW: Find context-appropriate replacement for vague term"""
        vague_lower = vague_term.lower()
        evidence_lower = evidence.lower()

        # Check each replacement rule
        for pattern, rule in self.context_replacements.items():
            if pattern in vague_lower:
                # Check if context keywords present
                if any(keyword in evidence_lower for keyword in rule['keywords']):
                    # Special handling for mottos
                    if rule.get('check_motto') and 'motto' not in relationship.lower():
                        return rule['replacement']
                    elif not rule.get('check_motto'):
                        return rule['replacement']

        return None

    def enrich_entity(
        self,
        entity: str,
        evidence_text: str,
        relationship: str,
        other_entity: str
    ) -> Optional[str]:
        """
        V14.3.2 ENHANCED: Try to enrich a vague entity with context.

        New features:
        - Resolve "unknown" publisher to actual publisher name
        - Better handling of generic activities/practices
        """
        entity_lower = entity.lower().strip()

        # Check document-specific mappings first
        if entity_lower in self.doc_entities:
            return self.doc_entities[entity_lower]

        # V8: Try context-aware replacement
        context_replacement = self._find_replacement(entity, evidence_text, relationship)
        if context_replacement:
            return context_replacement

        # V14.3.2 NEW: Handle "unknown" publisher
        # Try to find publisher name from evidence or metadata
        if entity_lower == 'unknown' and 'publish' in relationship.lower():
            # Look for publisher names in evidence (case-insensitive)
            publisher_patterns = [
                r'published by ([A-Z][A-Za-z\s]+(?:Press|Publishing|Books|Publishers?))',
                r'([A-Z][A-Za-z\s]+(?:Press|Publishing|Books|Publishers?))',
            ]
            for pattern in publisher_patterns:
                match = re.search(pattern, evidence_text, re.IGNORECASE)
                if match:
                    publisher = match.group(1).strip()
                    # Validate it's not too long (likely not a publisher)
                    if len(publisher.split()) <= 4:
                        return publisher
            # Can't resolve - leave as vague (will be flagged)
            return None

        # V14.3.2 NEW: Handle generic activities
        if entity_lower in {'community activities', 'personal life-hacks', 'activities', 'life-hacks'}:
            # Try to find specific activities in evidence
            activity_keywords = ['composting', 'gardening', 'soil building', 'cover cropping',
                               'mulching', 'rainwater', 'harvesting', 'permaculture']
            found_activities = [kw for kw in activity_keywords if kw in evidence_text.lower()]
            if found_activities:
                if len(found_activities) == 1:
                    return found_activities[0]
                # Multiple activities - still too vague, let blocker handle it
            return None

        # Try to extract qualifier from "the amount of X"
        if entity_lower.startswith('the amount'):
            match = re.search(r'the amount of ([^,\.]+)', evidence_text, re.IGNORECASE)
            if match:
                qualifier = match.group(1).strip()
                qualifier = re.sub(r'\s+(by|in|at)\s+.*', '', qualifier)
                return f"{qualifier}"

        # Try to identify process from context
        if entity_lower in {'the process', 'this process'}:
            processes = ['composting', 'pyrolysis', 'photosynthesis',
                        'decomposition', 'fermentation', 'soil building']
            for proc in processes:
                if proc in evidence_text.lower():
                    return f"{proc} process"

        # Fallback for book references
        if 'handbook' in entity_lower or 'book' in entity_lower:
            return 'Soil Stewardship Handbook'

        return None

    def enrich_relationship(self, rel: Any) -> Any:
        """Enrich a single relationship"""
        # Enrich source
        if self.is_vague(rel.source):
            enriched_source = self.enrich_entity(
                rel.source, rel.evidence_text,
                rel.relationship, rel.target
            )
            if enriched_source:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['CONTEXT_ENRICHED_SOURCE'] = True
                rel.flags['original_source'] = rel.source
                rel.source = enriched_source
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['VAGUE_SOURCE'] = True

        # Enrich target
        if self.is_vague(rel.target):
            enriched_target = self.enrich_entity(
                rel.target, rel.evidence_text,
                rel.relationship, rel.source
            )
            if enriched_target:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['CONTEXT_ENRICHED_TARGET'] = True
                rel.flags['original_target'] = rel.target
                rel.target = enriched_target
            else:
                if rel.flags is None:
                    rel.flags = {}
                rel.flags['VAGUE_TARGET'] = True

        return rel

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to enrich vague entities"""

        # Update doc_entities from context if available
        if context.document_metadata:
            title = context.document_metadata.get('title')
            if title:
                self.doc_entities.update({
                    'this book': title,
                    'the book': title,
                    'this handbook': title,
                    'the handbook': title,
                })

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        enriched_count = 0
        vague_count = 0

        for rel in relationships:
            rel = self.enrich_relationship(rel)

            # Count enrichment types
            if rel.flags and (rel.flags.get('CONTEXT_ENRICHED_SOURCE') or \
                            rel.flags.get('CONTEXT_ENRICHED_TARGET')):
                enriched_count += 1
                self.stats['modified_count'] += 1

            if rel.flags and (rel.flags.get('VAGUE_SOURCE') or \
                            rel.flags.get('VAGUE_TARGET')):
                vague_count += 1

            processed.append(rel)

        # Update stats
        self.stats['enriched'] = enriched_count
        self.stats['vague_flagged'] = vague_count

        logger.info(
            f"   {self.name} (V8 enhanced): {enriched_count} enriched with context-aware replacement, "
            f"{vague_count} flagged as vague"
        )

        return processed
