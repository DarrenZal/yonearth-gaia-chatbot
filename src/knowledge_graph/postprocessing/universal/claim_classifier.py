"""
Claim Classification Module

Classifies relationships into types: factual, philosophical, opinion, recommendation.

Features:
- Predicate-based classification (factual relationships)
- Evidence text analysis (opinions, recommendations)
- Philosophical statement detection
- Multiple classification support (relationship can have multiple types)
- Filtering of incidental prescriptive claims

Version History:
- v1.0.0 (V11.2): Initial implementation
- v1.1.0: Added filtering for incidental recommendations
"""

import re
import logging
from typing import List, Set, Optional, Dict, Any

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class ClaimClassifier(PostProcessingModule):
    """
    Classifies relationships as factual, philosophical, opinion, or recommendation.
    Filters incidental prescriptive claims unless they represent core thesis.

    Content Types: All
    Priority: 105 (runs after all modifications, before deduplication)
    """

    name = "ClaimClassifier"
    description = "Classify relationships by type (factual, philosophical, opinion, recommendation)"
    content_types = ["book", "podcast", "article"]
    priority = 105
    dependencies = []
    version = "1.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Factual predicates - verifiable facts
        self.factual_predicates = self.config.get('factual_predicates', {
            'authored', 'wrote', 'published', 'founded', 'located', 'contains',
            'born', 'died', 'established', 'created', 'produced', 'released',
            'named', 'called', 'titled', 'dated', 'measured', 'counted',
            'discovered', 'invented', 'built', 'constructed', 'designed',
            'has_part', 'part_of', 'member_of', 'employed_by', 'works_for',
            'educated_at', 'graduated_from', 'awarded', 'received',
            'published_in', 'appeared_in', 'cited_in', 'referenced_in'
        })

        # Philosophical predicates - abstract relationships
        self.philosophical_predicates = self.config.get('philosophical_predicates', {
            'represents', 'symbolizes', 'embodies', 'reflects', 'signifies',
            'manifests', 'expresses', 'conveys', 'demonstrates', 'illustrates',
            'reveals', 'suggests', 'implies', 'indicates', 'means',
            'is_essence_of', 'is_nature_of', 'transcends', 'encompasses'
        })

        # Opinion markers - subjective statements
        self.opinion_markers = self.config.get('opinion_markers', [
            r'\bbelieves?\b', r'\bthinks?\b', r'\bfeels?\b', r'\bopines?\b',
            r'\bclaims?\b', r'\bargues?\b', r'\bcontends?\b', r'\bmaintains?\b',
            r'\basserts?\b', r'\bproposes?\b', r'\bsuggests?\b',
            r'\bin my opinion\b', r'\bin his view\b', r'\bin her view\b',
            r'\baccording to\b', r'\bseems? to\b', r'\bappears? to\b'
        ])

        # Recommendation markers - prescriptive advice
        self.recommendation_markers = self.config.get('recommendation_markers', [
            r'\bshould\b', r'\bmust\b', r'\bshall\b', r'\bought to\b',
            r'\bneed to\b', r'\bhave to\b', r'\brecommends?\b', r'\badvises?\b',
            r'\bsuggests?\b', r'\burges?\b', r'\bencourages?\b', r'\bcounsels?\b',
            r'\bit is important\b', r'\bit is essential\b', r'\bit is vital\b',
            r'\bit is critical\b', r'\bit is necessary\b',
            r'\bwe should\b', r'\byou should\b', r'\bone should\b',
            r'\bwe must\b', r'\byou must\b', r'\bone must\b',
            r'\bconsider\b.*\bing\b', r'\btry to\b', r'\bmake sure\b'
        ])

        # Core thesis indicators - signals that recommendation is central
        self.core_thesis_indicators = self.config.get('core_thesis_indicators', [
            r'\bcentral argument\b', r'\bmain thesis\b', r'\bcore idea\b',
            r'\bkey principle\b', r'\bfundamental principle\b',
            r'\bprimary recommendation\b', r'\bmain recommendation\b',
            r'\bcentral claim\b', r'\bprimary claim\b',
            r'\bauthor\'s main\b', r'\bauthor\'s central\b', r'\bauthor\'s key\b',
            r'\bbook\'s main\b', r'\bbook\'s central\b', r'\bbook\'s key\b',
            r'\boverall message\b', r'\bcentral message\b',
            r'\bprimary focus\b', r'\bmain focus\b'
        ])

        # Incidental recommendation indicators - signals to filter
        self.incidental_indicators = self.config.get('incidental_indicators', [
            r'\bact now\b', r'\btake action now\b', r'\bdon\'t wait\b',
            r'\bimmediately\b', r'\bright now\b', r'\btoday\b',
            r'\bthis moment\b', r'\bthis instant\b',
            r'\bcall to action\b', r'\bsign up\b', r'\bsubscribe\b',
            r'\bbuy now\b', r'\bpurchase\b', r'\border now\b',
            r'\bvisit our\b', r'\bcheck out\b', r'\blearn more at\b'
        ])

        # Filter recommendations by default unless marked as core thesis
        self.filter_recommendations = self.config.get('filter_recommendations', True)

    def is_factual(self, rel: Any) -> bool:
        """Check if relationship is factual (verifiable)"""
        return rel.relationship.lower().strip() in self.factual_predicates

    def is_philosophical(self, rel: Any) -> bool:
        """Check if relationship is philosophical/abstract"""
        return rel.relationship.lower().strip() in self.philosophical_predicates

    def contains_opinion_markers(self, text: str) -> bool:
        """Check if evidence text contains opinion markers"""
        text_lower = text.lower()
        for pattern in self.opinion_markers:
            if re.search(pattern, text_lower):
                return True
        return False

    def contains_recommendation_markers(self, text: str) -> bool:
        """Check if evidence text contains recommendation markers"""
        text_lower = text.lower()
        for pattern in self.recommendation_markers:
            if re.search(pattern, text_lower):
                return True
        return False

    def is_core_thesis(self, rel: Any) -> bool:
        """Check if recommendation represents core thesis of the work"""
        evidence = rel.evidence_text or ""
        text_lower = evidence.lower()
        
        # Check for core thesis indicators
        for pattern in self.core_thesis_indicators:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def is_incidental_recommendation(self, rel: Any) -> bool:
        """Check if recommendation is incidental (call to action, marketing, etc.)"""
        evidence = rel.evidence_text or ""
        text_lower = evidence.lower()
        
        # Check for incidental indicators
        for pattern in self.incidental_indicators:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def should_filter_recommendation(self, rel: Any) -> bool:
        """Determine if a recommendation should be filtered out"""
        if not self.filter_recommendations:
            return False
        
        # Keep if it's core thesis
        if self.is_core_thesis(rel):
            return False
        
        # Filter if it's incidental
        if self.is_incidental_recommendation(rel):
            return True
        
        # Filter by default (conservative approach - only keep core recommendations)
        return True

    def classify_relationship(self, rel: Any) -> Set[str]:
        """
        Classify relationship and return all applicable types.

        Returns:
            Set of classification flags (may be empty or contain multiple)
        """
        classifications = set()

        # Check factual
        if self.is_factual(rel):
            classifications.add('FACTUAL')

        # Check philosophical
        if self.is_philosophical(rel):
            classifications.add('PHILOSOPHICAL_CLAIM')

        # Check opinion (from evidence text)
        evidence = rel.evidence_text or ""
        if self.contains_opinion_markers(evidence):
            classifications.add('OPINION')

        # Check recommendation (from evidence text)
        if self.contains_recommendation_markers(evidence):
            classifications.add('RECOMMENDATION')

        # If nothing matched, it's factual by default (knowledge extraction assumes truth)
        # unless it's clearly abstract/metaphorical
        if not classifications:
            # Check for abstract/vague concepts
            abstract_indicators = ['essence', 'nature', 'spirit', 'soul', 'energy', 'consciousness']
            if any(ind in rel.source.lower() or ind in rel.target.lower() for ind in abstract_indicators):
                classifications.add('PHILOSOPHICAL_CLAIM')
            else:
                classifications.add('FACTUAL')

        return classifications

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """
        Classify all relationships in batch and filter incidental recommendations.

        Args:
            relationships: List of relationship objects
            context: Processing context

        Returns:
            Filtered relationships with classification flags added
        """

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0
        self.stats['filtered_count'] = 0

        classification_counts = {
            'FACTUAL': 0,
            'PHILOSOPHICAL_CLAIM': 0,
            'OPINION': 0,
            'RECOMMENDATION': 0
        }

        filtered_relationships = []

        for rel in relationships:
            # Classify relationship
            classifications = self.classify_relationship(rel)

            # Check if this is a recommendation that should be filtered
            if 'RECOMMENDATION' in classifications:
                if self.should_filter_recommendation(rel):
                    self.stats['filtered_count'] += 1
                    logger.debug(f"Filtered incidental recommendation: {rel.source} -> {rel.target}")
                    continue

            # Add classification flags
            if classifications:
                if rel.flags is None:
                    rel.flags = {}

                for classification in classifications:
                    rel.flags[classification] = True
                    classification_counts[classification] += 1

                self.stats['modified_count'] += 1

            filtered_relationships.append(rel)

        # Update stats
        self.stats['factual_count'] = classification_counts['FACTUAL']
        self.stats['philosophical_count'] = classification_counts['PHILOSOPHICAL_CLAIM']
        self.stats['opinion_count'] = classification_counts['OPINION']
        self.stats['recommendation_count'] = classification_counts['RECOMMENDATION']

        logger.info(
            f"   {self.name}: Classified {self.stats['modified_count']} relationships - "
            f"Factual: {classification_counts['FACTUAL']}, "
            f"Philosophical: {classification_counts['PHILOSOPHICAL_CLAIM']}, "
            f"Opinion: {classification_counts['OPINION']}, "
            f"Recommendation: {classification_counts['RECOMMENDATION']}, "
            f"Filtered: {self.stats['filtered_count']}"
        )

        return filtered_relationships