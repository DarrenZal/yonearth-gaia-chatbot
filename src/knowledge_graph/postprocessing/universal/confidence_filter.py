"""
ConfidenceFilter: Filter relationships based on p_true thresholds

V14.0: Implements configurable thresholds with flag-specific overrides
- Base threshold: 0.5 (conservative, V13.1 calibrated)
- Flag-specific thresholds: Higher for PHILOSOPHICAL_CLAIM, METAPHOR, OPINION
- Loads from config/filtering_thresholds.yaml
"""

from pathlib import Path
from typing import List, Dict, Any
import yaml
import logging

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)


class ConfidenceFilter(PostProcessingModule):
    """
    Filter relationships based on p_true confidence thresholds.

    V14.0 Design:
    - Conservative base threshold (0.5) for domain knowledge
    - Aggressive thresholds for low-quality flags (PHILOSOPHICAL_CLAIM, METAPHOR, OPINION)
    - Loads configuration from YAML file

    Expected Impact:
    - Filters 8-10 low-quality philosophical/metaphorical relationships (0.9-1.1%)
    - Maintains valid domain knowledge through conservative base
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Load thresholds from config file
        self.thresholds_config = self._load_thresholds_config()

        self.base_threshold = self.thresholds_config.get('base_threshold', 0.5)
        self.flag_specific_thresholds = self.thresholds_config.get('flag_specific_thresholds', {})

        logger.info(f"ConfidenceFilter initialized:")
        logger.info(f"  - Base threshold: {self.base_threshold}")
        logger.info(f"  - Flag-specific thresholds: {len(self.flag_specific_thresholds)} flags configured")

    def _load_thresholds_config(self) -> Dict[str, Any]:
        """Load filtering thresholds from YAML config file"""
        config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "filtering_thresholds.yaml"

        if not config_path.exists():
            logger.warning(f"Thresholds config not found: {config_path}")
            logger.warning("Using default thresholds: base=0.5")
            return {
                'base_threshold': 0.5,
                'flag_specific_thresholds': {}
            }

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ Loaded thresholds from: {config_path.name}")
                return config
        except Exception as e:
            logger.error(f"Failed to load thresholds config: {e}")
            logger.warning("Using default thresholds: base=0.5")
            return {
                'base_threshold': 0.5,
                'flag_specific_thresholds': {}
            }

    def _get_threshold_for_relationship(self, relationship: Any) -> float:
        """
        Determine the threshold for a relationship based on its flags.

        Logic:
        1. Check classification_flags for flag-specific thresholds
        2. Check flags dict for module flags with thresholds
        3. V14: Apply 0.7 threshold for unresolved pronouns
        4. Return highest applicable threshold (most conservative)
        5. Default to base_threshold if no special flags present
        """
        applicable_thresholds = [self.base_threshold]

        # Check classification flags (PHILOSOPHICAL_CLAIM, METAPHOR, etc.)
        if relationship.classification_flags:
            for flag in relationship.classification_flags:
                if flag in self.flag_specific_thresholds:
                    applicable_thresholds.append(self.flag_specific_thresholds[flag])

        # Check module flags (signals_conflict_true, etc.)
        if relationship.flags:
            for flag_key in relationship.flags.keys():
                if flag_key in self.flag_specific_thresholds:
                    applicable_thresholds.append(self.flag_specific_thresholds[flag_key])

            # V14: Special handling for unresolved pronouns
            # If pronouns couldn't be resolved, require higher confidence (0.7)
            unresolved_pronoun_flags = [
                'PRONOUN_UNRESOLVED_SOURCE',
                'PRONOUN_UNRESOLVED_TARGET',
                'POSSESSIVE_PRONOUN_UNRESOLVED_SOURCE',
                'POSSESSIVE_PRONOUN_UNRESOLVED_TARGET'
            ]
            if any(flag in relationship.flags for flag in unresolved_pronoun_flags):
                applicable_thresholds.append(0.7)

        # Return highest threshold (most conservative)
        return max(applicable_thresholds)

    @property
    def name(self) -> str:
        return "ConfidenceFilter"

    @property
    def priority(self) -> int:
        return 120  # Run after Deduplicator (110)

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """
        Filter relationships based on p_true thresholds.

        Returns:
            Filtered list of relationships that meet threshold requirements
        """
        logger.info(f"🔍 {self.name}: Filtering {len(relationships)} relationships by confidence...")

        filtered = []
        filter_stats = {
            'total_input': len(relationships),
            'passed': 0,
            'filtered': 0,
            'filter_reasons': {}
        }

        for rel in relationships:
            # Determine applicable threshold
            threshold = self._get_threshold_for_relationship(rel)

            # Check if relationship meets threshold
            if rel.p_true >= threshold:
                filtered.append(rel)
                filter_stats['passed'] += 1
            else:
                filter_stats['filtered'] += 1

                # Track reason for filtering
                reason = f"p_true={rel.p_true:.2f} < threshold={threshold:.2f}"
                if rel.classification_flags:
                    reason += f" (flags: {', '.join(rel.classification_flags)})"

                filter_stats['filter_reasons'][reason] = filter_stats['filter_reasons'].get(reason, 0) + 1

                logger.debug(f"  ❌ Filtered: ({rel.source}, {rel.relationship}, {rel.target}) - {reason}")

        # Log summary
        logger.info(f"✅ {self.name}: {filter_stats['passed']}/{filter_stats['total_input']} relationships passed")
        logger.info(f"   Filtered: {filter_stats['filtered']} relationships")

        if filter_stats['filter_reasons']:
            logger.info(f"   Top filter reasons:")
            for reason, count in sorted(filter_stats['filter_reasons'].items(), key=lambda x: -x[1])[:5]:
                logger.info(f"     - {reason}: {count} relationships")

        # Store stats in context
        self._store_stats(context, filter_stats)

        return filtered
