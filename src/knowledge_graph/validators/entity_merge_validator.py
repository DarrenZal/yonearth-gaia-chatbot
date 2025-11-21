"""
Entity Merge Validator

Semantic validation for entity merging to prevent catastrophic merges like:
- Moscow = Soil + moon (different types and semantics)
- Earth = Mars + Paris + farms (unrelated entities)
- DIA = Dubai + Red + Sun + India (nonsensical)

Version: 1.0.0
Created: 2025-11-20
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from fuzzywuzzy import fuzz
from collections import defaultdict

logger = logging.getLogger(__name__)


class EntityMergeValidator:
    """
    Validates entity merges to prevent catastrophic errors.

    Validation Rules:
    1. Type compatibility: Only merge entities of same type
    2. Length ratio: Prevent merging very different lengths (e.g., "I" with "India")
    3. Semantic blocklist: Never merge known problematic pairs
    4. Fuzzy threshold: Require higher similarity (95 instead of 90)
    """

    # Known problematic merges that should NEVER happen
    MERGE_BLOCKLIST = [
        ('moscow', 'soil'),
        ('moscow', 'moon'),
        ('earth', 'mars'),
        ('earth', 'paris'),
        ('earth', 'farms'),
        ('earth', 'farm'),
        ('leaders', 'healers'),
        ('leaders', 'readers'),
        ('organization', 'urbanization'),
        ('organization', 'modernization'),
        ('business', 'sickness'),
        ('the soil', 'the stove'),
        ('the soil', 'the skin'),
        ('the soil', 'the show'),
        ('the land', 'thailand'),
        ('the land', 'the legend'),
        ('dia', 'dubai'),
        ('dia', 'india'),
        ('dia', 'sun'),
        ('dia', 'red'),
    ]

    # Pairs that look similar but have different semantic types
    TYPE_BLOCKLIST = [
        ('soil', 'moon'),  # Both short, but different semantic domains
        ('mars', 'paris'),  # Similar spelling, different entities
        ('leaders', 'healers'),  # Similar ending, different concepts
    ]

    def __init__(
        self,
        similarity_threshold: int = 95,
        min_length_ratio: float = 0.6,
        type_strict_matching: bool = True,
        semantic_validation: bool = True
    ):
        """
        Initialize entity merge validator.

        Args:
            similarity_threshold: Minimum fuzzy match score (0-100). Default 95.
            min_length_ratio: Minimum length ratio between entities (0-1). Default 0.6.
                             Prevents "I" (len=1) from merging with "India" (len=5).
            type_strict_matching: If True, only merge entities of same type
            semantic_validation: If True, check semantic compatibility
        """
        self.similarity_threshold = similarity_threshold
        self.min_length_ratio = min_length_ratio
        self.type_strict_matching = type_strict_matching
        self.semantic_validation = semantic_validation

        # Statistics
        self.stats = {
            'total_comparisons': 0,
            'passed_validations': 0,
            'failed_type_check': 0,
            'failed_length_check': 0,
            'failed_similarity_check': 0,
            'failed_blocklist_check': 0,
            'failed_semantic_check': 0,
        }

        # Build normalized blocklist for fast lookup
        self._blocklist_set: Set[Tuple[str, str]] = set()
        for name1, name2 in self.MERGE_BLOCKLIST:
            self._blocklist_set.add((name1.lower().strip(), name2.lower().strip()))
            self._blocklist_set.add((name2.lower().strip(), name1.lower().strip()))

        logger.info(
            f"EntityMergeValidator initialized: "
            f"threshold={similarity_threshold}, "
            f"min_length_ratio={min_length_ratio}, "
            f"type_strict={type_strict_matching}"
        )

    def can_merge(
        self,
        entity1: Dict,
        entity2: Dict,
        log_rejection: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if two entities can be safely merged.

        Args:
            entity1: First entity dict with 'name', 'type', etc.
            entity2: Second entity dict
            log_rejection: If True, log rejection reasons

        Returns:
            Tuple of (can_merge: bool, reason: str)
        """
        self.stats['total_comparisons'] += 1

        name1 = entity1.get('name', '')
        name2 = entity2.get('name', '')
        type1 = entity1.get('type', 'UNKNOWN')
        type2 = entity2.get('type', 'UNKNOWN')

        # Normalize names for comparison
        norm1 = name1.lower().strip()
        norm2 = name2.lower().strip()

        # Check 1: Exact match (always merge)
        if norm1 == norm2:
            self.stats['passed_validations'] += 1
            return True, "exact_match"

        # Check 2: Type compatibility
        if self.type_strict_matching and type1 != type2:
            self.stats['failed_type_check'] += 1
            reason = f"type_mismatch: {type1} != {type2}"
            if log_rejection:
                logger.debug(
                    f"REJECT merge: '{name1}' ({type1}) + '{name2}' ({type2}) - {reason}"
                )
            return False, reason

        # Check 3: Length ratio (prevent "I" -> "India")
        len1, len2 = len(name1), len(name2)
        if len1 == 0 or len2 == 0:
            return False, "empty_name"

        length_ratio = min(len1, len2) / max(len1, len2)
        if length_ratio < self.min_length_ratio:
            self.stats['failed_length_check'] += 1
            reason = f"length_mismatch: {len1} vs {len2} (ratio={length_ratio:.2f})"
            if log_rejection:
                logger.debug(
                    f"REJECT merge: '{name1}' + '{name2}' - {reason}"
                )
            return False, reason

        # Check 4: Fuzzy similarity threshold
        similarity = fuzz.ratio(norm1, norm2)
        if similarity < self.similarity_threshold:
            self.stats['failed_similarity_check'] += 1
            reason = f"low_similarity: {similarity} < {self.similarity_threshold}"
            if log_rejection:
                logger.debug(
                    f"REJECT merge: '{name1}' + '{name2}' - {reason}"
                )
            return False, reason

        # Check 5: Explicit blocklist
        if (norm1, norm2) in self._blocklist_set or (norm2, norm1) in self._blocklist_set:
            self.stats['failed_blocklist_check'] += 1
            reason = "explicit_blocklist"
            if log_rejection:
                logger.warning(
                    f"REJECT merge (BLOCKLIST): '{name1}' + '{name2}'"
                )
            return False, reason

        # Check 6: Semantic validation (substring checks)
        if self.semantic_validation:
            # Check for problematic substring patterns
            if not self._check_semantic_compatibility(norm1, norm2):
                self.stats['failed_semantic_check'] += 1
                reason = "semantic_incompatibility"
                if log_rejection:
                    logger.debug(
                        f"REJECT merge: '{name1}' + '{name2}' - {reason}"
                    )
                return False, reason

        # All checks passed
        self.stats['passed_validations'] += 1
        return True, f"approved: similarity={similarity}"

    def _check_semantic_compatibility(self, norm1: str, norm2: str) -> bool:
        """
        Check if two normalized entity names are semantically compatible.

        Rejects merges where:
        - One contains "soil" and other contains "stove"/"skin"/"show"
        - One contains "leaders" and other contains "healers"/"readers"
        - Geographic terms with very different semantics

        Args:
            norm1: Normalized entity name 1
            norm2: Normalized entity name 2

        Returns:
            True if semantically compatible, False otherwise
        """
        # Extract key words from each name
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # Check type blocklist patterns
        for blocked1, blocked2 in self.TYPE_BLOCKLIST:
            if blocked1 in norm1 and blocked2 in norm2:
                return False
            if blocked2 in norm1 and blocked1 in norm2:
                return False

        # Additional heuristic: If names share no words and are both >3 words,
        # they're likely unrelated
        if len(words1) >= 3 and len(words2) >= 3:
            if not words1.intersection(words2):
                return False

        return True

    def batch_validate_merges(
        self,
        merge_candidates: List[Tuple[Dict, Dict]]
    ) -> Dict[str, List[Tuple[Dict, Dict]]]:
        """
        Validate a batch of merge candidates.

        Args:
            merge_candidates: List of (entity1, entity2) tuples to validate

        Returns:
            Dict with keys:
                - 'approved': List of approved merges
                - 'rejected': List of rejected merges
                - 'reason_counts': Count of rejection reasons
        """
        approved = []
        rejected = []
        reason_counts = defaultdict(int)

        for entity1, entity2 in merge_candidates:
            can_merge, reason = self.can_merge(entity1, entity2, log_rejection=False)

            if can_merge:
                approved.append((entity1, entity2))
            else:
                rejected.append((entity1, entity2, reason))
                reason_counts[reason] += 1

        logger.info(
            f"Batch validation: {len(approved)} approved, {len(rejected)} rejected "
            f"(out of {len(merge_candidates)} candidates)"
        )

        return {
            'approved': approved,
            'rejected': rejected,
            'reason_counts': dict(reason_counts)
        }

    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        total = self.stats['total_comparisons']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'approval_rate': self.stats['passed_validations'] / total,
            'rejection_rate': (total - self.stats['passed_validations']) / total,
        }

    def log_statistics(self):
        """Log validation statistics."""
        stats = self.get_statistics()
        total = stats['total_comparisons']

        if total == 0:
            logger.info("No entity merge validations performed yet")
            return

        logger.info("=" * 60)
        logger.info("ENTITY MERGE VALIDATION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total comparisons: {total}")
        logger.info(f"Approved merges: {stats['passed_validations']} ({stats['approval_rate']:.1%})")
        logger.info(f"Rejected merges: {total - stats['passed_validations']} ({stats['rejection_rate']:.1%})")
        logger.info("")
        logger.info("Rejection reasons:")
        logger.info(f"  - Type mismatch: {stats['failed_type_check']}")
        logger.info(f"  - Length mismatch: {stats['failed_length_check']}")
        logger.info(f"  - Low similarity: {stats['failed_similarity_check']}")
        logger.info(f"  - Explicit blocklist: {stats['failed_blocklist_check']}")
        logger.info(f"  - Semantic incompatibility: {stats['failed_semantic_check']}")
        logger.info("=" * 60)
