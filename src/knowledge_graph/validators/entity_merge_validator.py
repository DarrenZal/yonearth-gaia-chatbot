"""
Entity Merge Validator

Semantic validation for entity merging to prevent catastrophic merges like:
- Moscow = Soil + moon (different types and semantics)
- Earth = Mars + Paris + farms (unrelated entities)
- DIA = Dubai + Red + Sun + India (nonsensical)

Version: 2.1.0
Created: 2025-11-20
Updated: 2025-11-21 - Added normalization, token overlap, type-specific thresholds, neighbor veto
Updated: 2025-12-04 - Added SEMANTIC_BLOCKLIST for known bad merges (Phase 6)
"""

import logging
import re
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

    # SEMANTIC_BLOCKLIST: Known bad merges identified from merge history analysis
    # These pairs look similar due to string metrics but have different meanings
    # Checked FIRST in can_merge() before any other validation
    # Added in Phase 6 (2025-12-04) based on entity_merges.json analysis
    SEMANTIC_BLOCKLIST = [
        # Core problematic pairs from original specification
        ('mood', 'food'),
        ('floods', 'food'),
        ('future revelations', 'future generations'),
        ('older generations', 'future generations'),
        ('country', 'community'),
        ('commune', 'community'),
        ('joanna macy', 'chris johnstone'),
        ('y on earth', 'earth water press'),
        # Discovered from merge history analysis
        ('water', 'nature'),
        ('ocean', 'japan'),
        ('delta', 'dia'),
        ('corn', 'imax'),
        ('trees', 'imax'),
        ('char', 'imax'),
        ('yoga', 'tour'),
        ('work', 'tour'),
        ('tour', 'talk'),
        ('study', 'ted'),
        ('society', 'soviets'),
        ('brands', 'brics'),
        ('nations', 'nazi ss'),
        ('cooking', 'coaching'),
        ('parenting', 'gardening'),
        ('schools', 'scholars'),
        ('mothers', 'vendors'),
        ('eating', 'satsang'),
        ('cacao', 'jetta'),
        ('allies', 'allianz'),
        ('board', 'norad'),
        ('dirt', 'dvds'),
        ('zoom', 'cocoa'),
        ('zeal', 'terna'),
        ('true', 'terna'),
        ('interns', 'terna'),
        ('salt', 'wahl'),
        ('kale', 'wahl'),
        ('nettles', 'metals'),
        ('mountain', 'montana'),
        ('mountains', 'montana'),
        ('poisons', 'poisonous'),
        ('go', 'hft'),
        ('women', 'dove'),
        ('berry', 'wendell berry'),  # Generic term vs proper name
        ('price', 'steven price'),  # Generic term vs proper name
        ('grace', 'stephen grace'),  # Generic term vs proper name
        ('northern california', 'southern california'),
        ('southern europe', 'northern europe'),
        ('washington state', 'washington square'),
        ('patagonia', 'adagonia'),
        ('demeter', 'domtar'),
        ('saraya', 'santana'),
        ('valerian', 'material'),
        ('skiers', 'waners'),
        ('plastics', 'plastic'),  # Singular/plural should not cross-domain merge
        # Additional confirmed bad merges from analyze_bad_merges.py (2025-12-04)
        ('the past', 'the earth'),
        ('the west', 'the earth'),
        ('the south', 'earth'),
        ('personal health', 'soil health'),
        ('fridays', 'friends'),
        ('a tree', 'a future'),
        ('the water', 'the work'),
        ('other years', 'mother earth'),
        ('hearth experience', 'human experience'),
        ('y-north.org', 'yonearth.org'),  # Different websites
        ('soil', 'moscow'),
        ('sun', 'dia'),
        ('legacy', 'we act'),
        ('people', 'mohawk people'),  # Generic vs specific
        ('compost', 'composting'),  # Noun vs verb form in different contexts
        ('educators', 'education'),
    ]

    # Common abbreviations to expand before comparison
    ABBREVIATION_MAP = {
        'dr': 'doctor',
        'dr.': 'doctor',
        'mr': 'mister',
        'mr.': 'mister',
        'mrs': 'misses',
        'mrs.': 'misses',
        'ms': 'miss',
        'ms.': 'miss',
        'st': 'saint',
        'st.': 'saint',
        'co': 'company',
        'co.': 'company',
        'corp': 'corporation',
        'corp.': 'corporation',
        'inc': 'incorporated',
        'inc.': 'incorporated',
        'ltd': 'limited',
        'ltd.': 'limited',
    }

    # Stop words to remove (titles, articles)
    STOP_TITLES = {'the', 'a', 'an', 'and', 'of', 'for'}

    # Entity types that can use flexible Tier 2 matching (85-94%)
    FLEXIBLE_TYPES = {'PERSON', 'ORGANIZATION', 'CONCEPT'}

    # Entity types that require strict Tier 1 matching (≥95%)
    STRICT_TYPES = {'PLACE', 'EVENT', 'LOCATION', 'REGION'}

    # Country / region synonym map (normalized forms -> canonical ISO3-ish IDs, lowercase)
    COUNTRY_SYNONYMS = {
        # United States
        'usa': 'usa',
        'u s a': 'usa',
        'u.s.a': 'usa',
        'u.s.a.': 'usa',
        'us': 'usa',
        'u s': 'usa',
        'u.s': 'usa',
        'u.s.': 'usa',
        'united states of america': 'usa',
        'united states': 'usa',
        'america': 'usa',
        # United Kingdom
        'uk': 'gbr',
        'u k': 'gbr',
        'u.k': 'gbr',
        'u.k.': 'gbr',
        'united kingdom': 'gbr',
        'britain': 'gbr',
        'great britain': 'gbr',
        # United Arab Emirates
        'uae': 'are',
        'u.a.e': 'are',
        'u.a.e.': 'are',
        'united arab emirates': 'are',
        # Democratic Republic of the Congo
        'drc': 'cod',
        'd.r.c': 'cod',
        'd.r.c.': 'cod',
        'democratic republic of congo': 'cod',
        'dr congo': 'cod',
        'congo drc': 'cod',
        'democratic republic of the congo': 'cod',
        'democratic republic congo': 'cod',
        # Republic of the Congo (leave bare "congo" unmapped to avoid ambiguity)
        'republic of congo': 'cog',
        'republic of the congo': 'cog',
        # Korea
        'south korea': 'kor',
        'republic of korea': 'kor',
        'north korea': 'prk',
        # Others commonly mentioned
        'russia': 'rus',
        'russian federation': 'rus',
        'iran': 'irn',
        'iraq': 'irq',
        'united arab republic': 'egy',  # historic ambiguity; map to Egypt cautiously
    }

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
            'failed_semantic_blocklist': 0,  # Phase 6: Semantic blocklist rejections
        }

        # Build normalized blocklist for fast lookup (MERGE_BLOCKLIST)
        self._blocklist_set: Set[Tuple[str, str]] = set()
        for name1, name2 in self.MERGE_BLOCKLIST:
            self._blocklist_set.add((name1.lower().strip(), name2.lower().strip()))
            self._blocklist_set.add((name2.lower().strip(), name1.lower().strip()))

        # Build semantic blocklist set for fast lookup (Phase 6)
        # This is checked FIRST in can_merge() - bidirectional and case-insensitive
        self._semantic_blocklist_set: Set[Tuple[str, str]] = set()
        for name1, name2 in self.SEMANTIC_BLOCKLIST:
            self._semantic_blocklist_set.add((name1.lower().strip(), name2.lower().strip()))
            self._semantic_blocklist_set.add((name2.lower().strip(), name1.lower().strip()))

        # Cache of canonical country ids
        self._country_ids: Set[str] = set(self.COUNTRY_SYNONYMS.values())

        logger.info(
            f"EntityMergeValidator initialized: "
            f"threshold={similarity_threshold}, "
            f"min_length_ratio={min_length_ratio}, "
            f"type_strict={type_strict_matching}"
        )

    def _country_id(self, normalized_phrase: str) -> Optional[str]:
        """
        Return canonical country id if normalized phrase is a known alias or id.
        """
        if not normalized_phrase:
            return None
        if normalized_phrase in self.COUNTRY_SYNONYMS:
            return self.COUNTRY_SYNONYMS[normalized_phrase]
        if normalized_phrase in self._country_ids:
            return normalized_phrase
        return None

    def _normalize_name(self, name: str) -> str:
        """
        Normalize entity name for fair comparison.

        Handles:
        - Case normalization (lowercase)
        - Acronym detection (N.A.S.A. → nasa)
        - Possessive removal ('s → '')
        - Punctuation collapse (hyphens, periods → spaces)
        - Abbreviation expansion (dr. → doctor)
        - Stop word removal (the, a, an, etc.)
        - Whitespace normalization

        Examples:
            "Dr. Bronner's" → "doctor bronner"
            "Y-on-Earth" → "y on earth"
            "N.A.S.A." → "nasa"
            "The Soil" → "soil"
        """
        if not name:
            return ""

        raw_name = name
        # Lowercase
        name = name.lower()

        # Raw phrase normalization (before stop-word removal) for country detection
        raw_phrase = re.sub(r'[^a-z0-9 ]', ' ', name)
        raw_phrase = re.sub(r'\s+', ' ', raw_phrase).strip()
        if raw_phrase in self.COUNTRY_SYNONYMS:
            if raw_phrase != "us" or any(c.isupper() for c in raw_name) or "." in raw_name:
                return self.COUNTRY_SYNONYMS[raw_phrase]

        # Detect and handle acronyms (e.g., "n.a.s.a." → "nasa")
        # Pattern: single letter followed by period, repeated
        if re.match(r'^([a-z]\.)+$', name) or re.match(r'^([a-z]\. )+[a-z]\.?$', name):
            # This is an acronym - just remove all periods
            return name.replace(".", "").replace(" ", "")

        # Remove possessive 's
        name = re.sub(r"'s\b", "", name)

        # Replace hyphens and periods with spaces
        name = name.replace("-", " ").replace(".", " ")

        # Split into tokens
        tokens = name.split()

        # Expand abbreviations and remove stop words
        normalized_tokens = []
        for token in tokens:
            # Clean trailing punctuation
            token_clean = token.strip('.,!?;:\'"')
            if not token_clean:
                continue

            # Expand abbreviations
            token_expanded = self.ABBREVIATION_MAP.get(token_clean, token_clean)

            # Skip stop words
            if token_expanded not in self.STOP_TITLES:
                normalized_tokens.append(token_expanded)

        # Join and normalize whitespace
        normalized_phrase = " ".join(normalized_tokens)

        # Country/region synonym collapsing (operates on full normalized phrase)
        if normalized_phrase in self.COUNTRY_SYNONYMS:
            # Avoid mapping conversational "us" unless the raw name looked like an acronym
            if normalized_phrase != "us" or any(c.isupper() for c in raw_name) or "." in raw_name:
                normalized_phrase = self.COUNTRY_SYNONYMS[normalized_phrase]
        elif normalized_phrase in self._country_ids:
            # Already a canonical country id
            normalized_phrase = normalized_phrase

        return normalized_phrase

    def _token_overlap(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        """
        Calculate token overlap ratio with fuzzy matching.

        For each token in the smaller set, finds if there's a similar token
        in the larger set (>= 85% similarity counts as a match).

        Returns: matched_tokens / max(|tokens1|, |tokens2|)

        Examples:
            {"bronner"} ∩ {"bronners"} = 1/1 = 1.0 (fuzzy match)
            {"bronner"} ∩ {"doctor", "bronner"} = 1/2 = 0.5
            {"soil"} ∩ {"soil", "conservation"} = 1/2 = 0.5
            {"doctor", "bronner"} ∩ {"doctor", "bronner"} = 2/2 = 1.0
        """
        if not tokens1 or not tokens2:
            return 0.0

        # Exact intersection first
        exact_matches = tokens1.intersection(tokens2)
        matched_count = len(exact_matches)

        # For unmatched tokens, try fuzzy matching
        unmatched1 = tokens1 - exact_matches
        unmatched2 = tokens2 - exact_matches

        for token1 in unmatched1:
            for token2 in unmatched2:
                # Check if tokens are similar (handles "bronner" vs "bronners")
                if fuzz.ratio(token1, token2) >= 85:
                    matched_count += 1
                    break  # Count each token only once

        return matched_count / max(len(tokens1), len(tokens2))

    def _has_punctuation_only_difference(self, name1: str, name2: str, norm1: str, norm2: str) -> bool:
        """
        Check if two names differ only in punctuation (apostrophes, hyphens, periods).

        Examples:
            "Bronner's" vs "Bronners" → True
            "Y-on-Earth" vs "Y on Earth" → True
            "Dr." vs "Dr" → True
            "Moscow" vs "Moon" → False
        """
        # Remove all punctuation and whitespace from originals
        clean1 = re.sub(r'[^a-z0-9]', '', name1.lower())
        clean2 = re.sub(r'[^a-z0-9]', '', name2.lower())

        # If cleaned versions are identical, difference was only punctuation
        return clean1 == clean2

    def _only_title_difference(self, tokens1: Set[str], tokens2: Set[str]) -> bool:
        """
        Determine if the only difference between two token sets is honorific/title tokens.

        Handles simple plural stripping so that "bronner" ≈ "bronner(s)".
        """
        title_tokens = {
            'dr', 'doctor', 'mr', 'mister', 'mrs', 'misses', 'ms', 'miss',
            'prof', 'professor', 'sir', 'jr', 'sr'
        }

        def _base_tokens(tokens: Set[str]) -> Set[str]:
            base = set()
            for t in tokens:
                if t.endswith('s') and len(t) > 3:
                    base.add(t[:-1])
                else:
                    base.add(t)
            return base

        base1 = _base_tokens(tokens1)
        base2 = _base_tokens(tokens2)

        non_title1 = {t for t in base1 if t not in title_tokens}
        non_title2 = {t for t in base2 if t not in title_tokens}

        if non_title1 != non_title2:
            return False

        diff1 = base1 - base2
        diff2 = base2 - base1
        return (diff1.issubset(title_tokens) or diff2.issubset(title_tokens))

    def can_merge(
        self,
        entity1: Dict,
        entity2: Dict,
        log_rejection: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if two entities can be safely merged using two-tier threshold strategy.

        Tier 1 (≥95%): High confidence merges with full validation
        Tier 2 (85-94%): Medium confidence - requires token overlap + type-specific rules
        Tier 3 (<85%): Rejected

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

        # Check 1: Empty name guard
        if not name1 or not name2:
            return False, "empty_name"

        # Check 2: SEMANTIC BLOCKLIST (Phase 6) - checked FIRST before all other checks
        # Case-insensitive, bidirectional lookup
        norm1_lower = name1.lower().strip()
        norm2_lower = name2.lower().strip()
        if (norm1_lower, norm2_lower) in self._semantic_blocklist_set:
            self.stats['failed_semantic_blocklist'] += 1
            reason = "semantic_blocklist"
            if log_rejection:
                logger.warning(
                    f"REJECT merge (SEMANTIC_BLOCKLIST): '{name1}' + '{name2}'"
                )
            return False, reason

        # Check 3: Type compatibility (strict)
        if self.type_strict_matching and type1 != type2:
            self.stats['failed_type_check'] += 1
            reason = f"type_mismatch: {type1} != {type2}"
            if log_rejection:
                logger.debug(
                    f"REJECT merge: '{name1}' ({type1}) + '{name2}' ({type2}) - {reason}"
                )
            return False, reason

        # Check 4: Explicit blocklist (early exit - before expensive normalization)
        # Note: norm1_lower and norm2_lower already computed for semantic blocklist check
        if (norm1_lower, norm2_lower) in self._blocklist_set:
            self.stats['failed_blocklist_check'] += 1
            reason = "explicit_blocklist"
            if log_rejection:
                logger.warning(
                    f"REJECT merge (BLOCKLIST): '{name1}' + '{name2}'"
                )
            return False, reason

        # Check 5: Advanced normalization
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)

        # Exact match after normalization (always merge)
        if norm1 == norm2:
            self.stats['passed_validations'] += 1
            return True, "exact_match_normalized"

        # Calculate token overlap early (needed for length ratio bypass)
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())
        overlap = self._token_overlap(tokens1, tokens2)
        single_word = len(tokens1) == 1 or len(tokens2) == 1

        # Check 6: Length ratio after normalization (prevent "I" -> "India")
        # Skip if decent token overlap (abbreviation cases like "Dr. Bronner's" + "Bronners")
        # Also skip if only 1-2 tokens total (short names have naturally low length ratios)
        len1_norm, len2_norm = len(norm1), len(norm2)
        total_tokens = len(tokens1) + len(tokens2)
        if len1_norm > 3 and len2_norm > 3 and overlap < 0.5 and total_tokens > 3:
            length_ratio = min(len1_norm, len2_norm) / max(len1_norm, len2_norm)
            if length_ratio < self.min_length_ratio:
                self.stats['failed_length_check'] += 1
                reason = f"length_mismatch: {len1_norm} vs {len2_norm} (ratio={length_ratio:.2f})"
                if log_rejection:
                    logger.debug(
                        f"REJECT merge: '{name1}' + '{name2}' - {reason}"
                    )
                return False, reason

        # Calculate fuzzy similarity on normalized names
        char_score = fuzz.ratio(norm1, norm2)

        # Check 7: Punctuation-only difference (special case)
        if self._has_punctuation_only_difference(name1, name2, norm1, norm2):
            if char_score >= 92:  # Lower threshold for punctuation diffs
                self.stats['passed_validations'] += 1
                return True, f"punctuation_only: score={char_score}"

        # Check 8: Substring/contains relationship (special case for abbreviations)
        # "Bronners" contains "Bronner", "Dr Bronners" contains "Bronner"
        if norm1 in norm2 or norm2 in norm1:
            # For substring matches, require high token overlap
            if overlap >= 0.7:  # 70% token overlap
                self.stats['passed_validations'] += 1
                return True, f"substring_match: overlap={overlap:.2f}"

        # Check 9: Title-only differences for flexible types (recovers Dr./Doctor abbreviations)
        if type1 in self.FLEXIBLE_TYPES and overlap >= 0.5 and char_score >= 60:
            if self._only_title_difference(tokens1, tokens2):
                if self.semantic_validation and not self._check_semantic_compatibility(norm1_lower, norm2_lower):
                    self.stats['failed_semantic_check'] += 1
                    reason = "title_only_semantic_block"
                    if log_rejection:
                        logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
                    return False, reason

                self.stats['passed_validations'] += 1
                return True, f"title_only_difference: overlap={overlap:.2f}, score={char_score}"

        # Check 10: PERSON name variations (first+last vs first+middle+last)
        # "Aaron Perry" should merge with "Aaron William Perry"
        if type1 == 'PERSON' and len(tokens1) >= 2 and len(tokens2) >= 2:
            # Check if one is a subset of the other (all tokens in shorter are in longer)
            if tokens1.issubset(tokens2) or tokens2.issubset(tokens1):
                # Get ordered tokens from original names for first/last comparison
                tokens1_ordered = norm1.split()
                tokens2_ordered = norm2.split()
                shorter = tokens1_ordered if len(tokens1_ordered) < len(tokens2_ordered) else tokens2_ordered
                longer = tokens2_ordered if len(tokens1_ordered) < len(tokens2_ordered) else tokens1_ordered

                # Verify first name and last name match
                if shorter[0] == longer[0] and shorter[-1] == longer[-1]:
                    self.stats['passed_validations'] += 1
                    return True, f"person_name_variation: '{name1}' + '{name2}' (first/last match)"

        # === TWO-TIER THRESHOLD STRATEGY ===

        # TIER 1: High confidence (≥95%)
        if char_score >= 95:
            # Still check semantic compatibility
            if self.semantic_validation and not self._check_semantic_compatibility(norm1_lower, norm2_lower):
                self.stats['failed_semantic_check'] += 1
                reason = "semantic_incompatibility"
                if log_rejection:
                    logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
                return False, reason

            self.stats['passed_validations'] += 1
            return True, f"tier1_approved: score={char_score}"

        # TIER 2: Medium confidence (85-94%) - Requires additional validation
        elif 85 <= char_score < 95:
            # Only allow Tier 2 for flexible types (not PLACE/EVENT)
            if type1 in self.STRICT_TYPES:
                self.stats['failed_similarity_check'] += 1
                reason = f"strict_type_tier2_blocked: {type1} requires ≥95% (got {char_score})"
                if log_rejection:
                    logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
                return False, reason

            # Require stronger token support for Tier 2 merges
            if overlap < 0.7:
                self.stats['failed_similarity_check'] += 1
                reason = f"tier2_low_overlap: {overlap:.2f} < 0.7 (score={char_score})"
                if log_rejection:
                    logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
                return False, reason

            # Single-word names need higher score and shared substring
            if single_word:
                if char_score < 90:
                    self.stats['failed_similarity_check'] += 1
                    reason = f"tier2_single_word_low: {char_score} < 90"
                    if log_rejection:
                        logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
                    return False, reason

                # Check for shared 3+ char substring
                if not any(len(t) >= 3 for t in tokens1.intersection(tokens2)):
                    self.stats['failed_similarity_check'] += 1
                    reason = "tier2_no_shared_substring"
                    if log_rejection:
                        logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
                    return False, reason

            # Semantic compatibility check
            if self.semantic_validation and not self._check_semantic_compatibility(norm1_lower, norm2_lower):
                self.stats['failed_semantic_check'] += 1
                reason = "tier2_semantic_incompatibility"
                if log_rejection:
                    logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
                return False, reason

            # All Tier 2 checks passed
            self.stats['passed_validations'] += 1
            return True, f"tier2_approved: score={char_score}, overlap={overlap:.2f}, type={type1}"

        # TIER 3: Low confidence (<85%) - Always reject
        else:
            self.stats['failed_similarity_check'] += 1
            reason = f"tier3_rejected: {char_score} < 85"
            if log_rejection:
                logger.debug(f"REJECT merge: '{name1}' + '{name2}' - {reason}")
            return False, reason

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
        logger.info(f"  - Semantic blocklist: {stats['failed_semantic_blocklist']}")
        logger.info(f"  - Type mismatch: {stats['failed_type_check']}")
        logger.info(f"  - Length mismatch: {stats['failed_length_check']}")
        logger.info(f"  - Low similarity: {stats['failed_similarity_check']}")
        logger.info(f"  - Explicit blocklist: {stats['failed_blocklist_check']}")
        logger.info(f"  - Semantic incompatibility: {stats['failed_semantic_check']}")
        logger.info("=" * 60)
