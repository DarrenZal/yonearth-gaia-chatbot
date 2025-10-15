"""
List Splitter Module

Splits list targets (e.g., "A, B, and C") into separate relationships.

Features:
- POS tagging to distinguish adjective series from noun lists
- 'and' conjunction pattern handling (V8)
- Preserves adjective series (e.g., "physical, mental, spiritual growth")
- Splits true lists (e.g., "families, communities and planet")
- Smart POS-based "and" splitting (V11.2.2): Only splits noun phrases

Version History:
- v1.0.0 (V6): Basic list splitting with POS tagging
- v1.1.0 (V8): Enhanced 'and' conjunction patterns
- v1.2.0 (V11.2): Split on ALL " and " conjunctions, handle compound terms
- v1.3.0 (V11.2.2): Fixed aggressive splitting - only split "and" when connecting nouns
"""

import re
import logging
from typing import Optional, List, Dict, Any

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)

# Try to load spaCy for POS tagging
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None


class ListSplitter(PostProcessingModule):
    """
    Splits list targets into separate relationships using POS tagging.

    Content Types: Universal (works for all content types)
    Priority: 40 (runs early, before other target processing)
    """

    name = "ListSplitter"
    description = "Splits list targets with POS-aware 'and' splitting (nouns only)"
    content_types = ["all"]
    priority = 40
    dependencies = []
    version = "1.3.0"  # V11.2.2 enhanced

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.min_list_length = self.config.get('min_list_length', 15)
        self.use_pos_tagging = self.config.get('use_pos_tagging', True) and (nlp is not None)

        # V8 NEW: Enhanced list patterns with 'and' conjunctions
        self.list_patterns = [
            # Pattern 1: A, B, and C (Oxford comma)
            r'([^,]+),\s*([^,]+),\s*and\s+([^,]+)',
            # Pattern 2: A, B and C (no Oxford comma)
            r'([^,]+),\s*([^,]+)\s+and\s+([^,]+)',
            # Pattern 3: A and B (simple conjunction)
            r'([^,]+)\s+and\s+([^,]+)',
            # Pattern 4: A, B (simple comma)
            r'([^,]+),\s*([^,]+)',
        ]

        if self.use_pos_tagging:
            logger.info("✅ ListSplitter using POS tagging for intelligent splitting + 'and' conjunctions (V8)")
        else:
            logger.info("⚠️  ListSplitter using fallback logic (no POS tagging)")

    def is_adjective_series(self, target: str) -> bool:
        """Use POS tagging to detect adjective series"""
        if not self.use_pos_tagging:
            return False

        doc = nlp(target)
        tokens = [token for token in doc]
        if len(tokens) < 3:
            return False

        # Find last noun
        last_noun_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].pos_ == 'NOUN':
                last_noun_idx = i
                break

        if last_noun_idx is None:
            return False

        # Check if tokens before last noun are mostly adjectives
        prefix_tokens = tokens[:last_noun_idx]
        if not prefix_tokens:
            return False

        adjective_count = sum(1 for t in prefix_tokens if t.pos_ == 'ADJ')
        coord_count = sum(1 for t in prefix_tokens if t.dep_ == 'cc')

        return adjective_count >= len(prefix_tokens) * 0.6

    def is_list_target(self, target: str) -> bool:
        """Check if target contains a list pattern (commas or 'and')"""
        # V8 ENHANCEMENT: Also check for 'and' without commas
        if ',' not in target and ' and ' not in target:
            return False

        if len(target) < self.min_list_length:
            return False

        # Check if it's an adjective series
        if self.is_adjective_series(target):
            return False  # Don't split adjective series

        # Check for list patterns
        if ' and ' in target or ',' in target:
            return True

        comma_count = target.count(',')
        if comma_count >= 2:
            return True

        return False

    def should_split_on_and(self, target: str, and_match_start: int, and_match_end: int) -> bool:
        """
        V11.2.2 FIX: Use POS tagging to determine if 'and' should cause a split.

        Only split when 'and' connects NOUN phrases.
        Don't split when 'and' connects:
        - Adverbs (e.g., "literally and deeply")
        - Verbs (e.g., "heals and restores")
        - Adjectives in compound phrases

        Args:
            target: Full target text
            and_match_start: Character position where regex match starts (includes leading space)
            and_match_end: Character position where regex match ends (includes trailing space)

        Returns:
            True if should split, False otherwise
        """
        if not self.use_pos_tagging:
            return True  # Fallback: split all

        # Parse the target with spaCy
        doc = nlp(target)

        # Find the "and" token by looking for tokens within the match range
        # Note: and_match_start includes the leading space from regex r'\s+and\s+'
        and_token = None
        for token in doc:
            if token.lower_ == 'and' and and_match_start <= token.idx < and_match_end:
                and_token = token
                break

        if and_token is None:
            return True  # Fallback if can't find token

        # Check the POS tags of words connected by "and"
        # Look at the token before "and" and the token after "and"
        and_idx = and_token.i

        if and_idx == 0 or and_idx >= len(doc) - 1:
            return False  # "and" at start/end - don't split

        left_token = doc[and_idx - 1]
        right_token = doc[and_idx + 1]

        # Define noun POS tags
        noun_pos = {'NOUN', 'PROPN', 'PRON'}

        # Check if BOTH sides are nouns
        left_is_noun = left_token.pos_ in noun_pos
        right_is_noun = right_token.pos_ in noun_pos

        # Only split if both sides are nouns
        if left_is_noun and right_is_noun:
            return True

        # Special case: Check if this is a noun phrase split
        # Example: "families, communities and planet" → last token is "planet" (NOUN)
        # But left token might be punctuation, so check noun phrase heads
        if left_token.head.pos_ in noun_pos and right_token.pos_ in noun_pos:
            return True

        # Otherwise, don't split (it's connecting adverbs, verbs, adjectives, etc.)
        return False

    def split_target_list(self, target: str) -> List[str]:
        """V11.2.2 FIX: Split on commas AND 'and' conjunctions, but only when 'and' connects nouns"""

        # Check if target contains list indicators
        if ',' not in target and ' and ' not in target:
            return [target]

        # Compound terms that should NOT be split (even if they contain "and")
        compound_terms = [
            'bread and butter', 'research and development', 'trial and error',
            'supply and demand', 'law and order', 'give and take',
            'ups and downs', 'back and forth', 'black and white',
            'body and soul', 'peace and quiet', 'life and death'
        ]

        # Check if entire target is a compound term
        target_lower = target.lower().strip()
        for compound in compound_terms:
            if target_lower == compound:
                return [target]

        # V11.2.2 FIX: Use POS tagging to decide which " and " to split on
        if self.use_pos_tagging and ' and ' in target:
            # Find all occurrences of " and "
            and_pattern = re.compile(r'\s+and\s+', re.IGNORECASE)

            # Build a list of split positions (commas always split, "and" conditionally)
            split_positions = []

            # Add comma positions
            for match in re.finditer(r',\s*', target):
                split_positions.append((match.start(), match.end(), True))  # Always split

            # Add "and" positions (check POS first)
            for match in and_pattern.finditer(target):
                should_split = self.should_split_on_and(target, match.start(), match.end())
                if should_split:
                    split_positions.append((match.start(), match.end(), True))
                # If shouldn't split, don't add to list

            # Sort by position
            split_positions.sort(key=lambda x: x[0])

            # Extract items based on split positions
            items = []
            last_end = 0
            for start, end, _ in split_positions:
                item = target[last_end:start].strip()
                if item:
                    items.append(item)
                last_end = end

            # Add remaining text after last split
            if last_end < len(target):
                item = target[last_end:].strip()
                if item:
                    items.append(item)
        else:
            # Fallback: Split on commas and " and " (old V11.2 behavior)
            items = re.split(r',\s*|\s+and\s+', target)
            items = [item.strip() for item in items if item.strip()]

        # If we only got 1 item, return it
        if len(items) <= 1:
            return [target]

        # Filter out empty strings and duplicates while preserving order
        seen = set()
        unique_items = []
        for item in items:
            item_lower = item.lower().strip()
            if item_lower and item_lower not in seen:
                seen.add(item_lower)
                unique_items.append(item.strip())

        # If splitting resulted in items that are too short (< 2 characters),
        # it's probably not a real list
        if any(len(item) < 2 for item in unique_items):
            return [target]

        return unique_items if len(unique_items) > 1 else [target]

    def split_relationship(self, rel: Any) -> List[Any]:
        """Split a single relationship with list target into multiple"""
        items = self.split_target_list(rel.target)

        if len(items) <= 1:
            return [rel]

        split_rels = []
        for i, item in enumerate(items):
            # Create new relationship with same attributes
            # V11.2.1 FIX: Don't pass @property fields (knowledge_plausibility, pattern_prior,
            # claim_uid, extraction_metadata) - they're computed, not __init__ parameters
            new_rel_dict = {
                'source': rel.source,
                'relationship': rel.relationship,
                'target': item,
                'source_type': rel.source_type,
                'target_type': rel.target_type,
                'context': rel.evidence_text,  # V11.2.1: Use 'context' for ModuleRelationship
                'page': rel.evidence.get('page_number', 0),  # V11.2.1: Use 'page' for ModuleRelationship
                'text_confidence': rel.text_confidence,
                'p_true': rel.p_true,
                'signals_conflict': rel.signals_conflict,
                'conflict_explanation': rel.conflict_explanation,
                'suggested_correction': getattr(rel, 'suggested_correction', None),
                'classification_flags': getattr(rel, 'classification_flags', []),
                'candidate_uid': rel.candidate_uid + f"_split_{i}",
                # Note: evidence_text, evidence, flags are set by __post_init__
            }

            # Create new relationship using same class
            new_rel = type(rel)(**new_rel_dict)

            # Update flags
            if new_rel.flags is None:
                new_rel.flags = {}
            new_rel.flags['LIST_SPLIT'] = True
            new_rel.flags['split_index'] = i
            new_rel.flags['split_total'] = len(items)
            new_rel.flags['original_target'] = rel.target

            split_rels.append(new_rel)

        return split_rels

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """Process batch of relationships to split list targets"""

        # Reset stats
        self.stats['processed_count'] = len(relationships)
        self.stats['modified_count'] = 0

        processed = []
        split_count = 0
        adjective_series_preserved = 0
        original_count = len(relationships)

        for rel in relationships:
            # Track adjective series that we preserve
            if ',' in rel.target and self.is_adjective_series(rel.target):
                adjective_series_preserved += 1
                processed.append(rel)
            elif self.is_list_target(rel.target):
                split_rels = self.split_relationship(rel)
                processed.extend(split_rels)
                if len(split_rels) > 1:
                    split_count += 1
                    self.stats['modified_count'] += 1
            else:
                processed.append(rel)

        new_count = len(processed)
        logger.info(
            f"   {self.name}: {split_count} lists split, "
            f"{adjective_series_preserved} adjective series preserved, "
            f"{original_count} → {new_count} relationships"
        )

        return processed
