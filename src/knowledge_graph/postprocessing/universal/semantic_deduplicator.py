"""
Semantic Deduplication Module

Detects and consolidates semantically duplicate relationships using embedding similarity.
Addresses V14.0's issue of 25 redundant 'is-a' relationships (4.2% of total issues).

Example duplicates:
- (X, is-a, source of Y for Z1), (X, is-a, source of Y for Z2), (X, is-a, source of Y for Z3)
- (X, is-a, foundation of A), (X, is-a, foundation of B)
- Multiple variations of same relationship with slight wording differences

Strategy:
1. Embed all relationships using sentence transformers
2. Find semantic duplicates using cosine similarity (0.85-0.90 threshold)
3. Keep relationship with highest p_true score
4. Add deduplication metadata flags
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import numpy as np

from ..base import PostProcessingModule, ProcessingContext

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, fall back to simple hashing if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, SemanticDeduplicator will use simpler matching")


class SemanticDeduplicator(PostProcessingModule):
    """
    Detects and removes semantically duplicate relationships.

    V14.1 NEW MODULE - Addresses 25 redundant 'is-a' relationships from V14.0.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize semantic deduplicator.

        Args:
            config: Configuration dict with optional keys:
                - similarity_threshold: Cosine similarity threshold (default: 0.87)
                - model_name: Sentence transformer model (default: 'all-MiniLM-L6-v2')
                - group_by_source: Whether to group by source entity first (default: True)
                - consolidate_is_a_sources: Whether to consolidate 'is-a source of' patterns (default: True)
        """
        # Call parent __init__ to get self.enabled and other base attributes
        super().__init__(config)

        self.similarity_threshold = self.config.get('similarity_threshold', 0.87)
        self.model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        self.group_by_source = self.config.get('group_by_source', True)
        self.consolidate_is_a_sources = self.config.get('consolidate_is_a_sources', True)

        # Initialize embedding model if available
        self.model = None
        self.embeddings_available = EMBEDDINGS_AVAILABLE
        if self.embeddings_available:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"✅ Loaded sentence transformer model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.embeddings_available = False

        # Module-specific statistics (in addition to base stats)
        self.dedup_stats = {
            'duplicates_found': 0,
            'relationships_removed': 0,
            'duplicate_groups': 0
        }

    @property
    def name(self) -> str:
        return "SemanticDeduplicator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def priority(self) -> int:
        return 115  # After Deduplicator (110), before ConfidenceFilter (120)

    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """
        Find and remove semantically duplicate relationships.

        Args:
            relationships: List of ModuleRelationship objects
            context: Processing context

        Returns:
            Deduplicated list of relationships
        """
        if not relationships:
            return relationships

        logger.info(f"🔍 {self.name}: Processing {len(relationships)} relationships")

        # Group by source entity if configured (more efficient)
        if self.group_by_source:
            deduplicated = self._process_by_source_groups(relationships)
        else:
            deduplicated = self._process_all_relationships(relationships)

        removed_count = len(relationships) - len(deduplicated)
        self.dedup_stats['relationships_removed'] = removed_count
        self.stats['filtered_count'] = removed_count  # Update base stats too

        logger.info(f"✅ {self.name}: Removed {removed_count} duplicate relationships")
        logger.info(f"   - Duplicate groups found: {self.dedup_stats['duplicate_groups']}")
        logger.info(f"   - Total duplicates: {self.dedup_stats['duplicates_found']}")

        return deduplicated

    def _process_by_source_groups(self, relationships: List[Any]) -> List[Any]:
        """Process relationships grouped by source entity (more efficient)."""
        # Group by source
        by_source = defaultdict(list)
        for rel in relationships:
            source = getattr(rel, 'source', '')
            by_source[source].append(rel)

        # Process each source group
        deduplicated = []
        for source, group in by_source.items():
            if len(group) == 1:
                # No duplicates possible with single relationship
                deduplicated.extend(group)
            else:
                # Find duplicates within this source group
                group_deduplicated = self._find_and_remove_duplicates(group)
                deduplicated.extend(group_deduplicated)

        return deduplicated

    def _process_all_relationships(self, relationships: List[Any]) -> List[Any]:
        """Process all relationships without grouping."""
        return self._find_and_remove_duplicates(relationships)

    def _find_and_remove_duplicates(self, relationships: List[Any]) -> List[Any]:
        """
        Find semantic duplicates and keep only the best one from each group.

        Strategy:
        1. If embeddings available: Use cosine similarity
        2. Else: Use simple string matching fallback
        3. Keep relationship with highest p_true score
        4. Add deduplication flags to kept relationships
        """
        if len(relationships) <= 1:
            return relationships

        # Find duplicate groups
        if self.embeddings_available and self.model:
            duplicate_groups = self._find_duplicates_with_embeddings(relationships)
        else:
            duplicate_groups = self._find_duplicates_with_strings(relationships)

        if not duplicate_groups:
            return relationships

        # Track which relationships to keep
        to_keep = set(range(len(relationships)))

        # Process each duplicate group
        for group_indices in duplicate_groups:
            if len(group_indices) < 2:
                continue

            self.dedup_stats['duplicate_groups'] += 1
            self.dedup_stats['duplicates_found'] += len(group_indices)

            # Find best relationship in group (highest p_true)
            best_idx = self._find_best_relationship(relationships, group_indices)

            # Mark the best one as deduplicated source
            best_rel = relationships[best_idx]
            if not hasattr(best_rel, 'flags') or best_rel.flags is None:
                best_rel.flags = {}
            best_rel.flags['DEDUPLICATED_KEEPER'] = True
            best_rel.flags['duplicate_group_size'] = len(group_indices)

            # Remove others from keep set
            for idx in group_indices:
                if idx != best_idx:
                    to_keep.discard(idx)

        # Return only relationships to keep
        return [relationships[i] for i in sorted(to_keep)]

    def _find_duplicates_with_embeddings(self, relationships: List[Any]) -> List[List[int]]:
        """Find duplicate groups using embedding similarity."""
        # Create relationship text for embedding
        rel_texts = []
        for rel in relationships:
            source = getattr(rel, 'source', '')
            predicate = getattr(rel, 'predicate', '')
            target = getattr(rel, 'target', '')
            # Combine into text for embedding
            rel_text = f"{source} {predicate} {target}"
            rel_texts.append(rel_text)

        # Generate embeddings
        try:
            embeddings = self.model.encode(rel_texts, convert_to_numpy=True)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}, falling back to string matching")
            return self._find_duplicates_with_strings(relationships)

        # Calculate similarity matrix
        similarity_matrix = self._cosine_similarity_matrix(embeddings)

        # Find duplicate groups
        duplicate_groups = []
        processed = set()

        for i in range(len(relationships)):
            if i in processed:
                continue

            # Find all similar relationships
            similar_indices = [i]
            for j in range(i + 1, len(relationships)):
                if j in processed:
                    continue

                if similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_indices.append(j)
                    processed.add(j)

            if len(similar_indices) > 1:
                duplicate_groups.append(similar_indices)

            processed.add(i)

        return duplicate_groups

    def _find_duplicates_with_strings(self, relationships: List[Any]) -> List[List[int]]:
        """Fallback: Find duplicates using simple string matching."""
        # Group by exact match of (source, predicate, target)
        groups = defaultdict(list)
        for i, rel in enumerate(relationships):
            source = getattr(rel, 'source', '').lower().strip()
            predicate = getattr(rel, 'predicate', '').lower().strip()
            target = getattr(rel, 'target', '').lower().strip()
            key = (source, predicate, target)
            groups[key].append(i)

        # Return groups with > 1 member
        return [indices for indices in groups.values() if len(indices) > 1]

    def _cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix for embeddings."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Calculate similarity matrix
        similarity = np.dot(normalized, normalized.T)

        return similarity

    def _find_best_relationship(self, relationships: List[Any], indices: List[int]) -> int:
        """
        Find the best relationship from a duplicate group.

        Selection criteria (in order):
        1. Highest p_true score
        2. Fewest flags (cleaner relationship)
        3. Shortest evidence text (more concise)
        4. First one (stable sort)
        """
        best_idx = indices[0]
        best_rel = relationships[best_idx]
        best_p_true = getattr(best_rel, 'p_true', 0.0)
        best_flag_count = len(getattr(best_rel, 'flags', {}) or {})
        best_evidence_len = len(getattr(best_rel, 'evidence_text', ''))

        for idx in indices[1:]:
            rel = relationships[idx]
            p_true = getattr(rel, 'p_true', 0.0)
            flag_count = len(getattr(rel, 'flags', {}) or {})
            evidence_len = len(getattr(rel, 'evidence_text', ''))

            # Compare criteria
            if p_true > best_p_true:
                best_idx = idx
                best_rel = rel
                best_p_true = p_true
                best_flag_count = flag_count
                best_evidence_len = evidence_len
            elif p_true == best_p_true:
                # Tie-breaker 1: Fewer flags
                if flag_count < best_flag_count:
                    best_idx = idx
                    best_rel = rel
                    best_flag_count = flag_count
                    best_evidence_len = evidence_len
                elif flag_count == best_flag_count:
                    # Tie-breaker 2: Shorter evidence
                    if evidence_len < best_evidence_len:
                        best_idx = idx
                        best_rel = rel
                        best_evidence_len = evidence_len

        return best_idx

    def get_summary(self) -> Dict[str, Any]:
        """Return module statistics (override base method to add dedup-specific stats)."""
        summary = super().get_summary()
        summary.update({
            'duplicates_found': self.dedup_stats['duplicates_found'],
            'relationships_removed': self.dedup_stats['relationships_removed'],
            'duplicate_groups': self.dedup_stats['duplicate_groups'],
            'embeddings_available': self.embeddings_available,
            'similarity_threshold': self.similarity_threshold
        })
        return summary
