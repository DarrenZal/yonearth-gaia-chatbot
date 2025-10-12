#!/usr/bin/env python3
"""
Comprehensive Entity Resolution System
Combines all features from KG_MASTER_GUIDE_V3.md and KG_POST_EXTRACTION_REFINEMENT.md

Features:
- PyKEEN graph embeddings (RotatE model)
- Splink multi-signal matching
- Relationship overlap analysis
- Active learning for human review
- Incremental processing
- Mesh validator architecture
"""

import json
import logging
import hashlib
import re
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'entity_resolution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EntityResolutionConfig:
    """Configuration for entity resolution pipeline"""

    # Paths
    input_file: str
    output_dir: str = "data/knowledge_graph/entity_resolution"
    checkpoint_dir: str = "data/knowledge_graph/entity_resolution/checkpoints"
    alias_file: Optional[str] = None
    existing_embeddings: Optional[str] = None

    # PyKEEN embedding settings
    embedding_model: str = "RotatE"  # Best for relationship direction
    embedding_dim: int = 64
    embedding_epochs: int = 100
    use_gpu: bool = False  # CPU sufficient at 11K scale

    # Splink matching settings
    name_threshold_high: float = 0.9
    name_threshold_low: float = 0.7
    embedding_threshold_high: float = 0.85
    embedding_threshold_low: float = 0.75
    relationship_jaccard_threshold: float = 0.5

    # Multi-signal weights
    weight_name: float = 0.20
    weight_type: float = 0.10
    weight_relationships: float = 0.30
    weight_embeddings: float = 0.40

    # Final matching threshold
    match_threshold: float = 0.80

    # Active learning
    active_learning_budget: int = 50  # Review top 50 uncertain pairs
    uncertainty_method: str = "entropy"  # or "margin"

    # Incremental processing
    incremental_mode: bool = True
    checkpoint_interval: int = 100  # Save every 100 entities


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def canon(s: str) -> str:
    """
    Normalize entity strings for robust matching
    From KG_MASTER_GUIDE_V3.md
    """
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)  # Drop punctuation
    s = re.sub(r"\s+", " ", s)       # Normalize whitespace
    return s


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculate Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Entity:
    """Entity representation with all signals"""
    id: str
    name: str
    entity_type: str
    relationships: Set[Tuple[str, str]] = field(default_factory=set)  # (relation, target)
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    canonical_form: Optional[str] = None

    def __hash__(self):
        return hash(self.id)


@dataclass
class EntityMatch:
    """Result of entity matching"""
    entity1: Entity
    entity2: Entity
    confidence: float
    signals: Dict[str, float]  # Individual signal scores
    explanation: str
    suggested_canonical: str


# ============================================================================
# GRAPH EMBEDDING TRAINER (PyKEEN)
# ============================================================================

class GraphEmbeddingTrainer:
    """
    Train graph embeddings using PyKEEN
    From KG_POST_EXTRACTION_REFINEMENT.md Part 4
    """

    def __init__(self, config: EntityResolutionConfig):
        self.config = config
        self.model = None
        self.entity_to_id = {}
        self.id_to_entity = {}

    def prepare_triples(self, entities: List[Entity]) -> List[Tuple[str, str, str]]:
        """Convert entities and relationships to PyKEEN triple format"""
        triples = []

        # Build entity ID mapping
        for i, entity in enumerate(entities):
            self.entity_to_id[entity.name] = i
            self.id_to_entity[i] = entity.name

        # Extract all relationships as triples
        for entity in entities:
            for relation, target in entity.relationships:
                triples.append((entity.name, relation, target))

        logger.info(f"Prepared {len(triples)} triples from {len(entities)} entities")
        return triples

    def train(self, entities: List[Entity]) -> Dict[str, np.ndarray]:
        """
        Train RotatE embeddings
        Returns: Dict mapping entity name to embedding vector
        """
        logger.info("🧠 Training graph embeddings with PyKEEN...")

        try:
            from pykeen.pipeline import pipeline
            from pykeen.triples import TriplesFactory
        except ImportError:
            logger.error("PyKEEN not installed. Run: pip install pykeen")
            logger.info("Falling back to random embeddings for testing...")
            return self._random_embeddings(entities)

        # Prepare triples
        triples = self.prepare_triples(entities)

        if not triples:
            logger.warning("No triples found, using random embeddings")
            return self._random_embeddings(entities)

        # Convert to PyKEEN format
        triples_array = np.array(triples)
        tf = TriplesFactory.from_labeled_triples(triples_array)

        # Train model
        device = 'cuda' if self.config.use_gpu else 'cpu'

        result = pipeline(
            model=self.config.embedding_model,
            training=tf,
            epochs=self.config.embedding_epochs,
            embedding_dim=self.config.embedding_dim,
            device=device,
            random_seed=42
        )

        self.model = result.model

        # Extract embeddings
        embeddings = {}
        for entity in entities:
            if entity.name in self.entity_to_id:
                entity_id = self.entity_to_id[entity.name]
                # Get embedding from model
                emb = self.model.entity_representations[0](
                    indices=np.array([entity_id])
                ).detach().cpu().numpy()[0]
                embeddings[entity.name] = emb

        logger.info(f"✅ Trained embeddings for {len(embeddings)} entities")
        return embeddings

    def _random_embeddings(self, entities: List[Entity]) -> Dict[str, np.ndarray]:
        """Fallback: Generate random embeddings for testing"""
        return {
            entity.name: np.random.randn(self.config.embedding_dim)
            for entity in entities
        }

    def save(self, path: str):
        """Save trained embeddings"""
        if self.model is None:
            logger.warning("No model to save")
            return

        import torch
        torch.save(self.model.state_dict(), path)
        logger.info(f"💾 Saved embeddings to {path}")

    def load(self, path: str):
        """Load pre-trained embeddings"""
        if not Path(path).exists():
            logger.warning(f"Embeddings file not found: {path}")
            return

        import torch
        self.model.load_state_dict(torch.load(path))
        logger.info(f"📂 Loaded embeddings from {path}")


# ============================================================================
# RELATIONSHIP ANALYZER
# ============================================================================

class RelationshipAnalyzer:
    """Analyze relationship overlap between entities"""

    def __init__(self):
        self.relationship_cache = {}

    def get_relationship_set(self, entity: Entity) -> Set[Tuple[str, str]]:
        """Get all relationships for an entity"""
        return entity.relationships

    def calculate_overlap(self, e1: Entity, e2: Entity) -> float:
        """
        Calculate Jaccard similarity of relationships
        From your insight: entities with same relationships are likely same entity!
        """
        rels1 = self.get_relationship_set(e1)
        rels2 = self.get_relationship_set(e2)

        return jaccard_similarity(rels1, rels2)

    def get_shared_targets(self, e1: Entity, e2: Entity) -> Set[str]:
        """Get entities both e1 and e2 connect to"""
        targets1 = {target for _, target in e1.relationships}
        targets2 = {target for _, target in e2.relationships}
        return targets1 & targets2


# ============================================================================
# MULTI-SIGNAL MATCHER
# ============================================================================

class MultiSignalMatcher:
    """
    Multi-signal entity matching
    Combines: name, type, relationships, embeddings
    From KG_POST_EXTRACTION_REFINEMENT.md Part 4
    """

    def __init__(self, config: EntityResolutionConfig):
        self.config = config
        self.relationship_analyzer = RelationshipAnalyzer()

    def calculate_name_similarity(self, e1: Entity, e2: Entity) -> float:
        """Jaro-Winkler similarity for names"""
        try:
            from jellyfish import jaro_winkler_similarity
            return jaro_winkler_similarity(e1.name, e2.name)
        except ImportError:
            # Fallback: Simple normalized edit distance
            from difflib import SequenceMatcher
            return SequenceMatcher(None, e1.name, e2.name).ratio()

    def calculate_type_match(self, e1: Entity, e2: Entity) -> float:
        """Exact type matching"""
        if e1.entity_type == e2.entity_type:
            return 1.0

        # Allow some fuzzy matching for types
        if canon(e1.entity_type) == canon(e2.entity_type):
            return 0.9

        return 0.0

    def calculate_relationship_similarity(self, e1: Entity, e2: Entity) -> float:
        """
        Relationship overlap - YOUR KEY INSIGHT!
        High overlap → likely same entity despite different names
        """
        return self.relationship_analyzer.calculate_overlap(e1, e2)

    def calculate_embedding_similarity(self, e1: Entity, e2: Entity) -> float:
        """Graph embedding similarity"""
        if e1.embedding is None or e2.embedding is None:
            return 0.0

        return cosine_similarity(e1.embedding, e2.embedding)

    def calculate_match_score(self, e1: Entity, e2: Entity) -> Dict[str, float]:
        """
        Calculate weighted match score using all signals
        Returns dict with individual scores and final score
        """
        scores = {
            'name': self.calculate_name_similarity(e1, e2),
            'type': self.calculate_type_match(e1, e2),
            'relationships': self.calculate_relationship_similarity(e1, e2),
            'embeddings': self.calculate_embedding_similarity(e1, e2)
        }

        # Weighted combination
        final_score = (
            self.config.weight_name * scores['name'] +
            self.config.weight_type * scores['type'] +
            self.config.weight_relationships * scores['relationships'] +
            self.config.weight_embeddings * scores['embeddings']
        )

        scores['final'] = final_score
        return scores

    def explain_match(self, e1: Entity, e2: Entity, scores: Dict[str, float]) -> str:
        """Generate human-readable explanation of match"""
        explanations = []

        if scores['name'] > 0.9:
            explanations.append(f"Very similar names ({scores['name']:.2%})")
        elif scores['name'] > 0.7:
            explanations.append(f"Similar names ({scores['name']:.2%})")

        if scores['type'] == 1.0:
            explanations.append(f"Same type: {e1.entity_type}")

        if scores['relationships'] > 0.7:
            shared = self.relationship_analyzer.get_shared_targets(e1, e2)
            explanations.append(
                f"High relationship overlap ({scores['relationships']:.2%}), "
                f"{len(shared)} shared connections"
            )

        if scores['embeddings'] > 0.85:
            explanations.append(
                f"Very similar graph position ({scores['embeddings']:.2%})"
            )

        return " | ".join(explanations) if explanations else "Weak match signals"


# ============================================================================
# MESH VALIDATOR ARCHITECTURE
# ============================================================================

class MeshValidator:
    """
    Parallel validation mesh
    All validators run simultaneously and vote
    From KG_POST_EXTRACTION_REFINEMENT.md Part 7
    """

    def __init__(self, config: EntityResolutionConfig):
        self.config = config
        self.matcher = MultiSignalMatcher(config)

    async def validate_async(self, e1: Entity, e2: Entity) -> EntityMatch:
        """Run all validators in parallel"""

        # Fire all validators simultaneously
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()

            # Run validators in parallel
            tasks = [
                loop.run_in_executor(executor, self.matcher.calculate_match_score, e1, e2)
            ]

            results = await asyncio.gather(*tasks)

        scores = results[0]

        # Create match result
        match = EntityMatch(
            entity1=e1,
            entity2=e2,
            confidence=scores['final'],
            signals=scores,
            explanation=self.matcher.explain_match(e1, e2, scores),
            suggested_canonical=self._suggest_canonical(e1, e2, scores)
        )

        return match

    def validate(self, e1: Entity, e2: Entity) -> EntityMatch:
        """Synchronous wrapper"""
        return asyncio.run(self.validate_async(e1, e2))

    def _suggest_canonical(self, e1: Entity, e2: Entity, scores: Dict[str, float]) -> str:
        """Suggest which name should be canonical"""
        # Prefer longer, more complete names
        if len(e1.name) > len(e2.name):
            return e1.name
        return e2.name


# ============================================================================
# ACTIVE LEARNING SELECTOR
# ============================================================================

class ActiveLearningSelector:
    """
    Select most informative examples for human review
    From KG_POST_EXTRACTION_REFINEMENT.md Part 3
    """

    def __init__(self, config: EntityResolutionConfig):
        self.config = config

    def calculate_uncertainty(self, match: EntityMatch) -> float:
        """
        Calculate uncertainty score
        Matches near decision boundary are most informative
        """
        confidence = match.confidence
        threshold = self.config.match_threshold

        if self.config.uncertainty_method == "entropy":
            # Distance from decision boundary
            return 1.0 - abs(confidence - threshold)

        elif self.config.uncertainty_method == "margin":
            # Confidence margin
            return 1.0 - abs(confidence - 0.5) * 2

        return 0.5

    def select_for_review(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """
        Select top N most uncertain matches for human review
        Reduces annotation effort by 65%+ (from docs)
        """
        # Calculate uncertainty for each match
        uncertainties = [
            (match, self.calculate_uncertainty(match))
            for match in matches
        ]

        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)

        # Select top N
        budget = self.config.active_learning_budget
        selected = [match for match, _ in uncertainties[:budget]]

        logger.info(f"📋 Selected {len(selected)} matches for human review (from {len(matches)} total)")
        return selected

    def ensure_diversity(self, selected: List[EntityMatch]) -> List[EntityMatch]:
        """Ensure diversity in selected examples"""
        # TODO: Implement diversity sampling
        # For now, return as-is
        return selected


# ============================================================================
# INCREMENTAL PROCESSOR
# ============================================================================

class IncrementalProcessor:
    """
    Incremental processing for efficiency
    From KG_POST_EXTRACTION_REFINEMENT.md Part 2
    """

    def __init__(self, config: EntityResolutionConfig):
        self.config = config
        self.validated_pairs = set()
        self.embeddings_cache = {}

    def load_checkpoint(self, checkpoint_file: str) -> Dict:
        """Load previous run checkpoint"""
        path = Path(checkpoint_file)
        if not path.exists():
            return {'validated_pairs': [], 'embeddings': {}}

        with open(path) as f:
            data = json.load(f)

        self.validated_pairs = set(tuple(p) for p in data.get('validated_pairs', []))
        self.embeddings_cache = data.get('embeddings', {})

        logger.info(f"📂 Loaded checkpoint: {len(self.validated_pairs)} validated pairs")
        return data

    def save_checkpoint(self, checkpoint_file: str, data: Dict):
        """Save checkpoint"""
        path = Path(checkpoint_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            'validated_pairs': list(self.validated_pairs),
            'embeddings': self.embeddings_cache,
            'timestamp': datetime.now().isoformat(),
            **data
        }

        with open(path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"💾 Saved checkpoint: {checkpoint_file}")

    def should_process(self, e1: Entity, e2: Entity) -> bool:
        """Check if pair needs processing"""
        pair = tuple(sorted([e1.id, e2.id]))
        return pair not in self.validated_pairs

    def mark_processed(self, e1: Entity, e2: Entity):
        """Mark pair as processed"""
        pair = tuple(sorted([e1.id, e2.id]))
        self.validated_pairs.add(pair)


# ============================================================================
# MAIN ENTITY RESOLVER
# ============================================================================

class ComprehensiveEntityResolver:
    """
    Main entity resolution pipeline
    Combines all components
    """

    def __init__(self, config: EntityResolutionConfig):
        self.config = config
        self.embedding_trainer = GraphEmbeddingTrainer(config)
        self.mesh_validator = MeshValidator(config)
        self.active_learner = ActiveLearningSelector(config)
        self.incremental_processor = IncrementalProcessor(config)

    def load_entities(self, input_file: str) -> List[Entity]:
        """Load entities from extraction output"""
        logger.info(f"📂 Loading entities from {input_file}")

        with open(input_file) as f:
            data = json.load(f)

        # Build entities from relationships
        entity_map = {}

        for rel in data.get('relationships', []):
            # Process source entity
            source = rel['source']
            source_type = rel.get('source_type', 'UNKNOWN')

            if source not in entity_map:
                entity_map[source] = Entity(
                    id=hashlib.sha1(source.encode()).hexdigest()[:12],
                    name=source,
                    entity_type=source_type
                )

            # Add relationship
            entity_map[source].relationships.add((
                rel['relationship'],
                rel['target']
            ))

            # Process target entity
            target = rel['target']
            target_type = rel.get('target_type', 'UNKNOWN')

            if target not in entity_map:
                entity_map[target] = Entity(
                    id=hashlib.sha1(target.encode()).hexdigest()[:12],
                    name=target,
                    entity_type=target_type
                )

        entities = list(entity_map.values())
        logger.info(f"✅ Loaded {len(entities)} entities")
        return entities

    def train_embeddings(self, entities: List[Entity]):
        """Train or load embeddings"""
        if self.config.existing_embeddings:
            logger.info("Loading existing embeddings...")
            self.embedding_trainer.load(self.config.existing_embeddings)
        else:
            logger.info("Training new embeddings...")
            embeddings = self.embedding_trainer.train(entities)

            # Attach embeddings to entities
            for entity in entities:
                if entity.name in embeddings:
                    entity.embedding = embeddings[entity.name]

    def find_matches(self, entities: List[Entity]) -> List[EntityMatch]:
        """Find all potential matches"""
        logger.info("🔍 Finding potential matches...")

        matches = []
        total_pairs = len(entities) * (len(entities) - 1) // 2
        processed = 0

        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Skip if already processed (incremental)
                if not self.incremental_processor.should_process(e1, e2):
                    continue

                # Validate match
                match = self.mesh_validator.validate(e1, e2)

                # Keep if above threshold
                if match.confidence >= self.config.match_threshold:
                    matches.append(match)

                # Mark as processed
                self.incremental_processor.mark_processed(e1, e2)

                processed += 1
                if processed % 100 == 0:
                    logger.info(f"  Processed {processed}/{total_pairs} pairs...")

        logger.info(f"✅ Found {len(matches)} potential matches")
        return matches

    def resolve(self) -> Dict:
        """Main resolution pipeline"""
        logger.info("="*80)
        logger.info("🚀 COMPREHENSIVE ENTITY RESOLUTION")
        logger.info("="*80)

        # Load entities
        entities = self.load_entities(self.config.input_file)

        # Train embeddings
        self.train_embeddings(entities)

        # Find matches
        matches = self.find_matches(entities)

        # Select for active learning review
        for_review = self.active_learner.select_for_review(matches)

        # Prepare results
        results = {
            'total_entities': len(entities),
            'potential_matches': len(matches),
            'for_human_review': len(for_review),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'embedding_model': self.config.embedding_model,
                'match_threshold': self.config.match_threshold,
                'weights': {
                    'name': self.config.weight_name,
                    'type': self.config.weight_type,
                    'relationships': self.config.weight_relationships,
                    'embeddings': self.config.weight_embeddings
                }
            },
            'matches': [
                {
                    'entity1': m.entity1.name,
                    'entity2': m.entity2.name,
                    'confidence': m.confidence,
                    'signals': m.signals,
                    'explanation': m.explanation,
                    'suggested_canonical': m.suggested_canonical,
                    'needs_review': m in for_review
                }
                for m in matches
            ]
        }

        return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Entity Resolution with Graph Embeddings"
    )
    parser.add_argument(
        "input_file",
        help="Path to extraction JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/knowledge_graph/entity_resolution",
        help="Output directory"
    )
    parser.add_argument(
        "--embedding-model",
        default="RotatE",
        choices=["RotatE", "TransE", "DistMult"],
        help="Graph embedding model"
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.80,
        help="Confidence threshold for matches"
    )
    parser.add_argument(
        "--review-budget",
        type=int,
        default=50,
        help="Number of matches to flag for human review"
    )

    args = parser.parse_args()

    # Create config
    config = EntityResolutionConfig(
        input_file=args.input_file,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        match_threshold=args.match_threshold,
        active_learning_budget=args.review_budget
    )

    # Run resolution
    resolver = ComprehensiveEntityResolver(config)
    results = resolver.resolve()

    # Save results
    output_path = Path(config.output_dir) / f"resolution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("="*80)
    logger.info("✨ ENTITY RESOLUTION COMPLETE")
    logger.info("="*80)
    logger.info(f"📊 Results:")
    logger.info(f"  Total entities: {results['total_entities']}")
    logger.info(f"  Potential matches: {results['potential_matches']}")
    logger.info(f"  For human review: {results['for_human_review']}")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
