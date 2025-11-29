#!/usr/bin/env python3
"""
One-off Discourse Graph Transformer (Hybrid Model - Option B)

Transforms existing knowledge graph to add discourse graph elements:
- Identifies opinion/recommendation/philosophical relationships
- Creates Claim nodes for multi-source statements
- Adds attribution edges (Person --MAKES_CLAIM--> Claim)
- Calculates consensus scores

This is a PROTOTYPE for testing discourse graph benefits.
For production, these transformations should be in the extraction pipeline.

Usage:
    python scripts/transform_to_discourse_graph.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscourseGraphTransformer:
    """
    Transforms knowledge graph to include discourse graph elements.

    Hybrid Model (Option B):
    - Keep factual relationships as-is
    - Transform opinion/philosophical/recommendation relationships into claims
    - Add attribution and consensus tracking
    """

    def __init__(self):
        self.claims = {}
        self.claim_id_counter = 0
        self.attribution_edges = []

        # Predicates that warrant claim transformation
        self.claim_worthy_predicates = {
            # Advocacy/belief
            'advocates_for', 'believes', 'supports', 'opposes',
            # Recommendations
            'recommends', 'suggests', 'advises', 'encourages',
            # Philosophical
            'represents', 'symbolizes', 'embodies', 'reflects',
            'signifies', 'manifests', 'expresses'
        }

        # Factual predicates - keep as-is
        self.factual_predicates = {
            'authored', 'wrote', 'published', 'founded', 'located',
            'born', 'died', 'established', 'created', 'produced',
            'has_part', 'part_of', 'member_of', 'employed_by', 'works_for'
        }

    def is_claim_worthy(self, relationship: Dict) -> bool:
        """Check if relationship should be transformed into claim"""
        # Handle both formats: unified graph (predicate) and original (relationship_type)
        predicate = (relationship.get('predicate') or relationship.get('relationship_type', '')).lower().strip()

        # Check if marked by ClaimClassifier
        classification_flags = relationship.get('classification_flags', [])
        if classification_flags:
            if any(flag in ['opinion', 'recommendation', 'philosophical']
                   for flag in classification_flags):
                return True

        # Check if predicate is claim-worthy
        if predicate in self.claim_worthy_predicates:
            return True

        # Check if explicitly factual (don't transform)
        if predicate in self.factual_predicates:
            return False

        return False

    def create_claim_statement(self, relationship: Dict) -> str:
        """
        Create claim statement from relationship.

        Format: "{subject} {predicate} {object}"
        Example: "Permaculture is beneficial for sustainable agriculture"
        """
        # Handle both formats: unified graph (source/target/predicate) and original (source_entity/target_entity/relationship_type)
        source = relationship.get('source') or relationship.get('source_entity')
        target = relationship.get('target') or relationship.get('target_entity')
        predicate = relationship.get('predicate') or relationship.get('relationship_type', '')
        predicate = predicate.replace('_', ' ')

        # Use description if available (more natural language)
        description = relationship.get('description', '')
        if description and len(description) > 20:
            return description

        # Use evidence text if available
        evidence = relationship.get('evidence', {})
        if isinstance(evidence, dict):
            evidence_text = evidence.get('text', '')
            if evidence_text and len(evidence_text) > 20:
                return evidence_text

        # Otherwise construct from triple
        return f"{source} {predicate} {target}"

    def find_similar_claim(self, claim_text: str, similarity_threshold: int = 85) -> str:
        """Find existing similar claim by text similarity"""
        claim_text_lower = claim_text.lower()

        for claim_id, claim_data in self.claims.items():
            existing_text = claim_data['claim_text'].lower()
            similarity = fuzz.ratio(claim_text_lower, existing_text)

            if similarity >= similarity_threshold:
                return claim_id

        return None

    def add_or_update_claim(
        self,
        claim_text: str,
        source_entity: str,
        target_entity: str,
        predicate: str,
        source_metadata: Dict
    ) -> str:
        """
        Add new claim or update existing with attribution.

        Returns claim_id
        """
        # Check for existing similar claim
        existing_claim_id = self.find_similar_claim(claim_text)

        if existing_claim_id:
            # Update existing claim with new attribution
            claim = self.claims[existing_claim_id]
            claim['attributions'].append({
                'source': source_entity,
                'source_type': 'person',  # Could enhance type detection
                'provenance': source_metadata
            })
            claim['source_count'] += 1
            return existing_claim_id

        # Create new claim
        claim_id = f"claim_{self.claim_id_counter}"
        self.claim_id_counter += 1

        self.claims[claim_id] = {
            'id': claim_id,
            'type': 'CLAIM',
            'claim_text': claim_text,
            'about': target_entity,  # The concept being discussed
            'predicate': predicate,
            'attributions': [{
                'source': source_entity,
                'source_type': 'person',
                'provenance': source_metadata
            }],
            'source_count': 1,
            'consensus_score': 0.0  # Will calculate later
        }

        return claim_id

    def transform_relationships(
        self,
        relationships: List[Dict],
        keep_original: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Transform claim-worthy relationships.

        Returns:
            (transformed_relationships, attribution_edges)
        """
        logger.info(f"Transforming {len(relationships)} relationships...")

        transformed = []
        claim_worthy_count = 0

        for rel in relationships:
            if self.is_claim_worthy(rel):
                claim_worthy_count += 1

                # Create claim statement
                claim_text = self.create_claim_statement(rel)

                # Handle both formats
                source_entity = rel.get('source') or rel.get('source_entity')
                target_entity = rel.get('target') or rel.get('target_entity')
                predicate = rel.get('predicate') or rel.get('relationship_type')

                # Extract metadata
                metadata = rel.get('metadata', {})

                # Add or update claim
                claim_id = self.add_or_update_claim(
                    claim_text=claim_text,
                    source_entity=source_entity,
                    target_entity=target_entity,
                    predicate=predicate,
                    source_metadata=metadata
                )

                # Create attribution edge: Person --MAKES_CLAIM--> Claim
                attribution = {
                    'source_entity': source_entity,
                    'target_entity': claim_id,
                    'relationship_type': 'MAKES_CLAIM',
                    'description': f"{source_entity} makes claim: {claim_text}",
                    'metadata': {
                        **metadata,
                        'original_predicate': predicate,
                        'discourse_element': 'attribution'
                    }
                }
                self.attribution_edges.append(attribution)

                # Create concept link: Claim --ABOUT--> Concept
                about_edge = {
                    'source_entity': claim_id,
                    'target_entity': target_entity,
                    'relationship_type': 'ABOUT',
                    'description': f"Claim about {target_entity}",
                    'metadata': {
                        'discourse_element': 'claim_topic'
                    }
                }
                transformed.append(about_edge)

                # Optionally keep original relationship
                if keep_original:
                    rel_copy = dict(rel)
                    rel_copy['metadata'] = {
                        **rel_copy.get('metadata', {}),
                        'transformed_to_claim': claim_id
                    }
                    transformed.append(rel_copy)
            else:
                # Keep factual relationships as-is
                transformed.append(rel)

        logger.info(f"  Claim-worthy relationships: {claim_worthy_count}")
        logger.info(f"  Created {len(self.claims)} unique claims")
        logger.info(f"  Created {len(self.attribution_edges)} attribution edges")

        return transformed, self.attribution_edges

    def calculate_consensus_scores(self):
        """Calculate consensus scores for claims based on source count"""
        if not self.claims:
            return

        max_sources = max(claim['source_count'] for claim in self.claims.values())

        for claim in self.claims.values():
            # Simple consensus score: source_count / max_sources
            claim['consensus_score'] = claim['source_count'] / max_sources if max_sources > 0 else 0

            # Add source diversity metrics
            provenances = [attr['provenance'] for attr in claim['attributions']]
            episodes = set(p.get('episode_number') for p in provenances if p.get('episode_number'))
            books = set(p.get('book_slug') for p in provenances if p.get('book_slug'))

            claim['source_diversity'] = {
                'episode_count': len(episodes),
                'book_count': len(books),
                'total_sources': len(episodes) + len(books)
            }

    def get_claims_as_entities(self) -> List[Dict]:
        """Convert claims to entity format for knowledge graph"""
        entities = []

        for claim_id, claim in self.claims.items():
            entities.append({
                'id': claim_id,
                'name': claim['claim_text'][:100],  # Truncate for readability
                'type': 'CLAIM',
                'description': claim['claim_text'],
                'metadata': {
                    'about': claim['about'],
                    'predicate': claim['predicate'],
                    'source_count': claim['source_count'],
                    'consensus_score': claim['consensus_score'],
                    'source_diversity': claim['source_diversity'],
                    'attributions': claim['attributions']
                }
            })

        return entities


def transform_unified_graph(input_file: Path, output_file: Path):
    """
    Transform unified knowledge graph to include discourse elements.

    Args:
        input_file: Path to unified_knowledge_graph.json
        output_file: Path to save discourse_graph.json
    """
    logger.info(f"Loading unified graph: {input_file}")

    with open(input_file) as f:
        data = json.load(f)

    logger.info(f"  Entities: {len(data.get('entities', []))}")
    logger.info(f"  Relationships: {len(data.get('relationships', []))}")

    # Transform
    transformer = DiscourseGraphTransformer()

    transformed_rels, attribution_edges = transformer.transform_relationships(
        data.get('relationships', []),
        keep_original=True  # Keep original for comparison
    )

    # Calculate consensus
    transformer.calculate_consensus_scores()

    # Get claims as entities
    claim_entities = transformer.get_claims_as_entities()

    # Merge everything
    all_relationships = transformed_rels + attribution_edges

    # Handle entities - unified graph has dict format, convert to dict with claims added
    entities = data.get('entities', {})
    if isinstance(entities, dict):
        # Add claim entities to dict
        for claim_ent in claim_entities:
            entities[claim_ent['id']] = claim_ent
        all_entities = entities
    else:
        # List format (shouldn't happen but handle it)
        all_entities = entities + claim_entities

    # Save
    output_data = {
        'graph_type': 'discourse_graph_hybrid',
        'description': 'Hybrid knowledge + discourse graph (Option B)',
        'entities': all_entities,
        'relationships': all_relationships,
        'discourse_stats': {
            'claims_created': len(claim_entities),
            'attribution_edges': len(attribution_edges),
            'multi_source_claims': sum(1 for c in transformer.claims.values() if c['source_count'] > 1),
            'max_sources_per_claim': max((c['source_count'] for c in transformer.claims.values()), default=0)
        },
        'original_stats': data.get('classification_metadata', {})
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"✅ Saved discourse graph: {output_file}")
    logger.info(f"📊 Discourse Stats:")
    logger.info(f"  - Claims created: {len(claim_entities)}")
    logger.info(f"  - Attribution edges: {len(attribution_edges)}")
    logger.info(f"  - Multi-source claims: {sum(1 for c in transformer.claims.values() if c['source_count'] > 1)}")
    logger.info(f"  - Entities: {len(data.get('entities', []))} → {len(all_entities)} (+{len(claim_entities)})")
    logger.info(f"  - Relationships: {len(data.get('relationships', []))} → {len(all_relationships)} (+{len(attribution_edges)})")


def main():
    project_root = Path("/home/claudeuser/yonearth-gaia-chatbot")

    # Input: unified knowledge graph (default to latest hybrid build)
    input_file = project_root / "data/knowledge_graph_unified/unified_hybrid.json"

    # Output: discourse graph
    output_file = project_root / "data/knowledge_graph_unified/discourse_graph_hybrid.json"

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Expected: data/knowledge_graph_unified/unified_normalized.json")
        return

    logger.info("="*60)
    logger.info("DISCOURSE GRAPH TRANSFORMATION (Hybrid Model - Option B)")
    logger.info("="*60)
    logger.info(f"Input:  {input_file.name}")
    logger.info(f"Output: {output_file.name}")
    logger.info("")

    transform_unified_graph(input_file, output_file)


if __name__ == "__main__":
    main()
