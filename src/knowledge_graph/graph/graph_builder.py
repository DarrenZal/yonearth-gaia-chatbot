"""
Graph builder for merging entities, deduplicating, and creating the knowledge graph.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from fuzzywuzzy import fuzz
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds unified knowledge graph from extracted entities and relationships."""

    def __init__(self, extraction_dir: str, neo4j_client: Neo4jClient):
        """
        Initialize graph builder.

        Args:
            extraction_dir: Directory containing extraction JSON files
            neo4j_client: Neo4j client instance
        """
        self.extraction_dir = Path(extraction_dir)
        self.client = neo4j_client
        self.entities = {}  # entity_id -> entity_data
        self.relationships = []
        self.entity_name_to_id = defaultdict(list)  # name -> [entity_ids]
        self.similarity_threshold = 90  # Fuzzy matching threshold

    def load_extractions(self) -> Dict[str, Any]:
        """
        Load all extraction files and gather statistics.

        Returns:
            Statistics about loaded data
        """
        extraction_files = list(self.extraction_dir.glob("episode_*_extraction.json"))

        total_files = 0
        total_entities_raw = 0
        total_relationships_raw = 0

        for file_path in extraction_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                episode_num = data.get('episode_number', 'unknown')
                entities = data.get('entities', [])
                relationships = data.get('relationships', [])

                total_entities_raw += len(entities)
                total_relationships_raw += len(relationships)
                total_files += 1

                # Store entities with unique IDs
                for entity in entities:
                    entity_id = self._generate_entity_id(entity, episode_num)
                    entity['id'] = entity_id
                    entity['episode_number'] = episode_num
                    self.entities[entity_id] = entity
                    self.entity_name_to_id[entity['name'].lower()].append(entity_id)

                # Store relationships
                for rel in relationships:
                    rel['episode_number'] = episode_num
                    self.relationships.append(rel)

                logger.info(f"Loaded {file_path.name}: {len(entities)} entities, {len(relationships)} relationships")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        stats = {
            'files_loaded': total_files,
            'total_entities_raw': total_entities_raw,
            'total_relationships_raw': total_relationships_raw,
            'unique_entities_before_dedup': len(self.entities)
        }

        logger.info(f"Loaded {total_files} files with {total_entities_raw} entities and {total_relationships_raw} relationships")
        return stats

    def _generate_entity_id(self, entity: Dict[str, Any], episode_num: Any) -> str:
        """Generate unique entity ID."""
        name = entity.get('name', 'unknown').replace(' ', '_').lower()
        entity_type = entity.get('type', 'unknown')
        chunk_id = entity.get('metadata', {}).get('chunk_id', 'unknown')
        return f"{entity_type}_{name}_{episode_num}_{chunk_id}"

    def deduplicate_entities(self) -> Dict[str, Any]:
        """
        Deduplicate entities using fuzzy matching and merge metadata.

        Returns:
            Statistics about deduplication
        """
        logger.info("Starting entity deduplication...")

        # Group entities by type for more accurate matching
        entities_by_type = defaultdict(list)
        for entity_id, entity in self.entities.items():
            entities_by_type[entity['type']].append((entity_id, entity))

        merged_entities = {}
        entity_id_mapping = {}  # old_id -> new_id

        for entity_type, entity_list in entities_by_type.items():
            logger.info(f"Deduplicating {len(entity_list)} entities of type {entity_type}")

            processed = set()

            for entity_id, entity in entity_list:
                if entity_id in processed:
                    continue

                # Find similar entities
                similar_entities = [(entity_id, entity)]
                entity_name = entity['name'].lower()

                for other_id, other_entity in entity_list:
                    if other_id == entity_id or other_id in processed:
                        continue

                    other_name = other_entity['name'].lower()

                    # Check exact match or fuzzy match
                    if entity_name == other_name or fuzz.ratio(entity_name, other_name) >= self.similarity_threshold:
                        similar_entities.append((other_id, other_entity))
                        processed.add(other_id)

                # Merge similar entities
                merged_entity = self._merge_entities(similar_entities)
                canonical_id = similar_entities[0][0]  # Use first entity's ID as canonical
                merged_entities[canonical_id] = merged_entity

                # Create mapping for old IDs to canonical ID
                for old_id, _ in similar_entities:
                    entity_id_mapping[old_id] = canonical_id

                processed.add(entity_id)

        self.entities = merged_entities
        self.entity_id_mapping = entity_id_mapping

        stats = {
            'entities_after_dedup': len(merged_entities),
            'entities_merged': len(entity_id_mapping) - len(merged_entities)
        }

        logger.info(f"Deduplication complete: {len(merged_entities)} unique entities (merged {stats['entities_merged']})")
        return stats

    def _merge_entities(self, entities: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Merge multiple entity records into one.

        Args:
            entities: List of (entity_id, entity_data) tuples

        Returns:
            Merged entity
        """
        if len(entities) == 1:
            return entities[0][1]

        # Use first entity as base
        merged = dict(entities[0][1])

        # Collect all chunks, episodes, descriptions
        all_chunks = set()
        all_episodes = set()
        all_descriptions = []

        for _, entity in entities:
            metadata = entity.get('metadata', {})
            chunks = metadata.get('chunks', [])
            all_chunks.update(chunks)

            episode = entity.get('episode_number')
            if episode is not None:
                all_episodes.add(episode)

            desc = entity.get('description', '').strip()
            if desc:
                all_descriptions.append(desc)

        # Use longest description
        if all_descriptions:
            merged['description'] = max(all_descriptions, key=len)

        # Update metadata
        merged['metadata'] = merged.get('metadata', {})
        merged['metadata']['chunks'] = sorted(list(all_chunks))
        merged['metadata']['episodes'] = sorted(list(all_episodes))
        merged['mention_count'] = len(all_chunks)
        merged['episode_count'] = len(all_episodes)

        return merged

    def deduplicate_relationships(self) -> Dict[str, Any]:
        """
        Deduplicate relationships and update entity references.

        Returns:
            Statistics about relationship deduplication
        """
        logger.info("Deduplicating relationships...")

        unique_relationships = {}

        for rel in self.relationships:
            source = rel['source_entity']
            target = rel['target_entity']
            rel_type = rel['relationship_type']

            # Create unique key for relationship
            rel_key = f"{source}_{rel_type}_{target}".lower()

            if rel_key not in unique_relationships:
                unique_relationships[rel_key] = rel
            else:
                # Merge metadata
                existing = unique_relationships[rel_key]
                existing_chunks = existing.get('metadata', {}).get('chunks', [])
                new_chunks = rel.get('metadata', {}).get('chunks', [])
                all_chunks = list(set(existing_chunks + new_chunks))
                existing['metadata']['chunks'] = all_chunks

        self.relationships = list(unique_relationships.values())

        stats = {
            'unique_relationships': len(self.relationships)
        }

        logger.info(f"Relationship deduplication complete: {len(self.relationships)} unique relationships")
        return stats

    def calculate_entity_importance(self):
        """Calculate importance scores for entities based on mentions and connections."""
        logger.info("Calculating entity importance scores...")

        # Count outgoing and incoming relationships
        outgoing_count = defaultdict(int)
        incoming_count = defaultdict(int)

        for rel in self.relationships:
            source = rel['source_entity'].lower()
            target = rel['target_entity'].lower()
            outgoing_count[source] += 1
            incoming_count[target] += 1

        # Calculate importance score
        for entity_id, entity in self.entities.items():
            name = entity['name'].lower()
            mention_count = entity.get('mention_count', 1)
            episode_count = entity.get('episode_count', 1)
            connections = outgoing_count[name] + incoming_count[name]

            # Importance score: weighted combination of mentions, episodes, and connections
            importance = (mention_count * 0.3) + (episode_count * 0.4) + (connections * 0.3)
            entity['importance_score'] = round(importance, 2)

        logger.info("Entity importance calculation complete")

    def populate_neo4j(self) -> Dict[str, Any]:
        """
        Populate Neo4j database with entities and relationships.

        Returns:
            Statistics about database population
        """
        logger.info("Populating Neo4j database...")

        # Clear existing data
        self.client.clear_database()

        # Create indexes
        self.client.create_indexes()

        # Batch insert entities
        entity_queries = []
        for entity_id, entity in self.entities.items():
            query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                type: $type,
                description: $description,
                mention_count: $mention_count,
                episode_count: $episode_count,
                importance_score: $importance_score,
                episodes: $episodes,
                chunks: $chunks
            })
            """
            params = {
                'id': entity_id,
                'name': entity.get('name', ''),
                'type': entity.get('type', ''),
                'description': entity.get('description', ''),
                'mention_count': entity.get('mention_count', 0),
                'episode_count': entity.get('episode_count', 0),
                'importance_score': entity.get('importance_score', 0.0),
                'episodes': entity.get('metadata', {}).get('episodes', []),
                'chunks': entity.get('metadata', {}).get('chunks', [])
            }
            entity_queries.append((query, params))

        # Execute in batches
        batch_size = 100
        total_entities = 0
        for i in range(0, len(entity_queries), batch_size):
            batch = entity_queries[i:i + batch_size]
            self.client.batch_execute(batch)
            total_entities += len(batch)
            logger.info(f"Inserted {total_entities}/{len(entity_queries)} entities")

        # Build entity name to ID mapping for relationships
        name_to_id = {}
        for entity_id, entity in self.entities.items():
            name_to_id[entity['name'].lower()] = entity_id

        # Insert relationships
        relationship_queries = []
        relationships_created = 0

        for rel in self.relationships:
            source_name = rel['source_entity'].lower()
            target_name = rel['target_entity'].lower()

            source_id = name_to_id.get(source_name)
            target_id = name_to_id.get(target_name)

            if not source_id or not target_id:
                continue

            rel_type = rel['relationship_type'].replace(' ', '_').upper()

            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            CREATE (source)-[r:{rel_type} {{
                description: $description,
                episode: $episode
            }}]->(target)
            """
            params = {
                'source_id': source_id,
                'target_id': target_id,
                'description': rel.get('description', ''),
                'episode': rel.get('episode_number', None)
            }
            relationship_queries.append((query, params))

        # Execute relationship insertions in batches
        total_relationships = 0
        for i in range(0, len(relationship_queries), batch_size):
            batch = relationship_queries[i:i + batch_size]
            try:
                self.client.batch_execute(batch)
                total_relationships += len(batch)
                logger.info(f"Inserted {total_relationships}/{len(relationship_queries)} relationships")
            except Exception as e:
                logger.error(f"Error inserting relationship batch: {e}")

        stats = {
            'entities_inserted': total_entities,
            'relationships_inserted': total_relationships
        }

        logger.info(f"Neo4j population complete: {total_entities} entities, {total_relationships} relationships")
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'entity_type_distribution': self._get_entity_type_distribution(),
            'relationship_type_distribution': self._get_relationship_type_distribution(),
            'top_entities_by_importance': self._get_top_entities(10),
            'top_entities_by_mentions': self._get_top_entities_by_mentions(10),
            'top_entities_by_episodes': self._get_top_entities_by_episodes(10)
        }

    def _get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types."""
        distribution = defaultdict(int)
        for entity in self.entities.values():
            distribution[entity['type']] += 1
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    def _get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types."""
        distribution = defaultdict(int)
        for rel in self.relationships:
            distribution[rel['relationship_type']] += 1
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    def _get_top_entities(self, limit: int) -> List[Dict[str, Any]]:
        """Get top entities by importance score."""
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda x: x.get('importance_score', 0),
            reverse=True
        )
        return [
            {
                'name': e['name'],
                'type': e['type'],
                'importance_score': e.get('importance_score', 0),
                'mention_count': e.get('mention_count', 0),
                'episode_count': e.get('episode_count', 0)
            }
            for e in sorted_entities[:limit]
        ]

    def _get_top_entities_by_mentions(self, limit: int) -> List[Dict[str, Any]]:
        """Get top entities by mention count."""
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda x: x.get('mention_count', 0),
            reverse=True
        )
        return [
            {
                'name': e['name'],
                'type': e['type'],
                'mention_count': e.get('mention_count', 0)
            }
            for e in sorted_entities[:limit]
        ]

    def _get_top_entities_by_episodes(self, limit: int) -> List[Dict[str, Any]]:
        """Get top entities by episode count."""
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda x: x.get('episode_count', 0),
            reverse=True
        )
        return [
            {
                'name': e['name'],
                'type': e['type'],
                'episode_count': e.get('episode_count', 0)
            }
            for e in sorted_entities[:limit]
        ]
