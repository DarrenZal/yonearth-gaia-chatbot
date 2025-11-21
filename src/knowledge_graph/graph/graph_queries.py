"""
Query interface for the knowledge graph with common graph queries.
"""

from typing import List, Dict, Any, Optional
from .neo4j_client import Neo4jClient
import logging

logger = logging.getLogger(__name__)


class GraphQueries:
    """Common graph queries for the YonEarth knowledge graph."""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize graph queries.

        Args:
            neo4j_client: Neo4j client instance
        """
        self.client = neo4j_client

    def find_entity_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Find entities by name (case-insensitive, partial match).

        Args:
            name: Entity name to search for

        Returns:
            List of matching entities
        """
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e.id as id, e.name as name, e.type as type,
               e.description as description, e.importance_score as importance
        ORDER BY e.importance_score DESC
        LIMIT 20
        """
        return self.client.execute_query(query, {'name': name})

    def get_entity_details(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific entity.

        Args:
            entity_id: Entity ID

        Returns:
            Entity details or None
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as connections
        RETURN e.id as id, e.name as name, e.type as type,
               e.description as description, e.mention_count as mentions,
               e.episode_count as episodes, e.importance_score as importance,
               e.chunks as chunks, connections
        """
        results = self.client.execute_query(query, {'entity_id': entity_id})
        return results[0] if results else None

    def get_related_entities(self, entity_name: str, max_hops: int = 1,
                            limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity.

        Args:
            entity_name: Name of the entity
            max_hops: Maximum relationship hops (1 or 2)
            limit: Maximum number of results

        Returns:
            List of related entities with relationship info
        """
        if max_hops == 1:
            query = """
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE toLower(source.name) = toLower($name)
            RETURN target.name as name, target.type as type,
                   type(r) as relationship, r.description as rel_description,
                   target.importance_score as importance
            ORDER BY importance DESC
            LIMIT $limit
            """
        else:
            query = """
            MATCH path = (source:Entity)-[*1..2]->(target:Entity)
            WHERE toLower(source.name) = toLower($name)
            AND source <> target
            WITH target, length(path) as hops, min(length(path)) as min_hops
            RETURN DISTINCT target.name as name, target.type as type,
                   min_hops as hops, target.importance_score as importance
            ORDER BY hops, importance DESC
            LIMIT $limit
            """

        return self.client.execute_query(query, {'name': entity_name, 'limit': limit})

    def find_shortest_path(self, entity1: str, entity2: str) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            Path information or None
        """
        query = """
        MATCH path = shortestPath(
            (e1:Entity)-[*]-(e2:Entity)
        )
        WHERE toLower(e1.name) = toLower($entity1)
        AND toLower(e2.name) = toLower($entity2)
        RETURN length(path) as path_length,
               [node in nodes(path) | node.name] as nodes,
               [rel in relationships(path) | type(rel)] as relationships
        LIMIT 1
        """
        results = self.client.execute_query(query, {'entity1': entity1, 'entity2': entity2})
        return results[0] if results else None

    def get_entity_neighborhood(self, entity_name: str, radius: int = 1) -> Dict[str, Any]:
        """
        Get the neighborhood subgraph around an entity.

        Args:
            entity_name: Entity name
            radius: Neighborhood radius (number of hops)

        Returns:
            Subgraph with nodes and edges
        """
        query = f"""
        MATCH path = (center:Entity)-[*1..{radius}]-(neighbor:Entity)
        WHERE toLower(center.name) = toLower($name)
        WITH collect(DISTINCT neighbor) + collect(DISTINCT center) as nodes,
             collect(DISTINCT relationships(path)) as rels
        UNWIND rels as rel_list
        UNWIND rel_list as rel
        WITH nodes,
             collect(DISTINCT {{
                 source: startNode(rel).name,
                 target: endNode(rel).name,
                 type: type(rel)
             }}) as edges
        RETURN [n in nodes | {{
                 id: n.id,
                 name: n.name,
                 type: n.type,
                 importance: n.importance_score
             }}] as nodes,
             edges
        """
        results = self.client.execute_query(query, {'name': entity_name})

        if results:
            return {
                'nodes': results[0].get('nodes', []),
                'edges': results[0].get('edges', [])
            }
        return {'nodes': [], 'edges': []}

    def find_entities_by_type(self, entity_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find all entities of a specific type.

        Args:
            entity_type: Entity type (PERSON, ORGANIZATION, CONCEPT, etc.)
            limit: Maximum number of results

        Returns:
            List of entities
        """
        query = """
        MATCH (e:Entity)
        WHERE e.type = $type
        RETURN e.name as name, e.type as type, e.description as description,
               e.importance_score as importance, e.episode_count as episodes
        ORDER BY importance DESC
        LIMIT $limit
        """
        return self.client.execute_query(query, {'type': entity_type, 'limit': limit})

    def find_entities_in_episode(self, episode_number: int) -> List[Dict[str, Any]]:
        """
        Find all entities mentioned in a specific episode.

        Args:
            episode_number: Episode number

        Returns:
            List of entities
        """
        query = """
        MATCH (e:Entity)
        WHERE $episode IN e.episodes
        RETURN e.name as name, e.type as type, e.description as description,
               e.mention_count as mentions, e.importance_score as importance
        ORDER BY importance DESC
        """
        return self.client.execute_query(query, {'episode': episode_number})

    def find_common_entities(self, episode1: int, episode2: int) -> List[Dict[str, Any]]:
        """
        Find entities that appear in both episodes.

        Args:
            episode1: First episode number
            episode2: Second episode number

        Returns:
            List of common entities
        """
        query = """
        MATCH (e:Entity)
        WHERE $ep1 IN e.episodes AND $ep2 IN e.episodes
        RETURN e.name as name, e.type as type, e.description as description,
               e.importance_score as importance
        ORDER BY importance DESC
        """
        return self.client.execute_query(query, {'ep1': episode1, 'ep2': episode2})

    def get_most_important_entities(self, limit: int = 20,
                                   entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get most important entities in the graph.

        Args:
            limit: Maximum number of results
            entity_type: Optional filter by entity type

        Returns:
            List of important entities
        """
        if entity_type:
            query = """
            MATCH (e:Entity)
            WHERE e.type = $type
            RETURN e.name as name, e.type as type, e.description as description,
                   e.importance_score as importance, e.mention_count as mentions,
                   e.episode_count as episodes
            ORDER BY importance DESC
            LIMIT $limit
            """
            return self.client.execute_query(query, {'type': entity_type, 'limit': limit})
        else:
            query = """
            MATCH (e:Entity)
            RETURN e.name as name, e.type as type, e.description as description,
                   e.importance_score as importance, e.mention_count as mentions,
                   e.episode_count as episodes
            ORDER BY importance DESC
            LIMIT $limit
            """
            return self.client.execute_query(query, {'limit': limit})

    def find_bridging_entities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find entities that connect different parts of the graph (high betweenness).

        Args:
            limit: Maximum number of results

        Returns:
            List of bridging entities
        """
        # Approximate using degree and diversity of connections
        query = """
        MATCH (e:Entity)-[r]-(other:Entity)
        WITH e, count(DISTINCT other) as connections,
             count(DISTINCT other.type) as type_diversity,
             count(DISTINCT r) as total_relations
        RETURN e.name as name, e.type as type, connections,
               type_diversity, total_relations,
               e.importance_score as importance
        ORDER BY type_diversity DESC, connections DESC
        LIMIT $limit
        """
        return self.client.execute_query(query, {'limit': limit})

    def search_by_description(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search entities by description content.

        Args:
            keyword: Keyword to search in descriptions
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.description) CONTAINS toLower($keyword)
        RETURN e.name as name, e.type as type, e.description as description,
               e.importance_score as importance
        ORDER BY importance DESC
        LIMIT $limit
        """
        return self.client.execute_query(query, {'keyword': keyword, 'limit': limit})

    def get_entity_timeline(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get episodes where entity appears, ordered chronologically.

        Args:
            entity_name: Entity name

        Returns:
            List of episodes with mention counts
        """
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($name)
        UNWIND e.episodes as episode
        RETURN episode, size([c in e.chunks WHERE c CONTAINS ('ep' + toString(episode))]) as mentions
        ORDER BY episode
        """
        return self.client.execute_query(query, {'name': entity_name})

    def find_co_occurring_entities(self, entity_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find entities that appear in same episodes as the given entity.

        Args:
            entity_name: Entity name
            limit: Maximum number of results

        Returns:
            List of co-occurring entities
        """
        query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE toLower(e1.name) = toLower($name)
        AND e1 <> e2
        WITH e1, e2, [ep IN e1.episodes WHERE ep IN e2.episodes] as common_episodes
        WHERE size(common_episodes) > 0
        RETURN e2.name as name, e2.type as type,
               size(common_episodes) as shared_episodes,
               common_episodes,
               e2.importance_score as importance
        ORDER BY shared_episodes DESC, importance DESC
        LIMIT $limit
        """
        return self.client.execute_query(query, {'name': entity_name, 'limit': limit})

    def get_relationship_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find common relationship patterns in the graph.

        Args:
            limit: Maximum number of results

        Returns:
            List of relationship patterns with counts
        """
        query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        WITH source.type as source_type, type(r) as rel_type,
             target.type as target_type, count(*) as count
        RETURN source_type, rel_type, target_type, count
        ORDER BY count DESC
        LIMIT $limit
        """
        return self.client.execute_query(query, {'limit': limit})
