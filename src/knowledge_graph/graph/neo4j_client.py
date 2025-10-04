"""
Neo4j client for managing graph database connections and operations.
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for interacting with Neo4j graph database."""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "yonearth2024"):
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results
        """
        if not self.driver:
            raise RuntimeError("Not connected to database. Call connect() first.")

        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Execute a write query in a transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            Query results
        """
        if not self.driver:
            raise RuntimeError("Not connected to database. Call connect() first.")

        def _execute(tx):
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]

        with self.driver.session() as session:
            return session.execute_write(_execute)

    def batch_execute(self, queries: List[tuple]):
        """
        Execute multiple queries in a single transaction.

        Args:
            queries: List of (query, parameters) tuples

        Returns:
            Number of queries executed
        """
        if not self.driver:
            raise RuntimeError("Not connected to database. Call connect() first.")

        def _execute_batch(tx):
            count = 0
            for query, params in queries:
                tx.run(query, params or {})
                count += 1
            return count

        with self.driver.session() as session:
            return session.execute_write(_execute_batch)

    def clear_database(self):
        """Delete all nodes and relationships from the database."""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write(query)
        logger.info("Cleared all data from Neo4j database")

    def create_indexes(self):
        """Create indexes for efficient querying."""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
        ]

        for index_query in indexes:
            try:
                self.execute_write(index_query)
                logger.info(f"Created index: {index_query}")
            except Exception as e:
                logger.warning(f"Index creation skipped or failed: {e}")

    def get_node_count(self) -> int:
        """Get total number of nodes in the database."""
        result = self.execute_query("MATCH (n) RETURN count(n) as count")
        return result[0]['count'] if result else 0

    def get_relationship_count(self) -> int:
        """Get total number of relationships in the database."""
        result = self.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
        return result[0]['count'] if result else 0

    def get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types."""
        query = """
        MATCH (e:Entity)
        RETURN e.type as type, count(e) as count
        ORDER BY count DESC
        """
        results = self.execute_query(query)
        return {r['type']: r['count'] for r in results}

    def get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        results = self.execute_query(query)
        return {r['rel_type']: r['count'] for r in results}

    def get_most_connected_entities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get entities with most connections."""
        query = f"""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as degree
        RETURN e.name as name, e.type as type, degree
        ORDER BY degree DESC
        LIMIT {limit}
        """
        return self.execute_query(query)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
