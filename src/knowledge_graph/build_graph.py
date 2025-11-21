"""
Main script to build the unified knowledge graph from all extractions.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from src.knowledge_graph.graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph.graph_builder import GraphBuilder
from src.knowledge_graph.graph.graph_queries import GraphQueries

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to build the knowledge graph."""
    logger.info("=" * 80)
    logger.info("YONEARTH KNOWLEDGE GRAPH BUILDER - AGENT 8")
    logger.info("=" * 80)

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    extraction_dir = base_dir / "data" / "knowledge_graph" / "entities"
    output_dir = base_dir / "data" / "knowledge_graph" / "graph"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Statistics collection
    all_stats = {
        'timestamp': datetime.now().isoformat(),
        'agent': 'Agent 8 - Graph Builder',
        'steps': {}
    }

    try:
        # Step 1: Connect to Neo4j
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Connecting to Neo4j Database")
        logger.info("=" * 80)

        with Neo4jClient() as client:
            logger.info("Successfully connected to Neo4j at bolt://localhost:7687")

            # Step 2: Initialize Graph Builder
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: Loading Extraction Files")
            logger.info("=" * 80)

            builder = GraphBuilder(str(extraction_dir), client)
            load_stats = builder.load_extractions()
            all_stats['steps']['1_load_extractions'] = load_stats

            logger.info(f"\nLoaded Statistics:")
            logger.info(f"  - Files processed: {load_stats['files_loaded']}")
            logger.info(f"  - Raw entities: {load_stats['total_entities_raw']}")
            logger.info(f"  - Raw relationships: {load_stats['total_relationships_raw']}")
            logger.info(f"  - Unique entities (before dedup): {load_stats['unique_entities_before_dedup']}")

            # Step 3: Deduplicate Entities
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Deduplicating Entities (Fuzzy Matching)")
            logger.info("=" * 80)

            dedup_stats = builder.deduplicate_entities()
            all_stats['steps']['2_deduplicate_entities'] = dedup_stats

            logger.info(f"\nDeduplication Results:")
            logger.info(f"  - Unique entities (after dedup): {dedup_stats['entities_after_dedup']}")
            logger.info(f"  - Entities merged: {dedup_stats['entities_merged']}")
            reduction = (dedup_stats['entities_merged'] / load_stats['total_entities_raw']) * 100
            logger.info(f"  - Reduction: {reduction:.1f}%")

            # Step 4: Deduplicate Relationships
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Deduplicating Relationships")
            logger.info("=" * 80)

            rel_stats = builder.deduplicate_relationships()
            all_stats['steps']['3_deduplicate_relationships'] = rel_stats

            logger.info(f"\nRelationship Deduplication:")
            logger.info(f"  - Unique relationships: {rel_stats['unique_relationships']}")

            # Step 5: Calculate Entity Importance
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: Calculating Entity Importance Scores")
            logger.info("=" * 80)

            builder.calculate_entity_importance()
            logger.info("Entity importance scores calculated")

            # Step 6: Populate Neo4j
            logger.info("\n" + "=" * 80)
            logger.info("STEP 6: Populating Neo4j Database")
            logger.info("=" * 80)

            pop_stats = builder.populate_neo4j()
            all_stats['steps']['4_populate_neo4j'] = pop_stats

            logger.info(f"\nDatabase Population:")
            logger.info(f"  - Entities inserted: {pop_stats['entities_inserted']}")
            logger.info(f"  - Relationships inserted: {pop_stats['relationships_inserted']}")

            # Step 7: Generate Statistics
            logger.info("\n" + "=" * 80)
            logger.info("STEP 7: Generating Graph Statistics")
            logger.info("=" * 80)

            graph_stats = builder.get_statistics()
            all_stats['steps']['5_graph_statistics'] = graph_stats

            logger.info(f"\nGraph Statistics:")
            logger.info(f"  - Total entities: {graph_stats['total_entities']}")
            logger.info(f"  - Total relationships: {graph_stats['total_relationships']}")

            logger.info(f"\nEntity Type Distribution:")
            for entity_type, count in list(graph_stats['entity_type_distribution'].items())[:10]:
                logger.info(f"  - {entity_type}: {count}")

            logger.info(f"\nRelationship Type Distribution:")
            for rel_type, count in list(graph_stats['relationship_type_distribution'].items())[:10]:
                logger.info(f"  - {rel_type}: {count}")

            logger.info(f"\nTop Entities by Importance:")
            for i, entity in enumerate(graph_stats['top_entities_by_importance'], 1):
                logger.info(f"  {i}. {entity['name']} ({entity['type']}) - Score: {entity['importance_score']:.2f}")

            # Step 8: Neo4j Statistics
            logger.info("\n" + "=" * 80)
            logger.info("STEP 8: Neo4j Database Statistics")
            logger.info("=" * 80)

            node_count = client.get_node_count()
            rel_count = client.get_relationship_count()
            entity_dist = client.get_entity_type_distribution()
            rel_dist = client.get_relationship_type_distribution()
            most_connected = client.get_most_connected_entities(10)

            neo4j_stats = {
                'node_count': node_count,
                'relationship_count': rel_count,
                'entity_type_distribution': entity_dist,
                'relationship_type_distribution': rel_dist,
                'most_connected_entities': most_connected
            }
            all_stats['steps']['6_neo4j_statistics'] = neo4j_stats

            logger.info(f"\nNeo4j Database:")
            logger.info(f"  - Total nodes: {node_count}")
            logger.info(f"  - Total relationships: {rel_count}")

            logger.info(f"\nMost Connected Entities:")
            for entity in most_connected:
                logger.info(f"  - {entity['name']} ({entity['type']}): {entity['degree']} connections")

            # Step 9: Test Queries
            logger.info("\n" + "=" * 80)
            logger.info("STEP 9: Testing Graph Queries")
            logger.info("=" * 80)

            queries = GraphQueries(client)

            # Test query: Find entities related to "permaculture"
            logger.info("\nTest Query 1: Find entities containing 'permaculture'")
            results = queries.find_entity_by_name("permaculture")
            for result in results[:5]:
                logger.info(f"  - {result['name']} ({result['type']})")

            # Test query: Find entities in Episode 44
            logger.info("\nTest Query 2: Find entities in Episode 44")
            results = queries.find_entities_in_episode(44)
            logger.info(f"  Found {len(results)} entities in Episode 44")
            for result in results[:5]:
                logger.info(f"  - {result['name']} ({result['type']})")

            # Test query: Most important concepts
            logger.info("\nTest Query 3: Most important CONCEPT entities")
            results = queries.get_most_important_entities(5, entity_type="CONCEPT")
            for result in results:
                logger.info(f"  - {result['name']}: {result['importance']:.2f}")

            # Save statistics
            stats_file = output_dir / "graph_build_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(all_stats, f, indent=2)
            logger.info(f"\nStatistics saved to: {stats_file}")

            # Generate summary report
            logger.info("\n" + "=" * 80)
            logger.info("FINAL SUMMARY")
            logger.info("=" * 80)

            logger.info(f"\nKnowledge Graph Successfully Built!")
            logger.info(f"  - Total Unique Entities: {graph_stats['total_entities']}")
            logger.info(f"  - Total Relationships: {graph_stats['total_relationships']}")
            logger.info(f"  - Entity Types: {len(graph_stats['entity_type_distribution'])}")
            logger.info(f"  - Relationship Types: {len(graph_stats['relationship_type_distribution'])}")
            logger.info(f"  - Neo4j Nodes: {node_count}")
            logger.info(f"  - Neo4j Edges: {rel_count}")
            logger.info(f"\nNeo4j Browser: http://localhost:7474")
            logger.info(f"  Username: neo4j")
            logger.info(f"  Password: yonearth2024")

            return all_stats

    except Exception as e:
        logger.error(f"\nError building knowledge graph: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
