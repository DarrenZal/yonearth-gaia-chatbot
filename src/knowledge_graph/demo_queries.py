"""
Demo script showcasing knowledge graph query capabilities.
"""

import json
from src.knowledge_graph.graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph.graph_queries import GraphQueries


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_entities(entities, title="Results"):
    """Print a list of entities."""
    print(f"\n{title}:")
    for i, entity in enumerate(entities[:10], 1):
        name = entity.get('name', 'N/A')
        entity_type = entity.get('type', 'N/A')
        importance = entity.get('importance', entity.get('importance_score', 0))
        print(f"  {i}. {name} ({entity_type}) - Importance: {importance:.2f}")


def main():
    """Run demo queries."""
    print_section("YONEARTH KNOWLEDGE GRAPH DEMO")
    print("\nConnecting to Neo4j...")

    with Neo4jClient() as client:
        queries = GraphQueries(client)

        # Demo 1: Search for entities
        print_section("DEMO 1: Search for 'permaculture' entities")
        results = queries.find_entity_by_name("permaculture")
        print_entities(results, f"Found {len(results)} entities")

        # Demo 2: Find entities in an episode
        print_section("DEMO 2: Entities in Episode 44")
        results = queries.find_entities_in_episode(44)
        print_entities(results, f"Found {len(results)} entities in Episode 44")

        # Demo 3: Most important concepts
        print_section("DEMO 3: Most Important CONCEPT Entities")
        results = queries.get_most_important_entities(10, entity_type="CONCEPT")
        print_entities(results, "Top 10 Concepts")

        # Demo 4: Most important people
        print_section("DEMO 4: Most Important PERSON Entities")
        results = queries.get_most_important_entities(10, entity_type="PERSON")
        print_entities(results, "Top 10 People")

        # Demo 5: Related entities
        print_section("DEMO 5: Entities Related to 'Regenerative Agriculture'")
        results = queries.get_related_entities("regenerative agriculture", max_hops=1, limit=10)
        print(f"\nFound {len(results)} related entities:")
        for i, rel in enumerate(results[:10], 1):
            print(f"  {i}. {rel['name']} ({rel['type']}) via {rel['relationship']}")

        # Demo 6: Co-occurring entities
        print_section("DEMO 6: Entities Co-occurring with 'Sustainability'")
        results = queries.find_co_occurring_entities("Sustainability", limit=10)
        print(f"\nFound {len(results)} co-occurring entities:")
        for i, entity in enumerate(results[:10], 1):
            shared = entity.get('shared_episodes', 0)
            print(f"  {i}. {entity['name']} ({entity['type']}) - {shared} shared episodes")

        # Demo 7: Bridging entities
        print_section("DEMO 7: Bridging Entities (Connecting Different Topics)")
        results = queries.find_bridging_entities(10)
        print(f"\nTop 10 bridging entities:")
        for i, entity in enumerate(results, 1):
            name = entity.get('name', 'N/A')
            entity_type = entity.get('type', 'N/A')
            connections = entity.get('connections', 0)
            diversity = entity.get('type_diversity', 0)
            print(f"  {i}. {name} ({entity_type}) - {connections} connections, {diversity} entity types")

        # Demo 8: Shortest path
        print_section("DEMO 8: Shortest Path Between Entities")
        result = queries.find_shortest_path("Permaculture", "Soil")
        if result:
            print(f"\nPath from 'Permaculture' to 'Soil':")
            print(f"  Path length: {result['path_length']} hops")
            print(f"  Nodes: {' → '.join(result['nodes'])}")
            print(f"  Relationships: {' → '.join(result['relationships'])}")
        else:
            print("\nNo path found between these entities.")

        # Demo 9: Entity neighborhood
        print_section("DEMO 9: Neighborhood of 'Climate Change'")
        result = queries.get_entity_neighborhood("climate change", radius=1)
        print(f"\nNeighborhood subgraph:")
        print(f"  Nodes: {len(result['nodes'])}")
        print(f"  Edges: {len(result['edges'])}")
        if result['nodes']:
            print("\n  Sample nodes:")
            for node in result['nodes'][:5]:
                print(f"    - {node['name']} ({node['type']})")

        # Demo 10: Relationship patterns
        print_section("DEMO 10: Common Relationship Patterns")
        results = queries.get_relationship_patterns(15)
        print(f"\nTop relationship patterns:")
        for i, pattern in enumerate(results, 1):
            print(f"  {i}. {pattern['source_type']} -{pattern['rel_type']}-> {pattern['target_type']} ({pattern['count']}x)")

        # Demo 11: Entity details
        print_section("DEMO 11: Detailed Entity Information")
        results = queries.find_entity_by_name("regenerative agriculture")
        if results:
            entity_id = results[0]['id']
            details = queries.get_entity_details(entity_id)
            if details:
                print(f"\nEntity: {details['name']} ({details['type']})")
                print(f"Description: {details['description'][:200]}...")
                print(f"Mentions: {details['mentions']}")
                print(f"Episodes: {details['episodes']}")
                print(f"Importance: {details['importance']:.2f}")
                print(f"Connections: {details['connections']}")

        # Summary statistics
        print_section("KNOWLEDGE GRAPH SUMMARY")
        node_count = client.get_node_count()
        rel_count = client.get_relationship_count()
        entity_dist = client.get_entity_type_distribution()
        rel_dist = client.get_relationship_type_distribution()

        print(f"\nGraph Statistics:")
        print(f"  Total Entities: {node_count}")
        print(f"  Total Relationships: {rel_count}")
        print(f"  Entity Types: {len(entity_dist)}")
        print(f"  Relationship Types: {len(rel_dist)}")

        print(f"\nTop Entity Types:")
        for entity_type, count in list(entity_dist.items())[:10]:
            print(f"  - {entity_type}: {count}")

        print(f"\nTop Relationship Types:")
        for rel_type, count in list(rel_dist.items())[:10]:
            print(f"  - {rel_type}: {count}")

        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        print("\nNeo4j Browser: http://localhost:7474")
        print("Username: neo4j")
        print("Password: yonearth2024")
        print("\n")


if __name__ == "__main__":
    main()
