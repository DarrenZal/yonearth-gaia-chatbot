#!/usr/bin/env python3
"""
Build unified knowledge graph from processed entities and relationships.

Outputs:
- unified_v2.json: Full graph with entities and relationships
- graph_stats_v2.json: Statistics about the graph

Usage:
    python scripts/build_unified_graph_v2.py
    python scripts/build_unified_graph_v2.py --dry-run
    python scripts/build_unified_graph_v2.py --stats-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import settings


class UnifiedGraphBuilder:
    """Builds the unified knowledge graph."""

    def __init__(
        self,
        entities_path: Path = None,
        relationships_path: Path = None,
        output_dir: Path = None
    ):
        self.entities_path = entities_path or Path("data/knowledge_graph_unified/entities_deduplicated.json")
        self.relationships_path = relationships_path or Path("data/knowledge_graph_unified/relationships_processed.json")
        self.output_dir = output_dir or Path("data/knowledge_graph_unified")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_entities(self) -> Tuple[Dict[str, Dict], Set[str]]:
        """
        Load entities into a dict keyed by normalized name.

        Returns:
            Tuple of (entities_dict, valid_entity_names_set)
        """
        if not self.entities_path.exists():
            raise FileNotFoundError(f"Entities file not found: {self.entities_path}")

        with open(self.entities_path) as f:
            entities_list = json.load(f)

        # Build dict keyed by name (prefer higher mention count if duplicates)
        entities = {}
        valid_names = set()

        for entity in entities_list:
            name = entity.get("name", "")
            if not name:
                continue

            name_lower = name.lower()
            valid_names.add(name_lower)

            # If we already have this entity, keep the one with higher mention count
            if name in entities:
                existing = entities[name]
                if entity.get("mention_count", 1) > existing.get("mention_count", 1):
                    entities[name] = entity
            else:
                entities[name] = entity

        return entities, valid_names

    def load_relationships(self) -> List[Dict]:
        """Load relationships from JSON file."""
        if not self.relationships_path.exists():
            raise FileNotFoundError(f"Relationships file not found: {self.relationships_path}")

        with open(self.relationships_path) as f:
            return json.load(f)

    def deduplicate_relationships(
        self,
        relationships: List[Dict],
        valid_entities: Set[str]
    ) -> List[Dict]:
        """
        Deduplicate relationships and remove orphans.

        Groups by (source, predicate, target) and merges duplicates.
        """
        # Group by (source, predicate, target)
        grouped: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
        orphaned = 0

        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            predicate = rel.get("predicate", "RELATES_TO")

            # Skip if entities don't exist (case-insensitive check)
            if source.lower() not in valid_entities or target.lower() not in valid_entities:
                orphaned += 1
                continue

            key = (source, predicate, target)
            grouped[key].append(rel)

        print(f"  Removed {orphaned} orphaned relationships")

        deduplicated = []
        for (source, predicate, target), group in grouped.items():
            # Merge multiple occurrences of the same relationship
            all_sources = set()
            for rel in group:
                source_key = f"{rel.get('source_type', 'unknown')}_{rel.get('source_id', 'unknown')}"
                all_sources.add(source_key)

            merged = {
                "source": source,
                "predicate": predicate,
                "target": target,
                "mention_count": len(group),
                "sources": sorted(list(all_sources))
            }
            deduplicated.append(merged)

        return deduplicated

    def build(self, dry_run: bool = False) -> Dict:
        """Build the unified graph."""
        print("Loading entities...")
        entities, valid_names = self.load_entities()
        print(f"  Loaded {len(entities)} unique entities")

        print("\nLoading relationships...")
        relationships = self.load_relationships()
        print(f"  Loaded {len(relationships)} relationships")

        print("\nDeduplicating relationships...")
        relationships_dedup = self.deduplicate_relationships(relationships, valid_names)
        print(f"  Deduplicated to {len(relationships_dedup)} unique relationships")

        # Build graph structure
        extraction_model = getattr(settings, "graph_extraction_model", "unknown")
        extraction_mode = getattr(settings, "graph_extraction_mode", "unknown")
        graph = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "extraction_model": extraction_model,
                "extraction_method": extraction_mode,
                "pipeline_version": "2.0",
                "entity_count": len(entities),
                "relationship_count": len(relationships_dedup)
            },
            "entities": entities,
            "relationships": relationships_dedup
        }

        # Generate stats
        stats = self._generate_stats(entities, relationships_dedup)

        if not dry_run:
            # Save graph
            output_path = self.output_dir / "unified_v2.json"
            with open(output_path, 'w') as f:
                json.dump(graph, f, indent=2)
            print(f"\nSaved unified graph to: {output_path}")

            # Save stats
            stats_path = self.output_dir / "graph_stats_v2.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved stats to: {stats_path}")

        self._print_stats(stats)

        return graph

    def _generate_stats(self, entities: Dict[str, Dict], relationships: List[Dict]) -> Dict:
        """Generate graph statistics."""
        # Entity type distribution
        type_counts = defaultdict(int)
        for entity in entities.values():
            type_counts[entity.get("type", "UNKNOWN")] += 1

        # Relationship predicate distribution
        predicate_counts = defaultdict(int)
        for rel in relationships:
            predicate_counts[rel.get("predicate", "UNKNOWN")] += 1

        # Top entities by mention count
        sorted_entities = sorted(
            entities.values(),
            key=lambda x: x.get("mention_count", 1),
            reverse=True
        )
        top_entities = [
            {
                "name": e.get("name"),
                "type": e.get("type"),
                "mentions": e.get("mention_count", 1),
                "is_fictional": e.get("is_fictional", False)
            }
            for e in sorted_entities[:30]
        ]

        # Fictional entity count
        fictional_count = sum(1 for e in entities.values() if e.get("is_fictional", False))

        # Entity connectivity (how many relationships each entity has)
        entity_connections = defaultdict(int)
        for rel in relationships:
            entity_connections[rel.get("source", "")] += 1
            entity_connections[rel.get("target", "")] += 1

        # Most connected entities
        sorted_connections = sorted(
            entity_connections.items(),
            key=lambda x: -x[1]
        )
        most_connected = [
            {
                "name": name,
                "connections": count,
                "type": entities.get(name, {}).get("type", "UNKNOWN")
            }
            for name, count in sorted_connections[:20]
        ]

        # Source distribution
        source_counts = defaultdict(int)
        for entity in entities.values():
            sources = entity.get("sources", [])
            for source in sources:
                source_type = source.split("_")[0] if "_" in source else "unknown"
                source_counts[source_type] += 1

        return {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "fictional_entities": fictional_count,
            "entity_types": dict(type_counts),
            "relationship_predicates": dict(predicate_counts),
            "top_entities_by_mentions": top_entities,
            "most_connected_entities": most_connected,
            "source_distribution": dict(source_counts),
            "generated_at": datetime.now().isoformat()
        }

    def _print_stats(self, stats: Dict):
        """Print statistics summary."""
        print("\n" + "=" * 60)
        print("UNIFIED GRAPH STATISTICS")
        print("=" * 60)

        print(f"\nTotal entities: {stats['total_entities']}")
        print(f"Total relationships: {stats['total_relationships']}")
        print(f"Fictional entities: {stats['fictional_entities']}")

        print(f"\nEntity types:")
        for etype, count in sorted(stats["entity_types"].items(), key=lambda x: -x[1]):
            print(f"  {etype}: {count}")

        print(f"\nTop 10 relationship predicates:")
        sorted_predicates = sorted(
            stats["relationship_predicates"].items(),
            key=lambda x: -x[1]
        )
        for pred, count in sorted_predicates[:10]:
            print(f"  {pred}: {count}")

        print(f"\nTop 10 entities by mention count:")
        for entity in stats["top_entities_by_mentions"][:10]:
            fictional_marker = " [FICTIONAL]" if entity.get("is_fictional") else ""
            print(f"  {entity['name']} ({entity['type']}): {entity['mentions']} mentions{fictional_marker}")

        print(f"\nTop 10 most connected entities:")
        for entity in stats["most_connected_entities"][:10]:
            print(f"  {entity['name']} ({entity['type']}): {entity['connections']} connections")

        if stats.get("source_distribution"):
            print(f"\nSource distribution:")
            for source_type, count in sorted(stats["source_distribution"].items(), key=lambda x: -x[1]):
                print(f"  {source_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Build unified knowledge graph")
    parser.add_argument("--entities-path", type=str, help="Path to entities_deduplicated.json")
    parser.add_argument("--relationships-path", type=str, help="Path to relationships_processed.json")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Process without saving")
    parser.add_argument("--stats-only", action="store_true", help="Show stats from previous run")

    args = parser.parse_args()

    if args.stats_only:
        stats_path = Path("data/knowledge_graph_unified/graph_stats_v2.json")
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            print(json.dumps(stats, indent=2))
        else:
            print("No stats file found. Run graph building first.")
        return

    builder = UnifiedGraphBuilder(
        entities_path=Path(args.entities_path) if args.entities_path else None,
        relationships_path=Path(args.relationships_path) if args.relationships_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    try:
        builder.build(dry_run=args.dry_run)

        if args.dry_run:
            print("\n[DRY RUN - No files saved]")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have run the deduplication step first:")
        print("  python scripts/deduplicate_entities.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
