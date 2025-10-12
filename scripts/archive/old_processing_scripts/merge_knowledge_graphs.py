#!/usr/bin/env python3
"""
Knowledge Graph Merger

Merges multiple knowledge graph extractions (books + episodes) into a single
unified knowledge graph with:
- Entity aliasing and deduplication
- Relationship deduplication by claim_uid
- Cross-source relationship aggregation
- Quality metrics and statistics

Usage:
    python3 scripts/merge_knowledge_graphs.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Set, Any

# Add data/processed to path for entity resolver
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "processed"))
from entity_resolver import EntityAliasResolver

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
BOOKS_KG_DIR = DATA_DIR / "knowledge_graph_books_v3_2_2"
EPISODES_KG_DIR = DATA_DIR / "knowledge_graph_v3_2_2"
OUTPUT_DIR = DATA_DIR / "knowledge_graph_unified"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
UNIFIED_KG_FILE = OUTPUT_DIR / f"unified_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
STATS_FILE = OUTPUT_DIR / f"merge_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


class KnowledgeGraphMerger:
    """Merges multiple KG extractions into unified graph"""

    def __init__(self, use_aliasing=True):
        self.use_aliasing = use_aliasing
        self.alias_resolver = EntityAliasResolver() if use_aliasing else None

        # Merged data structures
        self.relationships = {}  # claim_uid -> relationship
        self.entities = set()
        self.sources = []  # List of source files

        # Statistics
        self.stats = {
            'total_sources': 0,
            'books_processed': 0,
            'episodes_processed': 0,
            'total_relationships_before_merge': 0,
            'total_relationships_after_merge': 0,
            'duplicate_relationships_removed': 0,
            'entities_before_aliasing': 0,
            'entities_after_aliasing': 0,
            'aliases_applied': 0,
            'conflict_count': 0,
            'high_confidence_count': 0,
            'medium_confidence_count': 0,
            'low_confidence_count': 0,
        }

    def resolve_entity(self, entity: str) -> str:
        """Resolve entity to canonical form if aliasing enabled"""
        if self.alias_resolver:
            canonical = self.alias_resolver.resolve(entity)
            if canonical != entity:
                self.stats['aliases_applied'] += 1
            return canonical
        return entity

    def load_kg_file(self, kg_file: Path) -> Dict:
        """Load a knowledge graph extraction file"""
        with open(kg_file, 'r') as f:
            return json.load(f)

    def merge_relationship(self, rel: Dict, source_file: str, source_type: str):
        """Merge a single relationship into unified graph"""

        # Track pre-merge count
        self.stats['total_relationships_before_merge'] += 1

        # Resolve entities through aliasing
        rel['source'] = self.resolve_entity(rel['source'])
        rel['target'] = self.resolve_entity(rel['target'])

        # Track entities
        self.entities.add(rel['source'])
        self.entities.add(rel['target'])

        # Get claim_uid (unique identifier for this fact)
        claim_uid = rel.get('claim_uid')
        if not claim_uid:
            # Generate if missing (shouldn't happen in v3.2.2)
            import hashlib
            components = [rel['source'], rel['relationship'], rel['target']]
            claim_uid = hashlib.sha1("|".join(components).encode()).hexdigest()

        # Check if we've seen this relationship before
        if claim_uid in self.relationships:
            # Duplicate - track it but keep the one with higher confidence
            existing = self.relationships[claim_uid]
            if rel.get('p_true', 0) > existing.get('p_true', 0):
                # Replace with higher confidence version
                self.relationships[claim_uid] = rel
                self.relationships[claim_uid]['source_files'] = existing.get('source_files', []) + [source_file]
            else:
                # Keep existing, just add source
                self.relationships[claim_uid]['source_files'] = existing.get('source_files', []) + [source_file]

            self.stats['duplicate_relationships_removed'] += 1
        else:
            # New relationship
            rel['source_files'] = [source_file]
            rel['source_type'] = source_type  # 'book' or 'episode'
            self.relationships[claim_uid] = rel

        # Track quality stats
        p_true = rel.get('p_true', 0)
        if p_true >= 0.75:
            self.stats['high_confidence_count'] += 1
        elif p_true >= 0.5:
            self.stats['medium_confidence_count'] += 1
        else:
            self.stats['low_confidence_count'] += 1

        if rel.get('signals_conflict'):
            self.stats['conflict_count'] += 1

    def merge_kg_files(self, kg_files: List[Path], source_type: str):
        """Merge multiple KG files of the same type"""
        for kg_file in kg_files:
            if kg_file.name.startswith('checkpoint') or kg_file.name.startswith('progress') or kg_file.name.startswith('summary'):
                continue

            print(f"  📖 Merging: {kg_file.name}")
            data = self.load_kg_file(kg_file)

            # Track source
            self.sources.append({
                'file': kg_file.name,
                'type': source_type,
                'relationships': len(data.get('relationships', [])),
                'doc_id': data.get('book_title') or data.get('episode_number', 'unknown')
            })

            # Merge all relationships
            for rel in data.get('relationships', []):
                self.merge_relationship(rel, kg_file.name, source_type)

            if source_type == 'book':
                self.stats['books_processed'] += 1
            else:
                self.stats['episodes_processed'] += 1

    def build_unified_graph(self) -> Dict:
        """Build final unified knowledge graph"""

        # Calculate final stats
        self.stats['total_sources'] = len(self.sources)
        self.stats['total_relationships_after_merge'] = len(self.relationships)
        self.stats['entities_after_aliasing'] = len(self.entities)

        # Build relationship type distribution
        rel_types = Counter(r['relationship'] for r in self.relationships.values())

        # Build source type distribution
        source_types = Counter(r['source_type'] for r in self.relationships.values())

        # Top entities by degree
        entity_degrees = defaultdict(int)
        for rel in self.relationships.values():
            entity_degrees[rel['source']] += 1
            entity_degrees[rel['target']] += 1

        top_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)[:20]

        # Build unified graph structure
        unified_graph = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': 'unified_v1.0',
                'merger_version': 'v3.2.2',
                'aliasing_enabled': self.use_aliasing,
                'sources': self.sources
            },
            'statistics': {
                **self.stats,
                'relationship_types': dict(rel_types.most_common(20)),
                'source_type_distribution': dict(source_types),
                'top_entities_by_degree': [{'entity': e, 'degree': d} for e, d in top_entities],
                'deduplication_rate': f"{(self.stats['duplicate_relationships_removed'] / self.stats['total_relationships_before_merge'] * 100):.1f}%" if self.stats['total_relationships_before_merge'] > 0 else "0%",
                'compression_ratio': f"{self.stats['total_relationships_before_merge'] / self.stats['total_relationships_after_merge']:.2f}x" if self.stats['total_relationships_after_merge'] > 0 else "N/A"
            },
            'entities': sorted(list(self.entities)),
            'relationships': list(self.relationships.values())
        }

        return unified_graph

    def save_unified_graph(self, unified_graph: Dict, output_file: Path):
        """Save unified graph to JSON"""
        with open(output_file, 'w') as f:
            json.dump(unified_graph, f, indent=2)

    def save_statistics(self, stats: Dict, stats_file: Path):
        """Save detailed statistics"""
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    print("=" * 80)
    print("🔗 KNOWLEDGE GRAPH MERGER v3.2.2")
    print("=" * 80)
    print()

    # Initialize merger
    print("🔧 Initializing merger with entity aliasing...")
    merger = KnowledgeGraphMerger(use_aliasing=True)
    print()

    # Count entities before merge
    print("📊 Collecting source files...")

    # Books
    book_files = list(BOOKS_KG_DIR.glob("*.json"))
    book_files = [f for f in book_files if not f.name.startswith('checkpoint')]
    print(f"  📚 Found {len(book_files)} book extractions")

    # Episodes
    episode_files = list(EPISODES_KG_DIR.glob("*.json"))
    episode_files = [f for f in episode_files if not any(f.name.startswith(p) for p in ['checkpoint', 'progress', 'summary'])]
    print(f"  🎙️  Found {len(episode_files)} episode extractions")
    print()

    # Merge books
    print("📖 Merging book knowledge graphs...")
    merger.merge_kg_files(book_files, 'book')
    print()

    # Merge episodes
    print("🎙️  Merging episode knowledge graphs...")
    merger.merge_kg_files(episode_files, 'episode')
    print()

    # Build unified graph
    print("🔗 Building unified knowledge graph...")
    unified_graph = merger.build_unified_graph()
    print()

    # Save
    print(f"💾 Saving unified graph to: {UNIFIED_KG_FILE}")
    merger.save_unified_graph(unified_graph, UNIFIED_KG_FILE)

    print(f"💾 Saving statistics to: {STATS_FILE}")
    merger.save_statistics(unified_graph['statistics'], STATS_FILE)
    print()

    # Display statistics
    print("=" * 80)
    print("✅ MERGE COMPLETE")
    print("=" * 80)
    stats = unified_graph['statistics']
    print(f"Sources:")
    print(f"  Books: {stats['books_processed']}")
    print(f"  Episodes: {stats['episodes_processed']}")
    print(f"  Total: {stats['total_sources']}")
    print()
    print(f"Relationships:")
    print(f"  Before merge: {stats['total_relationships_before_merge']:,}")
    print(f"  After merge: {stats['total_relationships_after_merge']:,}")
    print(f"  Duplicates removed: {stats['duplicate_relationships_removed']:,}")
    print(f"  Compression ratio: {stats['compression_ratio']}")
    print()
    print(f"Entities:")
    print(f"  Unique entities: {stats['entities_after_aliasing']:,}")
    print(f"  Aliases applied: {stats['aliases_applied']:,}")
    print()
    print(f"Quality:")
    print(f"  High confidence (p≥0.75): {stats['high_confidence_count']:,}")
    print(f"  Medium confidence: {stats['medium_confidence_count']:,}")
    print(f"  Low confidence: {stats['low_confidence_count']:,}")
    print(f"  Conflicts: {stats['conflict_count']:,}")
    print()
    print(f"Top 5 Relationship Types:")
    for rel_type, count in list(stats['relationship_types'].items())[:5]:
        print(f"  {count:4d}x  {rel_type}")
    print()
    print(f"Top 5 Entities by Degree:")
    for item in stats['top_entities_by_degree'][:5]:
        print(f"  {item['degree']:4d}x  {item['entity']}")
    print()
    print(f"📁 Output files:")
    print(f"  Unified graph: {UNIFIED_KG_FILE}")
    print(f"  Statistics: {STATS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
