#!/usr/bin/env python3
"""
Merge Book and Episode Knowledge Graphs into a Unified Knowledge Graph

This script merges:
1. Book KG from kg_unified_discourse/outputs/book_extractions/
2. Episode KG from data/knowledge_graph_v3_2_2/

Output:
- data/knowledge_graph_unified/unified.json (entities + relationships)
- data/knowledge_graph_unified/adjacency.json (graph edges)
- data/knowledge_graph_unified/stats.json (counts, orphans, top predicates)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import hashlib
from datetime import datetime
import subprocess
import yaml

# Paths
PROJECT_ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
# Defaults (override via CLI args in main())
BOOK_KG_PATH = PROJECT_ROOT / "kg_unified_discourse/outputs/book_extractions/all_books_complete.json"
EPISODE_KG_DIR = PROJECT_ROOT / "data/knowledge_graph_v3_2_2"
OUTPUT_DIR = PROJECT_ROOT / "data/knowledge_graph_unified"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class KnowledgeGraphMerger:
    def __init__(self):
        self.entities = {}
        self.relationships = []
        self.adjacency = defaultdict(lambda: defaultdict(list))
        self.stats = {
            "entity_count": 0,
            "relationship_count": 0,
            "orphan_entities": [],
            "orphan_edges": [],
            "top_predicates": {},
            "top_entity_types": {},
            "sources": {
                "books": [],
                "episodes": []
            },
            "merge_timestamp": datetime.now().isoformat(),
            "ontology_version": "1.0.0",
            "build_id": self._get_build_id(),
            "type_remappings": {},
            "predicate_remappings": {}
        }
        self.load_ontology()

    def _get_build_id(self) -> str:
        """Generate build ID from timestamp and git SHA"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                             cwd=PROJECT_ROOT).decode().strip()
            return f"{timestamp}_{git_sha}"
        except:
            return timestamp

    def load_ontology(self):
        """Load ontology for type and predicate validation"""
        ontology_path = PROJECT_ROOT / "data/knowledge_graph/ontology.yaml"

        if ontology_path.exists():
            with open(ontology_path, 'r') as f:
                self.ontology = yaml.safe_load(f)

            # Extract canonical types
            self.canonical_types = set()
            for type_info in self.ontology.get('domain', {}).get('types', []):
                self.canonical_types.add(type_info['name'])

            # Extract canonical predicates
            self.canonical_predicates = set()
            for pred_info in self.ontology.get('domain', {}).get('predicates', []):
                self.canonical_predicates.add(pred_info['name'])

            # Load type synonyms for normalization
            self.type_synonyms = self.ontology.get('normalization', {}).get('type_synonyms', {})

            # Load predicate synonyms
            self.predicate_synonyms = self.ontology.get('normalization', {}).get('predicate_synonyms', {})

            print(f"Loaded ontology v{self.ontology.get('version', 'unknown')} with {len(self.canonical_types)} types and {len(self.canonical_predicates)} predicates")
        else:
            print("Warning: Ontology file not found. Proceeding without validation.")
            self.ontology = None
            self.canonical_types = set()
            self.canonical_predicates = set()
            self.type_synonyms = {}
            self.predicate_synonyms = {}

    def normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to canonical form"""
        if not entity_type:
            return "UNKNOWN"

        entity_type_upper = entity_type.upper()

        # Direct match
        if entity_type_upper in self.canonical_types:
            return entity_type_upper

        # Check synonyms
        for canonical, synonyms in self.type_synonyms.items():
            if entity_type in synonyms or entity_type_upper in [s.upper() for s in synonyms]:
                self.stats['type_remappings'][entity_type] = canonical
                return canonical

        # Special remappings for common issues
        if entity_type_upper == "STRING":
            self.stats['type_remappings'][entity_type] = "CONCEPT"
            return "CONCEPT"
        elif entity_type_upper == "GROUP":
            self.stats['type_remappings'][entity_type] = "ORGANIZATION"
            return "ORGANIZATION"
        elif entity_type_upper == "LANGUAGE":
            self.stats['type_remappings'][entity_type] = "CONCEPT"
            return "CONCEPT"

        # Default to CONCEPT for unknown types
        self.stats['type_remappings'][entity_type] = "CONCEPT"
        return "CONCEPT"

    def normalize_predicate(self, predicate: str) -> Tuple[str, str]:
        """Normalize predicate to canonical form, return (normalized, original)"""
        if not predicate:
            return ("related_to", predicate)

        predicate_lower = predicate.lower()

        # Direct match
        if predicate_lower in self.canonical_predicates:
            return (predicate_lower, predicate)

        # Check synonyms
        for canonical, synonyms in self.predicate_synonyms.items():
            if predicate_lower in [s.lower() for s in synonyms]:
                self.stats['predicate_remappings'][predicate] = canonical
                return (canonical, predicate)

        # Special mappings for overly generic predicates
        generic_mappings = {
            "is": "part_of",
            "are": "part_of",
            "was": "part_of",
            "were": "part_of",
            "has": "includes",
            "have": "includes",
            "is a": "part_of",
            "is associated with": "linked_to",
            "associated with": "linked_to",  # Add mapping without "is"
            "provides": "supports",
            "related to": "linked_to"
        }

        if predicate_lower in generic_mappings:
            canonical = generic_mappings[predicate_lower]
            self.stats['predicate_remappings'][predicate] = canonical
            return (canonical, predicate)

        # Keep original but track it
        return (predicate_lower, predicate)

    def load_book_kg(self):
        """Load book knowledge graph"""
        print(f"Loading book KG from {BOOK_KG_PATH}")

        if not BOOK_KG_PATH.exists():
            print(f"Warning: Book KG file not found at {BOOK_KG_PATH}")
            return

        with open(BOOK_KG_PATH, 'r') as f:
            data = json.load(f)

        # Extract entities
        if 'entities' in data:
            for entity_name, entity_info in data['entities'].items():
                # Normalize entity type
                normalized_type = self.normalize_entity_type(entity_info.get('type', 'UNKNOWN'))
                entity_info['type'] = normalized_type
                entity_info['original_type'] = entity_info.get('type')  # Preserve original

                if entity_name not in self.entities:
                    self.entities[entity_name] = entity_info
                else:
                    # Merge sources
                    existing_sources = set(self.entities[entity_name].get('sources', []))
                    new_sources = set(entity_info.get('sources', []))
                    self.entities[entity_name]['sources'] = list(existing_sources | new_sources)

        # Extract relationships
        if 'relationships' in data:
            for rel in data['relationships']:
                self._add_relationship(rel)

        # Track sources
        if 'entities' in data:
            books = set()
            for entity_info in data['entities'].values():
                books.update(entity_info.get('sources', []))
            self.stats['sources']['books'] = list(books)

        print(f"Loaded {len(data.get('entities', {}))} entities and {len(data.get('relationships', []))} relationships from books")

    def load_episode_kgs(self):
        """Load all episode knowledge graphs"""
        print(f"Loading episode KGs from {EPISODE_KG_DIR}")

        if not EPISODE_KG_DIR.exists():
            print(f"Warning: Episode KG directory not found at {EPISODE_KG_DIR}")
            return

        # Support both v3_2_2 inputs and postprocessed files
        episode_files = sorted(EPISODE_KG_DIR.glob("episode_*_v3_2_2.json"))
        if not episode_files:
            episode_files = sorted(EPISODE_KG_DIR.glob("episode_*_post.json"))
        if not episode_files:
            episode_files = sorted(EPISODE_KG_DIR.glob("episode_*.json"))
        print(f"Found {len(episode_files)} episode KG files")

        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    data = json.load(f)

                episode_num = data.get('episode', 'unknown')
                self.stats['sources']['episodes'].append(episode_num)

                # Extract relationships (episode format doesn't have separate entities)
                for rel in data.get('relationships', []):
                    # Create entities from relationship endpoints
                    source = rel.get('source')
                    target = rel.get('target')

                    if source:
                        if source not in self.entities:
                            source_type = self.normalize_entity_type(rel.get('source_type', 'UNKNOWN'))
                            self.entities[source] = {
                                'type': source_type,
                                'original_type': rel.get('source_type'),
                                'description': f"Entity from episode {episode_num}",
                                'sources': [f"episode_{episode_num}"]
                            }
                        else:
                            # Add episode to sources
                            sources = set(self.entities[source].get('sources', []))
                            sources.add(f"episode_{episode_num}")
                            self.entities[source]['sources'] = list(sources)

                    if target:
                        if target not in self.entities:
                            target_type = self.normalize_entity_type(rel.get('target_type', 'UNKNOWN'))
                            self.entities[target] = {
                                'type': target_type,
                                'original_type': rel.get('target_type'),
                                'description': f"Entity from episode {episode_num}",
                                'sources': [f"episode_{episode_num}"]
                            }
                        else:
                            # Add episode to sources
                            sources = set(self.entities[target].get('sources', []))
                            sources.add(f"episode_{episode_num}")
                            self.entities[target]['sources'] = list(sources)

                    # Add the relationship
                    self._add_relationship({
                        'source': source,
                        'target': target,
                        'predicate': rel.get('relationship', 'related_to'),
                        'confidence': rel.get('p_true', 0.5),
                        'evidence': rel.get('evidence_text', ''),
                        'source_type': rel.get('source_type', 'UNKNOWN').upper(),
                        'target_type': rel.get('target_type', 'UNKNOWN').upper(),
                        'episode': episode_num
                    })

            except Exception as e:
                print(f"Error loading {episode_file}: {e}")
                continue

        print(f"Loaded episodes: {sorted(self.stats['sources']['episodes'])[:5]}... (showing first 5)")

    def _add_relationship(self, rel: Dict):
        """Add a relationship and update adjacency"""
        source = rel.get('source')
        target = rel.get('target')
        raw_predicate = rel.get('predicate', rel.get('relationship', 'related_to'))

        # Skip if missing source or target
        if not source or not target:
            return

        # Normalize predicate
        predicate, original_predicate = self.normalize_predicate(raw_predicate)

        # Create relationship ID to avoid duplicates
        rel_id = hashlib.md5(f"{source}_{predicate}_{target}".encode()).hexdigest()

        # Check if relationship already exists
        for existing_rel in self.relationships:
            if existing_rel.get('id') == rel_id:
                return  # Skip duplicate

        # Add relationship
        self.relationships.append({
            'id': rel_id,
            'source': source,
            'target': target,
            'predicate': predicate,
            'original_predicate': original_predicate,
            'confidence': rel.get('confidence', rel.get('p_true', 0.5)),
            'evidence': rel.get('evidence', rel.get('evidence_text', '')),
            'source_type': self.normalize_entity_type(rel.get('source_type', 'UNKNOWN')),
            'target_type': self.normalize_entity_type(rel.get('target_type', 'UNKNOWN')),
            'metadata': {
                'episode': rel.get('episode'),
                'book': rel.get('book')
            }
        })

        # Update adjacency
        self.adjacency[source][predicate].append(target)

    def validate_and_fix_links(self):
        """Validate all relationships and fix zero-links bug"""
        print("\nValidating relationships and fixing zero-links bug...")

        valid_relationships = []
        invalid_count = 0
        orphan_edges = []

        for rel in self.relationships:
            source = rel.get('source')
            target = rel.get('target')
            predicate = rel.get('predicate')

            # Validate relationship has all required fields
            if source and target and predicate:
                # Check if entities exist (orphan edge detection)
                source_exists = source in self.entities
                target_exists = target in self.entities

                # Track orphan edges but still include them
                if not source_exists or not target_exists:
                    orphan_edges.append({
                        'source': source,
                        'target': target,
                        'predicate': predicate,
                        'source_exists': source_exists,
                        'target_exists': target_exists
                    })

                # Ensure source and target exist in entities
                if not source_exists:
                    self.entities[source] = {
                        'type': rel.get('source_type', 'UNKNOWN'),
                        'description': 'Auto-generated entity from relationship',
                        'sources': []
                    }
                if not target_exists:
                    self.entities[target] = {
                        'type': rel.get('target_type', 'UNKNOWN'),
                        'description': 'Auto-generated entity from relationship',
                        'sources': []
                    }

                valid_relationships.append(rel)
            else:
                invalid_count += 1

        self.relationships = valid_relationships
        self.stats['orphan_edges'] = orphan_edges[:100]  # Store first 100 for inspection

        # Calculate orphan edge percentage
        orphan_percentage = (len(orphan_edges) / len(valid_relationships) * 100) if valid_relationships else 0

        print(f"Validated {len(valid_relationships)} relationships, removed {invalid_count} invalid ones")
        print(f"Found {len(orphan_edges)} orphan edges ({orphan_percentage:.2f}%)")

        if orphan_percentage > 0.5:
            print(f"WARNING: Orphan edge rate ({orphan_percentage:.2f}%) exceeds 0.5% threshold")

    def find_orphan_entities(self):
        """Find entities with no relationships"""
        entities_in_relationships = set()

        for rel in self.relationships:
            entities_in_relationships.add(rel['source'])
            entities_in_relationships.add(rel['target'])

        orphans = []
        for entity_name in self.entities.keys():
            if entity_name not in entities_in_relationships:
                orphans.append(entity_name)

        self.stats['orphan_entities'] = orphans
        print(f"Found {len(orphans)} orphan entities")

    def calculate_stats(self):
        """Calculate statistics about the merged graph"""
        self.stats['entity_count'] = len(self.entities)
        self.stats['relationship_count'] = len(self.relationships)

        # Count entity types
        type_counter = Counter()
        for entity_info in self.entities.values():
            type_counter[entity_info.get('type', 'UNKNOWN')] += 1
        self.stats['top_entity_types'] = dict(type_counter.most_common(10))

        # Count predicates
        predicate_counter = Counter()
        for rel in self.relationships:
            predicate_counter[rel['predicate']] += 1
        self.stats['top_predicates'] = dict(predicate_counter.most_common(15))

        # Count multi-source entities
        multi_source_entities = 0
        for entity_info in self.entities.values():
            if len(entity_info.get('sources', [])) > 1:
                multi_source_entities += 1
        self.stats['multi_source_entities'] = multi_source_entities

        # Calculate validation metrics
        orphan_entity_rate = (len(self.stats['orphan_entities']) / self.stats['entity_count'] * 100) if self.stats['entity_count'] > 0 else 0
        self.stats['orphan_entity_percentage'] = orphan_entity_rate

        # Count type remappings
        self.stats['type_remap_count'] = len(self.stats['type_remappings'])
        self.stats['predicate_remap_count'] = len(self.stats['predicate_remappings'])

        # Check if types are canonical
        non_canonical_types = set()
        for entity_info in self.entities.values():
            if entity_info.get('type') not in self.canonical_types:
                non_canonical_types.add(entity_info.get('type'))
        self.stats['non_canonical_types'] = list(non_canonical_types)

        print(f"\nGraph Statistics:")
        print(f"  - Entities: {self.stats['entity_count']}")
        print(f"  - Relationships: {self.stats['relationship_count']}")
        print(f"  - Multi-source entities: {multi_source_entities}")
        print(f"  - Orphan entities: {len(self.stats['orphan_entities'])} ({orphan_entity_rate:.2f}%)")
        print(f"  - Type remappings: {self.stats['type_remap_count']}")
        print(f"  - Predicate remappings: {self.stats['predicate_remap_count']}")
        print(f"  - Top entity types: {list(self.stats['top_entity_types'].keys())[:5]}")
        print(f"  - Top predicates: {list(self.stats['top_predicates'].keys())[:5]}")

        # Validation checks
        if orphan_entity_rate > 2.0:
            print(f"  ⚠️  Warning: Orphan entity rate ({orphan_entity_rate:.2f}%) exceeds 2% threshold")
        if non_canonical_types:
            print(f"  ⚠️  Warning: Found {len(non_canonical_types)} non-canonical types after normalization")

    def save_outputs(self):
        """Save unified graph, adjacency, and stats"""
        print("\nSaving outputs...")

        # Save unified graph
        unified_path = OUTPUT_DIR / "unified.json"
        with open(unified_path, 'w') as f:
            json.dump({
                'entities': self.entities,
                'relationships': self.relationships,
                'metadata': {
                    'entity_count': len(self.entities),
                    'relationship_count': len(self.relationships),
                    'created_at': datetime.now().isoformat()
                }
            }, f, indent=2)
        print(f"Saved unified graph to {unified_path}")

        # Save adjacency
        adjacency_path = OUTPUT_DIR / "adjacency.json"
        # Convert defaultdict to regular dict for JSON serialization
        adjacency_dict = {}
        for source, predicates in self.adjacency.items():
            adjacency_dict[source] = dict(predicates)
        with open(adjacency_path, 'w') as f:
            json.dump(adjacency_dict, f, indent=2)
        print(f"Saved adjacency to {adjacency_path}")

        # Save stats
        stats_path = OUTPUT_DIR / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")

def main():
    import argparse
    print("=" * 60)
    print("Knowledge Graph Merger")
    print("=" * 60)

    # Declare globals before referencing them for defaults
    global BOOK_KG_PATH
    global EPISODE_KG_DIR
    global OUTPUT_DIR

    default_book = str(BOOK_KG_PATH)
    default_episode_dir = str(EPISODE_KG_DIR)
    default_out = str(OUTPUT_DIR)

    ap = argparse.ArgumentParser(description="Merge unified KG from books + episodes")
    ap.add_argument("--book-kg", type=str, default=default_book, help="Path to merged book KG JSON")
    ap.add_argument("--episode-dir", type=str, default=default_episode_dir, help="Directory with episode KG JSON files")
    ap.add_argument("--out-dir", type=str, default=default_out, help="Output directory for unified KG")
    args = ap.parse_args()

    # Override globals for this run
    BOOK_KG_PATH = Path(args.book_kg)
    EPISODE_KG_DIR = Path(args.episode_dir)
    OUTPUT_DIR = Path(args.out_dir)

    merger = KnowledgeGraphMerger()

    # Load book KG
    merger.load_book_kg()

    # Load episode KGs
    merger.load_episode_kgs()

    # Validate and fix
    merger.validate_and_fix_links()

    # Find orphans
    merger.find_orphan_entities()

    # Calculate stats
    merger.calculate_stats()

    # Save outputs
    merger.save_outputs()

    print("\n✅ Knowledge graph merge complete!")
    print(f"   Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
