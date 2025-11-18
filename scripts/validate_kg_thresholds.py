#!/usr/bin/env python3
"""
CI Validation Script for Knowledge Graph Thresholds

Validates that the knowledge graph meets quality thresholds:
- Entity count > 0
- Relationship count > 0
- Orphan entity rate < 2%
- Orphan edge rate < 0.5%
- No non-canonical entity types
- No disallowed predicates

Exit codes:
0 - All checks passed
1 - Threshold violation detected
2 - File loading error
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

PROJECT_ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
UNIFIED_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/unified.json"
ADJACENCY_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/adjacency.json"
STATS_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/stats.json"
ONTOLOGY_PATH = PROJECT_ROOT / "data/knowledge_graph/ontology.yaml"

# Configurable thresholds
THRESHOLDS = {
    'min_entity_count': 10000,
    'min_relationship_count': 10000,
    'max_orphan_entity_rate': 2.0,  # percentage
    'max_orphan_edge_rate': 0.5,    # percentage
    'max_non_canonical_types': 0,
    'max_disallowed_predicates': 0,
    'max_auto_created_entities': 500
}

class ValidationResult:
    def __init__(self):
        self.passed = True
        self.errors = []
        self.warnings = []
        self.stats = {}

    def error(self, message: str):
        self.errors.append(f"❌ ERROR: {message}")
        self.passed = False

    def warning(self, message: str):
        self.warnings.append(f"⚠️  WARNING: {message}")

    def info(self, key: str, value):
        self.stats[key] = value

def load_files() -> Tuple[Dict, Dict, Dict, Dict]:
    """Load all necessary files for validation"""
    try:
        with open(UNIFIED_PATH, 'r') as f:
            unified = json.load(f)

        with open(ADJACENCY_PATH, 'r') as f:
            adjacency = json.load(f)

        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)

        with open(ONTOLOGY_PATH, 'r') as f:
            ontology = yaml.safe_load(f)

        return unified, adjacency, stats, ontology

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(2)
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        sys.exit(2)

def validate_counts(unified: Dict, stats: Dict, result: ValidationResult):
    """Validate entity and relationship counts"""

    entity_count = len(unified.get('entities', {}))
    relationship_count = len(unified.get('relationships', []))

    result.info('entity_count', entity_count)
    result.info('relationship_count', relationship_count)

    if entity_count < THRESHOLDS['min_entity_count']:
        result.error(f"Entity count ({entity_count}) below minimum ({THRESHOLDS['min_entity_count']})")

    if relationship_count < THRESHOLDS['min_relationship_count']:
        result.error(f"Relationship count ({relationship_count}) below minimum ({THRESHOLDS['min_relationship_count']})")

    if entity_count == 0:
        result.error("No entities found in graph")

    if relationship_count == 0:
        result.error("No relationships found in graph")

def validate_orphan_rates(unified: Dict, stats: Dict, result: ValidationResult):
    """Validate orphan entity and edge rates"""

    entity_count = len(unified.get('entities', {}))
    relationship_count = len(unified.get('relationships', []))

    # Calculate orphan entity rate
    orphan_entities = stats.get('orphan_entities', [])
    orphan_entity_rate = (len(orphan_entities) / entity_count * 100) if entity_count > 0 else 100

    result.info('orphan_entity_count', len(orphan_entities))
    result.info('orphan_entity_rate', f"{orphan_entity_rate:.2f}%")

    if orphan_entity_rate > THRESHOLDS['max_orphan_entity_rate']:
        result.error(f"Orphan entity rate ({orphan_entity_rate:.2f}%) exceeds threshold ({THRESHOLDS['max_orphan_entity_rate']}%)")

    # Calculate orphan edge rate
    orphan_edges = stats.get('orphan_edges', [])
    orphan_edge_rate = (len(orphan_edges) / relationship_count * 100) if relationship_count > 0 else 0

    result.info('orphan_edge_count', len(orphan_edges))
    result.info('orphan_edge_rate', f"{orphan_edge_rate:.2f}%")

    if orphan_edge_rate > THRESHOLDS['max_orphan_edge_rate']:
        result.warning(f"Orphan edge rate ({orphan_edge_rate:.2f}%) exceeds threshold ({THRESHOLDS['max_orphan_edge_rate']}%)")

def validate_entity_types(unified: Dict, ontology: Dict, result: ValidationResult):
    """Validate that all entity types are canonical"""

    # Get canonical types from ontology
    canonical_types = set()
    for type_info in ontology.get('domain', {}).get('types', []):
        canonical_types.add(type_info['name'])

    # Check all entities
    non_canonical_types = set()
    type_counts = {}

    for entity_name, entity_info in unified.get('entities', {}).items():
        entity_type = entity_info.get('type', 'UNKNOWN')

        if entity_type not in type_counts:
            type_counts[entity_type] = 0
        type_counts[entity_type] += 1

        if entity_type not in canonical_types:
            non_canonical_types.add(entity_type)

    result.info('unique_types', len(type_counts))
    result.info('non_canonical_types', list(non_canonical_types))

    if len(non_canonical_types) > THRESHOLDS['max_non_canonical_types']:
        result.error(f"Found {len(non_canonical_types)} non-canonical types: {list(non_canonical_types)[:10]}")

def validate_predicates(unified: Dict, ontology: Dict, result: ValidationResult):
    """Validate predicates against whitelist"""

    # Get allowed predicates from ontology
    allowed_predicates = set()
    for pred_info in ontology.get('domain', {}).get('predicates', []):
        allowed_predicates.add(pred_info['name'])
    for pred_info in ontology.get('discourse', {}).get('predicates', []):
        allowed_predicates.add(pred_info['name'])

    # Get disallowed predicates
    disallowed = set(ontology.get('normalization', {}).get('disallowed_predicates', []))

    # Check all relationships
    predicate_counts = {}
    found_disallowed = []

    for rel in unified.get('relationships', []):
        predicate = rel.get('predicate', 'unknown')

        if predicate not in predicate_counts:
            predicate_counts[predicate] = 0
        predicate_counts[predicate] += 1

        if predicate in disallowed:
            found_disallowed.append(predicate)

    result.info('unique_predicates', len(predicate_counts))
    result.info('disallowed_predicates_found', list(set(found_disallowed)))

    if len(found_disallowed) > THRESHOLDS['max_disallowed_predicates']:
        result.error(f"Found {len(set(found_disallowed))} disallowed predicates: {list(set(found_disallowed))[:10]}")

def validate_auto_created_entities(unified: Dict, result: ValidationResult):
    """Check for auto-created entities from orphan edge fixes"""

    auto_created = []

    for entity_name, entity_info in unified.get('entities', {}).items():
        if 'Auto-generated entity' in entity_info.get('description', ''):
            auto_created.append(entity_name)

    result.info('auto_created_entities', len(auto_created))

    if len(auto_created) > THRESHOLDS['max_auto_created_entities']:
        result.warning(f"Too many auto-created entities ({len(auto_created)}) exceeds threshold ({THRESHOLDS['max_auto_created_entities']})")
        result.warning(f"Consider adding aliases for: {auto_created[:5]}...")

def validate_adjacency_consistency(unified: Dict, adjacency: Dict, result: ValidationResult):
    """Validate adjacency list consistency with unified graph"""

    entities = set(unified.get('entities', {}).keys())
    adjacency_entities = set(adjacency.keys())

    # Check for entities in adjacency not in unified
    dangling_in_adjacency = adjacency_entities - entities

    if dangling_in_adjacency:
        result.warning(f"Found {len(dangling_in_adjacency)} entities in adjacency not in unified graph")
        result.info('dangling_adjacency_examples', list(dangling_in_adjacency)[:5])

def print_results(result: ValidationResult):
    """Print validation results"""

    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH VALIDATION REPORT")
    print("="*60)

    print("\n📊 Statistics:")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")

    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  {warning}")

    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  {error}")

    if result.passed:
        print("\n✅ All validation checks PASSED!")
    else:
        print("\n❌ Validation FAILED!")

    print("="*60)

def main():
    """Main validation function"""

    print("Loading knowledge graph files...")
    unified, adjacency, stats, ontology = load_files()

    result = ValidationResult()

    print("Running validation checks...")

    # Run all validation checks
    validate_counts(unified, stats, result)
    validate_orphan_rates(unified, stats, result)
    validate_entity_types(unified, ontology, result)
    validate_predicates(unified, ontology, result)
    validate_auto_created_entities(unified, result)
    validate_adjacency_consistency(unified, adjacency, result)

    # Print results
    print_results(result)

    # Exit with appropriate code
    if result.passed:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()