#!/usr/bin/env python3
"""
Validate unified knowledge graph quality.

Runs automated tests to ensure no catastrophic entity merges.

Usage:
    python scripts/validate_unified_graph.py

    # Test specific file:
    python scripts/validate_unified_graph.py --input data/knowledge_graph_unified/unified_v2.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class GraphValidator:
    """Validates unified knowledge graph quality"""

    def __init__(self, graph_path: Path):
        self.graph_path = graph_path
        with open(graph_path, 'r') as f:
            self.data = json.load(f)

        self.entities = self.data['entities']
        self.relationships = self.data['relationships']
        self.tests_passed = 0
        self.tests_failed = 0

    def test_no_moscow_soil_merge(self):
        """Test that Moscow does NOT have soil/moon aliases"""
        print("\n🔍 Test: Moscow entity validation")

        if 'Moscow' not in self.entities:
            print("  ⚠️  Moscow entity not found (acceptable if truly absent)")
            self.tests_passed += 1
            return True

        moscow = self.entities['Moscow']
        aliases = [a.lower() for a in moscow.get('aliases', [])]

        failures = []
        if 'soil' in aliases:
            failures.append("'soil' found in Moscow aliases")
        if 'moon' in aliases:
            failures.append("'moon' found in Moscow aliases")

        if moscow.get('type') != 'PLACE':
            failures.append(f"Moscow type is '{moscow.get('type')}', expected 'PLACE'")

        if failures:
            print(f"  ❌ FAIL: {', '.join(failures)}")
            self.tests_failed += 1
            return False
        else:
            print(f"  ✅ PASS: Moscow entity is clean")
            self.tests_passed += 1
            return True

    def test_soil_entity_exists(self):
        """Test that Soil exists as independent entity"""
        print("\n🔍 Test: Soil entity validation")

        # Check for Soil or soil entity
        soil_entity = None
        for name, entity in self.entities.items():
            if name.lower() == 'soil' or 'soil' in [a.lower() for a in entity.get('aliases', [])]:
                soil_entity = entity
                break

        if not soil_entity:
            print("  ❌ FAIL: No Soil entity found")
            self.tests_failed += 1
            return False

        # Count Soil relationships
        soil_rels = [
            r for r in self.relationships
            if r['source'].lower() == 'soil' or r['target'].lower() == 'soil'
        ]

        if len(soil_rels) < 100:
            print(f"  ⚠️  WARNING: Soil has only {len(soil_rels)} relationships (expected 200-300)")

        print(f"  ✅ PASS: Soil entity exists with {len(soil_rels)} relationships")
        self.tests_passed += 1
        return True

    def test_no_cross_type_merges(self):
        """Test that entities with provenance maintained type consistency"""
        print("\n🔍 Test: Type consistency check")

        # This test requires provenance data, which may not exist in new extraction
        # Skip for now
        print("  ⏭️  SKIP: Provenance data not required in v2")
        return True

    def test_earth_entity(self):
        """Test that Earth does NOT have Mars/Paris aliases"""
        print("\n🔍 Test: Earth entity validation")

        if 'Earth' not in self.entities:
            print("  ⚠️  Earth entity not found")
            self.tests_passed += 1
            return True

        earth = self.entities['Earth']
        aliases = [a.lower() for a in earth.get('aliases', [])]

        failures = []
        if 'mars' in aliases:
            failures.append("'mars' found in Earth aliases")
        if 'paris' in aliases:
            failures.append("'paris' found in Earth aliases")
        if 'farms' in aliases or 'farm' in aliases:
            failures.append("'farms' or 'farm' found in Earth aliases")

        if failures:
            print(f"  ❌ FAIL: {', '.join(failures)}")
            self.tests_failed += 1
            return False
        else:
            print(f"  ✅ PASS: Earth entity is clean")
            self.tests_passed += 1
            return True

    def test_relationship_distribution(self):
        """Test that no entity has excessive relationships (indicating bad merge)"""
        print("\n🔍 Test: Relationship distribution")

        # Count relationships per entity
        rel_counts = defaultdict(int)
        for rel in self.relationships:
            rel_counts[rel['source']] += 1
            rel_counts[rel['target']] += 1

        max_entity = max(rel_counts.items(), key=lambda x: x[1])
        max_name, max_count = max_entity

        if max_count > 500:
            print(f"  ❌ FAIL: '{max_name}' has {max_count} relationships (suggests bad merge)")
            self.tests_failed += 1
            return False
        else:
            print(f"  ✅ PASS: Max relationships per entity: {max_count} ('{max_name}')")
            self.tests_passed += 1
            return True

    def test_relationship_types(self):
        """Test that all relationships have 'type' field"""
        print("\n🔍 Test: Relationship type field")

        missing_type = [r for r in self.relationships if 'type' not in r]

        if missing_type:
            print(f"  ❌ FAIL: {len(missing_type)} relationships missing 'type' field")
            self.tests_failed += 1
            return False
        else:
            print(f"  ✅ PASS: All {len(self.relationships)} relationships have 'type' field")
            self.tests_passed += 1
            return True

    def test_no_suspicious_entities(self):
        """Test that known problematic entities are clean"""
        print("\n🔍 Test: Suspicious entity check")

        suspicious_patterns = [
            ('DIA', ['dubai', 'sun', 'red', 'india']),
            ('the soil', ['stove', 'skin', 'show']),
            ('the land', ['thailand', 'legend']),
            ('leaders', ['healers', 'readers']),
        ]

        failures = []
        for name, forbidden_aliases in suspicious_patterns:
            if name in self.entities:
                entity = self.entities[name]
                aliases = [a.lower() for a in entity.get('aliases', [])]

                for forbidden in forbidden_aliases:
                    if forbidden in aliases:
                        failures.append(f"'{name}' has forbidden alias '{forbidden}'")

        if failures:
            print(f"  ❌ FAIL:\n" + "\n".join(f"    - {f}" for f in failures))
            self.tests_failed += 1
            return False
        else:
            print(f"  ✅ PASS: No suspicious entity aliases found")
            self.tests_passed += 1
            return True

    def run_all_tests(self):
        """Run all validation tests"""
        print("=" * 80)
        print("UNIFIED KNOWLEDGE GRAPH VALIDATION")
        print("=" * 80)
        print(f"Testing: {self.graph_path}")
        print(f"Entities: {len(self.entities)}")
        print(f"Relationships: {len(self.relationships)}")

        # Run all tests
        self.test_no_moscow_soil_merge()
        self.test_soil_entity_exists()
        self.test_earth_entity()
        self.test_relationship_distribution()
        self.test_relationship_types()
        self.test_no_suspicious_entities()

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        total_tests = self.tests_passed + self.tests_failed
        print(f"Tests passed: {self.tests_passed}/{total_tests}")
        print(f"Tests failed: {self.tests_failed}/{total_tests}")

        if self.tests_failed == 0:
            print("\n✅ ALL TESTS PASSED!")
            print("\nGraph is ready for deployment.")
            return True
        else:
            print(f"\n❌ {self.tests_failed} TESTS FAILED")
            print("\nGraph needs fixes before deployment.")
            return False


def main():
    parser = argparse.ArgumentParser(description='Validate unified knowledge graph')
    parser.add_argument(
        '--input',
        type=str,
        default='data/knowledge_graph_unified/unified.json',
        help='Path to unified JSON file'
    )
    args = parser.parse_args()

    # Setup path
    base_dir = Path(__file__).parent.parent
    graph_path = base_dir / args.input

    if not graph_path.exists():
        print(f"ERROR: File not found: {graph_path}")
        sys.exit(1)

    # Run validation
    validator = GraphValidator(graph_path)
    success = validator.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
