#!/usr/bin/env python3
"""
Entity Deduplication for Knowledge Graph

This script normalizes and merges duplicate entities that appear with
different capitalizations, punctuation, or minor variations.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EntityNormalizer:
    """Normalize and deduplicate entities in knowledge graph"""

    # Canonical entity mappings for important/official names
    CANONICAL_ENTITIES = {
        # YonEarth variants
        "yonearthcommunity": "YonEarth Community",
        "yonearth": "YonEarth",
        "yonearthorg": "YonEarth.org",
        "whyonearthcommunity": "Y on Earth Community",
        "whyonearth": "Y on Earth",
        "earthcoastproductions": "Earth Coast Productions",

        # Common organizations
        "bcorporation": "B Corporation",
        "bcorp": "B Corporation",

        # Common terms
        "covid19": "COVID-19",
        "covid": "COVID-19",
        "coronavirus": "COVID-19",

        # Standardize concepts
        "selfcare": "self-care",
        "regenerativeagriculture": "regenerative agriculture",
        "climatechange": "climate change",
        "carbonsequestration": "carbon sequestration",
    }

    def __init__(self):
        """Initialize normalizer"""
        self.entity_variants = defaultdict(set)
        self.entity_counts = Counter()
        self.canonical_map = {}
        self.stats = {
            'total_entities': 0,
            'unique_normalized': 0,
            'merge_groups': 0,
            'total_merged': 0
        }

    def normalize_key(self, entity: str) -> str:
        """Create normalized key for entity comparison"""
        if not entity:
            return ""

        # Convert to lowercase and strip
        key = entity.lower().strip()

        # Remove punctuation and special characters
        key = re.sub(r'[^a-z0-9]', '', key)

        return key

    def smart_capitalize(self, text: str) -> str:
        """Apply smart capitalization rules"""
        # List of words that should remain lowercase
        lowercase_words = {'and', 'or', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}

        # List of acronyms that should be uppercase
        acronyms = {'usa', 'uk', 'ai', 'ml', 'ceo', 'ngo', 'pdf', 'api', 'gis', 'gmo'}

        words = text.split()
        result = []

        for i, word in enumerate(words):
            word_lower = word.lower()

            # First word is always capitalized
            if i == 0:
                result.append(word.capitalize())
            # Check if it's an acronym
            elif word_lower in acronyms:
                result.append(word.upper())
            # Check if it should remain lowercase
            elif word_lower in lowercase_words:
                result.append(word_lower)
            # Otherwise capitalize
            else:
                result.append(word.capitalize())

        return ' '.join(result)

    def determine_canonical(self, variants: Set[str]) -> str:
        """Determine the canonical form from a set of variants"""
        if not variants:
            return ""

        # Check if any variant matches our canonical entities
        for variant in variants:
            key = self.normalize_key(variant)
            if key in self.CANONICAL_ENTITIES:
                return self.CANONICAL_ENTITIES[key]

        # Sort variants by frequency and quality
        scored_variants = []
        for variant in variants:
            score = 0

            # Prefer variants with proper capitalization
            if variant[0].isupper():
                score += 10

            # Prefer variants without all caps (unless acronym)
            if not variant.isupper() or len(variant) <= 4:
                score += 5

            # Prefer variants with normal spacing
            if ' ' in variant and not '  ' in variant:
                score += 3

            # Add frequency score
            score += self.entity_counts.get(variant, 0)

            scored_variants.append((score, variant))

        # Return highest scoring variant
        scored_variants.sort(reverse=True)
        return scored_variants[0][1]

    def analyze_corpus(self, relationships_dir: Path) -> Dict:
        """Analyze all entities in the corpus"""
        print("Analyzing entity corpus...")

        files = list(relationships_dir.glob("episode_*_extraction.json"))
        print(f"Found {len(files)} extraction files")

        # First pass: collect all entities
        for file_path in files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                for rel in data.get("relationships", []):
                    # Process source entity
                    source = rel.get("source_entity", "").strip()
                    if source:
                        self.stats['total_entities'] += 1
                        key = self.normalize_key(source)
                        self.entity_variants[key].add(source)
                        self.entity_counts[source] += 1

                    # Process target entity
                    target = rel.get("target_entity", "").strip()
                    if target:
                        self.stats['total_entities'] += 1
                        key = self.normalize_key(target)
                        self.entity_variants[key].add(target)
                        self.entity_counts[target] += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Second pass: determine canonical forms
        print(f"\nFound {len(self.entity_variants)} unique normalized entities")

        for norm_key, variants in self.entity_variants.items():
            if len(variants) > 1:
                self.stats['merge_groups'] += 1
                self.stats['total_merged'] += len(variants) - 1

            canonical = self.determine_canonical(variants)
            self.canonical_map[norm_key] = {
                'canonical': canonical,
                'variants': list(variants),
                'total_occurrences': sum(self.entity_counts[v] for v in variants)
            }

        self.stats['unique_normalized'] = len(self.entity_variants)

        return self.stats

    def process_file(self, file_path: Path, output_path: Path = None):
        """Process a single file and normalize entities"""
        with open(file_path) as f:
            data = json.load(f)

        # Normalize each relationship
        normalized_relationships = []
        for rel in data.get("relationships", []):
            normalized_rel = rel.copy()

            # Normalize source entity
            source = rel.get("source_entity", "").strip()
            if source:
                source_key = self.normalize_key(source)
                if source_key in self.canonical_map:
                    canonical_data = self.canonical_map[source_key]
                    normalized_rel["source_entity_canonical"] = canonical_data['canonical']
                    normalized_rel["source_entity_raw"] = source
                else:
                    normalized_rel["source_entity_canonical"] = source
                    normalized_rel["source_entity_raw"] = source

            # Normalize target entity
            target = rel.get("target_entity", "").strip()
            if target:
                target_key = self.normalize_key(target)
                if target_key in self.canonical_map:
                    canonical_data = self.canonical_map[target_key]
                    normalized_rel["target_entity_canonical"] = canonical_data['canonical']
                    normalized_rel["target_entity_raw"] = target
                else:
                    normalized_rel["target_entity_canonical"] = target
                    normalized_rel["target_entity_raw"] = target

            normalized_relationships.append(normalized_rel)

        data["relationships"] = normalized_relationships

        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        return data

    def save_entity_map(self, output_path: Path):
        """Save the entity normalization map"""
        output_data = {
            'canonical_entities': dict(self.CANONICAL_ENTITIES),
            'entity_mappings': self.canonical_map,
            'statistics': self.stats
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved entity map to {output_path}")

    def show_merge_report(self, limit: int = 20):
        """Show report of entities to be merged"""
        print("\n" + "=" * 70)
        print("ENTITY MERGE REPORT")
        print("=" * 70)

        # Find groups with most variants
        merge_groups = []
        for norm_key, data in self.canonical_map.items():
            if len(data['variants']) > 1:
                merge_groups.append({
                    'canonical': data['canonical'],
                    'variants': data['variants'],
                    'count': len(data['variants']),
                    'occurrences': data['total_occurrences']
                })

        # Sort by number of variants
        merge_groups.sort(key=lambda x: (-x['count'], -x['occurrences']))

        print(f"\nTop {limit} entity groups to merge:")
        print("-" * 70)

        for i, group in enumerate(merge_groups[:limit], 1):
            print(f"\n{i}. '{group['canonical']}' ({group['occurrences']} total occurrences)")
            print(f"   Merging {group['count']} variants:")
            for variant in sorted(group['variants']):
                if variant != group['canonical']:
                    print(f"     - {variant}")

        print(f"\n📊 SUMMARY:")
        print(f"  Total entities: {self.stats['total_entities']:,}")
        print(f"  Unique (after normalization): {self.stats['unique_normalized']:,}")
        print(f"  Entity groups to merge: {self.stats['merge_groups']:,}")
        print(f"  Total duplicates to merge: {self.stats['total_merged']:,}")
        print(f"  Reduction: {(self.stats['total_merged'] / self.stats['unique_normalized'] * 100):.1f}%")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate entities in knowledge graph")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze corpus and show merge report")
    parser.add_argument("--merge-variants", action="store_true",
                       help="Process files and merge entity variants")
    parser.add_argument("--input-dir", type=str,
                       default="data/knowledge_graph/relationships",
                       help="Input directory")
    parser.add_argument("--output-dir", type=str,
                       default="data/knowledge_graph/relationships_deduplicated",
                       help="Output directory for deduplicated files")
    parser.add_argument("--save-map", type=str,
                       default="data/knowledge_graph/entity_normalization_map.json",
                       help="Path to save entity normalization map")

    args = parser.parse_args()

    normalizer = EntityNormalizer()
    relationships_dir = Path(args.input_dir)

    # Always analyze first
    normalizer.analyze_corpus(relationships_dir)

    if args.analyze:
        normalizer.show_merge_report(limit=30)
        normalizer.save_entity_map(Path(args.save_map))

    if args.merge_variants:
        print("\n" + "=" * 70)
        print("PROCESSING FILES")
        print("=" * 70)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(relationships_dir.glob("episode_*_extraction.json"))
        print(f"Processing {len(files)} files...")

        for i, file_path in enumerate(files, 1):
            output_path = output_dir / file_path.name
            normalizer.process_file(file_path, output_path)

            if i % 10 == 0:
                print(f"  Processed {i}/{len(files)} files...")

        print(f"\n✅ Deduplication complete!")
        print(f"📁 Output: {output_dir}")

        # Save the entity map
        normalizer.save_entity_map(Path(args.save_map))


if __name__ == "__main__":
    main()