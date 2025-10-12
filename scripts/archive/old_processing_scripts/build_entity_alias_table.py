#!/usr/bin/env python3
"""
Entity Alias Table Builder

Analyzes extraction results to build a comprehensive alias resolution table
for entity deduplication in knowledge graph extractions.

Usage:
    python3 scripts/build_entity_alias_table.py
"""

import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
BOOKS_KG_DIR = DATA_DIR / "knowledge_graph_books_v3_2_2"
EPISODES_KG_DIR = DATA_DIR / "knowledge_graph_v3_2_2"  # Update when episodes extracted
OUTPUT_FILE = DATA_DIR / "processed" / "entity_aliases.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def normalize(s: str) -> str:
    """
    Normalize entity strings for comparison.
    Same as canon() in master guide.
    """
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)  # Drop punctuation/dashes
    s = re.sub(r"\s+", " ", s)       # Normalize whitespace
    return s


def extract_entities_from_kg(kg_file: Path) -> Set[str]:
    """Extract all unique entities from a knowledge graph file"""
    entities = set()

    with open(kg_file, 'r') as f:
        data = json.load(f)

    for rel in data.get('relationships', []):
        entities.add(rel['source'])
        entities.add(rel['target'])

    return entities


def build_alias_groups(entities: Set[str]) -> Dict[str, List[str]]:
    """Group entities by normalized form to find duplicates"""
    normalized_groups = defaultdict(list)

    for entity in entities:
        norm = normalize(entity)
        normalized_groups[norm].append(entity)

    # Only keep groups with multiple variants
    return {norm: variants for norm, variants in normalized_groups.items()
            if len(variants) > 1}


def select_canonical_form(variants: List[str]) -> str:
    """
    Select the best canonical form from variants.

    Rules:
    1. Prefer proper capitalization over all lowercase/uppercase
    2. Prefer longer forms (more specific)
    3. Prefer forms without special Unicode characters
    """
    # Score each variant
    scored = []
    for v in variants:
        score = 0

        # Prefer title case or proper capitalization
        if v[0].isupper() and not v.isupper():
            score += 10

        # Prefer longer (more descriptive)
        score += len(v) * 0.1

        # Penalize all uppercase (likely acronym needing expansion)
        if v.isupper():
            score -= 5

        # Penalize special Unicode characters
        if any(ord(c) > 127 for c in v):
            score -= 3

        scored.append((score, v))

    # Return highest scoring
    return max(scored, key=lambda x: x[0])[1]


def build_alias_table(alias_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build final alias table mapping normalized form -> canonical form.
    """
    alias_table = {}

    for norm, variants in alias_groups.items():
        canonical = select_canonical_form(variants)
        alias_table[norm] = canonical

    return alias_table


def add_manual_aliases(alias_table: Dict[str, str]) -> Dict[str, str]:
    """Add known manual aliases from domain knowledge"""
    manual_aliases = {
        # Common variations
        normalize("Y on Earth"): "Y on Earth",
        normalize("YonEarth"): "Y on Earth",
        normalize("yon earth"): "Y on Earth",
        normalize("IBI"): "International Biochar Initiative",
        normalize("International Biochar Initiative"): "International Biochar Initiative",

        # Add more as discovered during review
    }

    # Merge manual with discovered (manual takes precedence)
    result = alias_table.copy()
    result.update(manual_aliases)

    return result


def save_alias_table(alias_table: Dict[str, str], output_file: Path):
    """Save alias table to JSON with statistics"""

    # Group by canonical form for better inspection
    by_canonical = defaultdict(list)
    for norm, canonical in alias_table.items():
        if norm != normalize(canonical):  # Only include actual aliases
            by_canonical[canonical].append(norm)

    output_data = {
        "version": "1.0",
        "total_aliases": len(alias_table),
        "canonical_entities": len(by_canonical),
        "alias_table": alias_table,
        "aliases_by_canonical": dict(by_canonical),
        "stats": {
            "entities_with_multiple_forms": len(by_canonical),
            "total_normalized_forms": len(alias_table),
            "compression_ratio": f"{len(alias_table) / len(by_canonical):.2f}x" if by_canonical else "N/A"
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_data


def main():
    print("=" * 80)
    print("🔧 ENTITY ALIAS TABLE BUILDER")
    print("=" * 80)
    print()

    # Extract entities from all KG files
    all_entities = set()

    # Books
    for kg_file in BOOKS_KG_DIR.glob("*.json"):
        if kg_file.name.startswith("checkpoint"):
            continue
        print(f"📖 Extracting entities from: {kg_file.name}")
        entities = extract_entities_from_kg(kg_file)
        all_entities.update(entities)
        print(f"   Found {len(entities)} entities")

    # Episodes (if they exist)
    if EPISODES_KG_DIR.exists():
        for kg_file in EPISODES_KG_DIR.glob("*.json"):
            if kg_file.name.startswith("checkpoint"):
                continue
            print(f"🎙️  Extracting entities from: {kg_file.name}")
            entities = extract_entities_from_kg(kg_file)
            all_entities.update(entities)
            print(f"   Found {len(entities)} entities")

    print()
    print(f"📊 Total unique entities: {len(all_entities)}")
    print()

    # Build alias groups
    print("🔍 Identifying duplicate entities...")
    alias_groups = build_alias_groups(all_entities)
    print(f"   Found {len(alias_groups)} groups with duplicates")
    print()

    # Show top duplicates
    print("📋 Top 10 Duplicate Groups:")
    print("-" * 80)
    for i, (norm, variants) in enumerate(sorted(alias_groups.items(),
                                                  key=lambda x: len(x[1]),
                                                  reverse=True)[:10], 1):
        canonical = select_canonical_form(variants)
        print(f"{i:2d}. {canonical} ({len(variants)} variants)")
        for v in variants:
            marker = "✓" if v == canonical else " "
            print(f"    {marker} {v}")
    print()

    # Build final alias table
    print("🛠️  Building alias resolution table...")
    alias_table = build_alias_table(alias_groups)
    alias_table = add_manual_aliases(alias_table)
    print(f"   Created {len(alias_table)} alias mappings")
    print()

    # Save
    print(f"💾 Saving alias table to: {OUTPUT_FILE}")
    output_data = save_alias_table(alias_table, OUTPUT_FILE)
    print()

    # Statistics
    print("=" * 80)
    print("✅ ALIAS TABLE COMPLETE")
    print("=" * 80)
    print(f"Total normalized forms: {output_data['total_aliases']}")
    print(f"Canonical entities: {output_data['canonical_entities']}")
    print(f"Compression ratio: {output_data['stats']['compression_ratio']}")
    print()
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("=" * 80)

    # Create a simple resolver class for import
    resolver_code = '''"""
Simple entity alias resolver - auto-generated
"""
import json
from pathlib import Path

class EntityAliasResolver:
    """Production entity alias resolver"""

    def __init__(self, alias_file=None):
        if alias_file is None:
            alias_file = Path(__file__).parent.parent / "data" / "processed" / "entity_aliases.json"

        with open(alias_file, 'r') as f:
            data = json.load(f)

        self.alias_table = data['alias_table']
        self.version = data['version']

    def resolve(self, entity: str) -> str:
        """Resolve entity to its canonical form"""
        import re
        import unicodedata

        # Normalize for lookup
        norm = unicodedata.normalize("NFKC", entity).casefold().strip()
        norm = re.sub(r"[^\\w\\s]", " ", norm)
        norm = re.sub(r"\\s+", " ", norm)

        # Return canonical or original
        return self.alias_table.get(norm, entity)

    def get_stats(self):
        """Get resolver statistics"""
        return {
            'version': self.version,
            'total_aliases': len(self.alias_table)
        }
'''

    resolver_file = DATA_DIR / "processed" / "entity_resolver.py"
    with open(resolver_file, 'w') as f:
        f.write(resolver_code)

    print(f"📦 Created resolver module: {resolver_file}")
    print()


if __name__ == "__main__":
    main()
