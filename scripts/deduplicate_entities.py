#!/usr/bin/env python3
"""
Deduplicate entities from processed extraction results.

Entities with the same normalized name are merged:
- Descriptions are combined (longest kept)
- Aliases are unioned
- Source references are aggregated
- Fictional status is preserved if any instance is fictional

Usage:
    python scripts/deduplicate_entities.py
    python scripts/deduplicate_entities.py --entities-path /path/to/entities.json
    python scripts/deduplicate_entities.py --stats-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from datetime import datetime

import json

sys.path.insert(0, str(Path(__file__).parent.parent))


class EntityDeduplicator:
    """Deduplicates entities across chunks."""

    def __init__(
        self,
        entities_path: Path = None,
        output_dir: Path = None
    ):
        self.entities_path = entities_path or Path("data/knowledge_graph_unified/entities_processed.json")
        self.output_dir = output_dir or Path("data/knowledge_graph_unified")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Fictional registry (used to keep narrative characters marked fictional even if mixed with episode mentions)
        self._fictional_aliases: Set[str] = set()
        self._fictional_sources: Set[str] = set()
        self._load_fictional_registry()

        # Try to load merge validator for additional checks
        try:
            from src.knowledge_graph.validators.entity_merge_validator import EntityMergeValidator
            self.merge_validator = EntityMergeValidator()
            self._has_validator = True
        except ImportError:
            self._has_validator = False

        # Stats tracking
        self.stats = {
            "input_entities": 0,
            "output_entities": 0,
            "duplicates_merged": 0,
            "by_type": defaultdict(int),
            "largest_merge_groups": [],
            "fictional_entities": 0,
        }

    def load_entities(self) -> List[Dict]:
        """Load entities from JSON file."""
        if not self.entities_path.exists():
            raise FileNotFoundError(f"Entities file not found: {self.entities_path}")

        with open(self.entities_path) as f:
            entities = json.load(f)

        self.stats["input_entities"] = len(entities)
        return entities

    def _normalize_key(self, name: str, entity_type: str) -> Tuple[str, str]:
        """Create a normalized key for entity deduplication."""
        # Normalize name: lowercase, strip whitespace
        normalized_name = name.lower().strip()
        # Normalize type: uppercase, strip
        normalized_type = entity_type.upper().strip() if entity_type else "UNKNOWN"
        return (normalized_name, normalized_type)

    def _load_fictional_registry(self) -> None:
        """Load fictional character aliases and source identifiers, if available."""
        registry_path = Path("data/fictional_characters.json")
        if not registry_path.exists():
            return
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except Exception:
            return

        for _, data in registry.get("characters", {}).items():
            if data.get("full_name"):
                self._fictional_aliases.add(data["full_name"].lower())
            for alias in data.get("aliases", []):
                self._fictional_aliases.add(alias.lower())

        for _, src in registry.get("sources", {}).items():
            for ident in src.get("source_identifiers", []):
                self._fictional_sources.add(ident.lower())

        # Also treat book_* sources as narrative when present in entity sources
        self._fictional_sources.update({"book_veriditas", "book_ourbiggestdeal"})

    def deduplicate(self, entities: List[Dict]) -> List[Dict]:
        """Deduplicate entities by normalized name and type."""
        # Group by (normalized_name, type)
        grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

        for entity in entities:
            name = entity.get("name", "")
            entity_type = entity.get("type", "UNKNOWN")
            key = self._normalize_key(name, entity_type)
            grouped[key].append(entity)

        deduplicated = []
        merge_groups = []

        for (name_lower, entity_type), group in grouped.items():
            merged = self._merge_group(group)
            deduplicated.append(merged)
            self.stats["by_type"][entity_type] += 1

            if merged.get("is_fictional"):
                self.stats["fictional_entities"] += 1

            # Track large merge groups for analysis
            if len(group) > 1:
                self.stats["duplicates_merged"] += len(group) - 1
                merge_groups.append({
                    "name": merged["name"],
                    "type": entity_type,
                    "count": len(group),
                    "sources": merged.get("sources", [])[:5]  # First 5 sources
                })

        # Sort merge groups by count and keep top 20
        merge_groups.sort(key=lambda x: -x["count"])
        self.stats["largest_merge_groups"] = merge_groups[:20]

        self.stats["output_entities"] = len(deduplicated)
        return deduplicated

    def _merge_group(self, group: List[Dict]) -> Dict:
        """Merge a group of same-name entities."""
        if len(group) == 1:
            entity = group[0].copy()
            entity["mention_count"] = 1
            entity["sources"] = [f"{entity.get('source_type', 'unknown')}_{entity.get('source_id', 'unknown')}"]
            return entity

        # Use first entity as base (they should all have the same canonical name)
        # Prefer the one with the most complete data (longest description)
        group_sorted = sorted(group, key=lambda e: len(e.get("description", "")), reverse=True)
        base = group_sorted[0]

        merged = base.copy()

        # Collect all unique values
        all_aliases: Set[str] = set()
        all_sources: Set[str] = set()
        descriptions: List[str] = []
        fictional_seen = False
        non_fictional_seen = False

        for entity in group:
            # Union aliases
            aliases = entity.get("aliases", [])
            if aliases:
                all_aliases.update(aliases)

            # Track sources
            source_key = f"{entity.get('source_type', 'unknown')}_{entity.get('source_id', 'unknown')}"
            all_sources.add(source_key)

            # Collect descriptions (for longest selection)
            desc = entity.get("description", "")
            if desc and desc not in descriptions:
                descriptions.append(desc)

            # If any instance is fictional, mark as fictional
            if entity.get("is_fictional", False):
                fictional_seen = True
            else:
                non_fictional_seen = True

            # Also add original names to aliases
            original_name = entity.get("original_name")
            if original_name:
                all_aliases.add(original_name)

        # Update merged entity
        merged["aliases"] = sorted(list(all_aliases)) if all_aliases else []
        merged["description"] = max(descriptions, key=len) if descriptions else ""
        name_lower = merged.get("name", "").lower()
        has_fictional_alias = name_lower in self._fictional_aliases
        has_narrative_source = any(
            s.lower() in self._fictional_sources or s.lower().startswith("book_")
            for s in all_sources
        )

        merged["is_fictional"] = (
            (fictional_seen and not non_fictional_seen)
            or (has_fictional_alias and has_narrative_source)
        )
        merged["mention_count"] = len(group)
        merged["sources"] = sorted(list(all_sources))

        # Clean up source_type and source_id since we now have 'sources' list
        merged.pop("source_type", None)
        merged.pop("source_id", None)

        # Update provenance
        merged["provenance"] = {
            "extraction": "gpt-4o-mini-batch",
            "deduplicated_at": datetime.now().isoformat(),
            "merged_from_count": len(group)
        }

        return merged

    def process(self, dry_run: bool = False) -> List[Dict]:
        """Run the deduplication process."""
        print("Loading entities...")
        entities = self.load_entities()
        print(f"  Loaded {len(entities)} entities")

        print("\nDeduplicating...")
        deduplicated = self.deduplicate(entities)
        print(f"  Deduplicated to {len(deduplicated)} unique entities")

        if not dry_run:
            # Save deduplicated entities
            output_path = self.output_dir / "entities_deduplicated.json"
            with open(output_path, 'w') as f:
                json.dump(deduplicated, f, indent=2)
            print(f"\nSaved to: {output_path}")

            # Save deduplication stats
            stats_path = self.output_dir / "deduplication_stats.json"
            stats_to_save = {
                k: dict(v) if isinstance(v, defaultdict) else v
                for k, v in self.stats.items()
            }
            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            print(f"Stats saved to: {stats_path}")

        return deduplicated

    def print_stats(self):
        """Print deduplication statistics."""
        print("\n" + "=" * 50)
        print("DEDUPLICATION STATISTICS")
        print("=" * 50)

        input_count = self.stats["input_entities"]
        output_count = self.stats["output_entities"]
        reduction = 100 * (1 - output_count / input_count) if input_count > 0 else 0

        print(f"\nInput entities: {input_count}")
        print(f"Output entities: {output_count}")
        print(f"Reduction: {reduction:.1f}%")
        print(f"Duplicates merged: {self.stats['duplicates_merged']}")
        print(f"Fictional entities: {self.stats['fictional_entities']}")

        if self.stats["by_type"]:
            print(f"\nEntity types:")
            sorted_types = sorted(
                self.stats["by_type"].items(),
                key=lambda x: -x[1]
            )
            for entity_type, count in sorted_types:
                print(f"  {entity_type}: {count}")

        if self.stats["largest_merge_groups"]:
            print(f"\nLargest merge groups (entities mentioned in multiple chunks):")
            for item in self.stats["largest_merge_groups"][:10]:
                print(f"  {item['name']} ({item['type']}): {item['count']} mentions")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate extracted entities")
    parser.add_argument("--entities-path", type=str, help="Path to entities_processed.json")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Process without saving")
    parser.add_argument("--stats-only", action="store_true", help="Show stats from previous run")

    args = parser.parse_args()

    if args.stats_only:
        stats_path = Path("data/knowledge_graph_unified/deduplication_stats.json")
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            print(json.dumps(stats, indent=2))
        else:
            print("No stats file found. Run deduplication first.")
        return

    deduplicator = EntityDeduplicator(
        entities_path=Path(args.entities_path) if args.entities_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    try:
        deduplicator.process(dry_run=args.dry_run)
        deduplicator.print_stats()

        if args.dry_run:
            print("\n[DRY RUN - No files saved]")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have run the processing step first:")
        print("  python scripts/process_batch_results.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
