#!/usr/bin/env python3
"""
Post-processing pipeline for batch extraction results.

Workflow:
1. Load batch results (keyed by parent_chunk_id)
2. Load parent chunks (for source text context)
3. Apply quality filters (Phases 1-6)
4. Resolve entities to canonical forms
5. Output clean entities and relationships

Usage:
    python scripts/process_batch_results.py
    python scripts/process_batch_results.py --dry-run
    python scripts/process_batch_results.py --stats-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.knowledge_graph.validators.entity_quality_filter import EntityQualityFilter
from src.knowledge_graph.validators.entity_merge_validator import EntityMergeValidator
from src.knowledge_graph.resolvers.entity_resolver import EntityResolver
from src.knowledge_graph.ontology import EntityType, RelationshipType
from src.knowledge_graph.postprocessing.universal.enhanced_list_splitter import (
    is_list_entity,
    split_list_entity,
    split_compound_entity,
)


class BatchResultProcessor:
    """Processes batch extraction results through quality filters."""

    def __init__(
        self,
        batch_results_path: Path = None,
        parent_chunks_path: Path = None,
        output_dir: Path = None
    ):
        self.batch_results_path = batch_results_path or Path("data/batch_jobs/results/extraction_results.json")
        self.parent_chunks_path = parent_chunks_path or Path("data/batch_jobs/parent_chunks.json")
        self.output_dir = output_dir or Path("data/knowledge_graph_unified")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Content metadata (for reality tags, etc.)
        self.content_metadata = {}
        metadata_path = Path("data/batch_jobs/content_metadata.json")
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    self.content_metadata = json.load(f)
            except Exception:
                self.content_metadata = {}

        # Initialize filters
        self.quality_filter = EntityQualityFilter()
        self.merge_validator = EntityMergeValidator()

        # Try to load entity resolver (may fail if registry doesn't exist)
        try:
            self.entity_resolver = EntityResolver(Path("data/canonical_entities.json"))
            self._has_resolver = True
        except FileNotFoundError:
            print("Warning: canonical_entities.json not found. Entity resolution will be skipped.")
            self._has_resolver = False

        # Try to load fictional character tagger
        try:
            from src.knowledge_graph.postprocessing.content_specific.books.fictional_character_tagger import FictionalCharacterTagger
            self.fictional_tagger = FictionalCharacterTagger()
            self._has_fictional_tagger = True
        except FileNotFoundError:
            print("Warning: fictional_characters.json not found. Fictional tagging will be skipped.")
            self._has_fictional_tagger = False

        # Stats tracking
        self.stats = {
            "total_chunks_processed": 0,
            "total_entities_raw": 0,
            "total_relationships_raw": 0,
            "entities_after_quality_filter": 0,
            "entities_after_list_split": 0,
            "entities_after_resolution": 0,
            "entities_fictional_tagged": 0,
            "relationships_valid": 0,
            "relationships_orphaned": 0,
            "filter_reasons": defaultdict(int),
            "resolution_methods": defaultdict(int),
        }

    def load_batch_results(self) -> Dict[str, Dict]:
        """Load batch extraction results keyed by parent_chunk_id."""
        if not self.batch_results_path.exists():
            raise FileNotFoundError(f"Batch results not found at: {self.batch_results_path}")

        with open(self.batch_results_path) as f:
            return json.load(f)

    def load_parent_chunks(self) -> Dict[str, Dict]:
        """Load parent chunks for source text context."""
        if not self.parent_chunks_path.exists():
            raise FileNotFoundError(f"Parent chunks not found at: {self.parent_chunks_path}")

        with open(self.parent_chunks_path) as f:
            chunks = json.load(f)
        return {chunk["id"]: chunk for chunk in chunks}

    def _apply_quality_filter(self, entity: Dict) -> Tuple[bool, str]:
        """Apply quality filter to a single entity."""
        return self.quality_filter.filter_entity(entity)

    def _apply_list_splitting(self, entity: Dict) -> List[Dict]:
        """Split list entities into individual entities."""
        name = entity.get("name", "")

        # Check if it's a list entity
        if is_list_entity(name):
            split_entities = split_list_entity(entity)
            if len(split_entities) > 1:
                return split_entities

        # Check for compound PERSON names
        if entity.get("type", "").upper() == "PERSON":
            compound_entities = split_compound_entity(entity)
            if len(compound_entities) > 1:
                return compound_entities

        return [entity]

    def _apply_resolution(self, entity: Dict) -> Tuple[Dict, str, float, str]:
        """
        Resolve entity name to canonical form.

        Returns:
            Tuple of (entity, resolved_name, confidence, method)
        """
        if not self._has_resolver:
            return entity, entity.get("name", ""), 0.0, "no_resolver"

        name = entity.get("name", "")
        entity_type = entity.get("type", "")

        resolved_name, confidence, method = self.entity_resolver.resolve(name, entity_type)

        return entity, resolved_name, confidence, method

    def _apply_fictional_tagging(self, entity: Dict, source_id: str, reality_tag: str) -> bool:
        """Check if entity should be tagged as fictional."""
        if not self._has_fictional_tagger:
            return False

        # Only tag when the content is marked as narrative/fictional
        if str(reality_tag or "").lower() != "fictional":
            return False

        # Enrich sources with variants of the source_id to maximize registry matching
        sources = set(entity.get("sources", []))
        if source_id:
            sources.add(source_id)
            sources.add(source_id.lower())
            sources.add(source_id.replace("_", "-").lower())
            # Insert hyphens between CamelCase words (OurBiggestDeal -> our-biggest-deal)
            camel = "".join(["-" + c.lower() if c.isupper() else c for c in source_id]).lstrip("-")
            sources.add(camel.replace(" ", "-"))
        entity["sources"] = list(sources)

        tagged = self.fictional_tagger.tag_entity(entity, strict_mode=True)
        return tagged.get("is_fictional", False)

    def process_entities(
        self,
        entities: List[Dict],
        parent_chunk: Dict
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Process entities through quality filters.

        Returns:
            Tuple of (filtered_entities, name_mapping)
            name_mapping maps original names to resolved names
        """
        filtered = []
        name_mapping = {}
        source_type = parent_chunk.get("source_type", "episode")
        source_id = parent_chunk.get("source_id", "unknown")

        chunk_metadata = self.content_metadata.get(parent_chunk.get("id", ""), {})
        reality_tag = chunk_metadata.get("reality_tag", "")

        for entity in entities:
            self.stats["total_entities_raw"] += 1

            # 1. Quality filter
            passes, reason = self._apply_quality_filter(entity)
            if not passes:
                self.stats["filter_reasons"][reason] += 1
                continue

            self.stats["entities_after_quality_filter"] += 1

            # 2. List splitting
            split_entities = self._apply_list_splitting(entity)

            for split_entity in split_entities:
                self.stats["entities_after_list_split"] += 1
                processed = self._process_single_entity(
                    split_entity, source_type, source_id, reality_tag, name_mapping
                )
                if processed:
                    filtered.append(processed)

        return [e for e in filtered if e is not None], name_mapping

    def _process_single_entity(
        self,
        entity: Dict,
        source_type: str,
        source_id: str,
        reality_tag: str,
        name_mapping: Dict[str, str]
    ) -> Optional[Dict]:
        """Process a single entity through resolution and tagging."""
        entity_copy = dict(entity)
        name = entity_copy.get("name", "")
        entity_type = entity_copy.get("type", "CONCEPT")

        # 1) Fictional tagging first to avoid resolving narrative names into real canonicals
        is_fictional = self._apply_fictional_tagging(entity_copy, source_id, reality_tag)
        if is_fictional:
            self.stats["entities_fictional_tagged"] += 1

        # 2) Entity resolution (skip for fictional to avoid conflation)
        if is_fictional:
            resolved_name, confidence, method = name, 1.0, "fictional_skip"
        else:
            _, resolved_name, confidence, method = self._apply_resolution(entity_copy)

        self.stats["resolution_methods"][method] += 1
        self.stats["entities_after_resolution"] += 1

        if resolved_name != name:
            name_mapping[name] = resolved_name

        # Normalize entity type against ontology unless it's explicitly fictional
        if is_fictional and str(entity_type).upper().startswith("FICTIONAL"):
            normalized_type = entity_type
        else:
            normalized_type = EntityType.normalize(entity_type).value

        # Build processed entity
        extraction_label = f"{settings.graph_extraction_model}-{settings.graph_extraction_mode}"
        processed = {
            "name": resolved_name,
            "original_name": name if resolved_name != name else None,
            "type": normalized_type,
            "description": entity_copy.get("description", ""),
            "aliases": entity_copy.get("aliases", []),
            "is_fictional": is_fictional,
            "resolution_confidence": confidence,
            "resolution_method": method,
            "source_type": source_type,
            "source_id": source_id,
            "sources": entity_copy.get("sources", []),
            "provenance": {
                "extraction": extraction_label,
                "processed_at": datetime.now().isoformat()
            }
        }

        # Add original name to aliases if resolved
        if processed["original_name"] and processed["original_name"] not in processed["aliases"]:
            aliases = list(processed["aliases"]) if processed["aliases"] else []
            aliases.append(processed["original_name"])
            processed["aliases"] = aliases

        # Remove None values for cleaner output
        return {k: v for k, v in processed.items() if v is not None}

    def process_relationships(
        self,
        relationships: List[Dict],
        name_mapping: Dict[str, str],
        valid_entities: set,
        parent_chunk: Dict
    ) -> List[Dict]:
        """
        Process relationships, applying name resolution and orphan detection.

        Args:
            relationships: Raw relationships from extraction
            name_mapping: Mapping of original names to resolved names
            valid_entities: Set of valid entity names after filtering
            parent_chunk: Source chunk for provenance
        """
        processed = []
        source_type = parent_chunk.get("source_type", "episode")
        source_id = parent_chunk.get("source_id", "unknown")
        extraction_label = f"{settings.graph_extraction_model}-{settings.graph_extraction_mode}"

        for rel in relationships:
            self.stats["total_relationships_raw"] += 1

            source = rel.get("source", "")
            target = rel.get("target", "")
            predicate = rel.get("predicate", "RELATES_TO")

            # Apply name resolution
            resolved_source = name_mapping.get(source, source)
            resolved_target = name_mapping.get(target, target)

            # Normalize predicate to ontology
            normalized_predicate = RelationshipType.normalize(predicate).value

            # Check for orphans (entities not in valid set)
            source_valid = resolved_source.lower() in valid_entities
            target_valid = resolved_target.lower() in valid_entities

            if not source_valid or not target_valid:
                self.stats["relationships_orphaned"] += 1
                continue

            self.stats["relationships_valid"] += 1

            rel_processed = {
                "source": resolved_source,
                "target": resolved_target,
                "predicate": normalized_predicate,
                "source_type": source_type,
                "source_id": source_id,
                "provenance": {
                    "extraction": extraction_label,
                    "processed_at": datetime.now().isoformat()
                }
            }

            # Add original names if different
            if resolved_source != source:
                rel_processed["original_source"] = source
            if resolved_target != target:
                rel_processed["original_target"] = target

            processed.append(rel_processed)

        return processed

    def process_all(self, dry_run: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all batch results through the pipeline.

        Returns:
            Tuple of (all_entities, all_relationships)
        """
        print("Loading batch results...")
        batch_results = self.load_batch_results()
        print(f"  Loaded {len(batch_results)} chunk results")

        print("Loading parent chunks...")
        parent_chunks = self.load_parent_chunks()
        print(f"  Loaded {len(parent_chunks)} parent chunks")

        all_entities = []
        all_relationships = []
        entity_names_lower = set()

        print("\nProcessing chunks...")
        for i, (chunk_id, extraction) in enumerate(batch_results.items()):
            self.stats["total_chunks_processed"] += 1

            # Get parent chunk for context
            parent_chunk = parent_chunks.get(chunk_id, {
                "source_type": "unknown",
                "source_id": chunk_id
            })

            # Process entities
            entities, name_mapping = self.process_entities(
                extraction.get("entities", []),
                parent_chunk
            )
            all_entities.extend(entities)

            # Track valid entity names for relationship validation
            for entity in entities:
                entity_names_lower.add(entity["name"].lower())

            # Process relationships
            relationships = self.process_relationships(
                extraction.get("relationships", []),
                name_mapping,
                entity_names_lower,
                parent_chunk
            )
            all_relationships.extend(relationships)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(batch_results)} chunks...")

        print(f"  Processed {len(batch_results)} chunks total")

        if not dry_run:
            self._save_results(all_entities, all_relationships)

        return all_entities, all_relationships

    def _save_results(self, entities: List[Dict], relationships: List[Dict]):
        """Save processed entities and relationships."""
        # Save entities
        entities_path = self.output_dir / "entities_processed.json"
        with open(entities_path, 'w') as f:
            json.dump(entities, f, indent=2)
        print(f"\nSaved {len(entities)} entities to: {entities_path}")

        # Save relationships
        relationships_path = self.output_dir / "relationships_processed.json"
        with open(relationships_path, 'w') as f:
            json.dump(relationships, f, indent=2)
        print(f"Saved {len(relationships)} relationships to: {relationships_path}")

        # Save stats
        stats_path = self.output_dir / "processing_stats.json"
        # Convert defaultdict to regular dict for JSON serialization
        stats_to_save = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in self.stats.items()
        }
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"Saved stats to: {stats_path}")

    def print_stats(self):
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("PROCESSING STATISTICS")
        print("=" * 50)

        print(f"\nChunks processed: {self.stats['total_chunks_processed']}")

        print(f"\nEntities:")
        print(f"  Raw extracted: {self.stats['total_entities_raw']}")
        print(f"  After quality filter: {self.stats['entities_after_quality_filter']}")
        print(f"  After list split: {self.stats['entities_after_list_split']}")
        print(f"  After resolution: {self.stats['entities_after_resolution']}")
        print(f"  Fictional tagged: {self.stats['entities_fictional_tagged']}")

        print(f"\nRelationships:")
        print(f"  Raw extracted: {self.stats['total_relationships_raw']}")
        print(f"  Valid (both entities exist): {self.stats['relationships_valid']}")
        print(f"  Orphaned (missing entity): {self.stats['relationships_orphaned']}")

        if self.stats["filter_reasons"]:
            print(f"\nFilter rejection reasons:")
            sorted_reasons = sorted(
                self.stats["filter_reasons"].items(),
                key=lambda x: -x[1]
            )
            for reason, count in sorted_reasons:
                print(f"  {reason}: {count}")

        if self.stats["resolution_methods"]:
            print(f"\nResolution methods:")
            sorted_methods = sorted(
                self.stats["resolution_methods"].items(),
                key=lambda x: -x[1]
            )
            for method, count in sorted_methods:
                print(f"  {method}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Process batch extraction results")
    parser.add_argument("--dry-run", action="store_true", help="Process without saving")
    parser.add_argument("--stats-only", action="store_true", help="Only show stats from previous run")
    parser.add_argument("--results-path", type=str, help="Path to extraction_results.json")
    parser.add_argument("--chunks-path", type=str, help="Path to parent_chunks.json")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed files")

    args = parser.parse_args()

    if args.stats_only:
        stats_path = Path("data/knowledge_graph_unified/processing_stats.json")
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            print(json.dumps(stats, indent=2))
        else:
            print("No stats file found. Run processing first.")
        return

    processor = BatchResultProcessor(
        batch_results_path=Path(args.results_path) if args.results_path else None,
        parent_chunks_path=Path(args.chunks_path) if args.chunks_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    try:
        entities, relationships = processor.process_all(dry_run=args.dry_run)
        processor.print_stats()

        if args.dry_run:
            print("\n[DRY RUN - No files saved]")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have run the batch extraction first:")
        print("  python scripts/extract_episodes_batch.py --download")
        sys.exit(1)


if __name__ == "__main__":
    main()
