#!/usr/bin/env python3
"""
Apply Entity Normalization to Knowledge Graph

This script performs three-pass entity normalization to reduce entity count by 10-15%:
1. Pass 1: Exact key matching from existing alias_map
2. Pass 2: Apply normalization map from triage report
3. Pass 3: Fuzzy matching (Jaccard >0.85 or Levenshtein ≤3)

Key features:
- Max 100 merges per root entity (prevent super-nodes)
- Generate merges.json audit trail
- Preserve all relationships during merge
- Update both unified.json and adjacency.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import hashlib
from datetime import datetime
import Levenshtein
import re
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path("/home/claudeuser/yonearth-gaia-chatbot")
UNIFIED_KG_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/unified.json"
ADJACENCY_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/adjacency.json"
STATS_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/stats.json"
TRIAGE_REPORT_PATH = PROJECT_ROOT / "data/knowledge_graph_unified/orphan_triage_report.json"
OUTPUT_DIR = PROJECT_ROOT / "data/knowledge_graph_unified"

# Output files
MERGES_LOG_PATH = OUTPUT_DIR / "entity_merges.json"
NORMALIZED_UNIFIED_PATH = OUTPUT_DIR / "unified_normalized.json"
NORMALIZED_ADJACENCY_PATH = OUTPUT_DIR / "adjacency_normalized.json"
NORMALIZED_STATS_PATH = OUTPUT_DIR / "stats_normalized.json"

class EntityNormalizer:
    def __init__(self):
        self.entities = {}
        self.relationships = []
        self.adjacency = defaultdict(lambda: defaultdict(list))
        self.merges = {}  # old_entity -> canonical_entity
        self.merge_counts = defaultdict(int)  # canonical -> count of merges
        self.audit_trail = []
        self.stats = {
            "original_entity_count": 0,
            "normalized_entity_count": 0,
            "total_merges": 0,
            "pass1_merges": 0,
            "pass2_merges": 0,
            "pass3_merges": 0,
            "relationship_updates": 0,
            "timestamp": datetime.now().isoformat()
        }

    def load_knowledge_graph(self):
        """Load the current unified knowledge graph"""
        print("Loading knowledge graph...")

        # Load unified graph
        with open(UNIFIED_KG_PATH, 'r') as f:
            data = json.load(f)
            self.entities = data['entities']
            self.relationships = data['relationships']

        # Load adjacency
        with open(ADJACENCY_PATH, 'r') as f:
            adj_data = json.load(f)
            for source, targets in adj_data.items():
                for target, rels in targets.items():
                    self.adjacency[source][target] = rels

        # Load stats
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
            self.stats["original_entity_count"] = stats.get("entity_count", len(self.entities))

        print(f"Loaded {len(self.entities)} entities and {len(self.relationships)} relationships")

    def load_triage_report(self):
        """Load the orphan triage report for normalization suggestions"""
        if TRIAGE_REPORT_PATH.exists():
            with open(TRIAGE_REPORT_PATH, 'r') as f:
                return json.load(f)
        return {}

    def normalize_string(self, text):
        """Normalize a string for comparison"""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove punctuation except for meaningful ones
        text = re.sub(r'[^\w\s\-\'&]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def calculate_jaccard_similarity(self, str1, str2):
        """Calculate Jaccard similarity between two strings"""
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union) if union else 0.0

    def pass1_exact_alias_matching(self):
        """Pass 1: Exact key matching from existing alias_map"""
        print("\n=== Pass 1: Exact Alias Matching ===")

        # Build alias map from entities that have aliases
        alias_map = {}
        for entity_name, entity_data in self.entities.items():
            if 'aliases' in entity_data and entity_data['aliases']:
                for alias in entity_data['aliases']:
                    alias_lower = alias.lower()
                    if alias_lower != entity_name.lower():
                        alias_map[alias_lower] = entity_name

        print(f"Found {len(alias_map)} existing aliases")

        # Check each entity against the alias map
        merges = []
        for entity_name in list(self.entities.keys()):
            entity_lower = entity_name.lower()
            if entity_lower in alias_map:
                canonical = alias_map[entity_lower]
                if canonical != entity_name and canonical in self.entities:
                    merges.append((entity_name, canonical, "exact_alias"))

        # Apply merges
        for old_entity, canonical, method in merges:
            if self.merge_entities(old_entity, canonical, method):
                self.stats["pass1_merges"] += 1

        print(f"Pass 1 completed: {self.stats['pass1_merges']} entities merged")

    def pass2_triage_normalization(self):
        """Pass 2: Apply normalization map from triage report"""
        print("\n=== Pass 2: Triage Report Normalization ===")

        triage_report = self.load_triage_report()

        if not triage_report:
            print("No triage report found, skipping Pass 2")
            return

        # Apply recommended alias mappings
        alias_mappings = triage_report.get('recommended_fixes', {}).get('alias_mappings', {})

        merges = []
        for old_entity, canonical in alias_mappings.items():
            if old_entity in self.entities and canonical in self.entities:
                merges.append((old_entity, canonical, "triage_alias"))

        # Apply alias suggestions with high confidence
        for suggestion in triage_report.get('alias_suggestions', []):
            missing = suggestion['missing']
            if suggestion['candidates']:
                best_candidate, similarity = suggestion['candidates'][0]
                # Convert percentage string to float
                sim_score = float(similarity.rstrip('%')) / 100

                # Only merge if similarity is high enough
                if sim_score >= 0.75:  # 75% threshold
                    if best_candidate in self.entities:
                        # Create the missing entity if needed
                        if missing not in self.entities:
                            self.entities[missing] = {
                                "type": self.entities[best_candidate].get('type', 'CONCEPT'),
                                "auto_created": True,
                                "sources": []
                            }
                        merges.append((missing, best_candidate, f"triage_suggestion_{sim_score:.2f}"))

        # Apply merges
        for old_entity, canonical, method in merges:
            if self.merge_entities(old_entity, canonical, method):
                self.stats["pass2_merges"] += 1

        print(f"Pass 2 completed: {self.stats['pass2_merges']} entities merged")

    def pass3_fuzzy_matching(self):
        """Pass 3: Fuzzy matching (Jaccard >0.85 or Levenshtein ≤3)"""
        print("\n=== Pass 3: Fuzzy Matching ===")

        # Group entities by type for more efficient comparison
        entities_by_type = defaultdict(list)
        for entity_name, entity_data in self.entities.items():
            entity_type = entity_data.get('type', 'UNKNOWN')
            entities_by_type[entity_type].append(entity_name)

        merges = []
        processed = set()

        for entity_type, entity_list in tqdm(entities_by_type.items(), desc="Processing entity types"):
            # Only compare entities of the same type
            for i, entity1 in enumerate(entity_list):
                if entity1 in processed or entity1 not in self.entities:
                    continue

                entity1_norm = self.normalize_string(entity1)

                for entity2 in entity_list[i+1:]:
                    if entity2 in processed or entity2 not in self.entities:
                        continue

                    entity2_norm = self.normalize_string(entity2)

                    # Skip if already the same after normalization
                    if entity1_norm == entity2_norm:
                        merges.append((entity2, entity1, "normalized_exact"))
                        processed.add(entity2)
                        continue

                    # Check Levenshtein distance
                    lev_dist = Levenshtein.distance(entity1_norm, entity2_norm)
                    if lev_dist <= 3:
                        # Choose the canonical form (prefer the one with more evidence)
                        canonical = self.choose_canonical(entity1, entity2)
                        other = entity2 if canonical == entity1 else entity1
                        merges.append((other, canonical, f"levenshtein_{lev_dist}"))
                        processed.add(other)
                        continue

                    # Check Jaccard similarity
                    jaccard_sim = self.calculate_jaccard_similarity(entity1, entity2)
                    if jaccard_sim > 0.85:
                        canonical = self.choose_canonical(entity1, entity2)
                        other = entity2 if canonical == entity1 else entity1
                        merges.append((other, canonical, f"jaccard_{jaccard_sim:.2f}"))
                        processed.add(other)

        # Apply merges
        for old_entity, canonical, method in merges:
            if self.merge_entities(old_entity, canonical, method):
                self.stats["pass3_merges"] += 1

        print(f"Pass 3 completed: {self.stats['pass3_merges']} entities merged")

    def choose_canonical(self, entity1, entity2):
        """Choose the canonical entity based on evidence count and format"""
        # Safety check - if either entity doesn't exist, return the one that does
        if entity1 not in self.entities:
            return entity2
        if entity2 not in self.entities:
            return entity1

        e1_data = self.entities[entity1]
        e2_data = self.entities[entity2]

        # Prefer the one with more evidence
        e1_evidence = len(e1_data.get('evidence', []))
        e2_evidence = len(e2_data.get('evidence', []))

        if e1_evidence != e2_evidence:
            return entity1 if e1_evidence > e2_evidence else entity2

        # Prefer the one with proper capitalization (more uppercase letters)
        e1_caps = sum(1 for c in entity1 if c.isupper())
        e2_caps = sum(1 for c in entity2 if c.isupper())

        if e1_caps != e2_caps:
            return entity1 if e1_caps > e2_caps else entity2

        # Default to the first one alphabetically
        return min(entity1, entity2)

    def merge_entities(self, old_entity, canonical_entity, method):
        """Merge old_entity into canonical_entity"""

        # Check merge limit (prevent super-nodes)
        if self.merge_counts[canonical_entity] >= 100:
            return False

        # Skip if already merged or same entity
        if old_entity == canonical_entity:
            return False
        if old_entity in self.merges:
            return False

        # Perform the merge
        if old_entity in self.entities and canonical_entity in self.entities:
            old_data = self.entities[old_entity]
            canonical_data = self.entities[canonical_entity]

            # Merge evidence
            old_evidence = old_data.get('evidence', [])
            canonical_evidence = canonical_data.get('evidence', [])
            canonical_data['evidence'] = canonical_evidence + old_evidence

            # Merge aliases
            old_aliases = old_data.get('aliases', [])
            canonical_aliases = canonical_data.get('aliases', [])
            # Add the old entity name as an alias
            all_aliases = set(canonical_aliases + old_aliases + [old_entity])
            # Remove the canonical name from aliases
            all_aliases.discard(canonical_entity)
            canonical_data['aliases'] = list(all_aliases)

            # Merge sources
            old_sources = old_data.get('sources', [])
            canonical_sources = canonical_data.get('sources', [])
            all_sources = list(set(canonical_sources + old_sources))
            canonical_data['sources'] = all_sources

            # Update relationships
            for rel in self.relationships:
                updated = False
                if rel['source'] == old_entity:
                    rel['source'] = canonical_entity
                    updated = True
                if rel['target'] == old_entity:
                    rel['target'] = canonical_entity
                    updated = True
                if updated:
                    self.stats["relationship_updates"] += 1

            # Update adjacency
            # Replace old entity in adjacency lists
            if old_entity in self.adjacency:
                for target, rels in self.adjacency[old_entity].items():
                    if canonical_entity not in self.adjacency:
                        self.adjacency[canonical_entity] = {}
                    if target not in self.adjacency[canonical_entity]:
                        self.adjacency[canonical_entity][target] = []
                    self.adjacency[canonical_entity][target].extend(rels)
                del self.adjacency[old_entity]

            # Update references to old entity in other adjacency entries
            for source in self.adjacency:
                if old_entity in self.adjacency[source]:
                    rels = self.adjacency[source][old_entity]
                    if canonical_entity not in self.adjacency[source]:
                        self.adjacency[source][canonical_entity] = []
                    self.adjacency[source][canonical_entity].extend(rels)
                    del self.adjacency[source][old_entity]

            # Remove the old entity
            del self.entities[old_entity]

            # Record the merge
            self.merges[old_entity] = canonical_entity
            self.merge_counts[canonical_entity] += 1
            self.stats["total_merges"] += 1

            # Add to audit trail
            self.audit_trail.append({
                "old_entity": old_entity,
                "canonical_entity": canonical_entity,
                "method": method,
                "timestamp": datetime.now().isoformat()
            })

            return True

        return False

    def remove_duplicate_relationships(self):
        """Remove duplicate relationships after merging"""
        print("\nRemoving duplicate relationships...")

        seen = set()
        unique_relationships = []

        for rel in self.relationships:
            # Skip self-loops unless they're meaningful
            if rel['source'] == rel['target']:
                if rel.get('predicate') not in ['part_of', 'includes', 'supports']:
                    continue

            # Create a unique key for the relationship
            rel_key = (rel['source'], rel['predicate'], rel['target'])

            if rel_key not in seen:
                seen.add(rel_key)
                unique_relationships.append(rel)

        removed = len(self.relationships) - len(unique_relationships)
        self.relationships = unique_relationships

        print(f"Removed {removed} duplicate relationships")

    def rebuild_adjacency(self):
        """Rebuild adjacency from relationships"""
        print("\nRebuilding adjacency list...")

        self.adjacency = defaultdict(lambda: defaultdict(list))

        for rel in self.relationships:
            source = rel['source']
            target = rel['target']
            predicate = rel['predicate']

            self.adjacency[source][target].append({
                "predicate": predicate,
                "confidence": rel.get('confidence', 1.0),
                "evidence_count": len(rel.get('evidence', []))
            })

        print(f"Adjacency rebuilt with {len(self.adjacency)} source nodes")

    def save_results(self):
        """Save normalized graph and audit trail"""
        print("\nSaving results...")

        # Update final stats
        self.stats["normalized_entity_count"] = len(self.entities)
        self.stats["normalized_relationship_count"] = len(self.relationships)
        self.stats["entity_reduction"] = self.stats["original_entity_count"] - self.stats["normalized_entity_count"]
        self.stats["reduction_percentage"] = (self.stats["entity_reduction"] / self.stats["original_entity_count"] * 100) if self.stats["original_entity_count"] > 0 else 0

        # Save normalized unified graph
        unified_output = {
            "entities": self.entities,
            "relationships": self.relationships,
            "metadata": {
                "normalization_applied": True,
                "timestamp": self.stats["timestamp"],
                "original_count": self.stats["original_entity_count"],
                "normalized_count": self.stats["normalized_entity_count"]
            }
        }

        with open(NORMALIZED_UNIFIED_PATH, 'w') as f:
            json.dump(unified_output, f, indent=2)
        print(f"Saved normalized unified graph to {NORMALIZED_UNIFIED_PATH}")

        # Save normalized adjacency
        # Convert defaultdict to regular dict for JSON serialization
        adjacency_output = {source: dict(targets) for source, targets in self.adjacency.items()}

        with open(NORMALIZED_ADJACENCY_PATH, 'w') as f:
            json.dump(adjacency_output, f, indent=2)
        print(f"Saved normalized adjacency to {NORMALIZED_ADJACENCY_PATH}")

        # Save merge audit trail
        merge_output = {
            "stats": self.stats,
            "merges": self.merges,
            "merge_counts": dict(self.merge_counts),
            "audit_trail": self.audit_trail
        }

        with open(MERGES_LOG_PATH, 'w') as f:
            json.dump(merge_output, f, indent=2)
        print(f"Saved merge audit trail to {MERGES_LOG_PATH}")

        # Save updated stats
        with open(NORMALIZED_STATS_PATH, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Saved stats to {NORMALIZED_STATS_PATH}")

    def run(self):
        """Run the complete normalization pipeline"""
        print("=" * 60)
        print("Entity Normalization Pipeline")
        print("=" * 60)

        # Load data
        self.load_knowledge_graph()

        # Run three normalization passes
        self.pass1_exact_alias_matching()
        self.pass2_triage_normalization()
        self.pass3_fuzzy_matching()

        # Clean up
        self.remove_duplicate_relationships()
        self.rebuild_adjacency()

        # Save results
        self.save_results()

        # Print summary
        print("\n" + "=" * 60)
        print("Normalization Summary")
        print("=" * 60)
        print(f"Original entities: {self.stats['original_entity_count']:,}")
        print(f"Normalized entities: {self.stats['normalized_entity_count']:,}")
        print(f"Total merges: {self.stats['total_merges']:,}")
        print(f"  - Pass 1 (exact alias): {self.stats['pass1_merges']:,}")
        print(f"  - Pass 2 (triage): {self.stats['pass2_merges']:,}")
        print(f"  - Pass 3 (fuzzy): {self.stats['pass3_merges']:,}")
        print(f"Entity reduction: {self.stats['entity_reduction']:,} ({self.stats['reduction_percentage']:.1f}%)")
        print(f"Relationship updates: {self.stats['relationship_updates']:,}")

        # Check if we met the target
        if self.stats['reduction_percentage'] >= 10:
            print("\n✅ Successfully achieved 10%+ entity reduction!")
        else:
            print(f"\n⚠️ Only achieved {self.stats['reduction_percentage']:.1f}% reduction (target: 10-15%)")


if __name__ == "__main__":
    normalizer = EntityNormalizer()
    normalizer.run()