#!/usr/bin/env python3
"""
Relationship Normalization and Graph Embedding System

This script normalizes the raw relationship types extracted by GPT-4o-mini
into a hierarchical ontology for better querying and analysis.

Architecture:
- Raw (300+ types) → Domain (100) → Canonical (50) → Abstract (10-15)
- Preserves nuance while enabling broad queries
- Adds embedding support for semantic similarity
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RelationshipNormalizer:
    """Normalizes and enriches relationship types with hierarchical ontology"""

    # Abstract level (highest - 10-15 types)
    ABSTRACT_RELATIONSHIPS = {
        "CREATES": "Entity brings something into existence",
        "INFLUENCES": "Entity affects or shapes another",
        "ASSOCIATES": "Entities have connection or affiliation",
        "LOCATES": "Entity has spatial relationship",
        "TEMPORALIZES": "Entity has time-based relationship",
        "ATTRIBUTES": "Entity has property or characteristic",
        "TRANSFORMS": "Entity changes state or form",
        "EXCHANGES": "Entities trade or transfer something",
        "HIERARCHIZES": "Entities have parent-child or rank relation",
        "DEPENDS": "Entity requires or relies on another"
    }

    # Canonical level (middle - ~50 types)
    CANONICAL_RELATIONSHIPS = {
        # Creation cluster
        "FOUNDED": ["FOUNDED", "ESTABLISHED", "CREATED", "INITIATED", "LAUNCHED", "STARTED", "BUILT"],
        "DEVELOPS": ["DEVELOPS", "BUILDS", "CONSTRUCTS", "DESIGNS", "IMPLEMENTS", "GROWS"],
        "PRODUCES": ["PRODUCES", "MAKES", "GENERATES", "CREATES", "MANUFACTURES", "YIELDS"],

        # Influence cluster
        "INSPIRED_BY": ["INSPIRED_BY", "INFLUENCED_BY", "MOTIVATED_BY", "LIGHTED_FIRE_IN",
                        "PROVIDED_EXPOSURE_TO", "SHAPED_BY"],
        "TEACHES": ["TEACHES", "EDUCATES", "TRAINS", "MENTORS", "GUIDES", "INSTRUCTS"],
        "ADVOCATES_FOR": ["ADVOCATES_FOR", "PROMOTES", "CHAMPIONS", "SUPPORTS", "ENCOURAGES"],

        # Association cluster
        "WORKS_FOR": ["WORKS_FOR", "EMPLOYED_BY", "SERVES", "WORKS_AT", "WORKS_WITH"],
        "COLLABORATES_WITH": ["COLLABORATES_WITH", "PARTNERS_WITH", "WORKS_WITH",
                              "COOPERATES_WITH", "TEAMS_WITH"],
        "MEMBER_OF": ["MEMBER_OF", "BELONGS_TO", "PART_OF", "AFFILIATED_WITH", "ASSOCIATED_WITH"],

        # Location cluster
        "LOCATED_IN": ["LOCATED_IN", "BASED_IN", "SITUATED_IN", "FOUND_IN", "RESIDES_IN"],
        "LIVES_IN": ["LIVES_IN", "RESIDES_IN", "INHABITS", "DWELLS_IN", "SETTLED_IN"],

        # Temporal cluster
        "STARTED_IN": ["STARTED_IN", "BEGAN_IN", "INITIATED_IN", "COMMENCED_IN"],
        "FOUNDED_IN": ["FOUNDED_IN", "ESTABLISHED_IN", "CREATED_IN", "FORMED_IN"],
        "OCCURRED_IN": ["OCCURRED_IN", "HAPPENED_IN", "TOOK_PLACE_IN", "TRANSPIRED_IN"],

        # Attribute cluster
        "HAS_PROPERTY": ["HAS_SIZE", "HAS_DURATION", "HAS_COST", "HAS_QUANTITY",
                         "HAS_PERCENTAGE", "HAS_AGE", "HAS_FEATURE"],
        "HAS_ROLE": ["SERVES_AS", "ACTS_AS", "FUNCTIONS_AS", "OPERATES_AS"],

        # Exchange cluster
        "PROVIDES": ["PROVIDES", "SUPPLIES", "DELIVERS", "OFFERS", "FURNISHES", "DISTRIBUTES"],
        "SELLS": ["SELLS", "MARKETS", "TRADES", "VENDS", "RETAILS"],
        "FUNDED_BY": ["FUNDED_BY", "SPONSORED_BY", "FINANCED_BY", "SUPPORTED_BY", "BACKED_BY"],

        # Transformation cluster
        "TRANSFORMS_INTO": ["TRANSFORMS_INTO", "BECOMES", "CHANGES_TO", "EVOLVES_INTO", "CONVERTS_TO"],
        "AFFECTS": ["AFFECTS", "IMPACTS", "INFLUENCES", "MODIFIES", "ALTERS"],

        # Hierarchical cluster
        "OWNS": ["OWNS", "POSSESSES", "CONTROLS", "HOLDS", "HAS"],
        "MANAGES": ["MANAGES", "DIRECTS", "LEADS", "OVERSEES", "ADMINISTERS", "SUPERVISES"],
        "CONTAINS": ["CONTAINS", "INCLUDES", "COMPRISES", "ENCOMPASSES", "INCORPORATES"],

        # Dependency cluster
        "REQUIRES": ["REQUIRES", "NEEDS", "DEPENDS_ON", "RELIES_ON", "NECESSITATES"],
        "ENABLES": ["ENABLES", "ALLOWS", "FACILITATES", "PERMITS", "EMPOWERS"],
        "USES": ["USES", "EMPLOYS", "UTILIZES", "APPLIES", "LEVERAGES"]
    }

    # Mapping from canonical to abstract
    CANONICAL_TO_ABSTRACT = {
        "FOUNDED": "CREATES",
        "DEVELOPS": "CREATES",
        "PRODUCES": "CREATES",
        "INSPIRED_BY": "INFLUENCES",
        "TEACHES": "INFLUENCES",
        "ADVOCATES_FOR": "INFLUENCES",
        "WORKS_FOR": "ASSOCIATES",
        "COLLABORATES_WITH": "ASSOCIATES",
        "MEMBER_OF": "HIERARCHIZES",
        "LOCATED_IN": "LOCATES",
        "LIVES_IN": "LOCATES",
        "STARTED_IN": "TEMPORALIZES",
        "FOUNDED_IN": "TEMPORALIZES",
        "OCCURRED_IN": "TEMPORALIZES",
        "HAS_PROPERTY": "ATTRIBUTES",
        "HAS_ROLE": "ATTRIBUTES",
        "PROVIDES": "EXCHANGES",
        "SELLS": "EXCHANGES",
        "FUNDED_BY": "EXCHANGES",
        "TRANSFORMS_INTO": "TRANSFORMS",
        "AFFECTS": "TRANSFORMS",
        "OWNS": "HIERARCHIZES",
        "MANAGES": "HIERARCHIZES",
        "CONTAINS": "HIERARCHIZES",
        "REQUIRES": "DEPENDS",
        "ENABLES": "DEPENDS",
        "USES": "DEPENDS"
    }

    def __init__(self, use_embeddings: bool = False, api_key: str = None):
        """
        Initialize normalizer

        Args:
            use_embeddings: Whether to use OpenAI embeddings for similarity
            api_key: OpenAI API key for embeddings
        """
        self.use_embeddings = use_embeddings
        self.canonical_map_cache = {}
        self.embedding_cache = {}

        if use_embeddings:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required for embeddings")
            self.client = OpenAI(api_key=self.api_key)
            self._precompute_canonical_embeddings()

    def _precompute_canonical_embeddings(self):
        """Precompute embeddings for all canonical relationship types"""
        print("Precomputing canonical relationship embeddings...")
        for canonical in self.CANONICAL_RELATIONSHIPS.keys():
            if canonical not in self.embedding_cache:
                self.embedding_cache[canonical] = self._get_embedding(canonical)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text.replace("_", " ").lower()
        )
        embedding = np.array(response.data[0].embedding)
        self.embedding_cache[text] = embedding
        return embedding

    def normalize_relationship(self, raw_type: str,
                             use_similarity: bool = None) -> Dict[str, str]:
        """
        Normalize a raw relationship type to canonical and abstract levels

        Args:
            raw_type: The raw relationship type from extraction
            use_similarity: Override default embedding behavior

        Returns:
            Dictionary with normalized types at different levels
        """
        raw_upper = raw_type.upper()

        # Check cache
        if raw_upper in self.canonical_map_cache:
            canonical = self.canonical_map_cache[raw_upper]
        else:
            # Try exact match first
            canonical = None
            for canon_type, variants in self.CANONICAL_RELATIONSHIPS.items():
                if raw_upper in [v.upper() for v in variants]:
                    canonical = canon_type
                    break

            # If no exact match and embeddings enabled, use similarity
            if not canonical and (use_similarity or
                                (use_similarity is None and self.use_embeddings)):
                canonical = self._find_similar_canonical(raw_type)

            # Default to RELATED_TO if no match
            if not canonical:
                canonical = "RELATED_TO" if "RELATE" in raw_upper else raw_upper

            self.canonical_map_cache[raw_upper] = canonical

        # Get abstract level
        abstract = self.CANONICAL_TO_ABSTRACT.get(canonical, "ASSOCIATES")

        return {
            "raw": raw_type,
            "canonical": canonical,
            "abstract": abstract,
            "confidence": 1.0 if canonical in self.CANONICAL_RELATIONSHIPS else 0.7
        }

    def _find_similar_canonical(self, raw_type: str, threshold: float = 0.7) -> Optional[str]:
        """Find most similar canonical type using embeddings"""
        raw_embedding = self._get_embedding(raw_type)

        best_match = None
        best_score = 0

        for canonical, canonical_embedding in self.embedding_cache.items():
            if canonical in self.CANONICAL_RELATIONSHIPS:  # Only canonical types
                similarity = cosine_similarity(
                    raw_embedding.reshape(1, -1),
                    canonical_embedding.reshape(1, -1)
                )[0][0]

                if similarity > best_score:
                    best_score = similarity
                    best_match = canonical

        return best_match if best_score >= threshold else None

    def analyze_corpus(self, relationships_dir: Path) -> Dict:
        """
        Analyze all extracted relationships to understand the corpus

        Returns statistics and mapping recommendations
        """
        raw_types = Counter()
        entity_type_pairs = Counter()

        files = list(relationships_dir.glob("episode_*_extraction.json"))
        print(f"Analyzing {len(files)} extraction files...")

        for file_path in files:
            with open(file_path) as f:
                data = json.load(f)
                for rel in data.get("relationships", []):
                    raw_type = rel.get("relationship_type", "UNKNOWN")
                    raw_types[raw_type] += 1

                    # Track entity type pairs
                    source_type = rel.get("source_type", "?")
                    target_type = rel.get("target_type", "?")
                    entity_type_pairs[(source_type, raw_type, target_type)] += 1

        print(f"Found {len(raw_types)} unique relationship types")

        # Normalize all types
        normalized_mapping = {}
        for raw_type in raw_types:
            normalized = self.normalize_relationship(raw_type)
            normalized_mapping[raw_type] = normalized

        # Compute statistics
        canonical_counts = Counter()
        abstract_counts = Counter()
        unmapped_types = []

        for raw_type, norm in normalized_mapping.items():
            canonical_counts[norm["canonical"]] += raw_types[raw_type]
            abstract_counts[norm["abstract"]] += raw_types[raw_type]
            if norm["confidence"] < 1.0:
                unmapped_types.append((raw_type, raw_types[raw_type]))

        return {
            "total_relationships": sum(raw_types.values()),
            "unique_raw_types": len(raw_types),
            "unique_canonical_types": len(canonical_counts),
            "unique_abstract_types": len(abstract_counts),
            "top_raw_types": raw_types.most_common(20),
            "top_canonical_types": canonical_counts.most_common(15),
            "top_abstract_types": abstract_counts.most_common(10),
            "unmapped_types": sorted(unmapped_types, key=lambda x: -x[1])[:20],
            "entity_type_patterns": entity_type_pairs.most_common(20),
            "normalization_map": normalized_mapping
        }

    def process_extraction_file(self, file_path: Path, output_path: Path = None):
        """
        Process a single extraction file and add normalized relationships
        """
        with open(file_path) as f:
            data = json.load(f)

        # Normalize each relationship
        enriched_relationships = []
        for rel in data.get("relationships", []):
            normalized = self.normalize_relationship(rel.get("relationship_type", ""))

            # Add normalized fields
            enriched_rel = rel.copy()
            enriched_rel["relationship_canonical"] = normalized["canonical"]
            enriched_rel["relationship_abstract"] = normalized["abstract"]
            enriched_rel["normalization_confidence"] = normalized["confidence"]

            enriched_relationships.append(enriched_rel)

        data["relationships"] = enriched_relationships

        # Save enriched version
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        return data

    def process_all_extractions(self, relationships_dir: Path, output_dir: Path):
        """
        Process all extraction files and save normalized versions
        """
        files = sorted(relationships_dir.glob("episode_*_extraction.json"))
        print(f"Processing {len(files)} extraction files...")

        for i, file_path in enumerate(files, 1):
            output_path = output_dir / f"normalized_{file_path.name}"
            self.process_extraction_file(file_path, output_path)

            if i % 10 == 0:
                print(f"  Processed {i}/{len(files)} files...")

        print(f"✅ Normalized relationships saved to {output_dir}")


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Normalize extracted relationships")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze corpus and show statistics")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize all extraction files")
    parser.add_argument("--use-embeddings", action="store_true",
                       help="Use OpenAI embeddings for similarity matching")
    parser.add_argument("--input-dir", type=str,
                       default="data/knowledge_graph/relationships",
                       help="Input directory with raw extractions")
    parser.add_argument("--output-dir", type=str,
                       default="data/knowledge_graph/relationships_normalized",
                       help="Output directory for normalized extractions")

    args = parser.parse_args()

    # Initialize normalizer
    normalizer = RelationshipNormalizer(use_embeddings=args.use_embeddings)

    relationships_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.analyze:
        print("=" * 80)
        print("RELATIONSHIP CORPUS ANALYSIS")
        print("=" * 80)

        stats = normalizer.analyze_corpus(relationships_dir)

        print(f"\n📊 STATISTICS:")
        print(f"  Total relationships: {stats['total_relationships']:,}")
        print(f"  Unique raw types: {stats['unique_raw_types']}")
        print(f"  Unique canonical types: {stats['unique_canonical_types']}")
        print(f"  Unique abstract types: {stats['unique_abstract_types']}")

        print(f"\n🔝 TOP RAW TYPES:")
        for rtype, count in stats['top_raw_types']:
            norm = stats['normalization_map'][rtype]
            print(f"  {rtype}: {count} → {norm['canonical']} → {norm['abstract']}")

        print(f"\n🎯 TOP CANONICAL TYPES:")
        for ctype, count in stats['top_canonical_types']:
            print(f"  {ctype}: {count}")

        print(f"\n🌐 TOP ABSTRACT TYPES:")
        for atype, count in stats['top_abstract_types']:
            print(f"  {atype}: {count}")

        if stats['unmapped_types']:
            print(f"\n⚠️  UNMAPPED TYPES (need review):")
            for utype, count in stats['unmapped_types'][:10]:
                print(f"  {utype}: {count}")

        print(f"\n🔗 TOP ENTITY TYPE PATTERNS:")
        for (source, rel, target), count in stats['entity_type_patterns']:
            print(f"  {source} --{rel}--> {target}: {count}")

    if args.normalize:
        print("=" * 80)
        print("NORMALIZING RELATIONSHIPS")
        print("=" * 80)

        normalizer.process_all_extractions(relationships_dir, output_dir)

        print("\n✅ Normalization complete!")
        print(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()