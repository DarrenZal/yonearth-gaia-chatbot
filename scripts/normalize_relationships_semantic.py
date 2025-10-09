#!/usr/bin/env python3
"""
Semantic Relationship Normalization with Rich Domain Ontology

This script normalizes raw relationship types into a semantically-rich
hierarchical ontology that preserves domain knowledge while enabling
sophisticated graph queries.

Architecture:
- Raw (837+ types) → Domain (150 semantic) → Canonical (50) → Abstract (10-15)
- Domain layer is the KEY semantic intelligence layer for NL→Graph queries
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SemanticRelationshipNormalizer:
    """Normalizes relationships with rich semantic domain ontology"""

    # ============================================
    # DOMAIN ONTOLOGY - The Semantic Intelligence Layer
    # ============================================

    DOMAIN_ONTOLOGY = {
        # === EDUCATIONAL & MENTORSHIP ===
        "TEACHES": {
            "raw_patterns": ["TEACH", "INSTRUCT", "EDUCATE", "TRAIN"],
            "canonical": "EDUCATES",
            "abstract": "INFLUENCES",
            "properties": {"implies_expertise": True, "transfer_knowledge": True}
        },
        "MENTORS": {
            "raw_patterns": ["MENTOR", "GUIDE", "COACH", "ADVISE", "COUNSEL"],
            "canonical": "EDUCATES",
            "abstract": "INFLUENCES",
            "properties": {"implies_seniority": True, "personal_guidance": True}
        },
        "LEARNS_FROM": {
            "raw_patterns": ["LEARN", "STUDY", "TRAINED_BY", "TAUGHT_BY"],
            "canonical": "EDUCATES",
            "abstract": "INFLUENCES",
            "properties": {"reverse_of": "TEACHES"}
        },

        # === FINANCIAL & ECONOMIC ===
        "FUNDS": {
            "raw_patterns": ["FUND", "FINANCE", "BANKROLL", "SUBSIDIZE"],
            "canonical": "FINANCES",
            "abstract": "EXCHANGES",
            "properties": {"has_amount": True, "financial": True}
        },
        "SPONSORS": {
            "raw_patterns": ["SPONSOR", "BACK", "SUPPORT_FINANCIALLY"],
            "canonical": "FINANCES",
            "abstract": "EXCHANGES",
            "properties": {"has_duration": True, "marketing_aspect": True}
        },
        "INVESTS_IN": {
            "raw_patterns": ["INVEST", "STAKE", "CAPITALIZE"],
            "canonical": "FINANCES",
            "abstract": "EXCHANGES",
            "properties": {"expects_return": True, "has_risk": True}
        },
        "DONATES_TO": {
            "raw_patterns": ["DONATE", "GIFT", "CONTRIBUTE", "GIVE"],
            "canonical": "FINANCES",
            "abstract": "EXCHANGES",
            "properties": {"charitable": True, "no_return_expected": True}
        },
        "SELLS": {
            "raw_patterns": ["SELL", "MARKET", "VEND", "RETAIL", "TRADE"],
            "canonical": "TRANSACTS",
            "abstract": "EXCHANGES",
            "properties": {"commercial": True, "has_price": True}
        },

        # === ORGANIZATIONAL & PROFESSIONAL ===
        "WORKS_FOR": {
            "raw_patterns": ["WORK", "EMPLOY", "SERVE", "STAFF"],
            "canonical": "EMPLOYS",
            "abstract": "ASSOCIATES",
            "properties": {"employment": True, "has_role": True}
        },
        "LEADS": {
            "raw_patterns": ["LEAD", "HEAD", "DIRECT", "MANAGE", "OVERSEE"],
            "canonical": "MANAGES",
            "abstract": "HIERARCHIZES",
            "properties": {"authority": True, "responsibility": True}
        },
        "FOUNDED": {
            "raw_patterns": ["FOUND", "ESTABLISH", "CREATE", "START", "LAUNCH"],
            "canonical": "CREATES",
            "abstract": "CREATES",
            "properties": {"origin_point": True, "founder_status": True}
        },
        "OWNS": {
            "raw_patterns": ["OWN", "POSSESS", "CONTROL", "HOLD"],
            "canonical": "CONTROLS",
            "abstract": "HIERARCHIZES",
            "properties": {"legal_ownership": True, "control": True}
        },

        # === COLLABORATION & PARTNERSHIP ===
        "COLLABORATES_WITH": {
            "raw_patterns": ["COLLABORAT", "COOPERAT", "PARTNER", "WORK_WITH", "TEAM"],
            "canonical": "PARTNERS",
            "abstract": "ASSOCIATES",
            "properties": {"mutual": True, "equal_status": True}
        },
        "PARTNERS_WITH": {
            "raw_patterns": ["PARTNER", "ALLY", "JOIN", "UNITE"],
            "canonical": "PARTNERS",
            "abstract": "ASSOCIATES",
            "properties": {"formal_agreement": True, "shared_goals": True}
        },
        "MEMBER_OF": {
            "raw_patterns": ["MEMBER", "BELONG", "PART_OF", "AFFILIATE"],
            "canonical": "BELONGS",
            "abstract": "HIERARCHIZES",
            "properties": {"membership": True, "group_identity": True}
        },

        # === PRACTICES & METHODS ===
        "PRACTICES": {
            "raw_patterns": ["PRACTICE", "IMPLEMENT", "USE", "EMPLOY", "APPLY"],
            "canonical": "IMPLEMENTS",
            "abstract": "USES",
            "properties": {"methodology": True, "active_use": True}
        },
        "ADVOCATES_FOR": {
            "raw_patterns": ["ADVOCATE", "PROMOTE", "CHAMPION", "SUPPORT", "ENDORSE"],
            "canonical": "PROMOTES",
            "abstract": "INFLUENCES",
            "properties": {"public_stance": True, "activism": True}
        },
        "RESEARCHES": {
            "raw_patterns": ["RESEARCH", "STUDY", "INVESTIGATE", "EXPLORE", "ANALYZE"],
            "canonical": "INVESTIGATES",
            "abstract": "CREATES",
            "properties": {"scientific": True, "discovery": True}
        },
        "DEVELOPS": {
            "raw_patterns": ["DEVELOP", "BUILD", "DESIGN", "ENGINEER", "CONSTRUCT"],
            "canonical": "CREATES",
            "abstract": "CREATES",
            "properties": {"innovation": True, "technical": True}
        },

        # === SPATIAL & LOCATION ===
        "LOCATED_IN": {
            "raw_patterns": ["LOCATED", "SITUATED", "BASED", "FOUND_IN", "RESIDES"],
            "canonical": "LOCATES",
            "abstract": "LOCATES",
            "properties": {"physical_location": True, "geographic": True}
        },
        "OPERATES_IN": {
            "raw_patterns": ["OPERATE", "ACTIVE_IN", "SERVE", "COVER"],
            "canonical": "LOCATES",
            "abstract": "LOCATES",
            "properties": {"service_area": True, "operational": True}
        },
        "ORIGINATES_FROM": {
            "raw_patterns": ["ORIGINATE", "COME_FROM", "HAIL", "SOURCE"],
            "canonical": "LOCATES",
            "abstract": "LOCATES",
            "properties": {"origin": True, "historical": True}
        },

        # === TEMPORAL ===
        "STARTED_IN": {
            "raw_patterns": ["START", "BEGIN", "COMMENCE", "INITIATE", "LAUNCH"],
            "canonical": "TEMPORALIZES",
            "abstract": "TEMPORALIZES",
            "properties": {"start_point": True, "temporal": True}
        },
        "ENDED_IN": {
            "raw_patterns": ["END", "FINISH", "CONCLUDE", "TERMINATE", "CEASE"],
            "canonical": "TEMPORALIZES",
            "abstract": "TEMPORALIZES",
            "properties": {"end_point": True, "temporal": True}
        },
        "OCCURRED_IN": {
            "raw_patterns": ["OCCUR", "HAPPEN", "TAKE_PLACE", "TRANSPIRE"],
            "canonical": "TEMPORALIZES",
            "abstract": "TEMPORALIZES",
            "properties": {"event": True, "temporal": True}
        },

        # === KNOWLEDGE & EXPERTISE ===
        "HAS_EXPERTISE": {
            "raw_patterns": ["EXPERTISE", "SKILL", "KNOWLEDGE", "PROFICIENCY", "COMPETENCE"],
            "canonical": "HAS_KNOWLEDGE",
            "abstract": "ATTRIBUTES",
            "properties": {"cognitive": True, "professional": True}
        },
        "HAS_CERTIFICATION": {
            "raw_patterns": ["CERTIF", "CREDENTIAL", "QUALIFICATION", "LICENSE"],
            "canonical": "HAS_KNOWLEDGE",
            "abstract": "ATTRIBUTES",
            "properties": {"verified": True, "formal": True}
        },
        "HAS_EXPERIENCE": {
            "raw_patterns": ["EXPERIENCE", "BACKGROUND", "HISTORY", "TRACK_RECORD"],
            "canonical": "HAS_KNOWLEDGE",
            "abstract": "ATTRIBUTES",
            "properties": {"temporal_aspect": True, "accumulated": True}
        },

        # === INFLUENCE & IMPACT ===
        "INFLUENCES": {
            "raw_patterns": ["INFLUENCE", "AFFECT", "SHAPE", "SWAY", "IMPACT"],
            "canonical": "INFLUENCES",
            "abstract": "INFLUENCES",
            "properties": {"causal": True, "directional": True}
        },
        "INSPIRED_BY": {
            "raw_patterns": ["INSPIRE", "MOTIVATE", "LIGHTED_FIRE", "SPARK", "ENERGIZE"],
            "canonical": "INFLUENCES",
            "abstract": "INFLUENCES",
            "properties": {"emotional": True, "transformative": True}
        },
        "TRANSFORMS": {
            "raw_patterns": ["TRANSFORM", "CHANGE", "CONVERT", "ALTER", "MODIFY"],
            "canonical": "TRANSFORMS",
            "abstract": "TRANSFORMS",
            "properties": {"state_change": True, "significant": True}
        },

        # === PRODUCTION & CREATION ===
        "PRODUCES": {
            "raw_patterns": ["PRODUCE", "MANUFACTURE", "MAKE", "CREATE", "GENERATE"],
            "canonical": "PRODUCES",
            "abstract": "CREATES",
            "properties": {"output": True, "tangible": True}
        },
        "PROVIDES": {
            "raw_patterns": ["PROVIDE", "SUPPLY", "DELIVER", "OFFER", "FURNISH"],
            "canonical": "PROVIDES",
            "abstract": "EXCHANGES",
            "properties": {"service": True, "availability": True}
        },
        "DISTRIBUTES": {
            "raw_patterns": ["DISTRIBUTE", "SPREAD", "DISSEMINATE", "ALLOCATE"],
            "canonical": "PROVIDES",
            "abstract": "EXCHANGES",
            "properties": {"logistics": True, "reach": True}
        },

        # === ATTRIBUTES & PROPERTIES ===
        # More specific than generic HAS_PROPERTY!
        "HAS_CAPACITY": {
            "raw_patterns": ["CAPACITY", "ABILITY", "CAPABILITY", "POTENTIAL"],
            "canonical": "HAS_ABILITY",
            "abstract": "ATTRIBUTES",
            "properties": {"potential": True, "measurable": True}
        },
        "HAS_MISSION": {
            "raw_patterns": ["MISSION", "PURPOSE", "GOAL", "OBJECTIVE", "AIM"],
            "canonical": "HAS_PURPOSE",
            "abstract": "ATTRIBUTES",
            "properties": {"intentional": True, "strategic": True}
        },
        "HAS_VALUE": {
            "raw_patterns": ["VALUE", "WORTH", "IMPORTANCE", "SIGNIFICANCE"],
            "canonical": "HAS_QUALITY",
            "abstract": "ATTRIBUTES",
            "properties": {"evaluative": True, "subjective": True}
        },
        "HAS_QUANTITY": {
            "raw_patterns": ["QUANTITY", "AMOUNT", "NUMBER", "COUNT", "SIZE"],
            "canonical": "HAS_MEASURE",
            "abstract": "ATTRIBUTES",
            "properties": {"quantitative": True, "measurable": True}
        },
        "HAS_DURATION": {
            "raw_patterns": ["DURATION", "LENGTH", "PERIOD", "TIME", "SPAN"],
            "canonical": "HAS_TEMPORAL",
            "abstract": "ATTRIBUTES",
            "properties": {"temporal": True, "measurable": True}
        },

        # === COMMUNICATION & INFORMATION ===
        "MENTIONS": {
            "raw_patterns": ["MENTION", "CITE", "REFERENCE", "NOTE", "DISCUSS"],
            "canonical": "REFERENCES",
            "abstract": "COMMUNICATES",
            "properties": {"informational": True, "citation": True}
        },
        "DISCUSSES": {
            "raw_patterns": ["DISCUSS", "TALK", "CONVERSE", "DEBATE", "EXPLAIN"],
            "canonical": "COMMUNICATES",
            "abstract": "COMMUNICATES",
            "properties": {"dialogue": True, "detailed": True}
        },
        "ANNOUNCES": {
            "raw_patterns": ["ANNOUNCE", "DECLARE", "PROCLAIM", "BROADCAST"],
            "canonical": "COMMUNICATES",
            "abstract": "COMMUNICATES",
            "properties": {"public": True, "formal": True}
        },

        # === DEPENDENCY & REQUIREMENT ===
        "REQUIRES": {
            "raw_patterns": ["REQUIRE", "NEED", "DEPEND", "NECESSITATE"],
            "canonical": "DEPENDS_ON",
            "abstract": "DEPENDS",
            "properties": {"prerequisite": True, "essential": True}
        },
        "ENABLES": {
            "raw_patterns": ["ENABLE", "ALLOW", "FACILITATE", "PERMIT", "EMPOWER"],
            "canonical": "SUPPORTS",
            "abstract": "DEPENDS",
            "properties": {"empowering": True, "positive": True}
        },
        "PREVENTS": {
            "raw_patterns": ["PREVENT", "BLOCK", "INHIBIT", "STOP", "HINDER"],
            "canonical": "CONSTRAINS",
            "abstract": "DEPENDS",
            "properties": {"blocking": True, "negative": True}
        }
    }

    # Canonical to Abstract mapping (simplified from domain)
    CANONICAL_TO_ABSTRACT = {
        "EDUCATES": "INFLUENCES",
        "FINANCES": "EXCHANGES",
        "TRANSACTS": "EXCHANGES",
        "EMPLOYS": "ASSOCIATES",
        "MANAGES": "HIERARCHIZES",
        "CREATES": "CREATES",
        "CONTROLS": "HIERARCHIZES",
        "PARTNERS": "ASSOCIATES",
        "BELONGS": "HIERARCHIZES",
        "IMPLEMENTS": "USES",
        "PROMOTES": "INFLUENCES",
        "INVESTIGATES": "CREATES",
        "LOCATES": "LOCATES",
        "TEMPORALIZES": "TEMPORALIZES",
        "HAS_KNOWLEDGE": "ATTRIBUTES",
        "HAS_ABILITY": "ATTRIBUTES",
        "HAS_PURPOSE": "ATTRIBUTES",
        "HAS_QUALITY": "ATTRIBUTES",
        "HAS_MEASURE": "ATTRIBUTES",
        "HAS_TEMPORAL": "ATTRIBUTES",
        "INFLUENCES": "INFLUENCES",
        "TRANSFORMS": "TRANSFORMS",
        "PRODUCES": "CREATES",
        "PROVIDES": "EXCHANGES",
        "REFERENCES": "COMMUNICATES",
        "COMMUNICATES": "COMMUNICATES",
        "DEPENDS_ON": "DEPENDS",
        "SUPPORTS": "DEPENDS",
        "CONSTRAINS": "DEPENDS"
    }

    def __init__(self, use_embeddings: bool = False, api_key: str = None):
        """Initialize normalizer with rich semantic ontology"""
        self.use_embeddings = use_embeddings
        self.domain_map_cache = {}
        self.embedding_cache = {}
        self.pattern_index = self._build_pattern_index()

        if use_embeddings:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required for embeddings")
            self.client = OpenAI(api_key=self.api_key)
            self._precompute_domain_embeddings()

    def _build_pattern_index(self) -> Dict[str, str]:
        """Build reverse index from patterns to domain types"""
        pattern_index = {}
        for domain_type, config in self.DOMAIN_ONTOLOGY.items():
            for pattern in config["raw_patterns"]:
                pattern_index[pattern] = domain_type
        return pattern_index

    def _precompute_domain_embeddings(self):
        """Precompute embeddings for all domain types"""
        print("Precomputing domain ontology embeddings...")
        for domain_type in self.DOMAIN_ONTOLOGY.keys():
            if domain_type not in self.embedding_cache:
                # Use the domain type and its patterns for embedding
                text = domain_type.replace("_", " ").lower()
                self.embedding_cache[domain_type] = self._get_embedding(text)

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
                              source_entity_type: str = None,
                              target_entity_type: str = None,
                              use_similarity: bool = None) -> Dict[str, any]:
        """
        Normalize a raw relationship type through semantic hierarchy

        Args:
            raw_type: The raw relationship type from extraction
            source_entity_type: Type of source entity (for context)
            target_entity_type: Type of target entity (for context)
            use_similarity: Override default embedding behavior

        Returns:
            Dictionary with normalized types at all levels
        """
        raw_upper = raw_type.upper().replace("-", "_").replace(" ", "_")

        # Check cache
        cache_key = f"{raw_upper}:{source_entity_type}:{target_entity_type}"
        if cache_key in self.domain_map_cache:
            return self.domain_map_cache[cache_key]

        # Find best domain match
        domain_type = self._find_domain_type(raw_upper, use_similarity)

        if domain_type and domain_type in self.DOMAIN_ONTOLOGY:
            config = self.DOMAIN_ONTOLOGY[domain_type]
            result = {
                "raw": raw_type,
                "domain": domain_type,
                "canonical": config["canonical"],
                "abstract": config["abstract"],
                "properties": config.get("properties", {}),
                "confidence": 1.0
            }
        else:
            # Fallback for unmapped types
            result = {
                "raw": raw_type,
                "domain": raw_upper,  # Keep original as domain
                "canonical": self._guess_canonical(raw_upper),
                "abstract": "RELATES",
                "properties": {},
                "confidence": 0.5
            }

        self.domain_map_cache[cache_key] = result
        return result

    def _find_domain_type(self, raw_upper: str, use_similarity: bool = None) -> Optional[str]:
        """Find the best domain type for a raw relationship"""

        # 1. Try exact pattern match
        for domain_type, config in self.DOMAIN_ONTOLOGY.items():
            for pattern in config["raw_patterns"]:
                if pattern in raw_upper or raw_upper == pattern:
                    return domain_type

        # 2. Try partial match on key terms
        raw_parts = raw_upper.split("_")
        for part in raw_parts:
            if len(part) > 3:  # Skip short words
                for pattern, domain_type in self.pattern_index.items():
                    if part.startswith(pattern[:min(len(part), len(pattern))]):
                        return domain_type

        # 3. Use embeddings if enabled
        if use_similarity or (use_similarity is None and self.use_embeddings):
            return self._find_similar_domain(raw_upper)

        return None

    def _find_similar_domain(self, raw_type: str, threshold: float = 0.7) -> Optional[str]:
        """Find most similar domain type using embeddings"""
        raw_embedding = self._get_embedding(raw_type)

        best_match = None
        best_score = 0

        for domain_type, domain_embedding in self.embedding_cache.items():
            if domain_type in self.DOMAIN_ONTOLOGY:
                similarity = cosine_similarity(
                    raw_embedding.reshape(1, -1),
                    domain_embedding.reshape(1, -1)
                )[0][0]

                if similarity > best_score:
                    best_score = similarity
                    best_match = domain_type

        return best_match if best_score >= threshold else None

    def _guess_canonical(self, raw_upper: str) -> str:
        """Guess canonical type for unmapped relationships"""
        # Common patterns
        if "HAS_" in raw_upper:
            if any(t in raw_upper for t in ["TIME", "DURATION", "DATE", "PERIOD"]):
                return "HAS_TEMPORAL"
            elif any(t in raw_upper for t in ["SIZE", "QUANTITY", "NUMBER", "AMOUNT"]):
                return "HAS_MEASURE"
            elif any(t in raw_upper for t in ["SKILL", "EXPERTISE", "KNOWLEDGE"]):
                return "HAS_KNOWLEDGE"
            else:
                return "HAS_QUALITY"
        elif any(t in raw_upper for t in ["WORK", "EMPLOY", "JOB"]):
            return "EMPLOYS"
        elif any(t in raw_upper for t in ["FUND", "PAY", "FINANCE"]):
            return "FINANCES"
        elif any(t in raw_upper for t in ["LOCATED", "BASED", "SITUATED"]):
            return "LOCATES"
        else:
            return "RELATES"

    def analyze_corpus(self, relationships_dir: Path) -> Dict:
        """Analyze all extracted relationships and build statistics"""
        raw_types = Counter()
        domain_mappings = Counter()
        canonical_mappings = Counter()
        abstract_mappings = Counter()
        unmapped_types = []

        files = list(relationships_dir.glob("episode_*_extraction.json"))
        print(f"Analyzing {len(files)} extraction files...")

        for file_path in files:
            with open(file_path) as f:
                data = json.load(f)
                for rel in data.get("relationships", []):
                    raw_type = rel.get("relationship_type", "UNKNOWN")
                    raw_types[raw_type] += 1

                    # Normalize
                    normalized = self.normalize_relationship(
                        raw_type,
                        rel.get("source_type"),
                        rel.get("target_type")
                    )

                    domain_mappings[normalized["domain"]] += 1
                    canonical_mappings[normalized["canonical"]] += 1
                    abstract_mappings[normalized["abstract"]] += 1

                    if normalized["confidence"] < 1.0:
                        unmapped_types.append((raw_type, raw_types[raw_type]))

        return {
            "total_relationships": sum(raw_types.values()),
            "unique_raw_types": len(raw_types),
            "unique_domain_types": len(domain_mappings),
            "unique_canonical_types": len(canonical_mappings),
            "unique_abstract_types": len(abstract_mappings),
            "top_raw_types": raw_types.most_common(20),
            "top_domain_types": domain_mappings.most_common(20),
            "top_canonical_types": canonical_mappings.most_common(15),
            "top_abstract_types": abstract_mappings.most_common(10),
            "unmapped_types": sorted(set(unmapped_types), key=lambda x: -x[1])[:20]
        }

    def process_extraction_file(self, file_path: Path, output_path: Path = None):
        """Process a single extraction file with semantic normalization"""
        with open(file_path) as f:
            data = json.load(f)

        # Normalize each relationship
        enriched_relationships = []
        for rel in data.get("relationships", []):
            normalized = self.normalize_relationship(
                rel.get("relationship_type", ""),
                rel.get("source_type"),
                rel.get("target_type")
            )

            # Add normalized fields while preserving original
            enriched_rel = rel.copy()
            enriched_rel["relationship_domain"] = normalized["domain"]
            enriched_rel["relationship_canonical"] = normalized["canonical"]
            enriched_rel["relationship_abstract"] = normalized["abstract"]
            enriched_rel["semantic_properties"] = normalized["properties"]
            enriched_rel["normalization_confidence"] = normalized["confidence"]

            enriched_relationships.append(enriched_rel)

        data["relationships"] = enriched_relationships

        # Save enriched version
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        return data


def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic normalization of relationships")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze corpus with semantic ontology")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize all extraction files")
    parser.add_argument("--use-embeddings", action="store_true",
                       help="Use OpenAI embeddings for similarity")
    parser.add_argument("--input-dir", type=str,
                       default="data/knowledge_graph/relationships",
                       help="Input directory")
    parser.add_argument("--output-dir", type=str,
                       default="data/knowledge_graph/relationships_semantic",
                       help="Output directory")

    args = parser.parse_args()

    # Initialize normalizer with semantic ontology
    normalizer = SemanticRelationshipNormalizer(use_embeddings=args.use_embeddings)

    relationships_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.analyze:
        print("=" * 80)
        print("SEMANTIC RELATIONSHIP ANALYSIS")
        print("=" * 80)

        stats = normalizer.analyze_corpus(relationships_dir)

        print(f"\n📊 STATISTICS:")
        print(f"  Total relationships: {stats['total_relationships']:,}")
        print(f"  Unique raw types: {stats['unique_raw_types']}")
        print(f"  Unique domain types: {stats['unique_domain_types']} (semantic ontology)")
        print(f"  Unique canonical types: {stats['unique_canonical_types']}")
        print(f"  Unique abstract types: {stats['unique_abstract_types']}")

        print(f"\n🔝 TOP DOMAIN TYPES (Semantic Layer):")
        for dtype, count in stats['top_domain_types']:
            if dtype in normalizer.DOMAIN_ONTOLOGY:
                props = normalizer.DOMAIN_ONTOLOGY[dtype].get("properties", {})
                print(f"  {dtype}: {count}")
                if props:
                    print(f"    Properties: {list(props.keys())}")

        print(f"\n🎯 TOP CANONICAL TYPES:")
        for ctype, count in stats['top_canonical_types']:
            print(f"  {ctype}: {count}")

        if stats['unmapped_types']:
            print(f"\n⚠️  UNMAPPED TYPES (need ontology extension):")
            for utype, count in stats['unmapped_types'][:10]:
                print(f"  {utype}: {count}")

    if args.normalize:
        print("=" * 80)
        print("SEMANTIC NORMALIZATION")
        print("=" * 80)

        normalizer.process_all_extractions(relationships_dir, output_dir)

        print("\n✅ Semantic normalization complete!")
        print(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()