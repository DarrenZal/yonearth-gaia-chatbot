#!/usr/bin/env python3
"""
Build unified knowledge graph using hybrid approach:
- Use existing ACE-postprocessed episode files (keep quality benefits)
- Extract books fresh with new extractors
- Apply strict validation during graph building

This gives us the best of both worlds:
- ACE postprocessing benefits (pronoun resolution, discourse analysis)
- No catastrophic merges (Moscow != Soil)

Usage:
    python scripts/build_unified_graph_hybrid.py
"""

import json
import logging
import sys
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Paths for graph embeddings (used for graph-informed deduplication)
ROOT = Path(__file__).parent.parent
GRAPHSAGE_EMB_PATH = ROOT / "data/graphrag_hierarchy/graphsage_embeddings.npy"
GRAPHSAGE_IDS_PATH = ROOT / "data/graphrag_hierarchy/graphsage_entity_ids.json"

from src.knowledge_graph.validators.entity_merge_validator import EntityMergeValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridGraphBuilder:
    """Builds unified graph from ACE episodes + fresh book extractions"""

    # Type normalization mapping - fixes case inconsistency from different extractors
    TYPE_NORMALIZATION = {
        # Concept/abstract
        'concept': 'CONCEPT', 'Concept': 'CONCEPT', 'idea': 'CONCEPT', 'Idea': 'CONCEPT',
        # People
        'person': 'PERSON', 'Person': 'PERSON', 'individual': 'PERSON', 'Individual': 'PERSON',
        'human': 'PERSON', 'Human': 'PERSON', 'people': 'PERSON',
        # Organizations
        'organization': 'ORGANIZATION', 'Organization': 'ORGANIZATION',
        'company': 'ORGANIZATION', 'Company': 'ORGANIZATION',
        'business': 'ORGANIZATION', 'Business': 'ORGANIZATION',
        'group': 'ORGANIZATION', 'Group': 'ORGANIZATION',
        # Locations
        'place': 'PLACE', 'Place': 'PLACE', 'location': 'PLACE', 'Location': 'PLACE',
        'region': 'PLACE', 'Region': 'PLACE',
        # Products
        'product': 'PRODUCT', 'Product': 'PRODUCT',
        # Events
        'event': 'EVENT', 'Event': 'EVENT',
        # Practices/Activities
        'practice': 'PRACTICE', 'Practice': 'PRACTICE',
        'activity': 'PRACTICE', 'Activity': 'PRACTICE',
        'action': 'PRACTICE', 'Action': 'PRACTICE',
        # Species/organisms
        'species': 'SPECIES', 'Species': 'SPECIES',
        'plant': 'SPECIES', 'Plant': 'SPECIES',
        'animal': 'SPECIES', 'Animal': 'SPECIES',
        # Technology
        'technology': 'TECHNOLOGY', 'Technology': 'TECHNOLOGY',
        'website': 'TECHNOLOGY', 'Website': 'TECHNOLOGY',
        # Ecosystem
        'ecosystem': 'ECOSYSTEM', 'Ecosystem': 'ECOSYSTEM',
        # Misc/plurals/variants
        'organizations': 'ORGANIZATION', 'Organizations': 'ORGANIZATION',
        'communities': 'COMMUNITY', 'Communities': 'COMMUNITY',
        'families': 'FAMILY', 'Families': 'FAMILY',
        'stories': 'STORY', 'Stories': 'STORY',
        'archives': 'ARCHIVE', 'Archives': 'ARCHIVE',
        'roles': 'ROLE', 'Roles': 'ROLE',
        'communications': 'COMMUNICATION', 'Communications': 'COMMUNICATION',
        'projects': 'PROJECT', 'Projects': 'PROJECT',
        'technologies': 'TECHNOLOGY', 'Technologies': 'TECHNOLOGY',
        'countries': 'PLACE', 'Countries': 'PLACE', 'country': 'PLACE', 'Country': 'PLACE',
        'locations': 'PLACE', 'Locations': 'PLACE',
        'places': 'PLACE', 'Places': 'PLACE',
        # Generic fallback
        'string': 'CONCEPT', 'entity': 'CONCEPT', 'object': 'CONCEPT', 'noun': 'CONCEPT',
    }

    # Canonical labels for country ids (ids follow validator COUNTRY_SYNONYMS values)
    COUNTRY_LABELS = {
        'usa': 'United States',
        'gbr': 'United Kingdom',
        'are': 'United Arab Emirates',
        'cod': 'Democratic Republic of the Congo',
        'cog': 'Republic of the Congo',
        'kor': 'South Korea',
        'prk': 'North Korea',
        'rus': 'Russia',
        'irn': 'Iran',
        'irq': 'Iraq',
    }

    def __init__(
        self,
        similarity_threshold: int = 95,
        validator: EntityMergeValidator = None
    ):
        self.similarity_threshold = similarity_threshold
        self.validator = validator
        self.entities = {}
        self.relationships = []
        self.entity_id_counter = 0
        self.name_to_entity_id: Dict[str, str] = {}

    def _normalize_type(self, entity_type: str) -> str:
        """Normalize entity type to canonical uppercase form."""
        if not entity_type:
            return 'UNKNOWN'
        # Check normalization map first, otherwise uppercase
        return self.TYPE_NORMALIZATION.get(entity_type, entity_type.upper())

    def _ensure_entity(self, name: str, entity_type: str, source: str, episode_number: Any = None, book_slug: str = None):
        """Create or update an entity synthesized from ACE relationships."""
        if not name:
            return

        # Filter out sentence-like or overly long phrases posing as entities
        if self._should_filter_name(name):
            logger.debug(f"Skipping entity that looks like a sentence/phrase: '{name}'")
            return

        key = name.lower()
        if key in self.name_to_entity_id:
            return

        entity_id = f"ace_entity_{self.entity_id_counter}"
        self.entity_id_counter += 1

        # Normalize type to canonical form
        normalized_type = self._normalize_type(entity_type)

        self.entities[entity_id] = {
            'id': entity_id,
            'name': name,
            'type': normalized_type,
            'description': '',
            'aliases': [],
            'metadata': {
                'episode_number': episode_number,
                'book_slug': book_slug,
                'source': source
            },
            'provenance': [{
                'source': source,
                'episode_number': episode_number,
                'book_slug': book_slug
            }]
        }
        self.name_to_entity_id[key] = entity_id

    def _lookup_type(self, name: str) -> Optional[str]:
        """Lookup entity type by name."""
        if not name:
            return None
        ent_id = self.name_to_entity_id.get(name.lower())
        if ent_id and ent_id in self.entities:
            return self.entities[ent_id].get('type')
        return None

    @staticmethod
    def _should_filter_name(name: str) -> bool:
        """
        Heuristic filter to drop sentence-like or noisy entity names.

        Criteria:
        - Very long (> 8 tokens)
        - Long and no proper-case tokens (likely a phrase)
        - Contains sentence punctuation (comma/semicolon/colon)
        - Looks like a clause or question (e.g., starts with "what is", "how to", "why we", etc.)
        """
        if not name:
            return True

        lower = name.strip().lower()
        tokens = lower.split()

        if len(tokens) > 8:
            return True

        # If name is long and has no capital letters, likely a phrase
        if len(tokens) > 6 and not any(ch.isupper() for ch in name):
            return True

        if any(p in name for p in [',', ';', ':']):
            return True

        # Generic clause/question detection
        clause_prefixes = (
            "what is",
            "what are",
            "how to",
            "why we",
            "people who",
            "things that",
            "ideas about",
            "learning about",
            "learn about",
            "looking at",
            "talking about",
            "discussion of",
            "particular area of",
            "other parts of",
            "part of the",
            "two different parts",
            "many different regions",
            "the woods in the",
            "two days boat ride",
            "the global south in",
            "the town they were born in",
        )
        if any(lower.startswith(p) for p in clause_prefixes):
            return True

        # Reject names that are mostly stop/connector words and >5 tokens
        stop_like = {"the", "a", "an", "of", "in", "on", "for", "with", "and", "to", "at", "by", "from", "about"}
        if len(tokens) > 5:
            non_stop = [t for t in tokens if t not in stop_like]
            if len(non_stop) <= 2:
                return True

        return False

    def load_ace_episodes(self, episodes_dir: Path) -> Dict[str, Any]:
        """Load ACE-postprocessed episode files"""
        logger.info("Loading ACE-postprocessed episodes...")

        episode_files = sorted(episodes_dir.glob('episode_*_post.json'))
        if not episode_files:
            logger.error(f"No ACE episode files found in {episodes_dir}")
            return {'files_loaded': 0}

        total_entities = 0
        total_relationships = 0

        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    data = json.load(f)

                episode_num = data.get('episode_number', 'unknown')

                # Load entities
                # ACE files are relationship-centric; synthesize entities from relationships
                # and keep relationship typing.
                for rel in data.get('relationships', []):
                    # Ensure source entity exists
                    src_name = rel.get('source', rel.get('source_entity', '')) or ''
                    src_type = rel.get('source_type', 'UNKNOWN')
                    if src_name:
                        self._ensure_entity(
                            name=src_name,
                            entity_type=src_type,
                            source='ace_postprocessed',
                            episode_number=episode_num
                        )

                    # Ensure target entity exists
                    tgt_name = rel.get('target', rel.get('target_entity', '')) or ''
                    tgt_type = rel.get('target_type', 'UNKNOWN')
                    if tgt_name:
                        self._ensure_entity(
                            name=tgt_name,
                            entity_type=tgt_type,
                            source='ace_postprocessed',
                            episode_number=episode_num
                        )

                    self.relationships.append({
                        'source_entity': src_name,
                        'target_entity': tgt_name,
                        'relationship_type': rel.get('predicate', rel.get('relationship', 'UNKNOWN')),
                        'description': rel.get('description', ''),
                        'metadata': {
                            'episode_number': episode_num,
                            'source': 'ace_postprocessed',
                            'evidence_text': rel.get('evidence_text'),
                            'doc_sha256': rel.get('evidence', {}).get('doc_sha256') if rel.get('evidence') else None
                        },
                        'source_type': src_type,
                        'target_type': tgt_type
                    })
                    total_relationships += 1

                if len(episode_files) < 20 or episode_file == episode_files[-1]:
                    logger.info(f"Loaded {episode_file.name}: {len(data.get('entities', []))} entities, {len(data.get('relationships', []))} relationships")

            except Exception as e:
                logger.error(f"Error loading {episode_file.name}: {e}")
                continue

        logger.info(f"✓ Loaded {len(episode_files)} ACE episodes: {len(self.entities)} entities (synthesized), {total_relationships} relationships")

        return {
            'files_loaded': len(episode_files),
            'total_entities': total_entities,
            'total_relationships': total_relationships
        }

    def load_book_extractions(self, entities_dir: Path) -> Dict[str, Any]:
        """Load fresh book extractions from simple extractor"""
        logger.info("Loading fresh book extractions...")

        book_files = sorted(entities_dir.glob('book_*_extraction.json'))
        if not book_files:
            logger.warning(f"No book extraction files found in {entities_dir}")
            return {'files_loaded': 0}

        total_entities = 0
        total_relationships = 0

        for book_file in book_files:
            try:
                with open(book_file, 'r') as f:
                    data = json.load(f)

                book_slug = data.get('book_slug', book_file.stem.replace('book_', '').replace('_extraction', ''))

                # Load entities
                for entity in data.get('entities', []):
                    entity_id = f"book_{book_slug}_{self.entity_id_counter}"
                    self.entity_id_counter += 1

                    # Normalize type to canonical form
                    normalized_type = self._normalize_type(entity.get('type', 'UNKNOWN'))

                    self.entities[entity_id] = {
                        'id': entity_id,
                        'name': entity.get('name', ''),
                        'type': normalized_type,
                        'description': entity.get('description', ''),
                        'aliases': entity.get('aliases', []),
                        'metadata': {
                            'book_slug': book_slug,
                            'source': 'fresh_extraction',
                            **(entity.get('metadata') or {})
                        },
                        'provenance': [{
                            'source': 'book_extraction',
                            'book_slug': book_slug
                        }]
                    }
                    total_entities += 1
                    self.name_to_entity_id[self.entities[entity_id]['name'].lower()] = entity_id

                # Load relationships
                for rel in data.get('relationships', []):
                    src = rel.get('source_entity', '')
                    tgt = rel.get('target_entity', '')
                    # Best-effort typing for book relationships
                    src_type = self._lookup_type(src)
                    tgt_type = self._lookup_type(tgt)

                    self.relationships.append({
                        'source_entity': src,
                        'target_entity': tgt,
                        'relationship_type': rel.get('relationship_type', 'UNKNOWN'),
                        'description': rel.get('description', ''),
                        'metadata': {
                            'book_slug': book_slug,
                            'source': 'fresh_extraction'
                        },
                        'source_type': src_type,
                        'target_type': tgt_type
                    })
                    total_relationships += 1

                logger.info(f"Loaded {book_file.name}: {len(data.get('entities', []))} entities, {len(data.get('relationships', []))} relationships")

            except Exception as e:
                logger.error(f"Error loading {book_file.name}: {e}")
                continue

        logger.info(f"✓ Loaded {len(book_files)} books: {total_entities} entities, {total_relationships} relationships")

        return {
            'files_loaded': len(book_files),
            'total_entities': total_entities,
            'total_relationships': total_relationships
        }

    def deduplicate_with_validation(self) -> Dict[str, Any]:
        """Deduplicate entities with strict validation"""
        logger.info("Deduplicating entities with validation...")
        logger.info(f"  Similarity threshold: {self.similarity_threshold}")
        logger.info(f"  Validation enabled: {self.validator is not None}")

        from fuzzywuzzy import fuzz
        from collections import defaultdict

        # Group by type
        entities_by_type = defaultdict(list)
        for entity_id, entity in self.entities.items():
            entities_by_type[entity['type']].append((entity_id, entity))

        merged_entities = {}
        entity_id_mapping = {}
        name_canonical_map = {}

        # Normalization helper for candidate generation (aligns with validator)
        if self.validator:
            normalize_name = self.validator._normalize_name
            flexible_types = getattr(self.validator, "FLEXIBLE_TYPES", set())
            country_id_fn = getattr(self.validator, "_country_id", lambda _: None)
        else:
            normalize_name = lambda n: n.lower() if isinstance(n, str) else ""
            flexible_types = set()
            country_id_fn = lambda _: None

        def _token_overlap(tok1, tok2):
            if not tok1 or not tok2:
                return 0.0
            shared = len(tok1.intersection(tok2))
            return shared / max(len(tok1), len(tok2))

        # Load GraphSAGE embeddings if available (for graph-informed deduplication)
        graph_emb_lookup = None
        if GRAPHSAGE_EMB_PATH.exists() and GRAPHSAGE_IDS_PATH.exists():
            try:
                graphsage_embeddings = np.load(GRAPHSAGE_EMB_PATH)
                with GRAPHSAGE_IDS_PATH.open() as f:
                    graphsage_ids = json.load(f)
                graph_emb_lookup = {name: graphsage_embeddings[i] for i, name in enumerate(graphsage_ids)}
                logger.info(f"  Loaded GraphSAGE embeddings for {len(graph_emb_lookup):,} entities (graph-informed deduplication enabled)")
            except Exception as e:
                logger.warning(f"  Could not load GraphSAGE embeddings: {e}")
                graph_emb_lookup = None
        else:
            logger.info("  GraphSAGE embeddings not available - using string-based deduplication only")

        def _graph_similarity(name1: str, name2: str) -> float:
            """Compute cosine similarity between two entities using graph embeddings"""
            if graph_emb_lookup is None:
                return 0.0
            emb1 = graph_emb_lookup.get(name1)
            emb2 = graph_emb_lookup.get(name2)
            if emb1 is None or emb2 is None:
                return 0.0
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot / (norm1 * norm2))

        for entity_type, entity_list in entities_by_type.items():
            logger.info(f"  Deduplicating {len(entity_list)} entities of type {entity_type}")

            processed = set()

            # Precompute normalized names for this type bucket to avoid repeated work
            normalized_names = {
                ent_id: normalize_name(ent.get('name', ''))
                for ent_id, ent in entity_list
            }
            country_ids = {
                ent_id: country_id_fn(normalized_names[ent_id])
                for ent_id, _ in entity_list
            }
            norm_tokens = {
                ent_id: set(normalized_names[ent_id].split()) if normalized_names[ent_id] else set()
                for ent_id, _ in entity_list
            }

            for entity_id, entity in entity_list:
                if entity_id in processed:
                    continue

                similar_entities = [(entity_id, entity)]
                entity_name = entity.get('name', '')
                entity_name_lower = entity_name.lower()
                entity_name_norm = normalized_names.get(entity_id, "")

                for other_id, other_entity in entity_list:
                    if other_id == entity_id or other_id in processed:
                        continue

                    other_name = other_entity.get('name', '')
                    other_name_lower = other_name.lower()
                    other_name_norm = normalized_names.get(other_id, "")
                    entity_country = country_ids.get(entity_id)
                    other_country = country_ids.get(other_id)
                    tokens_self = norm_tokens.get(entity_id, set())
                    tokens_other = norm_tokens.get(other_id, set())

                    raw_similarity = fuzz.ratio(entity_name_lower, other_name_lower)
                    norm_similarity = fuzz.ratio(entity_name_norm, other_name_norm) if entity_name_norm and other_name_norm else raw_similarity

                    # Candidate selection:
                    # - Always accept exact raw match
                    # - Accept raw_score >= threshold
                    # - For flexible types, also accept normalized_score >= 85 to let Tier 2 fire
                    # - For PLACE, accept if both normalize to same canonical country id
                    # - For flexible types, accept moderate matches with token overlap (abbreviation cases)
                    is_candidate = False
                    if entity_name_lower == other_name_lower or raw_similarity >= self.similarity_threshold:
                        is_candidate = True
                    elif entity_type == 'PLACE' and entity_country and entity_country == other_country:
                        is_candidate = True
                    elif entity_type in flexible_types and norm_similarity >= 85:
                        is_candidate = True
                    elif entity_type in flexible_types:
                        overlap = _token_overlap(tokens_self, tokens_other)
                        if overlap >= 0.5 and norm_similarity >= 60:
                            is_candidate = True
                    elif entity_type == 'PERSON':
                        # Special handling for person names: "Aaron Perry" vs "Aaron William Perry"
                        # Check if one name is a subset of the other (first+last vs first+middle+last)
                        # NOTE: Use original names for ordering (sets don't preserve order)
                        entity_tokens_ordered = entity_name_lower.split()
                        other_tokens_ordered = other_name_lower.split()

                        # If both have 2+ tokens and one is a subset of the other, consider merging
                        if len(entity_tokens_ordered) >= 2 and len(other_tokens_ordered) >= 2:
                            # Check if shorter name's tokens are all in longer name (use sets for subset check)
                            if tokens_self.issubset(tokens_other) or tokens_other.issubset(tokens_self):
                                # Use ordered lists for first/last comparison
                                shorter_tokens = entity_tokens_ordered if len(entity_tokens_ordered) < len(other_tokens_ordered) else other_tokens_ordered
                                longer_tokens = other_tokens_ordered if len(entity_tokens_ordered) < len(other_tokens_ordered) else entity_tokens_ordered

                                # Check if first token (first name) and last token (last name) match
                                if shorter_tokens[0] == longer_tokens[0] and shorter_tokens[-1] == longer_tokens[-1]:
                                    is_candidate = True

                    # UNCERTAINTY ZONE: String similarity 60-90%
                    # Use graph embedding similarity as tiebreaker
                    if not is_candidate and 60 <= raw_similarity < 90 and graph_emb_lookup is not None:
                        graph_sim = _graph_similarity(entity_name, other_name)
                        if graph_sim > 0.9:
                            is_candidate = True
                            logger.debug(f"    Graph-informed merge: '{entity_name}' <-> '{other_name}' "
                                       f"(string={raw_similarity}%, graph={graph_sim:.3f})")

                    if not is_candidate:
                        continue

                    # Validate merge if validator provided
                    if self.validator:
                        can_merge, reason = self.validator.can_merge(entity, other_entity, log_rejection=False)
                        if not can_merge:
                            continue

                    similar_entities.append((other_id, other_entity))
                    processed.add(other_id)

                # Merge
                merged_entity = self._merge_entities(similar_entities)
                canonical_id = similar_entities[0][0]

                # If PLACE with known country id, force canonical country label
                entity_country = country_ids.get(canonical_id)
                if entity_type == 'PLACE' and entity_country:
                    canonical_label = self.COUNTRY_LABELS.get(entity_country)
                    if canonical_label:
                        current_name = merged_entity.get('name')
                        if current_name != canonical_label:
                            alias_set = set(merged_entity.get('aliases', []))
                            if current_name:
                                alias_set.add(current_name)
                            merged_entity['name'] = canonical_label
                            merged_entity['aliases'] = sorted(alias_set)

                merged_entities[canonical_id] = merged_entity
                name_canonical_map[merged_entity['name'].lower()] = merged_entity['name']

                for old_id, _ in similar_entities:
                    entity_id_mapping[old_id] = canonical_id
                    original_name = self.entities[old_id]['name']
                    name_canonical_map[original_name.lower()] = merged_entity['name']

                processed.add(entity_id)

        self.entities = merged_entities
        self.name_canonical_map = name_canonical_map
        # Rebuild name index to canonical entities
        self.name_to_entity_id = {e['name'].lower(): ent_id for ent_id, e in merged_entities.items()}

        stats = {
            'entities_after_dedup': len(merged_entities),
            'entities_merged': len(entity_id_mapping) - len(merged_entities)
        }

        logger.info(f"✓ Deduplication complete: {len(merged_entities)} unique entities")

        return stats

    def _merge_entities(self, entities: List) -> Dict:
        """Merge multiple entity records"""
        if len(entities) == 1:
            return entities[0][1]

        merged = dict(entities[0][1])
        canonical_name = merged.get('name')

        all_metadata = []
        all_descriptions = []
        alias_set = set(merged.get('aliases', []))

        for _, entity in entities:
            if entity.get('description'):
                all_descriptions.append(entity['description'])
            if entity.get('metadata'):
                all_metadata.append(entity['metadata'])
            name = entity.get('name')
            if name and name != canonical_name:
                alias_set.add(name)
            alias_set.update(entity.get('aliases', []) or [])

        if all_descriptions:
            merged['description'] = max(all_descriptions, key=len)

        merged['metadata']['merged_from'] = all_metadata
        merged['aliases'] = sorted(list(alias_set))
        merged.setdefault('provenance', []).extend([{
            'id': md.get('id'),
            'source': md.get('source'),
            'episode_number': md.get('episode_number'),
            'book_slug': md.get('book_slug')
        } for md in all_metadata if isinstance(md, dict)])

        return merged

    def deduplicate_relationships(self) -> Dict[str, Any]:
        """Deduplicate relationships and fix entity references"""
        logger.info("Deduplicating relationships...")

        # Fix entity names using canonical map
        for rel in self.relationships:
            src = rel.get('source_entity', '')
            tgt = rel.get('target_entity', '')
            if src:
                rel['source_entity'] = self.name_canonical_map.get(src.lower(), src)
            if tgt:
                rel['target_entity'] = self.name_canonical_map.get(tgt.lower(), tgt)

        # Deduplicate
        unique_rels = {}
        for rel in self.relationships:
            key = (rel['source_entity'].lower(), rel['relationship_type'].lower(), rel['target_entity'].lower())
            if key not in unique_rels:
                unique_rels[key] = rel

        self.relationships = list(unique_rels.values())

        logger.info(f"✓ {len(self.relationships)} unique relationships")

        return {'unique_relationships': len(self.relationships)}

    def export_unified_json(self, output_path: Path):
        """Export to unified.json format"""
        logger.info(f"Exporting to {output_path}...")

        # Build entities dict
        entities_dict = {}
        for entity_id, entity in self.entities.items():
            name = entity['name']
            entities_dict[name] = {
                'type': entity['type'],
                'description': entity.get('description', ''),
                'sources': [entity['metadata'].get('episode_number') or entity['metadata'].get('book_slug')],
                'original_type': entity['type'],
                'evidence': entity.get('metadata', {}).get('evidence', []),
                'aliases': entity.get('aliases', []),
                'provenance': entity.get('provenance', [])
            }

        # Build relationships list
        relationships_list = []
        for rel in self.relationships:
            source_type = rel.get('source_type') or self._lookup_type(rel.get('source_entity'))
            target_type = rel.get('target_type') or self._lookup_type(rel.get('target_entity'))
            relationships_list.append({
                'source': rel['source_entity'],
                'target': rel['target_entity'],
                'predicate': rel['relationship_type'],
                'type': rel['relationship_type'],
                'description': rel.get('description', ''),
                'source_type': source_type or 'UNKNOWN',
                'target_type': target_type or 'UNKNOWN',
                'metadata': rel.get('metadata', {})
            })

        unified_data = {
            'entities': entities_dict,
            'relationships': relationships_list
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Exported {len(entities_dict)} entities, {len(relationships_list)} relationships")

        if self.validator:
            self.validator.log_statistics()


def main():
    parser = argparse.ArgumentParser(description='Build unified graph using hybrid approach')
    parser.add_argument('--similarity-threshold', type=int, default=95)
    parser.add_argument('--output', type=str, default='data/knowledge_graph_unified/unified_hybrid.json')
    parser.add_argument('--extract-books-first', action='store_true', help='Extract books before building')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("HYBRID KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 80)
    logger.info("Approach: ACE episodes + Fresh book extractions + Strict validation")
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info("")

    base_dir = Path(__file__).parent.parent
    ace_episodes_dir = base_dir / 'data' / 'knowledge_graph_unified' / 'episodes_postprocessed'
    book_entities_dir = base_dir / 'data' / 'knowledge_graph' / 'entities'
    output_path = base_dir / args.output

    # Check if ACE episodes exist
    if not ace_episodes_dir.exists() or not list(ace_episodes_dir.glob('episode_*_post.json')):
        logger.error(f"ERROR: No ACE episode files found in {ace_episodes_dir}")
        logger.error("These should have been synced from the old server.")
        sys.exit(1)

    # Extract books if requested
    if args.extract_books_first:
        logger.info("Extracting books first...")
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(base_dir / 'scripts' / 'extract_knowledge_from_books.py')
        ], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Book extraction failed: {result.stderr}")
            sys.exit(1)
        logger.info("✓ Book extraction complete")
        logger.info("")

    # Initialize validator
    validator = EntityMergeValidator(
        similarity_threshold=args.similarity_threshold,
        min_length_ratio=0.6,
        type_strict_matching=True,
        semantic_validation=True
    )

    # Build hybrid graph
    builder = HybridGraphBuilder(
        similarity_threshold=args.similarity_threshold,
        validator=validator
    )

    # Load ACE episodes
    logger.info("=" * 80)
    logger.info("STEP 1: Loading ACE-Postprocessed Episodes")
    logger.info("=" * 80)
    ace_stats = builder.load_ace_episodes(ace_episodes_dir)
    logger.info("")

    # Load books
    logger.info("=" * 80)
    logger.info("STEP 2: Loading Fresh Book Extractions")
    logger.info("=" * 80)
    book_stats = builder.load_book_extractions(book_entities_dir)
    logger.info("")

    # Deduplicate with validation
    logger.info("=" * 80)
    logger.info("STEP 3: Deduplicating with Strict Validation")
    logger.info("=" * 80)
    dedup_stats = builder.deduplicate_with_validation()
    logger.info("")

    # Deduplicate relationships
    logger.info("=" * 80)
    logger.info("STEP 4: Deduplicating Relationships")
    logger.info("=" * 80)
    rel_stats = builder.deduplicate_relationships()
    logger.info("")

    # Export
    logger.info("=" * 80)
    logger.info("STEP 5: Exporting Unified JSON")
    logger.info("=" * 80)
    builder.export_unified_json(output_path)
    logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"ACE episodes: {ace_stats['files_loaded']}")
    logger.info(f"Books: {book_stats['files_loaded']}")
    logger.info(f"Total entities: {len(builder.entities)}")
    logger.info(f"Total relationships: {len(builder.relationships)}")
    logger.info(f"\nOutput: {output_path}")
    logger.info("\nNext steps:")
    logger.info("1. Run validate_unified_graph.py to test quality")
    logger.info("2. Compare with current unified.json")
    logger.info("3. Deploy if tests pass")


if __name__ == '__main__':
    main()
