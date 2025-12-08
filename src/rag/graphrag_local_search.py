"""
GraphRAG Local Search - Entity-centric retrieval with relationship traversal.

Implements Microsoft GraphRAG's "Local Search" approach:
- Extract entities from query using lexicon matching
- Gather entity descriptions from the hierarchy
- Expand to 1-hop relationships with context
- Link to source episodes/books
"""
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document

from ..config import settings

logger = logging.getLogger(__name__)

# Minimum length for entity names/aliases to be indexed
MIN_ALIAS_LENGTH = 3

# Stoplist of common words that shouldn't be matched as entities
# These are short strings that appear frequently but aren't meaningful entities
ENTITY_STOPLIST = {
    # Common short words
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
    'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how', 'its', 'may',
    'new', 'now', 'old', 'see', 'way', 'who', 'did', 'get', 'let', 'put',
    'say', 'she', 'too', 'use', 'man', 'day', 'got', 'him', 'own', 'off',
    'why', 'try', 'ask', 'men', 'run', 'few', 'big', 'set', 'end', 'far',
    # Chemical/mathematical symbols often extracted as entities
    'co2', 'ph', 'pi', 'cu', 'om', 'na', 'mg', 'ca', 'fe',
    # Common abbreviations
    'etc', 'inc', 'llc', 'usa', 'org', 'com', 'net',
    # Single letters that get matched
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    # Common nouns that aren't specific entities
    'people', 'world', 'earth', 'water', 'soil', 'food', 'life', 'time',
    'work', 'year', 'years', 'things', 'part', 'place', 'home',
}


@dataclass
class EntityContext:
    """Context from a matched entity"""
    name: str
    type: str
    description: str
    aliases: List[str]
    sources: List[str]  # Episodes/books mentioning this entity
    mention_count: int
    relevance_score: float = 1.0


@dataclass
class RelationshipContext:
    """Context from a relationship between entities"""
    source: str
    predicate: str
    target: str
    weight: float = 1.0
    context: str = ""


@dataclass
class LocalSearchResult:
    """Result from local entity-centric search"""
    entities: List[EntityContext] = field(default_factory=list)
    relationships: List[RelationshipContext] = field(default_factory=list)
    source_episodes: Set[str] = field(default_factory=set)
    source_books: Set[str] = field(default_factory=set)


class GraphRAGLocalSearch:
    """
    Local search using entity extraction and relationship traversal.

    The hierarchy contains:
    - 26,219 entities with descriptions, aliases, and source references
    - 39,118 relationships with types and weights
    """

    def __init__(
        self,
        hierarchy_path: Optional[str] = None
    ):
        self.hierarchy_path = hierarchy_path or "/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag_hierarchy.json"

        self.hierarchy = None
        self.entities = {}  # {entity_name: entity_data}
        self.relationships = []  # List of relationship dicts
        self.alias_index = {}  # {alias_lower: entity_name}

        self.is_initialized = False

        self.stats = {
            'total_searches': 0,
            'entities_matched': 0,
            'relationships_found': 0
        }

    def initialize(self):
        """Load hierarchy and build entity index"""
        logger.info("Initializing GraphRAG Local Search...")

        start_time = time.time()

        with open(self.hierarchy_path, 'r', encoding='utf-8') as f:
            self.hierarchy = json.load(f)

        load_time = time.time() - start_time
        logger.info(f"Hierarchy loaded in {load_time:.2f}s")

        # Extract entities from level_0 (individual entities)
        self._build_entity_index()

        # Extract relationships
        self._build_relationship_index()

        self.is_initialized = True
        logger.info(f"GraphRAG Local Search initialized with {len(self.entities)} entities and {len(self.relationships)} relationships")

    def _is_valid_alias(self, alias: str) -> bool:
        """
        Check if an alias is valid for indexing.

        Filters out:
        - Too short (< MIN_ALIAS_LENGTH characters)
        - In stoplist
        - Pure numbers
        - Single repeated characters
        """
        if not alias:
            return False

        alias_lower = alias.lower().strip()

        # Length check
        if len(alias_lower) < MIN_ALIAS_LENGTH:
            return False

        # Stoplist check
        if alias_lower in ENTITY_STOPLIST:
            return False

        # Pure numbers
        if alias_lower.isdigit():
            return False

        # Single repeated character (e.g., "aaa", "ooo")
        if len(set(alias_lower)) == 1:
            return False

        return True

    def _build_entity_index(self):
        """Build entity index from hierarchy with filtering"""
        # Entities are stored directly in the hierarchy root
        entities_data = self.hierarchy.get('entities', {})

        if not entities_data:
            # Try clusters level_0
            clusters = self.hierarchy.get('clusters', {})
            entities_data = clusters.get('level_0', {})

        skipped_short = 0
        skipped_stoplist = 0

        for entity_name, entity_data in entities_data.items():
            self.entities[entity_name] = entity_data

            # Build alias index for fast lookup
            # Add the main name (only if valid)
            if self._is_valid_alias(entity_name):
                self.alias_index[entity_name.lower()] = entity_name
            else:
                skipped_short += 1

            # Add all aliases (only if valid)
            aliases = entity_data.get('aliases', [])
            for alias in aliases:
                if self._is_valid_alias(alias):
                    self.alias_index[alias.lower()] = entity_name
                elif alias and len(alias) < MIN_ALIAS_LENGTH:
                    skipped_short += 1
                elif alias and alias.lower() in ENTITY_STOPLIST:
                    skipped_stoplist += 1

        logger.info(f"Built entity index with {len(self.entities)} entities and {len(self.alias_index)} aliases")
        logger.info(f"Filtered out {skipped_short} short aliases and {skipped_stoplist} stoplist entries")

    def _build_relationship_index(self):
        """Build relationship index from hierarchy"""
        self.relationships = self.hierarchy.get('relationships', [])
        logger.info(f"Loaded {len(self.relationships)} relationships")

    def find_entities_in_text(self, text: str) -> List[str]:
        """
        Extract entity names by matching aliases in the text.

        Uses word-boundary matching to avoid false positives:
        - Requires aliases to appear as whole words/phrases
        - Prioritizes longer matches to handle overlapping entity names
        - Filters out matches that are clearly noise
        """
        text_lower = text.lower()
        found_entities = set()

        # Sort aliases by length (longest first) to prioritize specific matches
        sorted_aliases = sorted(self.alias_index.keys(), key=len, reverse=True)

        for alias in sorted_aliases:
            # Skip if alias is in stoplist or too short (double-check at match time)
            if alias in ENTITY_STOPLIST or len(alias) < MIN_ALIAS_LENGTH:
                continue

            # Use word boundary matching for more accurate results
            # This prevents "soil" from matching in "soilless" or "topsoil"
            # But allows "Aaron Perry" to match in "I met Aaron Perry yesterday"
            pattern = r'\b' + re.escape(alias) + r'\b'

            if re.search(pattern, text_lower):
                entity_name = self.alias_index[alias]
                if entity_name not in found_entities:
                    found_entities.add(entity_name)

        return list(found_entities)

    def get_entity_context(self, entity_name: str) -> Optional[EntityContext]:
        """Get full context for an entity"""
        entity_data = self.entities.get(entity_name)

        if not entity_data:
            return None

        return EntityContext(
            name=entity_name,
            type=entity_data.get('type', 'ENTITY'),
            description=entity_data.get('description', ''),
            aliases=entity_data.get('aliases', [])[:5],  # Limit aliases shown
            sources=entity_data.get('sources', []),
            mention_count=entity_data.get('mention_count', 1)
        )

    def get_entity_relationships(
        self,
        entity_name: str,
        max_relationships: int = 10
    ) -> List[RelationshipContext]:
        """Get relationships involving an entity"""
        relationships = []

        for rel in self.relationships:
            source = rel.get('source', '')
            target = rel.get('target', '')

            if source == entity_name or target == entity_name:
                rel_ctx = RelationshipContext(
                    source=source,
                    predicate=rel.get('predicate', rel.get('type', 'RELATED_TO')),
                    target=target,
                    weight=rel.get('strength', rel.get('weight', 1.0)),
                    context=rel.get('context', '')
                )
                relationships.append(rel_ctx)

                if len(relationships) >= max_relationships:
                    break

        return relationships

    def expand_entity_neighborhood(
        self,
        entity_names: List[str],
        max_neighbors: int = 10
    ) -> Set[str]:
        """
        Expand to 1-hop neighbors of the given entities.

        Returns the set of neighboring entity names.
        """
        neighbors = set()

        for entity_name in entity_names:
            for rel in self.relationships:
                source = rel.get('source', '')
                target = rel.get('target', '')

                if source == entity_name and target not in entity_names:
                    neighbors.add(target)
                elif target == entity_name and source not in entity_names:
                    neighbors.add(source)

                if len(neighbors) >= max_neighbors:
                    break

            if len(neighbors) >= max_neighbors:
                break

        return neighbors

    def search(
        self,
        query: str,
        k_entities: int = 10,
        k_relationships: int = 20,
        expand_neighbors: bool = True
    ) -> LocalSearchResult:
        """
        Perform local search based on entity extraction.

        Args:
            query: User's question
            k_entities: Maximum entities to return
            k_relationships: Maximum relationships to return
            expand_neighbors: Whether to include 1-hop neighbors

        Returns:
            LocalSearchResult with entities, relationships, and sources
        """
        if not self.is_initialized:
            raise RuntimeError("GraphRAG Local Search not initialized")

        self.stats['total_searches'] += 1

        # Step 1: Extract entities from query
        matched_entities = self.find_entities_in_text(query)
        logger.info(f"Matched {len(matched_entities)} entities in query")

        if not matched_entities:
            return LocalSearchResult()

        # Step 2: Optionally expand to neighbors
        all_entity_names = set(matched_entities)
        if expand_neighbors and len(matched_entities) < k_entities:
            neighbors = self.expand_entity_neighborhood(
                matched_entities,
                max_neighbors=k_entities - len(matched_entities)
            )
            all_entity_names.update(neighbors)

        # Step 3: Get entity contexts
        entities = []
        source_episodes = set()
        source_books = set()

        for entity_name in list(all_entity_names)[:k_entities]:
            ctx = self.get_entity_context(entity_name)
            if ctx:
                # Mark directly matched entities with higher relevance
                if entity_name in matched_entities:
                    ctx.relevance_score = 1.0
                else:
                    ctx.relevance_score = 0.7  # Neighbors have lower relevance

                entities.append(ctx)

                # Collect sources
                for source in ctx.sources:
                    if source.startswith('episode_'):
                        source_episodes.add(source)
                    elif source.startswith('book_'):
                        source_books.add(source)

        # Sort by relevance and mention count
        entities.sort(key=lambda e: (e.relevance_score, e.mention_count), reverse=True)

        self.stats['entities_matched'] += len(entities)

        # Step 4: Get relationships between matched entities
        relationships = []
        seen_rels = set()

        for entity_name in matched_entities:
            rels = self.get_entity_relationships(entity_name, max_relationships=5)
            for rel in rels:
                rel_key = (rel.source, rel.predicate, rel.target)
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    relationships.append(rel)

            if len(relationships) >= k_relationships:
                break

        self.stats['relationships_found'] += len(relationships)

        logger.info(f"Local search found {len(entities)} entities and {len(relationships)} relationships")

        return LocalSearchResult(
            entities=entities,
            relationships=relationships,
            source_episodes=source_episodes,
            source_books=source_books
        )

    def format_entity_context(
        self,
        entities: List[EntityContext],
        max_description_length: int = 300
    ) -> str:
        """
        Format entity contexts into a string for LLM consumption.
        """
        if not entities:
            return ""

        context_parts = []

        for entity in entities:
            description = entity.description[:max_description_length]
            if len(entity.description) > max_description_length:
                description += "..."

            source_preview = ", ".join(entity.sources[:3])
            if len(entity.sources) > 3:
                source_preview += f" (+{len(entity.sources) - 3} more)"

            part = f"""
Entity: {entity.name}
Type: {entity.type}
Description: {description}
Sources: {source_preview}
Mentions: {entity.mention_count}
"""
            context_parts.append(part)

        return "\n".join(context_parts)

    def _humanize_predicate(self, predicate: str) -> str:
        """
        Convert relationship type to natural language.

        Examples:
        - FOUNDED -> "founded"
        - WORKS_FOR -> "works for"
        - ADVOCATES_FOR -> "advocates for"
        - RELATED_TO -> "is related to"
        """
        # Handle common relationship types
        predicate_map = {
            'FOUNDED': 'founded',
            'WORKS_FOR': 'works for',
            'WORKS_WITH': 'works with',
            'ADVOCATES_FOR': 'advocates for',
            'PRACTICES': 'practices',
            'CREATED': 'created',
            'LEADS': 'leads',
            'MEMBER_OF': 'is a member of',
            'PART_OF': 'is part of',
            'LOCATED_IN': 'is located in',
            'RELATED_TO': 'is related to',
            'DISCUSSES': 'discusses',
            'INTERVIEWED': 'interviewed',
            'AUTHORED': 'authored',
            'SUPPORTS': 'supports',
            'OPPOSES': 'opposes',
            'IMPLEMENTS': 'implements',
        }

        upper_pred = predicate.upper().replace('-', '_').replace(' ', '_')
        if upper_pred in predicate_map:
            return predicate_map[upper_pred]

        # Default: lowercase and replace underscores with spaces
        return predicate.lower().replace('_', ' ')

    def format_relationship_context(
        self,
        relationships: List[RelationshipContext]
    ) -> str:
        """
        Format relationship contexts into natural language for LLM consumption.
        """
        if not relationships:
            return ""

        context_parts = []

        for rel in relationships:
            predicate_text = self._humanize_predicate(rel.predicate)

            # Build natural language sentence
            sentence = f"- {rel.source} {predicate_text} {rel.target}"

            # Add context if available
            if rel.context and len(rel.context) > 10:
                # Truncate long contexts
                ctx = rel.context[:150] + "..." if len(rel.context) > 150 else rel.context
                sentence += f" ({ctx})"

            context_parts.append(sentence)

        return "Relationships:\n" + "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            **self.stats,
            'entities_indexed': len(self.entities),
            'aliases_indexed': len(self.alias_index),
            'relationships_indexed': len(self.relationships)
        }


# Singleton instance for reuse
_local_search_instance: Optional[GraphRAGLocalSearch] = None


def get_local_search() -> GraphRAGLocalSearch:
    """Get or create the local search singleton"""
    global _local_search_instance

    if _local_search_instance is None:
        _local_search_instance = GraphRAGLocalSearch()
        _local_search_instance.initialize()

    return _local_search_instance
