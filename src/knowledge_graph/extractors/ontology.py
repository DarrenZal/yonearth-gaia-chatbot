"""
Knowledge Graph Ontology Schema for YonEarth Podcast Content

This module provides dataclasses for entities and relationships,
importing type definitions from the parent ontology module.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import from consolidated parent ontology
from ..ontology import EntityType, RelationshipType, Domain


@dataclass
class Entity:
    """
    Represents an entity extracted from podcast content.

    Attributes:
        name: The canonical name of the entity
        entity_type: The type classification (PERSON, FORMAL_ORGANIZATION, etc.)
        domains: Relevant knowledge domains this entity relates to
        description: A concise description of the entity
        aliases: Alternative names or spellings
        properties: Structured properties specific to entity type
        metadata: Additional structured data about the entity
        source_episodes: Episode numbers where this entity appears
        source_chunks: Specific chunk IDs containing mentions
        confidence: Extraction confidence score (0.0 to 1.0)
        created_at: Timestamp of entity creation
        updated_at: Timestamp of last update
    """
    name: str
    entity_type: EntityType
    domains: List[Domain] = field(default_factory=list)
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_episodes: List[int] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamps and normalize entity type."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

        # Normalize entity type if string
        if isinstance(self.entity_type, str):
            self.entity_type = EntityType.normalize(self.entity_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            'name': self.name,
            'entity_type': self.entity_type.value,
            'domains': [d.value for d in self.domains],
            'description': self.description,
            'aliases': self.aliases,
            'properties': self.properties,
            'metadata': self.metadata,
            'source_episodes': self.source_episodes,
            'source_chunks': self.source_chunks,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary representation."""
        entity_type = data.get('entity_type', 'CONCEPT')
        if isinstance(entity_type, str):
            entity_type = EntityType.normalize(entity_type)

        # Handle domains that might not be in the enum
        domains = []
        for d in data.get('domains', []):
            try:
                domains.append(Domain(d))
            except ValueError:
                pass  # Skip invalid domains

        return cls(
            name=data['name'],
            entity_type=entity_type,
            domains=domains,
            description=data.get('description', ''),
            aliases=data.get('aliases', []),
            properties=data.get('properties', {}),
            metadata=data.get('metadata', {}),
            source_episodes=data.get('source_episodes', []),
            source_chunks=data.get('source_chunks', []),
            confidence=data.get('confidence', 1.0),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
        )


@dataclass
class Relationship:
    """
    Represents a relationship between two entities.

    Attributes:
        source_entity: The entity at the start of the relationship
        target_entity: The entity at the end of the relationship
        relationship_type: The nature of the relationship (FOUNDED, WORKS_FOR, etc.)
        description: Context about this specific relationship
        properties: Structured properties specific to relationship type
        domains: Relevant knowledge domains for this relationship
        source_episodes: Episode numbers where this relationship is mentioned
        source_chunks: Specific chunk IDs containing the relationship
        confidence: Extraction confidence score (0.0 to 1.0)
        metadata: Additional structured data
        created_at: Timestamp of relationship creation
        updated_at: Timestamp of last update
    """
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    domains: List[Domain] = field(default_factory=list)
    source_episodes: List[int] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamps and normalize relationship type."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

        # Normalize relationship type if string
        if isinstance(self.relationship_type, str):
            self.relationship_type = RelationshipType.normalize(self.relationship_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            'source_entity': self.source_entity,
            'target_entity': self.target_entity,
            'relationship_type': self.relationship_type.value,
            'description': self.description,
            'properties': self.properties,
            'domains': [d.value for d in self.domains],
            'source_episodes': self.source_episodes,
            'source_chunks': self.source_chunks,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create relationship from dictionary representation."""
        rel_type = data.get('relationship_type', 'RELATES_TO')
        if isinstance(rel_type, str):
            rel_type = RelationshipType.normalize(rel_type)

        # Handle domains that might not be in the enum
        domains = []
        for d in data.get('domains', []):
            try:
                domains.append(Domain(d))
            except ValueError:
                pass  # Skip invalid domains

        return cls(
            source_entity=data['source_entity'],
            target_entity=data['target_entity'],
            relationship_type=rel_type,
            description=data.get('description', ''),
            properties=data.get('properties', {}),
            domains=domains,
            source_episodes=data.get('source_episodes', []),
            source_chunks=data.get('source_chunks', []),
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
        )


# Backwards compatibility - relationship type descriptions
RELATIONSHIP_TYPES = {
    'FOUNDED': 'Person founded organization',
    'WORKS_FOR': 'Person works for organization',
    'LEADS': 'Person leads organization',
    'MEMBER_OF': 'Entity is member of group',
    'HAS_COMMUNITY': 'Organization has community',
    'PRODUCES': 'Entity produces product',
    'AUTHORED': 'Person authored work',
    'HAS_WEBSITE': 'Entity has website',
    'LOCATED_IN': 'Entity located in place',
    'FOCUSES_ON': 'Entity focuses on concept',
    'ADVOCATES_FOR': 'Entity advocates for concept',
    'INTERVIEWED_ON': 'Person interviewed on show',
    'PARTNERS_WITH': 'Organization partners with organization',
    'RELATES_TO': 'Concept relates to concept',
    'ENABLES': 'Thing enables another thing',
    'PART_OF': 'Entity is part of larger entity',
    'MENTIONED_IN': 'Entity mentioned in media',
    # Legacy types for backwards compatibility
    'founded_by': 'Organization founded by person',
    'works_for': 'Person works for organization',
    'collaborates_with': 'Person/organization collaborates with another',
    'mentored_by': 'Person mentored by another person',
    'implements': 'Practice implements a concept',
    'related_to': 'General relationship between concepts',
    'part_of': 'Component of a larger system',
    'depends_on': 'One concept depends on another',
    'enables': 'One thing enables another',
    'influences': 'One thing influences another',
    'uses_technology': 'Practice uses a technology',
    'developed_by': 'Technology developed by person/org',
    'applied_in': 'Technology/practice applied in domain',
    'located_in': 'Entity located in a place',
    'operates_in': 'Organization operates in location',
    'originated_from': 'Concept/practice originated from location',
    'contributes_to': 'Entity contributes to a domain',
    'addresses': 'Practice/technology addresses a problem',
    'exemplifies': 'Entity exemplifies a concept',
}


def get_relationship_description(rel_type: str) -> str:
    """Get human-readable description for a relationship type."""
    # Try uppercase first (new format)
    upper_type = rel_type.upper().replace(" ", "_").replace("-", "_")
    if upper_type in RELATIONSHIP_TYPES:
        return RELATIONSHIP_TYPES[upper_type]
    # Try original case (legacy format)
    return RELATIONSHIP_TYPES.get(rel_type, 'Related entities')
