"""
Knowledge Graph Ontology Schema for YonEarth Podcast Content

This module defines the entity types, domains, and data structures
used for extracting and organizing knowledge from podcast transcripts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime


class EntityType(str, Enum):
    """
    Core entity types in the YonEarth knowledge graph.
    """
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    PRACTICE = "practice"
    TECHNOLOGY = "technology"
    LOCATION = "location"


class Domain(str, Enum):
    """
    Knowledge domains representing topical areas in regenerative systems.
    """
    ECONOMY = "economy"
    ECOLOGY = "ecology"
    COMMUNITY = "community"
    CULTURE = "culture"
    HEALTH = "health"
    TECHNOLOGY = "technology"
    POLICY = "policy"
    SCIENCE = "science"
    SPIRITUALITY = "spirituality"
    FOOD_SYSTEMS = "food_systems"


@dataclass
class Entity:
    """
    Represents an entity extracted from podcast content.

    Attributes:
        name: The canonical name of the entity
        entity_type: The type classification (person, organization, etc.)
        domains: Relevant knowledge domains this entity relates to
        description: A concise description of the entity
        aliases: Alternative names or spellings
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_episodes: List[int] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            'name': self.name,
            'entity_type': self.entity_type.value,
            'domains': [d.value for d in self.domains],
            'description': self.description,
            'aliases': self.aliases,
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
        return cls(
            name=data['name'],
            entity_type=EntityType(data['entity_type']),
            domains=[Domain(d) for d in data.get('domains', [])],
            description=data.get('description', ''),
            aliases=data.get('aliases', []),
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
        relationship_type: The nature of the relationship
        description: Context about this specific relationship
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
    relationship_type: str
    description: str = ""
    domains: List[Domain] = field(default_factory=list)
    source_episodes: List[int] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            'source_entity': self.source_entity,
            'target_entity': self.target_entity,
            'relationship_type': self.relationship_type,
            'description': self.description,
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
        return cls(
            source_entity=data['source_entity'],
            target_entity=data['target_entity'],
            relationship_type=data['relationship_type'],
            description=data.get('description', ''),
            domains=[Domain(d) for d in data.get('domains', [])],
            source_episodes=data.get('source_episodes', []),
            source_chunks=data.get('source_chunks', []),
            confidence=data.get('confidence', 1.0),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
        )


# Common relationship types
RELATIONSHIP_TYPES = {
    # People and Organizations
    'founded_by': 'Organization founded by person',
    'works_for': 'Person works for organization',
    'collaborates_with': 'Person/organization collaborates with another',
    'mentored_by': 'Person mentored by another person',

    # Concepts and Practices
    'implements': 'Practice implements a concept',
    'related_to': 'General relationship between concepts',
    'part_of': 'Component of a larger system',
    'depends_on': 'One concept depends on another',
    'enables': 'One thing enables another',
    'influences': 'One thing influences another',

    # Technology and Practices
    'uses_technology': 'Practice uses a technology',
    'developed_by': 'Technology developed by person/org',
    'applied_in': 'Technology/practice applied in domain',

    # Locations
    'located_in': 'Entity located in a place',
    'operates_in': 'Organization operates in location',
    'originated_from': 'Concept/practice originated from location',

    # Domain relationships
    'contributes_to': 'Entity contributes to a domain',
    'addresses': 'Practice/technology addresses a problem',
    'exemplifies': 'Entity exemplifies a concept',
}


def get_relationship_description(rel_type: str) -> str:
    """Get human-readable description for a relationship type."""
    return RELATIONSHIP_TYPES.get(rel_type, 'Related entities')
