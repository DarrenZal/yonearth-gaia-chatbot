"""
Knowledge Graph Ontology Definition.

This module defines the entity types, relationship types, and their properties
for the YonEarth podcast knowledge graph.
"""

from enum import Enum
from typing import Dict, List, Set


class EntityType(Enum):
    """Entity types in the knowledge graph."""

    # People
    PERSON = "Person"

    # Organizations
    ORGANIZATION = "Organization"
    COMPANY = "Company"
    NONPROFIT = "Nonprofit"
    UNIVERSITY = "University"
    GOVERNMENT_AGENCY = "GovernmentAgency"

    # Locations
    LOCATION = "Location"
    CITY = "City"
    STATE = "State"
    COUNTRY = "Country"

    # Concepts
    CONCEPT = "Concept"
    TECHNOLOGY = "Technology"
    PRACTICE = "Practice"
    MATERIAL = "Material"
    PRODUCT = "Product"

    # Topics
    TOPIC = "Topic"

    # Media
    EPISODE = "Episode"
    BOOK = "Book"
    PAPER = "Paper"


class RelationshipType(Enum):
    """Relationship types in the knowledge graph."""

    # Organizational relationships
    WORKS_AT = "works_at"
    FOUNDED = "founded"
    LEADS = "leads"
    PARTNERS_WITH = "partners_with"
    FUNDED_BY = "funded_by"
    MEMBER_OF = "member_of"

    # Knowledge relationships
    DISCUSSES = "discusses"
    EXPLAINS = "explains"
    ADVOCATES_FOR = "advocates_for"
    RESEARCHES = "researches"
    TEACHES = "teaches"

    # Conceptual relationships
    RELATES_TO = "relates_to"
    ENABLES = "enables"
    REQUIRES = "requires"
    IMPROVES = "improves"
    COMBINES_WITH = "combines_with"
    PART_OF = "part_of"
    PRODUCES = "produces"
    USES = "uses"

    # Spatial/Temporal relationships
    LOCATED_IN = "located_in"
    MENTIONED_IN = "mentioned_in"
    APPEARED_IN = "appeared_in"

    # Scientific relationships
    CAUSES = "causes"
    PREVENTS = "prevents"
    MEASURED_BY = "measured_by"


class RelationshipSchema:
    """Schema for relationship types including valid source/target entity types."""

    SCHEMAS: Dict[RelationshipType, Dict[str, any]] = {
        # Organizational relationships
        RelationshipType.WORKS_AT: {
            "source_types": [EntityType.PERSON],
            "target_types": [EntityType.ORGANIZATION, EntityType.COMPANY,
                           EntityType.UNIVERSITY, EntityType.NONPROFIT],
            "description": "Person works at or is employed by organization",
            "properties": ["role", "start_date", "end_date"]
        },
        RelationshipType.FOUNDED: {
            "source_types": [EntityType.PERSON],
            "target_types": [EntityType.ORGANIZATION, EntityType.COMPANY, EntityType.NONPROFIT],
            "description": "Person founded organization",
            "properties": ["year", "location"]
        },
        RelationshipType.LEADS: {
            "source_types": [EntityType.PERSON],
            "target_types": [EntityType.ORGANIZATION, EntityType.COMPANY,
                           EntityType.UNIVERSITY, EntityType.NONPROFIT],
            "description": "Person leads or directs organization",
            "properties": ["title", "start_date"]
        },
        RelationshipType.PARTNERS_WITH: {
            "source_types": [EntityType.ORGANIZATION, EntityType.COMPANY, EntityType.NONPROFIT],
            "target_types": [EntityType.ORGANIZATION, EntityType.COMPANY, EntityType.NONPROFIT],
            "description": "Organization partners with another organization",
            "properties": ["partnership_type", "start_date"]
        },
        RelationshipType.FUNDED_BY: {
            "source_types": [EntityType.ORGANIZATION, EntityType.COMPANY, EntityType.NONPROFIT],
            "target_types": [EntityType.ORGANIZATION, EntityType.COMPANY,
                           EntityType.GOVERNMENT_AGENCY],
            "description": "Organization receives funding from another entity",
            "properties": ["amount", "year", "program"]
        },

        # Knowledge relationships
        RelationshipType.DISCUSSES: {
            "source_types": [EntityType.PERSON, EntityType.EPISODE],
            "target_types": [EntityType.CONCEPT, EntityType.TECHNOLOGY,
                           EntityType.PRACTICE, EntityType.TOPIC],
            "description": "Person or episode discusses a concept",
            "properties": ["context", "depth"]
        },
        RelationshipType.EXPLAINS: {
            "source_types": [EntityType.PERSON, EntityType.EPISODE],
            "target_types": [EntityType.CONCEPT, EntityType.TECHNOLOGY, EntityType.PRACTICE],
            "description": "Person or episode explains how something works",
            "properties": ["detail_level"]
        },
        RelationshipType.ADVOCATES_FOR: {
            "source_types": [EntityType.PERSON, EntityType.ORGANIZATION],
            "target_types": [EntityType.PRACTICE, EntityType.CONCEPT, EntityType.TECHNOLOGY],
            "description": "Person or organization advocates for a practice or concept",
            "properties": ["strength", "context"]
        },
        RelationshipType.RESEARCHES: {
            "source_types": [EntityType.PERSON, EntityType.ORGANIZATION, EntityType.UNIVERSITY],
            "target_types": [EntityType.CONCEPT, EntityType.TECHNOLOGY, EntityType.MATERIAL],
            "description": "Entity conducts research on a topic",
            "properties": ["focus_area", "duration"]
        },

        # Conceptual relationships
        RelationshipType.RELATES_TO: {
            "source_types": [EntityType.CONCEPT, EntityType.TECHNOLOGY,
                           EntityType.PRACTICE, EntityType.TOPIC],
            "target_types": [EntityType.CONCEPT, EntityType.TECHNOLOGY,
                           EntityType.PRACTICE, EntityType.TOPIC],
            "description": "Concept relates to another concept",
            "properties": ["relationship_nature"]
        },
        RelationshipType.ENABLES: {
            "source_types": [EntityType.TECHNOLOGY, EntityType.PRACTICE, EntityType.CONCEPT],
            "target_types": [EntityType.TECHNOLOGY, EntityType.PRACTICE, EntityType.CONCEPT],
            "description": "Something enables or makes possible another thing",
            "properties": ["mechanism"]
        },
        RelationshipType.REQUIRES: {
            "source_types": [EntityType.TECHNOLOGY, EntityType.PRACTICE, EntityType.PRODUCT],
            "target_types": [EntityType.MATERIAL, EntityType.TECHNOLOGY, EntityType.CONCEPT],
            "description": "Something requires another thing to function",
            "properties": ["requirement_type"]
        },
        RelationshipType.IMPROVES: {
            "source_types": [EntityType.TECHNOLOGY, EntityType.PRACTICE, EntityType.MATERIAL],
            "target_types": [EntityType.CONCEPT, EntityType.PRACTICE, EntityType.MATERIAL],
            "description": "Something improves or enhances another thing",
            "properties": ["improvement_type", "magnitude"]
        },
        RelationshipType.PRODUCES: {
            "source_types": [EntityType.TECHNOLOGY, EntityType.PRACTICE, EntityType.ORGANIZATION],
            "target_types": [EntityType.PRODUCT, EntityType.MATERIAL],
            "description": "Entity produces a product or material",
            "properties": ["quantity", "method"]
        },
        RelationshipType.USES: {
            "source_types": [EntityType.TECHNOLOGY, EntityType.PRACTICE, EntityType.ORGANIZATION],
            "target_types": [EntityType.MATERIAL, EntityType.TECHNOLOGY, EntityType.PRODUCT],
            "description": "Entity uses a material, technology, or product",
            "properties": ["purpose", "method"]
        },

        # Spatial/Temporal relationships
        RelationshipType.LOCATED_IN: {
            "source_types": [EntityType.ORGANIZATION, EntityType.COMPANY,
                           EntityType.UNIVERSITY, EntityType.PERSON],
            "target_types": [EntityType.LOCATION, EntityType.CITY,
                           EntityType.STATE, EntityType.COUNTRY],
            "description": "Entity is located in a place",
            "properties": ["address"]
        },
        RelationshipType.MENTIONED_IN: {
            "source_types": [EntityType.PERSON, EntityType.ORGANIZATION,
                           EntityType.CONCEPT, EntityType.TECHNOLOGY],
            "target_types": [EntityType.EPISODE, EntityType.BOOK],
            "description": "Entity is mentioned in media",
            "properties": ["context", "timestamp"]
        },
        RelationshipType.APPEARED_IN: {
            "source_types": [EntityType.PERSON],
            "target_types": [EntityType.EPISODE],
            "description": "Person appeared as guest in episode",
            "properties": ["role", "date"]
        },

        # Scientific relationships
        RelationshipType.CAUSES: {
            "source_types": [EntityType.CONCEPT, EntityType.PRACTICE,
                           EntityType.TECHNOLOGY, EntityType.MATERIAL],
            "target_types": [EntityType.CONCEPT],
            "description": "Something causes an effect",
            "properties": ["mechanism", "evidence"]
        },
        RelationshipType.PREVENTS: {
            "source_types": [EntityType.PRACTICE, EntityType.TECHNOLOGY, EntityType.MATERIAL],
            "target_types": [EntityType.CONCEPT],
            "description": "Something prevents or mitigates a problem",
            "properties": ["mechanism", "effectiveness"]
        },
    }

    @classmethod
    def get_valid_relationships(cls, source_type: EntityType,
                               target_type: EntityType) -> List[RelationshipType]:
        """Get valid relationship types between two entity types."""
        valid = []
        for rel_type, schema in cls.SCHEMAS.items():
            if (source_type in schema["source_types"] and
                target_type in schema["target_types"]):
                valid.append(rel_type)
        return valid

    @classmethod
    def is_valid_relationship(cls, relationship_type: RelationshipType,
                            source_type: EntityType,
                            target_type: EntityType) -> bool:
        """Check if a relationship type is valid between two entity types."""
        if relationship_type not in cls.SCHEMAS:
            return False

        schema = cls.SCHEMAS[relationship_type]
        return (source_type in schema["source_types"] and
                target_type in schema["target_types"])


# Predefined topics for YonEarth content
YONEARTH_TOPICS = {
    "REGENERATIVE_AGRICULTURE": "Regenerative Agriculture",
    "SOIL_HEALTH": "Soil Health",
    "CARBON_SEQUESTRATION": "Carbon Sequestration",
    "BIOCHAR": "Biochar",
    "COMPOSTING": "Composting",
    "PERMACULTURE": "Permaculture",
    "CLIMATE_CHANGE": "Climate Change",
    "SUSTAINABILITY": "Sustainability",
    "ORGANIC_FARMING": "Organic Farming",
    "WATER_CONSERVATION": "Water Conservation",
    "BIODIVERSITY": "Biodiversity",
    "RENEWABLE_ENERGY": "Renewable Energy",
    "CIRCULAR_ECONOMY": "Circular Economy",
    "INDIGENOUS_KNOWLEDGE": "Indigenous Knowledge",
    "COMMUNITY_RESILIENCE": "Community Resilience",
    "FOOD_SYSTEMS": "Food Systems",
    "ECOSYSTEM_RESTORATION": "Ecosystem Restoration",
}
