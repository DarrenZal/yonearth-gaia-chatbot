"""
Knowledge Graph Ontology Definition.

This module loads the ontology from the master JSON schema and provides
Python enums and classes for type-safe knowledge graph operations.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


def _load_master_ontology() -> Dict:
    """Load the master ontology from JSON."""
    ontology_path = Path(__file__).parent.parent.parent / "data" / "ontology" / "yonearth_ontology.json"
    if ontology_path.exists():
        with open(ontology_path) as f:
            return json.load(f)
    else:
        # Fallback to hardcoded if file not found
        return _get_fallback_ontology()


def _get_fallback_ontology() -> Dict:
    """Fallback ontology if JSON file not available."""
    return {
        "entity_types": {
            "PERSON": {"label": "Person"},
            "FORMAL_ORGANIZATION": {"label": "Formal Organization"},
            "COMMUNITY": {"label": "Community"},
            "NETWORK": {"label": "Network"},
            "PLACE": {"label": "Place"},
            "CONCEPT": {"label": "Concept"},
            "PRODUCT": {"label": "Product"},
            "URL": {"label": "URL"},
        },
        "relationship_types": {},
        "domains": {}
    }


# Load master ontology
_MASTER_ONTOLOGY = _load_master_ontology()


class EntityType(str, Enum):
    """Entity types in the knowledge graph."""

    # Core types from master ontology
    PERSON = "PERSON"
    FORMAL_ORGANIZATION = "FORMAL_ORGANIZATION"
    COMMUNITY = "COMMUNITY"
    NETWORK = "NETWORK"
    PLACE = "PLACE"
    CONCEPT = "CONCEPT"
    PRODUCT = "PRODUCT"
    URL = "URL"

    # Subtypes (for backwards compatibility and specificity)
    ORGANIZATION = "ORGANIZATION"  # Generic, maps to FORMAL_ORGANIZATION
    COMPANY = "COMPANY"
    NONPROFIT = "NONPROFIT"
    UNIVERSITY = "UNIVERSITY"
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"
    CITY = "CITY"
    STATE = "STATE"
    COUNTRY = "COUNTRY"
    REGION = "REGION"
    LOCATION = "LOCATION"  # Legacy, maps to PLACE
    PRACTICE = "PRACTICE"
    TECHNOLOGY = "TECHNOLOGY"
    MATERIAL = "MATERIAL"
    TOPIC = "TOPIC"
    MOVEMENT = "MOVEMENT"
    BOOK = "BOOK"
    EPISODE = "EPISODE"
    PODCAST = "PODCAST"
    ARTICLE = "ARTICLE"
    PAPER = "PAPER"  # Legacy, maps to PRODUCT

    @classmethod
    def normalize(cls, entity_type: str) -> 'EntityType':
        """Normalize an entity type string to canonical form."""
        type_upper = entity_type.upper().replace(" ", "_")

        # Map subtypes to parent types for extraction
        subtype_mapping = {
            "ORGANIZATION": cls.FORMAL_ORGANIZATION,
            "COMPANY": cls.FORMAL_ORGANIZATION,
            "NONPROFIT": cls.FORMAL_ORGANIZATION,
            "UNIVERSITY": cls.FORMAL_ORGANIZATION,
            "GOVERNMENT_AGENCY": cls.FORMAL_ORGANIZATION,
            "GOVERNMENTAGENCY": cls.FORMAL_ORGANIZATION,
            "CITY": cls.PLACE,
            "STATE": cls.PLACE,
            "COUNTRY": cls.PLACE,
            "REGION": cls.PLACE,
            "LOCATION": cls.PLACE,
            "LANDMARK": cls.PLACE,
            "PRACTICE": cls.CONCEPT,
            "TECHNOLOGY": cls.CONCEPT,
            "MATERIAL": cls.CONCEPT,
            "TOPIC": cls.CONCEPT,
            "MOVEMENT": cls.CONCEPT,
            "BOOK": cls.PRODUCT,
            "EPISODE": cls.PRODUCT,
            "PODCAST": cls.PRODUCT,
            "ARTICLE": cls.PRODUCT,
            "PAPER": cls.PRODUCT,
            "TOOL": cls.PRODUCT,
        }

        if type_upper in subtype_mapping:
            return subtype_mapping[type_upper]

        try:
            return cls(type_upper)
        except ValueError:
            return cls.CONCEPT  # Default fallback


class RelationshipType(str, Enum):
    """Relationship types in the knowledge graph."""

    # Core relationships
    FOUNDED = "FOUNDED"
    WORKS_FOR = "WORKS_FOR"
    LEADS = "LEADS"
    MEMBER_OF = "MEMBER_OF"
    HAS_COMMUNITY = "HAS_COMMUNITY"
    PRODUCES = "PRODUCES"
    AUTHORED = "AUTHORED"
    HAS_WEBSITE = "HAS_WEBSITE"
    LOCATED_IN = "LOCATED_IN"
    FOCUSES_ON = "FOCUSES_ON"
    ADVOCATES_FOR = "ADVOCATES_FOR"
    INTERVIEWED_ON = "INTERVIEWED_ON"
    PARTNERS_WITH = "PARTNERS_WITH"
    PART_OF = "PART_OF"
    RELATES_TO = "RELATES_TO"
    ENABLES = "ENABLES"
    MENTIONED_IN = "MENTIONED_IN"

    # Legacy/backwards compatibility
    WORKS_AT = "WORKS_AT"  # Maps to WORKS_FOR
    DISCUSSES = "DISCUSSES"  # Maps to FOCUSES_ON
    EXPLAINS = "EXPLAINS"  # Maps to FOCUSES_ON
    APPEARED_IN = "APPEARED_IN"  # Maps to INTERVIEWED_ON
    RESEARCHES = "RESEARCHES"  # Maps to FOCUSES_ON
    TEACHES = "TEACHES"  # Maps to FOCUSES_ON
    FUNDED_BY = "FUNDED_BY"  # Keep for backwards compatibility
    REQUIRES = "REQUIRES"  # Maps to RELATES_TO
    IMPROVES = "IMPROVES"  # Maps to ENABLES
    COMBINES_WITH = "COMBINES_WITH"  # Maps to RELATES_TO
    USES = "USES"  # Maps to RELATES_TO
    CAUSES = "CAUSES"  # Maps to ENABLES
    PREVENTS = "PREVENTS"  # Maps to RELATES_TO
    MEASURED_BY = "MEASURED_BY"  # Maps to RELATES_TO

    @classmethod
    def normalize(cls, rel_type: str) -> 'RelationshipType':
        """Normalize a relationship type string to canonical form."""
        type_upper = rel_type.upper().replace(" ", "_").replace("-", "_")

        # Map legacy types
        legacy_mapping = {
            "WORKS_AT": cls.WORKS_FOR,
            "DISCUSSES": cls.FOCUSES_ON,
            "EXPLAINS": cls.FOCUSES_ON,
            "APPEARED_IN": cls.INTERVIEWED_ON,
            "FOUNDED_BY": cls.FOUNDED,  # Inverse
            "RESEARCHES": cls.FOCUSES_ON,
            "TEACHES": cls.FOCUSES_ON,
            "REQUIRES": cls.RELATES_TO,
            "IMPROVES": cls.ENABLES,
            "COMBINES_WITH": cls.RELATES_TO,
            "USES": cls.RELATES_TO,
            "CAUSES": cls.ENABLES,
            "PREVENTS": cls.RELATES_TO,
            "MEASURED_BY": cls.RELATES_TO,
        }

        if type_upper in legacy_mapping:
            return legacy_mapping[type_upper]

        try:
            return cls(type_upper)
        except ValueError:
            return cls.RELATES_TO  # Default fallback


class Domain(str, Enum):
    """Knowledge domains for categorizing entities and relationships."""

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


class RelationshipSchema:
    """Schema for relationship types including valid source/target entity types."""

    _schemas: Dict[RelationshipType, Dict[str, Any]] = None

    @classmethod
    def _load_schemas(cls) -> Dict[RelationshipType, Dict[str, Any]]:
        """Load relationship schemas from master ontology."""
        if cls._schemas is not None:
            return cls._schemas

        cls._schemas = {}
        rel_types = _MASTER_ONTOLOGY.get("relationship_types", {})

        for rel_name, rel_info in rel_types.items():
            try:
                rel_type = RelationshipType(rel_name)
                source_types = []
                target_types = []

                # Convert string type names to EntityType enums
                for t in rel_info.get("source_types", []):
                    try:
                        source_types.append(EntityType(t))
                    except ValueError:
                        pass

                for t in rel_info.get("target_types", []):
                    try:
                        target_types.append(EntityType(t))
                    except ValueError:
                        pass

                cls._schemas[rel_type] = {
                    "source_types": source_types,
                    "target_types": target_types,
                    "description": rel_info.get("description", ""),
                    "properties": rel_info.get("properties", [])
                }
            except (ValueError, KeyError):
                continue

        return cls._schemas

    @classmethod
    def get_schema(cls, relationship_type: RelationshipType) -> Optional[Dict[str, Any]]:
        """Get schema for a relationship type."""
        schemas = cls._load_schemas()
        return schemas.get(relationship_type)

    @classmethod
    def get_valid_relationships(cls, source_type: EntityType,
                                target_type: EntityType) -> List[RelationshipType]:
        """Get valid relationship types between two entity types."""
        schemas = cls._load_schemas()
        valid = []

        # Normalize to core types for matching
        source_normalized = EntityType.normalize(source_type.value)
        target_normalized = EntityType.normalize(target_type.value)

        for rel_type, schema in schemas.items():
            source_matches = (
                source_type in schema["source_types"] or
                source_normalized in schema["source_types"]
            )
            target_matches = (
                target_type in schema["target_types"] or
                target_normalized in schema["target_types"]
            )
            if source_matches and target_matches:
                valid.append(rel_type)
        return valid

    @classmethod
    def is_valid_relationship(cls, relationship_type: RelationshipType,
                              source_type: EntityType,
                              target_type: EntityType) -> bool:
        """Check if a relationship type is valid between two entity types."""
        schema = cls.get_schema(relationship_type)
        if schema is None:
            return False

        # Normalize to core types for matching
        source_normalized = EntityType.normalize(source_type.value)
        target_normalized = EntityType.normalize(target_type.value)

        source_matches = (
            source_type in schema["source_types"] or
            source_normalized in schema["source_types"]
        )
        target_matches = (
            target_type in schema["target_types"] or
            target_normalized in schema["target_types"]
        )
        return source_matches and target_matches

    # Backwards compatibility: SCHEMAS class variable
    @classmethod
    @property
    def SCHEMAS(cls) -> Dict[RelationshipType, Dict[str, Any]]:
        """Legacy accessor for schemas."""
        return cls._load_schemas()


def get_entity_info(entity_type: EntityType) -> Dict[str, Any]:
    """Get information about an entity type from the master ontology."""
    return _MASTER_ONTOLOGY.get("entity_types", {}).get(entity_type.value, {})


def get_relationship_info(rel_type: RelationshipType) -> Dict[str, Any]:
    """Get information about a relationship type from the master ontology."""
    return _MASTER_ONTOLOGY.get("relationship_types", {}).get(rel_type.value, {})


def get_predefined_topics() -> List[str]:
    """Get list of predefined topics from the ontology."""
    return _MASTER_ONTOLOGY.get("predefined_topics", [])


def get_domains() -> Dict[str, str]:
    """Get domains from the ontology."""
    return _MASTER_ONTOLOGY.get("domains", {})


# For backwards compatibility - export predefined topics dict
YONEARTH_TOPICS = {
    topic.upper().replace(" ", "_"): topic
    for topic in get_predefined_topics()
}
