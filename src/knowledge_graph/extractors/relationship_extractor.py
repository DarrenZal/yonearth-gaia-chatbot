"""
Relationship Extractor for Knowledge Graph Construction

This module extracts relationships between entities from podcast episode transcripts.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
from pydantic import BaseModel


class Relationship(BaseModel):
    """Represents a relationship between two entities"""
    source_entity: str
    relationship_type: str
    target_entity: str
    description: str
    metadata: Dict[str, Any] = {}


class RelationshipForExtraction(BaseModel):
    """Relationship schema for OpenAI structured output (without metadata)"""
    source_entity: str
    relationship_type: str
    target_entity: str
    description: str


class RelationshipExtractionResult(BaseModel):
    """Result of relationship extraction from a chunk"""
    relationships: List[Relationship]
    chunk_id: str = ""
    episode_number: int = 0


class RelationshipListResponse(BaseModel):
    """Schema for OpenAI structured output - list of relationships without metadata"""
    relationships: List[RelationshipForExtraction]


class RelationshipExtractor:
    """Extracts relationships between entities using OpenAI GPT models"""

    RELATIONSHIP_TYPES = [
        "FOUNDED",          # Person founded organization
        "WORKS_FOR",        # Person works for organization
        "LOCATED_IN",       # Organization/place located in place
        "PRACTICES",        # Person/org practices technique
        "PRODUCES",         # Organization produces product
        "USES",             # Person/org uses technology/product
        "ADVOCATES_FOR",    # Person advocates for concept
        "COLLABORATES_WITH", # Person/org collaborates with person/org
        "PART_OF",          # Organization part of larger organization
        "IMPLEMENTS",       # Organization implements practice
        "RESEARCHES",       # Person/org researches concept
        "TEACHES",          # Person teaches concept/practice
        "MENTIONS",         # General mention relationship
        "INFLUENCES",       # Entity influences another
        "RELATED_TO",       # General relationship
    ]

    EXTRACTION_PROMPT = """You are an expert at extracting relationships between entities from podcast transcripts about sustainability and regenerative agriculture.

Given this text and the entities that were extracted from it, identify ALL meaningful relationships between the entities.

For each relationship, provide:
1. source_entity: The entity name (must match an entity from the list)
2. relationship_type: One of {relationship_types}
3. target_entity: The entity name (must match an entity from the list)
4. description: A brief description of this specific relationship instance

Example format:
[
  {{
    "source_entity": "Joel Salatin",
    "relationship_type": "FOUNDED",
    "target_entity": "Polyface Farm",
    "description": "Joel Salatin founded and operates Polyface Farm in Virginia."
  }},
  {{
    "source_entity": "Polyface Farm",
    "relationship_type": "PRACTICES",
    "target_entity": "Rotational Grazing",
    "description": "Polyface Farm uses rotational grazing as a core practice for livestock management."
  }}
]

Entities present in this text:
{entities}

Text to analyze:
{text}

Return ONLY a valid JSON array of relationships. Return an empty array [] if no relationships are found."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the relationship extractor

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for extraction
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.rate_limit_delay = 0.05  # seconds between API calls (was 1.0 - very conservative)

    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        episode_number: int,
        chunk_id: str,
        retry_attempts: int = 3
    ) -> RelationshipExtractionResult:
        """Extract relationships from a text chunk

        Args:
            text: The text to extract relationships from
            entities: List of entities found in this text
            episode_number: Episode number for tracking
            chunk_id: Unique identifier for this chunk
            retry_attempts: Number of times to retry on failure

        Returns:
            RelationshipExtractionResult containing extracted relationships
        """
        # Format entities for the prompt
        entity_list = "\n".join([
            f"- {e['name']} ({e['type']})"
            for e in entities
        ])

        prompt = self.EXTRACTION_PROMPT.format(
            relationship_types=", ".join(self.RELATIONSHIP_TYPES),
            entities=entity_list if entity_list else "None",
            text=text
        )

        for attempt in range(retry_attempts):
            try:
                # Use structured outputs with Pydantic schema
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting relationships between entities."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format=RelationshipListResponse,
                    temperature=0.1,  # Low temperature for consistency
                )

                # Extract parsed response
                parsed = response.choices[0].message.parsed

                # Validate we got a response
                if not parsed or not parsed.relationships:
                    print(f"Empty or invalid response from OpenAI on attempt {attempt + 1}/{retry_attempts}")
                    if attempt < retry_attempts - 1:
                        continue
                    else:
                        print("All retry attempts returned empty responses")
                        return RelationshipExtractionResult(relationships=[])

                # Convert to Relationship objects with metadata
                relationships = []
                for rel_data in parsed.relationships:
                    relationship = Relationship(
                        source_entity=rel_data.source_entity,
                        relationship_type=rel_data.relationship_type,
                        target_entity=rel_data.target_entity,
                        description=rel_data.description,
                        metadata={
                            "episode_number": episode_number,
                            "chunk_id": chunk_id
                        }
                    )
                    relationships.append(relationship)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                return RelationshipExtractionResult(
                    relationships=relationships,
                    chunk_id=chunk_id,
                    episode_number=episode_number
                )

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == retry_attempts - 1:
                    raise
                time.sleep(2)
                continue

        # Return empty result if all attempts failed
        return RelationshipExtractionResult(
            relationships=[],
            chunk_id=chunk_id,
            episode_number=episode_number
        )

    def aggregate_relationships(self, results: List[RelationshipExtractionResult]) -> List[Relationship]:
        """Aggregate relationships from multiple chunks, deduplicating

        Args:
            results: List of extraction results from multiple chunks

        Returns:
            List of unique relationships
        """
        relationship_map = {}

        for result in results:
            for rel in result.relationships:
                # Create a key for deduplication
                key = (
                    rel.source_entity.lower().strip(),
                    rel.relationship_type,
                    rel.target_entity.lower().strip()
                )

                if key in relationship_map:
                    # Track multiple mentions
                    existing = relationship_map[key]
                    if "chunks" not in existing.metadata:
                        existing.metadata["chunks"] = []
                    existing.metadata["chunks"].append(rel.metadata.get("chunk_id"))

                    # Keep longer description
                    if len(rel.description) > len(existing.description):
                        existing.description = rel.description
                else:
                    # Add new relationship
                    rel.metadata["chunks"] = [rel.metadata.get("chunk_id")]
                    relationship_map[key] = rel

        return list(relationship_map.values())
