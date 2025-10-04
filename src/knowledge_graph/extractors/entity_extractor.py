"""
Entity Extractor for Knowledge Graph Construction

This module extracts entities from podcast episode transcripts using OpenAI's GPT models.
It identifies key entities like people, organizations, concepts, places, and more.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
from pydantic import BaseModel


class Entity(BaseModel):
    """Represents an extracted entity"""
    name: str
    type: str  # PERSON, ORGANIZATION, CONCEPT, PLACE, PRACTICE, PRODUCT, etc.
    description: str
    aliases: List[str] = []
    metadata: Dict[str, Any] = {}


class EntityForExtraction(BaseModel):
    """Entity schema for OpenAI structured output (without metadata)"""
    name: str
    type: str
    description: str
    aliases: List[str] = []


class EntityExtractionResult(BaseModel):
    """Result of entity extraction from a chunk"""
    entities: List[Entity]
    chunk_id: str = ""
    episode_number: int = 0


class EntityListResponse(BaseModel):
    """Schema for OpenAI structured output - list of entities without metadata"""
    entities: List[EntityForExtraction]


class EntityExtractor:
    """Extracts entities from text using OpenAI GPT models"""

    ENTITY_TYPES = [
        "PERSON",           # Individuals mentioned
        "ORGANIZATION",     # Companies, nonprofits, institutions
        "CONCEPT",          # Abstract ideas, theories, movements
        "PLACE",            # Locations, regions
        "PRACTICE",         # Techniques, methods, activities
        "PRODUCT",          # Specific products or services
        "EVENT",            # Conferences, gatherings, initiatives
        "TECHNOLOGY",       # Tools, software, hardware
        "ECOSYSTEM",        # Natural systems, biomes
        "SPECIES",          # Plants, animals, organisms
    ]

    EXTRACTION_PROMPT = """You are an expert at extracting structured entities from podcast transcripts about sustainability, regenerative agriculture, and environmental topics.

Extract ALL significant entities from this text. For each entity, provide:
1. name: The exact name/term as it appears
2. type: One of {entity_types}
3. description: A concise description (1-2 sentences) of what/who this is
4. aliases: Any alternative names or spellings mentioned (optional)

Focus on:
- People (guests, hosts, researchers, activists)
- Organizations (nonprofits, companies, institutions)
- Concepts (regenerative agriculture, permaculture, biochar, etc.)
- Places (farms, regions, countries, ecosystems)
- Practices (composting, cover cropping, soil testing)
- Products (specific brands, tools, materials)
- Events (conferences, gatherings, initiatives)
- Technologies (tools, software, equipment)
- Ecosystems (forests, wetlands, grasslands)
- Species (plants, animals, microorganisms)

Return ONLY a valid JSON array of entities. Example format:
[
  {{
    "name": "Joel Salatin",
    "type": "PERSON",
    "description": "Farmer and author known for regenerative agriculture and rotational grazing practices.",
    "aliases": ["Joel"]
  }},
  {{
    "name": "Polyface Farm",
    "type": "ORGANIZATION",
    "description": "Regenerative farm in Virginia demonstrating holistic management practices.",
    "aliases": []
  }},
  {{
    "name": "Rotational Grazing",
    "type": "PRACTICE",
    "description": "A livestock management technique where animals are moved between pastures to improve soil health.",
    "aliases": ["mob grazing", "management-intensive grazing"]
  }}
]

Text to analyze:
{text}

Return ONLY the JSON array, no other text."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the entity extractor

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

    def extract_entities(
        self,
        text: str,
        episode_number: int,
        chunk_id: str,
        retry_attempts: int = 3
    ) -> EntityExtractionResult:
        """Extract entities from a text chunk

        Args:
            text: The text to extract entities from
            episode_number: Episode number for tracking
            chunk_id: Unique identifier for this chunk
            retry_attempts: Number of times to retry on failure

        Returns:
            EntityExtractionResult containing extracted entities
        """
        prompt = self.EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.ENTITY_TYPES),
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
                            "content": "You are an expert at extracting structured information from text."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format=EntityListResponse,
                    temperature=0.1,  # Low temperature for consistency
                )

                # Extract parsed response
                parsed = response.choices[0].message.parsed

                # Validate we got a response
                if not parsed or not parsed.entities:
                    print(f"Empty or invalid response from OpenAI on attempt {attempt + 1}/{retry_attempts}")
                    if attempt < retry_attempts - 1:
                        continue
                    else:
                        print("All retry attempts returned empty responses")
                        return EntityExtractionResult(entities=[])

                # Convert to Entity objects with metadata
                entities = []
                for entity_data in parsed.entities:
                    entity = Entity(
                        name=entity_data.name,
                        type=entity_data.type,
                        description=entity_data.description,
                        aliases=entity_data.aliases,
                        metadata={
                            "episode_number": episode_number,
                            "chunk_id": chunk_id
                        }
                    )
                    entities.append(entity)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                return EntityExtractionResult(
                    entities=entities,
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
        return EntityExtractionResult(
            entities=[],
            chunk_id=chunk_id,
            episode_number=episode_number
        )

    def aggregate_entities(self, results: List[EntityExtractionResult]) -> List[Entity]:
        """Aggregate entities from multiple chunks, deduplicating similar entities

        Args:
            results: List of extraction results from multiple chunks

        Returns:
            List of unique entities with merged information
        """
        entity_map = {}

        for result in results:
            for entity in result.entities:
                # Create a normalized key for deduplication
                key = (entity.name.lower().strip(), entity.type)

                if key in entity_map:
                    # Merge with existing entity
                    existing = entity_map[key]

                    # Add any new aliases
                    for alias in entity.aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)

                    # Update metadata to track all chunks
                    if "chunks" not in existing.metadata:
                        existing.metadata["chunks"] = []
                    existing.metadata["chunks"].append(entity.metadata.get("chunk_id"))

                    # Keep the longer description
                    if len(entity.description) > len(existing.description):
                        existing.description = entity.description
                else:
                    # Add new entity
                    entity.metadata["chunks"] = [entity.metadata.get("chunk_id")]
                    entity_map[key] = entity

        return list(entity_map.values())
