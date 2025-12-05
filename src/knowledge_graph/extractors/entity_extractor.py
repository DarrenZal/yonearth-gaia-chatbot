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
    type: str  # PERSON, FORMAL_ORGANIZATION, COMMUNITY, NETWORK, CONCEPT, PLACE, PRODUCT, URL
    description: str
    aliases: List[str] = []
    metadata: Dict[str, Any] = {}


class RelationshipTriple(BaseModel):
    """Represents a relationship extracted between entities"""
    source: str
    predicate: str
    target: str


class EntityForExtraction(BaseModel):
    """Entity schema for OpenAI structured output (without metadata)"""
    name: str
    type: str
    description: str
    aliases: List[str] = []


class EntityExtractionResult(BaseModel):
    """Result of entity extraction from a chunk"""
    entities: List[Entity]
    relationships: List[RelationshipTriple] = []
    chunk_id: str = ""
    episode_number: int = 0


class ExtractionResponse(BaseModel):
    """Schema for OpenAI structured output - entities and relationships"""
    entities: List[EntityForExtraction]
    relationships: List[RelationshipTriple] = []


# Legacy alias for backwards compatibility
EntityListResponse = ExtractionResponse


class EntityExtractor:
    """Extracts entities from text using OpenAI GPT models"""

    # Core entity types from the consolidated ontology
    ENTITY_TYPES = [
        "PERSON",               # Named individuals only
        "FORMAL_ORGANIZATION",  # Legal entities: nonprofits, companies, agencies, universities
        "COMMUNITY",            # Informal groups based on shared interest
        "NETWORK",              # Distributed connections between entities
        "PLACE",                # Geographic locations
        "CONCEPT",              # Abstract ideas, practices, movements, topics
        "PRODUCT",              # Books, podcasts, tools, creations
        "URL",                  # Websites and domains
    ]

    # Core relationship types
    RELATIONSHIP_TYPES = [
        "FOUNDED",          # Person founded organization
        "WORKS_FOR",        # Person works for organization
        "LEADS",            # Person leads organization
        "MEMBER_OF",        # Entity is member of group
        "HAS_COMMUNITY",    # Organization has community
        "PRODUCES",         # Entity produces product
        "AUTHORED",         # Person authored book/article
        "HAS_WEBSITE",      # Entity has website
        "LOCATED_IN",       # Entity located in place
        "FOCUSES_ON",       # Entity focuses on concept
        "ADVOCATES_FOR",    # Entity advocates for cause
        "INTERVIEWED_ON",   # Person interviewed on show
        "PARTNERS_WITH",    # Org partners with org
        "PART_OF",          # Entity part of larger entity
        "RELATES_TO",       # Concepts relate to each other
    ]

    EXTRACTION_PROMPT = """You are an expert at extracting structured entities and relationships from podcast transcripts about sustainability, regenerative agriculture, and environmental topics.

=== ENTITY TYPES ===

Extract entities using these SPECIFIC types (not generic "ORGANIZATION"):

PERSON: Named individuals only
  - EXTRACT: "Aaron William Perry", "Vandana Shiva", "Paul Stamets"
  - DO NOT EXTRACT: "farmers", "scientists", "activists", "the speaker", "she", "he", "they"

FORMAL_ORGANIZATION: Legal entities (nonprofits, companies, agencies, universities)
  - EXTRACT: "Y on Earth", "Patagonia", "Rodale Institute", "EPA", "Stanford University"
  - Use for 501(c)(3)s, B-Corps, LLCs, government agencies

COMMUNITY: Informal groups based on shared interest (no formal legal status)
  - EXTRACT: "Y on Earth Community", "local permaculture guild", "transition town movement"

NETWORK: Distributed connections between entities
  - EXTRACT: "regenerative agriculture network", "B Corp network", "seed library network"

PLACE: Geographic locations
  - EXTRACT: "Boulder, Colorado", "Amazon rainforest", "Findhorn, Scotland"

CONCEPT: Abstract ideas, practices, movements, technologies, materials
  - EXTRACT: "regenerative agriculture", "permaculture", "biochar", "carbon sequestration"

PRODUCT: Books, podcasts, tools, creations
  - EXTRACT: "Y on Earth Podcast", "VIRIDITAS", "Mycelium Running"

URL: Websites and domains
  - EXTRACT: "yonearth.org", "patagonia.com"
  - NOTE: Always link URLs to their owner with HAS_WEBSITE relationship

=== RELATIONSHIPS ===

For each relationship found, extract a triple:
{{"source": "entity name", "predicate": "RELATIONSHIP_TYPE", "target": "entity name"}}

Valid predicates:
- FOUNDED: Person founded organization
- WORKS_FOR: Person works for organization
- LEADS: Person leads/directs organization
- MEMBER_OF: Entity is member of community/network
- HAS_COMMUNITY: Organization has associated community
- HAS_WEBSITE: Entity has website (use when URL mentioned)
- PRODUCES: Entity produces product
- AUTHORED: Person authored book/article
- LOCATED_IN: Entity located in place
- FOCUSES_ON: Entity focuses on concept
- ADVOCATES_FOR: Entity advocates for cause
- INTERVIEWED_ON: Person interviewed on show
- PARTNERS_WITH: Org partners with org
- PART_OF: Entity is part of larger entity
- RELATES_TO: Concepts relate to each other

=== SPECIAL HANDLING ===

URLs: When you see a website like "yonearth.org":
1. Extract: {{"name": "yonearth.org", "type": "URL"}}
2. Create: {{"source": "Y on Earth", "predicate": "HAS_WEBSITE", "target": "yonearth.org"}}

Org vs Community: These are SEPARATE entities that should be linked:
- "Y on Earth" (nonprofit) -> FORMAL_ORGANIZATION
- "Y on Earth Community" (the people) -> COMMUNITY
- Create: {{"source": "Y on Earth", "predicate": "HAS_COMMUNITY", "target": "Y on Earth Community"}}

=== DO NOT EXTRACT ===

- Pronouns: we, she, he, they, I
- Generic groups: farmers, scientists, activists, researchers
- Roles: the founder, the CEO, the speaker, the guest
- Combined names: "John and Jane Smith" -> extract as TWO separate PERSON entities

=== OUTPUT FORMAT ===

Return entities AND relationships. Example:
{{
  "entities": [
    {{"name": "Aaron William Perry", "type": "PERSON", "description": "Founder of Y on Earth", "aliases": ["Aaron Perry"]}},
    {{"name": "Y on Earth", "type": "FORMAL_ORGANIZATION", "description": "Nonprofit focused on regenerative sustainability", "aliases": []}},
    {{"name": "Y on Earth Community", "type": "COMMUNITY", "description": "Community of sustainability practitioners", "aliases": []}},
    {{"name": "yonearth.org", "type": "URL", "description": "Official website for Y on Earth", "aliases": []}}
  ],
  "relationships": [
    {{"source": "Aaron William Perry", "predicate": "FOUNDED", "target": "Y on Earth"}},
    {{"source": "Y on Earth", "predicate": "HAS_COMMUNITY", "target": "Y on Earth Community"}},
    {{"source": "Y on Earth", "predicate": "HAS_WEBSITE", "target": "yonearth.org"}}
  ]
}}

Text to analyze:
{text}

Extract all entities and relationships following the guidelines above."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the entity extractor

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for extraction
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        # Priority: explicit model > GRAPH_EXTRACTION_MODEL > OPENAI_MODEL > default
        self.model = model or os.getenv("GRAPH_EXTRACTION_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Extraction mode: "batch" for Batch API, "realtime" for direct API calls
        self.extraction_mode = os.getenv("GRAPH_EXTRACTION_MODE", "realtime")
        self.rate_limit_delay = 0.05  # seconds between API calls (was 1.0 - very conservative)

    def extract_entities(
        self,
        text: str,
        episode_number: int,
        chunk_id: str,
        retry_attempts: int = 3
    ) -> EntityExtractionResult:
        """Extract entities and relationships from a text chunk

        Args:
            text: The text to extract entities from
            episode_number: Episode number for tracking
            chunk_id: Unique identifier for this chunk
            retry_attempts: Number of times to retry on failure

        Returns:
            EntityExtractionResult containing extracted entities and relationships
        """
        prompt = self.EXTRACTION_PROMPT.format(text=text)

        for attempt in range(retry_attempts):
            try:
                # Use structured outputs with Pydantic schema
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at extracting structured information from text. Extract entities using SPECIFIC types (FORMAL_ORGANIZATION, not just ORGANIZATION). Also extract relationships between entities."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format=ExtractionResponse,
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
                        return EntityExtractionResult(entities=[], relationships=[])

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

                # Extract relationships
                relationships = parsed.relationships if parsed.relationships else []

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                return EntityExtractionResult(
                    entities=entities,
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
        return EntityExtractionResult(
            entities=[],
            relationships=[],
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

    # ==========================================================================
    # Batch Mode Methods
    # ==========================================================================

    def extract_entities_batch(
        self,
        parent_chunks: List,  # List[ParentChunk] from chunking module
        collector,  # BatchCollector
        content_profile: Optional[Any] = None  # ContentProfile from extract_content_batch.py
    ) -> None:
        """
        Add parent chunks to batch collector for later submission.

        This method queues extraction requests for the Batch API instead of
        making real-time API calls. Use this for large-scale extraction jobs.

        Args:
            parent_chunks: List of ParentChunk objects to extract from
            collector: BatchCollector instance to add requests to
            content_profile: Optional ContentProfile for customizing extraction prompts
                             Has attributes: content_type, reality_tag, system_prompt_focus
        """
        # Base system prompt
        base_system_prompt = (
            "You are an expert at extracting structured information from text. "
            "Extract entities using SPECIFIC types (FORMAL_ORGANIZATION, not just ORGANIZATION). "
            "Also extract relationships between entities."
        )

        # Customize prompt based on content profile if provided
        if content_profile is not None:
            profile_additions = f"""

=== CONTENT PROFILE: {content_profile.content_type.upper()} ===
Reality Tag: {content_profile.reality_tag}

{content_profile.system_prompt_focus}

IMPORTANT: Tag all extracted entities with:
- content_type: "{content_profile.content_type}"
- reality_tag: "{content_profile.reality_tag}"
"""
            system_prompt = base_system_prompt + profile_additions
        else:
            system_prompt = base_system_prompt

        for chunk in parent_chunks:
            collector.add_extraction_request(
                parent_chunk=chunk,
                system_prompt=system_prompt,
                user_prompt_template=self.EXTRACTION_PROMPT
            )

    def process_batch_results(
        self,
        results: List[Dict]
    ) -> Dict[str, EntityExtractionResult]:
        """
        Process downloaded batch results, keyed by parent_chunk_id.

        This method parses the raw batch API results and converts them into
        EntityExtractionResult objects that can be used with the post-processing
        pipeline (quality filters, entity resolver, etc.).

        Args:
            results: List of batch result objects from BatchCollector.download_results()
                    Each result has 'custom_id' (parent chunk ID) and 'response' (API response)

        Returns:
            Dict mapping parent_chunk_id to EntityExtractionResult
        """
        processed = {}

        for result in results:
            custom_id = result.get("custom_id")
            if not custom_id:
                continue

            response = result.get("response", {})
            body = response.get("body", {})
            choices = body.get("choices", [])

            if not choices:
                # No response - add empty result
                processed[custom_id] = EntityExtractionResult(
                    entities=[],
                    relationships=[],
                    chunk_id=custom_id,
                    episode_number=0
                )
                continue

            message = choices[0].get("message", {})
            content = message.get("content", "{}")

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON for chunk {custom_id}")
                processed[custom_id] = EntityExtractionResult(
                    entities=[],
                    relationships=[],
                    chunk_id=custom_id,
                    episode_number=0
                )
                continue

            # Convert to Entity objects
            entities = []
            for entity_data in parsed.get("entities", []):
                entity = Entity(
                    name=entity_data.get("name", ""),
                    type=entity_data.get("type", "CONCEPT"),
                    description=entity_data.get("description", ""),
                    aliases=entity_data.get("aliases", []),
                    metadata={"parent_chunk_id": custom_id}
                )
                entities.append(entity)

            # Convert relationships
            relationships = []
            for rel_data in parsed.get("relationships", []):
                relationships.append(RelationshipTriple(
                    source=rel_data.get("source", ""),
                    predicate=rel_data.get("predicate", "RELATES_TO"),
                    target=rel_data.get("target", "")
                ))

            # Extract episode number from chunk ID if present
            # Format: "episode_120_parent_0" or "book_viriditas_parent_0"
            episode_number = 0
            if custom_id.startswith("episode_"):
                try:
                    parts = custom_id.split("_")
                    if len(parts) >= 2:
                        episode_number = int(parts[1])
                except (ValueError, IndexError):
                    pass

            processed[custom_id] = EntityExtractionResult(
                entities=entities,
                relationships=relationships,
                chunk_id=custom_id,
                episode_number=episode_number
            )

        return processed

    def is_batch_mode(self) -> bool:
        """Check if extractor is configured for batch mode"""
        return self.extraction_mode.lower() == "batch"
