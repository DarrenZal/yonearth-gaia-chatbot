# YonEarth Podcast Knowledge Graph Implementation Plan

## Executive Summary

This plan outlines the implementation of a comprehensive knowledge graph system for the YonEarth podcast corpus, integrating:
- **Graphiti** graph database with Neo4j backend for temporal, episodic knowledge
- **GPT-4o-mini** for efficient entity and relationship extraction
- **Obsidian-compatible Markdown wiki** with bidirectional links
- **Interactive graph visualization** using D3.js force-directed layout
- **Parallel subagent processing** for 8x faster implementation

## Problem Statement

### Current System Limitations
Based on analysis of the existing t-SNE/K-Means visualization:

1. **Hard Clustering**: Forces each chunk into exactly ONE category
2. **Lost Semantic Relationships**: Only 2.51% of chunks mention both economic AND ecological terms, but all get assigned to single categories
3. **Artificial Separation**: Economy & Ecology are 97.51 units apart (3rd furthest pair), despite natural overlap in "ecological economics"
4. **Dimensionality Loss**: t-SNE compresses 1536D → 2D, losing nuanced relationships
5. **No Multi-Domain Concepts**: Cannot represent concepts that bridge multiple domains (Economy + Ecology + Community)

### Etymology Insight
As noted: "Economy" (οἰκονομία) and "Ecology" (οἰκολογία) both derive from *oikos* (household):
- **Economy**: Management of the household
- **Ecology**: Study of the household

These concepts are etymologically and semantically intertwined, yet current visualization separates them.

## Proposed Solution: Dual System Architecture

### System 1: Keep Existing t-SNE Map (Already Working)
- **Purpose**: Quick visual overview, chunk-level semantic clustering
- **Technology**: OpenAI embeddings → t-SNE → K-Means → Voronoi visualization
- **Use Case**: Bird's eye view of content distribution

### System 2: NEW Knowledge Graph + Wiki (This Implementation)
- **Purpose**: Precise conceptual relationships, multi-domain concepts, entity tracking
- **Technology**: GPT-4o-mini → Graphiti → Neo4j + Markdown Wiki + Graph Visualization
- **Use Case**: Deep semantic exploration, research, entity connections

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Source Layer                             │
│  172 Episode Transcripts (JSON) + 3 Books (PDF)                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Parallel Extraction Layer (8 Agents)                │
│  GPT-4o-mini Entity Extraction → Structured JSON                │
│  Entity Types: People, Orgs, Concepts, Practices, Tech, Locations│
│  Relationships: works_at, discusses, relates_to, enables, etc.   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Knowledge Graph Layer                          │
│  Graphiti (Temporal Graph DB) + Neo4j Backend                   │
│  - Episodic memory (episode-level context)                      │
│  - Entity deduplication & linking                               │
│  - Multi-label categorization (Economy+Ecology+Community)       │
└────────┬───────────────────────────────────────┬────────────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────────────┐      ┌──────────────────────────────┐
│   Markdown Wiki Layer    │      │   Visualization Layer        │
│  Obsidian-Compatible     │      │  Interactive D3.js Graph     │
│  - Entity pages          │      │  - Force-directed layout     │
│  - Episode summaries     │      │  - Multi-color nodes         │
│  - Concept networks      │      │  - Episode playback          │
│  - Bidirectional links   │      │  - Community detection       │
└─────────────────────────┘      └──────────────────────────────┘
```

## Ontology Design

### Entity Types

#### 1. People
- **Attributes**: name, role, bio, affiliations
- **Examples**: Rowdy Yeatts, Aaron William Perry, Dr. Elaine Ingham
- **Relationships**: `interviewed_by`, `works_at`, `founded`, `collaborates_with`

#### 2. Organizations
- **Attributes**: name, type, location, mission
- **Examples**: High Plains Biochar, Kiss the Ground, Y on Earth Community
- **Relationships**: `founded_by`, `partners_with`, `funded_by`, `located_in`

#### 3. Concepts
- **Attributes**: name, definition, domains (multi-label), importance_score
- **Examples**: Biochar, Regenerative Agriculture, Ecological Economics, Carbon Sequestration
- **Relationships**: `relates_to`, `enables`, `requires`, `contradicts`, `combines_with`

#### 4. Practices
- **Attributes**: name, description, difficulty, resources_needed
- **Examples**: Composting, Cover Cropping, Biochar Production, Permaculture Design
- **Relationships**: `performed_by`, `requires_tool`, `improves`, `part_of`

#### 5. Technologies
- **Attributes**: name, type, manufacturer, purpose
- **Examples**: RocketChar 301, Pyrolysis Reactor, Soil Testing Kit
- **Relationships**: `manufactured_by`, `used_for`, `enables_practice`, `improves_on`

#### 6. Locations
- **Attributes**: name, type (city/region/country), coordinates
- **Examples**: Laramie Wyoming, Amazon Basin, Colorado State University
- **Relationships**: `located_in`, `home_to`, `researched_at`

### Domain Categories (Multi-Label)

Primary domains from 5 Pillars framework:
- **Economy**: Markets, business models, carbon credits, funding
- **Ecology**: Ecosystems, biodiversity, soil health, carbon sequestration
- **Community**: Cooperation, social enterprises, networks, education
- **Culture**: Values, traditions, storytelling, worldviews
- **Health**: Human health, animal health, planetary health

Extended domains:
- **Technology**: Tools, innovations, engineering
- **Policy**: Regulations, government programs, incentives
- **Science**: Research, data, methodology
- **Spirituality**: Connection, meaning, sacred practices
- **Food Systems**: Agriculture, nutrition, supply chains

### Relationship Types

#### Organizational
- `works_at(Person, Organization)`
- `founded(Person, Organization)`
- `leads(Person, Organization)`
- `partners_with(Organization, Organization)`
- `funded_by(Organization, Organization)`

#### Knowledge
- `discusses(Person/Episode, Concept)`
- `explains(Person, Concept)`
- `advocates_for(Person, Practice)`
- `researches(Person/Organization, Concept)`

#### Conceptual
- `relates_to(Concept, Concept, strength: float)`
- `enables(Practice, Outcome)`
- `requires(Practice, Resource)`
- `improves(Practice, Metric)`
- `combines_with(Concept, Concept)`
- `part_of(Concept, Concept)`

#### Spatial/Temporal
- `located_in(Entity, Location)`
- `occurs_at(Episode, Timestamp)`
- `mentioned_in(Entity, Episode, timestamp: float)`

## Implementation Roadmap

### Phase 1: Environment Setup (Days 1-2)

#### 1.1 Install Dependencies
```bash
# Graphiti and Neo4j
pip install graphiti-core neo4j python-dotenv

# Additional tools
pip install openai anthropic tiktoken markdown obsidiantools

# Graph visualization
pip install networkx plotly pyvis
```

#### 1.2 Neo4j Setup
```bash
# Docker-based Neo4j
docker run \
    --name neo4j-yonearth \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/yonearth2024 \
    -v neo4j-data:/data \
    -d neo4j:latest
```

#### 1.3 Environment Configuration
Add to `.env`:
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yonearth2024

# Knowledge Graph Settings
KG_EXTRACTION_MODEL=gpt-4o-mini
KG_EXTRACTION_TEMPERATURE=0.1
KG_BATCH_SIZE=8
KG_MAX_ENTITIES_PER_CHUNK=20
```

#### 1.4 Directory Structure
```
yonearth-gaia-chatbot/
├── src/
│   └── knowledge_graph/
│       ├── __init__.py
│       ├── extractors/
│       │   ├── __init__.py
│       │   ├── entity_extractor.py      # GPT-4o-mini extraction
│       │   ├── relationship_extractor.py
│       │   └── ontology.py               # Schema definitions
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── graphiti_client.py        # Graphiti integration
│       │   ├── neo4j_client.py           # Direct Neo4j access
│       │   └── graph_builder.py          # Graph construction
│       ├── wiki/
│       │   ├── __init__.py
│       │   ├── markdown_generator.py     # Obsidian page generation
│       │   └── wiki_builder.py           # Wiki structure
│       └── visualization/
│           ├── __init__.py
│           ├── graph_layout.py           # Force-directed layout
│           └── export_visualization.py   # D3.js data export
├── data/
│   └── knowledge_graph/
│       ├── entities/                     # Extracted entities (JSON)
│       ├── relationships/                # Extracted relationships (JSON)
│       └── wiki/                         # Generated Markdown files
│           ├── people/
│           ├── organizations/
│           ├── concepts/
│           ├── practices/
│           ├── technologies/
│           ├── locations/
│           └── episodes/
├── scripts/
│   ├── build_knowledge_graph.py          # Main orchestration script
│   ├── extract_entities_parallel.py      # Parallel extraction
│   └── generate_wiki.py                  # Wiki generation
└── web/
    └── knowledge_graph.html              # Interactive visualization
```

### Phase 2: Ontology Implementation (Days 3-5)

#### 2.1 Define Entity Schema (`src/knowledge_graph/extractors/ontology.py`)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    PRACTICE = "practice"
    TECHNOLOGY = "technology"
    LOCATION = "location"

class Domain(Enum):
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
    id: str
    name: str
    type: EntityType
    domains: Set[Domain] = field(default_factory=set)
    attributes: Dict[str, str] = field(default_factory=dict)
    mentions: List[Dict] = field(default_factory=list)  # episode_id, timestamp, context

@dataclass
class Relationship:
    source_id: str
    target_id: str
    type: str
    strength: float = 1.0
    episode_id: Optional[str] = None
    timestamp: Optional[float] = None
    context: Optional[str] = None
```

#### 2.2 GPT-4o-mini Extraction Prompts

**System Prompt**:
```
You are an expert knowledge graph builder analyzing podcast transcripts about sustainability, regenerative practices, and ecological systems.

Your task is to extract structured entities and relationships from transcript chunks.

Entity Types:
- PERSON: Individuals (guests, hosts, mentioned experts)
- ORGANIZATION: Companies, nonprofits, institutions
- CONCEPT: Abstract ideas (biochar, regenerative agriculture, carbon sequestration)
- PRACTICE: Actions/techniques (composting, cover cropping)
- TECHNOLOGY: Tools, machines, innovations (RocketChar 301)
- LOCATION: Geographic places

Relationship Types:
- works_at, founded, leads (organizational)
- discusses, explains, advocates_for (knowledge)
- relates_to, enables, requires, improves (conceptual)
- located_in, mentioned_in (spatial/temporal)

Domain Categories (multi-label):
Economy, Ecology, Community, Culture, Health, Technology, Policy, Science, Spirituality, Food_Systems

Output Format:
{
  "entities": [
    {
      "name": "Rowdy Yeatts",
      "type": "PERSON",
      "domains": ["ECONOMY", "TECHNOLOGY", "ECOLOGY"],
      "attributes": {"role": "Founder & CEO", "organization": "High Plains Biochar"}
    }
  ],
  "relationships": [
    {
      "source": "Rowdy Yeatts",
      "target": "High Plains Biochar",
      "type": "founded",
      "strength": 1.0
    }
  ]
}
```

### Phase 3: Parallel Extraction Pipeline (Week 2)

#### 3.1 Episode Batch Distribution

**8 Parallel Agents** processing 172 episodes:
- Agent 1: Episodes 0-21
- Agent 2: Episodes 22-43
- Agent 3: Episodes 44-65
- Agent 4: Episodes 66-87
- Agent 5: Episodes 88-109
- Agent 6: Episodes 110-131
- Agent 7: Episodes 132-153
- Agent 8: Episodes 154-172

#### 3.2 Extraction Script (`scripts/extract_entities_parallel.py`)

```python
import asyncio
import json
from pathlib import Path
from openai import AsyncOpenAI
from typing import List, Dict
import os

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_from_chunk(chunk_text: str, episode_id: str, chunk_index: int) -> Dict:
    """Extract entities and relationships from a single chunk using GPT-4o-mini"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Episode {episode_id}, Chunk {chunk_index}:\n\n{chunk_text}"}
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

async def process_episode(episode_path: Path, output_dir: Path):
    """Process a single episode with chunking"""
    with open(episode_path) as f:
        episode_data = json.load(f)

    episode_id = episode_data['episode_number']
    transcript = episode_data['full_transcript']

    # Chunk transcript (500 tokens)
    chunks = chunk_transcript(transcript, chunk_size=500, overlap=50)

    # Extract from all chunks
    tasks = [
        extract_from_chunk(chunk, episode_id, i)
        for i, chunk in enumerate(chunks)
    ]

    results = await asyncio.gather(*tasks)

    # Save results
    output_file = output_dir / f"episode_{episode_id}_entities.json"
    with open(output_file, 'w') as f:
        json.dump({
            'episode_id': episode_id,
            'title': episode_data['title'],
            'extraction_results': results
        }, f, indent=2)

async def process_batch(episode_paths: List[Path], output_dir: Path):
    """Process a batch of episodes"""
    tasks = [process_episode(path, output_dir) for path in episode_paths]
    await asyncio.gather(*tasks)

# Main execution with 8 parallel workers
if __name__ == "__main__":
    # Divide episodes into 8 batches
    # Launch 8 async workers
    # Each worker processes ~21 episodes
```

#### 3.3 Entity Deduplication & Linking

After parallel extraction, merge results:
```python
def deduplicate_entities(all_entities: List[Entity]) -> List[Entity]:
    """Use fuzzy matching to deduplicate entities across episodes"""
    # Group by type
    # Use sentence embeddings for similarity
    # Merge entities with >85% similarity
    # Aggregate mentions, attributes, domains
```

### Phase 4: Knowledge Graph Construction (Week 3)

#### 4.1 Graphiti Integration (`src/knowledge_graph/graph/graphiti_client.py`)

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeNode, EntityNode
from graphiti_core.edges import Relationship

class YonEarthKnowledgeGraph:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    async def add_episode(self, episode_id: str, entities: List[Entity],
                          relationships: List[Relationship]):
        """Add an episode with its entities and relationships as episodic memory"""

        # Create episode node
        episode_node = EpisodeNode(
            episode_id=episode_id,
            timestamp=episode_data['publish_date'],
            content=episode_data['description']
        )

        await self.graphiti.add_episode(episode_node, entities, relationships)

    async def query_related_concepts(self, concept_name: str, domains: List[Domain]) -> List[Entity]:
        """Query concepts related to a given concept, filtered by domains"""
        # Use Graphiti's temporal search
        # Return entities within domain filters
```

#### 4.2 Graph Metrics & Analysis

```python
def calculate_graph_metrics(graph):
    """Calculate key graph metrics"""
    return {
        'total_entities': len(graph.nodes),
        'total_relationships': len(graph.edges),
        'entity_breakdown': count_by_type(graph.nodes),
        'domain_breakdown': count_by_domain(graph.nodes),
        'most_connected_concepts': top_nodes_by_degree(graph, k=20),
        'bridging_concepts': find_bridging_nodes(graph),  # High betweenness centrality
        'community_structure': detect_communities(graph)
    }
```

### Phase 5: Markdown Wiki Generation (Week 4)

#### 5.1 Obsidian Page Templates

**Concept Page Template**:
```markdown
---
type: concept
domains: [Economy, Ecology]
mentions: 15
importance: high
---

# {{concept_name}}

## Definition
{{definition}}

## Domains
- Economy
- Ecology

## Related Concepts
- [[Carbon Sequestration]]
- [[Regenerative Agriculture]]
- [[Soil Health]]

## Practices Using This Concept
- [[Biochar Production]]
- [[Cover Cropping]]

## Episodes Discussing This
- [[Episode 120 - Rowdy Yeatts, High Plains Biochar]]
- [[Episode 122 - Dr. David Laird, Biochar Science]]

## Key People
- [[Rowdy Yeatts]]
- [[Dr. David Laird]]

## Organizations
- [[High Plains Biochar]]

## Graph View
![[concept_graph_biochar.png]]
```

**Episode Page Template**:
```markdown
---
type: episode
episode_number: 120
publish_date: 2022-08-15
guest: Rowdy Yeatts
domains: [Economy, Ecology, Technology]
---

# Episode 120: Rowdy Yeatts, Founder & CEO, High Plains Biochar

## Summary
{{ai_generated_summary}}

## Key Concepts Discussed
- [[Biochar]]
- [[Carbon Sequestration]]
- [[Pyrolysis]]
- [[Carbon Credits]]
- [[Regenerative Agriculture]]

## People Mentioned
- [[Rowdy Yeatts]] (Guest)
- [[Aaron William Perry]] (Host)
- [[Kevin Johnson]] (Executive at High Plains Biochar)

## Organizations
- [[High Plains Biochar]]
- [[Colorado State University]]
- [[Rocky Mountain Farmers Union]]
- [[Microsoft]] (gBeta accelerator)

## Technologies
- [[RocketChar 301]]
- [[Pyrolysis Reactor]]

## Locations
- [[Laramie, Wyoming]]
- [[Fort Collins, Colorado]]

## Timeline
- 00:00 - Introduction
- 05:30 - What is biochar?
- 15:20 - Carbon credit markets
- 28:45 - RocketChar 301 technology

## Audio
[Listen on YonEarth.org]({{audio_url}})

## Transcript
[Full transcript]({{transcript_url}})
```

#### 5.2 Wiki Generation Script (`scripts/generate_wiki.py`)

```python
from jinja2 import Template
from pathlib import Path

def generate_wiki(knowledge_graph, output_dir: Path):
    """Generate complete Obsidian vault from knowledge graph"""

    # Create directory structure
    (output_dir / "people").mkdir(parents=True, exist_ok=True)
    (output_dir / "organizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (output_dir / "practices").mkdir(parents=True, exist_ok=True)
    (output_dir / "technologies").mkdir(parents=True, exist_ok=True)
    (output_dir / "locations").mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)

    # Generate pages for each entity
    for entity in knowledge_graph.entities:
        template = load_template(entity.type)
        content = template.render(entity=entity, graph=knowledge_graph)

        filepath = output_dir / entity.type.value / f"{sanitize_filename(entity.name)}.md"
        filepath.write_text(content)

    # Generate index pages
    generate_index_page(output_dir, knowledge_graph)
    generate_domain_pages(output_dir, knowledge_graph)
```

### Phase 6: Interactive Graph Visualization (Weeks 5-6)

#### 6.1 Export Graph Data for D3.js

```python
def export_for_visualization(knowledge_graph) -> Dict:
    """Export graph in format suitable for D3.js force-directed graph"""

    nodes = []
    for entity in knowledge_graph.entities:
        nodes.append({
            'id': entity.id,
            'name': entity.name,
            'type': entity.type.value,
            'domains': [d.value for d in entity.domains],
            'importance': calculate_importance(entity),
            'mentions': len(entity.mentions)
        })

    links = []
    for rel in knowledge_graph.relationships:
        links.append({
            'source': rel.source_id,
            'target': rel.target_id,
            'type': rel.type,
            'strength': rel.strength
        })

    return {
        'nodes': nodes,
        'links': links,
        'communities': detect_communities(knowledge_graph),
        'metadata': calculate_graph_metrics(knowledge_graph)
    }
```

#### 6.2 D3.js Visualization (`web/KnowledgeGraph.js`)

```javascript
class KnowledgeGraphVisualization {
    constructor(data) {
        this.data = data;
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        // Multi-color scale for multi-domain nodes
        this.colorScale = d3.scaleOrdinal()
            .domain(['economy', 'ecology', 'community', 'culture', 'health'])
            .range(['#FF9800', '#2196F3', '#4CAF50', '#9C27B0', '#F44336']);
    }

    renderNode(node) {
        // Multi-domain nodes get gradient fill
        if (node.domains.length > 1) {
            return this.createMultiColorNode(node);
        } else {
            return this.createSingleColorNode(node);
        }
    }

    createMultiColorNode(node) {
        // Create pie-chart style node showing multiple domains
        const colors = node.domains.map(d => this.colorScale(d));
        // SVG gradient or pie segments
    }
}
```

#### 6.3 Interactive Features

- **Hover**: Show entity details, episode mentions
- **Click**: Navigate to Obsidian wiki page
- **Filter**: By domain, entity type, importance
- **Search**: Find entities by name
- **Community highlighting**: Show detected communities
- **Zoom**: Focus on sub-graphs
- **Timeline**: Animate knowledge evolution across episodes

### Phase 7: Integration & Testing (Week 7)

#### 7.1 Web Interface Integration

Add to existing podcast map interface:
```html
<!-- New tab in header -->
<div class="map-header">
    <div class="view-selector">
        <button class="view-btn active" data-view="tsne">t-SNE Map</button>
        <button class="view-btn" data-view="knowledge-graph">Knowledge Graph</button>
        <button class="view-btn" data-view="wiki">Wiki</button>
    </div>
</div>
```

#### 7.2 API Endpoints

Add to `src/api/main.py`:
```python
@app.get("/api/knowledge-graph/entities/{entity_id}")
async def get_entity(entity_id: str):
    """Get entity details with related entities"""

@app.get("/api/knowledge-graph/search")
async def search_entities(query: str, entity_type: Optional[str] = None,
                          domains: Optional[List[str]] = None):
    """Search entities by name, type, domain"""

@app.get("/api/knowledge-graph/explore/{concept_name}")
async def explore_concept(concept_name: str, depth: int = 2):
    """Get concept neighborhood (entities within N hops)"""
```

#### 7.3 Testing Strategy

**Unit Tests**:
- Entity extraction accuracy (manual validation on 10 episodes)
- Relationship extraction precision/recall
- Entity deduplication correctness

**Integration Tests**:
- End-to-end pipeline (transcript → entities → graph → wiki → visualization)
- Graph database queries
- API endpoint responses

**User Acceptance Tests**:
- Can users find related concepts easily?
- Does multi-domain tagging make sense?
- Is wiki navigation intuitive?
- Does visualization load quickly with 10,000+ nodes?

## Parallel Execution Strategy

### Agent Task Distribution

**Agent 1: Environment Setup & Schema**
- Install Graphiti, Neo4j, dependencies
- Configure environment variables
- Create directory structure
- Define ontology schema (`ontology.py`)
- **Deliverable**: Working environment + schema definitions

**Agent 2: Entity Extractor Implementation**
- Implement `entity_extractor.py` with GPT-4o-mini
- Create extraction prompts
- Build chunking logic
- **Deliverable**: Working extraction module

**Agent 3: Relationship Extractor Implementation**
- Implement `relationship_extractor.py`
- Define relationship types
- Build relationship inference logic
- **Deliverable**: Working relationship extraction

**Agent 4: Episodes 0-43 Extraction**
- Run parallel extraction on episodes 0-43
- Save entity/relationship JSONs
- **Deliverable**: 44 episode extraction files

**Agent 5: Episodes 44-87 Extraction**
- Run parallel extraction on episodes 44-87
- Save entity/relationship JSONs
- **Deliverable**: 44 episode extraction files

**Agent 6: Episodes 88-131 Extraction**
- Run parallel extraction on episodes 88-131
- Save entity/relationship JSONs
- **Deliverable**: 44 episode extraction files

**Agent 7: Episodes 132-172 Extraction**
- Run parallel extraction on episodes 132-172
- Save entity/relationship JSONs
- **Deliverable**: 41 episode extraction files

**Agent 8: Graph Building & Deduplication**
- Implement Graphiti client
- Build entity deduplication logic
- Merge all extraction results
- Populate Neo4j database
- **Deliverable**: Populated knowledge graph

**Agent 9: Wiki Generation**
- Implement markdown generators
- Create Obsidian templates
- Generate all entity pages
- Generate episode pages
- **Deliverable**: Complete Obsidian vault

**Agent 10: Visualization Implementation**
- Export graph data for D3.js
- Implement force-directed layout
- Create interactive controls
- Integrate with web interface
- **Deliverable**: Working interactive visualization

### Timeline

**Hours 0-2**: Agents 1, 2, 3 (Setup, Entity Extractor, Relationship Extractor)
**Hours 2-4**: Agents 4, 5, 6, 7 (Parallel episode extraction)
**Hours 4-5**: Agent 8 (Graph building & deduplication)
**Hours 5-6**: Agent 9 (Wiki generation)
**Hours 6-7**: Agent 10 (Visualization)
**Hour 7-8**: Integration, testing, documentation

**Expected Speedup**: 8-10x faster than sequential execution

## Success Metrics

### Quantitative
- **Coverage**: >90% of named entities extracted
- **Accuracy**: >85% entity extraction accuracy (manual validation)
- **Completeness**: >80% of significant relationships captured
- **Performance**: Graph queries <100ms
- **Deduplication**: <5% duplicate entities in final graph

### Qualitative
- Multi-domain concepts properly tagged (e.g., "ecological economics" has both Economy + Ecology)
- Bridging concepts identified (high betweenness centrality)
- Wiki navigation feels intuitive (user testing)
- Visualization loads smoothly with 10,000+ nodes
- Episode context preserved in episodic memory

## Future Enhancements

### Phase 8: Advanced Features (Post-Launch)
1. **Temporal Analysis**: Track concept evolution across episodes
2. **Question Answering**: Graph-based Q&A using episodic retrieval
3. **Recommendation Engine**: "If you liked Episode X, try Episode Y" based on concept overlap
4. **Auto-Tagging**: New episodes automatically tagged with relevant concepts
5. **Knowledge Graph Embeddings**: Use graph structure + text for hybrid RAG
6. **Community Detection**: Identify topic communities (e.g., "Biochar Community", "Regenerative Ag Community")
7. **Export Formats**: RDF, GraphML, Gephi-compatible formats

## Conclusion

This implementation creates a sophisticated knowledge management system that:
- Preserves multi-domain nature of concepts (Economy + Ecology)
- Tracks entities and relationships across 172 episodes
- Generates searchable, browsable Obsidian wiki
- Provides interactive graph visualization
- Enables episodic memory and temporal queries
- Complements existing t-SNE visualization

**Estimated Total Time**: 7-8 weeks (or 7-8 hours with 10 parallel agents!)

**Key Innovation**: Multi-label domain tagging solves the "ecological economics" problem - concepts can span multiple domains simultaneously, reflecting their true semantic nature.
