"""
Wiki builder for generating complete Obsidian vault from knowledge graph.

Orchestrates the creation of all wiki pages, directories, and indexes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from .markdown_generator import MarkdownGenerator

logger = logging.getLogger(__name__)


class WikiBuilder:
    """Builds complete Obsidian wiki from knowledge graph data."""

    def __init__(self, output_dir: Path):
        """
        Initialize wiki builder.

        Args:
            output_dir: Root directory for wiki output
        """
        self.output_dir = Path(output_dir)
        self.generator = MarkdownGenerator()

        # Statistics
        self.stats = {
            'total_entities': 0,
            'total_episodes': 0,
            'total_relationships': 0,
            'people_count': 0,
            'organizations_count': 0,
            'concepts_count': 0,
            'practices_count': 0,
            'technologies_count': 0,
            'locations_count': 0,
            'episodes_count': 0,
            'entity_types_count': 0,
            'domains': set(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Entity storage
        self.entities_by_type = defaultdict(list)
        self.entities_by_name = {}
        self.episodes = {}
        self.domains = defaultdict(list)

    def create_directory_structure(self):
        """Create the wiki directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / 'people',
            self.output_dir / 'organizations',
            self.output_dir / 'concepts',
            self.output_dir / 'practices',
            self.output_dir / 'technologies',
            self.output_dir / 'locations',
            self.output_dir / 'episodes',
            self.output_dir / 'domains',
            self.output_dir / '_indexes',
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def load_extraction_files(self, extraction_dir: Path) -> List[Dict]:
        """
        Load all entity extraction JSON files.

        Args:
            extraction_dir: Directory containing extraction JSON files

        Returns:
            List of extraction data dictionaries
        """
        extraction_files = sorted(extraction_dir.glob('episode_*_extraction.json'))
        extractions = []

        logger.info(f"Loading {len(extraction_files)} extraction files...")

        for file_path in extraction_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    extractions.append(data)
                    logger.debug(f"Loaded {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return extractions

    def process_extractions(self, extractions: List[Dict]):
        """
        Process extraction data to build entity and episode collections.

        Args:
            extractions: List of extraction dictionaries
        """
        logger.info("Processing extractions...")

        for extraction in extractions:
            episode_number = extraction.get('episode_number')

            # Use merged metadata fields (from UnifiedBuilder) if available
            episode_title = extraction.get('episode_title') or extraction.get('title', f'Episode {episode_number}')
            guest_name = extraction.get('guest_name', '')

            # Process episode
            episode_entities = extraction.get('entities', [])
            episode_data = {
                'number': episode_number,
                'title': episode_title,
                'guest': guest_name,
                'date': extraction.get('date'),
                'publish_date': extraction.get('publish_date'),
                'host': extraction.get('host'),
                'url': extraction.get('url', f'https://yonearth.org/episode-{episode_number}'),
                'audio_url': extraction.get('audio_url', ''),
                'subtitle': extraction.get('subtitle', ''),
                'description': extraction.get('description', ''),
                'summary': extraction.get('summary', ''),
                'sponsors': extraction.get('sponsors', ''),
                'about_sections': extraction.get('about_sections', {}),
                'related_episodes': extraction.get('related_episodes', []),
                'transcript_length': extraction.get('transcript_length'),
                'key_concepts': [],
                'people': [],
                'organizations': [],
                'technologies': [],
                'places': [],
                'practices': [],
                'domains': set(),
                'entity_count': len(episode_entities)
            }

            # Process entities
            for entity_data in episode_entities:
                entity_type = entity_data.get('type', 'CONCEPT')
                entity_name = entity_data.get('name', '')
                entity_desc = entity_data.get('description', '')
                entity_aliases = entity_data.get('aliases', [])

                if not entity_name:
                    continue

                # Normalize entity name
                entity_key = entity_name.lower()

                # Get or create entity
                if entity_key not in self.entities_by_name:
                    entity = {
                        'name': entity_name,
                        'type': entity_type,
                        'description': entity_desc,
                        'aliases': entity_aliases,
                        'episodes': [],
                        'episode_count': 0,
                        'mention_count': 0,
                        'importance_score': 0,
                        'related_concepts': set(),
                        'related_practices': set(),
                        'related_technologies': set(),
                        'organizations': set(),
                        'people': set(),
                        'locations': set(),
                        'affiliations': set(),
                        'domains': set()
                    }
                    self.entities_by_name[entity_key] = entity
                    self.entities_by_type[entity_type].append(entity)
                else:
                    entity = self.entities_by_name[entity_key]

                # Update entity
                entity['episodes'].append({
                    'number': episode_number,
                    'title': episode_title
                })
                entity['episode_count'] += 1
                entity['mention_count'] += len(entity_data.get('metadata', {}).get('chunks', []))

                # Add to episode collections
                if entity_type == 'CONCEPT':
                    episode_data['key_concepts'].append(entity_name)
                elif entity_type == 'PERSON':
                    episode_data['people'].append(entity_name)
                elif entity_type == 'ORGANIZATION':
                    episode_data['organizations'].append(entity_name)
                elif entity_type == 'TECHNOLOGY':
                    episode_data['technologies'].append(entity_name)
                elif entity_type == 'PLACE':
                    episode_data['places'].append(entity_name)
                elif entity_type == 'PRACTICE':
                    episode_data['practices'].append(entity_name)

            # Store episode
            self.episodes[episode_number] = episode_data
            self.stats['episodes_count'] += 1

        # Calculate importance scores
        for entity in self.entities_by_name.values():
            entity['importance_score'] = (
                entity['episode_count'] * 10 +
                entity['mention_count']
            )

        # Update statistics
        self.stats['total_entities'] = len(self.entities_by_name)
        self.stats['total_episodes'] = len(self.episodes)
        self.stats['people_count'] = len(self.entities_by_type.get('PERSON', []))
        self.stats['organizations_count'] = len(self.entities_by_type.get('ORGANIZATION', []))
        self.stats['concepts_count'] = len(self.entities_by_type.get('CONCEPT', []))
        self.stats['practices_count'] = len(self.entities_by_type.get('PRACTICE', []))
        self.stats['technologies_count'] = len(self.entities_by_type.get('TECHNOLOGY', []))
        self.stats['locations_count'] = len(self.entities_by_type.get('PLACE', []))
        self.stats['entity_types_count'] = len(self.entities_by_type)

        logger.info(f"Processed {self.stats['total_entities']} entities from {self.stats['total_episodes']} episodes")

    def build_relationships(self):
        """Build relationships between entities based on co-occurrence."""
        logger.info("Building entity relationships...")

        # Group entities by episode
        episode_entities = defaultdict(lambda: defaultdict(list))

        for entity in self.entities_by_name.values():
            for ep in entity['episodes']:
                episode_entities[ep['number']][entity['type']].append(entity['name'])

        # Build co-occurrence relationships
        for ep_num, type_entities in episode_entities.items():
            # Connect people to organizations
            for person in type_entities.get('PERSON', []):
                person_entity = self.entities_by_name[person.lower()]
                for org in type_entities.get('ORGANIZATION', []):
                    person_entity['affiliations'].add(org)

            # Connect concepts to practices
            for concept in type_entities.get('CONCEPT', []):
                concept_entity = self.entities_by_name[concept.lower()]
                for practice in type_entities.get('PRACTICE', []):
                    concept_entity['related_practices'].add(practice)

            # Connect concepts to technologies
            for concept in type_entities.get('CONCEPT', []):
                concept_entity = self.entities_by_name[concept.lower()]
                for tech in type_entities.get('TECHNOLOGY', []):
                    concept_entity['related_technologies'].add(tech)

            # Connect concepts to each other (same episode)
            concepts = type_entities.get('CONCEPT', [])
            for i, concept1 in enumerate(concepts):
                concept1_entity = self.entities_by_name[concept1.lower()]
                for concept2 in concepts[i+1:]:
                    concept1_entity['related_concepts'].add(concept2)

        # Convert sets to sorted lists for templates
        for entity in self.entities_by_name.values():
            for key in ['related_concepts', 'related_practices', 'related_technologies',
                       'organizations', 'people', 'locations', 'affiliations', 'domains']:
                if isinstance(entity[key], set):
                    entity[key] = sorted(list(entity[key]))

    def generate_entity_pages(self):
        """Generate Markdown pages for all entities."""
        logger.info("Generating entity pages...")

        type_to_dir = {
            'PERSON': 'people',
            'ORGANIZATION': 'organizations',
            'CONCEPT': 'concepts',
            'PRACTICE': 'practices',
            'TECHNOLOGY': 'technologies',
            'PLACE': 'locations'
        }

        for entity_type, entities in self.entities_by_type.items():
            dir_name = type_to_dir.get(entity_type, 'concepts')  # Default to concepts
            type_dir = self.output_dir / dir_name

            # Ensure directory exists
            type_dir.mkdir(parents=True, exist_ok=True)

            for entity in entities:
                filename = self.generator.sanitize_filename(entity['name']) + '.md'
                filepath = type_dir / filename

                # Generate page content
                content = self.generator.generate_entity_page(entity, entity_type)

                # Write file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                logger.debug(f"Generated {filepath.name}")

            logger.info(f"Generated {len(entities)} {entity_type} pages")

    def generate_episode_pages(self):
        """Generate Markdown pages for all episodes."""
        logger.info("Generating episode pages...")

        episodes_dir = self.output_dir / 'episodes'

        for episode_num, episode_data in self.episodes.items():
            filename = f'Episode_{episode_num:03d}.md'
            filepath = episodes_dir / filename

            # Generate page content
            content = self.generator.generate_episode_page(episode_data)

            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.debug(f"Generated {filename}")

        logger.info(f"Generated {len(self.episodes)} episode pages")

    def generate_index_pages(self):
        """Generate main index and type index pages."""
        logger.info("Generating index pages...")

        # Main index
        index_content = self.generator.generate_index_page(self.stats)
        with open(self.output_dir / 'Index.md', 'w', encoding='utf-8') as f:
            f.write(index_content)
        logger.info("Generated main Index page")

        # Type indexes
        type_to_dir = {
            'PERSON': 'people',
            'ORGANIZATION': 'organizations',
            'CONCEPT': 'concepts',
            'PRACTICE': 'practices',
            'TECHNOLOGY': 'technologies',
            'PLACE': 'locations'
        }

        for entity_type, dir_name in type_to_dir.items():
            entities = self.entities_by_type.get(entity_type, [])
            if entities:
                content = self.generator.generate_type_index(entity_type, entities)
                filepath = self.output_dir / '_indexes' / f'{dir_name.title()}.md'
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Generated {entity_type} index")

        # Episodes index
        episodes_list = sorted(self.episodes.values(), key=lambda x: x['number'])
        self._generate_episodes_index(episodes_list)

    def _generate_episodes_index(self, episodes: List[Dict]):
        """Generate episodes index page."""
        from jinja2 import Environment, BaseLoader

        template = Environment(loader=BaseLoader()).from_string("""---
title: Episodes
type: episode_index
---

# Episodes

Total episodes: {{ episode_count }}

{% for episode in episodes %}
## [[Episode {{ episode.number }}]]: {{ episode.title }}
{% if episode.guest %}**Guest**: [[{{ episode.guest }}]]{% endif %}

*Entities: {{ episode.entity_count }}*

---
{% endfor %}

[[Index|Back to Index]]
""")

        content = template.render(
            episode_count=len(episodes),
            episodes=episodes
        )

        filepath = self.output_dir / '_indexes' / 'Episodes.md'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def generate_summary_pages(self):
        """Generate summary and analysis pages."""
        logger.info("Generating summary pages...")

        # Most connected concepts
        self._generate_most_connected_page()

        # Bridging concepts
        self._generate_bridging_concepts_page()

        # Knowledge map
        self._generate_knowledge_map_page()

        # Top people
        self._generate_top_people_page()

    def _generate_most_connected_page(self):
        """Generate page showing most connected concepts."""
        concepts = self.entities_by_type.get('CONCEPT', [])
        sorted_concepts = sorted(
            concepts,
            key=lambda x: len(x.get('related_concepts', [])) + len(x.get('related_practices', [])),
            reverse=True
        )[:50]

        from jinja2 import Environment, BaseLoader
        template = Environment(loader=BaseLoader()).from_string("""---
title: Most Connected Concepts
type: analysis
---

# Most Connected Concepts

These concepts have the most connections to other entities in the knowledge graph.

{% for concept in concepts %}
## {{ loop.index }}. [[{{ concept.name }}]]

**Description**: {{ concept.description[:200] }}...

**Connections**:
- Related Concepts: {{ concept.related_concepts | length }}
- Related Practices: {{ concept.related_practices | length }}
- Episodes: {{ concept.episode_count }}
- Mentions: {{ concept.mention_count }}

**Importance Score**: {{ concept.importance_score }}

---
{% endfor %}

[[Index|Back to Index]]
""")

        content = template.render(concepts=sorted_concepts)
        filepath = self.output_dir / 'Most_Connected_Concepts.md'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def _generate_bridging_concepts_page(self):
        """Generate page showing concepts that bridge multiple domains."""
        # This would require domain information from the extraction
        # For now, create a placeholder
        content = """---
title: Bridging Concepts
type: analysis
---

# Bridging Concepts

Concepts that connect multiple domains (Economy, Ecology, Community, Culture, Health).

*This page requires domain classification data.*

[[Index|Back to Index]]
"""
        filepath = self.output_dir / 'Bridging_Concepts.md'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def _generate_knowledge_map_page(self):
        """Generate high-level knowledge map overview."""
        from jinja2 import Environment, BaseLoader
        template = Environment(loader=BaseLoader()).from_string("""---
title: Knowledge Map
type: overview
---

# Knowledge Map

A high-level overview of the Y on Earth podcast knowledge graph.

## By the Numbers

- **Total Entities**: {{ stats.total_entities }}
- **Total Episodes**: {{ stats.total_episodes }}
- **People**: {{ stats.people_count }}
- **Organizations**: {{ stats.organizations_count }}
- **Concepts**: {{ stats.concepts_count }}
- **Practices**: {{ stats.practices_count }}
- **Technologies**: {{ stats.technologies_count }}
- **Locations**: {{ stats.locations_count }}

## Entity Distribution

### Most Common Entity Types
1. Concepts ({{ stats.concepts_count }})
2. People ({{ stats.people_count }})
3. Organizations ({{ stats.organizations_count }})
4. Practices ({{ stats.practices_count }})
5. Technologies ({{ stats.technologies_count }})
6. Locations ({{ stats.locations_count }})

## Explore

- [[Most Connected Concepts]]
- [[Bridging Concepts]]
- [[Episodes]]
- [[People]]
- [[Organizations]]

[[Index|Back to Index]]
""")

        content = template.render(stats=self.stats)
        filepath = self.output_dir / 'Knowledge_Map.md'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def _generate_top_people_page(self):
        """Generate page showing most frequently mentioned people."""
        people = self.entities_by_type.get('PERSON', [])
        sorted_people = sorted(people, key=lambda x: x['importance_score'], reverse=True)[:30]

        from jinja2 import Environment, BaseLoader
        template = Environment(loader=BaseLoader()).from_string("""---
title: Top People
type: analysis
---

# Top People

Most frequently mentioned people across all episodes.

{% for person in people %}
## {{ loop.index }}. [[{{ person.name }}]]

{{ person.description[:200] }}...

- **Episodes**: {{ person.episode_count }}
- **Mentions**: {{ person.mention_count }}
- **Importance Score**: {{ person.importance_score }}

{% if person.affiliations %}
**Affiliations**: {% for org in person.affiliations %}[[{{ org }}]]{% if not loop.last %}, {% endif %}{% endfor %}
{% endif %}

---
{% endfor %}

[[Index|Back to Index]]
""")

        content = template.render(people=sorted_people)
        filepath = self.output_dir / 'Top_People.md'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def build(self, extraction_dir: Path) -> Dict:
        """
        Build complete wiki from extraction files.

        Args:
            extraction_dir: Directory containing entity extraction JSON files

        Returns:
            Statistics dictionary
        """
        logger.info("Starting wiki build...")

        # Create directory structure
        self.create_directory_structure()

        # Load and process data
        extractions = self.load_extraction_files(extraction_dir)
        self.process_extractions(extractions)
        self.build_relationships()

        # Generate pages
        self.generate_entity_pages()
        self.generate_episode_pages()
        self.generate_index_pages()
        self.generate_summary_pages()

        logger.info("Wiki build complete!")

        return self.stats
