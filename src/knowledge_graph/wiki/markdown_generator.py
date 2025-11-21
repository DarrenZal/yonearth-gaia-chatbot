"""
Markdown page generator for Obsidian wiki.

Generates individual Markdown pages for entities with bidirectional links.
"""

import re
from typing import Dict, List, Set, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, BaseLoader, Template


class MarkdownGenerator:
    """Generates Obsidian-compatible Markdown pages for knowledge graph entities."""

    def __init__(self):
        """Initialize the markdown generator with Jinja2 templates."""
        self.templates = self._create_templates()

    def _create_templates(self) -> Dict[str, Template]:
        """Create Jinja2 templates for different entity types."""
        env = Environment(loader=BaseLoader())

        templates = {}

        # Person page template
        templates['PERSON'] = env.from_string("""---
type: person
name: {{ entity.name }}
aliases: {{ entity.aliases | join(', ') if entity.aliases else 'none' }}
episode_count: {{ entity.episode_count }}
---

# {{ entity.name }}

## Overview
{{ entity.description }}

{% if entity.affiliations %}
## Affiliations
{% for org in entity.affiliations %}
- [[{{ org }}]]
{% endfor %}
{% endif %}

{% if entity.related_concepts %}
## Related Concepts
{% for concept in entity.related_concepts %}
- [[{{ concept }}]]
{% endfor %}
{% endif %}

{% if entity.related_practices %}
## Related Practices
{% for practice in entity.related_practices %}
- [[{{ practice }}]]
{% endfor %}
{% endif %}

## Episode Appearances
{% for ep in entity.episodes %}
- [[Episode {{ ep.number }}]]: {{ ep.title }}
{% endfor %}

{% if entity.domains %}
## Domains
{% for domain in entity.domains %}
- [[{{ domain }}]]
{% endfor %}
{% endif %}

---
*Mentions: {{ entity.mention_count }} | Importance Score: {{ entity.importance_score }}*
""")

        # Organization page template
        templates['ORGANIZATION'] = env.from_string("""---
type: organization
name: {{ entity.name }}
aliases: {{ entity.aliases | join(', ') if entity.aliases else 'none' }}
episode_count: {{ entity.episode_count }}
---

# {{ entity.name }}

## Overview
{{ entity.description }}

{% if entity.mission %}
## Mission
{{ entity.mission }}
{% endif %}

{% if entity.people %}
## People
{% for person in entity.people %}
- [[{{ person }}]]
{% endfor %}
{% endif %}

{% if entity.locations %}
## Locations
{% for location in entity.locations %}
- [[{{ location }}]]
{% endfor %}
{% endif %}

{% if entity.related_concepts %}
## Related Concepts
{% for concept in entity.related_concepts %}
- [[{{ concept }}]]
{% endfor %}
{% endif %}

## Episode Mentions
{% for ep in entity.episodes %}
- [[Episode {{ ep.number }}]]: {{ ep.title }}
{% endfor %}

{% if entity.domains %}
## Domains
{% for domain in entity.domains %}
- [[{{ domain }}]]
{% endfor %}
{% endif %}

---
*Mentions: {{ entity.mention_count }} | Importance Score: {{ entity.importance_score }}*
""")

        # Concept page template
        templates['CONCEPT'] = env.from_string("""---
type: concept
name: {{ entity.name }}
aliases: {{ entity.aliases | join(', ') if entity.aliases else 'none' }}
episode_count: {{ entity.episode_count }}
domains: {{ entity.domains | join(', ') if entity.domains else 'none' }}
---

# {{ entity.name }}

## Definition
{{ entity.description }}

{% if entity.domains %}
## Domains
{% for domain in entity.domains %}
- [[Domain: {{ domain }}]]
{% endfor %}
{% endif %}

{% if entity.related_concepts %}
## Related Concepts
{% for concept in entity.related_concepts %}
- [[{{ concept }}]]
{% endfor %}
{% endif %}

{% if entity.related_practices %}
## Related Practices
{% for practice in entity.related_practices %}
- [[{{ practice }}]]
{% endfor %}
{% endif %}

{% if entity.related_technologies %}
## Related Technologies
{% for tech in entity.related_technologies %}
- [[{{ tech }}]]
{% endfor %}
{% endif %}

{% if entity.organizations %}
## Organizations
{% for org in entity.organizations %}
- [[{{ org }}]]
{% endfor %}
{% endif %}

## Episode Mentions
{% for ep in entity.episodes %}
- [[Episode {{ ep.number }}]]: {{ ep.title }}
{% endfor %}

---
*Mentions: {{ entity.mention_count }} | Importance Score: {{ entity.importance_score }}*
""")

        # Practice page template
        templates['PRACTICE'] = env.from_string("""---
type: practice
name: {{ entity.name }}
aliases: {{ entity.aliases | join(', ') if entity.aliases else 'none' }}
episode_count: {{ entity.episode_count }}
---

# {{ entity.name }}

## Description
{{ entity.description }}

{% if entity.requirements %}
## Requirements
{% for req in entity.requirements %}
- {{ req }}
{% endfor %}
{% endif %}

{% if entity.benefits %}
## Benefits
{% for benefit in entity.benefits %}
- {{ benefit }}
{% endfor %}
{% endif %}

{% if entity.related_concepts %}
## Related Concepts
{% for concept in entity.related_concepts %}
- [[{{ concept }}]]
{% endfor %}
{% endif %}

{% if entity.related_practices %}
## Related Practices
{% for practice in entity.related_practices %}
- [[{{ practice }}]]
{% endfor %}
{% endif %}

{% if entity.examples %}
## Examples
{% for example in entity.examples %}
- {{ example }}
{% endfor %}
{% endif %}

## Episode Mentions
{% for ep in entity.episodes %}
- [[Episode {{ ep.number }}]]: {{ ep.title }}
{% endfor %}

{% if entity.domains %}
## Domains
{% for domain in entity.domains %}
- [[Domain: {{ domain }}]]
{% endfor %}
{% endif %}

---
*Mentions: {{ entity.mention_count }} | Importance Score: {{ entity.importance_score }}*
""")

        # Technology page template
        templates['TECHNOLOGY'] = env.from_string("""---
type: technology
name: {{ entity.name }}
aliases: {{ entity.aliases | join(', ') if entity.aliases else 'none' }}
episode_count: {{ entity.episode_count }}
---

# {{ entity.name }}

## Overview
{{ entity.description }}

{% if entity.purpose %}
## Purpose
{{ entity.purpose }}
{% endif %}

{% if entity.manufacturers %}
## Manufacturers/Developers
{% for mfg in entity.manufacturers %}
- [[{{ mfg }}]]
{% endfor %}
{% endif %}

{% if entity.uses %}
## Uses
{% for use in entity.uses %}
- {{ use }}
{% endfor %}
{% endif %}

{% if entity.related_concepts %}
## Related Concepts
{% for concept in entity.related_concepts %}
- [[{{ concept }}]]
{% endfor %}
{% endif %}

{% if entity.related_practices %}
## Related Practices
{% for practice in entity.related_practices %}
- [[{{ practice }}]]
{% endfor %}
{% endif %}

## Episode Mentions
{% for ep in entity.episodes %}
- [[Episode {{ ep.number }}]]: {{ ep.title }}
{% endfor %}

{% if entity.domains %}
## Domains
{% for domain in entity.domains %}
- [[Domain: {{ domain }}]]
{% endfor %}
{% endif %}

---
*Mentions: {{ entity.mention_count }} | Importance Score: {{ entity.importance_score }}*
""")

        # Location page template
        templates['PLACE'] = env.from_string("""---
type: place
name: {{ entity.name }}
aliases: {{ entity.aliases | join(', ') if entity.aliases else 'none' }}
episode_count: {{ entity.episode_count }}
---

# {{ entity.name }}

## Overview
{{ entity.description }}

{% if entity.location_type %}
## Type
{{ entity.location_type }}
{% endif %}

{% if entity.organizations %}
## Organizations
{% for org in entity.organizations %}
- [[{{ org }}]]
{% endfor %}
{% endif %}

{% if entity.people %}
## People
{% for person in entity.people %}
- [[{{ person }}]]
{% endfor %}
{% endif %}

{% if entity.related_concepts %}
## Related Concepts
{% for concept in entity.related_concepts %}
- [[{{ concept }}]]
{% endfor %}
{% endif %}

## Episode Mentions
{% for ep in entity.episodes %}
- [[Episode {{ ep.number }}]]: {{ ep.title }}
{% endfor %}

{% if entity.domains %}
## Domains
{% for domain in entity.domains %}
- [[Domain: {{ domain }}]]
{% endfor %}
{% endif %}

---
*Mentions: {{ entity.mention_count }} | Importance Score: {{ entity.importance_score }}*
""")

        # Episode page template
        templates['EPISODE'] = env.from_string("""---
type: episode
episode_number: {{ episode.number }}
title: {{ episode.title }}
{% if episode.guest %}guest: {{ episode.guest }}{% endif %}
{% if episode.host %}host: {{ episode.host }}{% endif %}
{% if episode.publish_date %}date: {{ episode.publish_date }}{% elif episode.date %}date: {{ episode.date }}{% endif %}
---

# Episode {{ episode.number }}: {{ episode.title }}

{% if episode.subtitle %}
**{{ episode.subtitle }}**
{% endif %}

{% if episode.host %}
**Host**: [[{{ episode.host }}]]
{% endif %}

{% if episode.guest %}
**Guest**: [[{{ episode.guest }}]]
{% endif %}

{% if episode.publish_date %}
**Published**: {{ episode.publish_date }}
{% elif episode.date %}
**Date**: {{ episode.date }}
{% endif %}

{% if episode.description %}
## Description
{{ episode.description }}
{% elif episode.summary %}
## Summary
{{ episode.summary }}
{% endif %}

{% if episode.key_concepts %}
## Key Concepts
{% for concept in episode.key_concepts %}
- [[{{ concept }}]]
{% endfor %}
{% endif %}

{% if episode.people %}
## People Mentioned
{% for person in episode.people %}
- [[{{ person }}]]
{% endfor %}
{% endif %}

{% if episode.organizations %}
## Organizations Mentioned
{% for org in episode.organizations %}
- [[{{ org }}]]
{% endfor %}
{% endif %}

{% if episode.technologies %}
## Technologies Discussed
{% for tech in episode.technologies %}
- [[{{ tech }}]]
{% endfor %}
{% endif %}

{% if episode.places %}
## Locations Mentioned
{% for place in episode.places %}
- [[{{ place }}]]
{% endfor %}
{% endif %}

{% if episode.practices %}
## Practices Discussed
{% for practice in episode.practices %}
- [[{{ practice }}]]
{% endfor %}
{% endif %}

{% if episode.domains %}
## Domains
{% for domain in episode.domains %}
- [[Domain: {{ domain }}]]
{% endfor %}
{% endif %}

## Links
{% if episode.url %}
- [🎧 Listen to Episode]({{ episode.url }})
{% endif %}
{% if episode.audio_url %}
- [📻 Direct Audio]({{ episode.audio_url }})
{% endif %}

{% if episode.related_episodes %}
## Related Episodes
{% for related in episode.related_episodes %}
- [{{ related.title }}]({{ related.url }})
{% endfor %}
{% endif %}

{% if episode.about_sections %}
## About
{% for section_name, section_content in episode.about_sections.items() %}
### {{ section_name.replace('_', ' ').title() }}
{{ section_content }}

{% endfor %}
{% endif %}

{% if episode.sponsors %}
## Sponsors
{{ episode.sponsors }}
{% endif %}

---
*Total Entities: {{ episode.entity_count }}{% if episode.transcript_length %} | Transcript Length: {{ episode.transcript_length }} characters{% endif %}*
""")

        return templates

    def sanitize_filename(self, name: str) -> str:
        """Convert entity name to safe filename."""
        # Remove special characters and replace spaces with underscores
        safe_name = re.sub(r'[<>:"/\\|?*]', '', name)
        safe_name = safe_name.replace(' ', '_')
        safe_name = safe_name.replace('&', 'and')
        safe_name = safe_name[:200]  # Limit length
        return safe_name

    def generate_entity_page(self, entity: Dict, entity_type: str) -> str:
        """Generate markdown page for an entity."""
        template = self.templates.get(entity_type, self.templates['CONCEPT'])
        return template.render(entity=entity)

    def generate_episode_page(self, episode: Dict) -> str:
        """Generate markdown page for an episode."""
        template = self.templates['EPISODE']
        return template.render(episode=episode)

    def generate_index_page(self, stats: Dict) -> str:
        """Generate main index page."""
        template = Environment(loader=BaseLoader()).from_string("""---
title: Y on Earth Podcast Knowledge Graph
---

# Y on Earth Podcast Knowledge Graph

This wiki contains {{ stats.total_entities }} entities extracted from {{ stats.total_episodes }} podcast episodes.

## Browse by Type

- [[People]] ({{ stats.people_count }} people)
- [[Organizations]] ({{ stats.organizations_count }} organizations)
- [[Concepts]] ({{ stats.concepts_count }} concepts)
- [[Practices]] ({{ stats.practices_count }} practices)
- [[Technologies]] ({{ stats.technologies_count }} technologies)
- [[Locations]] ({{ stats.locations_count }} locations)
- [[Episodes]] ({{ stats.episodes_count }} episodes)

## Browse by Domain

{% for domain in stats.domains %}
- [[Domain: {{ domain }}]]
{% endfor %}

## Featured Content

- [[Most Connected Concepts]]
- [[Bridging Concepts]]
- [[Knowledge Map]]
- [[Top People]]
- [[Key Organizations]]

## Statistics

- Total Entities: {{ stats.total_entities }}
- Total Episodes: {{ stats.total_episodes }}
- Total Relationships: {{ stats.total_relationships }}
- Entity Types: {{ stats.entity_types_count }}

---
*Last updated: {{ stats.last_updated }}*
""")
        return template.render(stats=stats)

    def generate_domain_index(self, domain: str, entities: List[Dict]) -> str:
        """Generate index page for a specific domain."""
        template = Environment(loader=BaseLoader()).from_string("""---
title: Domain - {{ domain }}
type: domain_index
---

# Domain: {{ domain }}

This domain contains {{ entity_count }} entities across {{ episode_count }} episodes.

## Concepts
{% for entity in concepts %}
- [[{{ entity.name }}]] - {{ entity.description[:100] }}...
{% endfor %}

## Practices
{% for entity in practices %}
- [[{{ entity.name }}]] - {{ entity.description[:100] }}...
{% endfor %}

## Organizations
{% for entity in organizations %}
- [[{{ entity.name }}]] - {{ entity.description[:100] }}...
{% endfor %}

## Episodes
{% for ep in episodes %}
- [[Episode {{ ep.number }}]]: {{ ep.title }}
{% endfor %}

---
[[Index|Back to Index]]
""")

        concepts = [e for e in entities if e['type'] == 'CONCEPT']
        practices = [e for e in entities if e['type'] == 'PRACTICE']
        organizations = [e for e in entities if e['type'] == 'ORGANIZATION']
        episodes = list({e['episode_number']: e for e in entities if 'episode_number' in e}.values())

        return template.render(
            domain=domain,
            entity_count=len(entities),
            episode_count=len(episodes),
            concepts=concepts,
            practices=practices,
            organizations=organizations,
            episodes=episodes
        )

    def generate_type_index(self, entity_type: str, entities: List[Dict]) -> str:
        """Generate index page for entities of a specific type."""
        template = Environment(loader=BaseLoader()).from_string("""---
title: {{ entity_type_title }}
type: entity_index
---

# {{ entity_type_title }}

Total {{ entity_type_lower }}: {{ entity_count }}

{% for entity in entities %}
## [[{{ entity.name }}]]
{{ entity.description[:200] }}...

*Episodes: {{ entity.episode_count }} | Mentions: {{ entity.mention_count }}*

---
{% endfor %}

[[Index|Back to Index]]
""")

        return template.render(
            entity_type_title=entity_type.title() + 's',
            entity_type_lower=entity_type.lower() + 's',
            entity_count=len(entities),
            entities=sorted(entities, key=lambda x: x.get('importance_score', 0), reverse=True)
        )
