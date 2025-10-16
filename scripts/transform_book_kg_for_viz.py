#!/usr/bin/env python3
"""
Transform "Our Biggest Deal" book knowledge graph data into visualization format.

This script reads the extracted knowledge graph data from the book chapters
and transforms it into the format expected by the D3.js knowledge graph visualization.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any
import colorsys


def generate_color_palette(n: int) -> List[str]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def determine_domain(entity_type: str, context: str = "") -> str:
    """Determine the domain category for an entity based on its type and context."""
    type_lower = entity_type.lower()
    context_lower = context.lower()

    # Domain classification rules
    if type_lower in ['person', 'people']:
        return 'People'
    elif type_lower in ['organization', 'company', 'institution', 'foundation', 'bank']:
        return 'Organizations'
    elif type_lower in ['book', 'essay', 'document', 'publication']:
        return 'Publications'
    elif type_lower in ['concept', 'idea', 'framework', 'principle', 'theory']:
        return 'Concepts'
    elif type_lower in ['location', 'place', 'region', 'country']:
        return 'Places'
    elif type_lower in ['event', 'conference', 'summit']:
        return 'Events'
    elif type_lower in ['product', 'service', 'technology', 'tool']:
        return 'Products'
    else:
        # Default to concepts for unknown types
        return 'Concepts'


def load_chapter_data(book_dir: Path) -> List[Dict]:
    """Load all chapter extraction files from a book directory."""
    chapters_dir = book_dir / 'chapters'
    if not chapters_dir.exists():
        print(f"Warning: No chapters directory found at {chapters_dir}")
        return []

    chapter_files = sorted(chapters_dir.glob('*.json'))
    all_relationships = []

    for chapter_file in chapter_files:
        try:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'relationships' in data:
                    # Add metadata to each relationship
                    for rel in data['relationships']:
                        rel['_source_file'] = chapter_file.name
                        rel['_book'] = data.get('metadata', {}).get('book', 'unknown')
                        rel['_section'] = data.get('metadata', {}).get('section', 'unknown')
                    all_relationships.extend(data['relationships'])
        except Exception as e:
            print(f"Error loading {chapter_file}: {e}")

    return all_relationships


def transform_to_visualization_format(relationships: List[Dict]) -> Dict:
    """Transform book relationships into visualization format."""

    # Track entities and their metadata
    entities: Dict[str, Dict[str, Any]] = {}
    entity_mentions: Dict[str, int] = defaultdict(int)
    entity_pages: Dict[str, Set[int]] = defaultdict(set)
    entity_contexts: Dict[str, List[str]] = defaultdict(list)

    # Track relationships for graph links
    links = []

    # Process all relationships
    for rel in relationships:
        source_name = rel.get('source', '')
        target_name = rel.get('target', '')
        relationship_type = rel.get('relationship', 'RELATED_TO')
        source_type = rel.get('source_type', 'Unknown')
        target_type = rel.get('target_type', 'Unknown')
        context = rel.get('context', '')
        page = rel.get('page', 0)
        confidence = rel.get('p_true', 0.5)

        # Skip low confidence relationships
        if confidence < 0.7:
            continue

        # Create or update source entity
        if source_name and source_name not in entities:
            entities[source_name] = {
                'id': source_name,
                'name': source_name,
                'type': source_type,
                'contexts': []
            }

        # Create or update target entity
        if target_name and target_name not in entities:
            entities[target_name] = {
                'id': target_name,
                'name': target_name,
                'type': target_type,
                'contexts': []
            }

        # Track mentions and pages
        if source_name:
            entity_mentions[source_name] += 1
            entity_pages[source_name].add(page)
            if context and len(context) > 20:
                entity_contexts[source_name].append(context[:200])

        if target_name:
            entity_mentions[target_name] += 1
            entity_pages[target_name].add(page)
            if context and len(context) > 20:
                entity_contexts[target_name].append(context[:200])

        # Create link
        if source_name and target_name:
            links.append({
                'source': source_name,
                'target': target_name,
                'type': relationship_type,
                'relationship_type': relationship_type,
                'strength': min(confidence, 1.0),
                'page': page
            })

    # Calculate importance scores
    max_mentions = max(entity_mentions.values()) if entity_mentions else 1

    # Determine domains and entity types
    all_domains = set()
    all_entity_types = set()

    for entity_name, entity_data in entities.items():
        entity_type = entity_data['type']
        all_entity_types.add(entity_type)

        # Determine domain
        context_sample = ' '.join(entity_contexts[entity_name][:3])
        domain = determine_domain(entity_type, context_sample)
        all_domains.add(domain)

        # Update entity with full metadata
        entity_data.update({
            'domains': [domain],
            'mention_count': entity_mentions[entity_name],
            'page_count': len(entity_pages[entity_name]),
            'pages': sorted(list(entity_pages[entity_name])),
            'importance': entity_mentions[entity_name] / max_mentions,
            'description': entity_contexts[entity_name][0] if entity_contexts[entity_name] else f"{entity_type} from Our Biggest Deal",
            'aliases': []
        })

    # Create domain color mapping
    domain_list = sorted(list(all_domains))
    domain_colors = generate_color_palette(len(domain_list))
    domain_info = [
        {'name': domain, 'color': color}
        for domain, color in zip(domain_list, domain_colors)
    ]
    domain_color_map = {d['name']: d['color'] for d in domain_info}

    # Add domain colors to entities
    for entity_data in entities.values():
        entity_data['domain_colors'] = [
            domain_color_map[domain] for domain in entity_data['domains']
        ]

    # Calculate statistics
    all_page_numbers = []
    for page_set in entity_pages.values():
        all_page_numbers.extend(list(page_set))

    statistics = {
        'total_entities': len(entities),
        'total_relationships': len(links),
        'total_pages': max(all_page_numbers) if all_page_numbers else 0,
        'total_communities': len(all_domains),
        'avg_mentions_per_entity': sum(entity_mentions.values()) / len(entity_mentions) if entity_mentions else 0,
        'entity_types': sorted(list(all_entity_types))
    }

    # Build final visualization data structure
    viz_data = {
        'nodes': list(entities.values()),
        'links': links,
        'domains': domain_info,
        'entity_types': sorted(list(all_entity_types)),
        'statistics': statistics,
        'metadata': {
            'book_title': 'Our Biggest Deal',
            'author': 'Aaron William Perry',
            'extraction_version': 'v14_3_8',
            'total_entities': len(entities),
            'total_relationships': len(links)
        }
    }

    return viz_data


def main():
    """Main function to transform book KG data for visualization."""

    # Get the latest version directory
    base_dir = Path('/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/output/our_biggest_deal')

    # Find the latest version directory
    version_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('v14')]
    if not version_dirs:
        print("Error: No version directories found")
        return

    latest_version = sorted(version_dirs)[-1]
    print(f"Using latest version: {latest_version.name}")

    # Load all chapter data
    print("Loading chapter data...")
    relationships = load_chapter_data(latest_version)
    print(f"Loaded {len(relationships)} relationships")

    if not relationships:
        print("Error: No relationships found")
        return

    # Transform to visualization format
    print("Transforming data...")
    viz_data = transform_to_visualization_format(relationships)

    # Save to output file
    output_dir = Path('/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_books')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'our_biggest_deal_visualization.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Visualization data saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  - Entities: {viz_data['statistics']['total_entities']}")
    print(f"  - Relationships: {viz_data['statistics']['total_relationships']}")
    print(f"  - Entity Types: {len(viz_data['entity_types'])}")
    print(f"  - Domains: {len(viz_data['domains'])}")
    print(f"  - Pages: {viz_data['statistics']['total_pages']}")

    print(f"\nDomains:")
    for domain in viz_data['domains']:
        print(f"  - {domain['name']}: {domain['color']}")

    print(f"\nEntity Types:")
    for entity_type in viz_data['entity_types']:
        count = sum(1 for n in viz_data['nodes'] if n['type'] == entity_type)
        print(f"  - {entity_type}: {count}")


if __name__ == '__main__':
    main()
