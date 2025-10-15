#!/usr/bin/env python3
"""
Generate Visualization Data from v3.2.2 Knowledge Graph Extraction

Converts the raw v3.2.2 extraction (45,478 relationships) into the
D3.js visualization format used by KnowledgeGraph.html.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data")
EXTRACTION_DIR = DATA_DIR / "knowledge_graph_v3_2_2"
OUTPUT_FILE = EXTRACTION_DIR / "visualization_data.json"

# Domain colors (matching existing visualization)
DOMAIN_COLORS = {
    'Community': '#4CAF50',
    'Culture': '#9C27B0',
    'Economy': '#FF9800',
    'Ecology': '#2196F3',
    'Health': '#F44336'
}

# Simple domain assignment based on keywords
DOMAIN_KEYWORDS = {
    'Community': ['community', 'group', 'organization', 'network', 'collective'],
    'Culture': ['culture', 'art', 'music', 'tradition', 'indigenous', 'spiritual'],
    'Economy': ['economy', 'business', 'market', 'finance', 'trade', 'cost'],
    'Ecology': ['ecology', 'environment', 'nature', 'climate', 'soil', 'water', 'plant', 'animal', 'biodiversity'],
    'Health': ['health', 'wellness', 'food', 'nutrition', 'healing', 'medicine']
}


def assign_domains(entity_name, entity_type, description=""):
    """Assign domains to entity based on keywords."""
    text = f"{entity_name} {entity_type} {description}".lower()

    assigned_domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            assigned_domains.append(domain)

    # Default to Ecology if no match
    if not assigned_domains:
        assigned_domains = ['Ecology']

    return assigned_domains


def load_extractions():
    """Load all episode extractions."""
    logger.info("Loading v3.2.2 extractions...")

    all_relationships = []
    episodes_processed = 0

    for episode_file in sorted(EXTRACTION_DIR.glob("episode_*_v3_2_2.json")):
        try:
            with open(episode_file) as f:
                data = json.load(f)

            episode_num = data.get('episode', 0)
            relationships = data.get('relationships', [])

            # Add episode number to each relationship
            for rel in relationships:
                rel['episode'] = episode_num

            all_relationships.extend(relationships)
            episodes_processed += 1

            if episodes_processed % 20 == 0:
                logger.info(f"  Loaded {episodes_processed} episodes, {len(all_relationships)} relationships...")

        except Exception as e:
            logger.error(f"Error loading {episode_file}: {e}")
            continue

    logger.info(f"✓ Loaded {len(all_relationships)} relationships from {episodes_processed} episodes")
    return all_relationships


def build_graph(relationships):
    """Build node and link structures."""
    logger.info("Building graph structures...")

    # Entity tracking
    entity_info = defaultdict(lambda: {
        'mentions': 0,
        'episodes': set(),
        'types': set(),
        'relationships_in': [],
        'relationships_out': []
    })

    # Process relationships
    links = []
    for rel in relationships:
        source = rel['source']
        target = rel['target']
        episode = rel['episode']

        # Track entity info
        entity_info[source]['mentions'] += 1
        entity_info[source]['episodes'].add(episode)
        entity_info[source]['types'].add(rel.get('source_type', 'UNKNOWN'))
        entity_info[source]['relationships_out'].append(rel)

        entity_info[target]['mentions'] += 1
        entity_info[target]['episodes'].add(episode)
        entity_info[target]['types'].add(rel.get('target_type', 'UNKNOWN'))
        entity_info[target]['relationships_in'].append(rel)

        # Create link (only high confidence relationships)
        p_true = rel.get('p_true', 0.5)
        if p_true >= 0.7:  # Minimum threshold
            links.append({
                'source': source,
                'target': target,
                'type': rel['relationship'].upper().replace(' ', '_'),
                'strength': p_true
            })

    logger.info(f"✓ Found {len(entity_info)} unique entities")
    logger.info(f"✓ Created {len(links)} high-confidence links (p_true >= 0.7)")

    # Build nodes
    nodes = []
    entity_types_set = set()

    for entity_name, info in entity_info.items():
        # Determine primary type
        types_list = list(info['types'])
        primary_type = types_list[0] if types_list else 'UNKNOWN'
        entity_types_set.add(primary_type)

        # Calculate importance (based on mentions and episodes)
        mention_count = info['mentions']
        episode_count = len(info['episodes'])
        importance = min(1.0, (mention_count * 0.1 + episode_count * 0.05))

        # Assign domains
        domains = assign_domains(entity_name, primary_type)
        domain_colors = [DOMAIN_COLORS[d] for d in domains]

        # Generate description
        in_count = len(info['relationships_in'])
        out_count = len(info['relationships_out'])
        description = f"{primary_type} mentioned {mention_count} times across {episode_count} episodes. "
        description += f"Has {out_count} outgoing and {in_count} incoming relationships."

        nodes.append({
            'id': entity_name,
            'name': entity_name,
            'type': primary_type,
            'shape': 'circle',
            'description': description,
            'aliases': [],
            'domains': domains,
            'domain_colors': domain_colors,
            'importance': importance,
            'mention_count': mention_count,
            'episode_count': episode_count,
            'episodes': sorted(list(info['episodes'])),
            'community': 0  # Would need community detection for real values
        })

    logger.info(f"✓ Created {len(nodes)} nodes")
    logger.info(f"✓ Found {len(entity_types_set)} unique entity types")

    return nodes, links, sorted(list(entity_types_set))


def calculate_statistics(nodes, links):
    """Calculate graph statistics."""
    logger.info("Calculating statistics...")

    # Domain distribution
    domain_counts = defaultdict(int)
    for node in nodes:
        for domain in node['domains']:
            domain_counts[domain] += 1

    # Average importance and connections
    total_importance = sum(n['importance'] for n in nodes)
    total_connections = len(links) * 2  # Each link connects 2 nodes

    stats = {
        'total_nodes': len(nodes),
        'total_links': len(links),
        'total_communities': 1,  # Would need community detection
        'entity_types': len(set(n['type'] for n in nodes)),
        'domain_distribution': dict(domain_counts),
        'avg_importance': total_importance / len(nodes) if nodes else 0,
        'avg_connections': total_connections / len(nodes) if nodes else 0
    }

    logger.info(f"✓ Statistics: {stats['total_nodes']} nodes, {stats['total_links']} links")
    return stats


def generate_visualization_data():
    """Main generation function."""
    logger.info("=" * 80)
    logger.info("KNOWLEDGE GRAPH v3.2.2 VISUALIZATION GENERATOR")
    logger.info("=" * 80)
    logger.info("")

    # Load extractions
    relationships = load_extractions()

    # Build graph
    nodes, links, entity_types = build_graph(relationships)

    # Define domains
    domains = [
        {'name': name, 'color': color}
        for name, color in DOMAIN_COLORS.items()
    ]

    # Calculate statistics
    statistics = calculate_statistics(nodes, links)

    # Build output
    visualization_data = {
        'nodes': nodes,
        'links': links,
        'domains': domains,
        'entity_types': entity_types,
        'statistics': statistics,
        'metadata': {
            'version': 'v3.2.2',
            'generated': datetime.now().isoformat(),
            'source': 'knowledge_graph_v3_2_2 extraction',
            'total_episodes': 172,
            'total_relationships': len(relationships),
            'description': 'Full production extraction with 45,478 relationships, 93.1% high confidence'
        }
    }

    # Save
    logger.info("")
    logger.info(f"Writing visualization data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(visualization_data, f, indent=2)

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved {file_size_mb:.1f} MB")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✨ VISUALIZATION DATA GENERATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Nodes: {len(nodes)}")
    logger.info(f"Links: {len(links)}")
    logger.info(f"Entity types: {len(entity_types)}")
    logger.info(f"Domains: {len(domains)}")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Copy to production: sudo mkdir -p /var/www/yonearth/data/knowledge_graph_v3_2_2")
    logger.info(f"2. sudo cp {OUTPUT_FILE} /var/www/yonearth/data/knowledge_graph_v3_2_2/")
    logger.info("3. Test at https://earthdo.me/KnowledgeGraph.html")
    logger.info("")


if __name__ == "__main__":
    generate_visualization_data()
