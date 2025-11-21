"""
Export Knowledge Graph Data for D3.js Visualization

This script reads entity extraction files and generates a JSON format suitable
for interactive force-directed graph visualization using D3.js.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeGraphExporter:
    """Export knowledge graph data for visualization."""

    # Entity type to shape mapping
    ENTITY_SHAPES = {
        "PERSON": "circle",
        "ORGANIZATION": "square",
        "COMPANY": "square",
        "NONPROFIT": "square",
        "UNIVERSITY": "square",
        "CONCEPT": "circle",
        "TECHNOLOGY": "diamond",
        "PRACTICE": "diamond",
        "MATERIAL": "diamond",
        "PRODUCT": "diamond",
        "LOCATION": "triangle",
        "CITY": "triangle",
        "STATE": "triangle",
        "COUNTRY": "triangle",
        "TOPIC": "hexagon",
        "EVENT": "star"
    }

    # Domain colors (matching YonEarth 5 pillars)
    DOMAIN_COLORS = {
        "Community": "#4CAF50",  # Green
        "Culture": "#9C27B0",    # Purple
        "Economy": "#FF9800",    # Orange
        "Ecology": "#2196F3",    # Blue
        "Health": "#F44336"      # Red
    }

    # Keywords to domain mapping
    DOMAIN_KEYWORDS = {
        "Community": [
            "community", "social", "people", "local", "neighborhood", "group",
            "collaboration", "cooperative", "network", "volunteer", "engagement"
        ],
        "Culture": [
            "culture", "art", "tradition", "indigenous", "knowledge", "wisdom",
            "education", "awareness", "values", "spiritual", "ceremony"
        ],
        "Economy": [
            "economy", "business", "market", "finance", "trade", "investment",
            "entrepreneurship", "economic", "commerce", "industry", "enterprise"
        ],
        "Ecology": [
            "ecology", "environment", "nature", "ecosystem", "biodiversity",
            "conservation", "wildlife", "habitat", "species", "climate", "soil",
            "regenerative", "permaculture", "organic", "compost", "biochar"
        ],
        "Health": [
            "health", "wellness", "nutrition", "food", "medicine", "healing",
            "disease", "toxin", "safety", "wellbeing", "mental", "physical"
        ]
    }

    def __init__(self, data_dir: str = "/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph"):
        self.data_dir = Path(data_dir)
        self.entities_dir = self.data_dir / "entities"
        self.relationships_dir = self.data_dir / "relationships"
        self.output_file = self.data_dir / "visualization_data.json"

        # Graph data
        self.entities = {}  # name -> entity data
        self.relationships = []
        self.entity_mentions = defaultdict(int)  # Track importance
        self.entity_episodes = defaultdict(set)  # Track which episodes mention each entity

    def load_entity_files(self, max_episodes: int = None) -> None:
        """Load entity extraction files."""
        logger.info(f"Loading entity files from {self.entities_dir}")

        entity_files = sorted(self.entities_dir.glob("episode_*_extraction.json"))

        if max_episodes:
            entity_files = entity_files[:max_episodes]

        logger.info(f"Found {len(entity_files)} entity files")

        for file_path in entity_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                episode_num = data.get("episode_number", 0)
                entities = data.get("entities", [])

                for entity in entities:
                    name = entity.get("name", "").strip()
                    if not name:
                        continue

                    entity_type = entity.get("type", "CONCEPT")
                    description = entity.get("description", "")
                    aliases = entity.get("aliases", [])

                    # Track mentions and episodes
                    self.entity_mentions[name] += 1
                    self.entity_episodes[name].add(episode_num)

                    # Merge or add entity
                    if name not in self.entities:
                        self.entities[name] = {
                            "name": name,
                            "type": entity_type,
                            "description": description,
                            "aliases": aliases,
                            "episodes": set([episode_num]),
                            "domains": set()
                        }
                    else:
                        # Merge data
                        existing = self.entities[name]
                        existing["episodes"].add(episode_num)
                        if len(description) > len(existing["description"]):
                            existing["description"] = description
                        existing["aliases"].extend([a for a in aliases if a not in existing["aliases"]])

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded {len(self.entities)} unique entities")

    def assign_domains(self) -> None:
        """Assign domains to entities based on their descriptions."""
        logger.info("Assigning domains to entities...")

        for name, entity in self.entities.items():
            text = (entity["description"] + " " + " ".join(entity["aliases"])).lower()

            # Score each domain
            domain_scores = {}
            for domain, keywords in self.DOMAIN_KEYWORDS.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    domain_scores[domain] = score

            # Assign domains with significant scores
            if domain_scores:
                max_score = max(domain_scores.values())
                threshold = max(1, max_score * 0.5)  # At least 50% of max score

                for domain, score in domain_scores.items():
                    if score >= threshold:
                        entity["domains"].add(domain)

            # Default to Ecology if no clear domain
            if not entity["domains"]:
                entity["domains"].add("Ecology")

        logger.info("Domain assignment complete")

    def detect_relationships(self) -> None:
        """Load actual semantic relationships from extraction files."""
        logger.info("Loading semantic relationships from extraction files...")

        if not self.relationships_dir.exists():
            logger.warning(f"Relationships directory not found: {self.relationships_dir}")
            logger.warning("Falling back to co-occurrence detection...")
            self._detect_cooccurrence_relationships()
            return

        relationship_files = sorted(self.relationships_dir.glob("episode_*_extraction.json"))
        logger.info(f"Found {len(relationship_files)} relationship files")

        # Track relationship counts for deduplication and strength
        relationship_map = defaultdict(lambda: {"count": 0, "type": None})

        for file_path in relationship_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                relationships = data.get("relationships", [])

                for rel in relationships:
                    # Handle both old format (string) and new format (object)
                    if "source_entity" in rel:
                        # Old format: source_entity and target_entity as strings
                        source = rel.get("source_entity", "").strip()
                        target = rel.get("target_entity", "").strip()
                    else:
                        # New format: source and target as objects
                        source_obj = rel.get("source", {})
                        target_obj = rel.get("target", {})
                        source = source_obj.get("name", "").strip() if isinstance(source_obj, dict) else str(source_obj).strip()
                        target = target_obj.get("name", "").strip() if isinstance(target_obj, dict) else str(target_obj).strip()

                    rel_type = rel.get("relationship_type", "RELATED_TO")

                    # Skip if source or target not in our entities
                    if not source or not target:
                        continue
                    if source not in self.entities or target not in self.entities:
                        continue

                    # Create directed key to preserve relationship direction
                    key = (source, target, rel_type)

                    # Track relationship
                    relationship_map[key]["count"] += 1
                    # Store the relationship type
                    relationship_map[key]["type"] = rel_type

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        # Create relationships with actual semantic types
        for (source, target, rel_type), data in relationship_map.items():
            self.relationships.append({
                "source": source,
                "target": target,
                "type": data["type"],
                "strength": min(data["count"] / 3.0, 1.0)  # Normalize to 0-1
            })

        logger.info(f"Loaded {len(self.relationships)} semantic relationships")

        # Log relationship type distribution
        type_counts = Counter([r["type"] for r in self.relationships])
        logger.info(f"Relationship types: {dict(type_counts.most_common(10))}")

    def _detect_cooccurrence_relationships(self) -> None:
        """Fallback: Detect relationships based on co-occurrence in episodes."""
        logger.info("Detecting relationships through co-occurrence...")

        # Group entities by episode
        episode_entities = defaultdict(list)
        for name, entity in self.entities.items():
            for episode in entity["episodes"]:
                episode_entities[episode].append(name)

        # Find co-occurrences
        relationship_counts = defaultdict(int)

        for episode, entity_names in episode_entities.items():
            # Create relationships for entities in same episode
            for i, name1 in enumerate(entity_names):
                for name2 in entity_names[i+1:]:
                    pair = tuple(sorted([name1, name2]))
                    relationship_counts[pair] += 1

        # Create relationships with strength based on co-occurrence
        for (source, target), count in relationship_counts.items():
            if count >= 2:  # Only include relationships with 2+ co-occurrences
                self.relationships.append({
                    "source": source,
                    "target": target,
                    "type": "co_occurs_with",
                    "strength": min(count / 5.0, 1.0)  # Normalize to 0-1
                })

        logger.info(f"Detected {len(self.relationships)} co-occurrence relationships")

    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using NetworkX."""
        logger.info("Detecting communities...")

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        for name in self.entities.keys():
            G.add_node(name)

        # Add edges
        for rel in self.relationships:
            G.add_edge(rel["source"], rel["target"], weight=rel["strength"])

        # Detect communities using Louvain algorithm
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G)
        except ImportError:
            # Fallback: simple connected components
            logger.warning("python-louvain not available, using connected components")
            communities = {}
            for i, component in enumerate(nx.connected_components(G)):
                for node in component:
                    communities[node] = i

        logger.info(f"Detected {len(set(communities.values()))} communities")
        return communities

    def export_visualization_data(self) -> None:
        """Export data in D3.js compatible format."""
        logger.info("Exporting visualization data...")

        # Detect communities
        communities = self.detect_communities()

        # Prepare nodes
        nodes = []
        for name, entity in self.entities.items():
            # Calculate importance (0-1 scale)
            importance = min(self.entity_mentions[name] / 10.0, 1.0)

            nodes.append({
                "id": name,
                "name": name,
                "type": entity["type"],
                "shape": self.ENTITY_SHAPES.get(entity["type"], "circle"),
                "description": entity["description"],
                "aliases": entity["aliases"],
                "domains": list(entity["domains"]),
                "domain_colors": [self.DOMAIN_COLORS[d] for d in entity["domains"]],
                "importance": importance,
                "mention_count": self.entity_mentions[name],
                "episode_count": len(entity["episodes"]),
                "episodes": sorted(list(entity["episodes"])),
                "community": communities.get(name, 0)
            })

        # Prepare links
        links = []
        for rel in self.relationships:
            links.append({
                "source": rel["source"],
                "target": rel["target"],
                "type": rel["type"],
                "strength": rel["strength"]
            })

        # Calculate statistics
        stats = {
            "total_nodes": len(nodes),
            "total_links": len(links),
            "total_communities": len(set(communities.values())),
            "entity_types": dict(Counter([n["type"] for n in nodes])),
            "domain_distribution": dict(Counter([d for n in nodes for d in n["domains"]])),
            "avg_importance": np.mean([n["importance"] for n in nodes]),
            "avg_connections": len(links) * 2 / len(nodes) if nodes else 0
        }

        # Export data
        output_data = {
            "nodes": nodes,
            "links": links,
            "domains": [
                {"name": name, "color": color}
                for name, color in self.DOMAIN_COLORS.items()
            ],
            "entity_types": list(set([n["type"] for n in nodes])),
            "statistics": stats,
            "metadata": {
                "generated": "2025-10-01",
                "source": "YonEarth Podcast Episodes",
                "version": "1.0"
            }
        }

        # Write to file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Exported visualization data to {self.output_file}")
        logger.info(f"Statistics: {stats}")

    def export(self, max_episodes: int = None) -> str:
        """Main export method."""
        self.load_entity_files(max_episodes=max_episodes)
        self.assign_domains()
        self.detect_relationships()
        self.export_visualization_data()
        return str(self.output_file)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Export knowledge graph for visualization")
    parser.add_argument("--max-episodes", type=int, help="Maximum number of episodes to process")
    parser.add_argument("--data-dir", type=str,
                       default="/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph",
                       help="Data directory")

    args = parser.parse_args()

    exporter = KnowledgeGraphExporter(data_dir=args.data_dir)
    output_path = exporter.export(max_episodes=args.max_episodes)

    print(f"\n✅ Successfully exported visualization data to: {output_path}")
    print(f"📊 Nodes: {len(exporter.entities)}")
    print(f"🔗 Links: {len(exporter.relationships)}")


if __name__ == "__main__":
    main()
