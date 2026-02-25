"""
Add Episode and Book Entities to Knowledge Graph

This script enhances the KG visualization data by:
1. Creating EPISODE type nodes for each podcast episode
2. Creating BOOK type nodes for each book
3. Linking existing entities to their source episodes/books via relationships

Usage:
    python scripts/add_episode_book_entities.py [--output OUTPUT_FILE]

This is Task 1 of the KG Chat Entity Linking Enhancement project.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EpisodeBookEntityAdder:
    """Add episode and book entities to the knowledge graph."""

    # Domain for media entities
    MEDIA_DOMAIN = "Culture"
    MEDIA_COLOR = "#9C27B0"  # Purple

    def __init__(
        self,
        base_dir: str = "/Users/darrenzal/projects/yonearth-gaia-chatbot",
        visualization_file: Optional[str] = None
    ):
        self.base_dir = Path(base_dir)
        self.transcripts_dir = self.base_dir / "data" / "transcripts"
        self.books_dir = self.base_dir / "data" / "books"
        self.kg_v3_dir = self.base_dir / "data" / "knowledge_graph_v3_2_2"

        # Output file - can be local or on server
        if visualization_file:
            self.visualization_file = Path(visualization_file)
        else:
            self.visualization_file = self.base_dir / "data" / "knowledge_graph" / "visualization_data.json"

        # Data to be processed
        self.episodes: Dict[int, dict] = {}
        self.books: Dict[str, dict] = {}
        self.existing_data: Optional[dict] = None

        # Track entity-episode relationships
        self.entity_episodes: Dict[str, set] = defaultdict(set)

    def load_episodes(self) -> int:
        """Load episode metadata from transcript files."""
        logger.info(f"Loading episodes from {self.transcripts_dir}")

        count = 0
        for transcript_file in sorted(self.transcripts_dir.glob("episode_*.json")):
            try:
                with open(transcript_file, 'r') as f:
                    data = json.load(f)

                episode_num = data.get("episode_number")
                if episode_num is None:
                    # Try to extract from filename
                    filename = transcript_file.stem
                    if filename.startswith("episode_"):
                        try:
                            episode_num = int(filename.split("_")[1])
                        except (ValueError, IndexError):
                            continue

                if episode_num is not None:
                    self.episodes[episode_num] = {
                        "episode_number": episode_num,
                        "title": data.get("title", f"Episode {episode_num}"),
                        "audio_url": data.get("audio_url", ""),
                        "url": data.get("url", ""),
                        "publish_date": data.get("publish_date", ""),
                        "description": data.get("description", data.get("subtitle", "")),
                        "about_sections": data.get("about_sections", {})
                    }
                    count += 1

            except Exception as e:
                logger.warning(f"Error loading {transcript_file}: {e}")

        logger.info(f"Loaded {count} episodes")
        return count

    def load_books(self) -> int:
        """Load book metadata from book directories."""
        logger.info(f"Loading books from {self.books_dir}")

        count = 0
        for book_dir in self.books_dir.iterdir():
            if not book_dir.is_dir():
                continue

            metadata_file = book_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)

                book_id = book_dir.name
                self.books[book_id] = {
                    "book_id": book_id,
                    "title": data.get("title", book_id),
                    "author": data.get("author", ""),
                    "description": data.get("description", ""),
                    "publication_year": data.get("publication_year"),
                    "topics": data.get("topics", []),
                    "audiobook_url": data.get("audiobook_url", ""),
                    "ebook_url": data.get("ebook_url", ""),
                    "print_url": data.get("print_url", "")
                }
                count += 1

            except Exception as e:
                logger.warning(f"Error loading {metadata_file}: {e}")

        logger.info(f"Loaded {count} books")
        return count

    def load_entity_episode_mappings(self) -> None:
        """Load mappings of which entities appear in which episodes from KG extraction files."""
        logger.info(f"Loading entity-episode mappings from {self.kg_v3_dir}")

        for kg_file in self.kg_v3_dir.glob("episode_*_v3_2_2.json"):
            try:
                with open(kg_file, 'r') as f:
                    data = json.load(f)

                # Extract episode number from filename
                filename = kg_file.stem
                try:
                    episode_num = int(filename.split("_")[1])
                except (ValueError, IndexError):
                    continue

                # Extract entities from relationships
                relationships = data.get("relationships", [])
                for rel in relationships:
                    # Handle the v3_2_2 format with source/target entities
                    source = rel.get("source", "")
                    target = rel.get("target", "")

                    if source:
                        self.entity_episodes[source].add(episode_num)
                    if target:
                        self.entity_episodes[target].add(episode_num)

            except Exception as e:
                logger.warning(f"Error loading {kg_file}: {e}")

        logger.info(f"Mapped {len(self.entity_episodes)} entities to episodes")

    def load_existing_visualization(self) -> bool:
        """Load existing visualization data if available."""
        if self.visualization_file.exists():
            try:
                with open(self.visualization_file, 'r') as f:
                    self.existing_data = json.load(f)
                logger.info(f"Loaded existing visualization data: {len(self.existing_data.get('nodes', []))} nodes")
                return True
            except Exception as e:
                logger.warning(f"Could not load existing visualization: {e}")
        return False

    def extract_guest_from_title(self, title: str) -> Optional[str]:
        """Extract guest name from episode title."""
        # Common patterns: "Episode X - Guest Name" or "Episode X: Guest Name - Topic"
        if " - " in title:
            parts = title.split(" - ", 1)
            if len(parts) > 1:
                # Second part might be "Guest Name" or "Guest Name - Topic"
                guest_part = parts[1].split(" - ")[0] if " - " in parts[1] else parts[1]
                # Clean up common suffixes
                for suffix in [", Founder", ", CEO", ", Author", ", Director", ", President"]:
                    if suffix in guest_part:
                        guest_part = guest_part.split(suffix)[0]
                return guest_part.strip()
        return None

    def create_episode_nodes(self) -> List[dict]:
        """Create episode nodes for the KG."""
        nodes = []

        for episode_num, episode_data in self.episodes.items():
            # Create a clean name for the node
            title = episode_data["title"]
            guest_name = self.extract_guest_from_title(title)

            # Build description
            description = episode_data.get("description", "")
            about_sections = episode_data.get("about_sections", {})
            if about_sections:
                # Add about sections to description
                for key, value in about_sections.items():
                    if value and len(description) < 500:
                        description = f"{description}\n\n{value}" if description else value

            node = {
                "id": f"episode_{episode_num}",
                "name": title,
                "type": "EPISODE",
                "shape": "star",  # Use star shape for episodes
                "description": description[:500] if description else f"Y on Earth Podcast Episode {episode_num}",
                "aliases": [f"Episode {episode_num}", f"Ep {episode_num}", f"Ep. {episode_num}"],
                "domains": [self.MEDIA_DOMAIN],
                "domain_colors": [self.MEDIA_COLOR],
                "importance": 0.8,  # Episodes are important reference nodes
                "mention_count": 1,
                "episode_count": 1,
                "episodes": [episode_num],
                "community": 0,  # Will be updated later if needed
                # Episode-specific fields
                "episode_number": episode_num,
                "guest_name": guest_name,
                "audio_url": episode_data.get("audio_url", ""),
                "url": episode_data.get("url", ""),
                "publish_date": episode_data.get("publish_date", "")
            }

            nodes.append(node)

        logger.info(f"Created {len(nodes)} episode nodes")
        return nodes

    def create_book_nodes(self) -> List[dict]:
        """Create book nodes for the KG."""
        nodes = []

        for book_id, book_data in self.books.items():
            title = book_data["title"]

            node = {
                "id": f"book_{book_id}",
                "name": title,
                "type": "BOOK",
                "shape": "square",  # Use square shape for books
                "description": book_data.get("description", ""),
                "aliases": [book_id.replace("-", " ").title()],
                "domains": [self.MEDIA_DOMAIN],
                "domain_colors": [self.MEDIA_COLOR],
                "importance": 0.9,  # Books are high importance reference nodes
                "mention_count": 1,
                "episode_count": 0,
                "episodes": [],
                "community": 0,
                # Book-specific fields
                "book_id": book_id,
                "author": book_data.get("author", ""),
                "publication_year": book_data.get("publication_year"),
                "topics": book_data.get("topics", []),
                "audiobook_url": book_data.get("audiobook_url", ""),
                "ebook_url": book_data.get("ebook_url", ""),
                "print_url": book_data.get("print_url", "")
            }

            nodes.append(node)

        logger.info(f"Created {len(nodes)} book nodes")
        return nodes

    def create_source_relationships(self) -> List[dict]:
        """Create relationships linking entities to their source episodes."""
        links = []

        for entity_name, episode_nums in self.entity_episodes.items():
            for episode_num in episode_nums:
                if episode_num in self.episodes:
                    links.append({
                        "source": entity_name,
                        "target": f"episode_{episode_num}",
                        "type": "MENTIONED_IN",
                        "strength": 0.5
                    })

        logger.info(f"Created {len(links)} entity-episode relationships")
        return links

    def merge_into_visualization(self, episode_nodes: List[dict], book_nodes: List[dict],
                                  source_links: List[dict]) -> dict:
        """Merge new nodes and links into existing visualization data."""

        if self.existing_data:
            # Start with existing data
            nodes = self.existing_data.get("nodes", [])
            links = self.existing_data.get("links", [])

            # Remove any existing episode/book nodes (in case of re-run)
            nodes = [n for n in nodes if not n["id"].startswith("episode_") and not n["id"].startswith("book_")]
            links = [l for l in links if not l["target"].startswith("episode_") and not l["target"].startswith("book_")]

            # Add new nodes
            nodes.extend(episode_nodes)
            nodes.extend(book_nodes)

            # Add new links
            links.extend(source_links)

            # Update metadata
            output_data = self.existing_data.copy()
            output_data["nodes"] = nodes
            output_data["links"] = links

            # Update entity_types to include new types
            entity_types = set(output_data.get("entity_types", []))
            entity_types.add("EPISODE")
            entity_types.add("BOOK")
            output_data["entity_types"] = list(entity_types)

            # Update statistics
            output_data["statistics"]["total_nodes"] = len(nodes)
            output_data["statistics"]["total_links"] = len(links)

        else:
            # Create new visualization data structure
            nodes = episode_nodes + book_nodes
            links = source_links

            output_data = {
                "nodes": nodes,
                "links": links,
                "domains": [
                    {"name": "Community", "color": "#4CAF50"},
                    {"name": "Culture", "color": "#9C27B0"},
                    {"name": "Economy", "color": "#FF9800"},
                    {"name": "Ecology", "color": "#2196F3"},
                    {"name": "Health", "color": "#F44336"}
                ],
                "entity_types": ["EPISODE", "BOOK"],
                "statistics": {
                    "total_nodes": len(nodes),
                    "total_links": len(links),
                    "episode_count": len(episode_nodes),
                    "book_count": len(book_nodes)
                },
                "metadata": {
                    "generated": "2026-02-01",
                    "source": "YonEarth Podcast Episodes and Books",
                    "version": "2.0",
                    "includes_episodes": True,
                    "includes_books": True
                }
            }

        return output_data

    def export(self, output_file: Optional[str] = None) -> str:
        """Main export method."""

        # Load all data
        self.load_episodes()
        self.load_books()
        self.load_entity_episode_mappings()
        self.load_existing_visualization()

        # Create new nodes
        episode_nodes = self.create_episode_nodes()
        book_nodes = self.create_book_nodes()
        source_links = self.create_source_relationships()

        # Merge into visualization data
        output_data = self.merge_into_visualization(episode_nodes, book_nodes, source_links)

        # Write output
        output_path = Path(output_file) if output_file else self.visualization_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"✅ Exported visualization data to {output_path}")
        logger.info(f"   Total nodes: {len(output_data['nodes'])}")
        logger.info(f"   Total links: {len(output_data['links'])}")
        logger.info(f"   Episodes: {len(episode_nodes)}")
        logger.info(f"   Books: {len(book_nodes)}")

        return str(output_path)

    def generate_episode_json(self, output_file: str = None) -> str:
        """Generate just the episode/book data as a separate JSON file."""
        self.load_episodes()
        self.load_books()

        episode_nodes = self.create_episode_nodes()
        book_nodes = self.create_book_nodes()

        data = {
            "episodes": {node["id"]: node for node in episode_nodes},
            "books": {node["id"]: node for node in book_nodes},
            "metadata": {
                "generated": "2026-02-01",
                "episode_count": len(episode_nodes),
                "book_count": len(book_nodes)
            }
        }

        output_path = Path(output_file) if output_file else self.base_dir / "data" / "episode_book_entities.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✅ Exported episode/book entities to {output_path}")
        return str(output_path)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Add episode and book entities to knowledge graph")
    parser.add_argument("--base-dir", type=str,
                       default="/Users/darrenzal/projects/yonearth-gaia-chatbot",
                       help="Base directory of the project")
    parser.add_argument("--visualization-file", type=str,
                       help="Path to existing visualization_data.json (optional)")
    parser.add_argument("--output", type=str,
                       help="Output file path (defaults to data/knowledge_graph/visualization_data.json)")
    parser.add_argument("--episodes-only", action="store_true",
                       help="Generate only episode/book data as separate file")

    args = parser.parse_args()

    adder = EpisodeBookEntityAdder(
        base_dir=args.base_dir,
        visualization_file=args.visualization_file
    )

    if args.episodes_only:
        output_path = adder.generate_episode_json(args.output)
    else:
        output_path = adder.export(args.output)

    print(f"\n✅ Export complete: {output_path}")


if __name__ == "__main__":
    main()
