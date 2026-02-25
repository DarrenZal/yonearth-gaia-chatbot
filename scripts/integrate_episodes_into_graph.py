#!/usr/bin/env python3
"""
Integrate Episodes and Books into Knowledge Graph

This script adds EPISODE and BOOK type nodes to the main graphrag_hierarchy.json
with computed 3D positions based on entity co-occurrence.

Approach:
1. Load existing graph data (entities, force_layout)
2. Load episode/book metadata
3. Build entity→episode co-occurrence from existing entity sources
4. Compute episode positions as centroids of related entity positions (+ jitter)
5. Create episode/book nodes with proper structure
6. Create relationships (EPISODE→PERSON, EPISODE→CONCEPT)
7. Output updated graphrag_hierarchy.json and force_layout.json

Usage:
    python scripts/integrate_episodes_into_graph.py [--dry-run] [--output-dir OUTPUT]
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EpisodeGraphIntegrator:
    """Integrate episodes and books into the knowledge graph."""

    # Type colors (hex as int for JS)
    TYPE_COLORS = {
        'EPISODE': 0xFFD700,  # Gold
        'BOOK': 0x8B4513      # Saddle Brown
    }

    def __init__(
        self,
        base_dir: str = "/Users/darrenzal/projects/yonearth-gaia-chatbot",
        server_data_dir: Optional[str] = None
    ):
        self.base_dir = Path(base_dir)

        # Server data paths (where actual graph data lives)
        if server_data_dir:
            self.server_data = Path(server_data_dir)
        else:
            self.server_data = Path("/var/www/symbiocenelabs/YonEarth/graph/data/graphrag_hierarchy")

        # Local data
        self.local_data = self.base_dir / "data"

        # Data containers
        self.graph_data: Optional[dict] = None
        self.force_layout: Dict[str, List[float]] = {}
        self.episode_data: Optional[dict] = None

        # Computed data
        self.entity_episodes: Dict[str, set] = defaultdict(set)  # entity -> episodes
        self.episode_entities: Dict[str, set] = defaultdict(set)  # episode -> entities

    def load_graph_data(self, local_file: Optional[str] = None) -> bool:
        """Load the main graph data."""
        # Try local file first, then server
        paths_to_try = []

        if local_file:
            paths_to_try.append(Path(local_file))

        paths_to_try.extend([
            self.local_data / "graphrag_hierarchy.json",
            self.server_data / "graphrag_hierarchy.json",
            self.server_data / "graphrag_hierarchy_v6_fixed.json",
        ])

        for path in paths_to_try:
            if path.exists():
                logger.info(f"Loading graph data from {path}")
                try:
                    with open(path, 'r') as f:
                        self.graph_data = json.load(f)
                    logger.info(f"Loaded {len(self.graph_data.get('entities', {}))} entities")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        logger.error("Could not load graph data from any source")
        return False

    def load_force_layout(self, local_file: Optional[str] = None) -> bool:
        """Load the force layout positions."""
        paths_to_try = []

        if local_file:
            paths_to_try.append(Path(local_file))

        paths_to_try.extend([
            self.local_data / "force_layout.json",
            self.server_data / "force_layout.json",
        ])

        for path in paths_to_try:
            if path.exists():
                logger.info(f"Loading force layout from {path}")
                try:
                    with open(path, 'r') as f:
                        self.force_layout = json.load(f)
                    logger.info(f"Loaded positions for {len(self.force_layout)} entities")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        logger.error("Could not load force layout from any source")
        return False

    def load_episode_book_data(self) -> bool:
        """Load the episode/book metadata."""
        path = self.local_data / "episode_book_entities.json"

        if not path.exists():
            logger.error(f"Episode data not found: {path}")
            return False

        logger.info(f"Loading episode/book data from {path}")
        with open(path, 'r') as f:
            self.episode_data = json.load(f)

        logger.info(f"Loaded {len(self.episode_data.get('episodes', {}))} episodes")
        logger.info(f"Loaded {len(self.episode_data.get('books', {}))} books")
        return True

    def build_entity_episode_mappings(self):
        """Build bidirectional mappings between entities and episodes from sources."""
        if not self.graph_data:
            logger.error("Graph data not loaded")
            return

        entities = self.graph_data.get('entities', {})

        for entity_name, entity_data in entities.items():
            sources = entity_data.get('sources', [])
            for source in sources:
                # Sources are like "episode_120" or "book_y-on-earth"
                if source.startswith('episode_') or source.startswith('book_'):
                    self.entity_episodes[entity_name].add(source)
                    self.episode_entities[source].add(entity_name)

        logger.info(f"Mapped {len(self.entity_episodes)} entities to sources")
        logger.info(f"Mapped {len(self.episode_entities)} sources to entities")

        # Log sample
        sample_sources = list(self.episode_entities.keys())[:3]
        for source in sample_sources:
            logger.info(f"  {source}: {len(self.episode_entities[source])} entities")

    def compute_episode_position(self, episode_id: str) -> List[float]:
        """Compute position for an episode/book as the CENTROID of its extracted entities.

        This is the simplest and most logical approach:
        - Episode position = average position of all entities extracted from that episode
        - Works for both UMAP (semantic) and Force Graph (structural) views
        - No scaling needed - the centroid naturally falls within the graph bounds
        """
        related_entities = self.episode_entities.get(episode_id, set())

        positions = []
        for entity_name in related_entities:
            if entity_name in self.force_layout:
                pos = self.force_layout[entity_name]
                if isinstance(pos, list) and len(pos) == 3:
                    positions.append(pos)

        if positions:
            # Simple centroid - average of all related entity positions
            arr = np.array(positions)
            centroid = arr.mean(axis=0)

            # Tiny jitter (±3) just to prevent exact overlap if two episodes have identical centroids
            jitter = np.random.uniform(-3, 3, 3)
            position = (centroid + jitter).tolist()

            logger.debug(f"{episode_id}: centroid from {len(positions)} entities")
            return position
        else:
            # No related entities found - place at origin
            logger.warning(f"No entity positions found for {episode_id}, placing at origin")
            return [0.0, 0.0, 0.0]

    def scale_episode_positions(self, positions: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """No scaling needed - centroids are already in the right place.

        Episode positions are computed as centroids of their extracted entities,
        so they naturally fall within the graph bounds. Just log stats for verification.
        """
        if not positions:
            return positions

        # Log stats for verification
        ep_ids = list(positions.keys())
        ep_arr = np.array([positions[k] for k in ep_ids])

        logger.info(f"Episode/Book positions (centroids) - "
                   f"X: [{ep_arr[:,0].min():.1f}, {ep_arr[:,0].max():.1f}], "
                   f"Y: [{ep_arr[:,1].min():.1f}, {ep_arr[:,1].max():.1f}], "
                   f"Z: [{ep_arr[:,2].min():.1f}, {ep_arr[:,2].max():.1f}]")

        # Return unchanged - centroids are already correct
        return positions

    def compute_book_position(self, book_id: str) -> List[float]:
        """Compute position for a book as centroid of related entities + jitter."""
        # Same logic as episodes
        return self.compute_episode_position(book_id)

    def create_episode_entity(self, episode_id: str, episode_data: dict) -> dict:
        """Create an entity entry for an episode."""
        # Get related entities for sources
        related = list(self.episode_entities.get(episode_id, set()))[:20]  # Limit for performance

        return {
            "name": episode_data["name"],
            "type": "EPISODE",
            "description": episode_data.get("description", ""),
            "aliases": episode_data.get("aliases", []),
            "is_fictional": False,
            "resolution_confidence": 1.0,
            "resolution_method": "generated",
            "sources": [episode_id],  # Self-reference
            "provenance": {
                "extraction": "episode_integration",
                "generated_at": "2026-02-02"
            },
            "mention_count": len(related),
            # Episode-specific fields
            "episode_number": episode_data.get("episode_number"),
            "guest_name": episode_data.get("guest_name"),
            "audio_url": episode_data.get("audio_url", ""),
            "url": episode_data.get("url", ""),
            "publish_date": episode_data.get("publish_date", ""),
            # Related entities (for UI display)
            "related_entities": related[:10]
        }

    def create_book_entity(self, book_id: str, book_data: dict) -> dict:
        """Create an entity entry for a book."""
        related = list(self.episode_entities.get(book_id, set()))[:20]

        return {
            "name": book_data["name"],
            "type": "BOOK",
            "description": book_data.get("description", ""),
            "aliases": book_data.get("aliases", []),
            "is_fictional": False,
            "resolution_confidence": 1.0,
            "resolution_method": "generated",
            "sources": [book_id],
            "provenance": {
                "extraction": "episode_integration",
                "generated_at": "2026-02-02"
            },
            "mention_count": len(related),
            # Book-specific fields
            "book_id": book_data.get("book_id"),
            "author": book_data.get("author", ""),
            "audiobook_url": book_data.get("audiobook_url", ""),
            "ebook_url": book_data.get("ebook_url", ""),
            "print_url": book_data.get("print_url", ""),
            # Related entities
            "related_entities": related[:10]
        }

    def create_relationships(self) -> List[dict]:
        """Create relationships between episodes/books and their entities."""
        relationships = []

        # Episode/Book → Person (FEATURES)
        # Episode/Book → Concept (DISCUSSES)
        # Episode/Book → Organization (MENTIONS)

        for source_id, entities in self.episode_entities.items():
            for entity_name in entities:
                entity_data = self.graph_data['entities'].get(entity_name, {})
                entity_type = entity_data.get('type', 'CONCEPT')

                if entity_type == 'PERSON':
                    rel_type = 'FEATURES'
                elif entity_type == 'CONCEPT':
                    rel_type = 'DISCUSSES'
                elif entity_type == 'ORGANIZATION':
                    rel_type = 'MENTIONS'
                else:
                    rel_type = 'CONTAINS'

                relationships.append({
                    "source": source_id,
                    "target": entity_name,
                    "type": rel_type,
                    "weight": 1.0,
                    "provenance": "episode_integration"
                })

        logger.info(f"Created {len(relationships)} episode/book → entity relationships")
        return relationships

    def integrate(self, dry_run: bool = False) -> Tuple[dict, dict]:
        """Main integration method."""

        # Load all data (skip if already loaded)
        if not self.graph_data:
            if not self.load_graph_data():
                raise RuntimeError("Failed to load graph data")
        if not self.force_layout:
            if not self.load_force_layout():
                raise RuntimeError("Failed to load force layout")
        if not self.load_episode_book_data():
            raise RuntimeError("Failed to load episode/book data")

        # Build mappings
        self.build_entity_episode_mappings()

        # Create new entities and positions
        new_entities = {}
        new_positions = {}

        # Process episodes
        episodes = self.episode_data.get('episodes', {})
        for episode_id, episode_data in episodes.items():
            new_entities[episode_id] = self.create_episode_entity(episode_id, episode_data)
            new_positions[episode_id] = self.compute_episode_position(episode_id)

        logger.info(f"Created {len([e for e in episodes])} episode entities")

        # Process books
        books = self.episode_data.get('books', {})
        for book_id, book_data in books.items():
            new_entities[book_id] = self.create_book_entity(book_id, book_data)
            new_positions[book_id] = self.compute_book_position(book_id)

        logger.info(f"Created {len([b for b in books])} book entities")

        # Scale episode/book positions to spread them out properly
        new_positions = self.scale_episode_positions(new_positions)

        # Create relationships
        new_relationships = self.create_relationships()

        # Merge into graph data
        updated_graph = self.graph_data.copy()

        # Add new entities (keeping existing ones)
        for entity_id, entity_data in new_entities.items():
            if entity_id not in updated_graph['entities']:
                updated_graph['entities'][entity_id] = entity_data
            else:
                logger.warning(f"Entity {entity_id} already exists, skipping")

        # Add episode/book entities to clusters.level_0 so visualization renders them
        if 'clusters' not in updated_graph:
            updated_graph['clusters'] = {}
        if 'level_0' not in updated_graph['clusters']:
            updated_graph['clusters']['level_0'] = {}

        for entity_id, entity_data in new_entities.items():
            position = new_positions.get(entity_id, [0, 0, 0])
            updated_graph['clusters']['level_0'][entity_id] = {
                'id': entity_id,
                'entity_id': entity_id,
                'entity': entity_data,
                'umap_position': position,
                'position': position,
                'betweenness': 0.01,  # Low centrality for media nodes
                'type': entity_data.get('type', 'EPISODE')
            }

        logger.info(f"Added {len(new_entities)} entities to clusters.level_0")

        # Add new relationships
        existing_rels = updated_graph.get('relationships', [])
        updated_graph['relationships'] = existing_rels + new_relationships

        # Update metadata
        if 'metadata' not in updated_graph:
            updated_graph['metadata'] = {}
        updated_graph['metadata']['episode_integration'] = {
            'date': '2026-02-02',
            'episodes_added': len(episodes),
            'books_added': len(books),
            'relationships_added': len(new_relationships)
        }

        # Merge positions
        updated_layout = self.force_layout.copy()
        updated_layout.update(new_positions)

        logger.info(f"Integration complete:")
        logger.info(f"  Total entities: {len(updated_graph['entities'])}")
        logger.info(f"  Total relationships: {len(updated_graph['relationships'])}")
        logger.info(f"  Total positions: {len(updated_layout)}")

        return updated_graph, updated_layout

    def save_outputs(
        self,
        updated_graph: dict,
        updated_layout: dict,
        output_dir: str
    ):
        """Save the updated graph and layout files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save updated graph
        graph_file = output_path / "graphrag_hierarchy_with_episodes.json"
        with open(graph_file, 'w') as f:
            json.dump(updated_graph, f)
        logger.info(f"Saved updated graph to {graph_file}")

        # Save updated layout
        layout_file = output_path / "force_layout_with_episodes.json"
        with open(layout_file, 'w') as f:
            json.dump(updated_layout, f)
        logger.info(f"Saved updated layout to {layout_file}")

        # Generate JS patch for typeColors
        js_patch = self.generate_js_patch()
        patch_file = output_path / "typeColors_patch.js"
        with open(patch_file, 'w') as f:
            f.write(js_patch)
        logger.info(f"Saved JS patch to {patch_file}")

        return graph_file, layout_file, patch_file

    def generate_js_patch(self) -> str:
        """Generate JS code to add EPISODE and BOOK type colors."""
        return """// Add these lines to the typeColors object in GraphRAG3D_EmbeddingView.js
// After line ~145, add:

'EPISODE': 0xFFD700,    // Gold
'BOOK': 0x8B4513,       // Saddle Brown

// The full typeColors should look like:
/*
this.typeColors = {
    'PERSON': 0x4CAF50,
    'ORGANIZATION': 0x2196F3,
    'CONCEPT': 0x9C27B0,
    'PRACTICE': 0xFF9800,
    'PRODUCT': 0xF44336,
    'PLACE': 0x00BCD4,
    'EVENT': 0xFFEB3B,
    'WORK': 0x795548,
    'CLAIM': 0xE91E63,
    'EPISODE': 0xFFD700,    // Gold - NEW
    'BOOK': 0x8B4513        // Saddle Brown - NEW
};
*/
"""


def main():
    parser = argparse.ArgumentParser(description="Integrate episodes into knowledge graph")
    parser.add_argument("--base-dir", type=str,
                       default="/Users/darrenzal/projects/yonearth-gaia-chatbot",
                       help="Base directory of the project")
    parser.add_argument("--graph-file", type=str,
                       help="Path to graphrag_hierarchy.json (optional)")
    parser.add_argument("--layout-file", type=str,
                       help="Path to force_layout.json (optional)")
    parser.add_argument("--output-dir", type=str,
                       default="/Users/darrenzal/projects/yonearth-gaia-chatbot/data/integrated",
                       help="Output directory for updated files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't save files, just show what would be done")

    args = parser.parse_args()

    integrator = EpisodeGraphIntegrator(base_dir=args.base_dir)

    # If specific files provided, load from those
    if args.graph_file:
        integrator.load_graph_data(args.graph_file)
    if args.layout_file:
        integrator.load_force_layout(args.layout_file)

    updated_graph, updated_layout = integrator.integrate(dry_run=args.dry_run)

    if not args.dry_run:
        integrator.save_outputs(updated_graph, updated_layout, args.output_dir)
        print(f"\n✅ Integration complete. Files saved to {args.output_dir}")
        print("\nNext steps:")
        print("1. Review the generated files")
        print("2. Copy graphrag_hierarchy_with_episodes.json to server")
        print("3. Copy force_layout_with_episodes.json to server")
        print("4. Apply typeColors_patch.js to GraphRAG3D_EmbeddingView.js")
        print("5. Clear browser cache and test")
    else:
        print("\n✅ Dry run complete. No files saved.")
        print(f"Would create {len(updated_graph['entities'])} entities")
        print(f"Would create {len(updated_layout)} positions")


if __name__ == "__main__":
    main()
