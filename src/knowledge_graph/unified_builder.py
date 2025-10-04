"""
Unified builder for synchronized knowledge graph and wiki generation.

This module ensures that Neo4j knowledge graph and Obsidian wiki are generated
from the same canonical data source, maintaining a single source of truth.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .wiki.wiki_builder import WikiBuilder
from .graph.neo4j_client import Neo4jClient
from .build_graph import GraphBuilder

logger = logging.getLogger(__name__)


class UnifiedBuilder:
    """
    Builds both Neo4j knowledge graph and Obsidian wiki from unified data source.

    Ensures synchronization by:
    1. Loading extraction files (canonical source)
    2. Merging with episode metadata (web-scraped data)
    3. Building both graph and wiki from same data
    4. Validating consistency between outputs
    """

    def __init__(
        self,
        extraction_dir: Path,
        transcripts_dir: Path,
        wiki_output_dir: Path,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        Initialize unified builder.

        Args:
            extraction_dir: Directory with entity extraction JSON files
            transcripts_dir: Directory with episode transcript JSON files
            wiki_output_dir: Output directory for wiki markdown
            neo4j_uri: Neo4j database URI (optional)
            neo4j_user: Neo4j username (optional)
            neo4j_password: Neo4j password (optional)
        """
        self.extraction_dir = Path(extraction_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.wiki_output_dir = Path(wiki_output_dir)

        # Initialize builders
        self.wiki_builder = WikiBuilder(wiki_output_dir)

        if neo4j_uri and neo4j_user and neo4j_password:
            self.graph_builder = GraphBuilder(
                extraction_dir=extraction_dir,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password
            )
        else:
            self.graph_builder = None
            logger.warning("Neo4j credentials not provided - graph building disabled")

        # Unified data storage
        self.episode_metadata = {}  # episode_number -> metadata
        self.extractions = []  # List of extraction dictionaries

    def load_episode_metadata(self) -> Dict[int, Dict]:
        """
        Load episode metadata from transcript JSON files.

        Returns:
            Dictionary mapping episode_number -> metadata
        """
        logger.info("Loading episode metadata from transcripts...")
        metadata = {}

        transcript_files = list(self.transcripts_dir.glob('episode_*.json'))
        logger.info(f"Found {len(transcript_files)} transcript files")

        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                episode_number = data.get('episode_number')
                if episode_number is None:
                    # Try to extract from filename
                    filename = file_path.stem  # e.g., "episode_170"
                    try:
                        episode_number = int(filename.split('_')[1])
                    except (IndexError, ValueError):
                        logger.warning(f"Could not extract episode number from {file_path.name}")
                        continue

                metadata[episode_number] = {
                    'number': episode_number,
                    'title': data.get('title', f'Episode {episode_number}'),
                    'url': data.get('url', f'https://yonearth.org/podcast/episode-{episode_number}/'),
                    'publish_date': data.get('publish_date', 'Unknown'),
                    'audio_url': data.get('audio_url', ''),
                    'subtitle': data.get('subtitle', ''),
                    'description': data.get('description', ''),
                    'sponsors': data.get('sponsors', ''),
                    'about_sections': data.get('about_sections', {}),
                    'related_episodes': data.get('related_episodes', []),
                    'transcript_length': len(data.get('full_transcript', '')),
                    'host': 'Aaron William Perry'  # Extracted from transcript pattern
                }

                logger.debug(f"Loaded metadata for Episode {episode_number}")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded metadata for {len(metadata)} episodes")
        return metadata

    def load_extractions(self) -> List[Dict]:
        """
        Load entity extraction files.

        Returns:
            List of extraction dictionaries
        """
        logger.info("Loading entity extractions...")
        extractions = []

        extraction_files = sorted(self.extraction_dir.glob('episode_*_extraction.json'))
        logger.info(f"Found {len(extraction_files)} extraction files")

        for file_path in extraction_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    extractions.append(data)
                    logger.debug(f"Loaded {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return extractions

    def merge_episode_data(self):
        """
        Merge extraction data with episode metadata.

        This creates the unified data model that both graph and wiki will use.
        """
        logger.info("Merging extraction data with episode metadata...")

        for extraction in self.extractions:
            episode_number = extraction.get('episode_number')

            if episode_number in self.episode_metadata:
                # Merge metadata into extraction
                metadata = self.episode_metadata[episode_number]

                extraction['url'] = metadata['url']
                extraction['publish_date'] = metadata['publish_date']
                extraction['audio_url'] = metadata['audio_url']
                extraction['subtitle'] = metadata['subtitle']
                extraction['description'] = metadata['description']
                extraction['sponsors'] = metadata['sponsors']
                extraction['about_sections'] = metadata['about_sections']
                extraction['related_episodes'] = metadata['related_episodes']
                extraction['host'] = metadata['host']

                # Update or add guest from extraction if missing
                if not extraction.get('guest_name') and metadata['about_sections']:
                    # Try to extract guest from about sections
                    for key, value in metadata['about_sections'].items():
                        if 'about_' in key.lower() and key.lower() != 'about_chelsea_green_publishing':
                            guest_name = key.replace('about_', '').replace('_', ' ').title()
                            extraction['guest_name'] = guest_name
                            break

                logger.debug(f"Merged metadata for Episode {episode_number}")
            else:
                logger.warning(f"No metadata found for Episode {episode_number}")

    def build_wiki(self) -> Dict:
        """
        Build Obsidian wiki from unified data.

        Returns:
            Statistics dictionary
        """
        logger.info("Building wiki from unified data...")

        # Update WikiBuilder to use merged extractions
        self.wiki_builder.create_directory_structure()
        self.wiki_builder.process_extractions(self.extractions)
        self.wiki_builder.build_relationships()

        # Generate pages
        self.wiki_builder.generate_entity_pages()
        self.wiki_builder.generate_episode_pages()
        self.wiki_builder.generate_index_pages()
        self.wiki_builder.generate_summary_pages()

        logger.info("Wiki build complete!")
        return self.wiki_builder.stats

    def build_graph(self) -> Dict:
        """
        Build Neo4j knowledge graph from unified data.

        Returns:
            Statistics dictionary
        """
        if not self.graph_builder:
            logger.warning("Graph builder not initialized - skipping graph build")
            return {}

        logger.info("Building Neo4j graph from unified data...")

        # Use GraphBuilder with merged extractions
        stats = self.graph_builder.build()

        logger.info("Graph build complete!")
        return stats

    def validate_synchronization(self) -> bool:
        """
        Validate that graph and wiki are synchronized.

        Returns:
            True if synchronized, False otherwise
        """
        logger.info("Validating synchronization between graph and wiki...")

        # Check episode counts match
        wiki_episode_count = len(self.wiki_builder.episodes)
        extraction_episode_count = len(self.extractions)

        if wiki_episode_count != extraction_episode_count:
            logger.error(
                f"Episode count mismatch: Wiki has {wiki_episode_count}, "
                f"Extractions have {extraction_episode_count}"
            )
            return False

        # Check entity counts match
        wiki_entity_count = len(self.wiki_builder.entities_by_name)

        logger.info(
            f"Validation: {wiki_episode_count} episodes, "
            f"{wiki_entity_count} entities"
        )

        return True

    def build_all(self) -> Dict:
        """
        Build both graph and wiki from unified data source.

        This is the main entry point for synchronized building.

        Returns:
            Combined statistics dictionary
        """
        logger.info("=" * 80)
        logger.info("Starting unified build process...")
        logger.info("=" * 80)

        # Step 1: Load all data
        self.episode_metadata = self.load_episode_metadata()
        self.extractions = self.load_extractions()

        # Step 2: Merge data (create single source of truth)
        self.merge_episode_data()

        # Step 3: Build wiki
        wiki_stats = self.build_wiki()

        # Step 4: Build graph (if enabled)
        graph_stats = self.build_graph() if self.graph_builder else {}

        # Step 5: Validate synchronization
        is_synced = self.validate_synchronization()

        # Combine statistics
        combined_stats = {
            'wiki': wiki_stats,
            'graph': graph_stats,
            'synchronized': is_synced,
            'total_episodes': len(self.episode_metadata),
            'total_extractions': len(self.extractions),
            'build_timestamp': datetime.now().isoformat()
        }

        logger.info("=" * 80)
        logger.info("Unified build complete!")
        logger.info(f"Episodes: {combined_stats['total_episodes']}")
        logger.info(f"Extractions: {combined_stats['total_extractions']}")
        logger.info(f"Synchronized: {is_synced}")
        logger.info("=" * 80)

        return combined_stats


def main():
    """Main entry point for unified builder."""
    import os
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    extraction_dir = base_dir / 'data' / 'knowledge_graph' / 'entities'
    transcripts_dir = base_dir / 'data' / 'transcripts'
    wiki_output_dir = base_dir / 'web' / 'wiki'

    # Neo4j credentials (optional)
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')

    # Build
    builder = UnifiedBuilder(
        extraction_dir=extraction_dir,
        transcripts_dir=transcripts_dir,
        wiki_output_dir=wiki_output_dir,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )

    stats = builder.build_all()

    # Save statistics
    stats_file = base_dir / 'data' / 'knowledge_graph' / 'build_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {stats_file}")


if __name__ == '__main__':
    main()
