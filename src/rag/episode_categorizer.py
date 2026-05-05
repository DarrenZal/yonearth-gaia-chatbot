"""
Episode Categorizer Module for RAG Search Enhancement
Loads and parses episode categorization data from CSV to enhance search relevance
"""
import json
import logging
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class EpisodeCategory:
    """Represents an episode with its categorization data"""
    episode_id: int
    guest_name: str
    guest_title: str
    location: str
    categories: Set[str]

    def __post_init__(self):
        """Ensure categories is a set"""
        if not isinstance(self.categories, set):
            self.categories = set(self.categories) if self.categories else set()


@dataclass
class BookCategory:
    """Represents a YOE book with its taxonomy categories.

    Books were originally invisible to category search (Aaron's #3 issue,
    2026-05-02). Categories come from `yoe_categories` in
    data/books/<slug>/metadata.json — manually curated so book chapters
    surface for theme queries.
    """
    book_title: str
    author: str
    categories: Set[str]


class EpisodeCategorizer:
    """
    Loads and manages episode categorization data for enhanced search scoring.
    Also loads book-level categories so theme-based retrieval can pull both
    podcasts and books from the YOE library.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        books_dir: Optional[str] = None,
    ):
        """
        Initialize the episode categorizer

        Args:
            csv_path: Path to the episode CSV. If None, uses default location.
            books_dir: Path to the books directory (each book has a metadata.json
                with optional `yoe_categories`). If None, uses data/books.
        """
        self.csv_path = csv_path or str(settings.data_dir / "PodcastPipelineTracking.csv")
        self.books_dir = Path(books_dir) if books_dir else (settings.data_dir / "books")
        self.episodes: Dict[int, EpisodeCategory] = {}
        self.books: Dict[str, BookCategory] = {}  # keyed by book_title
        self.categories: Set[str] = set()
        self.category_synonyms: Dict[str, Set[str]] = {}
        self._load_episodes()
        self._load_books()
        self._build_category_synonyms()
    
    def _load_episodes(self):
        """Load episodes from CSV file"""
        try:
            logger.info(f"Loading episodes from {self.csv_path}")
            
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) < 3:
                logger.warning("CSV file has insufficient rows")
                return
            
            # Parse header row (row 2, 0-indexed row 1)
            header = rows[1]
            
            # Find category columns (starting from column 6 based on CSV analysis)
            category_start_idx = 6  # Categories start after basic info columns
            category_columns = []
            
            for i, col in enumerate(header[category_start_idx:], category_start_idx):
                if col and col.strip():
                    category_columns.append((i, col.strip()))
                    self.categories.add(col.strip())
            
            logger.info(f"Found {len(category_columns)} categories: {[cat for _, cat in category_columns]}")
            
            # Parse episode rows (starting from row 3, 0-indexed row 2)
            episodes_parsed = 0
            for row_idx in range(2, len(rows)):
                row = rows[row_idx]
                
                if not row or len(row) < 5:
                    continue
                
                try:
                    # Parse episode ID from first column
                    episode_id_str = row[0].strip()
                    if not episode_id_str:
                        continue
                    
                    episode_id = int(episode_id_str)
                    
                    # Parse basic info
                    guest_name = row[1].strip() if len(row) > 1 else ""
                    guest_title = row[2].strip() if len(row) > 2 else ""
                    location = row[3].strip() if len(row) > 3 else ""
                    
                    # Parse categories
                    episode_categories = set()
                    for col_idx, category_name in category_columns:
                        if col_idx < len(row) and row[col_idx].strip().upper() == 'X':
                            episode_categories.add(category_name)
                    
                    # Create episode object
                    episode = EpisodeCategory(
                        episode_id=episode_id,
                        guest_name=guest_name,
                        guest_title=guest_title,
                        location=location,
                        categories=episode_categories
                    )
                    
                    self.episodes[episode_id] = episode
                    episodes_parsed += 1
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing row {row_idx}: {e}")
                    continue
            
            logger.info(f"Loaded {episodes_parsed} episodes with categorization data")
            
        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading episodes: {e}")
    
    def _load_books(self):
        """Load book-level categories from data/books/<slug>/metadata.json."""
        if not self.books_dir.exists():
            logger.info(f"No books dir at {self.books_dir} — skipping book categorization")
            return
        for book_folder in self.books_dir.iterdir():
            if not book_folder.is_dir():
                continue
            meta_path = book_folder / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with meta_path.open() as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {meta_path}: {e}")
                continue
            title = meta.get("title")
            if not title:
                continue
            cats = set(meta.get("yoe_categories") or [])
            if not cats:
                # No categories declared — book is reachable via direct search
                # but won't appear in theme-saturation. That's fine.
                continue
            self.books[title] = BookCategory(
                book_title=title,
                author=meta.get("author", "Unknown"),
                categories=cats,
            )
            self.categories.update(cats)
        if self.books:
            logger.info(
                f"Loaded {len(self.books)} books with category metadata: "
                f"{sorted(self.books.keys())}"
            )

    def get_books_by_category(self, category: str) -> List[str]:
        """Return the list of book titles tagged with `category`."""
        return [
            title
            for title, book in self.books.items()
            if category in book.categories
        ]

    def _build_category_synonyms(self):
        """Build category synonyms for better matching"""
        # Define synonym mappings for better query matching
        # Keys MUST match the canonical labels in data/yoe_taxonomy.json verbatim
        # (including misspellings POLICY & GOVERNMT, PERMA-CULTURE, SUSTAIN-ABILITY,
        # which are Aaron's spreadsheet conventions).
        synonyms = {
            'HERBAL MEDICINE': {'herbs', 'herbalist', 'herbal', 'medicine', 'plant medicine', 'botanicals'},
            'FARMING & FOOD': {'farming', 'agriculture', 'food', 'crops', 'grow', 'cultivation'},
            'BIOCHAR': {'biochar', 'charcoal', 'carbon sequestration', 'soil carbon'},
            'HEALTH & WELLNESS': {'health', 'wellness', 'healing', 'medicine', 'therapeutic'},
            'CLIMATE & SCIENCE': {'climate', 'science', 'global warming', 'carbon', 'research'},
            'PERMA-CULTURE': {'permaculture', 'sustainable design', 'ecological design'},
            'SOIL': {'soil', 'earth', 'ground', 'dirt', 'soil health', 'compost', 'composting'},
            'ECOLOGY & NATURE': {'ecology', 'nature', 'environment', 'natural', 'ecological', 'water', 'watershed'},
            'BIO-DYNAMICS': {'biodynamic', 'biodynamics', 'rudolf steiner'},
            'REGEN / SOCIAL ENTERPRISE': {'regenerative', 'regeneration', 'restoration', 'social enterprise', 'benefit corporation', 'triple bottom line', 'mission-driven'},
            'INDIGENOUS WISDOM': {'indigenous', 'traditional', 'ancestral', 'native'},
            'BUSINESS': {'business', 'entrepreneurship', 'enterprise', 'company'},
            'POLICY & GOVERNMT': {'policy', 'government', 'politics', 'governance'},
            'EDUCATION': {'education', 'teaching', 'learning', 'school'},
            'TECHNOLOGY & MATERIALS': {'technology', 'materials', 'innovation', 'tech', 'renewable energy', 'solar', 'wind'},
            'SUSTAIN-ABILITY': {'sustainability', 'sustainable', 'green', 'eco'},
            'COMMUNITY': {'community', 'social', 'collective', 'group'},
            'IMPACT INVESTING': {'investing', 'investment', 'finance', 'capital'},
            'GREEN BUILDING': {'building', 'construction', 'architecture', 'green building'},
            'MEDIA BOOKS & CONTENT': {'media', 'book', 'content', 'publishing', 'author'},
            'GREEN FAITH': {'faith', 'religion', 'spiritual', 'sacred'},
            'ESOTERICA': {'esoteric', 'mystical', 'spiritual', 'metaphysical'}
        }
        
        # Build reverse lookup
        for category, terms in synonyms.items():
            self.category_synonyms[category] = terms
    
    def get_episode_categories(self, episode_id: int) -> Set[str]:
        """Get categories for a specific episode"""
        episode = self.episodes.get(episode_id)
        return episode.categories if episode else set()
    
    def get_episodes_by_category(self, category: str) -> List[int]:
        """Get all episode IDs that belong to a specific category"""
        matching_episodes = []
        for episode_id, episode in self.episodes.items():
            if category in episode.categories:
                matching_episodes.append(episode_id)
        return matching_episodes
    
    def analyze_query_categories(self, query: str) -> Dict[str, float]:
        """
        Analyze query to determine which categories it matches
        Returns dict of category -> relevance score
        """
        query_lower = query.lower()
        category_scores = {}
        
        # Check each category and its synonyms
        for category, synonyms in self.category_synonyms.items():
            score = 0.0
            
            # Check exact category name match
            if category.lower().replace(' & ', ' ').replace('&', '').replace(' ', '') in query_lower.replace(' ', ''):
                score += 1.0
            
            # Check synonym matches
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    score += 0.8
            
            # Check partial matches
            category_words = category.lower().split()
            for word in category_words:
                if len(word) > 3 and word in query_lower:
                    score += 0.5
            
            if score > 0:
                category_scores[category] = score
        
        return category_scores
    
    def score_episode_for_query(self, episode_id: int, query: str) -> float:
        """
        Score an episode's relevance to a query based on categorization
        Returns score between 0.0 and 1.0
        """
        episode = self.episodes.get(episode_id)
        if not episode:
            return 0.0
        
        # Analyze query categories
        query_categories = self.analyze_query_categories(query)
        
        if not query_categories:
            return 0.0
        
        # Calculate score based on category matches
        total_score = 0.0
        max_possible_score = 0.0
        
        for category, query_score in query_categories.items():
            max_possible_score += query_score
            if category in episode.categories:
                total_score += query_score
        
        # Normalize score
        if max_possible_score > 0:
            return total_score / max_possible_score
        
        return 0.0
    
    def get_top_episodes_for_query(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """
        Get top K episodes for a query based on category matching
        Returns list of (episode_id, score) tuples
        """
        episode_scores = []
        
        for episode_id in self.episodes.keys():
            score = self.score_episode_for_query(episode_id, query)
            if score > 0:
                episode_scores.append((episode_id, score))
        
        # Sort by score descending
        episode_scores.sort(key=lambda x: x[1], reverse=True)
        
        return episode_scores[:k]
    
    def get_available_categories(self) -> List[str]:
        """Get list of all available categories"""
        return sorted(list(self.categories))
    
    def get_episode_info(self, episode_id: int) -> Optional[EpisodeCategory]:
        """Get full episode information including categories"""
        return self.episodes.get(episode_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get categorizer statistics"""
        category_counts = {}
        for episode in self.episodes.values():
            for category in episode.categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_episodes': len(self.episodes),
            'total_categories': len(self.categories),
            'category_counts': category_counts,
            'episodes_per_category': {k: v for k, v in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)}
        }


def main():
    """Test episode categorizer functionality"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create categorizer
    categorizer = EpisodeCategorizer()
    
    # Test queries
    test_queries = [
        "herbal medicine",
        "biochar",
        "farming and agriculture",
        "health and wellness",
        "climate science"
    ]
    
    print("=== Episode Categorizer Test ===\n")
    
    # Show stats
    stats = categorizer.get_stats()
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Total Categories: {stats['total_categories']}")
    print(f"Top Categories by Episode Count:")
    for category, count in list(stats['episodes_per_category'].items())[:10]:
        print(f"  {category}: {count} episodes")
    
    print("\n" + "="*60)
    
    # Test queries
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        print("-" * 40)
        
        # Get top episodes
        top_episodes = categorizer.get_top_episodes_for_query(query, k=5)
        
        print(f"Top {len(top_episodes)} episodes for '{query}':")
        for episode_id, score in top_episodes:
            episode_info = categorizer.get_episode_info(episode_id)
            if episode_info:
                print(f"  Episode {episode_id}: {episode_info.guest_name} - {episode_info.guest_title}")
                print(f"    Score: {score:.3f}")
                print(f"    Categories: {', '.join(sorted(episode_info.categories))}")
                print()


if __name__ == "__main__":
    main()