"""
Keyword frequency indexer for building searchable word indexes of episodes
"""
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from ..config import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class KeywordIndexer:
    """Build and maintain keyword indexes for episode transcripts"""
    
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Episode indexes
        self.episode_word_frequencies = {}  # episode_id -> {word: frequency}
        self.word_episode_mapping = defaultdict(set)  # word -> {episode_ids}
        self.episode_metadata = {}  # episode_id -> metadata
        
        # Cache paths
        self.cache_dir = Path(settings.project_root) / "yonearth-chatbot" / "data" / "indexes"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_cache_path = self.cache_dir / "keyword_index.pkl"
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and extract keywords"""
        if not text:
            return []
            
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove URLs, email addresses, and other noise
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)  # Keep only words and spaces
        
        # Tokenize
        words = word_tokenize(text)
        
        # Filter out very short words, numbers, and stopwords
        filtered_words = []
        for word in words:
            if (len(word) >= 3 and  # At least 3 characters
                not word.isdigit() and  # Not just numbers
                word.isalpha() and  # Only alphabetic characters
                (not self.remove_stopwords or word not in self.stop_words)):
                
                # Apply stemming if enabled
                final_word = self.stemmer.stem(word) if self.stemmer else word
                filtered_words.append(final_word)
                
        return filtered_words
    
    def _load_episode_data(self, episode_path: Path) -> Optional[Dict[str, Any]]:
        """Load episode data from JSON file"""
        try:
            with open(episode_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate required fields
            if not data.get('full_transcript') or data['full_transcript'] == 'NO_TRANSCRIPT_AVAILABLE':
                logger.warning(f"No transcript available for {episode_path.name}")
                return None
                
            return data
        except Exception as e:
            logger.error(f"Error loading episode {episode_path}: {e}")
            return None
    
    def index_episode(self, episode_data: Dict[str, Any], episode_id: str) -> Dict[str, int]:
        """Index a single episode and return word frequencies"""
        transcript = episode_data.get('full_transcript', '')
        if not transcript or transcript == 'NO_TRANSCRIPT_AVAILABLE':
            return {}
            
        # Extract and preprocess words
        words = self._preprocess_text(transcript)
        
        # Count word frequencies
        word_frequencies = Counter(words)
        
        # Store in indexes
        self.episode_word_frequencies[episode_id] = dict(word_frequencies)
        
        # Update reverse mapping
        for word in word_frequencies.keys():
            self.word_episode_mapping[word].add(episode_id)
            
        # Store metadata
        self.episode_metadata[episode_id] = {
            'title': episode_data.get('title', ''),
            'episode_number': episode_data.get('episode_number', ''),
            'url': episode_data.get('url', ''),
            'publish_date': episode_data.get('publish_date', ''),
            'total_words': sum(word_frequencies.values()),
            'unique_words': len(word_frequencies)
        }
        
        logger.info(f"Indexed episode {episode_id}: {len(word_frequencies)} unique words, {sum(word_frequencies.values())} total words")
        return dict(word_frequencies)
    
    def build_index_from_episodes_dir(self, episodes_dir: Optional[Path] = None) -> None:
        """Build index from all episodes in directory"""
        episodes_dir = episodes_dir or settings.episodes_dir
        
        logger.info(f"Building keyword index from {episodes_dir}")
        
        # Clear existing indexes
        self.episode_word_frequencies.clear()
        self.word_episode_mapping.clear()
        self.episode_metadata.clear()
        
        # Process all episode files
        episode_files = list(episodes_dir.glob("episode_*.json"))
        successful_episodes = 0
        
        for episode_file in episode_files:
            episode_data = self._load_episode_data(episode_file)
            if episode_data:
                episode_id = episode_file.stem  # filename without extension
                self.index_episode(episode_data, episode_id)
                successful_episodes += 1
                
        logger.info(f"Successfully indexed {successful_episodes}/{len(episode_files)} episodes")
        logger.info(f"Total unique words in index: {len(self.word_episode_mapping)}")
        
        # Save index to cache
        self.save_index()
    
    def search_by_keywords(
        self, 
        keywords: List[str], 
        top_k: int = 10,
        min_frequency: int = 1
    ) -> List[Dict[str, Any]]:
        """Search episodes by keywords and return ranked results"""
        if not keywords:
            return []
            
        # Preprocess keywords
        processed_keywords = []
        for keyword in keywords:
            processed = self._preprocess_text(keyword)
            processed_keywords.extend(processed)
            
        if not processed_keywords:
            return []
            
        # Score episodes based on keyword frequencies
        episode_scores = defaultdict(float)
        
        for keyword in processed_keywords:
            if keyword in self.word_episode_mapping:
                for episode_id in self.word_episode_mapping[keyword]:
                    frequency = self.episode_word_frequencies[episode_id].get(keyword, 0)
                    if frequency >= min_frequency:
                        # Score based on frequency and total words (TF normalization)
                        total_words = max(self.episode_metadata[episode_id]['total_words'], 1)
                        tf_score = frequency / total_words
                        episode_scores[episode_id] += tf_score
                        
        # Sort episodes by score
        ranked_episodes = sorted(
            episode_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Build result objects
        results = []
        for episode_id, score in ranked_episodes:
            metadata = self.episode_metadata[episode_id]
            
            # Calculate keyword frequencies for this episode
            keyword_details = {}
            for keyword in processed_keywords:
                if keyword in self.episode_word_frequencies[episode_id]:
                    keyword_details[keyword] = self.episode_word_frequencies[episode_id][keyword]
                    
            result = {
                'episode_id': episode_id,
                'title': metadata['title'],
                'episode_number': metadata['episode_number'],
                'url': metadata['url'],
                'score': score,
                'keyword_frequencies': keyword_details,
                'total_words': metadata['total_words'],
                'matching_keywords': list(keyword_details.keys())
            }
            results.append(result)
            
        logger.info(f"Found {len(results)} episodes matching keywords: {keywords}")
        return results
    
    def get_episode_keywords(
        self, 
        episode_id: str, 
        top_k: int = 20
    ) -> List[Tuple[str, int]]:
        """Get top keywords for a specific episode"""
        if episode_id not in self.episode_word_frequencies:
            return []
            
        frequencies = self.episode_word_frequencies[episode_id]
        top_words = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return top_words
    
    def save_index(self) -> None:
        """Save index to cache file"""
        try:
            index_data = {
                'episode_word_frequencies': self.episode_word_frequencies,
                'word_episode_mapping': dict(self.word_episode_mapping),  # Convert defaultdict
                'episode_metadata': self.episode_metadata,
                'config': {
                    'use_stemming': self.use_stemming,
                    'remove_stopwords': self.remove_stopwords
                }
            }
            
            with open(self.index_cache_path, 'wb') as f:
                pickle.dump(index_data, f)
                
            logger.info(f"Saved keyword index to {self.index_cache_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self) -> bool:
        """Load index from cache file"""
        try:
            if not self.index_cache_path.exists():
                logger.info("No cached index found")
                return False
                
            with open(self.index_cache_path, 'rb') as f:
                index_data = pickle.load(f)
                
            self.episode_word_frequencies = index_data['episode_word_frequencies']
            self.word_episode_mapping = defaultdict(set, index_data['word_episode_mapping'])
            self.episode_metadata = index_data['episode_metadata']
            
            # Convert sets back from lists if needed
            for word, episodes in self.word_episode_mapping.items():
                if isinstance(episodes, list):
                    self.word_episode_mapping[word] = set(episodes)
                    
            logger.info(f"Loaded keyword index from cache: {len(self.episode_word_frequencies)} episodes, {len(self.word_episode_mapping)} words")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_episodes': len(self.episode_word_frequencies),
            'total_unique_words': len(self.word_episode_mapping),
            'avg_words_per_episode': sum(
                metadata['total_words'] 
                for metadata in self.episode_metadata.values()
            ) / max(len(self.episode_metadata), 1),
            'avg_unique_words_per_episode': sum(
                metadata['unique_words'] 
                for metadata in self.episode_metadata.values()
            ) / max(len(self.episode_metadata), 1)
        }


def main():
    """Test keyword indexer functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Create indexer
    indexer = KeywordIndexer()
    
    # Try to load from cache first
    if not indexer.load_index():
        # Build index from scratch
        indexer.build_index_from_episodes_dir()
    
    # Test search
    results = indexer.search_by_keywords(['biochar'], top_k=5)
    
    print(f"\nBiochar search results ({len(results)} found):")
    for result in results:
        print(f"Episode {result['episode_number']}: {result['title']}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Keyword frequencies: {result['keyword_frequencies']}")
        print()
        
    # Get index stats
    stats = indexer.get_stats()
    print(f"\nIndex statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()