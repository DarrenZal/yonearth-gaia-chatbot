#!/usr/bin/env python3
"""
Test script for hybrid search functionality
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
import json
from typing import Dict, List, Any
from collections import Counter, defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SimpleKeywordIndexer:
    """Simplified keyword indexer for testing"""
    
    def __init__(self, episodes_dir: Path):
        self.episodes_dir = episodes_dir
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Episode indexes
        self.episode_word_frequencies = {}  # episode_id -> {word: frequency}
        self.word_episode_mapping = defaultdict(set)  # word -> {episode_ids}
        self.episode_metadata = {}  # episode_id -> metadata
        
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
                word not in self.stop_words):
                
                # Apply stemming
                final_word = self.stemmer.stem(word)
                filtered_words.append(final_word)
                
        return filtered_words
    
    def _load_episode_data(self, episode_path: Path) -> Dict[str, Any]:
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
    
    def build_index(self):
        """Build index from all episodes"""
        logger.info(f"Building keyword index from {self.episodes_dir}")
        
        # Clear existing indexes
        self.episode_word_frequencies.clear()
        self.word_episode_mapping.clear()
        self.episode_metadata.clear()
        
        # Process all episode files
        episode_files = list(self.episodes_dir.glob("episode_*.json"))
        successful_episodes = 0
        
        for episode_file in episode_files:
            episode_data = self._load_episode_data(episode_file)
            if episode_data:
                episode_id = episode_file.stem  # filename without extension
                self._index_episode(episode_data, episode_id)
                successful_episodes += 1
                
        logger.info(f"Successfully indexed {successful_episodes}/{len(episode_files)} episodes")
        logger.info(f"Total unique words in index: {len(self.word_episode_mapping)}")
    
    def _index_episode(self, episode_data: Dict[str, Any], episode_id: str):
        """Index a single episode"""
        transcript = episode_data.get('full_transcript', '')
        if not transcript or transcript == 'NO_TRANSCRIPT_AVAILABLE':
            return
            
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
            'total_words': sum(word_frequencies.values()),
            'unique_words': len(word_frequencies)
        }
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 10):
        """Search episodes by keywords"""
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
                    if frequency >= 1:
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
            
        return results


def test_biochar_search():
    """Test biochar search specifically"""
    # Set up paths
    project_root = Path(__file__).parent.parent
    episodes_dir = project_root / "data" / "json"
    
    print(f"Episodes directory: {episodes_dir}")
    print(f"Episodes directory exists: {episodes_dir.exists()}")
    
    if not episodes_dir.exists():
        print("ERROR: Episodes directory not found!")
        return
    
    # Create indexer and build index
    indexer = SimpleKeywordIndexer(episodes_dir)
    indexer.build_index()
    
    # Test biochar search
    print("\n" + "="*50)
    print("TESTING BIOCHAR SEARCH")
    print("="*50)
    
    results = indexer.search_by_keywords(['biochar'], top_k=10)
    print(f"\nFound {len(results)} episodes with 'biochar':")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Episode {result['episode_number']}: {result['title']}")
        print(f"   Score: {result['score']:.6f}")
        print(f"   Keyword frequencies: {result['keyword_frequencies']}")
        print(f"   Total words: {result['total_words']}")
    
    # Also test some other terms for comparison
    print("\n" + "="*50)
    print("TESTING OTHER SEARCH TERMS")
    print("="*50)
    
    test_terms = ['regenerative agriculture', 'soil health', 'permaculture']
    for term in test_terms:
        results = indexer.search_by_keywords([term], top_k=3)
        print(f"\nTop 3 results for '{term}':")
        for result in results:
            print(f"  Episode {result['episode_number']}: {result['title']} (score: {result['score']:.4f})")


if __name__ == "__main__":
    test_biochar_search()