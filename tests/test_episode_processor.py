"""
Tests for episode processing functionality
"""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock

from src.ingestion.episode_processor import Episode, EpisodeProcessor


class TestEpisode:
    """Test Episode class functionality"""
    
    def test_episode_initialization(self, sample_episode_data):
        """Test Episode object creation"""
        episode = Episode(sample_episode_data)
        
        assert episode.episode_number == "42"
        assert episode.title == "Regenerative Agriculture with Test Guest"
        assert episode.guest_name == "Test Guest"
        assert episode.has_transcript is True
        
    def test_episode_guest_name_extraction(self):
        """Test guest name extraction from different sources"""
        # Test extraction from about_guest
        data = {
            "about_sections": {
                "about_guest": "John Smith is a leading expert in permaculture."
            },
            "full_transcript": "Long transcript here..."
        }
        episode = Episode(data)
        assert episode.guest_name == "John Smith"
        
        # Test extraction from subtitle
        data = {
            "subtitle": "Permaculture basics with Jane Doe",
            "full_transcript": "Long transcript here..."
        }
        episode = Episode(data)
        assert episode.guest_name == "Jane Doe"
        
        # Test fallback
        data = {"full_transcript": "Long transcript here..."}
        episode = Episode(data)
        assert episode.guest_name == "Guest"
    
    def test_episode_has_transcript(self):
        """Test transcript validation"""
        # Valid transcript
        data = {"full_transcript": "A" * 200}
        episode = Episode(data)
        assert episode.has_transcript is True
        
        # Too short transcript
        data = {"full_transcript": "Short"}
        episode = Episode(data)
        assert episode.has_transcript is False
        
        # No transcript
        data = {}
        episode = Episode(data)
        assert episode.has_transcript is False
    
    def test_episode_metadata(self, sample_episode_data):
        """Test metadata extraction"""
        episode = Episode(sample_episode_data)
        metadata = episode.metadata
        
        assert metadata["episode_number"] == "42"
        assert metadata["title"] == "Regenerative Agriculture with Test Guest"
        assert metadata["guest_name"] == "Test Guest"
        assert metadata["url"] == "https://yonearth.org/episode-42"
    
    def test_episode_to_document(self, sample_episode_data):
        """Test document conversion"""
        episode = Episode(sample_episode_data)
        doc = episode.to_document()
        
        assert "page_content" in doc
        assert "metadata" in doc
        assert doc["page_content"] == episode.transcript
        assert doc["metadata"]["episode_number"] == "42"


class TestEpisodeProcessor:
    """Test EpisodeProcessor functionality"""
    
    def test_processor_initialization(self, mock_settings):
        """Test processor initialization"""
        processor = EpisodeProcessor()
        assert processor.episodes_dir == mock_settings.episodes_dir
        assert processor.processed_dir == mock_settings.processed_dir
    
    def test_load_episodes(self, mock_settings, mock_episodes_dir):
        """Test episode loading"""
        processor = EpisodeProcessor()
        episodes = processor.load_episodes(limit=2)
        
        assert len(episodes) <= 2
        assert all(isinstance(ep, Episode) for ep in episodes)
        assert all(ep.has_transcript for ep in episodes)
    
    def test_load_episodes_with_invalid_files(self, mock_settings, tmp_path):
        """Test handling of invalid episode files"""
        # Create invalid JSON file
        invalid_file = tmp_path / "episode_invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json")
        
        # Create episode without transcript
        no_transcript_data = {"episode_number": "99", "title": "No Transcript"}
        no_transcript_file = tmp_path / "episode_99.json"
        with open(no_transcript_file, 'w') as f:
            json.dump(no_transcript_data, f)
        
        mock_settings.episodes_dir = tmp_path
        mock_settings.get_episode_files.return_value = [invalid_file, no_transcript_file]
        
        processor = EpisodeProcessor()
        episodes = processor.load_episodes()
        
        # Should handle errors gracefully
        assert len(episodes) == 0  # No valid episodes
    
    def test_get_diverse_episodes(self, mock_settings):
        """Test diverse episode selection"""
        processor = EpisodeProcessor()
        
        # Create test episodes
        episodes = []
        for i in range(10):
            episode_data = {
                "episode_number": str(i),
                "title": f"Episode {i}",
                "full_transcript": "A" * 200
            }
            episodes.append(Episode(episode_data))
        
        # Test with fewer episodes than requested
        diverse = processor.get_diverse_episodes(episodes, count=15)
        assert len(diverse) == 10  # Should return all episodes
        
        # Test with more episodes than requested
        diverse = processor.get_diverse_episodes(episodes, count=5)
        assert len(diverse) == 5  # Should return 5 episodes
        assert diverse[0].episode_number == "0"  # Should include first
    
    def test_save_processed_episodes(self, mock_settings, tmp_path):
        """Test saving episode metadata"""
        mock_settings.processed_dir = tmp_path
        processor = EpisodeProcessor()
        
        # Create test episodes
        episodes = [
            Episode({
                "episode_number": "1",
                "title": "Test Episode 1",
                "full_transcript": "A" * 200,
                "publish_date": "2023-01-01"
            })
        ]
        
        processor.save_processed_episodes(episodes)
        
        # Check if metadata file was created
        metadata_file = tmp_path / "episode_metadata.json"
        assert metadata_file.exists()
        
        # Check metadata content
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["episode_count"] == 1
        assert len(metadata["episodes"]) == 1
        assert metadata["episodes"][0]["number"] == "1"
        assert "processed_at" in metadata


@pytest.mark.integration
class TestEpisodeProcessorIntegration:
    """Integration tests for episode processing"""
    
    def test_full_episode_processing_workflow(self, mock_settings, mock_episodes_dir):
        """Test complete episode processing workflow"""
        processor = EpisodeProcessor()
        
        # Load episodes
        episodes = processor.load_episodes()
        assert len(episodes) > 0
        
        # Get diverse episodes
        diverse_episodes = processor.get_diverse_episodes(episodes, count=2)
        assert len(diverse_episodes) <= 2
        
        # Save metadata
        processor.save_processed_episodes(diverse_episodes)
        
        # Check that metadata was saved
        metadata_file = mock_settings.processed_dir / "episode_metadata.json"
        assert metadata_file.exists()