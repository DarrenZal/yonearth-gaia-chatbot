"""
Tests for text chunking functionality
"""
import pytest
from unittest.mock import patch, Mock

from src.ingestion.chunker import TranscriptChunker
from src.ingestion.episode_processor import Episode


class TestTranscriptChunker:
    """Test TranscriptChunker functionality"""
    
    def test_chunker_initialization(self):
        """Test chunker initialization with default parameters"""
        chunker = TranscriptChunker()
        assert chunker.chunk_size == 500  # From mock settings
        assert chunker.chunk_overlap == 50
        assert chunker.preserve_speaker_turns is True
    
    def test_chunker_initialization_with_params(self):
        """Test chunker initialization with custom parameters"""
        chunker = TranscriptChunker(
            chunk_size=1000,
            chunk_overlap=100,
            preserve_speaker_turns=False
        )
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100
        assert chunker.preserve_speaker_turns is False
    
    def test_detect_speaker_format_timestamp_dash(self):
        """Test detection of timestamp-dash speaker format"""
        chunker = TranscriptChunker()
        
        transcript = """
        5:45 – speaker1
        This is what speaker1 said.
        
        7:23 – speaker2
        This is what speaker2 said.
        """
        
        format_type = chunker._detect_speaker_format(transcript)
        assert format_type == "timestamp_dash"
    
    def test_detect_speaker_format_speaker_colon(self):
        """Test detection of speaker:colon format"""
        chunker = TranscriptChunker()
        
        transcript = """
        Aaron Perry: Welcome to the show.
        
        Guest Speaker: Thank you for having me.
        
        Aaron Perry: Let's talk about sustainability.
        """
        
        format_type = chunker._detect_speaker_format(transcript)
        assert format_type == "speaker_colon"
    
    def test_detect_speaker_format_none(self):
        """Test detection when no speaker format is found"""
        chunker = TranscriptChunker()
        
        transcript = """
        This is just regular text without any speaker indicators.
        It continues for multiple paragraphs without structure.
        """
        
        format_type = chunker._detect_speaker_format(transcript)
        assert format_type is None
    
    def test_split_by_speaker_turns_timestamp_dash(self):
        """Test splitting by timestamp-dash format"""
        chunker = TranscriptChunker(chunk_size=100)
        
        transcript = """
        5:45 – aaron
        Welcome to the YonEarth podcast.
        
        6:30 – guest
        Thank you for having me, Aaron.
        
        7:15 – aaron
        Let's talk about regenerative agriculture.
        """
        
        chunks = chunker._split_by_speaker_turns(transcript, "timestamp_dash")
        assert len(chunks) > 0
        assert "5:45 – aaron" in chunks[0]
    
    def test_split_by_speaker_turns_speaker_colon(self):
        """Test splitting by speaker:colon format"""
        chunker = TranscriptChunker(chunk_size=100)
        
        transcript = """
        Aaron Perry: Welcome to the show.
        This is a longer introduction that continues.
        
        Guest Speaker: Thank you for having me.
        I'm excited to be here today.
        
        Aaron Perry: Great to have you.
        """
        
        chunks = chunker._split_by_speaker_turns(transcript, "speaker_colon")
        assert len(chunks) > 0
        assert "Aaron Perry:" in chunks[0]
    
    @patch('src.ingestion.chunker.settings')
    def test_chunk_episode_with_speaker_format(self, mock_settings):
        """Test chunking episode with speaker format detection"""
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 20
        
        episode_data = {
            "episode_number": "42",
            "title": "Test Episode",
            "guest_name": "Test Guest",
            "url": "https://example.com",
            "full_transcript": """
            Aaron Perry: Welcome to the YonEarth podcast.
            Today we're discussing regenerative agriculture.
            
            Test Guest: Thank you for having me, Aaron.
            I'm excited to share about sustainable farming.
            
            Aaron Perry: Let's start with the basics.
            What is regenerative agriculture?
            
            Test Guest: Regenerative agriculture focuses on soil health.
            It's about rebuilding rather than depleting resources.
            """
        }
        
        episode = Episode(episode_data)
        chunker = TranscriptChunker()
        documents = chunker.chunk_episode(episode)
        
        assert len(documents) > 0
        
        # Check document structure
        doc = documents[0]
        assert hasattr(doc, 'page_content')
        assert hasattr(doc, 'metadata')
        
        # Check metadata
        assert doc.metadata["episode_number"] == "42"
        assert doc.metadata["chunk_index"] == 0
        assert "chunk_total" in doc.metadata
        assert doc.metadata["chunk_type"] == "speaker_turn"
    
    @patch('src.ingestion.chunker.settings')
    def test_chunk_episode_without_speaker_format(self, mock_settings):
        """Test chunking episode without speaker format"""
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 20
        
        episode_data = {
            "episode_number": "43",
            "title": "Plain Text Episode",
            "guest_name": "Guest",
            "url": "https://example.com",
            "full_transcript": """
            This is a plain text transcript without clear speaker indicators.
            It contains multiple paragraphs of content that should be chunked
            based on the standard text splitting algorithm rather than speaker turns.
            The content continues for several more sentences to ensure proper chunking.
            """
        }
        
        episode = Episode(episode_data)
        chunker = TranscriptChunker()
        documents = chunker.chunk_episode(episode)
        
        assert len(documents) > 0
        
        # Should use standard chunking
        doc = documents[0]
        assert doc.metadata["chunk_type"] == "speaker_turn"  # Still tries speaker detection
    
    def test_chunk_episode_preserve_speaker_turns_false(self):
        """Test chunking with speaker turn preservation disabled"""
        episode_data = {
            "episode_number": "44",
            "title": "Standard Chunking",
            "guest_name": "Guest",
            "url": "https://example.com",
            "full_transcript": """
            Aaron Perry: This should be chunked normally.
            Test Guest: Even though there are speaker indicators.
            """ * 10  # Make it long enough to chunk
        }
        
        episode = Episode(episode_data)
        chunker = TranscriptChunker(preserve_speaker_turns=False)
        documents = chunker.chunk_episode(episode)
        
        assert len(documents) > 0
        
        # Should use standard chunking
        doc = documents[0]
        assert doc.metadata["chunk_type"] == "standard"
    
    def test_chunk_episodes_multiple(self):
        """Test chunking multiple episodes"""
        episode_data_1 = {
            "episode_number": "1",
            "title": "Episode 1",
            "guest_name": "Guest 1",
            "url": "https://example.com/1",
            "full_transcript": "Aaron Perry: Welcome. Guest 1: Thank you." * 20
        }
        
        episode_data_2 = {
            "episode_number": "2",
            "title": "Episode 2",
            "guest_name": "Guest 2",
            "url": "https://example.com/2",
            "full_transcript": "Aaron Perry: Hello. Guest 2: Hi there." * 20
        }
        
        episodes = [Episode(episode_data_1), Episode(episode_data_2)]
        chunker = TranscriptChunker()
        documents = chunker.chunk_episodes(episodes)
        
        assert len(documents) > 0
        
        # Should have chunks from both episodes
        episode_numbers = {doc.metadata["episode_number"] for doc in documents}
        assert "1" in episode_numbers
        assert "2" in episode_numbers
    
    def test_chunk_episodes_with_error(self):
        """Test chunking with episodes that cause errors"""
        # Create episode with problematic data
        episode_data = {
            "episode_number": "error",
            "title": "Error Episode",
            "guest_name": "Error Guest",
            "url": "https://example.com/error",
            "full_transcript": None  # This might cause an error
        }
        
        episodes = [Episode(episode_data)]
        chunker = TranscriptChunker()
        
        # Should handle errors gracefully
        documents = chunker.chunk_episodes(episodes)
        
        # Might return empty list if error handling prevents processing
        assert isinstance(documents, list)


@pytest.mark.integration
class TestTranscriptChunkerIntegration:
    """Integration tests for transcript chunking"""
    
    def test_full_chunking_workflow(self, sample_episode_data):
        """Test complete chunking workflow"""
        episode = Episode(sample_episode_data)
        chunker = TranscriptChunker()
        
        # Chunk the episode
        documents = chunker.chunk_episode(episode)
        
        # Verify results
        assert len(documents) > 0
        
        # Check that all chunks have required metadata
        for doc in documents:
            assert "episode_number" in doc.metadata
            assert "title" in doc.metadata
            assert "guest_name" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "chunk_total" in doc.metadata
            assert len(doc.page_content) > 0
        
        # Check chunk indices are sequential
        chunk_indices = [doc.metadata["chunk_index"] for doc in documents]
        assert chunk_indices == list(range(len(documents)))