"""
Pytest configuration and fixtures for YonEarth chatbot tests
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Test data
SAMPLE_EPISODE_DATA = {
    "episode_number": "42",
    "title": "Regenerative Agriculture with Test Guest",
    "subtitle": "Healing the soil with sustainable practices",
    "description": "A conversation about regenerative farming techniques.",
    "publish_date": "2023-01-15",
    "audio_url": "https://example.com/audio/episode-42.mp3",
    "url": "https://yonearth.org/episode-42",
    "full_transcript": """
    Aaron Perry: Welcome to the YonEarth Community Podcast. Today we're talking about regenerative agriculture.
    
    Test Guest: Thank you for having me, Aaron. I'm excited to share about soil health and regenerative practices.
    
    Aaron Perry: Let's start with the basics. What is regenerative agriculture?
    
    Test Guest: Regenerative agriculture focuses on rebuilding soil health through practices like cover cropping, 
    rotational grazing, and reducing tillage. It's about working with natural systems.
    
    Aaron Perry: That's fascinating. How does this differ from conventional farming?
    
    Test Guest: Conventional farming often depletes soil over time, while regenerative practices actually 
    improve soil health, increase carbon sequestration, and support biodiversity.
    """,
    "about_sections": {
        "about_guest": "Test Guest is a regenerative agriculture expert with 20 years of experience in sustainable farming."
    },
    "related_episodes": []
}

SAMPLE_CITATIONS = [
    {
        "episode_number": "42",
        "title": "Regenerative Agriculture with Test Guest",
        "guest_name": "Test Guest",
        "url": "https://yonearth.org/episode-42",
        "relevance": "High"
    }
]


@pytest.fixture
def sample_episode_data():
    """Provide sample episode data for testing"""
    return SAMPLE_EPISODE_DATA.copy()


@pytest.fixture
def sample_episode_file(tmp_path):
    """Create a temporary episode JSON file"""
    episode_file = tmp_path / "episode_42.json"
    with open(episode_file, 'w') as f:
        json.dump(SAMPLE_EPISODE_DATA, f)
    return episode_file


@pytest.fixture
def mock_episodes_dir(tmp_path, sample_episode_data):
    """Create a temporary directory with multiple episode files"""
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    
    # Create multiple test episodes
    for i in range(3):
        episode_data = sample_episode_data.copy()
        episode_data["episode_number"] = str(i)
        episode_data["title"] = f"Test Episode {i}"
        
        episode_file = episodes_dir / f"episode_{i}.json"
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f)
    
    return episodes_dir


@pytest.fixture
def mock_settings(mock_episodes_dir):
    """Mock settings for testing"""
    with patch('src.config.settings') as mock_settings:
        mock_settings.episodes_dir = mock_episodes_dir
        mock_settings.processed_dir = mock_episodes_dir / "processed"
        mock_settings.processed_dir.mkdir(exist_ok=True)
        mock_settings.episodes_to_process = 3
        mock_settings.chunk_size = 500
        mock_settings.chunk_overlap = 50
        mock_settings.openai_api_key = "test-key"
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_environment = "test"
        mock_settings.pinecone_index_name = "test-index"
        mock_settings.gaia_personality_variant = "warm_mother"
        mock_settings.gaia_temperature = 0.7
        mock_settings.gaia_max_tokens = 1000
        mock_settings.openai_model = "gpt-3.5-turbo"
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.get_episode_files.return_value = list(mock_episodes_dir.glob("episode_*.json"))
        yield mock_settings


@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls"""
    with patch('openai.Embedding.create') as mock_embed, \
         patch('openai.ChatCompletion.create') as mock_chat:
        
        # Mock embedding response
        mock_embed.return_value = {
            'data': [{
                'embedding': [0.1] * 1536  # Standard embedding dimension
            }]
        }
        
        # Mock chat response
        mock_chat.return_value = {
            'choices': [{
                'message': {
                    'content': 'This is a test response from Gaia about regenerative agriculture.'
                }
            }]
        }
        
        yield mock_embed, mock_chat


@pytest.fixture
def mock_pinecone():
    """Mock Pinecone operations"""
    with patch('pinecone.init') as mock_init, \
         patch('pinecone.Index') as mock_index_class, \
         patch('pinecone.list_indexes') as mock_list, \
         patch('pinecone.create_index') as mock_create:
        
        # Mock index instance
        mock_index = Mock()
        mock_index.upsert.return_value = {"upserted_count": 10}
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "test-1",
                    "score": 0.9,
                    "metadata": {
                        "episode_number": "42",
                        "title": "Test Episode",
                        "guest_name": "Test Guest"
                    },
                    "values": [0.1] * 1536
                }
            ]
        }
        mock_index.describe_index_stats.return_value = {
            "total_vector_count": 100,
            "dimension": 1536
        }
        
        mock_index_class.return_value = mock_index
        mock_list.return_value = []
        
        yield mock_init, mock_index, mock_create


@pytest.fixture
def sample_documents():
    """Sample LangChain documents for testing"""
    from src.utils.lc_compat import Document
    
    return [
        Document(
            page_content="This is about regenerative agriculture and soil health.",
            metadata={
                "episode_number": "42",
                "title": "Test Episode",
                "guest_name": "Test Guest",
                "url": "https://example.com"
            }
        ),
        Document(
            page_content="Sustainable farming practices for the future.",
            metadata={
                "episode_number": "43",
                "title": "Sustainability Episode",
                "guest_name": "Eco Expert",
                "url": "https://example.com"
            }
        )
    ]


@pytest.fixture
def mock_redis():
    """Mock Redis for testing"""
    with patch('redis.from_url') as mock_redis:
        mock_client = Mock()
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_redis.return_value = mock_client
        yield mock_client


class MockResponse:
    """Mock HTTP response for testing API endpoints"""
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
    
    def json(self):
        return self.json_data
    
    @property
    def ok(self):
        return 200 <= self.status_code < 300
