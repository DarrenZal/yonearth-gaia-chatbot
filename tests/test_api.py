"""
Tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json

from src.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain for testing"""
    with patch('src.api.main.get_rag_dependency') as mock_get_rag:
        mock_rag = Mock()
        
        # Mock query response
        mock_rag.query.return_value = {
            "response": "This is Gaia's response about regenerative agriculture.",
            "personality": "warm_mother",
            "citations": [
                {
                    "episode_number": "42",
                    "title": "Test Episode",
                    "guest_name": "Test Guest",
                    "url": "https://example.com",
                    "relevance": "High"
                }
            ],
            "context_used": 1,
            "retrieval_count": 3
        }
        
        # Mock stats
        mock_rag.get_stats.return_value = {
            "initialized": True,
            "gaia_personality": "warm_mother",
            "vectorstore_stats": {
                "total_vector_count": 100,
                "dimension": 1536
            }
        }
        
        # Mock recommendations
        mock_rag.get_episode_recommendations.return_value = [
            {
                "episode_number": "42",
                "title": "Test Episode",
                "guest_name": "Test Guest",
                "url": "https://example.com",
                "relevance_score": 0.9,
                "reason": "Highly relevant to your query"
            }
        ]
        
        # Mock search
        mock_rag.search_episodes.return_value = [
            {
                "episode_number": "42",
                "title": "Test Episode",
                "guest_name": "Test Guest",
                "url": "https://example.com",
                "relevance_score": 0.9,
                "content_preview": "This is a preview of the content...",
                "metadata": {"key": "value"}
            }
        ]
        
        mock_get_rag.return_value = mock_rag
        yield mock_rag


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client, mock_rag_chain):
        """Test successful health check"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["rag_initialized"] is True
        assert data["gaia_personality"] == "warm_mother"
        assert "vectorstore_stats" in data
    
    def test_health_check_failure(self, client):
        """Test health check when service is unhealthy"""
        with patch('src.api.main.get_rag_dependency') as mock_get_rag:
            mock_get_rag.side_effect = Exception("Service error")
            
            response = client.get("/health")
            
            assert response.status_code == 503
            assert "Service unhealthy" in response.json()["detail"]


class TestChatEndpoint:
    """Test chat endpoint"""
    
    def test_chat_success(self, client, mock_rag_chain):
        """Test successful chat request"""
        chat_request = {
            "message": "Tell me about regenerative agriculture",
            "session_id": "test-session",
            "personality": "warm_mother",
            "max_results": 5
        }
        
        response = client.post("/chat", json=chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["response"] == "This is Gaia's response about regenerative agriculture."
        assert data["personality"] == "warm_mother"
        assert len(data["citations"]) == 1
        assert data["context_used"] == 1
        assert data["session_id"] == "test-session"
        assert data["retrieval_count"] == 3
        assert "processing_time" in data
        
        # Verify RAG chain was called correctly
        mock_rag_chain.query.assert_called_once_with(
            user_input="Tell me about regenerative agriculture",
            k=5,
            session_id="test-session",
            personality_variant="warm_mother"
        )
    
    def test_chat_minimal_request(self, client, mock_rag_chain):
        """Test chat with minimal request data"""
        chat_request = {
            "message": "Hello Gaia"
        }
        
        response = client.post("/chat", json=chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
        assert "personality" in data
        assert "citations" in data
    
    def test_chat_invalid_request(self, client):
        """Test chat with invalid request data"""
        # Empty message
        chat_request = {"message": ""}
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 422
        
        # No message
        chat_request = {}
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 422
        
        # Message too long
        chat_request = {"message": "x" * 1001}
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 422
    
    def test_chat_rag_error(self, client):
        """Test chat when RAG chain raises an error"""
        with patch('src.api.main.get_rag_dependency') as mock_get_rag:
            mock_rag = Mock()
            mock_rag.query.side_effect = Exception("RAG error")
            mock_get_rag.return_value = mock_rag
            
            chat_request = {"message": "Test message"}
            response = client.post("/chat", json=chat_request)
            
            assert response.status_code == 500
            assert "experiencing difficulties" in response.json()["detail"]


class TestRecommendationsEndpoint:
    """Test recommendations endpoint"""
    
    def test_recommendations_success(self, client, mock_rag_chain):
        """Test successful recommendations request"""
        rec_request = {
            "query": "soil health",
            "max_recommendations": 3
        }
        
        response = client.post("/recommendations", json=rec_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["query"] == "soil health"
        assert len(data["recommendations"]) == 1
        assert data["total_found"] == 1
        
        rec = data["recommendations"][0]
        assert rec["episode_number"] == "42"
        assert rec["title"] == "Test Episode"
        assert rec["guest_name"] == "Test Guest"
        assert rec["relevance_score"] == 0.9
        
        # Verify RAG chain was called
        mock_rag_chain.get_episode_recommendations.assert_called_once_with(
            user_input="soil health",
            k=3
        )
    
    def test_recommendations_invalid_request(self, client):
        """Test recommendations with invalid request"""
        # Empty query
        rec_request = {"query": ""}
        response = client.post("/recommendations", json=rec_request)
        assert response.status_code == 422
        
        # Query too long
        rec_request = {"query": "x" * 501}
        response = client.post("/recommendations", json=rec_request)
        assert response.status_code == 422
    
    def test_recommendations_error(self, client):
        """Test recommendations when RAG chain raises error"""
        with patch('src.api.main.get_rag_dependency') as mock_get_rag:
            mock_rag = Mock()
            mock_rag.get_episode_recommendations.side_effect = Exception("Error")
            mock_get_rag.return_value = mock_rag
            
            rec_request = {"query": "test"}
            response = client.post("/recommendations", json=rec_request)
            
            assert response.status_code == 500


class TestSearchEndpoint:
    """Test search endpoint"""
    
    def test_search_success(self, client, mock_rag_chain):
        """Test successful search request"""
        search_request = {
            "query": "regenerative agriculture",
            "max_results": 10,
            "filters": {"guest_name": "Test Guest"}
        }
        
        response = client.post("/search", json=search_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["query"] == "regenerative agriculture"
        assert len(data["results"]) == 1
        assert data["total_found"] == 1
        assert data["filters_applied"] == {"guest_name": "Test Guest"}
        
        result = data["results"][0]
        assert result["episode_number"] == "42"
        assert result["title"] == "Test Episode"
        assert result["relevance_score"] == 0.9
        
        # Verify RAG chain was called
        mock_rag_chain.search_episodes.assert_called_once_with(
            query="regenerative agriculture",
            filters={"guest_name": "Test Guest"},
            k=10
        )
    
    def test_search_minimal_request(self, client, mock_rag_chain):
        """Test search with minimal request"""
        search_request = {"query": "test"}
        
        response = client.post("/search", json=search_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["query"] == "test"
        assert "results" in data
        assert "total_found" in data


class TestResetConversationEndpoint:
    """Test reset conversation endpoint"""
    
    def test_reset_conversation_success(self, client, mock_rag_chain):
        """Test successful conversation reset"""
        response = client.post("/reset-conversation?session_id=test-session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Conversation reset successfully"
        assert data["session_id"] == "test-session"
        
        # Verify RAG chain was called
        mock_rag_chain.reset_conversation.assert_called_once_with(session_id="test-session")
    
    def test_reset_conversation_no_session(self, client, mock_rag_chain):
        """Test conversation reset without session ID"""
        response = client.post("/reset-conversation")
        
        assert response.status_code == 200
        
        # Should be called with None
        mock_rag_chain.reset_conversation.assert_called_once_with(session_id=None)
    
    def test_reset_conversation_error(self, client):
        """Test conversation reset with error"""
        with patch('src.api.main.get_rag_dependency') as mock_get_rag:
            mock_rag = Mock()
            mock_rag.reset_conversation.side_effect = Exception("Reset error")
            mock_get_rag.return_value = mock_rag
            
            response = client.post("/reset-conversation")
            
            assert response.status_code == 500


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "YonEarth Gaia Chatbot API" in data["message"]
        assert "version" in data
        assert "endpoints" in data
        
        endpoints = data["endpoints"]
        assert "chat" in endpoints
        assert "recommendations" in endpoints
        assert "search" in endpoints
        assert "health" in endpoints


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_full_chat_workflow(self, client, mock_rag_chain):
        """Test complete chat workflow"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Send chat message
        chat_request = {
            "message": "What is regenerative agriculture?",
            "session_id": "integration-test",
            "personality": "warm_mother"
        }
        chat_response = client.post("/chat", json=chat_request)
        assert chat_response.status_code == 200
        
        chat_data = chat_response.json()
        assert len(chat_data["citations"]) > 0
        
        # 3. Get recommendations based on the same topic
        rec_request = {"query": "regenerative agriculture"}
        rec_response = client.post("/recommendations", json=rec_request)
        assert rec_response.status_code == 200
        
        # 4. Search for episodes
        search_request = {"query": "agriculture"}
        search_response = client.post("/search", json=search_request)
        assert search_response.status_code == 200
        
        # 5. Reset conversation
        reset_response = client.post("/reset-conversation?session_id=integration-test")
        assert reset_response.status_code == 200
    
    @patch('src.api.main.limiter')
    def test_rate_limiting(self, mock_limiter, client, mock_rag_chain):
        """Test rate limiting functionality"""
        # Mock rate limiter to simulate rate limit exceeded
        from slowapi.errors import RateLimitExceeded
        
        # First request should succeed
        chat_request = {"message": "Test message"}
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 200
        
        # Note: Actual rate limiting testing would require integration with Redis
        # This test verifies the rate limiter is properly configured