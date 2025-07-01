"""
Tests for Gaia character functionality
"""
import pytest
from unittest.mock import patch, Mock, MagicMock

from src.character.gaia import GaiaCharacter, create_gaia
from src.character.gaia_personalities import get_personality, get_available_personalities


class TestGaiaPersonalities:
    """Test personality system"""
    
    def test_get_personality_warm_mother(self):
        """Test getting warm mother personality"""
        personality = get_personality("warm_mother")
        assert "nurturing spirit of Mother Earth" in personality
        assert "maternal warmth" in personality
    
    def test_get_personality_wise_guide(self):
        """Test getting wise guide personality"""
        personality = get_personality("wise_guide")
        assert "ancient wisdom of Earth" in personality
        assert "deep knowing" in personality
    
    def test_get_personality_earth_activist(self):
        """Test getting earth activist personality"""
        personality = get_personality("earth_activist")
        assert "fierce and loving guardian" in personality
        assert "passionate" in personality
    
    def test_get_personality_invalid_defaults_to_warm_mother(self):
        """Test that invalid personality returns warm mother"""
        personality = get_personality("invalid_personality")
        assert "nurturing spirit of Mother Earth" in personality
    
    def test_get_available_personalities(self):
        """Test getting list of available personalities"""
        personalities = get_available_personalities()
        assert "warm_mother" in personalities
        assert "wise_guide" in personalities
        assert "earth_activist" in personalities
        assert len(personalities) == 3


class TestGaiaCharacter:
    """Test GaiaCharacter functionality"""
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    @patch('src.config.settings')
    def test_gaia_initialization_default(self, mock_settings, mock_memory, mock_chat):
        """Test Gaia initialization with default settings"""
        mock_settings.gaia_personality_variant = "warm_mother"
        mock_settings.openai_model = "gpt-3.5-turbo"
        mock_settings.gaia_temperature = 0.7
        mock_settings.gaia_max_tokens = 1000
        mock_settings.openai_api_key = "test-key"
        
        gaia = GaiaCharacter()
        
        assert gaia.personality_variant == "warm_mother"
        mock_chat.assert_called_once()
        mock_memory.assert_called_once()
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_gaia_initialization_custom_personality(self, mock_memory, mock_chat):
        """Test Gaia initialization with custom personality"""
        gaia = GaiaCharacter(personality_variant="wise_guide")
        
        assert gaia.personality_variant == "wise_guide"
        assert "ancient wisdom" in gaia.personality_prompt
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_format_context_with_documents(self, mock_memory, mock_chat):
        """Test context formatting with retrieved documents"""
        gaia = GaiaCharacter()
        
        # Mock documents
        mock_docs = [
            Mock(
                metadata={
                    "episode_number": "42",
                    "title": "Test Episode",
                    "guest_name": "Test Guest"
                },
                page_content="This is about regenerative agriculture." * 20  # Long content
            ),
            Mock(
                metadata={
                    "episode_number": "43",
                    "title": "Another Episode",
                    "guest_name": "Another Guest"
                },
                page_content="This is about sustainability."
            )
        ]
        
        context = gaia._format_context(mock_docs)
        
        assert "Episode 42" in context
        assert "Test Episode" in context
        assert "Test Guest" in context
        assert "Episode 43" in context
        assert "..." in context  # Should truncate long content
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_format_context_no_documents(self, mock_memory, mock_chat):
        """Test context formatting with no documents"""
        gaia = GaiaCharacter()
        
        context = gaia._format_context([])
        
        assert "No specific episode content found" in context
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_extract_citations(self, mock_memory, mock_chat):
        """Test citation extraction from documents"""
        gaia = GaiaCharacter()
        
        mock_docs = [
            Mock(
                metadata={
                    "episode_number": "42",
                    "title": "Test Episode",
                    "guest_name": "Test Guest",
                    "url": "https://example.com/42"
                }
            ),
            Mock(
                metadata={
                    "episode_number": "43",
                    "title": "Another Episode",
                    "guest_name": "Another Guest",
                    "url": "https://example.com/43"
                }
            )
        ]
        
        citations = gaia._extract_citations(mock_docs)
        
        assert len(citations) == 2
        assert citations[0]["episode_number"] == "42"
        assert citations[0]["title"] == "Test Episode"
        assert citations[0]["guest_name"] == "Test Guest"
        assert citations[0]["url"] == "https://example.com/42"
        assert citations[0]["relevance"] == "High"
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_generate_response_success(self, mock_memory, mock_chat):
        """Test successful response generation"""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_llm_instance.return_value.content = "This is Gaia's response about regenerative agriculture."
        mock_chat.return_value = mock_llm_instance
        
        mock_memory_instance = Mock()
        mock_memory_instance.chat_memory.messages = []
        mock_memory_instance.save_context = Mock()
        mock_memory.return_value = mock_memory_instance
        
        gaia = GaiaCharacter()
        gaia.llm = mock_llm_instance
        gaia.memory = mock_memory_instance
        
        # Test documents
        mock_docs = [
            Mock(
                metadata={
                    "episode_number": "42",
                    "title": "Test Episode",
                    "guest_name": "Test Guest",
                    "url": "https://example.com"
                }
            )
        ]
        
        response = gaia.generate_response(
            user_input="Tell me about regenerative agriculture",
            retrieved_docs=mock_docs,
            session_id="test-session"
        )
        
        assert response["response"] == "This is Gaia's response about regenerative agriculture."
        assert response["personality"] == gaia.personality_variant
        assert len(response["citations"]) == 1
        assert response["context_used"] == 1
        assert response["session_id"] == "test-session"
        
        # Verify memory was updated
        mock_memory_instance.save_context.assert_called_once()
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_generate_response_error(self, mock_memory, mock_chat):
        """Test response generation with error"""
        # Setup mocks to raise an exception
        mock_llm_instance = Mock()
        mock_llm_instance.side_effect = Exception("API Error")
        mock_chat.return_value = mock_llm_instance
        
        gaia = GaiaCharacter()
        gaia.llm = mock_llm_instance
        
        response = gaia.generate_response("Test question")
        
        assert "error" in response
        assert "having trouble" in response["response"]
        assert response["context_used"] == 0
        assert len(response["citations"]) == 0
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_clear_memory(self, mock_memory, mock_chat):
        """Test clearing conversation memory"""
        mock_memory_instance = Mock()
        mock_memory.return_value = mock_memory_instance
        
        gaia = GaiaCharacter()
        gaia.memory = mock_memory_instance
        
        gaia.clear_memory()
        
        mock_memory_instance.clear.assert_called_once()
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_get_memory_summary(self, mock_memory, mock_chat):
        """Test getting memory summary"""
        mock_memory_instance = Mock()
        mock_memory_instance.buffer = "Previous conversation summary"
        mock_memory.return_value = mock_memory_instance
        
        gaia = GaiaCharacter()
        gaia.memory = mock_memory_instance
        
        summary = gaia.get_memory_summary()
        
        assert summary == "Previous conversation summary"
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_get_memory_summary_no_buffer(self, mock_memory, mock_chat):
        """Test getting memory summary when no buffer exists"""
        mock_memory_instance = Mock()
        mock_memory_instance.buffer = None
        mock_memory.return_value = mock_memory_instance
        
        gaia = GaiaCharacter()
        gaia.memory = mock_memory_instance
        
        summary = gaia.get_memory_summary()
        
        assert summary == "No conversation history yet."
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    @patch('src.character.gaia.ChatPromptTemplate')
    def test_switch_personality(self, mock_prompt, mock_memory, mock_chat):
        """Test switching personality variant"""
        mock_prompt_instance = Mock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        
        gaia = GaiaCharacter(personality_variant="warm_mother")
        
        gaia.switch_personality("wise_guide")
        
        assert gaia.personality_variant == "wise_guide"
        assert "ancient wisdom" in gaia.personality_prompt
        
        # Should recreate prompt template
        mock_prompt.from_messages.assert_called()
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_switch_personality_invalid(self, mock_memory, mock_chat):
        """Test switching to invalid personality"""
        gaia = GaiaCharacter()
        
        with pytest.raises(ValueError, match="Unknown personality variant"):
            gaia.switch_personality("invalid_personality")


class TestCreateGaia:
    """Test Gaia factory function"""
    
    @patch('src.character.gaia.GaiaCharacter')
    def test_create_gaia_default(self, mock_gaia_class):
        """Test creating Gaia with default personality"""
        mock_instance = Mock()
        mock_gaia_class.return_value = mock_instance
        
        result = create_gaia()
        
        mock_gaia_class.assert_called_once_with(personality_variant=None)
        assert result == mock_instance
    
    @patch('src.character.gaia.GaiaCharacter')
    def test_create_gaia_custom_personality(self, mock_gaia_class):
        """Test creating Gaia with custom personality"""
        mock_instance = Mock()
        mock_gaia_class.return_value = mock_instance
        
        result = create_gaia("earth_activist")
        
        mock_gaia_class.assert_called_once_with(personality_variant="earth_activist")
        assert result == mock_instance


@pytest.mark.integration
class TestGaiaCharacterIntegration:
    """Integration tests for Gaia character"""
    
    @patch('src.character.gaia.ChatOpenAI')
    @patch('src.character.gaia.ConversationSummaryBufferMemory')
    def test_full_conversation_flow(self, mock_memory, mock_chat):
        """Test complete conversation flow"""
        # Setup mocks for a realistic conversation
        mock_llm_instance = Mock()
        mock_llm_instance.return_value.content = "Welcome, dear one. Let me share wisdom about regenerative agriculture from Episode 42 with Test Guest."
        mock_chat.return_value = mock_llm_instance
        
        mock_memory_instance = Mock()
        mock_memory_instance.chat_memory.messages = []
        mock_memory_instance.save_context = Mock()
        mock_memory.return_value = mock_memory_instance
        
        gaia = GaiaCharacter(personality_variant="warm_mother")
        gaia.llm = mock_llm_instance
        gaia.memory = mock_memory_instance
        
        # Simulate retrieved documents
        mock_docs = [
            Mock(
                metadata={
                    "episode_number": "42",
                    "title": "Regenerative Agriculture Basics",
                    "guest_name": "Test Guest",
                    "url": "https://yonearth.org/episode-42"
                },
                page_content="Regenerative agriculture focuses on rebuilding soil health through natural practices."
            )
        ]
        
        # Generate response
        response = gaia.generate_response(
            user_input="What is regenerative agriculture?",
            retrieved_docs=mock_docs,
            session_id="test-session-123"
        )
        
        # Verify response structure
        assert "response" in response
        assert "personality" in response
        assert "citations" in response
        assert "context_used" in response
        assert "session_id" in response
        
        # Verify citations
        assert len(response["citations"]) == 1
        citation = response["citations"][0]
        assert citation["episode_number"] == "42"
        assert citation["title"] == "Regenerative Agriculture Basics"
        assert citation["guest_name"] == "Test Guest"
        
        # Verify memory was updated
        mock_memory_instance.save_context.assert_called_once_with(
            {"input": "What is regenerative agriculture?"},
            {"output": mock_llm_instance.return_value.content}
        )