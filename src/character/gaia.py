"""
Gaia character implementation with conversation memory and personality system
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config import settings
from .gaia_personalities import get_personality

logger = logging.getLogger(__name__)


class GaiaCharacter:
    """Gaia character with personality, memory, and context awareness"""
    
    def __init__(self, personality_variant: Optional[str] = None, model_name: Optional[str] = None):
        self.personality_variant = personality_variant or settings.gaia_personality_variant
        self.model_name = model_name or settings.openai_model
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=settings.gaia_temperature,
            max_tokens=settings.gaia_max_tokens,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Load personality
        self.personality_prompt = get_personality(self.personality_variant)
        
        # Create chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.personality_prompt),
            ("system", "Context from YonEarth Podcast Episodes:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        logger.info(f"Initialized Gaia character with '{self.personality_variant}' personality")
    
    def _format_context(self, retrieved_docs: List[Any]) -> str:
        """Format retrieved documents as context for Gaia"""
        if not retrieved_docs:
            return "No specific episode content found for this query."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Limit to top 5 results
            metadata = getattr(doc, 'metadata', {})
            content = getattr(doc, 'page_content', str(doc))
            
            episode_num = metadata.get('episode_number', 'Unknown')
            title = metadata.get('title', 'Unknown Episode')
            guest = metadata.get('guest_name', 'Guest')
            
            # Truncate content for context
            content_preview = content[:300] + "..." if len(content) > 300 else content
            
            context_parts.append(
                f"Episode {episode_num} - \"{title}\" with {guest}:\n{content_preview}"
            )
        
        return "\n\n".join(context_parts)
    
    def _create_citation_reminder(self) -> str:
        """Create reminder about proper citation format"""
        return """
IMPORTANT: Always cite your sources using this exact format:
- For specific quotes: "As [Guest Name] explained in Episode [Number], '[specific quote or insight]'..."
- For general concepts: "In Episode [Number] with [Guest Name], we learned about [topic]..."
- Always include episode numbers and guest names when referencing information.
"""
    
    def generate_response(
        self, 
        user_input: str, 
        retrieved_docs: List[Any] = None,
        session_id: Optional[str] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate Gaia's response to user input"""
        try:
            # Format context from retrieved documents
            context = self._format_context(retrieved_docs or [])
            
            # Get conversation history
            chat_history = self.memory.chat_memory.messages
            
            # Create the prompt with custom or default personality
            if custom_prompt:
                # Use custom prompt instead of default personality
                custom_chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", custom_prompt),
                    ("system", "Context from YonEarth Podcast Episodes:\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                full_prompt = custom_chat_prompt.format_messages(
                    context=context + self._create_citation_reminder(),
                    chat_history=chat_history,
                    input=user_input
                )
            else:
                # Use default personality prompt
                full_prompt = self.prompt.format_messages(
                    context=context + self._create_citation_reminder(),
                    chat_history=chat_history,
                    input=user_input
                )
            
            # Generate response
            response = self.llm(full_prompt)
            
            # Save to memory
            self.memory.save_context(
                {"input": user_input},
                {"output": response.content}
            )
            
            # Extract episode citations
            citations = self._extract_citations(retrieved_docs or [])
            
            result = {
                "response": response.content,
                "personality": self.personality_variant,
                "citations": citations,
                "context_used": len(retrieved_docs) if retrieved_docs else 0,
                "session_id": session_id
            }
            
            logger.info(f"Generated response with {len(citations)} citations")
            return result
            
        except Exception as e:
            logger.error(f"Error generating Gaia response: {e}")
            return {
                "response": "I apologize, dear one, but I'm having trouble accessing the wisdom right now. Please try again in a moment.",
                "error": str(e),
                "personality": self.personality_variant,
                "citations": [],
                "context_used": 0
            }
    
    def _extract_citations(self, retrieved_docs: List[Any]) -> List[Dict[str, Any]]:
        """Extract citation information from retrieved documents with deduplication by episode"""
        citations = []
        seen_episodes = set()
        
        for doc in retrieved_docs:  # Check all retrieved docs for unique episodes
            metadata = getattr(doc, 'metadata', {})
            episode_number = metadata.get('episode_number', 'Unknown')
            
            # Skip if we've already seen this episode
            if episode_number in seen_episodes:
                continue
                
            seen_episodes.add(episode_number)
            
            citation = {
                "episode_number": episode_number,
                "title": metadata.get('title', 'Unknown Episode'),
                "guest_name": metadata.get('guest_name', 'Guest'),
                "url": metadata.get('url', ''),
                "relevance": "High"  # Could add scoring later
            }
            citations.append(citation)
            
            # Limit to top 3 unique episodes
            if len(citations) >= 3:
                break
        
        return citations
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Cleared Gaia conversation memory")
    
    def get_memory_summary(self) -> str:
        """Get summary of conversation history"""
        try:
            if hasattr(self.memory, 'buffer') and self.memory.buffer:
                return self.memory.buffer
            return "No conversation history yet."
        except:
            return "Unable to retrieve conversation summary."
    
    def switch_personality(self, new_variant: str):
        """Switch to a different personality variant"""
        from .gaia_personalities import get_available_personalities
        
        # Allow 'custom' personality variant
        if new_variant != 'custom' and new_variant not in get_available_personalities():
            raise ValueError(f"Unknown personality variant: {new_variant}")
        
        self.personality_variant = new_variant
        
        # Don't update personality_prompt or template for custom - it will be handled in generate_response
        if new_variant != 'custom':
            self.personality_prompt = get_personality(new_variant)
            
            # Update prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.personality_prompt),
                ("system", "Context from YonEarth Podcast Episodes:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
        
        logger.info(f"Switched to '{new_variant}' personality")


def create_gaia(personality_variant: Optional[str] = None) -> GaiaCharacter:
    """Factory function to create Gaia character"""
    return GaiaCharacter(personality_variant=personality_variant)


def main():
    """Test Gaia character functionality"""
    logging.basicConfig(level=logging.INFO)
    
    # Create Gaia character
    gaia = create_gaia("warm_mother")
    
    # Test basic response
    response = gaia.generate_response(
        "Tell me about regenerative agriculture",
        retrieved_docs=[]
    )
    
    print(f"\nGaia Response ({gaia.personality_variant}):")
    print(response["response"])
    print(f"\nCitations: {len(response['citations'])}")


if __name__ == "__main__":
    main()