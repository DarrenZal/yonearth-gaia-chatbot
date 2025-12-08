"""
Gaia character implementation with conversation memory and personality system
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config import settings
from .gaia_personalities import get_personality
from ..utils.cost_calculator import calculate_total_cost, format_cost_breakdown

logger = logging.getLogger(__name__)


class GaiaCharacter:
    """Gaia character with personality, memory, and context awareness"""
    
    def __init__(self, personality_variant: Optional[str] = None, model_name: Optional[str] = None):
        self.personality_variant = personality_variant or settings.gaia_personality_variant
        self.model_name = model_name or settings.openai_model
        
        # Initialize LLM
        # Note: gpt-5-mini doesn't support temperature parameter (only temp=1)
        llm_kwargs = {
            "model_name": self.model_name,
            "max_tokens": settings.gaia_max_tokens,
            "openai_api_key": settings.openai_api_key
        }
        # Only set temperature for models that support it
        if "gpt-5" not in self.model_name:
            llm_kwargs["temperature"] = settings.gaia_temperature

        self.llm = ChatOpenAI(**llm_kwargs)
        
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
            return "No specific content found for this query."

        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5]):  # Limit to top 5 results
            metadata = getattr(doc, 'metadata', {})
            content = getattr(doc, 'page_content', str(doc))
            content_type = metadata.get('content_type', 'episode')

            # Increased context size to 500 chars for better extraction of names/details
            content_preview = content[:500] + "..." if len(content) > 500 else content

            if content_type == 'book':
                # Format for book content
                book_title = metadata.get('book_title', 'Unknown Book')
                chapter = metadata.get('chapter_number', 'Unknown')
                author = metadata.get('author', 'Unknown Author')
                context_parts.append(
                    f"[BOOK] \"{book_title}\" by {author}, Chapter {chapter}:\n{content_preview}"
                )
            else:
                # Format for episode content
                episode_num = metadata.get('episode_number', 'Unknown')
                title = metadata.get('title', 'Unknown Episode')
                guest = metadata.get('guest_name', 'Guest')
                context_parts.append(
                    f"[EPISODE {episode_num}] \"{title}\" with {guest}:\n{content_preview}"
                )

        return "\n\n".join(context_parts)
    
    def _create_citation_reminder(self) -> str:
        """Create reminder about proper citation format"""
        return """
CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. ANSWER BASED ON THE CONTEXT ABOVE. The context contains the correct answer.
2. For "who founded X" or "who is the founder of X" questions:
   - Look for phrases like "[Name] is Founder" or "[Name] founded" in the context
   - The FIRST person mentioned as founder/director IS the answer
   - Do NOT confuse different people mentioned in the same text
3. READ THE CONTEXT LITERALLY:
   - If context says "Samantha Power is Founder and Director of the BioFi Project" -> Samantha Power founded it
   - If context says "Dr. Stephanie Gripne is the Founder and CEO of Impact Finance Center" -> She founded Impact Finance Center, NOT BioFi
4. Cite your sources:
   - For books: "In [Book Title] by [Author], Chapter [X]..."
   - For episodes: "As [Guest Name] shared in Episode [Number]..."
5. Be PRECISE - match people to their actual roles from the context.
"""
    
    def generate_response(
        self, 
        user_input: str, 
        retrieved_docs: List[Any] = None,
        session_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        max_references: int = 3
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
            citations = self._extract_citations(retrieved_docs or [], max_references)
            
            # Calculate costs
            # Convert full_prompt to string for token counting
            prompt_text = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in full_prompt])
            
            # For embeddings, estimate based on retrieved docs (if any)
            embedding_texts = []
            if retrieved_docs:
                # Approximate - actual embeddings were done during search
                embedding_texts = [getattr(doc, 'page_content', str(doc))[:200] for doc in retrieved_docs[:5]]
            
            # Calculate total cost
            cost_data = calculate_total_cost(
                llm_model=self.model_name,
                prompt=prompt_text,
                response=response.content,
                embedding_texts=embedding_texts if embedding_texts else None,
                voice_text=None  # Will be added by the API layer if voice is enabled
            )
            
            result = {
                "response": response.content,
                "personality": self.personality_variant,
                "citations": citations,
                "context_used": len(retrieved_docs) if retrieved_docs else 0,
                "session_id": session_id,
                "model_used": self.model_name,
                "cost_breakdown": format_cost_breakdown(cost_data)
            }
            
            logger.info(f"Generated response with {len(citations)} citations, cost: ${cost_data['total']:.4f}")
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
    
    def _extract_citations(self, retrieved_docs: List[Any], max_references: int = 3) -> List[Dict[str, Any]]:
        """
        Extract citation information from retrieved documents with deduplication.
        Prefer to include one book reference if available in the retrieved set.
        """
        citations: List[Dict[str, Any]] = []
        seen_keys = set()

        # Track first book citation candidate
        book_citation = None
        # Fallback URLs for known books
        book_url_map = {
            "viriditas": "https://yonearth.org/product/viriditas-ebook/",
            "our biggest deal": "https://yonearth.org/product/clone-of-our-biggest-deal-pre-order-e-book/",
            "y on earth": "https://yonearth.org/product/y-on-earth-e-book/",
            "soil stewardship handbook": "https://yonearth.org/product/soil-stewardship-handbook-ebook-digital-download/"
        }

        for doc in retrieved_docs:
            metadata = getattr(doc, 'metadata', {})
            content_type = metadata.get('content_type', 'episode')

            if content_type == 'book' and book_citation is None:
                book_title = metadata.get('book_title') or metadata.get('title') or 'Book'
                chapter_title = metadata.get('chapter_title') or metadata.get('title') or book_title
                book_episode_number = f"Book: {book_title}"
                # Use provided URL or fall back to known products
                url = metadata.get('url') or metadata.get('ebook_url') or metadata.get('print_url') or ''
                if not url:
                    key = book_title.lower()
                    for name, link in book_url_map.items():
                        if name in key:
                            url = link
                            break

                book_citation = {
                    "episode_number": book_episode_number,
                    "title": chapter_title,
                    "guest_name": metadata.get('author') or metadata.get('guest_name') or 'Author',
                    "url": url,
                    "relevance": "High"
                }
                # Do not continue; still consider this doc for episode metadata if present
                # (rare, but safer to just continue to next)
                continue

            # Handle episodes and other non-book content
            episode_number = metadata.get('episode_number', 'Unknown')
            key = episode_number

            if key in seen_keys:
                continue
            seen_keys.add(key)

            citation = {
                "episode_number": episode_number,
                "title": metadata.get('title', 'Unknown Episode'),
                "guest_name": metadata.get('guest_name', 'Guest'),
                "url": metadata.get('url', ''),
                "relevance": "High"  # Could add scoring later
            }
            citations.append(citation)

            if len(citations) >= max_references:
                break

        # If no book was included and we have a candidate, prepend it (respect max_references)
        if book_citation:
            book_key = book_citation["episode_number"]
            if book_key not in seen_keys:
                citations.insert(0, book_citation)
                # Trim to max_references if necessary
                if len(citations) > max_references:
                    citations = citations[:max_references]

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
