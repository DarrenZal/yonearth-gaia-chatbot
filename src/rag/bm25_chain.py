"""
Advanced BM25 Hybrid RAG Chain - New pipeline with BM25 integration
Keeps existing RAG pipeline intact while adding state-of-the-art BM25 hybrid search
"""
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from ..utils.lc_compat import Document

from ..config import settings
from ..character.gaia import GaiaCharacter
from .vectorstore import YonEarthVectorStore, create_vectorstore
from .bm25_hybrid_retriever import BM25HybridRetriever
from ..ingestion.process_episodes import process_episodes_for_ingestion

logger = logging.getLogger(__name__)


def load_book_metadata() -> Dict[str, Dict[str, Any]]:
    """Load book metadata from JSON files"""
    book_metadata = {}
    books_dir = "/root/yonearth-gaia-chatbot/data/books"
    
    if os.path.exists(books_dir):
        for book_folder in os.listdir(books_dir):
            metadata_path = os.path.join(books_dir, book_folder, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        book_title = metadata.get('title', book_folder)
                        book_metadata[book_title] = metadata
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {book_folder}: {e}")
    
    return book_metadata


class BM25RAGChain:
    """
    Advanced RAG chain with BM25 hybrid search
    
    Features:
    - BM25 keyword search + semantic vector search
    - Reciprocal Rank Fusion for result combination
    - Cross-encoder reranking for improved relevance
    - Query-adaptive search strategy
    - Comparison metrics with original RAG chain
    """
    
    def __init__(self, initialize_data: bool = False):
        self.vectorstore = None
        self.bm25_retriever = None
        self.gaia = None
        self.is_initialized = False
        self.book_metadata = load_book_metadata()  # Load book metadata once
        self.search_stats = {
            'total_queries': 0,
            'bm25_queries': 0,
            'semantic_queries': 0,
            'hybrid_queries': 0,
            'reranked_queries': 0
        }
        
        if initialize_data:
            self.initialize()
    
    def initialize(self, recreate_index: bool = False):
        """Initialize the BM25 RAG chain with data and components"""
        logger.info("Initializing BM25 Hybrid RAG chain...")
        
        try:
            # Step 1: Process episodes if vectorstore is empty or recreating
            documents = None
            if recreate_index:
                logger.info("Processing episodes for vector database...")
                documents = process_episodes_for_ingestion()
            
            # Step 2: Create/connect to vectorstore
            logger.info("Setting up vector database...")
            self.vectorstore = create_vectorstore(
                documents=documents,
                recreate_index=recreate_index
            )
            
            # Step 3: Initialize BM25 hybrid retriever
            logger.info("Setting up BM25 hybrid retriever...")
            self.bm25_retriever = BM25HybridRetriever(self.vectorstore)
            
            # Step 4: Initialize Gaia character
            logger.info("Setting up Gaia character...")
            self.gaia = GaiaCharacter()
            
            self.is_initialized = True
            logger.info("BM25 RAG chain initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing BM25 RAG chain: {e}")
            raise
    
    def chat(
        self,
        message: str,
        search_method: str = "auto",  # "auto", "bm25", "semantic", "hybrid"
        k: int = 5,
        include_sources: bool = True,
        custom_prompt: Optional[str] = None,
        max_citations: int = 3,
        category_threshold: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat message using BM25 hybrid RAG
        
        Args:
            message: User's question/message
            search_method: Search strategy ("auto", "bm25", "semantic", "hybrid")
            k: Number of documents to retrieve
            include_sources: Whether to include source citations
            **kwargs: Additional parameters for Gaia
        
        Returns:
            Dict containing response, sources, metadata, and performance metrics
        """
        if not self.is_initialized:
            raise RuntimeError("BM25 RAG chain not initialized. Call initialize() first.")
        
        logger.info(f"Processing chat message: {message[:50]}...")
        
        # Update stats
        self.search_stats['total_queries'] += 1
        
        try:
            # Step 1: Retrieve relevant documents
            if search_method == "auto":
                # Use query analysis to determine best method
                query_analysis = self.bm25_retriever.analyze_query(message)
                search_method = query_analysis['suggested_method']
            
            documents = self._retrieve_documents(message, search_method, k, category_threshold)
            
            # Step 2: Handle model and personality selection
            personality_variant = kwargs.get('personality_variant')
            model_name = kwargs.get('model_name')
            
            # Create appropriate Gaia instance
            gaia_instance = self.gaia
            if model_name and model_name != self.gaia.model_name:
                # Create new Gaia instance with different model
                logger.info(f"Creating new Gaia instance with model: {model_name}")
                gaia_instance = GaiaCharacter(
                    personality_variant=personality_variant or self.gaia.personality_variant,
                    model_name=model_name
                )
            elif personality_variant and personality_variant != self.gaia.personality_variant:
                if personality_variant != 'custom' or custom_prompt is None:
                    self.gaia.switch_personality(personality_variant)
            
            # Step 3: Generate response using Gaia
            response_data = gaia_instance.generate_response(
                user_input=message,
                retrieved_docs=documents,
                session_id=kwargs.get('session_id'),
                custom_prompt=custom_prompt
            )
            
            # Step 4: Add BM25-specific metadata
            response_data.update({
                'search_method_used': search_method,
                'documents_retrieved': len(documents),
                'bm25_stats': self.bm25_retriever.get_stats(),
                'model_used': model_name or self.gaia.model_name,
                'performance_stats': self.search_stats.copy()
            })
            
            # Step 5: Add sources if requested
            if include_sources:
                response_data['sources'] = self._format_sources(documents, max_citations)
                response_data['episode_references'] = self._extract_episode_references(documents)
            
            logger.info(f"BM25 RAG response generated successfully using {search_method} search")
            return response_data
            
        except Exception as e:
            logger.error(f"Error in BM25 chat processing: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your question. Please try again.",
                'error': str(e),
                'search_method_used': search_method,
                'success': False
            }
    
    def _retrieve_documents(
        self, 
        query: str, 
        search_method: str, 
        k: int,
        category_threshold: float = 0.7
    ) -> List[Document]:
        """Retrieve documents using specified search method"""
        
        if search_method == "bm25":
            # Pure BM25 search
            results = self.bm25_retriever.bm25_search(query, k=k)
            documents = [doc for doc, score in results]
            self.search_stats['bm25_queries'] += 1
            
        elif search_method == "semantic":
            # Pure semantic search
            results = self.bm25_retriever.semantic_search(query, k=k)
            documents = [doc for doc, score in results]
            self.search_stats['semantic_queries'] += 1
            
        elif search_method in ["hybrid", "keyword_heavy", "semantic_heavy"]:
            # Full hybrid search with RRF and reranking
            documents = self.bm25_retriever.hybrid_search(query, k=k, category_threshold=category_threshold)
            self.search_stats['hybrid_queries'] += 1
            if self.bm25_retriever.use_reranker:
                self.search_stats['reranked_queries'] += 1
        
        else:
            # Fallback to hybrid
            documents = self.bm25_retriever.hybrid_search(query, k=k, category_threshold=category_threshold)
            self.search_stats['hybrid_queries'] += 1
        
        logger.info(f"Retrieved {len(documents)} documents using {search_method} method")
        return documents
    
    def _format_sources(self, documents: List[Document], max_citations: int = 3) -> List[Dict[str, Any]]:
        """Format source citations from retrieved documents with deduplication by episode/book"""
        sources = []
        seen_items = set()
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            content_type = metadata.get('content_type', 'episode')
            
            # Initialize variables for both book and episode content
            book_title = metadata.get('book_title', 'Unknown Book')
            chapter_int = 1  # Default chapter
            
            # Determine unique ID and fix chapter numbers for books
            if content_type == 'book':
                chapter_num = metadata.get('chapter_number', 'Unknown')
                
                # For VIRIDITAS book, map from chapter_number field which contains page numbers
                if book_title == 'VIRIDITAS: THE GREAT HEALING' and chapter_num != 'Unknown':
                    # chapter_number field contains page numbers for VIRIDITAS book
                    try:
                        chunk_num = int(float(chapter_num))  # Convert page number to int
                        
                        # Map chunk numbers to actual chapters based on table of contents
                        chapter_ranges = [
                            (0, 10, 0),      # Prelude
                            (11, 21, 1),     # Chapter 1: Urban Cacophony
                            (22, 33, 2),     # Chapter 2: Terror: A Deadly Chase
                            (34, 45, 3),     # Chapter 3: Taking Flight
                            (46, 60, 4),     # Chapter 4: Temple of the Apocalypse
                            (61, 71, 5),     # Chapter 5: Cresting the Horizon
                            (72, 82, 6),     # Chapter 6: Rendezvous with a Stranger
                            (83, 104, 7),    # Chapter 7: A Bizarre Sanctuary
                            (105, 133, 8),   # Chapter 8: Alpine Village
                            (134, 147, 9),   # Chapter 9: Securus Locus: Trust Nobody
                            (148, 165, 10),  # Chapter 10: Mesa Laboratory
                            (166, 170, 11),  # Chapter 11: A Mysterious Billionaire
                            (171, 190, 12),  # Chapter 12: Airborne
                            (191, 197, 13),  # Chapter 13: Billionaires & Bicycles
                            (198, 208, 14),  # Chapter 14: Respite at the Farm
                            (209, 216, 15),  # Chapter 15: The Garden
                            (217, 236, 16),  # Chapter 16: What Is Really Possible?
                            (237, 248, 17),  # Chapter 17: Superorganism
                            (249, 254, 18),  # Chapter 18: Wi Magua
                            (255, 267, 19),  # Chapter 19: The Great Darkness
                            (268, 272, 20),  # Chapter 20: From the Ashes
                            (273, 288, 21),  # Chapter 21: Spiral of No Return
                            (289, 300, 22),  # Chapter 22: Into the Wilderness
                            (301, 320, 23),  # Chapter 23: The Cave
                            (321, 338, 24),  # Chapter 24: Winter Solitude—Pregnant at the Hearth
                            (339, 351, 25),  # Chapter 25: Mountain Side Terror
                            (352, 354, 26),  # Chapter 26: Otto Awakens
                            (355, 391, 27),  # Chapter 27: A Walk Through History
                            (392, 407, 28),  # Chapter 28: The Ubiquity
                            (408, 428, 29),  # Chapter 29: Otto's Revelation
                            (429, 493, 30),  # Chapter 30: Gaia Speaks
                            (494, 495, 31),  # Chapter 31: A Joyful Journey
                            (496, 520, 32),  # Chapter 32: Birthing a New World—Water of Life
                            (521, 568, 33),  # Chapter 33: Weaving A New Culture Together
                        ]
                        
                        # Find which chapter this chunk belongs to
                        chapter_int = 1  # Default
                        for start_page, end_page, chapter_num in chapter_ranges:
                            if start_page <= chunk_num <= end_page:
                                chapter_int = max(1, chapter_num)  # Ensure minimum chapter 1
                                break
                        
                        # If chunk number is very high, estimate based on position
                        if chunk_num > 568:
                            chapter_int = 33  # Last chapter
                            
                    except (ValueError, IndexError):
                        chapter_int = 1  # Default to chapter 1 if parsing fails
                else:
                    # For non-VIRIDITAS books or if parsing fails, use original chapter number
                    if chapter_num and chapter_num != 'Unknown':
                        try:
                            chapter_int = int(float(chapter_num))
                        except (ValueError, TypeError):
                            chapter_int = 1
                
                # Create unique ID with corrected chapter number
                unique_id = f"book_{book_title}_ch{chapter_int}"
            else:
                episode_number = metadata.get('episode_number', 'Unknown')
                unique_id = f"episode_{episode_number}"
            
            # Skip if we've already seen this item
            if unique_id in seen_items:
                continue
                
            seen_items.add(unique_id)
            
            # Format source based on content type
            if content_type == 'book':
                # Use variables already calculated in the deduplication section above
                # book_title and chapter_int are already set
                chapter_title = metadata.get('chapter_title', '')
                author = metadata.get('author', 'Unknown Author')
                
                # Get URLs from metadata (default to ebook)
                book_url = ''
                audiobook_url = ''
                print_url = ''
                if book_title in self.book_metadata:
                    book_meta = self.book_metadata[book_title]
                    book_url = book_meta.get('ebook_url', '')  # Default to ebook
                    audiobook_url = book_meta.get('audiobook_url', '')
                    print_url = book_meta.get('print_url', '')
                
                # Create episode-compatible format
                source = {
                    'content_type': 'book',  # Keep internal distinction
                    'episode_id': unique_id,
                    'episode_number': f"Book: {book_title}",  # Show as book identifier
                    'title': f"Chapter {chapter_int}",
                    'guest_name': author,
                    'url': book_url,  # Default to ebook URL
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    # Keep book-specific fields for internal use
                    'book_title': book_title,
                    'author': author,
                    'chapter_number': chapter_int,  # Convert to int for API
                    'chapter_title': chapter_title,
                    # Add all URL options
                    'ebook_url': book_url,
                    'audiobook_url': audiobook_url,
                    'print_url': print_url
                }
            else:
                episode_number = metadata.get('episode_number')
                # Skip synthetic or non-episode docs that lack a real episode_number
                # (or use placeholders like "Unknown") to avoid confusing
                # "Episode unknown" references in the UI.
                if not episode_number:
                    continue
                if str(episode_number).strip().lower() == "unknown":
                    continue

                source = {
                    'content_type': 'episode',
                    'episode_id': metadata.get('episode_id', 'Unknown'),
                    'episode_number': str(episode_number),
                    'title': metadata.get('title', 'Unknown Title'),
                    'guest_name': metadata.get('guest_name', 'Unknown Guest'),
                    'url': metadata.get('url', ''),
                    'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
            
            # Add BM25-specific scores if available
            if 'keyword_component' in metadata:
                source['keyword_score'] = metadata['keyword_component']
            if 'semantic_component' in metadata:
                source['semantic_score'] = metadata['semantic_component']
            if 'final_score' in metadata:
                source['final_score'] = metadata['final_score']
            
            sources.append(source)
            
            # Limit to configured number of unique items
            if len(sources) >= max_citations:
                break
        
        return sources
    
    def _extract_episode_references(self, documents: List[Document]) -> List[str]:
        """Extract unique episode/book references from documents"""
        references = set()
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            content_type = metadata.get('content_type', 'episode')
            
            if content_type == 'book':
                book_title = metadata.get('book_title')
                chapter_number = metadata.get('chapter_number')
                if book_title:
                    # Use the same chapter correction logic as _format_sources
                    chapter_int = 1  # Default chapter
                    
                    # For VIRIDITAS book, map from chapter_number field which contains page numbers
                    if book_title == 'VIRIDITAS: THE GREAT HEALING' and chapter_number is not None:
                        # chapter_number field contains page numbers for VIRIDITAS book
                        try:
                            chunk_num = int(float(chapter_number))  # Convert page number to int
                            
                            # Map chunk numbers to actual chapters based on table of contents
                            chapter_ranges = [
                                (0, 10, 0),      # Prelude
                                (11, 21, 1),     # Chapter 1: Urban Cacophony
                                (22, 33, 2),     # Chapter 2: Terror: A Deadly Chase
                                (34, 45, 3),     # Chapter 3: Taking Flight
                                (46, 60, 4),     # Chapter 4: Temple of the Apocalypse
                                (61, 71, 5),     # Chapter 5: Cresting the Horizon
                                (72, 82, 6),     # Chapter 6: Rendezvous with a Stranger
                                (83, 104, 7),    # Chapter 7: A Bizarre Sanctuary
                                (105, 133, 8),   # Chapter 8: Alpine Village
                                (134, 147, 9),   # Chapter 9: Securus Locus: Trust Nobody
                                (148, 165, 10),  # Chapter 10: Mesa Laboratory
                                (166, 170, 11),  # Chapter 11: A Mysterious Billionaire
                                (171, 190, 12),  # Chapter 12: Airborne
                                (191, 197, 13),  # Chapter 13: Billionaires & Bicycles
                                (198, 208, 14),  # Chapter 14: Respite at the Farm
                                (209, 216, 15),  # Chapter 15: The Garden
                                (217, 236, 16),  # Chapter 16: What Is Really Possible?
                                (237, 248, 17),  # Chapter 17: Superorganism
                                (249, 254, 18),  # Chapter 18: Wi Magua
                                (255, 267, 19),  # Chapter 19: The Great Darkness
                                (268, 272, 20),  # Chapter 20: From the Ashes
                                (273, 288, 21),  # Chapter 21: Spiral of No Return
                                (289, 300, 22),  # Chapter 22: Into the Wilderness
                                (301, 320, 23),  # Chapter 23: The Cave
                                (321, 338, 24),  # Chapter 24: Winter Solitude—Pregnant at the Hearth
                                (339, 351, 25),  # Chapter 25: Mountain Side Terror
                                (352, 354, 26),  # Chapter 26: Otto Awakens
                                (355, 391, 27),  # Chapter 27: A Walk Through History
                                (392, 407, 28),  # Chapter 28: The Ubiquity
                                (408, 428, 29),  # Chapter 29: Otto's Revelation
                                (429, 493, 30),  # Chapter 30: Gaia Speaks
                                (494, 495, 31),  # Chapter 31: A Joyful Journey
                                (496, 520, 32),  # Chapter 32: Birthing a New World—Water of Life
                                (521, 568, 33),  # Chapter 33: Weaving A New Culture Together
                            ]
                            
                            # Find which chapter this chunk belongs to
                            for start_page, end_page, chapter_num in chapter_ranges:
                                if start_page <= chunk_num <= end_page:
                                    chapter_int = max(1, chapter_num)  # Ensure minimum chapter 1
                                    break
                            
                            # If chunk number is very high, estimate based on position
                            if chunk_num > 568:
                                chapter_int = 33  # Last chapter
                                
                        except (ValueError, IndexError):
                            chapter_int = 1  # Default to chapter 1 if parsing fails
                    else:
                        # For non-VIRIDITAS books or if parsing fails, use original chapter number
                        if chapter_number and chapter_number != 'Unknown':
                            try:
                                chapter_int = int(float(chapter_number))
                            except (ValueError, TypeError):
                                chapter_int = 1
                    
                    references.add(f"Book: {book_title} - Chapter {chapter_int}")
            else:
                episode_id = metadata.get('episode_id')
                episode_number = metadata.get('episode_number')
                
                if episode_id:
                    references.add(str(episode_id))
                elif episode_number:
                    references.add(str(episode_number))
        
        return sorted(list(references))
    
    def search_episodes(
        self, 
        query: str, 
        k: int = 10,
        search_method: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search for episodes using BM25 hybrid search
        
        Args:
            query: Search query
            k: Number of results to return
            search_method: Search method to use
        
        Returns:
            List of episode information with relevance scores
        """
        if not self.is_initialized:
            raise RuntimeError("BM25 RAG chain not initialized")
        
        documents = self._retrieve_documents(query, search_method, k)
        
        # Group results by episode or book
        content_groups = {}
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            content_type = metadata.get('content_type', 'episode')
            
            # Create unique ID based on content type
            if content_type == 'book':
                book_title = metadata.get('book_title', 'unknown')
                chapter_number = metadata.get('chapter_number', 0)
                
                # Apply chapter correction logic for VIRIDITAS book
                chapter_int = 1  # Default chapter
                if book_title == 'VIRIDITAS: THE GREAT HEALING' and chapter_number is not None:
                    # chapter_number field contains page numbers for VIRIDITAS book
                    try:
                        chunk_num = int(float(chapter_num))  # Convert page number to int
                        
                        # Map chunk numbers to actual chapters based on table of contents
                        chapter_ranges = [
                            (0, 10, 0),      # Prelude
                            (11, 21, 1),     # Chapter 1: Urban Cacophony
                            (22, 33, 2),     # Chapter 2: Terror: A Deadly Chase
                            (34, 45, 3),     # Chapter 3: Taking Flight
                            (46, 60, 4),     # Chapter 4: Temple of the Apocalypse
                            (61, 71, 5),     # Chapter 5: Cresting the Horizon
                            (72, 82, 6),     # Chapter 6: Rendezvous with a Stranger
                            (83, 104, 7),    # Chapter 7: A Bizarre Sanctuary
                            (105, 133, 8),   # Chapter 8: Alpine Village
                            (134, 147, 9),   # Chapter 9: Securus Locus: Trust Nobody
                            (148, 165, 10),  # Chapter 10: Mesa Laboratory
                            (166, 170, 11),  # Chapter 11: A Mysterious Billionaire
                            (171, 190, 12),  # Chapter 12: Airborne
                            (191, 197, 13),  # Chapter 13: Billionaires & Bicycles
                            (198, 208, 14),  # Chapter 14: Respite at the Farm
                            (209, 216, 15),  # Chapter 15: The Garden
                            (217, 236, 16),  # Chapter 16: What Is Really Possible?
                            (237, 248, 17),  # Chapter 17: Superorganism
                            (249, 254, 18),  # Chapter 18: Wi Magua
                            (255, 267, 19),  # Chapter 19: The Great Darkness
                            (268, 272, 20),  # Chapter 20: From the Ashes
                            (273, 288, 21),  # Chapter 21: Spiral of No Return
                            (289, 300, 22),  # Chapter 22: Into the Wilderness
                            (301, 320, 23),  # Chapter 23: The Cave
                            (321, 338, 24),  # Chapter 24: Winter Solitude—Pregnant at the Hearth
                            (339, 351, 25),  # Chapter 25: Mountain Side Terror
                            (352, 354, 26),  # Chapter 26: Otto Awakens
                            (355, 391, 27),  # Chapter 27: A Walk Through History
                            (392, 407, 28),  # Chapter 28: The Ubiquity
                            (408, 428, 29),  # Chapter 29: Otto's Revelation
                            (429, 493, 30),  # Chapter 30: Gaia Speaks
                            (494, 495, 31),  # Chapter 31: A Joyful Journey
                            (496, 520, 32),  # Chapter 32: Birthing a New World—Water of Life
                            (521, 568, 33),  # Chapter 33: Weaving A New Culture Together
                        ]
                        
                        # Find which chapter this chunk belongs to
                        for start_page, end_page, chapter_num in chapter_ranges:
                            if start_page <= chunk_num <= end_page:
                                chapter_int = max(1, chapter_num)  # Ensure minimum chapter 1
                                break
                        
                        # If chunk number is very high, estimate based on position
                        if chunk_num > 568:
                            chapter_int = 33  # Last chapter
                            
                    except (ValueError, IndexError):
                        chapter_int = 1  # Default to chapter 1 if parsing fails
                else:
                    # For non-VIRIDITAS books or if parsing fails, use original chapter number
                    if chapter_num and chapter_num != 'Unknown':
                        try:
                            chapter_int = int(float(chapter_num))
                        except (ValueError, TypeError):
                            chapter_int = 1
                
                content_id = f"book_{book_title}_ch{chapter_int}"
            else:
                content_id = metadata.get('episode_id', 'unknown')
            
            if content_id not in content_groups:
                if content_type == 'book':
                    content_groups[content_id] = {
                        'content_type': 'book',
                        'content_id': content_id,
                        'book_title': metadata.get('book_title', 'Unknown Book'),
                        'author': metadata.get('author', 'Unknown Author'),
                        'chapter_number': chapter_int,  # Use corrected chapter number
                        'chapter_title': metadata.get('chapter_title', ''),
                        'title': f"{metadata.get('book_title', 'Unknown Book')} - Chapter {chapter_int}",  # Use corrected chapter number
                        'chunks': [],
                        'max_score': 0
                    }
                else:
                    content_groups[content_id] = {
                        'content_type': 'episode',
                        'episode_id': content_id,
                        'episode_number': metadata.get('episode_number', 'Unknown'),
                        'title': metadata.get('title', 'Unknown Title'),
                        'url': metadata.get('url', ''),
                        'chunks': [],
                        'max_score': 0
                    }
            
            chunk_info = {
                'content': doc.page_content,
                'score': metadata.get('final_score', 0)
            }
            
            content_groups[content_id]['chunks'].append(chunk_info)
            content_groups[content_id]['max_score'] = max(
                content_groups[content_id]['max_score'], 
                chunk_info['score']
            )
        
        # Sort by relevance score
        results = sorted(
            content_groups.values(), 
            key=lambda x: x['max_score'], 
            reverse=True
        )
        
        return results
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance statistics for comparison with original RAG"""
        stats = self.search_stats.copy()
        
        if stats['total_queries'] > 0:
            stats['bm25_percentage'] = (stats['bm25_queries'] / stats['total_queries']) * 100
            stats['semantic_percentage'] = (stats['semantic_queries'] / stats['total_queries']) * 100
            stats['hybrid_percentage'] = (stats['hybrid_queries'] / stats['total_queries']) * 100
            stats['reranking_percentage'] = (stats['reranked_queries'] / stats['total_queries']) * 100
        
        stats.update(self.bm25_retriever.get_stats())
        
        return stats
    
    def compare_search_methods(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Compare different search methods for the same query
        Useful for A/B testing and performance analysis
        """
        if not self.is_initialized:
            raise RuntimeError("BM25 RAG chain not initialized")
        
        comparison = {
            'query': query,
            'methods': {}
        }
        
        methods = ["bm25", "semantic", "hybrid"]
        
        for method in methods:
            try:
                documents = self._retrieve_documents(query, method, k)
                
                comparison['methods'][method] = {
                    'documents_count': len(documents),
                    'content_referenced': self._extract_episode_references(documents),
                    'top_results': [
                        self._format_comparison_result(doc)
                        for doc in documents[:3]
                    ]
                }
            except Exception as e:
                comparison['methods'][method] = {
                    'error': str(e)
                }
        
        return comparison
    
    def _format_comparison_result(self, doc: Document) -> Dict[str, Any]:
        """Format a document for comparison display"""
        metadata = getattr(doc, 'metadata', {})
        content_type = metadata.get('content_type', 'episode')
        
        if content_type == 'book':
            book_title = metadata.get('book_title', 'Unknown')
            chapter_number = metadata.get('chapter_number', 0)
            
            # Apply chapter correction logic for VIRIDITAS book
            chapter_int = 1  # Default chapter
            if book_title == 'VIRIDITAS: THE GREAT HEALING' and chapter_number is not None:
                # chapter_number field contains page numbers for VIRIDITAS book
                try:
                    chunk_num = int(float(chapter_number))  # Convert page number to int
                    
                    # Map chunk numbers to actual chapters based on table of contents
                    chapter_ranges = [
                        (0, 10, 0),      # Prelude
                        (11, 21, 1),     # Chapter 1: Urban Cacophony
                        (22, 33, 2),     # Chapter 2: Terror: A Deadly Chase
                        (34, 45, 3),     # Chapter 3: Taking Flight
                        (46, 60, 4),     # Chapter 4: Temple of the Apocalypse
                        (61, 71, 5),     # Chapter 5: Cresting the Horizon
                        (72, 82, 6),     # Chapter 6: Rendezvous with a Stranger
                        (83, 104, 7),    # Chapter 7: A Bizarre Sanctuary
                        (105, 133, 8),   # Chapter 8: Alpine Village
                        (134, 147, 9),   # Chapter 9: Securus Locus: Trust Nobody
                        (148, 165, 10),  # Chapter 10: Mesa Laboratory
                        (166, 170, 11),  # Chapter 11: A Mysterious Billionaire
                        (171, 190, 12),  # Chapter 12: Airborne
                        (191, 197, 13),  # Chapter 13: Billionaires & Bicycles
                        (198, 208, 14),  # Chapter 14: Respite at the Farm
                        (209, 216, 15),  # Chapter 15: The Garden
                        (217, 236, 16),  # Chapter 16: What Is Really Possible?
                        (237, 248, 17),  # Chapter 17: Superorganism
                        (249, 254, 18),  # Chapter 18: Wi Magua
                        (255, 267, 19),  # Chapter 19: The Great Darkness
                        (268, 272, 20),  # Chapter 20: From the Ashes
                        (273, 288, 21),  # Chapter 21: Spiral of No Return
                        (289, 300, 22),  # Chapter 22: Into the Wilderness
                        (301, 320, 23),  # Chapter 23: The Cave
                        (321, 338, 24),  # Chapter 24: Winter Solitude—Pregnant at the Hearth
                        (339, 351, 25),  # Chapter 25: Mountain Side Terror
                        (352, 354, 26),  # Chapter 26: Otto Awakens
                        (355, 391, 27),  # Chapter 27: A Walk Through History
                        (392, 407, 28),  # Chapter 28: The Ubiquity
                        (408, 428, 29),  # Chapter 29: Otto's Revelation
                        (429, 493, 30),  # Chapter 30: Gaia Speaks
                        (494, 495, 31),  # Chapter 31: A Joyful Journey
                        (496, 520, 32),  # Chapter 32: Birthing a New World—Water of Life
                        (521, 568, 33),  # Chapter 33: Weaving A New Culture Together
                    ]
                    
                    # Find which chapter this chunk belongs to
                    for start_page, end_page, chapter_num in chapter_ranges:
                        if start_page <= chunk_num <= end_page:
                            chapter_int = max(1, chapter_num)  # Ensure minimum chapter 1
                            break
                    
                    # If chunk number is very high, estimate based on position
                    if chunk_num > 568:
                        chapter_int = 33  # Last chapter
                        
                except (ValueError, IndexError):
                    chapter_int = 1  # Default to chapter 1 if parsing fails
            else:
                # For non-VIRIDITAS books or if parsing fails, use original chapter number
                if chapter_number and chapter_number != 'Unknown':
                    try:
                        chapter_int = int(float(chapter_number))
                    except (ValueError, TypeError):
                        chapter_int = 1
            
            return {
                'content_type': 'book',
                'id': f"book_{book_title}_ch{chapter_int}",
                'title': f"{book_title} - Chapter {chapter_int}",
                'author': metadata.get('author', 'Unknown Author'),
                'preview': doc.page_content[:100] + "..."
            }
        else:
            return {
                'content_type': 'episode',
                'episode_id': metadata.get('episode_id', 'Unknown'),
                'title': metadata.get('title', 'Unknown'),
                'preview': doc.page_content[:100] + "..."
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health status of BM25 RAG chain"""
        return {
            'initialized': self.is_initialized,
            'vectorstore_available': self.vectorstore is not None,
            'bm25_retriever_available': self.bm25_retriever is not None,
            'gaia_available': self.gaia is not None,
            'bm25_index_ready': self.bm25_retriever.bm25 is not None if self.bm25_retriever else False,
            'reranker_available': self.bm25_retriever.reranker is not None if self.bm25_retriever else False,
            'performance_stats': self.search_stats,
            'component_stats': self.bm25_retriever.get_stats() if self.bm25_retriever else {}
        }


def main():
    """Test BM25 RAG chain functionality"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create and initialize BM25 RAG chain
    logger.info("Creating BM25 RAG chain...")
    bm25_chain = BM25RAGChain(initialize_data=True)
    
    # Test chat functionality
    test_queries = [
        "what is biochar and how is it used?",
        "tell me about regenerative agriculture practices",
        "episode 147 permaculture techniques",
        "how can I start composting at home?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: '{query}'")
        print(f"{'='*60}")
        
        # Test different search methods
        for method in ["bm25", "semantic", "hybrid"]:
            print(f"\n--- {method.upper()} METHOD ---")
            
            try:
                response = bm25_chain.chat(
                    message=query,
                    search_method=method,
                    k=3,
                    include_sources=True
                )
                
                print(f"Response: {response.get('response', 'No response')[:200]}...")
                print(f"Episodes referenced: {response.get('episode_references', [])}")
                print(f"Search method used: {response.get('search_method_used')}")
                
            except Exception as e:
                print(f"Error with {method}: {e}")
    
    # Show performance statistics
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    stats = bm25_chain.get_performance_comparison()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Health check
    print(f"\n{'='*60}")
    print("HEALTH CHECK")
    print(f"{'='*60}")
    
    health = bm25_chain.health_check()
    for key, value in health.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
