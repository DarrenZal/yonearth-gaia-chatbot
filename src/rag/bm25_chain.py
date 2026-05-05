"""
Advanced BM25 Hybrid RAG Chain - New pipeline with BM25 integration
Keeps existing RAG pipeline intact while adding state-of-the-art BM25 hybrid search
"""
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document

from ..config import settings
from ..character.gaia import GaiaCharacter
from .vectorstore import YonEarthVectorStore, create_vectorstore
from .bm25_hybrid_retriever import BM25HybridRetriever
from ..ingestion.process_episodes import process_episodes_for_ingestion

logger = logging.getLogger(__name__)


def _merge_unique(
    doc_lists: List[List[Document]],
    cap: int,
    weights: Optional[List[int]] = None,
) -> List[Document]:
    """Weighted round-robin merge of document lists, dedup by chunk identity.

    Each round pulls `weights[i]` items from `doc_lists[i]` (default 1 each)
    so a higher-weighted list dominates the output. This matters because
    theme-saturation should dominate theme queries (it's the whole point) but
    hybrid_search must still contribute high-relevance non-theme hits like
    resource pages (Soil Werks, Wele Waters).

    Default behavior with `weights=None` is even round-robin — same as before.
    For theme-aware retrieval, callers pass weights like [2, 1] to give
    theme-saturation 2× hybrid's slot share.

    Identity falls back through chunk_id → episode_id+content_prefix → content
    hash so theme-saturation chunks aren't re-added by hybrid_search.
    """
    def _ident(doc: Document) -> str:
        md = getattr(doc, 'metadata', {}) or {}
        return (
            md.get('chunk_id')
            or md.get('id')
            or f"ep{md.get('episode_number')}|{(doc.page_content or '')[:80]}"
        )

    if weights is None:
        weights = [1] * len(doc_lists)
    assert len(weights) == len(doc_lists)

    seen: set[str] = set()
    out: List[Document] = []
    iters = [iter(lst) for lst in doc_lists]
    list_done = [False] * len(iters)
    while not all(list_done) and len(out) < cap:
        progressed = False
        for i, it in enumerate(iters):
            if list_done[i]:
                continue
            for _ in range(max(1, weights[i])):
                try:
                    while True:
                        doc = next(it)
                        ident = _ident(doc)
                        if ident in seen:
                            continue
                        seen.add(ident)
                        out.append(doc)
                        progressed = True
                        if len(out) >= cap:
                            return out
                        break
                except StopIteration:
                    list_done[i] = True
                    break
        if not progressed:
            break
    return out


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
        max_citations: int = 5,
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
                custom_prompt=custom_prompt,
                mentioned_episodes=kwargs.get('mentioned_episodes')
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
                response_data['sources'] = self._format_sources(documents, max_citations, query=message)
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
        category_threshold: float = 0.55
    ) -> List[Document]:
        """Retrieve documents using specified search method.

        For methods that aren't pure BM25 / pure semantic, we run an additional
        theme-saturation pass first (Aaron's #2 issue, 2026-05-02): if the
        query matches a YOE theme via taxonomy, ensure every theme-tagged
        episode and book contributes at least one chunk. The hybrid result is
        then merged on top, deduped by chunk identity.
        """

        if search_method == "bm25":
            results = self.bm25_retriever.bm25_search(query, k=k)
            documents = [doc for doc, score in results]
            self.search_stats['bm25_queries'] += 1

        elif search_method == "semantic":
            results = self.bm25_retriever.semantic_search(query, k=k)
            documents = [doc for doc, score in results]
            self.search_stats['semantic_queries'] += 1

        elif search_method in ["hybrid", "keyword_heavy", "semantic_heavy", "category_heavy"]:
            theme_docs = self.bm25_retriever.theme_saturation_search(
                query, category_threshold=category_threshold, max_chunks=k * 3
            )
            hybrid_docs = self.bm25_retriever.hybrid_search(
                query, k=k, category_threshold=category_threshold
            )
            # Direct semantic top-K is a third channel: hybrid's category-first
            # fusion buries non-category content (like resource_soilwerks /
            # resource_welewaters which aren't theme-tagged), so we pull the
            # raw vector-similarity top-K alongside as a safety net for named
            # resources and proper nouns.
            semantic_docs = [
                doc for doc, _ in self.bm25_retriever.semantic_search(query, k=k)
            ]
            documents = _merge_unique(
                [theme_docs, hybrid_docs, semantic_docs],
                cap=max(k * 2, 12),
                # Theme dominates theme queries (it's the whole point), but we
                # still always sample 1 from each of hybrid + semantic per round.
                weights=[2, 1, 1] if theme_docs else [1, 1, 1],
            )
            self.search_stats['hybrid_queries'] += 1
            if self.bm25_retriever.use_reranker:
                self.search_stats['reranked_queries'] += 1
            logger.info(
                f"Theme-saturation contributed {len(theme_docs)} docs; "
                f"hybrid contributed {len(hybrid_docs)}; merged to {len(documents)}"
            )

        else:
            # Fallback: same merge behavior under the default method
            theme_docs = self.bm25_retriever.theme_saturation_search(
                query, category_threshold=category_threshold, max_chunks=k * 3
            )
            hybrid_docs = self.bm25_retriever.hybrid_search(
                query, k=k, category_threshold=category_threshold
            )
            semantic_docs = [
                doc for doc, _ in self.bm25_retriever.semantic_search(query, k=k)
            ]
            documents = _merge_unique(
                [theme_docs, hybrid_docs, semantic_docs],
                cap=max(k * 2, 12),
                weights=[2, 1, 1] if theme_docs else [1, 1, 1],
            )
            self.search_stats['hybrid_queries'] += 1

        logger.info(f"Retrieved {len(documents)} documents using {search_method} method")
        return documents
    
    def _format_sources(
        self,
        documents: List[Document],
        max_citations: int = 3,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Format source citations from retrieved documents with deduplication.

        `query` is used only for the optional book-citation reservation pass
        below (we want to know if the user signalled interest in a book).
        """
        query_for_routing = query
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
                source = {
                    'content_type': 'episode',
                    'episode_id': metadata.get('episode_id', 'Unknown'),
                    'episode_number': metadata.get('episode_number', 'Unknown'),
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

        # Book-slot reservation (Aaron's #3, refined 2026-05-04): when the
        # user signals interest in a book — by saying "book", "handbook", or
        # a specific YOE book title — and the natural top-N sources have no
        # book, swap one in. We only do the swap on book-signaled queries to
        # avoid displacing a relevant episode for theme queries like
        # "regenerative social enterprise" where a book isn't actually being
        # asked for.
        # 1) If the query names a SPECIFIC book by title, ensure that book
        # appears in citations even if another book naturally outranks it.
        # 2) Otherwise (no specific title), only swap a book in if no book is
        # already cited and the query signals book interest.
        named_book_canonical = self._named_book_in_query(query_for_routing)
        if named_book_canonical and max_citations >= 2:
            already_cites_named = any(
                s.get('content_type') == 'book'
                and s.get('book_title') == named_book_canonical
                for s in sources
            )
            if not already_cites_named:
                first_book_doc = self._find_specific_book_doc(documents, named_book_canonical)
                if first_book_doc is not None:
                    book_source = self._format_single_source(first_book_doc)
                    if book_source is not None:
                        # Replace any naturally-cited book first (we want the
                        # named book, not whichever ranked higher), else swap
                        # the lowest-priority slot.
                        replace_idx = next(
                            (i for i, s in enumerate(sources)
                             if s.get('content_type') == 'book'),
                            None,
                        )
                        if replace_idx is None:
                            if len(sources) >= max_citations:
                                replace_idx = len(sources) - 1
                            else:
                                sources.append(book_source)
                                return sources
                        sources[replace_idx] = book_source
                return sources

        if (
            max_citations >= 2
            and not any(s.get('content_type') == 'book' for s in sources)
            and self._query_signals_book(query_for_routing)
        ):
            first_book_doc = self._pick_book_for_reservation(documents, query_for_routing)
            if first_book_doc is not None:
                try:
                    book_source = self._format_single_source(first_book_doc)
                except Exception as e:
                    logger.warning(f"Could not format reserved book source: {e}")
                    book_source = None
                if book_source is not None:
                    # Replace the lowest-priority source with the book; keep top
                    # episode at sources[0] (it tends to be the most relevant).
                    if len(sources) >= max_citations:
                        sources[-1] = book_source
                    else:
                        sources.append(book_source)

        return sources

    @staticmethod
    def _named_book_in_query(query: Optional[str]) -> Optional[str]:
        """Return the canonical YOE book title if the query names it; else None."""
        if not query:
            return None
        q = query.lower()
        title_cues = {
            "Y on Earth: Get Smarter, Feel Better, Heal the Planet": ("y on earth",),
            "Soil Stewardship Handbook": ("soil stewardship",),
            "VIRIDITAS: THE GREAT HEALING": ("viriditas",),
        }
        for canonical, cues in title_cues.items():
            if any(cue in q for cue in cues):
                return canonical
        return None

    @staticmethod
    def _find_specific_book_doc(
        documents: List[Document], canonical: str
    ) -> Optional[Document]:
        """Return the first chunk whose book_title matches `canonical`."""
        for d in documents:
            md = getattr(d, 'metadata', {}) or {}
            if md.get('content_type') == 'book' and md.get('book_title') == canonical:
                return d
        return None

    @staticmethod
    def _pick_book_for_reservation(
        documents: List[Document],
        query: Optional[str],
    ) -> Optional[Document]:
        """Choose which book chunk to reserve when the query signals a book.

        If the query names a specific YOE book (or distinctive fragment of
        a book title), prefer a chunk from THAT book even if another book
        outranks it in the candidate pool. Otherwise fall back to the first
        book chunk in retrieval order.
        """
        book_docs = [
            d for d in documents
            if (getattr(d, 'metadata', {}) or {}).get('content_type') == 'book'
        ]
        if not book_docs:
            return None

        q = (query or "").lower()
        title_cues = {
            "Y on Earth: Get Smarter, Feel Better, Heal the Planet": ("y on earth",),
            "Soil Stewardship Handbook": ("soil stewardship",),
            "VIRIDITAS: THE GREAT HEALING": ("viriditas",),
        }
        for canonical, cues in title_cues.items():
            if any(cue in q for cue in cues):
                for d in book_docs:
                    if (getattr(d, 'metadata', {}) or {}).get('book_title') == canonical:
                        return d

        return book_docs[0]

    @staticmethod
    def _query_signals_book(query: Optional[str]) -> bool:
        """True if the query suggests a book citation would be relevant.

        Triggers on:
          (a) explicit book/chapter mentions ("the book says...", "in chapter 3");
          (b) named YOE book titles;
          (c) "how do/can I X" how-to questions, where book chapters often
              contain the procedural detail an episode interview only summarizes;
          (d) "what does X say about" / "tell me about" framings, which usually
              expect quotational-style content books are good for.

        Used to gate book-citation reservation so theme/discovery queries like
        "regenerative social enterprise" don't get a book forced on them at
        the expense of more relevant episodes.
        """
        if not query:
            return False
        q = query.lower()
        explicit = (
            "book", "books", "chapter", "chapters", "handbook",
            "y on earth", "soil stewardship", "viriditas",
        )
        if any(s in q for s in explicit):
            return True
        how_to = ("how do i", "how can i", "how do you", "how to ", "how does")
        if any(s in q for s in how_to):
            return True
        quotational = ("what does ", "tell me about ", "explain ")
        if any(s in q for s in quotational):
            return True
        return False

    def _format_single_source(self, doc: Document) -> Dict[str, Any]:
        """Build the dict shape returned by _format_sources, for one document.

        Reuses the same logic as the main loop so the reserved book slot has
        identical metadata to a naturally-ranked book citation.
        """
        # Round-trip through _format_sources with max_citations=1 and a single doc
        formatted = self._format_sources([doc], max_citations=1, query=None)
        if not formatted:
            raise ValueError("could not format source for doc")
        # _format_sources also runs the book-reservation pass on a single-book
        # input, but since the doc IS a book, that pass is a no-op.
        return formatted[0]
    
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