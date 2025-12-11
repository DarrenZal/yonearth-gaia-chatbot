"""
GraphRAG Chat API endpoints.

Provides endpoints for:
- GraphRAG-powered chat using community and entity search
- Comparison between GraphRAG and BM25 systems
- Health checks and community browsing
"""
import json
import logging
import os
import time
from typing import Optional, Dict, List

from fastapi import APIRouter, HTTPException, Depends

from .graphrag_chat_models import (
    GraphRAGChatRequest,
    GraphRAGChatResponse,
    GraphRAGCompareRequest,
    GraphRAGCompareResponse,
    ComparisonResult,
    GraphRAGHealthResponse,
    CommunityListResponse,
    EntitySearchRequest,
    EntitySearchResponse,
    CommunityContext,
    EntityContext,
    SourceCitation
)
from ..rag.graphrag_chain import GraphRAGChain, get_graphrag_chain
from ..rag.bm25_chain import BM25RAGChain
from ..voice.piper_client import PiperVoiceClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graphrag", tags=["GraphRAG Chat"])

# Cache for episode and book metadata
_episode_metadata_cache: Dict[int, Dict] = {}
_book_metadata_cache: Dict[str, Dict] = {}


def load_episode_metadata() -> Dict[int, Dict]:
    """Load episode metadata from processed data."""
    global _episode_metadata_cache
    if _episode_metadata_cache:
        return _episode_metadata_cache

    try:
        # Get base directory (where the repo root is)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        metadata_path = os.path.join(base_dir, 'data/processed/episode_metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                for ep in data.get('episodes', []):
                    _episode_metadata_cache[ep['number']] = ep
            logger.info(f"Loaded metadata for {len(_episode_metadata_cache)} episodes")
    except Exception as e:
        logger.error(f"Failed to load episode metadata: {e}")

    return _episode_metadata_cache


def load_book_metadata() -> Dict[str, Dict]:
    """Load book metadata from data/books directories."""
    global _book_metadata_cache
    if _book_metadata_cache:
        return _book_metadata_cache

    try:
        # Get base directory (where the repo root is)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        books_dir = os.path.join(base_dir, 'data/books')

        if os.path.exists(books_dir):
            for book_name in os.listdir(books_dir):
                book_path = os.path.join(books_dir, book_name)
                metadata_path = os.path.join(book_path, 'metadata.json')
                if os.path.isdir(book_path) and os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        # Key by various possible book IDs
                        book_id = book_name.lower().replace(' ', '-').replace('_', '-')
                        _book_metadata_cache[book_id] = metadata
                        _book_metadata_cache[book_name] = metadata
                        if metadata.get('title'):
                            _book_metadata_cache[metadata['title'].lower()] = metadata
            logger.info(f"Loaded metadata for {len(_book_metadata_cache)} books")
    except Exception as e:
        logger.error(f"Failed to load book metadata: {e}")

    return _book_metadata_cache


def transform_sources_for_frontend(
    source_episodes: List[str],
    source_books: List[str]
) -> List[SourceCitation]:
    """Transform source_episodes and source_books into frontend-compatible format."""
    sources = []
    episode_meta = load_episode_metadata()
    book_meta = load_book_metadata()

    # Process episodes
    for ep_id in source_episodes:
        try:
            # Extract episode number from ID like "episode_120"
            ep_num = int(ep_id.replace('episode_', ''))
            meta = episode_meta.get(ep_num, {})

            sources.append(SourceCitation(
                content_type="episode",
                episode_number=str(ep_num),
                title=meta.get('title', f'Episode {ep_num}'),
                guest_name=meta.get('guest'),
                url=f"https://yonearth.org/episodes/episode-{ep_num}/"
            ))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not parse episode ID {ep_id}: {e}")

    # Process books
    for book_id in source_books:
        try:
            # Extract book name from ID like "book_soil-stewardship-handbook"
            book_name = book_id.replace('book_', '').replace('-', ' ')

            # Try to find metadata
            meta = None
            for key, value in book_meta.items():
                if book_name.lower() in key.lower() or key.lower() in book_name.lower():
                    meta = value
                    break

            if meta:
                book_title = meta.get('title', book_name.title())
                sources.append(SourceCitation(
                    content_type="book",
                    episode_number=f"Book: {book_title}",
                    title=book_title,
                    guest_name=meta.get('author'),
                    url=meta.get('ebook_url') or meta.get('url'),
                    ebook_url=meta.get('ebook_url'),
                    audiobook_url=meta.get('audiobook_url'),
                    print_url=meta.get('print_url')
                ))
            else:
                # Fallback without metadata
                sources.append(SourceCitation(
                    content_type="book",
                    episode_number=f"Book: {book_name.title()}",
                    title=book_name.title(),
                    guest_name=None,
                    url=None
                ))
        except Exception as e:
            logger.warning(f"Could not parse book ID {book_id}: {e}")

    return sources

# Lazy-loaded chain instances
_graphrag_chain: Optional[GraphRAGChain] = None
_bm25_chain: Optional[BM25RAGChain] = None


def get_chain() -> GraphRAGChain:
    """Get or initialize the GraphRAG chain"""
    global _graphrag_chain
    if _graphrag_chain is None:
        logger.info("Initializing GraphRAG chain for API...")
        _graphrag_chain = get_graphrag_chain()
    return _graphrag_chain


def get_bm25() -> BM25RAGChain:
    """Get or initialize the BM25 chain for comparison"""
    global _bm25_chain
    if _bm25_chain is None:
        logger.info("Initializing BM25 chain for comparison...")
        _bm25_chain = BM25RAGChain(initialize_data=True)
    return _bm25_chain


@router.post("/chat", response_model=GraphRAGChatResponse)
async def graphrag_chat(request: GraphRAGChatRequest):
    """
    Chat with Gaia using GraphRAG search.

    GraphRAG combines:
    - **Global Search**: Community summaries for thematic understanding
    - **Local Search**: Entity extraction for specific details
    - **DRIFT Search**: Both combined (recommended)

    Search Modes:
    - `global`: Best for broad questions ("What are the main themes?")
    - `local`: Best for specific queries ("Who is Aaron Perry?")
    - `drift`: Best overall - combines both approaches
    - `auto`: Automatically detect best mode
    """
    try:
        chain = get_chain()

        result = chain.chat(
            message=request.message,
            search_mode=request.search_mode,
            community_level=request.community_level,
            k_communities=request.k_communities,
            k_entities=request.k_entities,
            k_chunks=request.k_chunks,
            personality=request.personality,
            custom_prompt=request.custom_prompt
        )

        # Generate voice if requested
        audio_data = None
        if request.enable_voice:
            try:
                logger.info("Generating voice with Piper TTS (Gaia voice)")
                voice_client = PiperVoiceClient(voice_name="en_US-kristin-medium")
                # Preprocess response for better speech
                speech_text = voice_client.preprocess_text_for_speech(result.get('response', ''))
                audio_data = voice_client.generate_speech_base64(speech_text)
                if audio_data:
                    logger.info(f"Voice generated successfully: {len(audio_data)} bytes base64")
                else:
                    logger.warning("Voice generation returned None")
            except Exception as voice_error:
                logger.error(f"Voice generation failed: {voice_error}")
                # Continue without voice rather than failing the entire request

        # Transform sources for frontend (limit to top 3)
        source_episodes = result.get('source_episodes', [])
        source_books = result.get('source_books', [])
        all_sources = transform_sources_for_frontend(source_episodes, source_books)
        sources = all_sources[:3]  # Limit to top 3 references

        return GraphRAGChatResponse(
            response=result.get('response', ''),
            search_mode=result.get('search_mode', request.search_mode),
            communities_used=[
                CommunityContext(**c) for c in result.get('communities_used', [])
            ],
            entities_matched=[
                EntityContext(**e) for e in result.get('entities_matched', [])
            ],
            relationships=result.get('relationships', []),
            source_episodes=source_episodes,
            source_books=source_books,
            sources=sources,
            processing_time=result.get('processing_time', 0),
            success=result.get('success', True),
            error=result.get('error'),
            audio_data=audio_data
        )

    except Exception as e:
        logger.error(f"GraphRAG chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=GraphRAGCompareResponse)
async def compare_with_bm25(request: GraphRAGCompareRequest):
    """
    Compare responses from GraphRAG and BM25 systems.

    Returns side-by-side responses from both systems for the same query,
    along with comparison metrics.
    """
    results = {
        'query': request.message,
        'bm25_result': None,
        'graphrag_result': None,
        'comparison_metrics': {}
    }

    # Get BM25 response
    if request.include_bm25:
        try:
            bm25_chain = get_bm25()
            start_time = time.time()

            bm25_response = bm25_chain.chat(
                message=request.message,
                search_method="hybrid",
                k=5,
                include_sources=True,
                personality_variant=request.personality
            )

            bm25_time = time.time() - start_time

            results['bm25_result'] = ComparisonResult(
                system="bm25",
                response=bm25_response.get('response', ''),
                processing_time=bm25_time,
                success=bm25_response.get('success', True),
                error=bm25_response.get('error'),
                metadata={
                    'search_method': bm25_response.get('search_method_used'),
                    'documents_retrieved': bm25_response.get('documents_retrieved'),
                    'episode_references': bm25_response.get('episode_references', []),
                    'sources': bm25_response.get('sources', [])[:3]
                }
            )

        except Exception as e:
            logger.error(f"BM25 comparison error: {e}")
            results['bm25_result'] = ComparisonResult(
                system="bm25",
                response="",
                processing_time=0,
                success=False,
                error=str(e),
                metadata={}
            )

    # Get GraphRAG response
    if request.include_graphrag:
        try:
            graphrag_chain = get_chain()
            start_time = time.time()

            graphrag_response = graphrag_chain.chat(
                message=request.message,
                search_mode=request.graphrag_mode,
                community_level=request.community_level,
                k_communities=request.k_communities,
                k_entities=request.k_entities,
                personality=request.personality
            )

            graphrag_time = time.time() - start_time

            # Extract episode numbers from source_episodes (format: "episode_123" -> 123)
            source_episodes = []
            for ep in graphrag_response.get('source_episodes', []):
                if isinstance(ep, str) and ep.startswith('episode_'):
                    try:
                        source_episodes.append(int(ep.replace('episode_', '')))
                    except ValueError:
                        pass
                elif isinstance(ep, int):
                    source_episodes.append(ep)

            results['graphrag_result'] = ComparisonResult(
                system="graphrag",
                response=graphrag_response.get('response', ''),
                processing_time=graphrag_time,
                success=graphrag_response.get('success', True),
                error=graphrag_response.get('error'),
                metadata={
                    'search_mode': graphrag_response.get('search_mode'),
                    'communities_count': len(graphrag_response.get('communities_used', [])),
                    'entities_count': len(graphrag_response.get('entities_matched', [])),
                    'relationships_count': len(graphrag_response.get('relationships', [])),
                    'communities_used': graphrag_response.get('communities_used', [])[:3],
                    'entities_matched': graphrag_response.get('entities_matched', [])[:5],
                    'source_episodes': source_episodes,  # Add for overlap calculation
                    'source_books': graphrag_response.get('source_books', [])
                }
            )

        except Exception as e:
            logger.error(f"GraphRAG comparison error: {e}")
            results['graphrag_result'] = ComparisonResult(
                system="graphrag",
                response="",
                processing_time=0,
                success=False,
                error=str(e),
                metadata={}
            )

    # Calculate comparison metrics
    if results['bm25_result'] and results['graphrag_result']:
        # Normalize episode references to integers for comparison
        # BM25 may return strings like "120", GraphRAG returns ints
        def normalize_episodes(episodes):
            normalized = set()
            for ep in episodes:
                try:
                    normalized.add(int(ep))
                except (ValueError, TypeError):
                    pass
            return normalized

        bm25_episodes = normalize_episodes(results['bm25_result'].metadata.get('episode_references', []))
        graphrag_episodes = normalize_episodes(results['graphrag_result'].metadata.get('source_episodes', []))

        # Overlap analysis
        common_episodes = bm25_episodes & graphrag_episodes
        total_episodes = bm25_episodes | graphrag_episodes

        results['comparison_metrics'] = {
            'bm25_episodes': sorted(list(bm25_episodes)),
            'graphrag_episodes': sorted(list(graphrag_episodes)),
            'common_episodes': sorted(list(common_episodes)),
            'episode_overlap_ratio': len(common_episodes) / len(total_episodes) if total_episodes else 0,
            'graphrag_communities': [
                c.get('title', c.get('name', ''))
                for c in results['graphrag_result'].metadata.get('communities_used', [])
            ],
            'graphrag_entities': [
                e.get('name', '')
                for e in results['graphrag_result'].metadata.get('entities_matched', [])
            ],
            'speed_comparison': {
                'bm25_time': results['bm25_result'].processing_time,
                'graphrag_time': results['graphrag_result'].processing_time,
                'faster_system': 'bm25' if results['bm25_result'].processing_time < results['graphrag_result'].processing_time else 'graphrag'
            }
        }

    return GraphRAGCompareResponse(**results)


@router.get("/health", response_model=GraphRAGHealthResponse)
async def health_check():
    """Check health of GraphRAG chain components"""
    try:
        chain = get_chain()
        health = chain.health_check()

        return GraphRAGHealthResponse(
            initialized=health.get('initialized', False),
            community_search_ready=health.get('community_search_ready', False),
            local_search_ready=health.get('local_search_ready', False),
            vectorstore_ready=health.get('vectorstore_ready', False),
            gaia_ready=health.get('gaia_ready', False),
            stats=health.get('stats', {})
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return GraphRAGHealthResponse(
            initialized=False,
            community_search_ready=False,
            local_search_ready=False,
            vectorstore_ready=False,
            gaia_ready=False,
            stats={'error': str(e)}
        )


@router.get("/communities/{level}", response_model=CommunityListResponse)
async def list_communities(level: int = 1, limit: int = 50, offset: int = 0):
    """
    List available communities at a given hierarchy level.

    - Level 1: 573 fine-grained topic clusters
    - Level 2: 73 broader theme clusters
    """
    if level not in [1, 2]:
        raise HTTPException(status_code=400, detail="Level must be 1 or 2")

    try:
        chain = get_chain()
        communities = chain.community_search.communities_by_level.get(level, [])

        # Paginate
        paginated = communities[offset:offset + limit]

        return CommunityListResponse(
            level=level,
            count=len(communities),
            communities=[
                {
                    'id': c['id'],
                    'name': c['name'],
                    'title': c['title'],
                    'summary': c['summary_text'][:200] + '...' if len(c['summary_text']) > 200 else c['summary_text'],
                    'entity_count': c['entity_count']
                }
                for c in paginated
            ]
        )

    except Exception as e:
        logger.error(f"List communities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/search", response_model=EntitySearchResponse)
async def search_entities(request: EntitySearchRequest):
    """
    Search for entities mentioned in text.

    Uses lexicon matching to find entities by name or alias.
    """
    try:
        chain = get_chain()

        # Find entities in text
        entity_names = chain.local_search.find_entities_in_text(request.query)

        # Get full context for each
        entities = []
        for name in entity_names[:request.k]:
            ctx = chain.local_search.get_entity_context(name)
            if ctx:
                entities.append(EntityContext(
                    name=ctx.name,
                    type=ctx.type,
                    description=ctx.description[:300] + '...' if len(ctx.description) > 300 else ctx.description,
                    sources=ctx.sources[:5],
                    mention_count=ctx.mention_count,
                    relevance_score=ctx.relevance_score
                ))

        return EntitySearchResponse(
            query=request.query,
            entities_found=entities,
            total_matches=len(entity_names)
        )

    except Exception as e:
        logger.error(f"Entity search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get GraphRAG chain statistics"""
    try:
        chain = get_chain()
        return chain.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
