"""
Pre-Configured Post-Processing Pipelines

Ready-to-use pipeline configurations for common content types.

Available Pipelines:
-------------------
- get_book_pipeline(): Full pipeline for book extraction
- get_podcast_pipeline(): Pipeline for podcast transcripts
- CustomPipeline: Builder for creating custom pipelines

Example usage:
    # Use pre-configured pipeline
    from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline
    from src.knowledge_graph.postprocessing import ProcessingContext

    pipeline = get_book_pipeline()
    context = ProcessingContext(content_type='book', document_metadata={'title': 'My Book'})
    processed, stats = pipeline.run(relationships, context)

    # Build custom pipeline
    from src.knowledge_graph.postprocessing.pipelines import CustomPipeline
    from src.knowledge_graph.postprocessing.universal import PronounResolver, ListSplitter

    pipeline = CustomPipeline().add_module(PronounResolver()).add_module(ListSplitter()).build()
"""

from .book_pipeline import get_book_pipeline
from .podcast_pipeline import get_podcast_pipeline
from .custom_pipeline import CustomPipeline

__all__ = [
    "get_book_pipeline",
    "get_podcast_pipeline",
    "CustomPipeline",
]
