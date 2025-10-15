"""
Modular Post-Processing System for Knowledge Graph Extraction

This package provides a composable "lego block" architecture for post-processing
extracted relationships. Modules can be mixed-and-matched for different content
types (books, podcasts, academic papers, news, etc.).

Architecture:
-----------
- base.py: Core abstractions (PostProcessingModule, PipelineOrchestrator)
- universal/: Modules that work for ANY content type
- content_specific/: Content-type specific modules (books, podcasts, etc.)
- figurative/: Language style processing (metaphors, idioms)
- pipelines/: Pre-configured pipelines for common content types

Usage:
------
from src.knowledge_graph.postprocessing import PipelineOrchestrator, ProcessingContext
from src.knowledge_graph.postprocessing.pipelines import get_book_pipeline

# Get pre-configured pipeline
pipeline_modules = get_book_pipeline(config)

# Create orchestrator
orchestrator = PipelineOrchestrator(pipeline_modules)

# Create context
context = ProcessingContext(
    content_type='book',
    document_metadata={'author': 'John Doe', 'title': 'Example Book'},
    pages_with_text=pages
)

# Run pipeline
relationships, stats = orchestrator.run(relationships, context)
"""

from .base import (
    PostProcessingModule,
    PipelineOrchestrator,
    ProcessingContext
)

__version__ = "1.0.0"

__all__ = [
    "PostProcessingModule",
    "PipelineOrchestrator",
    "ProcessingContext",
]
