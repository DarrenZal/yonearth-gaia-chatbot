"""
Custom Pipeline Builder

Flexible builder for creating custom post-processing pipelines.

Example usage:
    from src.knowledge_graph.postprocessing import CustomPipeline
    from src.knowledge_graph.postprocessing.universal import PronounResolver, ListSplitter
    from src.knowledge_graph.postprocessing.content_specific.books import PraiseQuoteDetector

    # Build custom pipeline
    pipeline = (
        CustomPipeline()
        .add_module(PraiseQuoteDetector())
        .add_module(PronounResolver())
        .add_module(ListSplitter())
        .build()
    )

    # Use pipeline
    from src.knowledge_graph.postprocessing import ProcessingContext
    context = ProcessingContext(content_type='book', document_metadata={})
    processed, stats = pipeline.run(relationships, context)
"""

from typing import List, Optional, Dict, Any

from ..base import PostProcessingModule, PipelineOrchestrator


class CustomPipeline:
    """
    Builder for creating custom post-processing pipelines.

    Provides a fluent interface for composing modules.
    """

    def __init__(self):
        self.modules: List[PostProcessingModule] = []

    def add_module(self, module: PostProcessingModule) -> 'CustomPipeline':
        """
        Add a module to the pipeline.

        Args:
            module: PostProcessingModule instance to add

        Returns:
            self for method chaining
        """
        self.modules.append(module)
        return self

    def add_modules(self, modules: List[PostProcessingModule]) -> 'CustomPipeline':
        """
        Add multiple modules to the pipeline.

        Args:
            modules: List of PostProcessingModule instances

        Returns:
            self for method chaining
        """
        self.modules.extend(modules)
        return self

    def remove_module(self, module_name: str) -> 'CustomPipeline':
        """
        Remove a module by name.

        Args:
            module_name: Name of the module to remove

        Returns:
            self for method chaining
        """
        self.modules = [m for m in self.modules if m.name != module_name]
        return self

    def clear(self) -> 'CustomPipeline':
        """
        Clear all modules from the pipeline.

        Returns:
            self for method chaining
        """
        self.modules = []
        return self

    def build(self) -> PipelineOrchestrator:
        """
        Build the pipeline orchestrator.

        Returns:
            PipelineOrchestrator with configured modules

        Raises:
            ValueError: If no modules added or validation fails
        """
        if not self.modules:
            raise ValueError("Cannot build pipeline with no modules")

        return PipelineOrchestrator(self.modules)

    def __repr__(self) -> str:
        module_names = [m.name for m in self.modules]
        return f"CustomPipeline(modules={module_names})"
