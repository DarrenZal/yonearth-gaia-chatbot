"""
Base classes and interfaces for modular post-processing system.

This module provides the foundation for composable "lego block" post-processing
modules that can be mixed-and-matched for different content types.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingContext:
    """
    Context passed to all post-processing modules.

    Contains metadata and configuration needed by modules to make decisions
    about how to process relationships.
    """

    # Content type identifier
    content_type: str  # 'book', 'podcast', 'academic', 'news', 'custom'

    # Document metadata
    document_metadata: Dict[str, Any] = field(default_factory=dict)

    # Content-specific data
    pages_with_text: Optional[List[Tuple[int, str]]] = None  # For books, PDFs
    audio_timestamps: Optional[List[Dict[str, Any]]] = None  # For podcasts, videos
    sections: Optional[List[Dict[str, Any]]] = None  # For structured documents

    # Processing configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    run_id: Optional[str] = None
    extraction_version: Optional[str] = None


class PostProcessingModule(ABC):
    """
    Abstract base class for all post-processing modules.

    Each module:
    - Is self-contained and independent
    - Declares its applicable content types
    - Can be composed with other modules in a pipeline
    - Collects statistics about its processing

    Modules should follow the single responsibility principle.
    """

    # Module metadata (override in subclasses)
    name: str = "BaseModule"
    description: str = "Base post-processing module"
    content_types: List[str] = ["all"]  # Which content types this applies to
    priority: int = 50  # Execution priority (lower = runs earlier)
    dependencies: List[str] = []  # Module names that must run before this one
    version: str = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the module with optional configuration.

        Args:
            config: Module-specific configuration dictionary
        """
        self.config = config or {}
        self.stats: Dict[str, Any] = {
            'processed_count': 0,
            'modified_count': 0,
            'filtered_count': 0,
            'error_count': 0
        }
        self.enabled = self.config.get('enabled', True)

    @abstractmethod
    def process_batch(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> List[Any]:
        """
        Process a batch of relationships.

        Args:
            relationships: List of ProductionRelationship objects
            context: Processing context with metadata and config

        Returns:
            Processed list of relationships (may be filtered or augmented)
        """
        pass

    def is_applicable(self, context: ProcessingContext) -> bool:
        """
        Check if this module should run for the given content type.

        Args:
            context: Processing context

        Returns:
            True if module should process, False to skip
        """
        if not self.enabled:
            return False

        return (
            "all" in self.content_types or
            context.content_type in self.content_types
        )

    def validate_dependencies(self, available_modules: List[str]) -> bool:
        """
        Check if all required dependencies are available.

        Args:
            available_modules: List of module names that will run

        Returns:
            True if all dependencies are met
        """
        return all(dep in available_modules for dep in self.dependencies)

    def reset_stats(self):
        """Reset module statistics"""
        self.stats = {
            'processed_count': 0,
            'modified_count': 0,
            'filtered_count': 0,
            'error_count': 0
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get module processing summary"""
        return {
            'name': self.name,
            'version': self.version,
            'stats': self.stats.copy()
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', priority={self.priority})"


class PipelineOrchestrator:
    """
    Orchestrates execution of a post-processing pipeline.

    Responsibilities:
    - Sort modules by priority
    - Validate dependencies
    - Skip non-applicable modules
    - Collect statistics from all modules
    - Handle errors gracefully
    """

    def __init__(
        self,
        modules: List[PostProcessingModule],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize orchestrator with a list of modules.

        Args:
            modules: List of PostProcessingModule instances
            config: Global configuration for the pipeline
        """
        self.modules = modules
        self.config = config or {}
        self.pipeline_stats: Dict[str, Any] = {}

    def _sort_modules(self) -> List[PostProcessingModule]:
        """Sort modules by priority (lower priority runs first)"""
        return sorted(self.modules, key=lambda m: m.priority)

    def _validate_pipeline(self) -> Tuple[bool, List[str]]:
        """
        Validate that all module dependencies are satisfied.

        Returns:
            (is_valid, error_messages)
        """
        errors = []
        module_names = [m.name for m in self.modules]

        for module in self.modules:
            if not module.validate_dependencies(module_names):
                missing = [dep for dep in module.dependencies if dep not in module_names]
                errors.append(
                    f"Module '{module.name}' missing dependencies: {missing}"
                )

        return len(errors) == 0, errors

    def run(
        self,
        relationships: List[Any],
        context: ProcessingContext
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Run the complete post-processing pipeline.

        Args:
            relationships: List of ProductionRelationship objects
            context: Processing context

        Returns:
            (processed_relationships, pipeline_stats)
        """
        # Validate pipeline
        is_valid, errors = self._validate_pipeline()
        if not is_valid:
            logger.error("Pipeline validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Pipeline validation failed. See logs for details.")

        # Sort modules by priority
        sorted_modules = self._sort_modules()

        # Initialize stats
        self.pipeline_stats = {
            'initial_count': len(relationships),
            'modules_run': [],
            'modules_skipped': [],
            'module_stats': {}
        }

        logger.info(f"🎨 Running post-processing pipeline with {len(sorted_modules)} modules...")
        logger.info(f"   Content type: {context.content_type}")
        logger.info(f"   Initial relationships: {len(relationships)}")
        logger.info("")

        # Execute each module
        for i, module in enumerate(sorted_modules, 1):
            # Check if module is applicable
            if not module.is_applicable(context):
                logger.info(f"  {i}/{len(sorted_modules)}: Skipping {module.name} (not applicable to {context.content_type})")
                self.pipeline_stats['modules_skipped'].append(module.name)
                continue

            # Run module
            logger.info(f"  {i}/{len(sorted_modules)}: {module.name}...")

            try:
                before_count = len(relationships)
                relationships = module.process_batch(relationships, context)
                after_count = len(relationships)

                # Collect stats
                self.pipeline_stats['modules_run'].append(module.name)
                self.pipeline_stats['module_stats'][module.name] = module.get_summary()

                # Log changes
                if after_count != before_count:
                    diff = after_count - before_count
                    logger.info(f"       {module.name}: {before_count} → {after_count} ({diff:+d})")
                else:
                    logger.info(f"       {module.name}: {after_count} relationships (no filtering)")

            except Exception as e:
                logger.error(f"  ❌ Error in {module.name}: {e}")
                module.stats['error_count'] += 1

                # Continue with next module (graceful degradation)
                if self.config.get('halt_on_error', False):
                    raise

        self.pipeline_stats['final_count'] = len(relationships)

        logger.info("")
        logger.info(f"✅ Pipeline complete:")
        logger.info(f"   - Modules run: {len(self.pipeline_stats['modules_run'])}")
        logger.info(f"   - Modules skipped: {len(self.pipeline_stats['modules_skipped'])}")
        logger.info(f"   - Final count: {self.pipeline_stats['final_count']} relationships")
        logger.info("")

        return relationships, self.pipeline_stats

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution"""
        return {
            'total_modules': len(self.modules),
            'modules_run': len(self.pipeline_stats.get('modules_run', [])),
            'modules_skipped': len(self.pipeline_stats.get('modules_skipped', [])),
            'initial_count': self.pipeline_stats.get('initial_count', 0),
            'final_count': self.pipeline_stats.get('final_count', 0),
            'module_summaries': self.pipeline_stats.get('module_stats', {})
        }
