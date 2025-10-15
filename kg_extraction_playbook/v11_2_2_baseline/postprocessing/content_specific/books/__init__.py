"""
Book-Specific Post-Processing Modules

Modules specifically designed for processing relationships extracted from books:
- Front matter processing (praise quotes, dedications)
- Bibliographic citation parsing
- Title validation
- Figurative language handling
"""

from .praise_quote_detector import PraiseQuoteDetector
from .bibliographic_citation_parser import BibliographicCitationParser
from .title_completeness_validator import TitleCompletenessValidator
from .figurative_language_filter import FigurativeLanguageFilter

__all__ = [
    "PraiseQuoteDetector",
    "BibliographicCitationParser",
    "TitleCompletenessValidator",
    "FigurativeLanguageFilter",
]
