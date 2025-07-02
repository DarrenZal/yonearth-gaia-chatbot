"""
Book processor for loading and preparing PDF books for ingestion
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..config import settings

logger = logging.getLogger(__name__)


class Chapter:
    """Represents a book chapter with metadata and content"""
    
    def __init__(self, number: int, title: str, content: str, page_range: Tuple[int, int]):
        self.number = number
        self.title = title
        self.content = content
        self.page_start, self.page_end = page_range
        self.word_count = len(content.split())
    
    @property
    def has_content(self) -> bool:
        """Check if chapter has valid content"""
        return bool(self.content and len(self.content) > 200)
    
    def to_document(self, book_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chapter to document format for vector storage"""
        metadata = book_metadata.copy()
        metadata.update({
            "chapter_number": self.number,
            "chapter_title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "word_count": self.word_count,
            "content_type": "book"
        })
        
        return {
            "page_content": self.content,
            "metadata": metadata
        }


class Book:
    """Represents a book with metadata and chapters"""
    
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.book_dir = metadata_path.parent
        self._load_metadata()
        self.chapters: List[Chapter] = []
        self.full_text = ""
        
    def _load_metadata(self):
        """Load book metadata from JSON file"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.title = metadata.get("title", "")
            self.author = metadata.get("author", "")
            self.publication_year = metadata.get("publication_year")
            self.isbn = metadata.get("isbn", "")
            self.category = metadata.get("category", "")
            self.topics = metadata.get("topics", [])
            self.description = metadata.get("description", "")
            self.file_path = metadata.get("file_path", "")
            self.language = metadata.get("language", "english")
            self.pages = metadata.get("pages")
            self.word_count = metadata.get("word_count")
            
        except Exception as e:
            logger.error(f"Error loading metadata from {self.metadata_path}: {e}")
            raise
    
    @property
    def pdf_path(self) -> Path:
        """Get the path to the PDF file"""
        return self.book_dir / self.file_path
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get book metadata for vector storage"""
        return {
            "book_title": self.title,
            "author": self.author,
            "publication_year": self.publication_year,
            "isbn": self.isbn,
            "category": self.category,
            "topics": self.topics,
            "description": self.description,
            "language": self.language
        }
    
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF file"""
        if not pdfplumber:
            raise ImportError("pdfplumber is required for PDF processing. Install with: pip install pdfplumber")
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        text_pages = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                logger.info(f"Extracting text from {len(pdf.pages)} pages in {self.pdf_path.name}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Clean up text
                        text = self._clean_text(text)
                        text_pages.append((page_num, text))
                    
                    if page_num % 10 == 0:
                        logger.info(f"Processed {page_num}/{len(pdf.pages)} pages")
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
        
        # Combine all pages
        self.full_text = "\n\n".join([text for page_num, text in text_pages])
        self.pages = len(text_pages)
        self.word_count = len(self.full_text.split())
        
        logger.info(f"Extracted {self.word_count} words from {self.pages} pages")
        return self.full_text
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely page numbers (single numbers or numbers with minimal text)
            if re.match(r'^\d+$', line) or re.match(r'^\d+\s*$', line):
                continue
            # Skip very short lines that might be headers/footers
            if len(line) < 3:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def detect_chapters(self) -> List[Chapter]:
        """Detect chapters in the text using heuristics"""
        if not self.full_text:
            logger.warning("No text available for chapter detection")
            return []
        
        chapters = []
        
        # Simple chapter detection patterns
        chapter_patterns = [
            r'^CHAPTER\s+(\d+)[\:\-\s]*(.*)$',
            r'^Chapter\s+(\d+)[\:\-\s]*(.*)$',
            r'^(\d+)[\.\)\s]+(.*)$',  # Numbered sections
        ]
        
        lines = self.full_text.split('\n')
        current_chapter_lines = []
        current_chapter_num = 0
        current_chapter_title = ""
        chapter_start_page = 1
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if this line looks like a chapter heading
            chapter_match = None
            for pattern in chapter_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    chapter_match = match
                    break
            
            if chapter_match and len(line) < 100:  # Chapter titles are usually short
                # Save previous chapter if exists
                if current_chapter_lines and current_chapter_num > 0:
                    chapter_content = '\n'.join(current_chapter_lines).strip()
                    if len(chapter_content) > 200:  # Minimum chapter length
                        chapter = Chapter(
                            number=current_chapter_num,
                            title=current_chapter_title,
                            content=chapter_content,
                            page_range=(chapter_start_page, chapter_start_page + len(current_chapter_lines) // 50)
                        )
                        chapters.append(chapter)
                
                # Start new chapter
                current_chapter_num = int(chapter_match.group(1))
                current_chapter_title = chapter_match.group(2).strip() if len(chapter_match.groups()) > 1 else f"Chapter {current_chapter_num}"
                current_chapter_lines = []
                chapter_start_page = i // 50 + 1  # Rough page estimation
                
            else:
                # Add line to current chapter
                if line:  # Skip empty lines
                    current_chapter_lines.append(line)
        
        # Don't forget the last chapter
        if current_chapter_lines and current_chapter_num > 0:
            chapter_content = '\n'.join(current_chapter_lines).strip()
            if len(chapter_content) > 200:
                chapter = Chapter(
                    number=current_chapter_num,
                    title=current_chapter_title,
                    content=chapter_content,
                    page_range=(chapter_start_page, self.pages or chapter_start_page + len(current_chapter_lines) // 50)
                )
                chapters.append(chapter)
        
        # If no chapters detected, create a single chapter with all content
        if not chapters and self.full_text:
            chapter = Chapter(
                number=1,
                title=self.title or "Full Text",
                content=self.full_text,
                page_range=(1, self.pages or 1)
            )
            chapters.append(chapter)
        
        self.chapters = chapters
        logger.info(f"Detected {len(chapters)} chapters in {self.title}")
        return chapters


class BookProcessor:
    """Process books for ingestion into vector database"""
    
    def __init__(self):
        self.books_dir = Path(settings.data_dir) / "books"
        self.processed_dir = settings.processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def find_books(self) -> List[Path]:
        """Find all books with metadata.json files"""
        if not self.books_dir.exists():
            logger.warning(f"Books directory not found: {self.books_dir}")
            return []
        
        book_metadata_files = list(self.books_dir.glob("*/metadata.json"))
        logger.info(f"Found {len(book_metadata_files)} books with metadata")
        return book_metadata_files
    
    def load_books(self, limit: Optional[int] = None) -> List[Book]:
        """Load books from metadata files"""
        metadata_files = self.find_books()
        
        if limit:
            metadata_files = metadata_files[:limit]
        
        books = []
        
        for metadata_path in metadata_files:
            try:
                book = Book(metadata_path)
                books.append(book)
                logger.info(f"Loaded book: {book.title} by {book.author}")
                
            except Exception as e:
                logger.error(f"Error loading book from {metadata_path}: {e}")
        
        logger.info(f"Loaded {len(books)} books")
        return books
    
    def process_book(self, book: Book) -> List[Dict[str, Any]]:
        """Process a single book into documents"""
        documents = []
        
        try:
            # Extract text from PDF
            book.extract_text_from_pdf()
            
            # Detect chapters
            chapters = book.detect_chapters()
            
            # Convert chapters to documents
            for chapter in chapters:
                if chapter.has_content:
                    doc = chapter.to_document(book.metadata)
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} documents from {book.title}")
            
        except Exception as e:
            logger.error(f"Error processing book {book.title}: {e}")
            
        return documents
    
    def save_processed_books(self, books: List[Book]):
        """Save processed books metadata for reference"""
        metadata = {
            "processed_at": datetime.now().isoformat(),
            "book_count": len(books),
            "books": [
                {
                    "title": book.title,
                    "author": book.author,
                    "publication_year": book.publication_year,
                    "category": book.category,
                    "chapters": len(book.chapters),
                    "pages": book.pages,
                    "word_count": book.word_count
                }
                for book in books
            ]
        }
        
        output_path = self.processed_dir / "book_metadata.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved book metadata to {output_path}")


def main():
    """Test book processing"""
    logging.basicConfig(level=logging.INFO)
    
    processor = BookProcessor()
    books = processor.load_books()
    
    for book in books:
        print(f"\nBook: {book.title}")
        print(f"Author: {book.author}")
        print(f"PDF Path: {book.pdf_path}")
        print(f"PDF Exists: {book.pdf_path.exists()}")
        
        if book.pdf_path.exists():
            documents = processor.process_book(book)
            print(f"Generated {len(documents)} documents")


if __name__ == "__main__":
    main()