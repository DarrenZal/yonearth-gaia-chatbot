"""
Text chunking module for splitting episode transcripts
"""
import re
import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..config import settings
from .episode_processor import Episode

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk book documents for vector storage"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        # Use larger chunks for books
        self.chunk_size = chunk_size or (settings.chunk_size + 250)  # 750 tokens
        self.chunk_overlap = chunk_overlap or (settings.chunk_overlap + 50)  # 100 tokens
        
        # Initialize text splitter optimized for books
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Chunk book documents into smaller pieces"""
        all_documents = []
        
        for doc in documents:
            try:
                content = doc["page_content"]
                metadata = doc["metadata"]
                
                # Split the content into chunks
                chunks = self.text_splitter.split_text(content)
                
                # Create LangChain documents with metadata
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "chunk_type": "book_chapter"
                    })
                    
                    langchain_doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    all_documents.append(langchain_doc)
                
                logger.info(f"Created {len(chunks)} chunks for chapter {metadata.get('chapter_number', '?')} of {metadata.get('book_title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Error chunking document: {e}")
                continue
        
        logger.info(f"Created {len(all_documents)} total chunks from {len(documents)} book documents")
        return all_documents


class TranscriptChunker:
    """Chunk episode transcripts for vector storage"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        preserve_speaker_turns: bool = True
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.preserve_speaker_turns = preserve_speaker_turns
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def _detect_speaker_format(self, transcript: str) -> Optional[str]:
        """Detect the speaker format in the transcript"""
        # Format 1: "5:45 – speaker_name"
        if re.search(r'\d+:\d+ – \w+', transcript):
            return "timestamp_dash"
        
        # Format 2: "Speaker 1:" or "Aaron Perry:"
        if re.search(r'^[A-Za-z\s]+\d*:', transcript, re.MULTILINE):
            return "speaker_colon"
            
        # Format 3: No clear speaker format
        return None
        
    def _split_by_speaker_turns(self, transcript: str, format_type: str) -> List[str]:
        """Split transcript by speaker turns"""
        chunks = []
        
        if format_type == "timestamp_dash":
            # Split by timestamp pattern
            pattern = r'(\d+:\d+ – \w+)'
            parts = re.split(pattern, transcript)
            
            current_chunk = ""
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    speaker_tag = parts[i]
                    content = parts[i + 1].strip()
                    turn = f"{speaker_tag}\n{content}"
                    
                    # Accumulate turns until reaching chunk size
                    if len(current_chunk) + len(turn) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = turn
                    else:
                        current_chunk += "\n\n" + turn if current_chunk else turn
                        
            if current_chunk:
                chunks.append(current_chunk.strip())
                
        elif format_type == "speaker_colon":
            # Split by speaker name pattern
            pattern = r'^([A-Za-z\s]+\d*:)'
            lines = transcript.split('\n')
            
            current_chunk = ""
            current_speaker = ""
            current_content = []
            
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    # New speaker detected
                    if current_speaker and current_content:
                        turn = f"{current_speaker} {' '.join(current_content)}"
                        
                        if len(current_chunk) + len(turn) > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = turn
                        else:
                            current_chunk += "\n\n" + turn if current_chunk else turn
                            
                    current_speaker = match.group(1)
                    current_content = [line[len(current_speaker):].strip()]
                else:
                    if line.strip():
                        current_content.append(line.strip())
                        
            # Add final turn
            if current_speaker and current_content:
                turn = f"{current_speaker} {' '.join(current_content)}"
                if current_chunk:
                    current_chunk += "\n\n" + turn
                else:
                    current_chunk = turn
                chunks.append(current_chunk.strip())
                
        return chunks
    
    def chunk_episode(self, episode: Episode) -> List[Document]:
        """Chunk a single episode into documents"""
        documents = []
        
        # Check if we should preserve speaker turns
        if self.preserve_speaker_turns:
            format_type = self._detect_speaker_format(episode.transcript)
            if format_type:
                logger.info(f"Detected {format_type} format for episode {episode.episode_number}")
                chunks = self._split_by_speaker_turns(episode.transcript, format_type)
            else:
                logger.info(f"No speaker format detected for episode {episode.episode_number}, using standard chunking")
                chunks = self.text_splitter.split_text(episode.transcript)
        else:
            chunks = self.text_splitter.split_text(episode.transcript)
            
        # Create documents with metadata
        for i, chunk in enumerate(chunks):
            metadata = episode.metadata.copy()
            metadata.update({
                "content_type": "episode",  # Add content type for consistency with books
                "chunk_index": i,
                "chunk_total": len(chunks),
                "chunk_type": "speaker_turn" if self.preserve_speaker_turns else "standard"
            })
            
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
            
        logger.info(f"Created {len(documents)} chunks for episode {episode.episode_number}")
        return documents
    
    def chunk_episodes(self, episodes: List[Episode]) -> List[Document]:
        """Chunk multiple episodes"""
        all_documents = []
        
        for episode in episodes:
            try:
                documents = self.chunk_episode(episode)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error chunking episode {episode.episode_number}: {e}")
                
        logger.info(f"Created {len(all_documents)} total chunks from {len(episodes)} episodes")
        return all_documents


def main():
    """Test chunking functionality"""
    import json
    logging.basicConfig(level=logging.INFO)
    
    # Load a sample episode
    sample_file = settings.episodes_dir / "episode_170.json"
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    from .episode_processor import Episode
    episode = Episode(data)
    
    # Test chunking
    chunker = TranscriptChunker()
    documents = chunker.chunk_episode(episode)
    
    print(f"\nCreated {len(documents)} chunks")
    print(f"\nFirst chunk preview:")
    print(documents[0].page_content[:500])
    print(f"\nMetadata: {documents[0].metadata}")


if __name__ == "__main__":
    main()