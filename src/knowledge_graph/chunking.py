"""
Parent-Child Chunking Module for Knowledge Graph Extraction

This module implements a "Greedy Accumulator" chunking strategy that respects
semantic boundaries (paragraphs, speaker turns, chapter headers) while creating
parent chunks for extraction and child chunks for RAG vector indexing.

Parent chunks: ~3,000 tokens (soft target), max 6,000 tokens (hard limit)
Child chunks: ~600 tokens with 100 token overlap, strictly nested within parents
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import tiktoken


@dataclass
class ParentChunk:
    """A parent chunk used for knowledge graph extraction"""
    id: str                    # "episode_120_parent_0" or "book_viriditas_parent_0"
    content: str               # ~3,000 tokens (semantic boundaries)
    source_type: str           # "episode" or "book"
    source_id: str             # episode_number or book_slug
    token_count: int           # Actual token count
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    """A child chunk used for RAG vector indexing"""
    id: str                    # "episode_120_parent_0_child_0"
    parent_id: str             # Links to parent (MUST be within same parent)
    content: str               # ~600 tokens
    start_offset: int          # Character position within parent
    end_offset: int            # Character end position within parent
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParentChildChunker:
    """
    Chunker that creates parent chunks for extraction and child chunks for RAG.

    Uses "Greedy Accumulator" algorithm with semantic boundaries:
    - Accumulates natural units (paragraphs, speaker turns) until target size
    - Respects hard boundaries (chapter headers in books)
    - Force-splits oversized units at sentence boundaries
    """

    def __init__(
        self,
        parent_target: int = 3000,
        parent_max: int = 6000,
        force_split_threshold: int = 1000,
        child_size: int = 600,
        child_overlap: int = 100
    ):
        """
        Initialize the chunker with size parameters.

        Args:
            parent_target: Soft target for parent chunk size in tokens
            parent_max: Hard limit for parent chunk size in tokens
            force_split_threshold: Force split speaker turns larger than this
            child_size: Target size for child chunks in tokens
            child_overlap: Overlap between child chunks in tokens
        """
        self.parent_target = parent_target
        self.parent_max = parent_max
        self.force_split_threshold = force_split_threshold
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def create_parent_chunks(
        self,
        text: str,
        source_type: str,
        source_id: str
    ) -> List[ParentChunk]:
        """
        Create parent chunks from text using Greedy Accumulator algorithm.

        Algorithm:
        1. Split text into smallest natural units (paragraphs or speaker turns)
        2. Accumulate units into buffer
        3. If buffer + next_unit > TARGET_SIZE: close chunk, start new
        4. Edge case: If single unit > MAX_SIZE, force split at sentence boundary
        5. Safety valve: If speaker turn > FORCE_SPLIT_THRESHOLD, split at paragraph

        Args:
            text: Full text content to chunk
            source_type: "episode" or "book"
            source_id: Episode number or book slug

        Returns:
            List of ParentChunk objects
        """
        if not text or not text.strip():
            return []

        # Detect and split by natural units
        if source_type == "book":
            units = self._split_book_units(text)
        else:  # episode
            units = self._split_podcast_units(text)

        if not units:
            return []

        chunks = []
        buffer: List[str] = []
        buffer_tokens = 0
        chunk_index = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            # Safety valve: Force split long speaker turns at paragraph breaks
            if unit_tokens > self.force_split_threshold:
                sub_units = self._split_at_paragraphs(unit)
                for sub_unit in sub_units:
                    sub_tokens = self.count_tokens(sub_unit)
                    buffer, buffer_tokens, chunk_index = self._accumulate(
                        buffer, buffer_tokens, sub_unit, sub_tokens,
                        chunks, source_type, source_id, chunk_index
                    )
                continue

            # Hard stop: New chapter in book
            if self._is_chapter_header(unit) and buffer:
                chunks.append(self._close_chunk(buffer, source_type, source_id, chunk_index))
                chunk_index += 1
                buffer = [unit]
                buffer_tokens = unit_tokens
                continue

            buffer, buffer_tokens, chunk_index = self._accumulate(
                buffer, buffer_tokens, unit, unit_tokens,
                chunks, source_type, source_id, chunk_index
            )

        # Close final buffer
        if buffer:
            chunks.append(self._close_chunk(buffer, source_type, source_id, chunk_index))

        return chunks

    def _accumulate(
        self,
        buffer: List[str],
        buffer_tokens: int,
        unit: str,
        unit_tokens: int,
        chunks: List[ParentChunk],
        source_type: str,
        source_id: str,
        chunk_index: int
    ) -> Tuple[List[str], int, int]:
        """
        Accumulate unit into buffer, close chunk if exceeds target.

        Returns:
            Tuple of (updated_buffer, updated_tokens, updated_index)
        """
        if buffer and (buffer_tokens + unit_tokens) > self.parent_target:
            chunks.append(self._close_chunk(buffer, source_type, source_id, chunk_index))
            chunk_index += 1
            buffer = [unit]
            buffer_tokens = unit_tokens
        else:
            buffer.append(unit)
            buffer_tokens += unit_tokens

        # Edge case: Single massive unit exceeds max
        if buffer_tokens > self.parent_max:
            forced = self._force_split_at_sentence(buffer, self.parent_max)
            for i, part in enumerate(forced[:-1]):
                chunks.append(self._close_chunk([part], source_type, source_id, chunk_index))
                chunk_index += 1
            buffer = [forced[-1]] if forced else []
            buffer_tokens = self.count_tokens(buffer[0]) if buffer else 0

        return buffer, buffer_tokens, chunk_index

    def _split_book_units(self, text: str) -> List[str]:
        """
        Split book text by paragraphs, preserving headers as separate units.

        Headers (lines starting with #) are kept as separate units to trigger
        chapter boundary detection.
        """
        units = []
        for para in text.split('\n\n'):
            para = para.strip()
            if para:
                units.append(para)
        return units

    def _split_podcast_units(self, text: str) -> List[str]:
        """
        Split podcast by speaker turns.

        Detects speaker patterns:
        - Timestamp format: "5:45 – speaker_name" or "5:45 - speaker_name"
        - Name format: "Speaker Name:" at start of line

        Falls back to paragraph splitting if no speaker patterns detected.
        """
        # Pattern for timestamp-based speaker turns
        timestamp_pattern = r'(\d+:\d+(?::\d+)?\s*[–\-]\s*\w+[^:]*:?)'

        # Pattern for name-based speaker turns (at start of line)
        name_pattern = r'^([A-Z][a-zA-Z\s]+:)'

        # Try timestamp format first
        parts = re.split(timestamp_pattern, text)
        if len(parts) > 1:
            return self._reconstruct_speaker_turns(parts)

        # Try name format
        parts = re.split(name_pattern, text, flags=re.MULTILINE)
        if len(parts) > 1:
            return self._reconstruct_speaker_turns(parts)

        # Fallback: split by paragraphs
        return self._split_at_paragraphs(text)

    def _reconstruct_speaker_turns(self, parts: List[str]) -> List[str]:
        """Reconstruct speaker turns from regex split parts"""
        units = []
        i = 0
        while i < len(parts):
            # Skip empty parts
            if not parts[i].strip():
                i += 1
                continue

            # If this looks like a speaker marker, combine with next part
            if i + 1 < len(parts) and (
                re.match(r'\d+:\d+', parts[i]) or
                re.match(r'^[A-Z][a-zA-Z\s]+:$', parts[i].strip())
            ):
                turn = parts[i] + (parts[i + 1] if i + 1 < len(parts) else '')
                units.append(turn.strip())
                i += 2
            else:
                if parts[i].strip():
                    units.append(parts[i].strip())
                i += 1
        return [u for u in units if u]

    def _split_at_paragraphs(self, text: str) -> List[str]:
        """Split text at paragraph breaks (double newlines)"""
        return [p.strip() for p in text.split('\n\n') if p.strip()]

    def _is_chapter_header(self, text: str) -> bool:
        """
        Check if text is a markdown chapter header.

        Matches: # Header, ## Header, ### Header
        """
        return bool(re.match(r'^#{1,3}\s+', text.strip()))

    def _force_split_at_sentence(self, buffer: List[str], max_tokens: int) -> List[str]:
        """
        Force split at sentence boundaries when exceeding max.

        Args:
            buffer: List of text units to combine and split
            max_tokens: Maximum tokens per resulting part

        Returns:
            List of text parts, each under max_tokens
        """
        combined = '\n\n'.join(buffer)

        # Split on sentence-ending punctuation followed by space
        sentences = re.split(r'([.!?]+\s+)', combined)

        result = []
        current = []
        current_tokens = 0

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            # Add back the punctuation + space if present
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > max_tokens and current:
                result.append(''.join(current))
                current = [sentence]
                current_tokens = sent_tokens
            else:
                current.append(sentence)
                current_tokens += sent_tokens

        if current:
            result.append(''.join(current))

        return result if result else [combined]

    def _close_chunk(
        self,
        buffer: List[str],
        source_type: str,
        source_id: str,
        index: int
    ) -> ParentChunk:
        """Close buffer into a ParentChunk"""
        content = '\n\n'.join(buffer)
        return ParentChunk(
            id=f"{source_type}_{source_id}_parent_{index}",
            content=content,
            source_type=source_type,
            source_id=source_id,
            token_count=self.count_tokens(content),
            metadata={"chunk_index": index}
        )

    def create_child_chunks(self, parent: ParentChunk) -> List[ChildChunk]:
        """
        Split parent into ~600 token child chunks.

        CRITICAL: Child chunks MUST strictly nest inside parent boundaries.
        Do NOT borrow text from adjacent parents to fill the window.
        When the window hits the end of the Parent, it stops.

        Args:
            parent: The parent chunk to split into children

        Returns:
            List of ChildChunk objects, each strictly within parent boundaries
        """
        children = []
        text = parent.content

        if not text:
            return children

        # Calculate character-based window from token targets
        chars_per_token = len(text) / max(parent.token_count, 1)
        window_chars = int(self.child_size * chars_per_token)
        overlap_chars = int(self.child_overlap * chars_per_token)
        step = max(window_chars - overlap_chars, 1)

        start = 0
        child_index = 0

        while start < len(text):
            end = min(start + window_chars, len(text))

            # Try to end at a sentence boundary for cleaner chunks
            if end < len(text):
                # Look for sentence-ending punctuation
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclaim = text.rfind('!', start, end)
                best_break = max(last_period, last_question, last_exclaim)

                # Only use the break if it's in the latter half of the chunk
                if best_break > start + window_chars // 2:
                    end = best_break + 1

            chunk_content = text[start:end].strip()

            if chunk_content:
                children.append(ChildChunk(
                    id=f"{parent.id}_child_{child_index}",
                    parent_id=parent.id,
                    content=chunk_content,
                    start_offset=start,
                    end_offset=end,
                    metadata={
                        "parent_index": parent.metadata.get("chunk_index"),
                        "child_index": child_index,
                        "source_type": parent.source_type,
                        "source_id": parent.source_id
                    }
                ))
                child_index += 1

            start = start + step

            # STRICT NESTING: Stop at parent boundary
            if start >= len(text):
                break

        return children

    def process_content(
        self,
        text: str,
        source_type: str,
        source_id: str
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Full pipeline: create parents, then children with linkage.

        Args:
            text: Full text content to process
            source_type: "episode" or "book"
            source_id: Episode number or book slug

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        parents = self.create_parent_chunks(text, source_type, source_id)

        all_children = []
        for parent in parents:
            children = self.create_child_chunks(parent)
            all_children.extend(children)

        return parents, all_children


def create_chunker_from_settings() -> ParentChildChunker:
    """
    Create a chunker configured from environment settings.

    Uses settings from src.config.settings:
    - parent_chunk_size
    - parent_chunk_max
    - child_chunk_size
    - child_chunk_overlap
    """
    from ..config import settings

    return ParentChildChunker(
        parent_target=settings.parent_chunk_size,
        parent_max=settings.parent_chunk_max,
        child_size=settings.child_chunk_size,
        child_overlap=settings.child_chunk_overlap
    )
