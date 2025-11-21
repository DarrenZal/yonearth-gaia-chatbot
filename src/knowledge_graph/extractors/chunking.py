"""
Transcript chunking utilities for knowledge graph extraction.

This module provides functions to split long transcripts into manageable chunks
for entity extraction, using accurate token counting with tiktoken.
"""

import tiktoken
from typing import List, Dict


def chunk_transcript(
    transcript: str,
    chunk_size: int = 500,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[Dict[str, any]]:
    """
    Chunk a transcript into overlapping segments based on token count.

    Args:
        transcript: The full transcript text to chunk
        chunk_size: Target size for each chunk in tokens (default: 500)
        overlap: Number of overlapping tokens between chunks (default: 50)
        encoding_name: Tiktoken encoding to use (default: "cl100k_base" for GPT-4)

    Returns:
        List of dicts containing chunk text and metadata:
        [
            {
                "text": str,
                "chunk_index": int,
                "start_token": int,
                "end_token": int,
                "token_count": int
            },
            ...
        ]

    Example:
        >>> chunks = chunk_transcript(transcript, chunk_size=500, overlap=50)
        >>> print(f"Created {len(chunks)} chunks")
        >>> print(f"First chunk has {chunks[0]['token_count']} tokens")
    """
    # Load the tokenizer
    encoding = tiktoken.get_encoding(encoding_name)

    # Tokenize the entire transcript
    tokens = encoding.encode(transcript)
    total_tokens = len(tokens)

    chunks = []
    chunk_index = 0
    start_pos = 0

    while start_pos < total_tokens:
        # Calculate end position for this chunk
        end_pos = min(start_pos + chunk_size, total_tokens)

        # Extract tokens for this chunk
        chunk_tokens = tokens[start_pos:end_pos]

        # Decode tokens back to text
        chunk_text = encoding.decode(chunk_tokens)

        # Create chunk metadata
        chunk_data = {
            "text": chunk_text,
            "chunk_index": chunk_index,
            "start_token": start_pos,
            "end_token": end_pos,
            "token_count": len(chunk_tokens)
        }

        chunks.append(chunk_data)

        # Move to next chunk with overlap
        # For the last chunk, we're done
        if end_pos >= total_tokens:
            break

        # Otherwise, move forward by (chunk_size - overlap)
        start_pos += (chunk_size - overlap)
        chunk_index += 1

    return chunks


def get_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Get the token count for a given text.

    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding to use

    Returns:
        Number of tokens in the text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))
