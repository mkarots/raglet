"""Text chunker implementation."""

import re
from typing import Any

from tinyrag.config.config import ChunkingConfig
from tinyrag.core.chunk import Chunk
from tinyrag.processing.interfaces import Chunker


class SentenceAwareChunker(Chunker):
    """Chunks text with sentence-aware strategy."""

    def __init__(self, config: ChunkingConfig):
        """Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Split text into chunks using sentence-aware strategy.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        # Approximate token count (rough: 1 token ≈ 4 chars)
        target_chars = self.config.size * 4
        overlap_chars = self.config.overlap * 4

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed size, finalize current chunk
            if current_chunk and current_size + sentence_size > target_chars:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=metadata.get("source", ""),
                        index=chunk_index,
                        metadata=metadata.copy(),
                    )
                )
                chunk_index += 1

                # Start new chunk with overlap
                if overlap_chars > 0:
                    overlap_text = chunk_text[-overlap_chars:]
                    current_chunk = [overlap_text] if overlap_text.strip() else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=metadata.get("source", ""),
                    index=chunk_index,
                    metadata=metadata.copy(),
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved)
        # Split on sentence endings followed by space or newline
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)

        # Filter empty sentences
        return [s.strip() for s in sentences if s.strip()]
