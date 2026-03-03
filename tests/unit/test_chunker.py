"""Unit tests for chunker."""

from raglet.config.config import ChunkingConfig
from raglet.processing.chunker import SentenceAwareChunker


class TestSentenceAwareChunker:
    """Test SentenceAwareChunker."""

    def test_chunk_simple_text(self):
        """Test chunking simple text."""
        config = ChunkingConfig(size=100, overlap=10)
        chunker = SentenceAwareChunker(config)

        text = "This is sentence one. This is sentence two. This is sentence three."
        metadata = {"source": "test.txt"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) > 0
        assert all(chunk.source == "test.txt" for chunk in chunks)
        assert all(chunk.metadata == metadata for chunk in chunks)

    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list."""
        config = ChunkingConfig()
        chunker = SentenceAwareChunker(config)

        chunks = chunker.chunk("", {})
        assert chunks == []

    def test_chunk_preserves_metadata(self):
        """Test that chunker preserves metadata."""
        config = ChunkingConfig(size=50, overlap=5)
        chunker = SentenceAwareChunker(config)

        text = "Short sentence."
        metadata = {"source": "test.txt", "custom": "value"}

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) > 0
        assert all(chunk.metadata["custom"] == "value" for chunk in chunks)

    def test_chunk_indexing(self):
        """Test that chunks have sequential indices."""
        config = ChunkingConfig(size=50, overlap=5)
        chunker = SentenceAwareChunker(config)

        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunker.chunk(text, {})

        indices = [chunk.index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_size_respects_config(self):
        """Test that chunks respect size configuration."""
        config = ChunkingConfig(size=20, overlap=5)
        chunker = SentenceAwareChunker(config)

        # Create text that should produce multiple chunks
        text = ". ".join([f"Sentence {i}" for i in range(10)])

        chunks = chunker.chunk(text, {})

        # Approximate: each chunk should be around target size
        # (allowing for sentence boundaries)
        assert len(chunks) > 1
