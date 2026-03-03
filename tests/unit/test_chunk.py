"""Unit tests for Chunk model."""

from raglet.core.chunk import Chunk


class TestChunk:
    """Test Chunk domain model."""

    def test_create_chunk(self):
        """Test creating a chunk."""
        chunk = Chunk(text="Test text", source="test.txt", index=0, metadata={"key": "value"})

        assert chunk.text == "Test text"
        assert chunk.source == "test.txt"
        assert chunk.index == 0
        assert chunk.metadata == {"key": "value"}
        assert chunk.score is None

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = Chunk(
            text="Test text", source="test.txt", index=0, metadata={"key": "value"}, score=0.95
        )

        result = chunk.to_dict()

        assert result["text"] == "Test text"
        assert result["source"] == "test.txt"
        assert result["index"] == 0
        assert result["metadata"] == {"key": "value"}
        assert result["score"] == 0.95

    def test_chunk_from_dict(self):
        """Test creating chunk from dictionary."""
        data = {
            "text": "Test text",
            "source": "test.txt",
            "index": 0,
            "metadata": {"key": "value"},
            "score": 0.95,
        }

        chunk = Chunk.from_dict(data)

        assert chunk.text == "Test text"
        assert chunk.source == "test.txt"
        assert chunk.index == 0
        assert chunk.metadata == {"key": "value"}
        assert chunk.score == 0.95

    def test_chunk_default_metadata(self):
        """Test chunk with default metadata."""
        chunk = Chunk(text="Test", source="test.txt", index=0)

        assert chunk.metadata == {}
        assert chunk.score is None
