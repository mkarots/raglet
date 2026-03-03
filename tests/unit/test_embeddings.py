"""Unit tests for embedding generator."""

import numpy as np
import pytest

from raglet.config.config import EmbeddingConfig
from raglet.core.chunk import Chunk
from raglet.embeddings.generator import SentenceTransformerGenerator


@pytest.mark.unit
class TestSentenceTransformerGenerator:
    """Test SentenceTransformerGenerator."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        config = EmbeddingConfig()
        generator = SentenceTransformerGenerator(config)
        assert generator.get_dimension() == 384  # all-MiniLM-L6-v2 dimension

    def test_get_dimension(self):
        """Test get_dimension returns correct dimension."""
        config = EmbeddingConfig(model="all-MiniLM-L6-v2")
        generator = SentenceTransformerGenerator(config)
        dimension = generator.get_dimension()
        assert isinstance(dimension, int)
        assert dimension > 0

    def test_generate_single(self):
        """Test generating embedding for single text."""
        config = EmbeddingConfig()
        generator = SentenceTransformerGenerator(config)
        embedding = generator.generate_single("test text")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (generator.get_dimension(),)
        assert embedding.dtype == np.float32

    def test_generate_empty_list(self):
        """Test generating embeddings for empty chunk list."""
        config = EmbeddingConfig()
        generator = SentenceTransformerGenerator(config)
        embeddings = generator.generate([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, generator.get_dimension())

    def test_generate_chunks(self):
        """Test generating embeddings for chunks."""
        config = EmbeddingConfig()
        generator = SentenceTransformerGenerator(config)

        chunks = [
            Chunk(text="First chunk", source="test.txt", index=0),
            Chunk(text="Second chunk", source="test.txt", index=1),
        ]

        embeddings = generator.generate(chunks)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(chunks), generator.get_dimension())
        assert embeddings.dtype == np.float32

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        config = EmbeddingConfig(batch_size=2)
        generator = SentenceTransformerGenerator(config)

        # Create more chunks than batch size
        chunks = [Chunk(text=f"Chunk {i}", source="test.txt", index=i) for i in range(5)]

        embeddings = generator.generate(chunks)
        assert embeddings.shape[0] == len(chunks)

    def test_invalid_model(self):
        """Test error handling for invalid model name."""
        config = EmbeddingConfig(model="nonexistent-model-xyz")
        with pytest.raises(ValueError, match="Failed to load"):
            SentenceTransformerGenerator(config)
