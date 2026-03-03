"""Unit tests for FAISS vector store."""

import numpy as np
import pytest

from raglet.config.config import SearchConfig
from raglet.core.chunk import Chunk
from raglet.vector_store.faiss_store import FAISSVectorStore


@pytest.mark.unit
class TestFAISSVectorStore:
    """Test FAISSVectorStore."""

    def test_init(self):
        """Test initialization."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)
        assert store.get_count() == 0

    def test_add_vectors(self):
        """Test adding vectors to store."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)

        vectors = np.random.rand(3, 384).astype(np.float32)
        chunks = [Chunk(text=f"Chunk {i}", source="test.txt", index=i) for i in range(3)]

        store.add_vectors(vectors, chunks)
        assert store.get_count() == 3

    def test_add_vectors_mismatch_length(self):
        """Test error when vectors and chunks have different lengths."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)

        vectors = np.random.rand(3, 384).astype(np.float32)
        chunks = [Chunk(text="One", source="test.txt", index=0)]

        with pytest.raises(ValueError, match="must have the same length"):
            store.add_vectors(vectors, chunks)

    def test_add_vectors_wrong_dimension(self):
        """Test error when vector dimension doesn't match."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)

        vectors = np.random.rand(2, 256).astype(np.float32)  # Wrong dimension
        chunks = [Chunk(text=f"Chunk {i}", source="test.txt", index=i) for i in range(2)]

        with pytest.raises(ValueError, match="does not match"):
            store.add_vectors(vectors, chunks)

    def test_search_empty_store(self):
        """Test search on empty store returns empty list."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)

        query = np.random.rand(384).astype(np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_search(self):
        """Test search returns relevant results."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)

        # Add some vectors
        vectors = np.random.rand(5, 384).astype(np.float32)
        chunks = [Chunk(text=f"Chunk {i}", source="test.txt", index=i) for i in range(5)]
        store.add_vectors(vectors, chunks)

        # Search with query vector
        query = np.random.rand(384).astype(np.float32)
        results = store.search(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(chunk, Chunk) for chunk in results)
        assert all(chunk.score is not None for chunk in results)
        # Results should be sorted by similarity (highest score first)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_wrong_dimension(self):
        """Test error when query dimension doesn't match."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)

        vectors = np.random.rand(2, 384).astype(np.float32)
        chunks = [Chunk(text=f"Chunk {i}", source="test.txt", index=i) for i in range(2)]
        store.add_vectors(vectors, chunks)

        query = np.random.rand(256).astype(np.float32)  # Wrong dimension
        with pytest.raises(ValueError, match="does not match"):
            store.search(query, top_k=5)

    def test_get_count(self):
        """Test get_count returns correct number."""
        config = SearchConfig()
        store = FAISSVectorStore(embedding_dim=384, config=config)

        assert store.get_count() == 0

        vectors = np.random.rand(3, 384).astype(np.float32)
        chunks = [Chunk(text=f"Chunk {i}", source="test.txt", index=i) for i in range(3)]
        store.add_vectors(vectors, chunks)

        assert store.get_count() == 3
