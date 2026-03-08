"""Unit tests for deferred embedding materialisation (ADR 009).

Tests the list-of-arrays internal storage and lazy vstack behaviour
that replaced the eager np.vstack in add_chunks().
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from raglet.config.config import RAGletConfig, SearchConfig
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.vector_store.faiss_store import FAISSVectorStore


def _make_chunks(n: int, offset: int = 0) -> list[Chunk]:
    return [
        Chunk(text=f"chunk {offset + i}", source="test.txt", index=offset + i) for i in range(n)
    ]


def _make_embeddings(n: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _mock_generator(dim: int = 16):
    gen = MagicMock()
    gen.get_dimension.return_value = dim
    gen.generate.side_effect = lambda chunks: _make_embeddings(len(chunks), dim)
    gen.generate_single.side_effect = lambda text: _make_embeddings(1, dim)[0]
    return gen


@pytest.mark.unit
class TestEmbeddingsProperty:
    """Test the embeddings property with lazy materialisation."""

    def test_empty_raglet_returns_empty_array(self):
        """Empty RAGlet returns (0, dim) array without error."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        raglet = RAGlet(
            chunks=[],
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )
        assert raglet.embeddings.shape == (0, 16)
        assert raglet.embeddings.dtype == np.float32

    def test_init_with_provided_embeddings(self):
        """Provided embeddings are stored and accessible via property."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        emb = _make_embeddings(3, 16)
        chunks = _make_chunks(3)

        raglet = RAGlet(
            chunks=chunks,
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
            embeddings=emb,
        )

        assert raglet.embeddings.shape == (3, 16)
        np.testing.assert_array_equal(raglet.embeddings, emb)

    def test_init_generates_embeddings_when_none_provided(self):
        """When embeddings=None, generator is called and result is accessible."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        chunks = _make_chunks(5)

        raglet = RAGlet(
            chunks=chunks,
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )

        assert raglet.embeddings.shape == (5, 16)
        gen.generate.assert_called_once_with(chunks)

    def test_setter_backward_compat(self):
        """Direct assignment to .embeddings works for backward compatibility."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        raglet = RAGlet(
            chunks=[],
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )

        new_emb = _make_embeddings(2, 16)
        raglet.embeddings = new_emb
        np.testing.assert_array_equal(raglet.embeddings, new_emb)


@pytest.mark.unit
class TestDeferredMaterialisation:
    """Test the O(1) append and deferred vstack pattern."""

    def test_add_chunks_does_not_vstack_immediately(self):
        """After add_chunks, _embedding_chunks has multiple arrays (no eager vstack)."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        initial = _make_chunks(3)
        raglet = RAGlet(
            chunks=initial,
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )

        # At this point _embedding_chunks should have exactly 1 array
        assert len(raglet._embedding_chunks) == 1

        raglet.add_chunks(_make_chunks(2, offset=3))
        # After add_chunks, a second array is appended — no vstack yet
        assert len(raglet._embedding_chunks) == 2

    def test_property_access_materialises(self):
        """Accessing .embeddings collapses _embedding_chunks into one array."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        raglet = RAGlet(
            chunks=_make_chunks(3),
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )
        raglet.add_chunks(_make_chunks(2, offset=3))
        assert len(raglet._embedding_chunks) == 2

        # Access the property
        emb = raglet.embeddings
        assert emb.shape == (5, 16)
        # Now internal list should be collapsed to 1
        assert len(raglet._embedding_chunks) == 1

    def test_multiple_add_chunks_then_materialise(self):
        """Multiple add_chunks accumulate arrays; materialisation merges all."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        raglet = RAGlet(
            chunks=_make_chunks(2),
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )

        for i in range(5):
            offset = 2 + i * 3
            raglet.add_chunks(_make_chunks(3, offset=offset))

        assert len(raglet._embedding_chunks) == 6  # 1 initial + 5 appends
        assert raglet.embeddings.shape == (17, 16)  # 2 + 5*3
        assert len(raglet._embedding_chunks) == 1

    def test_second_access_is_free(self):
        """After materialisation, repeated access returns the same array."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        raglet = RAGlet(
            chunks=_make_chunks(3),
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )
        raglet.add_chunks(_make_chunks(2, offset=3))

        first = raglet.embeddings
        second = raglet.embeddings
        assert first is second  # same object, no re-materialisation

    def test_close_clears_embedding_chunks(self):
        """close() empties _embedding_chunks to free memory."""
        gen = _mock_generator(dim=16)
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        raglet = RAGlet(
            chunks=_make_chunks(3),
            config=RAGletConfig(),
            embedding_generator=gen,
            vector_store=store,
        )
        raglet.close()
        assert raglet._embedding_chunks == []


@pytest.mark.unit
class TestGetAllVectors:
    """Test FAISSVectorStore.get_all_vectors()."""

    def test_empty_store(self):
        """get_all_vectors on empty store returns (0, dim) array."""
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        result = store.get_all_vectors()
        assert result.shape == (0, 16)
        assert result.dtype == np.float32

    def test_returns_indexed_vectors(self):
        """get_all_vectors returns vectors that were added."""
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        emb = _make_embeddings(5, 16)
        chunks = _make_chunks(5)

        store.add_vectors(emb, chunks)
        result = store.get_all_vectors()

        assert result.shape == (5, 16)
        assert result.dtype == np.float32

    def test_vectors_match_after_normalisation(self):
        """Reconstructed vectors match the L2-normalised originals.

        FAISS normalises vectors on add (for IndexFlatIP / cosine similarity),
        so get_all_vectors returns the normalised form, not the raw input.
        """
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        emb = _make_embeddings(3, 16).copy()
        expected = emb.copy()
        # Manually normalise to get the expected output
        norms = np.linalg.norm(expected, axis=1, keepdims=True)
        expected = expected / norms

        chunks = _make_chunks(3)
        store.add_vectors(emb, chunks)

        result = store.get_all_vectors()
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_incremental_add_then_retrieve(self):
        """Vectors from multiple add_vectors calls are all retrievable."""
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())

        for batch in range(3):
            emb = _make_embeddings(4, 16)
            chunks = _make_chunks(4, offset=batch * 4)
            store.add_vectors(emb, chunks)

        result = store.get_all_vectors()
        assert result.shape == (12, 16)

    def test_reset_then_get_all_vectors(self):
        """After reset, get_all_vectors returns empty array."""
        store = FAISSVectorStore(embedding_dim=16, config=SearchConfig())
        store.add_vectors(_make_embeddings(3, 16), _make_chunks(3))
        store.reset()

        result = store.get_all_vectors()
        assert result.shape == (0, 16)
