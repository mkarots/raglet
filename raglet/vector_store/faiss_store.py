"""FAISS vector store implementation."""

import faiss
import numpy as np

from raglet.config.config import SearchConfig
from raglet.core.chunk import Chunk
from raglet.vector_store.interfaces import VectorStore


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows in-place and return the array.

    Pure numpy replacement for faiss.normalize_L2, which segfaults on
    macOS ARM64 (Apple Silicon) with certain faiss-cpu builds when the
    array is large enough to trigger OpenMP threading.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vectors /= norms
    return vectors


class FAISSVectorStore(VectorStore):
    """Vector store using FAISS with cosine similarity (IndexFlatIP)."""

    def __init__(self, embedding_dim: int, config: SearchConfig):
        """Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embeddings
            config: Search configuration
        """
        self.config = config
        self.config.validate()

        self.embedding_dim = embedding_dim

        # IndexFlatIP (Inner Product) with normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: list[Chunk] = []

    def add_vectors(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add vectors to the store with associated chunks.

        Args:
            vectors: NumPy array of shape (n_vectors, embedding_dim)
            chunks: List of Chunk objects corresponding to vectors

        Raises:
            ValueError: If vectors and chunks have different lengths
        """
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Vectors ({len(vectors)}) and chunks ({len(chunks)}) " "must have the same length"
            )

        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) does not match "
                f"store dimension ({self.embedding_dim})"
            )

        if vectors.dtype != np.float32 or not vectors.flags['C_CONTIGUOUS']:
            vectors = np.asarray(vectors, dtype=np.float32, order='C')

        _normalize_l2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Chunk]:
        """Search for similar vectors.

        Args:
            query_vector: NumPy array of shape (embedding_dim,)
            top_k: Number of results to return

        Returns:
            List of Chunk objects with score attribute set, sorted by similarity
            (most similar first). Returns empty list if store is empty.
        """
        if self.get_count() == 0:
            return []

        if query_vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query vector dimension ({query_vector.shape[0]}) does not "
                f"match store dimension ({self.embedding_dim})"
            )

        if query_vector.dtype != np.float32 or not query_vector.flags['C_CONTIGUOUS']:
            query_vector = np.asarray(query_vector, dtype=np.float32, order='C')
        query_vector = query_vector.reshape(1, -1)

        _normalize_l2(query_vector)

        similarities, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                score = float(similarities[0][i])
                result_chunk = Chunk(
                    text=chunk.text,
                    source=chunk.source,
                    index=chunk.index,
                    metadata=chunk.metadata.copy(),
                    score=score,
                )
                results.append(result_chunk)

        return results

    def get_count(self) -> int:
        """Get the number of vectors stored.

        Returns:
            Number of vectors in the store
        """
        return int(self.index.ntotal)

    def get_all_vectors(self) -> np.ndarray:
        """Retrieve all indexed vectors from the FAISS index.

        Uses ``index.reconstruct_n()`` to bulk-read vectors directly from
        the C++ index without keeping a Python-side copy.

        Returns:
            Contiguous float32 array of shape (ntotal, embedding_dim).
            Empty (0, embedding_dim) array when the index is empty.
        """
        n = int(self.index.ntotal)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)  # type: ignore[no-any-return]
        return self.index.reconstruct_n(0, n)  # type: ignore[no-any-return]

    def reset(self) -> None:
        """Reset the vector store (clear all vectors and chunks).

        Useful for cleanup and preventing resource accumulation across iterations.
        This explicitly clears the FAISS index, which may help with OpenMP thread cleanup.
        """
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks.clear()
