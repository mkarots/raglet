"""FAISS vector store implementation."""

import faiss
import numpy as np

from raglet.config.config import SearchConfig
from raglet.core.chunk import Chunk
from raglet.vector_store.interfaces import VectorStore


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

        # Ensure vectors are float32 and contiguous (FAISS requirement)
        vectors = np.ascontiguousarray(vectors.astype(np.float32))

        # Normalize vectors for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(vectors)

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

        # Ensure query is float32 and contiguous
        query_vector = np.ascontiguousarray(query_vector.astype(np.float32)).reshape(1, -1)

        # Normalize query vector for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(query_vector)

        # Search returns inner product scores (cosine similarity) and indices
        similarities, indices = self.index.search(query_vector, top_k)

        # Inner product with normalized vectors = cosine similarity (0-1 range)
        results = []
        for i, idx in enumerate(indices[0]):
            # FAISS returns -1 for invalid indices when there aren't enough vectors
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]

                # Inner product is already similarity score (higher = more similar)
                # Range: -1.0 to 1.0 (typically 0.0 to 1.0 for normalized vectors)
                score = float(similarities[0][i])

                # Create a copy with score set
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
