"""Interfaces for embedding generation."""

from abc import ABC, abstractmethod

import numpy as np

from raglet.core.chunk import Chunk


class EmbeddingGenerator(ABC):
    """Interface for generating embeddings from text."""

    @abstractmethod
    def generate(self, chunks: list[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks.

        Args:
            chunks: List of Chunk objects to generate embeddings for

        Returns:
            NumPy array of shape (len(chunks), embedding_dim) with embeddings
        """
        pass

    @abstractmethod
    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Text string to generate embedding for

        Returns:
            NumPy array of shape (embedding_dim,) with embedding
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        pass
