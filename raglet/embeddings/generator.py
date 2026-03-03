"""Embedding generator implementation using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

from raglet.config.config import EmbeddingConfig
from raglet.core.chunk import Chunk
from raglet.embeddings.interfaces import EmbeddingGenerator


class SentenceTransformerGenerator(EmbeddingGenerator):
    """Generates embeddings using sentence-transformers models."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding generator.

        Args:
            config: Embedding configuration

        Raises:
            ValueError: If model cannot be loaded
        """
        self.config = config
        self.config.validate()

        try:
            self.model = SentenceTransformer(config.model, device=config.device)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise ValueError(f"Failed to load embedding model '{config.model}': {e}") from e

    def generate(self, chunks: list[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks.

        Args:
            chunks: List of Chunk objects to generate embeddings for

        Returns:
            NumPy array of shape (len(chunks), embedding_dim) with embeddings
        """
        if not chunks:
            return np.array([]).reshape(0, self._dimension)  # type: ignore[no-any-return]

        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )

        return np.array(embeddings, dtype=np.float32)  # type: ignore[no-any-return]

    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Text string to generate embedding for

        Returns:
            NumPy array of shape (embedding_dim,) with embedding
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)  # type: ignore[no-any-return]

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return int(self._dimension)
