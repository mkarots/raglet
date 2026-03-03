"""raglet - Portable memory for small text corpora."""

__version__ = "0.1.0"

from raglet.config.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RAGletConfig,
    SearchConfig,
)
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet

__all__ = [
    "RAGlet",
    "Chunk",
    "RAGletConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "SearchConfig",
]
