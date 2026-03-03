"""raglet - Portable memory for small text corpora."""

__version__ = "0.1.0"

from raglet.config.config import (
    ChunkingConfig,
    EmbeddingConfig,
    SearchConfig,
    TinyRAGConfig,
)
from raglet.core.chunk import Chunk
from raglet.core.rag import TinyRAG

__all__ = [
    "TinyRAG",
    "Chunk",
    "TinyRAGConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "SearchConfig",
]
