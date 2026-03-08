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

# NOTE: SentenceTransformerGenerator and FAISSVectorStore are NOT imported
# eagerly here. They are imported lazily in rag.py's __init__ and from_files
# (torch before faiss) to preserve the correct OpenMP init order on macOS
# while allowing `import raglet` to succeed without torch/faiss installed.

__all__ = [
    "RAGlet",
    "Chunk",
    "RAGletConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "SearchConfig",
]
