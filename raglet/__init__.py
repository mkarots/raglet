"""raglet - Portable memory for small text corpora."""

__version__ = "0.1.0"

# CRITICAL: Import order matters for threading compatibility on macOS
# PyTorch (via sentence-transformers) must initialize BEFORE FAISS
# to prevent OpenMP threading conflicts. Importing these modules at
# package level ensures correct initialization order.
#
# These imports happen when 'raglet' is imported, ensuring:
# 1. sentence-transformers loads first (via generator.py module-level import)
# 2. faiss loads second (via this import)
# This prevents segfaults when creating multiple RAGlet instances.
from raglet.config.config import (
    ChunkingConfig,
    EmbeddingConfig,
    RAGletConfig,
    SearchConfig,
)
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.embeddings.generator import SentenceTransformerGenerator  # noqa: F401
from raglet.vector_store.faiss_store import FAISSVectorStore  # noqa: F401

__all__ = [
    "RAGlet",
    "Chunk",
    "RAGletConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "SearchConfig",
]
