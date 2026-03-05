"""Storage backend interfaces."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np

    from raglet.core.chunk import Chunk
    from raglet.core.rag import RAGlet


class StorageBackend(ABC):
    """Abstract storage backend interface.

    Storage backends handle persistence of RAGlet instances.
    Core RAGlet operations don't depend on storage format.
    """

    @abstractmethod
    def save(
        self,
        raglet: "RAGlet",
        file_path: str,
        incremental: bool = False,
    ) -> None:
        """Save RAGlet to storage.

        Args:
            raglet: RAGlet instance to save
            file_path: Path to save file
            incremental: If True, append new chunks (if supported)
                        If False, full save (replace existing)

        Raises:
            ValueError: If save fails
            IOError: If file operations fail
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> "RAGlet":
        """Load RAGlet from storage.

        Args:
            file_path: Path to load file

        Returns:
            RAGlet instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            IOError: If file operations fail
        """
        pass

    @abstractmethod
    def supports_incremental(self) -> bool:
        """Check if backend supports incremental updates.

        Returns:
            True if backend supports incremental saves, False otherwise
        """
        pass

    @abstractmethod
    def add_chunks(
        self,
        file_path: str,
        chunks: list["Chunk"],
        embeddings: "np.ndarray",
        raglet: Optional["RAGlet"] = None,
    ) -> None:
        """Add chunks incrementally to existing storage.

        Args:
            file_path: Path to storage file
            chunks: New chunks to add
            embeddings: Embeddings for new chunks
            raglet: Optional RAGlet instance (for context)

        Raises:
            ValueError: If incremental updates not supported
            IOError: If file operations fail
        """
        pass
