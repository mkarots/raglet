"""Interfaces for document processing."""

from abc import ABC, abstractmethod
from typing import Any

from raglet.core.chunk import Chunk


class DocumentExtractor(ABC):
    """Interface for extracting text from files."""

    @abstractmethod
    def extract(self, file_path: str) -> str:
        """Extract text from file.

        Args:
            file_path: Path to the file to extract text from

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        pass

    @abstractmethod
    def can_extract(self, file_path: str) -> bool:
        """Check if this extractor can handle the file.

        Args:
            file_path: Path to the file

        Returns:
            True if this extractor can handle the file
        """
        pass


class Chunker(ABC):
    """Interface for chunking text."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        pass
