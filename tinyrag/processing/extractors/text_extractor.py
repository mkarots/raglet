"""Text file extractor."""

import os
from pathlib import Path

from tinyrag.processing.interfaces import DocumentExtractor


class TextExtractor(DocumentExtractor):
    """Extracts text from .txt files."""

    def __init__(self, encoding: str = "utf-8"):
        """Initialize text extractor.

        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding

    def can_extract(self, file_path: str) -> bool:
        """Check if file is a text file."""
        path = Path(file_path)
        return path.suffix.lower() == ".txt" or path.suffix == ""

    def extract(self, file_path: str) -> str:
        """Extract text from .txt file.

        Args:
            file_path: Path to text file

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, encoding=self.encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with error handling
            with open(file_path, encoding=self.encoding, errors="replace") as f:
                return f.read()
