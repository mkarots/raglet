"""Text file extractor."""

import os
from pathlib import Path

from raglet.processing.interfaces import DocumentExtractor


class TextExtractor(DocumentExtractor):
    """Extracts text from text-based files.

    Handles .txt files and serves as a fallback for any file type that doesn't
    have a specific extractor (e.g., source code files like .py, .js, etc.).
    Files are read as plain text with UTF-8 encoding.
    """

    def __init__(self, encoding: str = "utf-8"):
        """Initialize text extractor.

        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding

    def can_extract(self, file_path: str) -> bool:
        """Check if file is a text file.

        Returns True for .txt files and files without extension.
        Note: TextExtractor can extract any text-based file, but this method
        only returns True for .txt files to allow other extractors to handle
        their specific formats first. The factory falls back to TextExtractor
        for unknown file types.
        """
        path = Path(file_path)
        return path.suffix.lower() == ".txt" or path.suffix == ""

    def extract(self, file_path: str) -> str:
        """Extract text from any text-based file.

        Args:
            file_path: Path to text file (can be any text-based format)

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is invalid (falls back to error replacement)
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
