"""Text file extractor."""

import os
from pathlib import Path

from raglet.processing.interfaces import DocumentExtractor


class TextExtractor(DocumentExtractor):
    """Extracts text from text-based files.

    Handles all text files including:
    - .txt files
    - Source code files (.py, .js, .ts, .java, .cpp, .go, .rs, etc.)
    - Config files (Makefile, Dockerfile, .yaml, .json, .toml, etc.)
    - Markdown files (.md) - though MarkdownExtractor handles these first
    - Any other UTF-8 text file

    Files are read as plain text with UTF-8 encoding. Binary formats (PDF, DOCX, etc.)
    are not supported - raglet focuses on text files only.
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
        only returns True for .txt files to allow other extractors (like MarkdownExtractor)
        to handle their specific formats first. The factory falls back to TextExtractor
        for all other text file types (source code, config files, etc.).
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

        # Use errors="replace" by default to avoid double file read on encoding errors
        # This is more efficient and handles invalid UTF-8 gracefully
        with open(file_path, encoding=self.encoding, errors="replace") as f:
            return f.read()
