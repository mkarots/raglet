"""Markdown file extractor."""

import os
from pathlib import Path

from raglet.processing.interfaces import DocumentExtractor


class MarkdownExtractor(DocumentExtractor):
    """Extracts text from .md and .markdown files."""

    def __init__(self, encoding: str = "utf-8"):
        """Initialize markdown extractor.

        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding

    def can_extract(self, file_path: str) -> bool:
        """Check if file is a markdown file."""
        path = Path(file_path)
        return path.suffix.lower() in [".md", ".markdown"]

    def extract(self, file_path: str) -> str:
        """Extract text from markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            File contents as string (raw markdown)

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
