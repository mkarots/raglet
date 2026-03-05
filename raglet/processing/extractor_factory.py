"""Factory for document extractors."""

from typing import Optional

from raglet.processing.extractors import MarkdownExtractor, TextExtractor
from raglet.processing.interfaces import DocumentExtractor


def create_extractor(
    file_path: str, extractors: Optional[list[DocumentExtractor]] = None
) -> DocumentExtractor:
    """Create appropriate extractor for file.

    Args:
        file_path: Path to file
        extractors: Optional list of extractors to try (default: all available)

    Returns:
        DocumentExtractor instance

    Note:
        Falls back to TextExtractor for any file type that doesn't have a specific
        extractor. This allows source code files and other text-based files to be
        ingested as plain text until semantic extractors are implemented.
    """
    if extractors is None:
        extractors = [
            MarkdownExtractor(),
            TextExtractor(),
        ]

    # Try specific extractors first (e.g., MarkdownExtractor)
    for extractor in extractors:
        if extractor.can_extract(file_path):
            return extractor

    # Fallback: Use TextExtractor for any file type
    # This allows source code files (.py, .js, etc.) to be ingested as plain text
    return TextExtractor()
