"""Factory for document extractors."""

from tinyrag.processing.extractors import MarkdownExtractor, TextExtractor
from tinyrag.processing.interfaces import DocumentExtractor


def create_extractor(
    file_path: str, extractors: list[DocumentExtractor] = None
) -> DocumentExtractor:
    """Create appropriate extractor for file.

    Args:
        file_path: Path to file
        extractors: Optional list of extractors to try (default: all available)

    Returns:
        DocumentExtractor instance

    Raises:
        ValueError: If no extractor can handle the file
    """
    if extractors is None:
        extractors = [
            MarkdownExtractor(),
            TextExtractor(),
        ]

    for extractor in extractors:
        if extractor.can_extract(file_path):
            return extractor

    raise ValueError(f"No extractor found for file: {file_path}")
