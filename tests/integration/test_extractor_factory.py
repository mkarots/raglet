"""Integration tests for extractor factory."""

import os
import tempfile

import pytest

from raglet.processing.extractor_factory import create_extractor
from raglet.processing.extractors.markdown_extractor import MarkdownExtractor
from raglet.processing.extractors.text_extractor import TextExtractor


@pytest.mark.integration
class TestExtractorFactory:
    """Test extractor factory."""

    def test_factory_text_file(self):
        """Test factory creates TextExtractor for .txt files."""
        extractor = create_extractor("test.txt")

        assert isinstance(extractor, TextExtractor)
        assert extractor.can_extract("test.txt")

    def test_factory_markdown_file(self):
        """Test factory creates MarkdownExtractor for .md files."""
        extractor = create_extractor("test.md")

        assert isinstance(extractor, MarkdownExtractor)
        assert extractor.can_extract("test.md")

    def test_factory_no_extension(self):
        """Test factory handles files without extension."""
        extractor = create_extractor("test")

        assert isinstance(extractor, TextExtractor)

    def test_factory_unknown_file(self):
        """Test factory raises error for unknown file types."""
        with pytest.raises(ValueError, match="No extractor found"):
            create_extractor("test.unknown")

    def test_factory_extracts_correctly(self):
        """Test factory-created extractor works correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            extractor = create_extractor(temp_path)
            result = extractor.extract(temp_path)

            assert result == "Test content"
        finally:
            os.unlink(temp_path)
