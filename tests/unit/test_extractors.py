"""Unit tests for document extractors."""

import os
import tempfile

import pytest

from tinyrag.processing.extractors import MarkdownExtractor, TextExtractor


class TestTextExtractor:
    """Test TextExtractor."""

    def test_can_extract_txt(self):
        """Test can_extract for .txt files."""
        extractor = TextExtractor()
        assert extractor.can_extract("test.txt") is True
        assert extractor.can_extract("test.TXT") is True

    def test_can_extract_no_extension(self):
        """Test can_extract for files without extension."""
        extractor = TextExtractor()
        assert extractor.can_extract("test") is True

    def test_cannot_extract_md(self):
        """Test can_extract returns False for .md files."""
        extractor = TextExtractor()
        assert extractor.can_extract("test.md") is False

    def test_extract_text_file(self):
        """Test extracting text from file."""
        extractor = TextExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, world!\nThis is a test.")
            temp_path = f.name

        try:
            result = extractor.extract(temp_path)
            assert result == "Hello, world!\nThis is a test."
        finally:
            os.unlink(temp_path)

    def test_extract_nonexistent_file(self):
        """Test extracting from non-existent file raises error."""
        extractor = TextExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent.txt")


class TestMarkdownExtractor:
    """Test MarkdownExtractor."""

    def test_can_extract_md(self):
        """Test can_extract for .md files."""
        extractor = MarkdownExtractor()
        assert extractor.can_extract("test.md") is True
        assert extractor.can_extract("test.MD") is True

    def test_can_extract_markdown(self):
        """Test can_extract for .markdown files."""
        extractor = MarkdownExtractor()
        assert extractor.can_extract("test.markdown") is True

    def test_cannot_extract_txt(self):
        """Test can_extract returns False for .txt files."""
        extractor = MarkdownExtractor()
        assert extractor.can_extract("test.txt") is False

    def test_extract_markdown_file(self):
        """Test extracting text from markdown file."""
        extractor = MarkdownExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nThis is **bold** text.")
            temp_path = f.name

        try:
            result = extractor.extract(temp_path)
            assert result == "# Title\n\nThis is **bold** text."
        finally:
            os.unlink(temp_path)
