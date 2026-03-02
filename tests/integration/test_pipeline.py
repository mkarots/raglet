"""Integration tests for extract → chunk flow."""

import os
import tempfile

import pytest

from tinyrag.config.config import ChunkingConfig, TinyRAGConfig
from tinyrag.core.rag import TinyRAG


@pytest.mark.integration
class TestExtractChunkFlow:
    """Test extract → chunk integration."""

    def test_from_files_txt(self):
        """Test creating TinyRAG from text files."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "This is a test document. It has multiple sentences. Each sentence is important."
            )
            temp_path = f.name

        try:
            rag = TinyRAG.from_files([temp_path])

            assert len(rag.chunks) > 0
            assert all(chunk.source == temp_path for chunk in rag.chunks)
            assert all(len(chunk.text) > 0 for chunk in rag.chunks)
        finally:
            os.unlink(temp_path)

    def test_from_files_markdown(self):
        """Test creating TinyRAG from markdown files."""
        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nThis is a markdown document. It has **formatting**.")
            temp_path = f.name

        try:
            rag = TinyRAG.from_files([temp_path])

            assert len(rag.chunks) > 0
            assert all(chunk.source == temp_path for chunk in rag.chunks)
        finally:
            os.unlink(temp_path)

    def test_from_files_multiple(self):
        """Test creating TinyRAG from multiple files."""
        files = []

        try:
            # Create multiple temp files
            for i in range(3):
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
                f.write(f"Document {i}. This is test content.")
                f.close()
                files.append(f.name)

            rag = TinyRAG.from_files(files)

            assert len(rag.chunks) > 0
            # Check that chunks come from different sources
            sources = {chunk.source for chunk in rag.chunks}
            assert len(sources) == 3
        finally:
            for f in files:
                os.unlink(f)

    def test_from_files_with_config(self):
        """Test creating TinyRAG with custom config."""
        config = TinyRAGConfig(chunking=ChunkingConfig(size=100, overlap=10))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Short document.")
            temp_path = f.name

        try:
            rag = TinyRAG.from_files([temp_path], config=config)

            assert rag.config.chunking.size == 100
            assert rag.config.chunking.overlap == 10
        finally:
            os.unlink(temp_path)

    def test_get_all_chunks(self):
        """Test getting all chunks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.")
            temp_path = f.name

        try:
            rag = TinyRAG.from_files([temp_path])
            chunks = rag.get_all_chunks()

            assert chunks == rag.chunks
            assert len(chunks) > 0
        finally:
            os.unlink(temp_path)
