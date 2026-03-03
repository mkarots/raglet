"""End-to-end tests for basic functionality."""

import os
import tempfile

import pytest

from raglet.config.config import ChunkingConfig, RAGletConfig
from raglet.core.rag import RAGlet


@pytest.mark.e2e
class TestE2EBasic:
    """End-to-end tests for basic RAGlet functionality."""

    def test_e2e_create_from_files(self):
        """E2E test: Create RAGlet from files."""
        # Create test files
        test_dir = tempfile.mkdtemp()
        files = []

        try:
            # Create multiple test files
            for i in range(3):
                file_path = os.path.join(test_dir, f"doc_{i}.txt")
                with open(file_path, "w") as f:
                    f.write(
                        f"Document {i}. This is test content with multiple sentences. "
                        f"Each document has different content. "
                        f"Testing the full pipeline from files to chunks."
                    )
                files.append(file_path)

            # Create RAGlet
            rag = RAGlet.from_files(files)

            # Verify results
            assert len(rag.chunks) > 0
            assert len(rag.chunks) >= 3  # At least one chunk per file

            # Verify chunks have correct structure
            for chunk in rag.chunks:
                assert chunk.text
                assert chunk.source in files
                assert chunk.index >= 0
                assert isinstance(chunk.metadata, dict)

            # Verify all files are represented
            sources = {chunk.source for chunk in rag.chunks}
            assert len(sources) == 3

        finally:
            # Cleanup
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)
            os.rmdir(test_dir)

    def test_e2e_with_config(self):
        """E2E test: Create RAGlet with custom configuration."""
        test_dir = tempfile.mkdtemp()
        file_path = os.path.join(test_dir, "test.txt")

        try:
            with open(file_path, "w") as f:
                f.write("This is a longer document. " * 50)  # Create longer content

            # Create with custom config
            config = RAGletConfig(chunking=ChunkingConfig(size=100, overlap=10))

            rag = RAGlet.from_files([file_path], config=config)

            # Verify config is applied
            assert rag.config.chunking.size == 100
            assert rag.config.chunking.overlap == 10

            # Verify chunks were created
            assert len(rag.chunks) > 0

        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
            os.rmdir(test_dir)

    def test_e2e_get_all_chunks(self):
        """E2E test: Get all chunks from RAGlet."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test document with content.")
            temp_path = f.name

        try:
            rag = RAGlet.from_files([temp_path])

            # Get all chunks
            chunks = rag.get_all_chunks()

            # Verify
            assert chunks == rag.chunks
            assert len(chunks) > 0
            assert all(chunk.text for chunk in chunks)

        finally:
            os.unlink(temp_path)

    def test_e2e_mixed_file_types(self):
        """E2E test: Handle mixed file types."""
        test_dir = tempfile.mkdtemp()
        files = []

        try:
            # Create .txt file
            txt_file = os.path.join(test_dir, "doc.txt")
            with open(txt_file, "w") as f:
                f.write("Text file content.")
            files.append(txt_file)

            # Create .md file
            md_file = os.path.join(test_dir, "doc.md")
            with open(md_file, "w") as f:
                f.write("# Markdown\n\nMarkdown file content.")
            files.append(md_file)

            # Create RAGlet from mixed files
            rag = RAGlet.from_files(files)

            # Verify both file types processed
            sources = {chunk.source for chunk in rag.chunks}
            assert txt_file in sources
            assert md_file in sources
            assert len(sources) == 2

        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)
            os.rmdir(test_dir)
