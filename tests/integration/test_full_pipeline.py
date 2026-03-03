"""Integration tests for full pipeline."""

import os
import tempfile

import pytest

from raglet import RAGlet
from raglet.config.config import RAGletConfig


@pytest.mark.integration
class TestFullPipeline:
    """Test full pipeline: files → chunks → embeddings → search."""

    def test_from_files_completes_pipeline(self):
        """Test that from_files completes full pipeline."""
        # Create temporary test files
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test1.txt")
            file2 = os.path.join(tmpdir, "test2.txt")

            with open(file1, "w") as f:
                f.write("This is a test document about Python programming.")

            with open(file2, "w") as f:
                f.write("This document discusses machine learning and embeddings.")

            # Create RAGlet from files
            rag = RAGlet.from_files([file1, file2])

            # Verify chunks were created
            assert len(rag.chunks) > 0

            # Verify vector store has vectors
            assert rag.vector_store.get_count() == len(rag.chunks)

            # Verify embeddings were generated
            assert rag.embedding_generator is not None

    def test_search_returns_results(self):
        """Test that search returns relevant results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test1.txt")

            with open(file1, "w") as f:
                f.write(
                    "Python is a programming language. "
                    "Machine learning uses Python. "
                    "Embeddings are vector representations of text."
                )

            rag = RAGlet.from_files([file1])

            # Search for relevant content
            results = rag.search("What is Python?", top_k=3)

            assert len(results) > 0
            assert all(chunk.score is not None for chunk in results)

            # Verify results contain relevant text
            all_text = " ".join([r.text for r in results]).lower()
            assert "python" in all_text

    def test_search_with_custom_config(self):
        """Test search with custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test1.txt")

            with open(file1, "w") as f:
                f.write("Test document with multiple sentences. " * 10)

            config = RAGletConfig()
            config.search.default_top_k = 10

            rag = RAGlet.from_files([file1], config=config)

            results = rag.search("test", top_k=None)  # Uses config default
            assert len(results) <= 10

    def test_empty_files(self):
        """Test handling of empty files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "empty.txt")

            with open(file1, "w") as f:
                f.write("")

            rag = RAGlet.from_files([file1])

            # Should handle gracefully
            assert len(rag.chunks) == 0
            assert rag.vector_store.get_count() == 0

            # Search should return empty
            results = rag.search("anything")
            assert results == []
