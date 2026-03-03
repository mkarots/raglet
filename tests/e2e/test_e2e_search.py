"""End-to-end tests for search functionality."""

import os
import tempfile

import pytest

from raglet import RAGlet


@pytest.mark.e2e
class TestE2ESearch:
    """E2E tests for search functionality."""

    def test_e2e_create_and_search(self):
        """E2E test: Create RAGlet and search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, "python.txt")
            file2 = os.path.join(tmpdir, "ml.txt")

            with open(file1, "w") as f:
                f.write(
                    "Python is a high-level programming language. "
                    "It is known for its simplicity and readability. "
                    "Python supports multiple programming paradigms."
                )

            with open(file2, "w") as f:
                f.write(
                    "Machine learning is a subset of artificial intelligence. "
                    "It uses algorithms to learn from data. "
                    "Neural networks are a popular ML technique."
                )

            # Create RAGlet
            rag = RAGlet.from_files([file1, file2])

            # Verify it was created
            assert len(rag.chunks) > 0
            assert rag.vector_store.get_count() == len(rag.chunks)

            # Search for Python-related content
            results = rag.search("What is Python?", top_k=3)

            assert len(results) > 0
            assert all(chunk.score is not None for chunk in results)

            # Verify top result is relevant
            top_result_text = results[0].text.lower()
            assert "python" in top_result_text

            # Search for ML-related content
            ml_results = rag.search("machine learning", top_k=2)
            assert len(ml_results) > 0

            ml_text = " ".join([r.text.lower() for r in ml_results])
            assert "machine" in ml_text or "learning" in ml_text

    def test_e2e_search_with_top_k(self):
        """E2E test: Search with custom top_k."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test.txt")

            with open(file1, "w") as f:
                f.write("Sentence one. " * 20)

            rag = RAGlet.from_files([file1])

            # Search with top_k=1
            results = rag.search("sentence", top_k=1)
            assert len(results) == 1

            # Search with top_k=5
            results = rag.search("sentence", top_k=5)
            assert len(results) <= 5

    def test_e2e_empty_search(self):
        """E2E test: Search on empty RAGlet."""
        rag = RAGlet.from_files([])

        results = rag.search("anything")
        assert results == []
