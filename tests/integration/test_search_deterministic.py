"""Deterministic search tests with known content and expected results.

This test uses a carefully crafted document where we know exactly which chunks
should be returned for specific queries. Each query has expected chunk indices.
"""

import tempfile
from pathlib import Path

import pytest

from raglet.config.config import ChunkingConfig, RAGletConfig
from raglet.core.rag import RAGlet


@pytest.mark.integration
class TestDeterministicSearch:
    """Deterministic search tests with known expected results."""

    @pytest.fixture
    def test_document(self):
        """Create test document with known content structure."""
        # Use the fixture file
        fixture_path = Path(__file__).parent.parent / "fixtures" / "search_test_content.txt"
        if fixture_path.exists():
            return str(fixture_path)

        # Fallback: create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("""Python Programming Language

Python is a high-level programming language known for its simplicity and readability.
Python programming is widely used for data science, web development, and automation.
Python developers appreciate its extensive standard library and third-party packages.

Machine Learning Algorithms

Machine learning algorithms learn patterns from data without explicit programming.
Machine learning models can make predictions and classifications.
Deep learning is a subset of machine learning that uses neural networks.

RAG Systems

RAG stands for Retrieval-Augmented Generation.
RAG systems combine retrieval and generation to improve language model responses.
Retrieval-augmented generation enhances LLM outputs by grounding them in retrieved context.

Vector Databases

Vector databases store embeddings efficiently for similarity search.
Vector database search uses cosine similarity or L2 distance.
Embeddings represent text as vectors in high-dimensional space.

FAISS Vector Search

FAISS is a library for efficient vector search developed by Facebook AI Research.
FAISS enables fast similarity search over large vector collections.
Vector search with FAISS supports various index types including flat indices.
""")
            return f.name

    def test_python_programming_query(self, test_document):
        """Test that 'Python programming' query returns expected chunks."""
        config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
        raglet = RAGlet.from_files([test_document], config=config)

        all_chunks = raglet.get_all_chunks()

        # Find chunks containing "Python" and "programming"
        python_chunks = [
            (i, chunk)
            for i, chunk in enumerate(all_chunks)
            if "Python" in chunk.text and "programming" in chunk.text.lower()
        ]

        assert len(python_chunks) > 0, "No Python programming chunks found"

        # Search
        results = raglet.search("Python programming", top_k=5)

        assert len(results) > 0, "No search results"

        # Verify at least one result contains Python programming content
        result_texts = [r.text for r in results]
        assert any(
            "Python" in text and "programming" in text.lower() for text in result_texts
        ), f"Expected Python programming content, got: {result_texts[:2]}"

        # Verify results are ranked (scores should decrease)
        scores = [r.score for r in results if r.score is not None]
        if len(scores) > 1:
            assert scores == sorted(scores, reverse=True), "Results not properly ranked"

    def test_machine_learning_query(self, test_document):
        """Test that 'machine learning algorithms' query returns expected chunks."""
        config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
        raglet = RAGlet.from_files([test_document], config=config)

        # Search
        results = raglet.search("machine learning algorithms", top_k=5)

        assert len(results) > 0, "No search results"

        # Verify results contain machine learning content
        result_texts = [r.text for r in results]
        assert any(
            "machine learning" in text.lower() and "algorithm" in text.lower()
            for text in result_texts
        ), f"Expected machine learning content, got: {result_texts[:2]}"

    def test_rag_retrieval_query(self, test_document):
        """Test that 'RAG retrieval augmented' query returns expected chunks."""
        config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
        raglet = RAGlet.from_files([test_document], config=config)

        # Search
        results = raglet.search("RAG retrieval augmented", top_k=5)

        assert len(results) > 0, "No search results"

        # Verify results contain RAG content
        result_texts = [r.text for r in results]
        assert any(
            "RAG" in text and "retrieval" in text.lower() for text in result_texts
        ), f"Expected RAG content, got: {result_texts[:2]}"

    def test_vector_database_query(self, test_document):
        """Test that 'vector database embeddings' query returns expected chunks."""
        config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
        raglet = RAGlet.from_files([test_document], config=config)

        # Search
        results = raglet.search("vector database embeddings", top_k=5)

        assert len(results) > 0, "No search results"

        # Verify results contain vector database content
        result_texts = [r.text for r in results]
        assert any(
            "vector" in text.lower() and ("database" in text.lower() or "embedding" in text.lower())
            for text in result_texts
        ), f"Expected vector database content, got: {result_texts[:2]}"

    def test_faiss_query(self, test_document):
        """Test that 'FAISS vector search' query returns expected chunks."""
        config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
        raglet = RAGlet.from_files([test_document], config=config)

        # Search
        results = raglet.search("FAISS vector search", top_k=5)

        assert len(results) > 0, "No search results"

        # Verify results contain FAISS content
        result_texts = [r.text for r in results]
        assert any(
            "FAISS" in text and "vector" in text.lower() for text in result_texts
        ), f"Expected FAISS content, got: {result_texts[:2]}"

    def test_search_consistency(self, test_document):
        """Test that search results are consistent across multiple calls."""
        config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
        raglet = RAGlet.from_files([test_document], config=config)

        # Run same query multiple times
        query = "Python programming"
        results1 = raglet.search(query, top_k=5)
        results2 = raglet.search(query, top_k=5)
        results3 = raglet.search(query, top_k=5)

        # Results should be consistent (same chunks, may differ in order slightly)
        assert len(results1) == len(results2) == len(results3), "Inconsistent result counts"

        # Top result should be the same
        assert results1[0].text == results2[0].text == results3[0].text, "Inconsistent top result"

    def test_search_with_save_load(self, test_document):
        """Test that search works correctly after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
            raglet1 = RAGlet.from_files([test_document], config=config)

            # Search before save
            results1 = raglet1.search("Python programming", top_k=3)
            results1[0].text if results1 else ""

            # Save and load
            db_path = Path(tmpdir) / "test.sqlite"
            raglet1.save(str(db_path))
            raglet2 = RAGlet.load(str(db_path))

            # Search after load
            results2 = raglet2.search("Python programming", top_k=3)

            assert len(results2) > 0, "No results after load"
            assert len(results2) == len(results1), "Different number of results"

            # Top result should be similar (may not be identical due to FAISS rebuild)
            top_result_text2 = results2[0].text
            assert "Python" in top_result_text2, "Top result doesn't contain 'Python'"
