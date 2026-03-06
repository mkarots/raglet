"""Integration tests for search accuracy with known content.

This test uses a carefully crafted document where we know exactly which chunks
should be returned for specific queries. This ensures search works correctly.
"""

import tempfile
from pathlib import Path

import pytest

from raglet.config.config import ChunkingConfig, RAGletConfig
from raglet.core.rag import RAGlet


@pytest.mark.integration
class TestSearchAccuracy:
    """Test search accuracy with known content."""

    def create_test_document(self) -> str:
        """Create a test document with known content at specific positions.

        Returns:
            Path to created test file
        """
        # Create content where we know which chunks should match which queries
        # We'll use a pattern where specific topics appear at predictable positions

        content_parts = []

        # Add filler content to push specific topics to known chunk positions
        # Using chunk_size=512, we expect ~1-2 chunks per section

        # Chunks 0-2: General filler
        content_parts.extend(
            [
                "Introduction to computer science. " * 20,
                "Basic programming concepts. " * 20,
                "Software development practices. " * 20,
            ]
        )

        # Chunk ~3-4: Python-specific (should match "Python programming")
        content_parts.append(
            "Python is a high-level programming language. "
            "Python programming is widely used for data science. "
            "Python developers use it for web development. " * 15
        )

        # Chunks 5-7: More filler
        content_parts.extend(
            [
                "Database systems store data. " * 20,
                "Networking protocols enable communication. " * 20,
                "Operating systems manage resources. " * 20,
            ]
        )

        # Chunk ~8-9: Machine learning (should match "machine learning algorithms")
        content_parts.append(
            "Machine learning algorithms learn from data. "
            "Machine learning models can make predictions. "
            "Deep learning is a subset of machine learning. " * 15
        )

        # Chunks 10-12: More filler
        content_parts.extend(
            [
                "Cloud computing provides scalability. " * 20,
                "DevOps practices improve deployment. " * 20,
                "Security measures protect systems. " * 20,
            ]
        )

        # Chunk ~13-14: RAG systems (should match "RAG retrieval augmented")
        content_parts.append(
            "RAG stands for Retrieval-Augmented Generation. "
            "RAG systems combine retrieval and generation. "
            "Retrieval-augmented generation improves LLM responses. " * 15
        )

        # Chunks 15-17: More filler
        content_parts.extend(
            [
                "API design follows REST principles. " * 20,
                "Microservices architecture enables scaling. " * 20,
                "Containerization with Docker. " * 20,
            ]
        )

        # Chunk ~18-19: Vector databases (should match "vector database embeddings")
        content_parts.append(
            "Vector databases store embeddings efficiently. "
            "Vector database search uses similarity. "
            "Embeddings represent text as vectors. " * 15
        )

        # Chunks 20-22: More filler
        content_parts.extend(
            [
                "Testing ensures code quality. " * 20,
                "Code review improves collaboration. " * 20,
                "Documentation helps developers. " * 20,
            ]
        )

        # Chunk ~23-24: FAISS (should match "FAISS vector search")
        content_parts.append(
            "FAISS is a library for vector search. "
            "FAISS enables efficient similarity search. "
            "Vector search with FAISS is fast. " * 15
        )

        # More filler to ensure we have enough chunks
        content_parts.extend(
            [
                "Version control with Git. " * 20,
                "Continuous integration automates testing. " * 20,
                "Agile methodology for development. " * 20,
            ]
        )

        return "\n\n".join(content_parts)

    def test_search_returns_expected_chunks(self):
        """Test that search returns chunks containing expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test document
            test_file = Path(tmpdir) / "test_content.txt"
            test_file.write_text(self.create_test_document())

            # Create RAGlet with small chunks to get predictable positions
            config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
            raglet = RAGlet.from_files([str(test_file)], config=config)

            # Get all chunks to find which ones contain our target phrases
            all_chunks = raglet.get_all_chunks()

            # Find chunks containing specific phrases
            python_chunks = [
                i
                for i, chunk in enumerate(all_chunks)
                if "Python" in chunk.text and "programming" in chunk.text.lower()
            ]
            ml_chunks = [
                i
                for i, chunk in enumerate(all_chunks)
                if "machine learning" in chunk.text.lower() and "algorithm" in chunk.text.lower()
            ]
            rag_chunks = [
                i
                for i, chunk in enumerate(all_chunks)
                if "RAG" in chunk.text and "retrieval" in chunk.text.lower()
            ]
            vector_db_chunks = [
                i
                for i, chunk in enumerate(all_chunks)
                if "vector database" in chunk.text.lower()
                or ("vector" in chunk.text.lower() and "embedding" in chunk.text.lower())
            ]
            faiss_chunks = [
                i
                for i, chunk in enumerate(all_chunks)
                if "FAISS" in chunk.text and "vector" in chunk.text.lower()
            ]

            # Test 1: Search for "Python programming"
            results = raglet.search("Python programming", top_k=5)
            # Match chunks by (text, source, index) since search returns new Chunk objects with scores
            result_indices = [
                i
                for i, orig_chunk in enumerate(all_chunks)
                if any(
                    r.text == orig_chunk.text
                    and r.source == orig_chunk.source
                    and r.index == orig_chunk.index
                    for r in results
                )
            ]

            # At least one result should be from our Python chunks
            assert len(results) > 0, "No results for 'Python programming'"
            assert any(
                idx in python_chunks for idx in result_indices
            ), f"Expected Python chunks {python_chunks}, got {result_indices}"

            # Test 2: Search for "machine learning algorithms"
            results = raglet.search("machine learning algorithms", top_k=5)
            result_indices = [
                i
                for i, orig_chunk in enumerate(all_chunks)
                if any(
                    r.text == orig_chunk.text
                    and r.source == orig_chunk.source
                    and r.index == orig_chunk.index
                    for r in results
                )
            ]

            assert len(results) > 0, "No results for 'machine learning algorithms'"
            assert any(
                idx in ml_chunks for idx in result_indices
            ), f"Expected ML chunks {ml_chunks}, got {result_indices}"

            # Test 3: Search for "RAG retrieval augmented"
            results = raglet.search("RAG retrieval augmented", top_k=5)
            result_indices = [
                i
                for i, orig_chunk in enumerate(all_chunks)
                if any(
                    r.text == orig_chunk.text
                    and r.source == orig_chunk.source
                    and r.index == orig_chunk.index
                    for r in results
                )
            ]

            assert len(results) > 0, "No results for 'RAG retrieval augmented'"
            assert any(
                idx in rag_chunks for idx in result_indices
            ), f"Expected RAG chunks {rag_chunks}, got {result_indices}"

            # Test 4: Search for "vector database embeddings"
            results = raglet.search("vector database embeddings", top_k=5)
            result_indices = [
                i
                for i, orig_chunk in enumerate(all_chunks)
                if any(
                    r.text == orig_chunk.text
                    and r.source == orig_chunk.source
                    and r.index == orig_chunk.index
                    for r in results
                )
            ]

            assert len(results) > 0, "No results for 'vector database embeddings'"
            assert any(
                idx in vector_db_chunks for idx in result_indices
            ), f"Expected vector DB chunks {vector_db_chunks}, got {result_indices}"

            # Test 5: Search for "FAISS vector search"
            results = raglet.search("FAISS vector search", top_k=5)
            result_indices = [
                i
                for i, orig_chunk in enumerate(all_chunks)
                if any(
                    r.text == orig_chunk.text
                    and r.source == orig_chunk.source
                    and r.index == orig_chunk.index
                    for r in results
                )
            ]

            assert len(results) > 0, "No results for 'FAISS vector search'"
            assert any(
                idx in faiss_chunks for idx in result_indices
            ), f"Expected FAISS chunks {faiss_chunks}, got {result_indices}"

    def test_search_ranking_order(self):
        """Test that search results are ranked by relevance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create document with clear relevance hierarchy
            test_file = Path(tmpdir) / "ranking_test.txt"
            test_file.write_text(
                # High relevance: exact phrase match
                "Python programming language is versatile. " * 30
                + "\n\n"
                +
                # Medium relevance: partial match
                "Programming in Python is fun. " * 20
                + "\n\n"
                +
                # Low relevance: single word
                "Python is a snake. " * 10
            )

            config = RAGletConfig(chunking=ChunkingConfig(size=150, overlap=10))
            raglet = RAGlet.from_files([str(test_file)], config=config)

            # Search for "Python programming"
            results = raglet.search("Python programming", top_k=5)

            assert len(results) > 0, "No results"

            # Results should have scores (higher = more relevant)
            scores = [r.score for r in results if r.score is not None]
            assert len(scores) > 0, "No scores in results"

            # Scores should be in descending order (most relevant first)
            assert scores == sorted(scores, reverse=True), "Results not sorted by relevance"

            # Top result should contain the exact phrase
            top_result = results[0]
            assert "Python" in top_result.text and "programming" in top_result.text.lower()

    def test_search_preserves_after_save_load(self):
        """Test that search results are consistent after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test document
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text(self.create_test_document())

            config = RAGletConfig(chunking=ChunkingConfig(size=200, overlap=20))
            raglet1 = RAGlet.from_files([str(test_file)], config=config)

            # Search before save
            results1 = raglet1.search("Python programming", top_k=3)
            result_texts1 = [r.text for r in results1]

            # Save and load
            db_path = Path(tmpdir) / "test.sqlite"
            raglet1.save(str(db_path))
            raglet2 = RAGlet.load(str(db_path))

            # Search after load
            results2 = raglet2.search("Python programming", top_k=3)
            result_texts2 = [r.text for r in results2]

            # Results should be similar (may not be identical due to FAISS rebuild,
            # but should contain similar content)
            assert len(results2) > 0, "No results after load"
            assert len(results2) == len(results1), "Different number of results"

            # At least some overlap in results
            overlap = set(result_texts1) & set(result_texts2)
            assert len(overlap) > 0, "No overlap in search results after load"
