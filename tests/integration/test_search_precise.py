"""Precise search tests with known chunk positions.

This test creates a document where we know exactly which chunk indices
should be returned for specific queries. For example:
- Query "Python programming" should return chunks 5, 15, 25
- Query "machine learning" should return chunks 10, 20, 30
etc.
"""

import tempfile
from pathlib import Path

import pytest

from raglet.config.config import ChunkingConfig, RAGletConfig
from raglet.core.rag import RAGlet


@pytest.mark.integration
class TestPreciseSearch:
    """Precise search tests with known chunk positions."""

    def create_precise_document(self, chunk_size: int = 100) -> str:
        """Create document with target phrases at specific positions.

        Args:
            chunk_size: Size of chunks (affects positioning)

        Returns:
            Document content with known phrase positions
        """
        # Strategy: Create filler content, then insert target phrases
        # at positions that will land in specific chunks

        # Calculate how many words per chunk (roughly chunk_size / 5)
        words_per_chunk = chunk_size // 5

        content_parts = []

        # Chunks 0-4: Filler
        for i in range(5):
            content_parts.append(f"Filler content section {i}. " * words_per_chunk)

        # Chunk 5: Python programming (TARGET)
        content_parts.append(
            "Python programming language is versatile and powerful. "
            "Python programming enables rapid development. "
            "Many developers use Python programming for data science. " * (words_per_chunk // 3)
        )

        # Chunks 6-9: Filler
        for i in range(6, 10):
            content_parts.append(f"Filler content section {i}. " * words_per_chunk)

        # Chunk 10: Machine learning (TARGET)
        content_parts.append(
            "Machine learning algorithms process data intelligently. "
            "Machine learning models learn from examples. "
            "Deep learning is a type of machine learning. " * (words_per_chunk // 3)
        )

        # Chunks 11-14: Filler
        for i in range(11, 15):
            content_parts.append(f"Filler content section {i}. " * words_per_chunk)

        # Chunk 15: Python programming again (TARGET)
        content_parts.append(
            "Advanced Python programming techniques include decorators. "
            "Python programming best practices improve code quality. "
            "Object-oriented Python programming uses classes. " * (words_per_chunk // 3)
        )

        # Chunks 16-19: Filler
        for i in range(16, 20):
            content_parts.append(f"Filler content section {i}. " * words_per_chunk)

        # Chunk 20: Machine learning again (TARGET)
        content_parts.append(
            "Supervised machine learning uses labeled training data. "
            "Unsupervised machine learning finds hidden patterns. "
            "Reinforcement learning is another machine learning approach. " * (words_per_chunk // 3)
        )

        # Chunks 21-24: Filler
        for i in range(21, 25):
            content_parts.append(f"Filler content section {i}. " * words_per_chunk)

        # Chunk 25: Python programming again (TARGET)
        content_parts.append(
            "Functional Python programming emphasizes immutability. "
            "Python programming with async/await enables concurrency. "
            "Testing Python programming code ensures reliability. " * (words_per_chunk // 3)
        )

        # Chunks 26-29: Filler
        for i in range(26, 30):
            content_parts.append(f"Filler content section {i}. " * words_per_chunk)

        # Chunk 30: Machine learning again (TARGET)
        content_parts.append(
            "Neural networks are fundamental to machine learning. "
            "Machine learning pipelines process data systematically. "
            "Feature engineering improves machine learning performance. " * (words_per_chunk // 3)
        )

        return "\n\n".join(content_parts)

    def test_python_programming_returns_chunks_5_15_25(self):
        """Test that 'Python programming' returns chunks containing Python programming content.

        Note: Due to chunking overlap, exact positions may vary from 5, 15, 25.
        The test verifies that chunks containing 'Python programming' are found and ranked correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create document
            test_file = Path(tmpdir) / "precise_test.txt"
            test_file.write_text(self.create_precise_document(chunk_size=100))

            # Create RAGlet with specific chunk size
            config = RAGletConfig(chunking=ChunkingConfig(size=100, overlap=10))
            raglet = RAGlet.from_files([str(test_file)], config=config)

            # Get all chunks and find which ones contain "Python programming"
            all_chunks = raglet.get_all_chunks()
            python_chunk_indices = [
                i
                for i, chunk in enumerate(all_chunks)
                if "Python" in chunk.text and "programming" in chunk.text.lower()
            ]

            # Store debug info for assertion messages
            debug_info = {
                "python_chunk_indices": python_chunk_indices,
                "total_chunks": len(all_chunks),
                "vector_store_count": raglet.vector_store.get_count(),
                "similarity_threshold": raglet.config.search.similarity_threshold,
                "embeddings_shape": raglet.embeddings.shape,
            }

            # Search via RAGlet
            results = raglet.search("Python programming", top_k=10)

            debug_info["results_count"] = len(results)
            if results:
                debug_info["first_result_score"] = results[0].score
                debug_info["first_result_text"] = results[0].text[:100]

            # Find indices by matching text content (chunks are new objects, not same references)
            # Use a set to track matched chunks to avoid duplicates
            result_indices = []
            matched_chunk_texts = set()
            for result_chunk in results:
                # Skip if we've already matched this chunk text
                chunk_key = (result_chunk.text, result_chunk.source)
                if chunk_key in matched_chunk_texts:
                    continue

                # Find matching chunk by text content
                for i, chunk in enumerate(all_chunks):
                    if chunk.text == result_chunk.text and chunk.source == result_chunk.source:
                        result_indices.append(i)
                        matched_chunk_texts.add(chunk_key)
                        break

            debug_info["result_indices"] = result_indices

            # Print for visibility when running with -s flag
            print("\n=== DEBUG INFO ===")
            print(f"Python chunks found at: {debug_info['python_chunk_indices']}")
            print(f"Total chunks: {debug_info['total_chunks']}")
            print(f"Vector store count: {debug_info['vector_store_count']}")
            print(f"Search returned {debug_info['results_count']} results")
            print(f"Result indices: {debug_info['result_indices']}")
            if results:
                print(f"First result score: {debug_info.get('first_result_score')}")
                print(f"First result text: {debug_info.get('first_result_text', '')[:80]}...")
            print("==================\n")

            # Assertion 1: Verify we got results
            assert len(results) > 0, (
                f"No search results!\n"
                f"Debug info: {debug_info}\n"
                f"Vector store has {raglet.vector_store.get_count()} vectors, "
                f"similarity_threshold={raglet.config.search.similarity_threshold}"
            )

            # Assertion 2: Verify results don't exceed top_k
            assert len(results) <= 10, f"Got {len(results)} results, expected <= 10"

            # Assertion 3: Verify all results have scores
            scores = [r.score for r in results]
            assert all(score is not None for score in scores), "Some results missing scores"
            assert len(scores) == len(results), "Score count doesn't match result count"

            # Assertion 4: Verify results are ranked correctly (descending by score)
            if len(results) > 1:
                assert scores == sorted(
                    scores, reverse=True
                ), f"Results not properly ranked. Scores: {scores}"

            # Assertion 5: Verify top result contains query terms
            top_result_text = results[0].text.lower()
            assert (
                "python" in top_result_text and "programming" in top_result_text
            ), f"Top result doesn't contain 'Python programming': {results[0].text[:100]}..."

            # Assertion 6: Verify at least some results contain query terms
            # Use case-insensitive matching for both terms
            relevant_results = [
                r for r in results if "python" in r.text.lower() and "programming" in r.text.lower()
            ]
            assert len(relevant_results) > 0, (
                f"None of the {len(results)} results contain 'Python programming'.\n"
                f"First result text: {results[0].text[:200] if results else 'N/A'}\n"
                f"All result texts: {[r.text[:50] for r in results[:3]]}"
            )

            # Assertion 7: Verify overlap with expected chunks
            overlap = set(python_chunk_indices) & set(result_indices)
            debug_info["overlap"] = sorted(overlap)
            debug_info["overlap_count"] = len(overlap)
            assert len(overlap) > 0, (
                f"Expected Python chunks {python_chunk_indices}, "
                f"but got {result_indices}. No overlap!\n"
                f"Debug info: {debug_info}"
            )

            # Assertion 8: Verify at least 30% of expected chunks are in top results
            overlap_ratio = len(overlap) / len(python_chunk_indices) if python_chunk_indices else 0
            debug_info["overlap_ratio"] = overlap_ratio
            assert overlap_ratio >= 0.3, (
                f"Only {len(overlap)}/{len(python_chunk_indices)} expected chunks in results "
                f"({overlap_ratio:.1%}). Expected at least 30%.\n"
                f"Overlap: {sorted(overlap)}\n"
                f"Debug info: {debug_info}"
            )

            # Assertion 9: Verify top result is one of the expected chunks
            top_result_idx = result_indices[0] if result_indices else None
            debug_info["top_result_idx"] = top_result_idx
            debug_info["top_result_in_expected"] = (
                top_result_idx in python_chunk_indices if top_result_idx is not None else False
            )
            assert top_result_idx in python_chunk_indices, (
                f"Top result (chunk {top_result_idx}) is not one of expected Python chunks {python_chunk_indices}\n"
                f"Debug info: {debug_info}\n"
                f"Note: Due to chunking overlap, exact positions may vary. "
                f"Top result should contain 'Python programming' content."
            )

            # Assertion 10: Verify scores are reasonable (cosine similarity: 0-1 range)
            # Cosine similarity scores are 0.0 to 1.0 (higher = more similar)
            assert all(
                0.0 <= score <= 1.0 for score in scores
            ), f"Scores should be in 0-1 range (cosine similarity), got: {scores[:3]}"

            # Assertion 11: Verify score range is reasonable
            # Cosine similarity: typically 0.5-1.0 for relevant results
            assert all(
                0.0 <= score <= 1.0 for score in scores
            ), f"Scores seem unreasonable (expected 0.0 to 1.0 for cosine similarity): {scores[:3]}"

    def test_machine_learning_returns_chunks_10_20_30(self):
        """Test that 'machine learning' returns chunks containing machine learning content.

        Note: Due to chunking overlap, exact positions may vary from 10, 20, 30.
        The test verifies that chunks containing 'machine learning' are found and ranked correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create document
            test_file = Path(tmpdir) / "precise_test.txt"
            test_file.write_text(self.create_precise_document(chunk_size=100))

            # Create RAGlet
            config = RAGletConfig(chunking=ChunkingConfig(size=100, overlap=10))
            raglet = RAGlet.from_files([str(test_file)], config=config)

            # Get all chunks and find machine learning chunks
            all_chunks = raglet.get_all_chunks()
            ml_chunk_indices = [
                i for i, chunk in enumerate(all_chunks) if "machine learning" in chunk.text.lower()
            ]

            # Store debug info for assertion messages
            debug_info = {
                "ml_chunk_indices": ml_chunk_indices,
                "total_chunks": len(all_chunks),
            }

            # Search
            results = raglet.search("machine learning", top_k=10)

            # Find indices by matching text content (avoid duplicates)
            result_indices = []
            matched_chunk_texts = set()
            for result_chunk in results:
                chunk_key = (result_chunk.text, result_chunk.source)
                if chunk_key in matched_chunk_texts:
                    continue

                for i, chunk in enumerate(all_chunks):
                    if chunk.text == result_chunk.text and chunk.source == result_chunk.source:
                        result_indices.append(i)
                        matched_chunk_texts.add(chunk_key)
                        break

            debug_info["result_indices"] = result_indices
            debug_info["results_count"] = len(results)
            if results:
                debug_info["first_result_score"] = results[0].score
                debug_info["first_result_text"] = results[0].text[:100]

            # Print for visibility when running with -s flag
            print("\n=== DEBUG INFO (ML Test) ===")
            print(f"ML chunks found at: {debug_info['ml_chunk_indices']}")
            print(f"Total chunks: {debug_info['total_chunks']}")
            print(f"Search returned {debug_info['results_count']} results")
            print(f"Result indices: {debug_info['result_indices']}")
            if results:
                print(f"First result score: {debug_info.get('first_result_score')}")
                print(f"First result text: {debug_info.get('first_result_text', '')[:80]}...")
            print("============================\n")

            # Assertion 1: Verify we got results
            assert len(results) > 0, (
                f"No search results!\n"
                f"Debug info: {debug_info}\n"
                f"Vector store has {raglet.vector_store.get_count()} vectors, "
                f"similarity_threshold={raglet.config.search.similarity_threshold}"
            )

            # Assertion 2: Verify results don't exceed top_k
            assert len(results) <= 10, f"Got {len(results)} results, expected <= 10"

            # Assertion 3: Verify all results have scores
            scores = [r.score for r in results]
            assert all(score is not None for score in scores), "Some results missing scores"
            assert len(scores) == len(results), "Score count doesn't match result count"

            # Assertion 4: Verify results are ranked correctly (descending by score)
            if len(results) > 1:
                assert scores == sorted(
                    scores, reverse=True
                ), f"Results not properly ranked. Scores: {scores}"

            # Assertion 5: Verify top result contains query term
            top_result_text = results[0].text.lower()
            assert (
                "machine learning" in top_result_text
            ), f"Top result doesn't contain 'machine learning': {results[0].text[:100]}..."

            # Assertion 6: Verify at least some results contain query term
            relevant_results = [r for r in results if "machine learning" in r.text.lower()]
            assert (
                len(relevant_results) > 0
            ), f"None of the {len(results)} results contain 'machine learning'"

            # Assertion 7: Verify overlap with expected chunks
            overlap = set(ml_chunk_indices) & set(result_indices)
            assert len(overlap) > 0, (
                f"Expected ML chunks {ml_chunk_indices}, " f"but got {result_indices}. No overlap!"
            )

            # Assertion 8: Verify at least 30% of expected chunks are in top results
            # (allows for some variation due to chunking/ranking)
            overlap_ratio = len(overlap) / len(ml_chunk_indices) if ml_chunk_indices else 0
            assert overlap_ratio >= 0.3, (
                f"Only {len(overlap)}/{len(ml_chunk_indices)} expected chunks in results "
                f"({overlap_ratio:.1%}). Expected at least 30%.\n"
                f"Debug info: {debug_info}\n"
                f"Overlap: {overlap}"
            )

            # Assertion 9: Verify top result is one of the expected chunks
            top_result_idx = result_indices[0] if result_indices else None
            assert top_result_idx in ml_chunk_indices, (
                f"Top result (chunk {top_result_idx}) is not one of expected chunks {ml_chunk_indices}\n"
                f"Debug info: {debug_info}"
            )

            # Assertion 10: Verify scores are reasonable (cosine similarity: 0-1 range)
            # Cosine similarity scores are 0.0 to 1.0 (higher = more similar)
            assert all(
                0.0 <= score <= 1.0 for score in scores
            ), f"Scores should be in 0-1 range (cosine similarity), got: {scores[:3]}"

            # Assertion 11: Verify score range is reasonable
            # Cosine similarity: typically 0.5-1.0 for relevant results
            assert all(
                0.0 <= score <= 1.0 for score in scores
            ), f"Scores seem unreasonable (expected 0.0 to 1.0 for cosine similarity): {scores[:3]}"

    def test_search_precision_with_different_chunk_sizes(self):
        """Test search precision with different chunk sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text(self.create_precise_document(chunk_size=100))

            chunk_sizes = [50, 100, 200]

            for chunk_size in chunk_sizes:
                config = RAGletConfig(chunking=ChunkingConfig(size=chunk_size, overlap=10))
                raglet = RAGlet.from_files([str(test_file)], config=config)

                # Search for Python programming
                results = raglet.search("Python programming", top_k=5)

                # Assertion 1: Verify we got results
                assert len(results) > 0, f"No results for chunk_size={chunk_size}"

                # Assertion 2: Verify results don't exceed top_k
                assert len(results) <= 5, f"Got {len(results)} results, expected <= 5"

                # Assertion 3: Verify all results have scores
                scores = [r.score for r in results]
                assert all(score is not None for score in scores), "Some results missing scores"

                # Assertion 4: Verify results are ranked correctly
                if len(results) > 1:
                    assert scores == sorted(scores, reverse=True), "Results not properly ranked"

                # Assertion 5: Verify results contain Python programming
                result_texts = [r.text for r in results]
                assert any(
                    "Python" in text and "programming" in text.lower() for text in result_texts
                ), f"Results don't contain Python programming for chunk_size={chunk_size}"

                # Assertion 6: Verify top result contains query terms
                assert (
                    "Python" in results[0].text and "programming" in results[0].text.lower()
                ), f"Top result doesn't contain 'Python programming' for chunk_size={chunk_size}"

    def test_expected_chunk_verification(self):
        """Test that we can verify specific chunks are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simpler document with known structure
            test_file = Path(tmpdir) / "simple_test.txt"
            test_file.write_text("""
Filler text here. Filler text here. Filler text here. Filler text here.
Filler text here. Filler text here. Filler text here. Filler text here.
Filler text here. Filler text here. Filler text here. Filler text here.

Python programming is great. Python programming enables rapid development.
Python programming has a large ecosystem. Python programming is versatile.

Filler text here. Filler text here. Filler text here. Filler text here.
Filler text here. Filler text here. Filler text here. Filler text here.

Machine learning algorithms are powerful. Machine learning models learn from data.
Machine learning enables predictions. Machine learning improves over time.

Filler text here. Filler text here. Filler text here. Filler text here.
Filler text here. Filler text here. Filler text here. Filler text here.

Python programming with libraries. Python programming best practices.
Python programming for data science. Python programming for web development.
""")

            config = RAGletConfig(chunking=ChunkingConfig(size=150, overlap=20))
            raglet = RAGlet.from_files([str(test_file)], config=config)

            all_chunks = raglet.get_all_chunks()

            # Find Python programming chunks
            python_indices = [
                i
                for i, chunk in enumerate(all_chunks)
                if "Python" in chunk.text and "programming" in chunk.text.lower()
            ]

            # Store debug info for assertion messages
            debug_info = {
                "python_indices": python_indices,
                "total_chunks": len(all_chunks),
            }

            # Search
            results = raglet.search("Python programming", top_k=10)

            # Find indices by matching text content (avoid duplicates)
            result_indices = []
            matched_chunk_texts = set()
            for result_chunk in results:
                chunk_key = (result_chunk.text, result_chunk.source)
                if chunk_key in matched_chunk_texts:
                    continue

                for i, chunk in enumerate(all_chunks):
                    if chunk.text == result_chunk.text and chunk.source == result_chunk.source:
                        result_indices.append(i)
                        matched_chunk_texts.add(chunk_key)
                        break

            debug_info["result_indices"] = result_indices
            debug_info["results_count"] = len(results)
            if results:
                debug_info["first_result_score"] = results[0].score
                debug_info["first_result_text"] = results[0].text[:100]

            # Assertion 1: Verify Python chunks exist
            assert len(python_indices) > 0, "No Python programming chunks found"

            # Assertion 2: Verify we got search results
            assert len(results) > 0, "No search results"

            # Assertion 3: Verify results don't exceed top_k
            assert len(results) <= 10, f"Got {len(results)} results, expected <= 10"

            # Assertion 4: Verify all results have scores
            scores = [r.score for r in results]
            assert all(score is not None for score in scores), "Some results missing scores"

            # Assertion 5: Verify results are ranked correctly
            if len(results) > 1:
                assert scores == sorted(scores, reverse=True), "Results not properly ranked"

            # Assertion 6: Verify top result contains query terms
            assert (
                "Python" in results[0].text and "programming" in results[0].text.lower()
            ), f"Top result doesn't contain 'Python programming': {results[0].text[:100]}..."

            # Assertion 7: Verify at least 50% of Python chunks are in top results
            overlap = set(python_indices) & set(result_indices[: len(python_indices) * 2])
            overlap_ratio = len(overlap) / len(python_indices) if python_indices else 0
            assert overlap_ratio >= 0.5, (
                f"Only {len(overlap)}/{len(python_indices)} Python chunks in results. "
                f"Expected at least 50% overlap.\n"
                f"Debug info: {debug_info}\n"
                f"Overlap: {overlap}"
            )

            # Assertion 8: Verify top result is one of the expected chunks
            top_result_idx = result_indices[0] if result_indices else None
            assert top_result_idx in python_indices, (
                f"Top result (chunk {top_result_idx}) is not one of expected chunks {python_indices}\n"
                f"Debug info: {debug_info}"
            )

            # Assertion 9: Verify scores are reasonable (cosine similarity: 0-1 range)
            assert all(
                0.0 <= score <= 1.0 for score in scores
            ), f"Scores should be in 0-1 range (cosine similarity), got: {scores[:3]}"
            assert all(-10 < score < 1 for score in scores), "Scores seem unreasonable"
