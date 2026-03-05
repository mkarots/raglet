"""Integration tests for save/load operations."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from raglet.core.rag import RAGlet


@pytest.mark.integration
class TestSaveLoadIntegration:
    """Integration tests for save/load operations."""

    def test_save_and_load_from_files(self):
        """Test saving RAGlet created from files and loading it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file1 = Path(tmpdir) / "test1.txt"
            test_file1.write_text("This is test file 1.\nIt has multiple lines.")

            test_file2 = Path(tmpdir) / "test2.txt"
            test_file2.write_text("This is test file 2.\nAlso multiple lines.")

            # Create RAGlet from files
            raglet = RAGlet.from_files([str(test_file1), str(test_file2)])

            # Save to SQLite
            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))

            # Load back
            loaded = RAGlet.load(str(db_path))

            # Verify chunks
            assert len(loaded.chunks) == len(raglet.chunks)
            assert all(chunk.text in [c.text for c in raglet.chunks] for chunk in loaded.chunks)

            # Verify embeddings
            assert loaded.embeddings.shape == raglet.embeddings.shape
            np.testing.assert_array_almost_equal(loaded.embeddings, raglet.embeddings, decimal=5)

            # Verify search works
            results = loaded.search("test file")
            assert len(results) > 0

    def test_incremental_add_chunks(self):
        """Test adding chunks incrementally and saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial RAGlet
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Initial content.")
            raglet = RAGlet.from_files([str(test_file)])

            initial_count = len(raglet.chunks)

            # Save initial state
            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))

            # Add new chunks
            from raglet.core.chunk import Chunk

            new_chunks = [
                Chunk(text="New chunk 1", source="manual", index=0),
                Chunk(text="New chunk 2", source="manual", index=1),
            ]
            raglet.add_chunks(new_chunks)

            # Save incrementally
            raglet.save(str(db_path), incremental=True)

            # Load and verify
            loaded = RAGlet.load(str(db_path))
            assert len(loaded.chunks) == initial_count + 2

            # Verify new chunks are searchable
            results = loaded.search("New chunk")
            assert len(results) > 0

    def test_add_chunks_with_immediate_save(self):
        """Test add_chunks() with file_path saves immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial RAGlet
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Initial content.")
            raglet = RAGlet.from_files([str(test_file)])

            initial_count = len(raglet.chunks)

            # Save initial state
            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))

            # Add chunks with immediate save
            from raglet.core.chunk import Chunk

            new_chunks = [
                Chunk(text="New chunk", source="manual", index=0),
            ]
            raglet.add_chunks(new_chunks, file_path=str(db_path))

            # Load and verify
            loaded = RAGlet.load(str(db_path))
            assert len(loaded.chunks) == initial_count + 1

    def test_add_files(self):
        """Test add_files() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial RAGlet
            test_file1 = Path(tmpdir) / "test1.txt"
            test_file1.write_text("Initial content.")
            raglet = RAGlet.from_files([str(test_file1)])

            initial_count = len(raglet.chunks)

            # Add new files
            test_file2 = Path(tmpdir) / "test2.txt"
            test_file2.write_text("New file content.")
            test_file3 = Path(tmpdir) / "test3.txt"
            test_file3.write_text("Another new file.")

            raglet.add_files([str(test_file2), str(test_file3)])

            # Verify chunks were added
            assert len(raglet.chunks) > initial_count
            sources = {chunk.source for chunk in raglet.chunks}
            assert str(test_file1) in sources
            assert str(test_file2) in sources
            assert str(test_file3) in sources

            # Verify search works with new content
            results = raglet.search("New file")
            assert len(results) > 0

    def test_add_files_with_save(self):
        """Test add_files() with immediate save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save initial RAGlet
            test_file1 = Path(tmpdir) / "test1.txt"
            test_file1.write_text("Initial content.")
            raglet = RAGlet.from_files([str(test_file1)])

            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))

            # Load and add files with save
            loaded = RAGlet.load(str(db_path))
            initial_count = len(loaded.chunks)

            test_file2 = Path(tmpdir) / "test2.txt"
            test_file2.write_text("New content to add.")
            loaded.add_files([str(test_file2)], file_path=str(db_path))

            # Verify persistence
            final = RAGlet.load(str(db_path))
            assert len(final.chunks) == initial_count + len(
                RAGlet.from_files([str(test_file2)]).chunks
            )

    def test_from_sqlite_inspection(self):
        """Test from_sqlite() for inspection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save RAGlet
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content.")
            raglet = RAGlet.from_files([str(test_file)])

            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))

            # Load using from_sqlite()
            loaded = RAGlet.from_sqlite(str(db_path))

            assert len(loaded.chunks) == len(raglet.chunks)
            assert loaded.embeddings.shape == raglet.embeddings.shape

    def test_save_load_preserves_search_functionality(self):
        """Test that search functionality is preserved after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create RAGlet with multiple chunks
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text(
                "Python is a programming language.\n"
                "Machine learning uses algorithms.\n"
                "RAG systems combine retrieval and generation."
            )
            raglet = RAGlet.from_files([str(test_file)])

            # Save and load
            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))
            loaded = RAGlet.load(str(db_path))

            # Test search
            results = loaded.search("programming")
            assert len(results) > 0
            assert any("Python" in chunk.text for chunk in results)

            results2 = loaded.search("machine learning")
            assert len(results2) > 0
            assert any("algorithms" in chunk.text.lower() for chunk in results2)

    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Initial content.")
            raglet = RAGlet.from_files([str(test_file)])

            db_path = Path(tmpdir) / "test.sqlite"

            # Multiple save/load cycles
            for i in range(3):
                raglet.save(str(db_path))
                loaded = RAGlet.load(str(db_path))
                assert len(loaded.chunks) == len(raglet.chunks)

                # Add more chunks
                from raglet.core.chunk import Chunk

                new_chunks = [Chunk(text=f"Cycle {i} chunk", source="cycle", index=i)]
                raglet.add_chunks(new_chunks)

            # Final verification
            final_loaded = RAGlet.load(str(db_path))
            assert len(final_loaded.chunks) >= len(raglet.chunks) - 1
