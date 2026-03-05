"""Integration tests for directory format save/load."""

import tempfile
from pathlib import Path

import pytest

from raglet import RAGlet, RAGletConfig
from raglet.core.chunk import Chunk


@pytest.mark.integration
class TestDirectoryFormat:
    """Test directory format save/load integration."""

    def test_save_and_load_roundtrip(self):
        """Test save → load roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"

            # Create RAGlet
            chunks = [
                Chunk(text="Python is a programming language", source="test.txt", index=0),
                Chunk(text="Machine learning uses algorithms", source="test.txt", index=1),
            ]
            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)

            # Save
            raglet.save(str(dir_path))

            # Verify files exist
            assert (dir_path / "config.json").exists()
            assert (dir_path / "chunks.json").exists()
            assert (dir_path / "embeddings.npy").exists()
            assert (dir_path / "metadata.json").exists()

            # Load
            loaded = RAGlet.load(str(dir_path))

            # Verify data
            assert len(loaded.chunks) == 2
            assert loaded.chunks[0].text == "Python is a programming language"
            assert loaded.chunks[1].text == "Machine learning uses algorithms"
            assert loaded.embeddings.shape == (2, 384)  # all-MiniLM-L6-v2 has 384 dims
            assert loaded.config == config

    def test_incremental_save(self):
        """Test incremental save adds chunks without replacing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"

            # Create initial RAGlet
            chunks1 = [
                Chunk(text="Chunk 1", source="test.txt", index=0),
                Chunk(text="Chunk 2", source="test.txt", index=1),
            ]
            config = RAGletConfig()
            raglet1 = RAGlet.from_files([], config=config)
            raglet1.chunks = chunks1
            raglet1.embeddings = raglet1.embedding_generator.generate(chunks1)
            raglet1.vector_store.add_vectors(raglet1.embeddings, chunks1)

            # Initial save
            raglet1.save(str(dir_path))

            # Add more chunks
            chunks2 = [
                Chunk(text="Chunk 3", source="test.txt", index=2),
                Chunk(text="Chunk 4", source="test.txt", index=3),
            ]
            raglet2 = RAGlet.load(str(dir_path))
            raglet2.add_chunks(chunks2)
            raglet2.save(str(dir_path), incremental=True)

            # Load and verify
            loaded = RAGlet.load(str(dir_path))
            assert len(loaded.chunks) == 4
            assert loaded.chunks[0].text == "Chunk 1"
            assert loaded.chunks[3].text == "Chunk 4"

    def test_search_after_load(self):
        """Test search works after loading from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"

            # Create and save RAGlet
            raglet = RAGlet.from_files([], config=RAGletConfig())
            chunks = [
                Chunk(text="Python programming language", source="test.txt", index=0),
                Chunk(text="Machine learning algorithms", source="test.txt", index=1),
            ]
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(dir_path))

            # Load and search
            loaded = RAGlet.load(str(dir_path))
            results = loaded.search("Python", top_k=1)

            assert len(results) > 0
            assert "python" in results[0].text.lower()

    def test_auto_detection_directory(self):
        """Test auto-detection works for directory paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"

            # Create RAGlet
            raglet = RAGlet.from_files([], config=RAGletConfig())
            chunks = [Chunk(text="Test", source="test.txt", index=0)]
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)

            # Save (should auto-detect directory format)
            raglet.save(str(dir_path))

            # Load (should auto-detect directory format)
            loaded = RAGlet.load(str(dir_path))
            assert len(loaded.chunks) == 1

    def test_auto_detection_sqlite(self):
        """Test auto-detection works for SQLite files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = Path(tmpdir) / "test.sqlite"

            # Create RAGlet
            raglet = RAGlet.from_files([], config=RAGletConfig())
            chunks = [Chunk(text="Test", source="test.txt", index=0)]
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)

            # Save (should auto-detect SQLite format)
            raglet.save(str(sqlite_path))

            # Load (should auto-detect SQLite format)
            loaded = RAGlet.load(str(sqlite_path))
            assert len(loaded.chunks) == 1

    def test_format_interoperability(self):
        """Test converting between directory and SQLite formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"
            sqlite_path = Path(tmpdir) / "test.sqlite"

            # Create RAGlet
            raglet = RAGlet.from_files([], config=RAGletConfig())
            chunks = [
                Chunk(text="Chunk 1", source="test.txt", index=0),
                Chunk(text="Chunk 2", source="test.txt", index=1),
            ]
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)

            # Save to directory
            raglet.save(str(dir_path))

            # Load from directory and save to SQLite
            loaded = RAGlet.load(str(dir_path))
            loaded.save(str(sqlite_path))

            # Load from SQLite and verify
            sqlite_loaded = RAGlet.load(str(sqlite_path))
            assert len(sqlite_loaded.chunks) == 2
            assert sqlite_loaded.chunks[0].text == "Chunk 1"
