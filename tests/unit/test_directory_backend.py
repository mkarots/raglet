"""Unit tests for DirectoryStorageBackend."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from raglet.config.config import RAGletConfig
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.storage.directory_backend import DirectoryStorageBackend


@pytest.mark.unit
class TestDirectoryStorageBackend:
    """Test DirectoryStorageBackend."""

    def test_supports_incremental(self):
        """Test supports_incremental() returns True."""
        backend = DirectoryStorageBackend()
        assert backend.supports_incremental() is True

    def test_save_and_load_empty_raglet(self):
        """Test saving and loading empty RAGlet."""
        backend = DirectoryStorageBackend()
        config = RAGletConfig()
        raglet = RAGlet(chunks=[], config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"
            backend.save(raglet, str(dir_path))

            # Verify files exist
            assert (dir_path / "config.json").exists()
            assert (dir_path / "chunks.json").exists()
            assert (dir_path / "embeddings.npy").exists()
            assert (dir_path / "metadata.json").exists()

            # Load and verify
            loaded = backend.load(str(dir_path))
            assert len(loaded.chunks) == 0
            assert loaded.config == config

    def test_save_and_load_with_chunks(self):
        """Test saving and loading RAGlet with chunks."""
        backend = DirectoryStorageBackend()
        chunks = [
            Chunk(text="Test chunk 1", source="test.txt", index=0),
            Chunk(text="Test chunk 2", source="test.txt", index=1),
        ]
        config = RAGletConfig()
        raglet = RAGlet.from_files([], config=config)
        raglet.chunks = chunks
        raglet.embeddings = np.random.rand(2, 384).astype(np.float32)
        raglet.vector_store.add_vectors(raglet.embeddings, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"
            backend.save(raglet, str(dir_path))

            # Load and verify
            loaded = backend.load(str(dir_path))
            assert len(loaded.chunks) == 2
            assert loaded.chunks[0].text == "Test chunk 1"
            assert loaded.chunks[1].text == "Test chunk 2"
            assert loaded.embeddings.shape == (2, 384)

    def test_incremental_save(self):
        """Test incremental save appends chunks."""
        backend = DirectoryStorageBackend()
        chunks1 = [
            Chunk(text="Chunk 1", source="test.txt", index=0),
            Chunk(text="Chunk 2", source="test.txt", index=1),
        ]
        config = RAGletConfig()
        raglet1 = RAGlet.from_files([], config=config)
        raglet1.chunks = chunks1
        raglet1.embeddings = np.random.rand(2, 384).astype(np.float32)
        raglet1.vector_store.add_vectors(raglet1.embeddings, chunks1)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"

            # Initial save
            backend.save(raglet1, str(dir_path))

            # Add more chunks
            chunks2 = [
                Chunk(text="Chunk 3", source="test.txt", index=2),
                Chunk(text="Chunk 4", source="test.txt", index=3),
            ]
            raglet2 = RAGlet.from_files([], config=config)
            raglet2.chunks = chunks1 + chunks2
            raglet2.embeddings = np.random.rand(4, 384).astype(np.float32)
            raglet2.vector_store.add_vectors(raglet2.embeddings, raglet2.chunks)

            # Incremental save
            backend.save(raglet2, str(dir_path), incremental=True)

            # Load and verify
            loaded = backend.load(str(dir_path))
            assert len(loaded.chunks) == 4
            assert loaded.chunks[0].text == "Chunk 1"
            assert loaded.chunks[3].text == "Chunk 4"

    def test_add_chunks(self):
        """Test add_chunks() method."""
        backend = DirectoryStorageBackend()
        chunks1 = [
            Chunk(text="Chunk 1", source="test.txt", index=0),
        ]
        config = RAGletConfig()
        raglet1 = RAGlet.from_files([], config=config)
        raglet1.chunks = chunks1
        raglet1.embeddings = np.random.rand(1, 384).astype(np.float32)
        raglet1.vector_store.add_vectors(raglet1.embeddings, chunks1)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"

            # Initial save
            backend.save(raglet1, str(dir_path))

            # Add chunks incrementally
            new_chunks = [
                Chunk(text="Chunk 2", source="test.txt", index=1),
            ]
            new_embeddings = np.random.rand(1, 384).astype(np.float32)
            backend.add_chunks(str(dir_path), new_chunks, new_embeddings, raglet1)

            # Load and verify
            loaded = backend.load(str(dir_path))
            assert len(loaded.chunks) == 2
            assert loaded.chunks[0].text == "Chunk 1"
            assert loaded.chunks[1].text == "Chunk 2"

    def test_load_nonexistent_directory(self):
        """Test loading from nonexistent directory raises FileNotFoundError."""
        backend = DirectoryStorageBackend()
        with pytest.raises(FileNotFoundError):
            backend.load("/nonexistent/directory")

    def test_load_invalid_directory(self):
        """Test loading from invalid directory raises ValueError."""
        backend = DirectoryStorageBackend()
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValueError, match="not a directory"):
                backend.load(tmpfile.name)

    def test_load_missing_config(self):
        """Test loading from directory without config.json raises ValueError."""
        backend = DirectoryStorageBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"
            dir_path.mkdir()
            # Create chunks.json but not config.json
            (dir_path / "chunks.json").write_text("[]")

            with pytest.raises(ValueError, match="config.json not found"):
                backend.load(str(dir_path))

    def test_load_missing_chunks(self):
        """Test loading from directory without chunks.json raises ValueError."""
        backend = DirectoryStorageBackend()
        config = RAGletConfig()
        raglet = RAGlet(chunks=[], config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"
            backend.save(raglet, str(dir_path))

            # Remove chunks.json
            (dir_path / "chunks.json").unlink()

            with pytest.raises(ValueError, match="chunks.json not found"):
                backend.load(str(dir_path))

    def test_load_missing_embeddings(self):
        """Test loading from directory without embeddings.npy raises ValueError."""
        backend = DirectoryStorageBackend()
        config = RAGletConfig()
        raglet = RAGlet(chunks=[], config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"
            backend.save(raglet, str(dir_path))

            # Remove embeddings.npy
            (dir_path / "embeddings.npy").unlink()

            with pytest.raises(ValueError, match="embeddings.npy not found"):
                backend.load(str(dir_path))

    def test_embedding_dimension_mismatch(self):
        """Test loading with embedding dimension mismatch raises ValueError."""
        backend = DirectoryStorageBackend()
        chunks = [Chunk(text="Test", source="test.txt", index=0)]
        config = RAGletConfig()
        raglet = RAGlet.from_files([], config=config)
        raglet.chunks = chunks
        raglet.embeddings = np.random.rand(1, 384).astype(np.float32)
        raglet.vector_store.add_vectors(raglet.embeddings, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "test_raglet"
            backend.save(raglet, str(dir_path))

            # Corrupt config to use different model (different dimension)
            import json

            config_path = dir_path / "config.json"
            with open(config_path) as f:
                config_data = json.load(f)
            config_data["embedding"]["model"] = "all-mpnet-base-v2"  # 768 dims instead of 384
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            with pytest.raises(ValueError, match="Embedding dimension mismatch"):
                backend.load(str(dir_path))

    def test_round_trip_1000_chunks(self):
        """Save/load 1000 chunks and verify embedding fidelity.

        Exercises the get_all_vectors() save path and the
        ascontiguousarray load path at moderate scale.
        """
        backend = DirectoryStorageBackend()
        config = RAGletConfig()
        n = 1000
        chunks = [Chunk(text=f"Chunk number {i}", source="scale.txt", index=i) for i in range(n)]
        raglet = RAGlet(chunks=chunks, config=config)
        original_embeddings = raglet.embeddings.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir) / "scale_dir"
            backend.save(raglet, str(dir_path))

            loaded = backend.load(str(dir_path))
            assert len(loaded.chunks) == n
            assert loaded.chunks[0].text == "Chunk number 0"
            assert loaded.chunks[n - 1].text == f"Chunk number {n - 1}"
            np.testing.assert_array_almost_equal(loaded.embeddings, original_embeddings, decimal=5)
