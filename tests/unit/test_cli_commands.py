"""Unit tests for CLI command functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raglet.cli import (
    add_command,
    build_command,
    package_command,
    query_command,
)
from raglet.core.chunk import Chunk


@pytest.mark.unit
class TestCLICommands:
    """Test CLI command functions."""

    def test_build_command_creates_raglet(self):
        """Test build_command creates raglet directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_path = workspace / "test.raglet"

            # Create test files
            (workspace / "test.txt").write_text("Test content")

            # Mock args
            args = MagicMock()
            args.inputs = [str(workspace / "test.txt")]
            args.out = str(output_path)
            args.ignore = ".git,__pycache__"
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 0
            assert output_path.exists()
            assert output_path.is_dir()
            assert (output_path / "chunks.json").exists()

    def test_build_command_handles_no_files(self):
        """Test build_command handles case with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_path = workspace / "test.raglet"

            # Mock args
            args = MagicMock()
            args.inputs = []  # No inputs
            args.out = str(output_path)
            args.ignore = ""
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 1  # Should fail with no files

    def test_query_command_loads_and_searches(self):
        """Test query_command loads raglet and searches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "test.raglet.zip"

            # Create raglet
            chunks = [
                Chunk(text="Python programming", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.query = "Python"
            args.top_k = 5
            args.show_full = False

            # Run query command
            result = query_command(args)

            assert result == 0

    def test_query_command_handles_missing_raglet(self):
        """Test query_command handles missing raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "nonexistent.raglet.zip"

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.q = "test"
            args.top_k = 5
            args.show_full = False

            # Run query command (should fail)
            result = query_command(args)

            assert result == 1  # Should fail with missing raglet

    def test_add_command_adds_files(self):
        """Test add_command adds files to raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "test.raglet.zip"

            # Create initial raglet
            chunks = [
                Chunk(text="Initial", source="initial.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Create new file
            new_file = workspace / "new.txt"
            new_file.write_text("New content")

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(new_file)]
            args.out = None
            args.ignore = None
            args.max_files = None

            # Run add command
            result = add_command(args)

            assert result == 0

            # Verify file added
            loaded = RAGlet.load(str(kb_path))
            assert len(loaded.chunks) > len(chunks)

    def test_add_command_accepts_directory(self):
        """Test add_command accepts a directory input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            from raglet import RAGlet

            # Create initial raglet from a dummy file
            init_file = workspace / "init.txt"
            init_file.write_text("Initial content for raglet.")
            raglet = RAGlet.from_files([str(init_file)])
            raglet.save(str(kb_path))
            initial_count = len(raglet.chunks)

            # Create a directory with new files
            new_dir = workspace / "new_docs"
            new_dir.mkdir()
            (new_dir / "a.txt").write_text("Alpha content about searching.")
            (new_dir / "b.txt").write_text("Beta content about indexing.")

            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(new_dir)]
            args.out = None
            args.ignore = None
            args.max_files = None

            result = add_command(args)

            assert result == 0

            loaded = RAGlet.load(str(kb_path))
            assert len(loaded.chunks) > initial_count

    def test_add_command_accepts_glob(self):
        """Test add_command accepts glob patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            from raglet import RAGlet

            init_file = workspace / "init.txt"
            init_file.write_text("Initial content for raglet.")
            raglet = RAGlet.from_files([str(init_file)])
            raglet.save(str(kb_path))
            initial_count = len(raglet.chunks)

            # Create files matching a glob
            (workspace / "note_1.md").write_text("First note about embeddings.")
            (workspace / "note_2.md").write_text("Second note about vectors.")
            (workspace / "ignore.txt").write_text("Should not be matched by *.md")

            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(workspace / "*.md")]
            args.out = None
            args.ignore = None
            args.max_files = None

            result = add_command(args)

            assert result == 0

            loaded = RAGlet.load(str(kb_path))
            assert len(loaded.chunks) > initial_count
            sources = [c.source for c in loaded.chunks]
            assert not any("ignore.txt" in s for s in sources)

    def test_add_command_respects_ignore(self):
        """Test add_command --ignore filters out matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            from raglet import RAGlet

            init_file = workspace / "init.txt"
            init_file.write_text("Initial content for raglet.")
            raglet = RAGlet.from_files([str(init_file)])
            raglet.save(str(kb_path))

            new_dir = workspace / "src"
            new_dir.mkdir()
            (new_dir / "app.py").write_text("Application code content.")
            cache_dir = new_dir / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "app.pyc").write_text("bytecode")

            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(new_dir)]
            args.out = None
            args.ignore = "__pycache__"
            args.max_files = None

            result = add_command(args)

            assert result == 0

            loaded = RAGlet.load(str(kb_path))
            sources = [c.source for c in loaded.chunks]
            assert not any("__pycache__" in s for s in sources)

    def test_add_command_respects_max_files(self):
        """Test add_command --max-files limits files processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"

            from raglet import RAGlet

            init_file = workspace / "init.txt"
            init_file.write_text("Initial content for raglet.")
            raglet = RAGlet.from_files([str(init_file)])
            raglet.save(str(kb_path))

            new_dir = workspace / "docs"
            new_dir.mkdir()
            for i in range(5):
                (new_dir / f"doc_{i}.txt").write_text(f"Document {i} content about topic {i}.")

            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(new_dir)]
            args.out = None
            args.ignore = None
            args.max_files = 2

            result = add_command(args)
            assert result == 0

            loaded_limited = RAGlet.load(str(kb_path))

            # Compare: add all 5 files without limit
            raglet2 = RAGlet.from_files([str(init_file)])
            raglet2.save(str(kb_path))

            args2 = MagicMock()
            args2.raglet = str(kb_path)
            args2.files = [str(new_dir)]
            args2.out = None
            args2.ignore = None
            args2.max_files = None

            add_command(args2)
            loaded_all = RAGlet.load(str(kb_path))

            assert len(loaded_limited.chunks) < len(loaded_all.chunks)

    def test_add_command_handles_missing_raglet(self):
        """Test add_command handles missing raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "nonexistent.raglet.zip"

            # Create test file
            test_file = workspace / "test.txt"
            test_file.write_text("Test content")

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.files = [str(test_file)]
            args.out = None
            args.ignore = None
            args.max_files = None

            # Run add command (should fail)
            result = add_command(args)

            assert result == 1  # Should fail with missing raglet

    def test_package_command_creates_zip(self):
        """Test package_command creates zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "kb"
            zip_path = workspace / "export.zip"

            # Create knowledge base
            chunks = [
                Chunk(text="Test", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.format = "zip"
            args.out = str(zip_path)

            # Run package command
            result = package_command(args)

            assert result == 0
            assert zip_path.exists()

    def test_package_command_handles_missing_raglet(self):
        """Test package_command handles missing raglet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "nonexistent.raglet"

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.format = "zip"
            args.out = None

            # Run package command (should fail)
            result = package_command(args)

            assert result == 1  # Should fail with missing raglet

    def test_build_command_with_custom_output(self):
        """Test build_command with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            output_path = workspace / "custom_output.raglet"

            # Create test files
            (workspace / "test.txt").write_text("Test content")

            # Mock args
            args = MagicMock()
            args.inputs = [str(workspace / "test.txt")]
            args.out = str(output_path)
            args.ignore = ""
            args.max_files = None
            args.chunk_size = None
            args.chunk_overlap = None
            args.model = None

            # Run build command
            result = build_command(args)

            assert result == 0
            assert output_path.exists()
            assert output_path.is_dir()

    def test_query_command_with_custom_raglet_path(self):
        """Test query_command with custom raglet path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            kb_path = workspace / "custom.raglet.zip"

            # Create raglet
            chunks = [
                Chunk(text="Python", source="test.txt", index=0),
            ]
            from raglet import RAGlet, RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet.from_files([], config=config)
            raglet.chunks = chunks
            raglet.embeddings = raglet.embedding_generator.generate(chunks)
            raglet.vector_store.add_vectors(raglet.embeddings, chunks)
            raglet.save(str(kb_path))

            # Mock args
            args = MagicMock()
            args.raglet = str(kb_path)
            args.query = "Python"
            args.top_k = 5
            args.show_full = False

            # Run query command
            result = query_command(args)

            assert result == 0
