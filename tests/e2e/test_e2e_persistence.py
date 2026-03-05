"""End-to-end tests for persistence workflow."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from raglet.core.rag import RAGlet


@pytest.mark.e2e
class TestE2EPersistence:
    """End-to-end tests for full persistence workflow."""

    def test_full_workflow_save_load(self):
        """Test complete workflow: create → save → load → use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create RAGlet from files
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text(
                "Python is a high-level programming language.\n"
                "Machine learning is a subset of artificial intelligence.\n"
                "RAG systems combine retrieval and generation."
            )
            raglet1 = RAGlet.from_files([str(test_file)])

            # Step 2: Save to SQLite
            db_path = Path(tmpdir) / "knowledge.sqlite"
            raglet1.save(str(db_path))

            # Step 3: Load from SQLite
            raglet2 = RAGlet.load(str(db_path))

            # Step 4: Verify data integrity
            assert len(raglet2.chunks) == len(raglet1.chunks)
            assert raglet2.embeddings.shape == raglet1.embeddings.shape
            np.testing.assert_array_almost_equal(raglet2.embeddings, raglet1.embeddings, decimal=5)

            # Step 5: Use loaded RAGlet (search)
            results = raglet2.search("programming language")
            assert len(results) > 0
            assert any("Python" in chunk.text for chunk in results)

    def test_agentic_loop_simulation(self):
        """Test agentic loop pattern: load → search → add → save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initial setup: Create and save initial knowledge base
            initial_file = Path(tmpdir) / "initial.txt"
            initial_file.write_text("Initial knowledge base content.")
            raglet = RAGlet.from_files([str(initial_file)])

            db_path = Path(tmpdir) / "memory.sqlite"
            raglet.save(str(db_path))

            # Session 1: Load existing memory
            raglet = RAGlet.load(str(db_path))
            initial_count = len(raglet.chunks)

            # Search existing memory
            results = raglet.search("knowledge")
            assert len(results) > 0

            # Add new conversation chunks
            from raglet.core.chunk import Chunk

            conversation_chunks = [
                Chunk(text="User: What is RAG?", source="chat", index=0),
                Chunk(
                    text="Agent: RAG stands for Retrieval-Augmented Generation.",
                    source="chat",
                    index=1,
                ),
            ]
            raglet.add_chunks(conversation_chunks)

            # Save at end of session
            raglet.save(str(db_path), incremental=True)

            # Session 2: Load with all previous conversations
            raglet2 = RAGlet.load(str(db_path))
            assert len(raglet2.chunks) == initial_count + 2

            # Search should find new content
            results = raglet2.search("RAG")
            assert len(results) > 0
            assert any("Retrieval-Augmented" in chunk.text for chunk in results)

    def test_incremental_updates_across_sessions(self):
        """Test incremental updates across multiple sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"

            # Session 1: Create initial knowledge base
            file1 = Path(tmpdir) / "doc1.txt"
            file1.write_text("Document 1 content.")
            raglet = RAGlet.from_files([str(file1)])
            raglet.save(str(db_path))

            # Session 2: Add more content
            raglet = RAGlet.load(str(db_path))
            from raglet.core.chunk import Chunk

            raglet.add_chunks([Chunk(text="Session 2 addition", source="session2", index=0)])
            raglet.save(str(db_path), incremental=True)

            # Session 3: Add even more content
            raglet = RAGlet.load(str(db_path))
            raglet.add_chunks([Chunk(text="Session 3 addition", source="session3", index=0)])
            raglet.save(str(db_path), incremental=True)

            # Final verification
            final = RAGlet.load(str(db_path))
            assert len(final.chunks) >= 3  # Initial + 2 additions

            # All content should be searchable
            results = final.search("Session")
            assert len(results) >= 2

    def test_from_sqlite_inspection_workflow(self):
        """Test inspection workflow using from_sqlite()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save RAGlet
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content for inspection.")
            raglet = RAGlet.from_files([str(test_file)])

            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))

            # Inspect using from_sqlite()
            inspected = RAGlet.from_sqlite(str(db_path))

            # Verify inspection works
            assert len(inspected.chunks) == len(raglet.chunks)
            assert inspected.embeddings.shape == raglet.embeddings.shape

            # Should be able to search
            results = inspected.search("inspection")
            assert len(results) > 0

    def test_config_persistence(self):
        """Test that custom config is preserved across save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from raglet.config.config import RAGletConfig

            # Create with custom config
            config = RAGletConfig()
            config.chunking.size = 1024
            config.search.default_top_k = 10
            config.custom_metadata = {"project": "test", "version": "1.0"}

            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content.")
            raglet = RAGlet.from_files([str(test_file)], config=config)

            db_path = Path(tmpdir) / "test.sqlite"
            raglet.save(str(db_path))

            # Load and verify config
            loaded = RAGlet.load(str(db_path))
            assert loaded.config.chunking.size == 1024
            assert loaded.config.search.default_top_k == 10
            assert loaded.config.custom_metadata == {"project": "test", "version": "1.0"}

    def test_empty_raglet_save_load(self):
        """Test saving and loading empty RAGlet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from raglet.config.config import RAGletConfig

            config = RAGletConfig()
            raglet = RAGlet(chunks=[], config=config)

            db_path = Path(tmpdir) / "empty.sqlite"
            raglet.save(str(db_path))

            loaded = RAGlet.load(str(db_path))
            assert len(loaded.chunks) == 0
            assert loaded.embeddings.shape[0] == 0

            # Search should return empty
            results = loaded.search("anything")
            assert results == []
