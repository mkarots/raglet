"""SQLite storage backend implementation."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from raglet.config.config import RAGletConfig
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.storage.interfaces import StorageBackend


class SQLiteStorageBackend(StorageBackend):
    """SQLite-based storage backend for RAGlet instances."""

    VERSION = "1.0.0"

    def close(self) -> None:
        """Close the storage backend and free resources.

        SQLite connections are created per operation and closed immediately,
        so this is a no-op for consistency with other backends.
        """
        # SQLite connections are created per operation and closed in finally blocks
        # No persistent connection to close
        pass

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema if needed.

        Args:
            conn: SQLite connection
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                "index" INTEGER NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at)")

    _SAVE_BATCH = 512

    def _save_full(self, conn: sqlite3.Connection, raglet: RAGlet) -> None:
        """Save full RAGlet (replace all data).

        Inserts chunks and embeddings in fixed-size batches to keep peak
        memory bounded.  Embeddings are read from the FAISS index via
        ``get_all_vectors()`` so the lazy ``raglet.embeddings`` property
        is never materialised during save.

        Args:
            conn: SQLite connection
            raglet: RAGlet instance to save
        """
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM metadata")

        if raglet.chunks:
            all_vectors = raglet.vector_store.get_all_vectors()

            n = len(raglet.chunks)
            batch = self._SAVE_BATCH
            for start in range(0, n, batch):
                end = min(start + batch, n)

                conn.executemany(
                    'INSERT INTO chunks (text, source, "index", metadata) VALUES (?, ?, ?, ?)',
                    (
                        (c.text, c.source, c.index, json.dumps(c.metadata))
                        for c in raglet.chunks[start:end]
                    ),
                )

            chunk_ids = [row[0] for row in conn.execute("SELECT id FROM chunks ORDER BY id")]

            for start in range(0, n, batch):
                end = min(start + batch, n)
                conn.executemany(
                    "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                    ((chunk_ids[i], all_vectors[i].tobytes()) for i in range(start, end)),
                )

        # Always save metadata (even for empty RAGlet)
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("version", self.VERSION),
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("config", json.dumps(raglet.config.to_dict())),
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("chunk_count", str(len(raglet.chunks))),
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("embedding_dim", str(raglet.embedding_generator.get_dimension())),
        )

    def _add_chunks_incremental(self, conn: sqlite3.Connection, raglet: RAGlet) -> None:
        """Add chunks incrementally (append new chunks).

        Args:
            conn: SQLite connection
            raglet: RAGlet instance with new chunks
        """
        current_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        new_chunks = raglet.chunks[current_count:]

        if not new_chunks:
            return

        all_vectors = raglet.vector_store.get_all_vectors()

        n = len(new_chunks)
        batch = self._SAVE_BATCH
        for start in range(0, n, batch):
            end = min(start + batch, n)
            conn.executemany(
                'INSERT INTO chunks (text, source, "index", metadata) VALUES (?, ?, ?, ?)',
                (
                    (c.text, c.source, c.index, json.dumps(c.metadata))
                    for c in new_chunks[start:end]
                ),
            )

        max_id_result = conn.execute("SELECT MAX(id) FROM chunks").fetchone()
        max_id = max_id_result[0] if max_id_result[0] is not None else 0
        new_chunk_ids = list(range(max_id - n + 1, max_id + 1))

        for start in range(0, n, batch):
            end = min(start + batch, n)
            conn.executemany(
                "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                (
                    (new_chunk_ids[i], all_vectors[current_count + i].tobytes())
                    for i in range(start, end)
                ),
            )

        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("chunk_count", str(len(raglet.chunks))),
        )

    def _load_config(self, conn: sqlite3.Connection) -> RAGletConfig:
        """Load configuration from database.

        Args:
            conn: SQLite connection

        Returns:
            RAGletConfig instance

        Raises:
            ValueError: If config is missing or invalid
        """
        result = conn.execute("SELECT value FROM metadata WHERE key = ?", ("config",))
        row = result.fetchone()
        if row is None:
            # Use default config if not found
            return RAGletConfig()
        config_dict = json.loads(row[0])
        return RAGletConfig.from_dict(config_dict)

    def _load_chunks(self, conn: sqlite3.Connection) -> list[Chunk]:
        """Load chunks from database.

        Args:
            conn: SQLite connection

        Returns:
            List of Chunk objects
        """
        chunks = []
        for row in conn.execute('SELECT text, source, "index", metadata FROM chunks ORDER BY id'):
            text, source, index, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            chunks.append(Chunk(text=text, source=source, index=index, metadata=metadata))
        return chunks

    def _load_embeddings(self, conn: sqlite3.Connection) -> np.ndarray:
        """Load embeddings from database.

        Pre-allocates a single contiguous array and fills it row-by-row,
        avoiding the intermediate list-of-arrays that ``np.vstack`` would
        require (which doubles peak memory at scale).

        Args:
            conn: SQLite connection

        Returns:
            NumPy array of embeddings (shape: [n_chunks, embedding_dim])

        Raises:
            ValueError: If embeddings are missing or invalid
        """
        # Get embedding dimension from metadata
        result = conn.execute("SELECT value FROM metadata WHERE key = ?", ("embedding_dim",))
        row = result.fetchone()
        if row is None:
            raise ValueError("embedding_dim not found in metadata")

        embedding_dim = int(row[0])

        n = int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        if n == 0:
            return np.empty((0, embedding_dim), dtype=np.float32)  # type: ignore[no-any-return]

        embeddings = np.empty((n, embedding_dim), dtype=np.float32)
        for i, row in enumerate(conn.execute("SELECT embedding FROM embeddings ORDER BY chunk_id")):
            embeddings[i] = np.frombuffer(row[0], dtype=np.float32)

        return embeddings  # type: ignore[no-any-return]

    def save(
        self,
        raglet: RAGlet,
        file_path: str,
        incremental: bool = False,
    ) -> None:
        """Save RAGlet to SQLite file.

        Args:
            raglet: RAGlet instance to save
            file_path: Path to SQLite file
            incremental: If True, append new chunks; if False, replace all data

        Raises:
            ValueError: If save fails
            IOError: If file operations fail
        """
        file_path_obj = Path(file_path)

        # Ensure parent directory exists
        if file_path_obj.parent != file_path_obj:  # Not root directory
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Check if path is actually a directory (shouldn't happen, but safety check)
        if file_path_obj.exists() and file_path_obj.is_dir():
            raise ValueError(
                f"Cannot save SQLite file to directory path: {file_path}. "
                f"Use DirectoryStorageBackend for directories."
            )

        conn = sqlite3.connect(str(file_path))
        try:
            self._create_schema(conn)

            if incremental:
                self._add_chunks_incremental(conn, raglet)
            else:
                self._save_full(conn, raglet)

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise OSError(f"Failed to save RAGlet to {file_path}: {e}") from e
        finally:
            conn.close()

    def load(self, file_path: str) -> RAGlet:
        """Load RAGlet from SQLite file.

        Args:
            file_path: Path to SQLite file

        Returns:
            RAGlet instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            IOError: If file operations fail
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        conn = sqlite3.connect(str(file_path))
        try:
            config = self._load_config(conn)
            chunks = self._load_chunks(conn)
            embeddings = self._load_embeddings(conn)

            from raglet.embeddings.generator import SentenceTransformerGenerator

            embedding_generator = SentenceTransformerGenerator(config.embedding)

            if len(embeddings) > 0:
                saved_embedding_dim = embeddings.shape[1]
                model_embedding_dim = embedding_generator.get_dimension()

                if saved_embedding_dim != model_embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: Saved embeddings have dimension {saved_embedding_dim}, "
                        f"but model '{config.embedding.model}' produces dimension {model_embedding_dim}. "
                        f"This indicates the embeddings were generated with a different model. "
                        f"To fix: Regenerate embeddings with the correct model or use the model that matches "
                        f"the saved embeddings."
                    )

            return RAGlet(
                chunks=chunks,
                config=config,
                embedding_generator=embedding_generator,
                embeddings=embeddings,
            )
        except Exception as e:
            raise ValueError(f"Failed to load RAGlet from {file_path}: {e}") from e
        finally:
            conn.close()

    def supports_incremental(self) -> bool:
        """Check if backend supports incremental updates.

        Returns:
            True (SQLite supports incremental updates)
        """
        return True

    def add_chunks(
        self,
        file_path: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        raglet: Optional[RAGlet] = None,
    ) -> None:
        """Add chunks incrementally to existing storage.

        Processes chunks in batches to balance performance and safety.
        Uses batch inserts within each batch for efficiency.

        Args:
            file_path: Path to storage file
            chunks: New chunks to add
            embeddings: Embeddings for new chunks
            raglet: Optional RAGlet instance (for context)

        Raises:
            ValueError: If incremental updates not supported
            IOError: If file operations fail
        """
        if not chunks:
            return

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Batch size: balance between performance and safety
        # Smaller batches = safer (less memory, shorter transactions)
        # Larger batches = faster (fewer round-trips)
        BATCH_SIZE = 512

        conn = sqlite3.connect(str(file_path))
        try:
            self._create_schema(conn)

            # Get the current max chunk_id to calculate new IDs
            result = conn.execute("SELECT MAX(id) FROM chunks")
            max_id_row = result.fetchone()
            max_id = max_id_row[0] if max_id_row[0] is not None else 0

            # Process in batches
            total_chunks = len(chunks)
            for batch_start in range(0, total_chunks, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end]
                batch_size = len(batch_chunks)

                # Prepare chunk data for this batch
                chunk_data = [
                    (chunk.text, chunk.source, chunk.index, json.dumps(chunk.metadata))
                    for chunk in batch_chunks
                ]

                # Batch insert chunks for this batch
                conn.executemany(
                    'INSERT INTO chunks (text, source, "index", metadata) VALUES (?, ?, ?, ?)',
                    chunk_data,
                )

                # Calculate chunk IDs for this batch (sequential after current max_id)
                batch_chunk_ids = list(range(max_id + 1, max_id + 1 + batch_size))

                # Prepare embedding data for this batch
                embedding_data = [
                    (chunk_id, batch_embeddings[i].astype(np.float32, copy=False).tobytes())
                    for i, chunk_id in enumerate(batch_chunk_ids)
                ]

                # Batch insert embeddings for this batch
                conn.executemany(
                    "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                    embedding_data,
                )

                # Update max_id for next batch
                max_id += batch_size

                # Commit after each batch (safer - partial progress on failure)
                conn.commit()

            # Update metadata once at the end
            result = conn.execute("SELECT value FROM metadata WHERE key = ?", ("chunk_count",))
            row = result.fetchone()
            current_count = int(row[0]) if row else 0
            new_count = current_count + total_chunks
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("chunk_count", str(new_count)),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise OSError(f"Failed to add chunks to {file_path}: {e}") from e
        finally:
            conn.close()
