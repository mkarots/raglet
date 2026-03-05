"""SQLite storage backend implementation."""

import json
import sqlite3
from datetime import datetime
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
                index INTEGER NOT NULL,
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

    def _save_full(self, conn: sqlite3.Connection, raglet: RAGlet) -> None:
        """Save full RAGlet (replace all data).
        
        Args:
            conn: SQLite connection
            raglet: RAGlet instance to save
        """
        # Clear existing data
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM metadata")

        # Insert chunks and embeddings
        for i, chunk in enumerate(raglet.chunks):
            cursor = conn.execute(
                "INSERT INTO chunks (text, source, index, metadata) VALUES (?, ?, ?, ?)",
                (chunk.text, chunk.source, chunk.index, json.dumps(chunk.metadata)),
            )
            chunk_id = cursor.lastrowid

            # Store embedding as BLOB (float32)
            embedding_array = raglet.embeddings[i].astype(np.float32)
            embedding_bytes = embedding_array.tobytes()
            conn.execute(
                "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, embedding_bytes),
            )

        # Save metadata
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("version", self.VERSION),
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("created_at", datetime.utcnow().isoformat() + "Z"),
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

    def _add_chunks_incremental(
        self, conn: sqlite3.Connection, raglet: RAGlet
    ) -> None:
        """Add chunks incrementally (append new chunks).
        
        Args:
            conn: SQLite connection
            raglet: RAGlet instance with new chunks
        """
        # Get current chunk count
        current_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        new_chunks = raglet.chunks[current_count:]

        if not new_chunks:
            return

        # Add new chunks
        for i, chunk in enumerate(new_chunks):
            chunk_index = current_count + i
            cursor = conn.execute(
                "INSERT INTO chunks (text, source, index, metadata) VALUES (?, ?, ?, ?)",
                (chunk.text, chunk.source, chunk.index, json.dumps(chunk.metadata)),
            )
            chunk_id = cursor.lastrowid

            # Store embedding as BLOB (float32)
            embedding_array = raglet.embeddings[chunk_index].astype(np.float32)
            embedding_bytes = embedding_array.tobytes()
            conn.execute(
                "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, embedding_bytes),
            )

        # Update metadata
        new_count = len(raglet.chunks)
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("chunk_count", str(new_count)),
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
        for row in conn.execute(
            "SELECT text, source, index, metadata FROM chunks ORDER BY id"
        ):
            text, source, index, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            chunks.append(Chunk(text=text, source=source, index=index, metadata=metadata))
        return chunks

    def _load_embeddings(self, conn: sqlite3.Connection) -> np.ndarray:
        """Load embeddings from database.
        
        Args:
            conn: SQLite connection
            
        Returns:
            NumPy array of embeddings (shape: [n_chunks, embedding_dim])
            
        Raises:
            ValueError: If embeddings are missing or invalid
        """
        # Get embedding dimension from metadata
        result = conn.execute(
            "SELECT value FROM metadata WHERE key = ?", ("embedding_dim",)
        )
        row = result.fetchone()
        if row is None:
            raise ValueError("embedding_dim not found in metadata")

        embedding_dim = int(row[0])

        # Load all embeddings
        embeddings_list = []
        for row in conn.execute(
            "SELECT embedding FROM embeddings ORDER BY chunk_id"
        ):
            embedding_bytes = row[0]
            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings_list.append(embedding_array)

        if not embeddings_list:
            return np.array([]).reshape(0, embedding_dim)

        embeddings = np.vstack(embeddings_list)
        return embeddings

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
            raise IOError(f"Failed to save RAGlet to {file_path}: {e}") from e
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
            # Load config
            config = self._load_config(conn)

            # Load chunks
            chunks = self._load_chunks(conn)

            # Load embeddings
            embeddings = self._load_embeddings(conn)

            # Rebuild FAISS index
            from raglet.embeddings.generator import SentenceTransformerGenerator
            from raglet.vector_store.faiss_store import FAISSVectorStore

            embedding_generator = SentenceTransformerGenerator(config.embedding)
            
            # Validate embedding dimension matches model dimension
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
            
            vector_store = FAISSVectorStore(
                embedding_dim=embeddings.shape[1] if len(embeddings) > 0 else embedding_generator.get_dimension(),
                config=config.search,
            )

            if len(chunks) > 0:
                vector_store.add_vectors(embeddings, chunks)

            # Create RAGlet
            return RAGlet(
                chunks=chunks,
                config=config,
                embedding_generator=embedding_generator,
                vector_store=vector_store,
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

        conn = sqlite3.connect(str(file_path))
        try:
            self._create_schema(conn)

            # Add new chunks
            for i, chunk in enumerate(chunks):
                cursor = conn.execute(
                    "INSERT INTO chunks (text, source, index, metadata) VALUES (?, ?, ?, ?)",
                    (chunk.text, chunk.source, chunk.index, json.dumps(chunk.metadata)),
                )
                chunk_id = cursor.lastrowid

                # Store embedding as BLOB (float32)
                embedding_array = embeddings[i].astype(np.float32)
                embedding_bytes = embedding_array.tobytes()
                conn.execute(
                    "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, embedding_bytes),
                )

            # Update metadata
            result = conn.execute("SELECT value FROM metadata WHERE key = ?", ("chunk_count",))
            row = result.fetchone()
            current_count = int(row[0]) if row else 0
            new_count = current_count + len(chunks)
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("chunk_count", str(new_count)),
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise IOError(f"Failed to add chunks to {file_path}: {e}") from e
        finally:
            conn.close()
