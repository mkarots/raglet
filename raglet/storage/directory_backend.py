"""Directory storage backend implementation."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from raglet.config.config import RAGletConfig
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.storage.interfaces import StorageBackend


class DirectoryStorageBackend(StorageBackend):
    """Directory-based storage backend for RAGlet instances."""

    VERSION = "1.0.0"

    def save(
        self,
        raglet: RAGlet,
        file_path: str,
        incremental: bool = False,
    ) -> None:
        """Save RAGlet to directory structure.

        Args:
            raglet: RAGlet instance to save
            file_path: Path to directory (will be created if doesn't exist)
            incremental: If True, append new chunks (if supported)
                        If False, full save (replace existing)

        Raises:
            ValueError: If save fails
            IOError: If file operations fail
        """
        dir_path = Path(file_path)

        # Create directory if needed
        dir_path.mkdir(parents=True, exist_ok=True)

        if incremental and dir_path.exists() and (dir_path / "chunks.json").exists():
            # Incremental save: append new chunks
            # Only add chunks that aren't already saved
            self._add_chunks_incremental(dir_path, raglet)
        else:
            # Full save: replace all data
            self._save_full(dir_path, raglet)

    def _save_full(self, dir_path: Path, raglet: RAGlet) -> None:
        """Save full RAGlet (replace all data).

        Args:
            dir_path: Directory path
            raglet: RAGlet instance to save
        """
        # Save config.json
        config_path = dir_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(raglet.config.to_dict(), f, indent=2)

        # Save chunks.json
        chunks_path = dir_path / "chunks.json"
        chunks_data = [chunk.to_dict() for chunk in raglet.chunks]
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)

        # Save embeddings.npy
        embeddings_path = dir_path / "embeddings.npy"
        if len(raglet.embeddings) > 0:
            np.save(str(embeddings_path), raglet.embeddings.astype(np.float32))
        else:
            # Create empty array with correct dimension
            embedding_dim = raglet.embedding_generator.get_dimension()
            empty_embeddings = np.array([]).reshape(0, embedding_dim).astype(np.float32)
            np.save(str(embeddings_path), empty_embeddings)

        # Save metadata.json
        metadata_path = dir_path / "metadata.json"
        metadata = {
            "version": self.VERSION,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "chunk_count": len(raglet.chunks),
            "embedding_dim": (
                raglet.embeddings.shape[1]
                if len(raglet.embeddings) > 0
                else raglet.embedding_generator.get_dimension()
            ),
            "embedding_model": raglet.config.embedding.model,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _add_chunks_incremental(self, dir_path: Path, raglet: RAGlet) -> None:
        """Add chunks incrementally to existing directory.

        Args:
            dir_path: Directory path
            raglet: RAGlet instance with new chunks to add
        """
        # Load existing chunks to get current count
        chunks_path = dir_path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path) as f:
                existing_chunks_data = json.load(f)
            current_count = len(existing_chunks_data)
        else:
            current_count = 0

        # Get new chunks (chunks after current count)
        new_chunks = raglet.chunks[current_count:]

        if not new_chunks:
            # No new chunks to add
            return

        # Load all existing chunks
        if chunks_path.exists():
            with open(chunks_path) as f:
                existing_chunks_data = json.load(f)
            existing_chunks = [Chunk.from_dict(c) for c in existing_chunks_data]
        else:
            existing_chunks = []

        # Append new chunks
        updated_chunks = existing_chunks + new_chunks

        # Save updated chunks
        chunks_data = [chunk.to_dict() for chunk in updated_chunks]
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)

        # Append embeddings
        embeddings_path = dir_path / "embeddings.npy"
        if embeddings_path.exists():
            existing_embeddings = np.load(str(embeddings_path))
        else:
            # Create empty array with correct dimension
            embedding_dim = raglet.embedding_generator.get_dimension()
            existing_embeddings = np.array([]).reshape(0, embedding_dim).astype(np.float32)

        # Get embeddings for new chunks only
        new_embeddings = raglet.embeddings[current_count:]

        # Stack embeddings
        updated_embeddings = np.vstack([existing_embeddings, new_embeddings]).astype(np.float32)
        np.save(str(embeddings_path), updated_embeddings)

        # Update metadata
        metadata_path = dir_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata["chunk_count"] = len(updated_chunks)
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "version" not in metadata:
            metadata["version"] = self.VERSION

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, file_path: str) -> RAGlet:
        """Load RAGlet from directory structure.

        Args:
            file_path: Path to directory

        Returns:
            RAGlet instance

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If file format is invalid
            IOError: If file operations fail
        """
        dir_path = Path(file_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {file_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {file_path}")

        # Load config
        config_path = dir_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"config.json not found in {file_path}")
        with open(config_path) as f:
            config = RAGletConfig.from_dict(json.load(f))

        # Load chunks
        chunks_path = dir_path / "chunks.json"
        if not chunks_path.exists():
            raise ValueError(f"chunks.json not found in {file_path}")
        with open(chunks_path) as f:
            chunks_data = json.load(f)
        chunks = [Chunk.from_dict(c) for c in chunks_data]

        # Load embeddings
        embeddings_path = dir_path / "embeddings.npy"
        if not embeddings_path.exists():
            raise ValueError(f"embeddings.npy not found in {file_path}")
        embeddings = np.load(str(embeddings_path))

        # Validate embedding dimension matches model dimension
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

        # Create RAGlet (will rebuild FAISS index on init)
        return RAGlet(
            chunks=chunks,
            config=config,
            embedding_generator=embedding_generator,
            embeddings=embeddings,
        )

    def supports_incremental(self) -> bool:
        """Check if backend supports incremental updates.

        Returns:
            True (directory backend supports incremental saves)
        """
        return True

    def add_chunks(
        self,
        file_path: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        raglet: Optional[RAGlet] = None,
    ) -> None:
        """Add chunks incrementally to existing directory.

        Args:
            file_path: Path to directory
            chunks: New chunks to add
            embeddings: Embeddings for new chunks
            raglet: Optional RAGlet instance (for context)

        Raises:
            ValueError: If directory doesn't exist or format is invalid
            IOError: If file operations fail
        """
        dir_path = Path(file_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {file_path}")

        # Load existing chunks
        chunks_path = dir_path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path) as f:
                existing_chunks_data = json.load(f)
            existing_chunks = [Chunk.from_dict(c) for c in existing_chunks_data]
        else:
            existing_chunks = []

        # Append new chunks
        updated_chunks = existing_chunks + chunks

        # Save updated chunks
        chunks_data = [chunk.to_dict() for chunk in updated_chunks]
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)

        # Append embeddings
        embeddings_path = dir_path / "embeddings.npy"
        if embeddings_path.exists():
            existing_embeddings = np.load(str(embeddings_path))
        else:
            # Create empty array with correct dimension
            if raglet is not None:
                embedding_dim = raglet.embedding_generator.get_dimension()
            else:
                embedding_dim = embeddings.shape[1] if len(embeddings) > 0 else 384
            existing_embeddings = np.array([]).reshape(0, embedding_dim).astype(np.float32)

        # Validate dimension match
        if len(existing_embeddings) > 0 and len(embeddings) > 0:
            if existing_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: Existing embeddings have dimension {existing_embeddings.shape[1]}, "
                    f"but new embeddings have dimension {embeddings.shape[1]}"
                )

        # Stack embeddings
        updated_embeddings = np.vstack([existing_embeddings, embeddings]).astype(np.float32)
        np.save(str(embeddings_path), updated_embeddings)

        # Update metadata
        metadata_path = dir_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata["chunk_count"] = len(updated_chunks)
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "version" not in metadata:
            metadata["version"] = self.VERSION

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
