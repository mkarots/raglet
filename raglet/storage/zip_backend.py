"""Zip archive storage backend implementation."""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from raglet.config.config import RAGletConfig
from raglet.core.chunk import Chunk
from raglet.core.rag import RAGlet
from raglet.storage.interfaces import StorageBackend


class ZipStorageBackend(StorageBackend):
    """Zip archive storage backend for RAGlet instances.

    Note: Zip format is read-only. Use for export/import only.
    For incremental updates, use DirectoryStorageBackend or SQLiteStorageBackend.
    """

    VERSION = "1.0.0"

    def save(
        self,
        raglet: RAGlet,
        file_path: str,
        incremental: bool = False,
    ) -> None:
        """Save RAGlet to zip archive.

        Args:
            raglet: RAGlet instance to save
            file_path: Path to zip file
            incremental: Not supported for zip format (always full save)

        Raises:
            ValueError: If incremental=True (not supported)
            IOError: If file operations fail
        """
        if incremental:
            raise ValueError(
                "Zip format does not support incremental updates. "
                "Use DirectoryStorageBackend or SQLiteStorageBackend for incremental saves."
            )

        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save to zip
        with zipfile.ZipFile(str(file_path), "w", zipfile.ZIP_DEFLATED) as zipf:
            # Save config
            config_dict = raglet.config.to_dict()
            zipf.writestr("config.json", json.dumps(config_dict, indent=2))

            # Save chunks
            chunks_data = [chunk.to_dict() for chunk in raglet.chunks]
            zipf.writestr("chunks.json", json.dumps(chunks_data, indent=2))

            # Save embeddings
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                np.save(tmp.name, raglet.embeddings)
                tmp.flush()
                with open(tmp.name, "rb") as f:
                    zipf.writestr("embeddings.npy", f.read())
                Path(tmp.name).unlink()

            # Save metadata
            metadata = {
                "version": self.VERSION,
                "chunk_count": len(raglet.chunks),
                "embedding_dim": raglet.embeddings.shape[1] if len(raglet.embeddings) > 0 else 0,
                "embedding_model": raglet.config.embedding.model,
            }
            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

    def load(self, file_path: str) -> RAGlet:
        """Load RAGlet from zip archive.

        Args:
            file_path: Path to zip file

        Returns:
            RAGlet instance

        Raises:
            FileNotFoundError: If zip file doesn't exist
            ValueError: If zip format is invalid
            IOError: If file operations fail
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Zip file not found: {file_path}")
        if not file_path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            with zipfile.ZipFile(str(file_path), "r") as zipf:
                # Load config
                config_data = json.loads(zipf.read("config.json"))
                config = RAGletConfig.from_dict(config_data)

                # Load chunks
                chunks_data = json.loads(zipf.read("chunks.json"))
                chunks = [Chunk.from_dict(c) for c in chunks_data]

                # Load embeddings
                embeddings_bytes = zipf.read("embeddings.npy")
                embeddings = np.load(io.BytesIO(embeddings_bytes))

                # Create RAGlet (will rebuild FAISS index on init)
                return RAGlet(
                    chunks=chunks,
                    config=config,
                    embeddings=embeddings,
                )
        except KeyError as e:
            raise ValueError(f"Invalid zip format: missing file {e}") from e
        except Exception as e:
            raise OSError(f"Failed to load RAGlet from zip: {e}") from e

    def supports_incremental(self) -> bool:
        """Check if backend supports incremental updates.

        Returns:
            False (zip format doesn't support incremental updates)
        """
        return False

    def add_chunks(
        self,
        file_path: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        raglet: RAGlet,
    ) -> None:
        """Add chunks incrementally.

        Note: Zip doesn't support efficient incremental updates.
        This will raise an error.
        """
        raise ValueError(
            "Zip storage backend does not support incremental updates. "
            "Use DirectoryStorageBackend or SQLiteStorageBackend for incremental updates."
        )
