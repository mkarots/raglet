"""Embedding generator implementation using sentence-transformers."""

import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

# Import sentence-transformers at module level (not lazy)
# CRITICAL: PyTorch (via sentence-transformers) must initialize BEFORE FAISS
# to prevent OpenMP threading conflicts on macOS. Module-level import ensures
# this happens when the module loads, before FAISSVectorStore is created.
#
# Python's import system ensures this is only imported once (cached in sys.modules),
# so this import happens exactly once per Python process.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]

from raglet.config.config import EmbeddingConfig
from raglet.core.chunk import Chunk
from raglet.embeddings.interfaces import EmbeddingGenerator

if TYPE_CHECKING:
    from raglet.cli_utils import CLIOutput

# Model cache: reuse SentenceTransformer instances to avoid reloading weights
# Key: (model_name, device), Value: SentenceTransformer instance
_model_cache: dict[tuple[str, str], SentenceTransformer] = {}
_cache_lock = threading.Lock()


class SentenceTransformerGenerator(EmbeddingGenerator):
    """Generates embeddings using sentence-transformers models."""

    def __init__(self, config: EmbeddingConfig, output: Optional["CLIOutput"] = None):
        """Initialize embedding generator.

        Args:
            config: Embedding configuration
            output: Optional CLI output handler (uses CLI utils if available, otherwise silent)

        Raises:
            ValueError: If model cannot be loaded
        """
        self.config = config
        self.config.validate()
        self._output = output

        # Check if sentence-transformers is available
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required but not installed. "
                "Install with: pip install sentence-transformers"
            )

        # Get or create cached model instance
        cache_key = (config.model, config.device)

        with _cache_lock:
            if cache_key not in _model_cache:
                # First time loading this model - create new instance
                self._warn_model_loading(config.model)
                try:
                    _model_cache[cache_key] = SentenceTransformer(config.model, device=config.device)
                except Exception as e:
                    raise ValueError(f"Failed to load embedding model '{config.model}': {e}") from e

            # Use cached model instance
            self.model = _model_cache[cache_key]
            self._dimension = self.model.get_sentence_embedding_dimension()

    def _warn_model_loading(self, model_name: str) -> None:
        """Warn about model loading using CLI output if available."""
        # Use provided output, or try to get CLI output if available
        output = self._output
        if output is None:
            try:
                from raglet.cli_utils import get_output
                output = get_output()
            except Exception:
                # No CLI context available - silent (library usage)
                return

        if output:
            output.warning(
                f"Loading embedding model '{model_name}'... "
                "This may take up to a minute on first use."
            )

    def generate(self, chunks: list[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks.

        Args:
            chunks: List of Chunk objects to generate embeddings for

        Returns:
            NumPy array of shape (len(chunks), embedding_dim) with embeddings
        """
        if not chunks:
            return np.array([]).reshape(0, self._dimension)  # type: ignore[no-any-return]

        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )

        return np.array(embeddings, dtype=np.float32)  # type: ignore[no-any-return]

    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Text string to generate embedding for

        Returns:
            NumPy array of shape (embedding_dim,) with embedding
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)  # type: ignore[no-any-return]

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return int(self._dimension)
