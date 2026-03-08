"""Embedding generator implementation using sentence-transformers."""

import atexit
import os
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

# Disable loky's parallel processing to avoid semaphore leaks
# sentence-transformers uses loky internally, which can cause resource leaks
# Setting this environment variable disables loky's worker pool
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")


def _cleanup_torch_workers() -> None:
    """Release PyTorch shared-memory semaphores before process exit.

    PyTorch allocates POSIX named semaphores for shared-memory coordination
    between its internal CPU worker threads. These must be released before FAISS
    runs its own cleanup at process exit to prevent use-after-free crashes.

    Note: We also set environment variables (LOKY_MAX_CPU_COUNT=1, JOBLIB_MULTIPROCESSING=0)
    to minimize semaphore allocation, but cleanup is still needed for any that are created.
    """
    try:
        import torch.multiprocessing as mp
        for proc in mp.active_children():
            proc.terminate()
            proc.join(timeout=1)
    except Exception:
        pass  # best-effort; never raise from atexit


# Register cleanup handler at module level to ensure it runs before FAISS cleanup
atexit.register(_cleanup_torch_workers)

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
    SentenceTransformer = None

from raglet.config.config import EmbeddingConfig
from raglet.core.chunk import Chunk
from raglet.embeddings.interfaces import EmbeddingGenerator

if TYPE_CHECKING:
    from raglet.cli_utils import CLIOutput


# Module-level cache for SentenceTransformer models.
# Key: (model_name, device, use_fp16, torch_compile) — all fields that affect the
# loaded model's state are included so that differently-configured generators do
# not share an incompatible cached instance.
# Value: SentenceTransformer instance
_model_cache: dict[tuple[str, str, bool, bool], "SentenceTransformer"] = {}
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

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers (and PyTorch) are required but not installed. "
                "Install with: pip install sentence-transformers torch\n"
                "Note: PyTorch does not support Alpine Linux (musl). "
                "Use a glibc-based image (e.g. python:3.11-slim) instead of Alpine."
            )

        # Cache key includes every config field that affects the model's runtime state.
        # use_fp16 and torch_compile both mutate the model object, so they must be
        # part of the key — otherwise a fp32 generator could receive a fp16 model.
        cache_key = (config.model, config.device, config.use_fp16, config.torch_compile)
        self._cache_key = cache_key
        self._owns_model = False  # Track if we own the model or it's cached

        with _cache_lock:
            if cache_key not in _model_cache:
                # First time loading this (model, device, precision, compile) combination
                self._warn_model_loading(config.model)
                try:
                    # Use local_files_only=True to skip HuggingFace Hub network calls.
                    # This saves ~1-2s on every run by avoiding hf_hub_download/metadata checks.
                    # If the model isn't locally cached it will raise immediately rather than hanging.
                    model = SentenceTransformer(
                        config.model,
                        device=config.device,
                        local_files_only=True,
                    )

                    # --- float16 half-precision ---
                    # Converting weights to fp16 halves memory bandwidth pressure and
                    # typically gives 1.5–2× throughput on MPS/CUDA at negligible
                    # quality loss for retrieval tasks.
                    if config.use_fp16:
                        model.half()

                    # --- torch.compile graph optimisation ---
                    # Wraps the underlying transformer's forward pass with PyTorch's
                    # compiler. The first encode() call after this will trigger a
                    # compilation warmup (10–30 s); subsequent calls are faster.
                    # We target only the transformer module, not the full
                    # SentenceTransformer wrapper, to avoid compile-incompatible pooling ops.
                    if config.torch_compile:
                        try:
                            import torch

                            if hasattr(torch, "compile"):
                                transformer_module = model[0].auto_model
                                model[0].auto_model = torch.compile(
                                    transformer_module,
                                    mode="reduce-overhead",  # minimises per-call overhead
                                )
                        except Exception:
                            # torch.compile is best-effort: log silently and continue.
                            pass

                    _model_cache[cache_key] = model
                    self._owns_model = True
                except Exception as e:
                    raise ValueError(f"Failed to load embedding model '{config.model}': {e}") from e
            else:
                # Model already cached — reuse without re-applying transformations.
                self._owns_model = False

        # Get model from cache (safe to access without lock after the cache check above)
        self.model = _model_cache[cache_key]
        # Warmup: ensures compilation (if enabled) and MPS kernel JIT happen now,
        # not during the first real encode call.
        self.model.encode(["warmup"], show_progress_bar=False)
        self._dimension = self.model.get_sentence_embedding_dimension()

        # Pre-compute a character limit from the model's token window.
        # sentence-transformers truncates at max_seq_length tokens internally,
        # so any characters beyond that are tokenised then discarded.  By
        # pre-slicing the text we skip tokenisation of the wasted tail.
        # WordPiece averages ~4-5 chars/token on English; ×6 gives a safe
        # margin that never clips content the model would have kept.
        self._char_limit: int = self.model.max_seq_length * 6

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
            NumPy array of shape (len(chunks), embedding_dim) with float32 embeddings
        """
        if not chunks:
            return np.empty((0, self._dimension), dtype=np.float32)  # type: ignore[no-any-return]

        total_chunks = len(chunks)
        output = np.empty((total_chunks, self._dimension), dtype=np.float32)

        # Extract texts and truncate to the model's effective character window.
        # The model's tokeniser truncates at max_seq_length tokens anyway;
        # pre-slicing avoids tokenising characters that would be discarded.
        char_limit = self._char_limit
        texts = [chunk.text[:char_limit] for chunk in chunks]

        # batch_size controls the number of texts processed in a single GPU/CPU
        # forward pass.  The value comes from EmbeddingConfig, which selects
        # device-appropriate defaults (32 CPU / 128 MPS / 256 CUDA) and can be
        # overridden by the caller.  Previously this was hardcoded to 1000, which
        # ignored user configuration entirely.
        batch_embeddings = self.model.encode(
            texts,
            normalize_embeddings=False,  # FAISS normalises before indexing
            show_progress_bar=True,
            convert_to_numpy=True,       # avoids an extra tensor→ndarray conversion
            batch_size=self.config.batch_size,
        )

        # fp16 models return float16 arrays; FAISS requires float32.
        # astype with copy=False is a no-op when dtype already matches.
        if batch_embeddings.dtype != np.float32:
            batch_embeddings = batch_embeddings.astype(np.float32, copy=False)

        output[:] = batch_embeddings
        return output  # type: ignore[no-any-return]


    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Text string to generate embedding for

        Returns:
            NumPy array of shape (embedding_dim,) with embedding
        """
        embedding = self.model.encode(
            text[:self._char_limit],
            normalize_embeddings=False,  # FAISS normalises before indexing
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)  # type: ignore[no-any-return]

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return int(self._dimension)

    def close(self) -> None:
        """Close the embedding generator and free resources.

        Note: sentence-transformers uses loky internally for parallel processing,
        which may cause semaphore leak warnings. These are harmless warnings that
        don't affect functionality - loky executors are cleaned up at process shutdown.
        """
        if not hasattr(self, 'model') or self.model is None:
            return

        # Clear reference (but don't remove from cache if cached)
        # Note: We don't attempt to clean up loky executors here because:
        # 1. sentence-transformers manages them internally
        # 2. They're cleaned up at process shutdown
        # 3. Aggressive cleanup can interfere with other generators using cached models
        self.model = None


def clear_model_cache() -> None:
    """Clear the model cache and free all cached models.

    This function should be called when you want to free all cached models,
    for example at the end of a test suite or when switching models.

    Note: This will close all cached models, so any generators still using
    them will need to be recreated.
    """
    global _model_cache

    with _cache_lock:
        # Close all cached models
        for model in _model_cache.values():
            try:
                # Attempt cleanup similar to SentenceTransformerGenerator.close()
                if hasattr(model, '_modules'):
                    for module in model._modules.values():
                        if hasattr(module, 'shutdown'):
                            try:
                                module.shutdown(wait=True)
                            except Exception:
                                pass
            except Exception:
                pass

        _model_cache.clear()
