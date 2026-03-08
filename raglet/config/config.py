"""Configuration classes."""

from dataclasses import dataclass, field
from typing import Any, Optional


def _select_device() -> str:
    """Select best available inference device.

    Returns:
        Device string: "cuda" if CUDA is available, "mps" if Apple Silicon MPS
        is available, otherwise "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except ImportError:
        # torch not installed, fall back to cpu
        pass
    return "cpu"


def _default_batch_size() -> int:
    """Select a sensible default encode batch size based on the active device.

    Larger batches amortise GPU launch overhead; smaller batches avoid OOM on CPU.
    Values are based on empirical testing with all-MiniLM-L6-v2.

    Returns:
        Recommended batch size for the current device.
    """
    device = _select_device()
    if device == "cuda":
        return 256
    if device == "mps":
        return 128
    return 32


def _default_fp16() -> bool:
    """Enable fp16 by default on GPU devices (MPS/CUDA).

    Half-precision halves memory bandwidth pressure and gives 1.5-2x throughput
    on GPU at negligible quality loss for retrieval tasks. No benefit on CPU.
    """
    return _select_device() in ("mps", "cuda")


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    size: int = 256
    overlap: int = 50
    strategy: str = "sentence-aware"

    def validate(self) -> None:
        """Validate chunking configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.overlap >= self.size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.strategy not in ["fixed", "sentence-aware", "semantic"]:
            raise ValueError(f"Invalid chunk_strategy: {self.strategy}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "size": self.size,
            "overlap": self.overlap,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkingConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            ChunkingConfig instance
        """
        return cls(
            size=data.get("size", 256),
            overlap=data.get("overlap", 50),
            strategy=data.get("strategy", "sentence-aware"),
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = "all-MiniLM-L6-v2"
    batch_size: int = field(default_factory=_default_batch_size)
    """Number of chunks encoded per model forward pass.

    Defaults are device-aware: 32 (CPU), 128 (MPS), 256 (CUDA).
    Larger values improve GPU utilisation but increase VRAM usage. Tune
    downward if you hit out-of-memory errors on long chunks.
    """
    device: str = field(default_factory=_select_device)
    normalize: bool = True  # Deprecated: Normalization is handled by FAISS for consistency
    use_fp16: bool = field(default_factory=_default_fp16)
    """Enable float16 (half-precision) inference on MPS or CUDA devices.

    Defaults to True on MPS/CUDA, False on CPU. Typically gives 1.5-2x
    throughput at negligible quality loss for retrieval tasks. Embeddings
    are always returned as float32 regardless of this setting.
    """
    torch_compile: bool = False
    """Enable torch.compile() graph optimisation. Requires PyTorch ≥ 2.0.

    Gives roughly 15–25% additional throughput on MPS/CUDA after a one-time
    compilation warmup (typically 10–30 s on first call). Falls back
    gracefully if torch.compile is unavailable or raises an error.
    """

    def validate(self) -> None:
        """Validate embedding configuration.

        Falls back to CPU with a warning if the requested device is not
        available (e.g. ``device="cuda"`` on macOS, or ``device="mps"``
        on Linux).  Also adjusts ``use_fp16`` since half-precision has
        no benefit on CPU.

        Raises:
            ValueError: If configuration is invalid
        """
        import warnings

        if not self.model:
            raise ValueError("embedding model must be specified")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError("device must be 'cpu', 'cuda', or 'mps'")

        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    warnings.warn(
                        "device='cuda' requested but CUDA is not available. "
                        "Falling back to CPU.",
                        stacklevel=2,
                    )
                    self.device = "cpu"
                    self.use_fp16 = False
            except ImportError:
                warnings.warn(
                    "device='cuda' requested but PyTorch is not installed. " "Falling back to CPU.",
                    stacklevel=2,
                )
                self.device = "cpu"
                self.use_fp16 = False

        if self.device == "mps":
            try:
                import torch

                if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                    warnings.warn(
                        "device='mps' requested but MPS is not available. " "Falling back to CPU.",
                        stacklevel=2,
                    )
                    self.device = "cpu"
                    self.use_fp16 = False
            except ImportError:
                warnings.warn(
                    "device='mps' requested but PyTorch is not installed. " "Falling back to CPU.",
                    stacklevel=2,
                )
                self.device = "cpu"
                self.use_fp16 = False

        if self.use_fp16 and self.device == "cpu":
            warnings.warn(
                "use_fp16=True has no benefit on CPU. Setting use_fp16=False.",
                stacklevel=2,
            )
            self.use_fp16 = False

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "model": self.model,
            "batch_size": self.batch_size,
            "device": self.device,
            "normalize": self.normalize,
            "use_fp16": self.use_fp16,
            "torch_compile": self.torch_compile,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            EmbeddingConfig instance
        """
        return cls(
            model=data.get("model", "all-MiniLM-L6-v2"),
            batch_size=data.get("batch_size", _default_batch_size()),
            device=data.get("device", _select_device()),
            normalize=data.get("normalize", True),
            use_fp16=data.get("use_fp16", _default_fp16()),
            torch_compile=data.get("torch_compile", False),
        )


@dataclass
class SearchConfig:
    """Configuration for vector search."""

    default_top_k: int = 5
    similarity_threshold: Optional[float] = None
    index_type: str = "flat_ip"  # Cosine similarity (IndexFlatIP)

    def validate(self) -> None:
        """Validate search configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")
        if self.similarity_threshold is not None:
            if not 0.0 <= self.similarity_threshold <= 1.0:
                raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.index_type != "flat_ip":
            raise ValueError(
                f"Invalid index_type: {self.index_type}. Only 'flat_ip' (cosine similarity) is supported."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        result = {
            "default_top_k": self.default_top_k,
            "index_type": self.index_type,
        }
        if self.similarity_threshold is not None:
            result["similarity_threshold"] = self.similarity_threshold
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            SearchConfig instance
        """
        return cls(
            default_top_k=data.get("default_top_k", 5),
            similarity_threshold=data.get("similarity_threshold"),
            index_type=data.get("index_type", "flat_ip"),  # Default to cosine similarity
        )


@dataclass
class RAGletConfig:
    """Main configuration class."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate entire configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        self.chunking.validate()
        self.embedding.validate()
        self.search.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config with nested configs
        """
        return {
            "chunking": self.chunking.to_dict(),
            "embedding": self.embedding.to_dict(),
            "search": self.search.to_dict(),
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGletConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with config values (may include nested configs)

        Returns:
            RAGletConfig instance
        """
        chunking_data = data.get("chunking", {})
        embedding_data = data.get("embedding", {})
        search_data = data.get("search", {})

        return cls(
            chunking=ChunkingConfig.from_dict(chunking_data),
            embedding=EmbeddingConfig.from_dict(embedding_data),
            search=SearchConfig.from_dict(search_data),
            custom_metadata=data.get("custom_metadata", {}),
        )
