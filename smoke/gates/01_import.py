"""Gate: top-level imports resolve without errors."""

SECTION = "Imports"


def register(gate):
    gate(SECTION, "import-raglet",
         "from raglet import RAGlet succeeds", _import_raglet)
    gate(SECTION, "import-chunk",
         "from raglet import Chunk succeeds", _import_chunk)
    gate(SECTION, "import-config",
         "all config classes are importable from raglet", _import_config)
    gate(SECTION, "version-set",
         "raglet.__version__ is a valid semver string", _import_version)


def _import_raglet():
    from raglet import RAGlet  # noqa: F401
    return "RAGlet class available"


def _import_chunk():
    from raglet import Chunk  # noqa: F401
    return "Chunk class available"


def _import_config():
    from raglet import RAGletConfig, ChunkingConfig, EmbeddingConfig, SearchConfig  # noqa: F401
    return "RAGletConfig, ChunkingConfig, EmbeddingConfig, SearchConfig"


def _import_version():
    from raglet import __version__

    assert __version__, "__version__ is empty"
    parts = __version__.split(".")
    assert len(parts) >= 3, f"expected semver, got {__version__}"
    return f"v{__version__}"
