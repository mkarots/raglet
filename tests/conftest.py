"""Pytest configuration and shared fixtures."""

import pytest

from raglet.config.config import EmbeddingConfig, RAGletConfig
from raglet.embeddings.generator import SentenceTransformerGenerator

# Cache the embedding generator across tests to avoid reloading the model
# This significantly speeds up test execution
_embedding_generator_cache: dict[str, SentenceTransformerGenerator] = {}


def get_cached_embedding_generator(
    model_name: str = "all-MiniLM-L6-v2",
) -> SentenceTransformerGenerator:
    """Get a cached embedding generator for a model.

    This function caches SentenceTransformer model instances so they're only
    loaded once per test session, dramatically speeding up tests.

    Args:
        model_name: Name of the model to load (default: "all-MiniLM-L6-v2")

    Returns:
        SentenceTransformerGenerator instance (cached)
    """
    if model_name not in _embedding_generator_cache:
        config = EmbeddingConfig(model=model_name)
        _embedding_generator_cache[model_name] = SentenceTransformerGenerator(config)

    return _embedding_generator_cache[model_name]


@pytest.fixture(scope="session")
def cached_embedding_generator():
    """Provide a cached embedding generator for tests (default model).

    This fixture caches the SentenceTransformer model instance so it's only
    loaded once per test session, dramatically speeding up tests.

    Returns:
        SentenceTransformerGenerator instance (cached, default model)
    """
    return get_cached_embedding_generator()


@pytest.fixture(scope="session")
def cached_raglet_config():
    """Provide a cached RAGletConfig for tests.

    Returns:
        RAGletConfig instance with default test settings
    """
    return RAGletConfig(embedding=EmbeddingConfig(model="all-MiniLM-L6-v2"))


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup fixture that runs after each test.

    This ensures resources are properly cleaned up, but doesn't prevent
    model caching across tests.
    """
    yield
    # Any per-test cleanup can go here if needed


def pytest_sessionfinish(session, exitstatus):
    """Cleanup cached generators at end of test session."""
    for generator in _embedding_generator_cache.values():
        try:
            generator.close()
        except Exception:
            pass
    _embedding_generator_cache.clear()
