"""Gate: common patterns from the README work correctly."""

import os
import tempfile
from pathlib import Path

SECTION = "Usage patterns"


def register(gate):
    gate(SECTION, "load-or-create",
         "load existing raglet if present, otherwise create from files", _load_or_create)
    gate(SECTION, "context-llm",
         "extract context string from search results for LLM prompt", _context_for_llm)
    gate(SECTION, "with-stmt",
         "RAGlet works as a context manager via with statement", _context_manager)


def _load_or_create():
    """README pattern: load if exists, else create from files."""
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        raglet_path = os.path.join(tmp, "kb")
        sample = os.path.join(tmp, "doc.txt")
        with open(sample, "w") as f:
            f.write("Knowledge about the load-or-create pattern.")

        rag = (
            RAGlet.load(raglet_path)
            if Path(raglet_path).exists()
            else RAGlet.from_files([sample])
        )
        assert len(rag.get_all_chunks()) > 0

        rag.save(raglet_path)
        assert Path(raglet_path).exists()

        rag2 = (
            RAGlet.load(raglet_path)
            if Path(raglet_path).exists()
            else RAGlet.from_files([sample])
        )
        n = len(rag2.get_all_chunks())
        assert n > 0
        return f"create → save → reload: {n} chunks"


def _context_for_llm():
    """README pattern: extract context string from search results."""
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        sample = os.path.join(tmp, "doc.txt")
        with open(sample, "w") as f:
            f.write("raglet handles retrieval. You handle generation.")

        rag = RAGlet.from_files([sample])
        results = rag.search("retrieval", top_k=5)
        context = "\n\n".join(chunk.text for chunk in results)
        assert isinstance(context, str)
        assert len(context) > 0
        return f"{len(results)} results, {len(context)} chars of context"


def _context_manager():
    """RAGlet supports `with` statement."""
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        sample = os.path.join(tmp, "doc.txt")
        with open(sample, "w") as f:
            f.write("Context manager test content.")

        with RAGlet.from_files([sample]) as rag:
            results = rag.search("context manager", top_k=1)
            assert len(results) > 0
            return f"score={results[0].score:.3f}"
