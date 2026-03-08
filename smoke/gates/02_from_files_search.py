"""Gate: the README 'How it works' code example runs end-to-end."""

import os
import tempfile

SECTION = "from_files + search"

SAMPLE_TEXTS = {
    "api-design.md": (
        "We decided to keep the API surface minimal — just search(), add_text(), and save(). "
        "The goal is that a new user can be productive in under 5 minutes."
    ),
    "meeting-notes.md": (
        "API design discussion: favour explicit save() calls over auto-persistence. "
        "Incremental updates should be opt-in, not default behaviour."
    ),
    "architecture.txt": (
        "raglet is a Python library that creates portable knowledge bases. "
        "Each one is self-contained with chunks, embeddings, and metadata."
    ),
}


def _make_sample_dir(tmp: str) -> str:
    docs = os.path.join(tmp, "docs")
    os.makedirs(docs)
    for name, text in SAMPLE_TEXTS.items():
        with open(os.path.join(docs, name), "w") as f:
            f.write(text)
    return docs


def register(gate):
    gate(SECTION, "quickstart",
         "build from files then search returns scored results", _quickstart)
    gate(SECTION, "dir-input",
         "from_files accepts a directory path and indexes all files in it", _from_files_dir)
    gate(SECTION, "file-list",
         "from_files accepts an explicit list of file paths", _from_files_list)
    gate(SECTION, "ranked-scores",
         "search results are ranked by descending similarity score", _search_results)
    gate(SECTION, "chunk-attrs",
         "each result chunk exposes .text, .source, and .score", _chunk_attributes)


def _quickstart():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        notes = os.path.join(tmp, "notes.md")
        conv = os.path.join(tmp, "conversation.txt")
        with open(notes, "w") as f:
            f.write("We decided to use semantic search for the API design.")
        with open(conv, "w") as f:
            f.write("The team agreed that X should be handled by the retrieval layer.")

        rag = RAGlet.from_files([notes, conv])
        results = rag.search("what did we decide about X?", top_k=5)

        assert len(results) > 0, "search returned no results"
        for chunk in results:
            assert chunk.score is not None

        return f"{len(rag.get_all_chunks())} chunks, top score {results[0].score:.3f}"


def _from_files_dir():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        docs = _make_sample_dir(tmp)
        rag = RAGlet.from_files([docs])
        n = len(rag.get_all_chunks())
        assert n > 0
        return f"{len(SAMPLE_TEXTS)} files, {n} chunks"


def _from_files_list():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        docs = _make_sample_dir(tmp)
        files = [os.path.join(docs, name) for name in SAMPLE_TEXTS]
        rag = RAGlet.from_files(files)
        n = len(rag.get_all_chunks())
        assert n > 0
        return f"{len(files)} files, {n} chunks"


def _search_results():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        docs = _make_sample_dir(tmp)
        rag = RAGlet.from_files([docs])
        results = rag.search("what did we decide about the API design?", top_k=3)
        assert len(results) > 0, "search returned no results"
        assert results[0].score >= results[-1].score, "results not sorted by score"
        return f"{len(results)} results, scores {results[0].score:.3f}..{results[-1].score:.3f}"


def _chunk_attributes():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        docs = _make_sample_dir(tmp)
        rag = RAGlet.from_files([docs])
        results = rag.search("API design", top_k=1)
        chunk = results[0]
        assert hasattr(chunk, "text") and chunk.text, "chunk.text missing or empty"
        assert hasattr(chunk, "source") and chunk.source, "chunk.source missing or empty"
        assert hasattr(chunk, "score") and chunk.score is not None, "chunk.score missing"
        return f"text={len(chunk.text)} chars, source={os.path.basename(chunk.source)}, score={chunk.score:.3f}"
