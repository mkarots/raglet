"""Gate: save/load works for all three formats (directory, SQLite, zip)."""

import os
import tempfile

SECTION = "Storage formats"

SAMPLE_TEXT = (
    "raglet creates portable knowledge bases. "
    "Each one contains chunks, embeddings, and metadata."
)


def _build_rag(tmp: str):
    from raglet import RAGlet

    sample = os.path.join(tmp, "sample.txt")
    with open(sample, "w") as f:
        f.write(SAMPLE_TEXT)
    return RAGlet.from_files([sample])


def register(gate):
    gate(SECTION, "dir-roundtrip",
         "save to directory format then load and search", _directory_roundtrip)
    gate(SECTION, "sqlite-roundtrip",
         "save to SQLite format then load and search", _sqlite_roundtrip)
    gate(SECTION, "zip-roundtrip",
         "save to zip format then load and search", _zip_roundtrip)
    gate(SECTION, "dir-structure",
         "directory format contains config.json, chunks.json, embeddings.npy, metadata.json",
         _directory_structure)
    gate(SECTION, "auto-detect",
         "load() auto-detects format from file extension", _auto_detect)


def _directory_roundtrip():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        save_path = os.path.join(tmp, "kb")
        rag.save(save_path)

        loaded = RAGlet.load(save_path)
        results = loaded.search("knowledge base", top_k=1)
        assert len(results) > 0, "loaded raglet returned no results"
        n = len(loaded.get_all_chunks())
        return f"{n} chunks, top score {results[0].score:.3f}"


def _sqlite_roundtrip():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        save_path = os.path.join(tmp, "knowledge.sqlite")
        rag.save(save_path)
        size_kb = os.path.getsize(save_path) / 1024

        loaded = RAGlet.load(save_path)
        results = loaded.search("knowledge base", top_k=1)
        assert len(results) > 0, "loaded raglet returned no results"
        return f"{size_kb:.1f} KB, {len(loaded.get_all_chunks())} chunks"


def _zip_roundtrip():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        save_path = os.path.join(tmp, "export.zip")
        rag.save(save_path)
        size_kb = os.path.getsize(save_path) / 1024

        loaded = RAGlet.load(save_path)
        results = loaded.search("knowledge base", top_k=1)
        assert len(results) > 0, "loaded raglet returned no results"
        return f"{size_kb:.1f} KB, {len(loaded.get_all_chunks())} chunks"


def _directory_structure():
    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        save_path = os.path.join(tmp, "kb")
        rag.save(save_path)

        expected = {"config.json", "chunks.json", "embeddings.npy", "metadata.json"}
        actual = set(os.listdir(save_path))
        missing = expected - actual
        assert not missing, f"missing files in .raglet/: {missing}"
        return f"files: {', '.join(sorted(actual))}"


def _auto_detect():
    """load() should auto-detect format from the path."""
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)

        dir_path = os.path.join(tmp, "kb")
        sqlite_path = os.path.join(tmp, "test.sqlite")
        zip_path = os.path.join(tmp, "test.zip")

        rag.save(dir_path)
        rag.save(sqlite_path)
        rag.save(zip_path)

        formats_ok = []
        for path in [dir_path, sqlite_path, zip_path]:
            loaded = RAGlet.load(path)
            assert len(loaded.get_all_chunks()) > 0, f"load({path}) produced empty raglet"
            formats_ok.append(os.path.basename(path))
        return f"detected: {', '.join(formats_ok)}"
