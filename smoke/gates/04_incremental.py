"""Gate: incremental updates (add_text, add_file, add_files, incremental save)."""

import os
import tempfile

SECTION = "Incremental updates"


def _build_rag(tmp: str):
    from raglet import RAGlet

    sample = os.path.join(tmp, "base.txt")
    with open(sample, "w") as f:
        f.write("raglet is a portable knowledge base library.")
    return RAGlet.from_files([sample])


def register(gate):
    gate(SECTION, "add-text",
         "add_text appends new chunks to an existing raglet", _add_text)
    gate(SECTION, "add-text-meta",
         "add_text with custom source and metadata attaches to chunks", _add_text_metadata)
    gate(SECTION, "add-file",
         "add_file indexes a new file into an existing raglet", _add_file)
    gate(SECTION, "add-files",
         "add_files indexes multiple new files at once", _add_files)
    gate(SECTION, "incr-save-dir",
         "incremental save persists new chunks to directory format", _incremental_save_dir)
    gate(SECTION, "incr-save-sqlite",
         "incremental save persists new chunks to SQLite format", _incremental_save_sqlite)


def _add_text():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        before = len(rag.get_all_chunks())
        rag.add_text("Some new information about vector search.")
        after = len(rag.get_all_chunks())
        assert after > before, f"chunk count didn't increase: {before} → {after}"
        return f"{before} → {after} chunks"


def _add_text_metadata():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        rag.add_text("Chat message content", source="chat", metadata={"session": "abc"})
        results = rag.search("chat message", top_k=1)
        assert len(results) > 0
        assert results[0].source == "chat"
        return f"source={results[0].source}, score={results[0].score:.3f}"


def _add_file():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        before = len(rag.get_all_chunks())

        new_file = os.path.join(tmp, "extra.txt")
        with open(new_file, "w") as f:
            f.write("Additional context about embeddings and FAISS indexing.")
        rag.add_file(new_file)

        after = len(rag.get_all_chunks())
        assert after > before, f"chunk count didn't increase: {before} → {after}"
        return f"{before} → {after} chunks"


def _add_files():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        before = len(rag.get_all_chunks())

        files = []
        for i in range(3):
            path = os.path.join(tmp, f"doc_{i}.txt")
            with open(path, "w") as f:
                f.write(f"Document number {i} with content about topic {i}.")
            files.append(path)
        rag.add_files(files)

        after = len(rag.get_all_chunks())
        assert after > before, f"chunk count didn't increase: {before} → {after}"
        return f"{before} → {after} chunks (+{len(files)} files)"


def _incremental_save_dir():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        save_path = os.path.join(tmp, "kb")
        rag.save(save_path)

        rag.add_text("Incrementally added content.")
        rag.save(save_path, incremental=True)

        loaded = RAGlet.load(save_path)
        results = loaded.search("incrementally added", top_k=1)
        assert len(results) > 0, "incremental content not found after reload"
        return f"{len(loaded.get_all_chunks())} chunks after reload"


def _incremental_save_sqlite():
    from raglet import RAGlet

    with tempfile.TemporaryDirectory() as tmp:
        rag = _build_rag(tmp)
        save_path = os.path.join(tmp, "knowledge.sqlite")
        rag.save(save_path)

        rag.add_text("Incrementally added content for SQLite.")
        rag.save(save_path, incremental=True)

        loaded = RAGlet.load(save_path)
        results = loaded.search("incrementally added", top_k=1)
        assert len(results) > 0, "incremental content not found after reload"
        return f"{len(loaded.get_all_chunks())} chunks after reload"
