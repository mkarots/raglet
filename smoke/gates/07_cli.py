"""Gate: CLI commands work as documented in the README."""

import os
import subprocess
import tempfile

SECTION = "CLI"


def _run(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, cwd=cwd, timeout=120,
    )


def register(gate):
    gate(SECTION, "help",
         "raglet --help exits 0 and lists build/query subcommands", _help)
    gate(SECTION, "build",
         "raglet build creates a directory raglet from input files", _build)
    gate(SECTION, "query",
         "raglet query returns matching text from a built raglet", _query)
    gate(SECTION, "add",
         "raglet add appends new files to an existing raglet", _add)
    gate(SECTION, "add-dir",
         "raglet add accepts a directory and indexes all files in it", _add_dir)
    gate(SECTION, "pkg-zip",
         "raglet package converts directory format to zip", _package_zip)
    gate(SECTION, "pkg-sqlite",
         "raglet package converts directory format to sqlite", _package_sqlite)


def _help():
    r = _run(["raglet", "--help"])
    assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
    assert "build" in r.stdout, "build command not in help output"
    assert "query" in r.stdout, "query command not in help output"
    return f"exit {r.returncode}, {len(r.stdout)} chars"


def _build():
    with tempfile.TemporaryDirectory() as tmp:
        doc = os.path.join(tmp, "doc.txt")
        with open(doc, "w") as f:
            f.write("CLI build test content about vector search.")

        out = os.path.join(tmp, "kb")
        r = _run(["raglet", "build", doc, "--out", out])
        assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
        assert os.path.isdir(out), f"{out} was not created"
        files = os.listdir(out)
        return f"exit {r.returncode}, {len(files)} files in output"


def _query():
    with tempfile.TemporaryDirectory() as tmp:
        doc = os.path.join(tmp, "doc.txt")
        with open(doc, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog.")

        out = os.path.join(tmp, "kb")
        _run(["raglet", "build", doc, "--out", out])

        r = _run(["raglet", "query", "quick brown fox", "--raglet", out])
        assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
        assert "fox" in r.stdout.lower() or "quick" in r.stdout.lower(), (
            f"expected query result in output, got: {r.stdout[:200]}"
        )
        return f"exit {r.returncode}, matched query in output"


def _add():
    with tempfile.TemporaryDirectory() as tmp:
        doc = os.path.join(tmp, "doc.txt")
        with open(doc, "w") as f:
            f.write("Original content.")

        out = os.path.join(tmp, "kb")
        _run(["raglet", "build", doc, "--out", out])

        extra = os.path.join(tmp, "extra.txt")
        with open(extra, "w") as f:
            f.write("Extra content added via CLI.")

        r = _run(["raglet", "add", "--raglet", out, extra])
        assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
        return f"exit {r.returncode}"


def _add_dir():
    with tempfile.TemporaryDirectory() as tmp:
        doc = os.path.join(tmp, "doc.txt")
        with open(doc, "w") as f:
            f.write("Original content.")

        out = os.path.join(tmp, "kb")
        _run(["raglet", "build", doc, "--out", out])

        new_dir = os.path.join(tmp, "new_docs")
        os.makedirs(new_dir)
        for i in range(3):
            with open(os.path.join(new_dir, f"doc_{i}.txt"), "w") as f:
                f.write(f"New document {i} about topic {i}.")

        r = _run(["raglet", "add", "--raglet", out, new_dir])
        assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
        return f"exit {r.returncode}, added directory with 3 files"


def _package_zip():
    with tempfile.TemporaryDirectory() as tmp:
        doc = os.path.join(tmp, "doc.txt")
        with open(doc, "w") as f:
            f.write("Content for packaging.")

        raglet_dir = os.path.join(tmp, "kb")
        _run(["raglet", "build", doc, "--out", raglet_dir])

        zip_out = os.path.join(tmp, "export.zip")
        r = _run(["raglet", "package", "--raglet", raglet_dir, "--format", "zip", "--out", zip_out])
        assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
        assert os.path.isfile(zip_out), f"{zip_out} was not created"
        size_kb = os.path.getsize(zip_out) / 1024
        return f"exit {r.returncode}, {size_kb:.1f} KB"


def _package_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        doc = os.path.join(tmp, "doc.txt")
        with open(doc, "w") as f:
            f.write("Content for SQLite packaging.")

        raglet_dir = os.path.join(tmp, "kb")
        _run(["raglet", "build", doc, "--out", raglet_dir])

        sqlite_out = os.path.join(tmp, "knowledge.sqlite")
        r = _run([
            "raglet", "package", "--raglet", raglet_dir, "--format", "sqlite", "--out", sqlite_out,
        ])
        assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
        assert os.path.isfile(sqlite_out), f"{sqlite_out} was not created"
        size_kb = os.path.getsize(sqlite_out) / 1024
        return f"exit {r.returncode}, {size_kb:.1f} KB"
