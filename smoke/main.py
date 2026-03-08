"""raglet launch readiness — smoke tests against README claims.

Runs inside a clean Docker container to verify that everything
the README promises actually works after `pip install raglet`.

Usage:
    docker build -t raglet-readiness ./raglet-readiness
    docker run --rm raglet-readiness
    docker run --rm raglet-readiness --verbose
"""

import os
import sys
import time
import importlib
import pkgutil

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SECTION_FMT = "\033[1m\033[94m"  # bold blue
RESET = "\033[0m"
DIM = "\033[2m"

verbose = "--verbose" in sys.argv or "-v" in sys.argv

if not verbose:
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

results: list[tuple[str, str, str, bool, float, str]] = []


def gate(section: str, name: str, desc: str, fn):
    """Run a single gate test and record the result."""
    start = time.time()
    try:
        detail = fn()
        elapsed = time.time() - start
        results.append((section, name, desc, True, elapsed, ""))
        print(f"  [{PASS}] {name} ({elapsed:.2f}s)")
        print(f"         {DIM}{desc}{RESET}")
        if verbose and detail:
            print(f"         {detail}")
    except Exception as e:
        elapsed = time.time() - start
        msg = f"{type(e).__name__}: {e}"
        results.append((section, name, desc, False, elapsed, msg))
        print(f"  [{FAIL}] {name} ({elapsed:.2f}s)")
        print(f"         {DIM}{desc}{RESET}")
        print(f"         {msg}")


def discover_and_run():
    """Import all gate modules and run their register() functions."""
    import gates

    for importer, modname, ispkg in pkgutil.iter_modules(gates.__path__):
        module = importlib.import_module(f"gates.{modname}")
        if hasattr(module, "register"):
            section_name = getattr(module, "SECTION", modname)
            print(f"\n{SECTION_FMT}{section_name}{RESET}")
            module.register(gate)


def print_summary():
    total = len(results)
    passed = sum(1 for *_, ok, _, _ in results if ok)
    failed_count = total - passed
    total_time = sum(t for *_, t, _ in results)

    print(f"\n{'─' * 50}")
    print(f"  {passed}/{total} passed, {failed_count} failed  ({total_time:.1f}s)")
    if failed_count:
        print(f"\n  Failed gates:")
        for section, name, desc, ok, _, msg in results:
            if not ok:
                print(f"    [{section}] {name}")
                print(f"      {msg}")
    print()


if __name__ == "__main__":
    from raglet import __version__

    print(f"raglet v{__version__} — launch readiness")
    if verbose:
        print(f"(verbose mode)")
    print(f"{'─' * 50}")

    discover_and_run()
    print_summary()

    failed_any = any(not ok for *_, ok, _, _ in results)
    sys.exit(1 if failed_any else 0)
