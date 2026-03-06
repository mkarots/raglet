"""Utility functions for raglet."""

import glob
from pathlib import Path

from raglet.core.rag import DEFAULT_IGNORE_PATTERNS


def expand_file_inputs(
    inputs: list[str],
    ignore_patterns: list[str] | None = None,
) -> list[str]:
    """Expand file inputs (files, directories, glob patterns) into a list of files.

    Supports:
    - Individual files: `["file.txt"]`
    - Directories: `["docs/"]` (recursively finds all files)
    - Glob patterns: `["*.py"]`, `["**/*.md"]`, `["docs/**/*.txt"]`

    Args:
        inputs: List of file paths, directory paths, or glob patterns
        ignore_patterns: Optional list of patterns to ignore.
                        If None, uses DEFAULT_IGNORE_PATTERNS.
                        If provided, merges with DEFAULT_IGNORE_PATTERNS.

    Returns:
        List of file paths (filtered by ignore patterns)

    Raises:
        ValueError: If no files found after filtering
    """
    all_files = []
    for input_path in inputs:
        path = Path(input_path)

        if path.is_file():
            # Individual files
            all_files.append(str(path))
        elif path.is_dir():
            # Directories: recursively find all files
            all_files.extend([str(f) for f in path.rglob("*") if f.is_file()])
        else:
            # Try glob pattern
            matches = glob.glob(input_path, recursive=True)
            if matches:
                all_files.extend([f for f in matches if Path(f).is_file()])
            else:
                # If no glob match, treat as file path (will raise FileNotFoundError later)
                all_files.append(input_path)

    # Filter out ignore patterns
    if ignore_patterns:
        ignore_patterns = ignore_patterns + DEFAULT_IGNORE_PATTERNS
    else:
        ignore_patterns = DEFAULT_IGNORE_PATTERNS

    filtered_files = []
    for file in all_files:
        if not any(pattern in file for pattern in ignore_patterns):
            filtered_files.append(file)

    if not filtered_files:
        raise ValueError("No files found to process.")

    return filtered_files
