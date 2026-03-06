#!/usr/bin/env python3
"""raglet CLI - Command-line interface for raglet."""

import argparse
import sys
from pathlib import Path

from raglet import RAGlet, RAGletConfig
from raglet.cli_utils import get_output, init_output


def build_command(args: argparse.Namespace) -> int:
    """Build raglet from inputs (files, directories, or glob patterns).

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    output = get_output()

    if not args.inputs:
        output.error("No inputs provided. Specify files, directories, or glob patterns.")
        return 1

    # Expand inputs: files, directories, glob patterns
    from raglet.utils import expand_file_inputs

    try:
        ignore_patterns = args.ignore.split(",") if args.ignore else None
        filtered_files = expand_file_inputs(args.inputs, ignore_patterns=ignore_patterns)
    except ValueError as e:
        output.error(str(e))
        return 1

    # Determine output path (build only creates directories)
    # --out is required, so args.out is guaranteed to be set
        output_path = Path(args.out)

    # Analyze original inputs to show what's being processed
    dirs = [f for f in args.inputs if Path(f).is_dir()]
    file_inputs = [f for f in args.inputs if Path(f).is_file()]
    globs = [f for f in args.inputs if not Path(f).exists() and ("*" in f or "?" in f)]

    parts = []
    if dirs:
        parts.append(f"{len(dirs)} director{'y' if len(dirs) == 1 else 'ies'}")
    if file_inputs:
        parts.append(f"{len(file_inputs)} file{'s' if len(file_inputs) != 1 else ''}")
    if globs:
        parts.append(f"{len(globs)} glob pattern{'s' if len(globs) != 1 else ''}")

    if parts:
        input_desc = ", ".join(parts)
        output.info(
            f"Found {input_desc} ({len(filtered_files)} file{'s' if len(filtered_files) != 1 else ''} total)..."
        )
    else:
        output.info(f"Found {len(filtered_files)} files to process...")

    output.progress(f"Building raglet: {output_path}")

    try:
        # Create config
        config = RAGletConfig()
        if args.chunk_size:
            config.chunking.size = args.chunk_size
        if args.chunk_overlap:
            config.chunking.overlap = args.chunk_overlap
        if args.model:
            config.embedding.model = args.model

        # Show configuration
        output.section("Configuration:")
        output.info(f"  Chunk size: {config.chunking.size}")
        output.info(f"  Chunk overlap: {config.chunking.overlap}")
        output.info(f"  Embedding model: {config.embedding.model}")
        output.info(f"  Device: {config.embedding.device}")

        # Limit files if specified
        files_to_process = filtered_files[: args.max_files] if args.max_files else filtered_files
        if args.max_files and len(files_to_process) < len(filtered_files):
            output.info(
                f"  Processing {len(files_to_process)} of {len(filtered_files)} files (--max-files limit)"
            )

        # Build RAGlet
        raglet = RAGlet.from_files(files_to_process, config=config, output=output)

        # Save
        output.verbose_msg("Saving raglet...")
        raglet.save(str(output_path))

        output.success(f"Raglet built: {len(raglet.chunks)} chunks → {output_path}")
        return 0

    except Exception as e:
        output.error(f"Building raglet failed: {e}")
        if getattr(args, "verbose", False):
        import traceback

        traceback.print_exc()
        return 1


def query_command(args: argparse.Namespace) -> int:
    """Query raglet (single query, non-interactive).

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    output = get_output()

    raglet_path = Path(args.raglet)
    if not raglet_path.exists():
        output.error(f"Raglet not found at {raglet_path}.")
        output.info("Create one with: raglet build <inputs> --out <path>")
        return 1

    try:
        # Load RAGlet
        output.verbose_msg(f"Loading raglet from {raglet_path}...")
        raglet = RAGlet.load(str(raglet_path))

        # Search
        output.verbose_msg(f"Searching for: {args.query}")
        results = raglet.search(args.query, top_k=args.top_k)

        if not results:
            output.info("No results found.")
            return 0

        # Print results
        output.header(f"Found {len(results)} result{'s' if len(results) != 1 else ''}:")
        for i, result in enumerate(results, 1):
            score = result.score if result.score is not None else 0.0
            output.print(f"\n{i}. [{score:.3f}] {result.source}:{result.index}")
            preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
            output.print(f"   {preview}")
            if args.show_full:
                output.print(f"   Full text: {result.text}")

        return 0

    except Exception as e:
        output.error(f"Querying raglet failed: {e}")
        return 1


def add_command(args: argparse.Namespace) -> int:
    """Add files to existing raglet.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    output = get_output()

    raglet_path = Path(args.raglet)
    if not raglet_path.exists():
        output.error(f"Raglet not found at {raglet_path}.")
        output.info("Create one with: raglet build <inputs> --out <path>")
        return 1

    if not args.files:
        output.error("No files specified to add.")
        return 1

    # Resolve file paths
    files_to_add = []
    for file_arg in args.files:
        file_path = Path(file_arg)
        if file_path.exists() and file_path.is_file():
            files_to_add.append(str(file_path))
            output.info(f"Adding file: {file_path}")
        else:
            output.warning(f"File not found: {file_path}")

    if not files_to_add:
        output.error("No valid files to add.")
        return 1

    try:
        # Load existing RAGlet
        output.verbose_msg(f"Loading raglet from {raglet_path}...")
        raglet = RAGlet.load(str(raglet_path))
        initial_chunks = len(raglet.chunks)

        # Show configuration
        output.section("Configuration:")
        output.info(f"  Chunk size: {raglet.config.chunking.size}")
        output.info(f"  Chunk overlap: {raglet.config.chunking.overlap}")
        output.info(f"  Embedding model: {raglet.config.embedding.model}")

        # Add files
        raglet.add_files(files_to_add, output=output)

        # Determine output path
        output_path = Path(args.out) if args.out else raglet_path

        # Check if backend supports incremental updates
        backend = raglet._get_default_backend(str(output_path))
        if backend.supports_incremental():
            output.verbose_msg("Saving incrementally...")
        raglet.save(str(output_path), incremental=True)
        else:
            output.verbose_msg("Saving (full save, incremental not supported)...")
            raglet.save(str(output_path), incremental=False)

        new_chunks = len(raglet.chunks) - initial_chunks
        output.success(
            f"Added {len(files_to_add)} file{'s' if len(files_to_add) != 1 else ''} ({new_chunks} new chunks)"
        )
        output.info(f"  Total chunks: {len(raglet.chunks)}")
        return 0

    except Exception as e:
        output.error(f"Adding files failed: {e}")
        return 1


def package_command(args: argparse.Namespace) -> int:
    """Convert raglet between formats (directory, zip, sqlite).

    Supports all conversions:
    - Directory ↔ Zip
    - Directory ↔ SQLite
    - Zip ↔ SQLite

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    output = get_output()

    raglet_path = Path(args.raglet)
    if not raglet_path.exists():
        output.error(f"Raglet not found at {raglet_path}.")
        return 1

    # Determine output path
    if args.out:
        output_path = Path(args.out)
    else:
        # Default: same name with new extension based on target format
        if args.format == "zip":
            output_path = raglet_path.with_suffix(".zip")
        elif args.format == "sqlite":
            output_path = raglet_path.with_suffix(".sqlite")
        else:  # dir
            # For directory output, use the path as-is or add .raglet suffix
            if raglet_path.suffix in [".zip", ".sqlite", ".db"]:
                output_path = raglet_path.with_suffix("")
            else:
                output_path = raglet_path / ".raglet" if raglet_path.is_file() else raglet_path

    try:
        # Load RAGlet (auto-detects format)
        output.verbose_msg(f"Loading raglet from {raglet_path}...")
        raglet = RAGlet.load(str(raglet_path))

        # Save in new format
        output.progress(f"Converting to {args.format} format...")
        raglet.save(str(output_path))

        output.success(f"Packaged raglet: {output_path} (format: {args.format})")
        return 0

    except Exception as e:
        output.error(f"Packaging raglet failed: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="raglet - Portable memory for small text corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build raglet from files (creates directory)
  raglet build file1.txt file2.md --out my-raglet/
  raglet build ./docs --out docs/
  raglet build "*.md" --out docs-kb/

  # Query raglet (works with any format)
  raglet query "python" --raglet my-raglet/ --top-k 5
  raglet query "python" --raglet export.zip --top-k 5
  raglet query "python" --raglet knowledge.sqlite --top-k 5

  # Package (convert between formats)
  raglet package --raglet my-raglet/ --format zip --out export.zip
  raglet package --raglet export.zip --format sqlite --out knowledge.sqlite
  raglet package --raglet knowledge.sqlite --format dir --out .raglet/

  # Add files incrementally (works with any format)
  raglet add --raglet my-raglet/ file3.txt
  raglet add --raglet knowledge.sqlite file3.txt
        """,
    )

    # Global flags
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build raglet from inputs")
    build_parser.add_argument(
        "inputs",
        nargs="+",
        help="Files, directories, or glob patterns to process",
    )
    build_parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory path (required)",
    )
    build_parser.add_argument(
        "--ignore",
        type=str,
        default=".git,__pycache__,.venv,node_modules,.raglet",
        help="Patterns to ignore (comma-separated)",
    )
    build_parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (default: 512)",
    )
    build_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap (default: 50)",
    )
    build_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model (default: all-MiniLM-L6-v2)",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query raglet (single query)")
    query_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    query_parser.add_argument(
        "--raglet",
        type=str,
        required=True,
        help="Path to raglet (.zip, .sqlite, or directory)",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    query_parser.add_argument(
        "--show-full",
        action="store_true",
        help="Show full text of results",
    )

    # Add command
    add_parser = subparsers.add_parser("add", help="Add files to existing raglet")
    add_parser.add_argument(
        "--raglet",
        type=str,
        required=True,
        help="Path to existing raglet",
    )
    add_parser.add_argument(
        "files",
        nargs="+",
        help="Files to add",
    )
    add_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: same as --raglet)",
    )

    # Package command
    package_parser = subparsers.add_parser(
        "package",
        help="Convert raglet between formats (directory, zip, sqlite)",
    )
    package_parser.add_argument(
        "--raglet",
        type=str,
        required=True,
        help="Path to existing raglet (directory, .zip, or .sqlite)",
    )
    package_parser.add_argument(
        "--format",
        type=str,
        choices=["dir", "zip", "sqlite"],
        required=True,
        help="Target format (directory, zip, or sqlite)",
    )
    package_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: source.{format})",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize CLI output
    init_output(
        quiet=args.quiet,
        verbose=args.verbose,
        use_colors=None if not args.no_color else False,
    )

    # Warn about loading time (embedding models are heavy) - only if not quiet
    output = get_output()
    if not output.quiet:
        output.warning(
            "raglet is loading... Embedding models may take a few seconds to load on first use."
        )

    # Route to command handler
    if args.command == "build":
        return build_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "add":
        return add_command(args)
    elif args.command == "package":
        return package_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
