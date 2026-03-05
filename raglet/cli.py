#!/usr/bin/env python3
"""raglet CLI - Command-line interface for raglet."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

from raglet import RAGlet, RAGletConfig
from raglet.core.chunk import Chunk


def build_command(args: argparse.Namespace) -> int:
    """Build raglet from inputs (files, directories, or glob patterns).

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not args.inputs:
        print("Error: No inputs provided. Specify files, directories, or glob patterns.", file=sys.stderr)
        return 1

    # Collect all input files
    all_files = []
    for input_path in args.inputs:
        path = Path(input_path)
        
        if path.is_file():
            # Individual files are always included
            all_files.append(str(path))
        elif path.is_dir():
            # For directories, find all files recursively
            # (extractors will handle what they can, fallback to TextExtractor)
            # Users can use glob patterns like "*.py" if they want to filter
            all_files.extend([str(f) for f in path.rglob("*") if f.is_file()])
        else:
            # Try glob pattern (glob pattern defines what to match)
            matches = glob.glob(input_path, recursive=True)
            if matches:
                all_files.extend([f for f in matches if Path(f).is_file()])
            else:
                print(f"Warning: Input not found: {input_path}", file=sys.stderr)

    # Filter out common ignores
    ignore_patterns = args.ignore.split(",") if args.ignore else [
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        ".raglet",
    ]
    filtered_files = []
    for file in all_files:
        if not any(pattern in file for pattern in ignore_patterns):
            filtered_files.append(file)

    if not filtered_files:
        print("Error: No files found to process.", file=sys.stderr)
        return 1

    # Determine output path and format
    if args.out:
        output_path = Path(args.out)
        # Auto-detect format from extension if not specified
        if args.format:
            format_type = args.format
        else:
            if output_path.suffix == ".zip":
                format_type = "zip"
            elif output_path.suffix in [".sqlite", ".db"]:
                format_type = "sqlite"
            elif output_path.is_dir() or output_path.suffix == "":
                format_type = "dir"
            else:
                format_type = "zip"  # default
    else:
        # Default output
        format_type = args.format or "zip"
        if format_type == "zip":
            output_path = Path("raglet-out.zip")
        elif format_type == "sqlite":
            output_path = Path("raglet-out.sqlite")
        else:
            output_path = Path("raglet-out")

    print(f"Found {len(filtered_files)} files to process...")
    print(f"Building raglet: {output_path} (format: {format_type})")

    try:
        # Create config
        config = RAGletConfig()
        if args.chunk_size:
            config.chunking.size = args.chunk_size
        if args.chunk_overlap:
            config.chunking.overlap = args.chunk_overlap
        if args.model:
            config.embedding.model = args.model

        # Limit files if specified
        files_to_process = filtered_files[: args.max_files] if args.max_files else filtered_files

        # Build RAGlet
        raglet = RAGlet.from_files(files_to_process, config=config)

        # Save
        raglet.save(str(output_path))

        print(f"✓ Raglet built: {len(raglet.chunks)} chunks")
        return 0

    except Exception as e:
        print(f"Error building raglet: {e}", file=sys.stderr)
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
    raglet_path = Path(args.raglet)
    if not raglet_path.exists():
        print(f"Error: Raglet not found at {raglet_path}.", file=sys.stderr)
        print("Create one with: raglet build <inputs> --out <path>", file=sys.stderr)
        return 1

    try:
        # Load RAGlet
        raglet = RAGlet.load(str(raglet_path))

        # Search
        results = raglet.search(args.q, top_k=args.top_k)

        if not results:
            print("No results found.")
            return 0

        # Print results
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            score = result.score if result.score is not None else 0.0
            print(f"{i}. [{score:.3f}] {result.source}:{result.index}")
            print(f"   {result.text[:200]}...")
            if args.show_full:
                print(f"   Full text: {result.text}\n")
            else:
                print()

        return 0

    except Exception as e:
        print(f"Error querying raglet: {e}", file=sys.stderr)
        return 1


def add_command(args: argparse.Namespace) -> int:
    """Add files to existing raglet.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    raglet_path = Path(args.raglet)
    if not raglet_path.exists():
        print(f"Error: Raglet not found at {raglet_path}.", file=sys.stderr)
        print("Create one with: raglet build <inputs> --out <path>", file=sys.stderr)
        return 1

    if not args.files:
        print("Error: No files specified to add.", file=sys.stderr)
        return 1

    # Resolve file paths
    files_to_add = []
    for file_arg in args.files:
        file_path = Path(file_arg)
        if file_path.exists() and file_path.is_file():
            files_to_add.append(str(file_path))
        else:
            print(f"Warning: File not found: {file_path}", file=sys.stderr)

    if not files_to_add:
        print("Error: No valid files to add.", file=sys.stderr)
        return 1

    try:
        # Load existing RAGlet
        raglet = RAGlet.load(str(raglet_path))

        # Add files
        raglet.add_files(files_to_add)

        # Determine output path
        output_path = Path(args.out) if args.out else raglet_path

        # Save incrementally
        raglet.save(str(output_path), incremental=True)

        print(f"✓ Added {len(files_to_add)} files to raglet")
        print(f"  Total chunks: {len(raglet.chunks)}")
        return 0

    except Exception as e:
        print(f"Error adding files: {e}", file=sys.stderr)
        return 1


def package_command(args: argparse.Namespace) -> int:
    """Convert raglet between formats (zip, sqlite, dir).

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    raglet_path = Path(args.raglet)
    if not raglet_path.exists():
        print(f"Error: Raglet not found at {raglet_path}.", file=sys.stderr)
        return 1

    if args.format not in ["zip", "sqlite"]:
        print(f"Error: Invalid format '{args.format}'. Use 'zip' or 'sqlite'.", file=sys.stderr)
        return 1

    # Determine output path
    if args.out:
        output_path = Path(args.out)
    else:
        # Default: same name with new extension
        if args.format == "zip":
            output_path = raglet_path.with_suffix(".zip")
        else:
            output_path = raglet_path.with_suffix(".sqlite")

    try:
        # Load RAGlet
        raglet = RAGlet.load(str(raglet_path))

        # Save in new format
        raglet.save(str(output_path))

        print(f"✓ Packaged raglet: {output_path} (format: {args.format})")
        return 0

    except Exception as e:
        print(f"Error packaging raglet: {e}", file=sys.stderr)
        return 1


def chat_command(args: argparse.Namespace) -> int:
    """Interactive agentic chat loop with raglet context.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    raglet_path = Path(args.raglet)

    # Setup: Create or load raglet
    if raglet_path.exists():
        print(f"📂 Loading raglet from {raglet_path}...")
        try:
            raglet = RAGlet.load(str(raglet_path))
            print(f"   Loaded {len(raglet.chunks)} chunks")
        except Exception as e:
            print(f"Error loading raglet: {e}", file=sys.stderr)
            return 1
    else:
        print(f"🆕 Creating new raglet at {raglet_path}...")
        config = RAGletConfig()
        raglet = RAGlet(chunks=[], config=config)
        raglet.save(str(raglet_path))
        print(f"   Created empty raglet")

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Anthropic API key required.", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key flag.", file=sys.stderr)
        return 1

    # Import Anthropic
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package not installed.", file=sys.stderr)
        print("Install with: pip install anthropic", file=sys.stderr)
        return 1

    client = Anthropic(api_key=api_key)

    # Chat loop
    print("\n💬 Starting chat loop (type 'exit' to quit):\n")

    conversation_count = 0
    while True:
        # Get user input
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Saving and exiting...")
            raglet.save(str(raglet_path), incremental=True)
            print(f"   Saved {len(raglet.chunks)} total chunks")
            return 0

        if user_query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Saving and exiting...")
            raglet.save(str(raglet_path), incremental=True)
            print(f"   Saved {len(raglet.chunks)} total chunks")
            break

        if not user_query:
            continue

        # Step 1: Search raglet for context
        print("🔍 Searching raglet...")
        relevant_chunks = raglet.search(user_query, top_k=args.top_k)
        print(f"   Found {len(relevant_chunks)} relevant chunks")

        # Step 2: Build context
        if relevant_chunks:
            context_text = "\n\n".join([
                f"[Source: {chunk.source}]\n{chunk.text}"
                for chunk in relevant_chunks
            ])
            prompt = f"""Based on the following context from the knowledge base, answer the user's question.

Context:
{context_text}

User Question: {user_query}

Provide a helpful answer based on the context provided."""
        else:
            prompt = f"""Answer the user's question.

User Question: {user_query}

Provide a helpful answer."""

        # Step 3: Generate response using Claude API
        print("🤖 Generating response...")
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            response_text = response.content[0].text
        except Exception as e:
            print(f"Error generating response: {e}", file=sys.stderr)
            continue

        print(f"Agent: {response_text}\n")

        # Step 4: Store conversation in raglet
        conversation_chunks = [
            Chunk(
                text=user_query,
                source="chat",
                index=conversation_count * 2,
                metadata={"type": "user_query", "session": "cli"},
            ),
            Chunk(
                text=response_text,
                source="chat",
                index=conversation_count * 2 + 1,
                metadata={"type": "agent_response", "session": "cli"},
            ),
        ]

        raglet.add_chunks(conversation_chunks)
        conversation_count += 1

        # Step 5: Save periodically (every 2 conversations)
        if conversation_count % 2 == 0:
            print("💾 Saving raglet...")
            raglet.save(str(raglet_path), incremental=True)
            print(f"   Raglet now contains {len(raglet.chunks)} chunks\n")

    print(f"\n✅ Done! Raglet saved to {raglet_path}")
    return 0


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Warn about loading time (embedding models are heavy)
    print(
        "⚠️  raglet is loading... Embedding models may take a few seconds to load on first use.",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="raglet - Portable memory for small text corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build raglet from files
  raglet build file1.txt file2.md --out my-raglet.zip
  raglet build ./docs --out my-raglet.sqlite --format sqlite
  raglet build "*.md" --out output.zip

  # Query raglet
  raglet query --raglet my-raglet.zip --q "python" --top-k 5

  # Package (convert format)
  raglet package --raglet ./my-raglet-dir --format zip --out my-raglet.zip

  # Add files incrementally
  raglet add --raglet my-raglet.zip file3.txt

  # Interactive chat
  raglet chat --raglet my-raglet.zip
        """,
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
        default=None,
        help="Output path (default: raglet-out.{format})",
    )
    build_parser.add_argument(
        "--format",
        type=str,
        choices=["zip", "sqlite", "dir"],
        default=None,
        help="Output format (auto-detected from --out extension if not specified)",
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
        "--raglet",
        type=str,
        required=True,
        help="Path to raglet (.zip, .sqlite, or directory)",
    )
    query_parser.add_argument(
        "--q",
        type=str,
        required=True,
        help="Search query",
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
    package_parser = subparsers.add_parser("package", help="Convert raglet between formats")
    package_parser.add_argument(
        "--raglet",
        type=str,
        required=True,
        help="Path to existing raglet",
    )
    package_parser.add_argument(
        "--format",
        type=str,
        choices=["zip", "sqlite"],
        required=True,
        help="Target format",
    )
    package_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: source.{format})",
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive agentic chat loop with raglet context",
    )
    chat_parser.add_argument(
        "--raglet",
        type=str,
        required=True,
        help="Path to raglet (creates if doesn't exist)",
    )
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to use (default: 5)",
    )
    chat_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    chat_parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command handler
    if args.command == "build":
        return build_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "add":
        return add_command(args)
    elif args.command == "package":
        return package_command(args)
    elif args.command == "chat":
        return chat_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
