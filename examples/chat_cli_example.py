#!/usr/bin/env python3
"""Example: Interactive agentic chat loop with raglet.

This demonstrates how to use raglet in an agentic loop pattern:
1. Search raglet for relevant context
2. Use external LLM (Claude) to generate response
3. Store conversation in raglet for future retrieval

This is a demonstration script showing the "bring your own LLM" pattern.
raglet itself is retrieval-only - this script shows one integration example.

Usage:
    # Install optional dependencies
    pip install raglet[chat]

    # With environment variable (works with directory, zip, or sqlite)
    export ANTHROPIC_API_KEY="your-api-key"
    python examples/chat_cli_example.py --raglet .raglet/
    python examples/chat_cli_example.py --raglet my-raglet.zip
    python examples/chat_cli_example.py --raglet knowledge.sqlite

    # With API key flag
    python examples/chat_cli_example.py --raglet .raglet/ --api-key sk-...

    # Docker example
    docker run -it -v $(pwd):/data -e ANTHROPIC_API_KEY=sk-... raglet python examples/chat_cli_example.py --raglet /data/.raglet/
"""

import argparse
import os
import sys
from pathlib import Path

from raglet import RAGlet, RAGletConfig


def main() -> int:
    """Main entry point for chat example."""
    parser = argparse.ArgumentParser(
        description="Interactive agentic chat loop demonstration (requires external LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raglet",
        type=str,
        required=True,
        help="Path to raglet (creates if doesn't exist)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to use (default: 5)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6)",
    )

    args = parser.parse_args()

    raglet_path = Path(args.raglet)

    # Setup: Create or load raglet
    if raglet_path.exists():
        print(f"Loading raglet from {raglet_path}...")
        try:
            raglet = RAGlet.load(str(raglet_path))
            print(f"Loaded {len(raglet.chunks)} chunks")
        except Exception as e:
            print(f"Error loading raglet: {e}", file=sys.stderr)
            return 1
    else:
        print(f"Creating new raglet at {raglet_path}...")
        raglet = RAGlet(chunks=[], config=RAGletConfig())
        raglet.save(str(raglet_path))
        print("Created empty raglet")

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Anthropic API key required.", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY environment variable or use --api-key flag.", file=sys.stderr)
        print("Install anthropic package: pip install anthropic", file=sys.stderr)
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
    print("\nStarting chat loop (type 'exit' to quit)\n")

    conversation_count = 0
    while True:
        # Get user input
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving and exiting...")
            backend = raglet._get_default_backend(str(raglet_path))
            if backend.supports_incremental():
                raglet.save(str(raglet_path), incremental=True)
            else:
                raglet.save(str(raglet_path), incremental=False)
            print(f"Saved {len(raglet.chunks)} total chunks")
            return 0

        if user_query.lower() in ["exit", "quit", "q"]:
            print("\nSaving and exiting...")
            backend = raglet._get_default_backend(str(raglet_path))
            if backend.supports_incremental():
                raglet.save(str(raglet_path), incremental=True)
            else:
                raglet.save(str(raglet_path), incremental=False)
            print(f"Saved {len(raglet.chunks)} total chunks")
            break

        if not user_query:
            continue

        # Step 1: Search raglet for context
        relevant_chunks = raglet.search(user_query, top_k=args.top_k)

        # Step 2: Build context
        if relevant_chunks:
            context_text = "\n\n".join(
                [f"[Source: {chunk.source}]\n{chunk.text}" for chunk in relevant_chunks]
            )
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

        print(f"\nAgent: {response_text}\n")

        # Step 4: Store conversation in raglet
        raglet.add_text(
            text=user_query,
            source="chat",
            metadata={"type": "user_query", "session": "cli"},
        )
        raglet.add_text(
            text=response_text,
            source="chat",
            metadata={"type": "agent_response", "session": "cli"},
        )
        conversation_count += 1

        # Step 5: Save periodically (every 5 conversations)
        if conversation_count % 5 == 0:
            backend = raglet._get_default_backend(str(raglet_path))
            if backend.supports_incremental():
                raglet.save(str(raglet_path), incremental=True)
            else:
                raglet.save(str(raglet_path), incremental=False)
            print(f"Saved raglet ({len(raglet.chunks)} chunks)\n")

    print(f"\nDone! Raglet saved to {raglet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
