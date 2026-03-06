#!/usr/bin/env python3
"""Minimal agent loop with raglet - the most ergonomic API.

This shows the simplest possible agentic loop:
1. Load/create memory
2. Search for context
3. Use with your LLM
4. Store conversation (batched saves)

Batching Strategy:
    Instead of saving every turn, we accumulate chat history and save when
    it exceeds a threshold (SAVE_THRESHOLD chars). This reduces I/O operations
    while still preserving data reasonably (only lose last N chars if crash).

Usage:
    python examples/minimal_agent_loop.py
"""

from pathlib import Path
from raglet import RAGlet


def your_llm(context: str, query: str) -> str:
    """Placeholder for your LLM integration.
    
    Replace this with your actual LLM call (OpenAI, Anthropic, etc.).
    """
    return f"[LLM Response for: {query}]"


# Configuration
RAGLET_PATH = ".raglet/"
SAVE_THRESHOLD = 1000  # Save when chat history exceeds N characters

# Load or create memory
rag = RAGlet.load(RAGLET_PATH) if Path(RAGLET_PATH).exists() else RAGlet.from_files(["docs/"])

# Track unsaved chat history
unsaved_chars = 0

# Agent loop
while True:
    query = input("You: ")
    if query == "exit":
        # Save remaining chat history before exit
        if unsaved_chars > 0:
            rag.save(RAGLET_PATH)
            print("[Saved remaining chat history]\n")
        break
    
    # Search → Build context → LLM → Store
    context = "\n\n".join([c.text for c in rag.search(query, top_k=5)])
    response = your_llm(context, query)
    
    # Add to memory (not saved yet - batched saves)
    rag.add_text(query, source="chat")
    rag.add_text(response, source="chat")
    
    # Track unsaved content
    unsaved_chars += len(query) + len(response)
    
    # Batch save when threshold reached
    if unsaved_chars >= SAVE_THRESHOLD:
        rag.save(RAGLET_PATH)
        unsaved_chars = 0
        print(f"[Saved {SAVE_THRESHOLD}+ chars to disk]\n")
    
    print(f"Agent: {response}\n")
