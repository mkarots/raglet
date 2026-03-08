"""Example: Agentic loop with raglet memory.

This demonstrates how to use raglet in an agentic chat loop:
1. Load or create memory
2. Search existing memory
3. Generate response using Claude API
4. Store conversation
5. Save incrementally

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python examples/agentic_loop_example.py
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from raglet.core.rag import RAGlet


def simulate_llm(context_chunks, user_query, api_key: Optional[str] = None):
    """Generate LLM response using Claude API.
    
    Args:
        context_chunks: List of Chunk objects for context
        user_query: User's query
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
    
    Returns:
        LLM response string
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )
    
    # Get API key from parameter or environment
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Provide via api_key parameter or "
            "set ANTHROPIC_API_KEY environment variable."
        )
    
    # Build context from chunks
    context_text = "\n\n".join([
        f"[Source: {chunk.source}]\n{chunk.text}"
        for chunk in context_chunks[:5]  # Use top 5 chunks
    ])
    
    # Create client and generate response
    client = Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Based on the following context from the knowledge base, answer the user's question.

Context:
{context_text}

User Question: {user_query}

Provide a helpful answer based on the context provided."""
            }
        ]
    )
    
    return response.content[0].text


def agentic_loop_example(memory_path: Optional[str] = None):
    """Example agentic loop with raglet memory.
    
    Args:
        memory_path: Optional path to memory (directory or SQLite file).
                     Defaults to ".raglet/" directory (directory format).
    """
    # Setup: Create or load memory
    if memory_path:
        memory_file = Path(memory_path)
    else:
        memory_file = Path("memory")
    
    if memory_file.exists():
        print(f"📂 Loading existing memory from {memory_file}...")
        raglet = RAGlet.load(str(memory_file))
        print(f"   Loaded {len(raglet.chunks)} chunks from memory")
    else:
        print(f"🆕 Creating new memory at {memory_file}...")
        # Initialize with some knowledge
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "initial_knowledge.txt"
            init_file.write_text(
                "Python is a programming language.\n"
                "RAG stands for Retrieval-Augmented Generation.\n"
                "raglet is a portable knowledge base library."
            )
            raglet = RAGlet.from_files([str(init_file)])
            raglet.save(str(memory_file))
            print(f"   Created memory with {len(raglet.chunks)} initial chunks")
    
    # Chat loop
    print("\n💬 Starting chat loop (type 'exit' to quit):\n")
    
    conversation_count = 0
    while True:
        # Get user input
        user_query = input("You: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Saving and exiting...")
            raglet.save(str(memory_file), incremental=True)
            print(f"   Saved {len(raglet.chunks)} total chunks")
            break
        
        if not user_query:
            continue
        
        # Step 1: Search existing memory
        print("🔍 Searching memory...")
        relevant_chunks = raglet.search(user_query, top_k=3)
        print(f"   Found {len(relevant_chunks)} relevant chunks")
        
        # Step 2: Generate response using Claude API
        print("🤖 Generating response...")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        response = simulate_llm(relevant_chunks, user_query, api_key=api_key)
        print(f"Agent: {response}\n")
        
        # Step 3: Store conversation in memory
        from raglet.core.chunk import Chunk
        
        conversation_chunks = [
            Chunk(
                text=user_query,
                source="chat",
                index=conversation_count * 2,
                metadata={"type": "user_query", "session": "example"},
            ),
            Chunk(
                text=response,
                source="chat",
                index=conversation_count * 2 + 1,
                metadata={"type": "agent_response", "session": "example"},
            ),
        ]
        
        raglet.add_chunks(conversation_chunks)
        conversation_count += 1
        
        # Step 4: Save periodically (every 2 conversations)
        if conversation_count % 2 == 0:
            print("💾 Saving memory...")
            raglet.save(str(memory_file), incremental=True)
            print(f"   Memory now contains {len(raglet.chunks)} chunks\n")
    
    print(f"\n✅ Done! Memory saved to {memory_file}")


if __name__ == "__main__":
    agentic_loop_example()
