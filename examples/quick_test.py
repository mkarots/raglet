#!/usr/bin/env python3
"""Quick test: Create → Save → Load → Search → Add → Save"""

from pathlib import Path
import tempfile

from raglet.core.rag import RAGlet
from raglet.core.chunk import Chunk

# Create test file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write("Python is a programming language.\nRAG systems combine retrieval and generation.")
    test_file = f.name

try:
    # 1. Create RAGlet
    print("1. Creating RAGlet...")
    raglet = RAGlet.from_files([test_file])
    print(f"   ✓ Created with {len(raglet.chunks)} chunks")
    
    # 2. Save
    print("\n2. Saving to SQLite...")
    raglet.save("test_memory.sqlite")
    print("   ✓ Saved")
    
    # 3. Load
    print("\n3. Loading from SQLite...")
    loaded = RAGlet.load("test_memory.sqlite")
    print(f"   ✓ Loaded {len(loaded.chunks)} chunks")
    
    # 4. Search
    print("\n4. Searching...")
    results = loaded.search("programming")
    print(f"   ✓ Found {len(results)} results")
    print(f"   Top result: {results[0].text[:50]}...")
    
    # 5. Add files
    print("\n5. Adding new files...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Machine learning is a subset of AI.\nDeep learning uses neural networks.")
        new_file = f.name
    
    try:
        loaded.add_files([new_file])
        print(f"   ✓ Added file, total chunks: {len(loaded.chunks)}")
        
        # 6. Add chunks (simulate LLM conversation)
        print("\n6. Adding conversation chunks...")
        new_chunks = [
            Chunk(text="User: What is RAG?", source="chat", index=0),
            Chunk(text="Agent: RAG is Retrieval-Augmented Generation.", source="chat", index=1),
        ]
        loaded.add_chunks(new_chunks)
        print(f"   ✓ Added {len(new_chunks)} chunks")
    finally:
        Path(new_file).unlink()
    
    # 7. Save incrementally
    print("\n7. Saving incrementally...")
    loaded.save("test_memory.sqlite", incremental=True)
    print("   ✓ Saved")
    
    # 8. Verify
    print("\n8. Verifying...")
    final = RAGlet.load("test_memory.sqlite")
    print(f"   ✓ Final count: {len(final.chunks)} chunks")
    
    print("\n✅ All tests passed!")
    
finally:
    Path(test_file).unlink()
    if Path("test_memory.sqlite").exists():
        print(f"\n💾 Memory file saved: test_memory.sqlite")
        print("   (Delete it manually if you want)")
