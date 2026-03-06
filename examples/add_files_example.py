"""Example: Using add_files() to add new documents to existing RAGlet."""

import tempfile
from pathlib import Path

from raglet.core.rag import RAGlet


def main():
    """Demonstrate add_files() usage."""
    print("📚 Example: Adding files to existing RAGlet\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Create initial RAGlet with some files
        print("1️⃣ Creating initial RAGlet...")
        file1 = Path(tmpdir) / "doc1.txt"
        file1.write_text("Python is a programming language.\nIt's used for data science.")
        
        file2 = Path(tmpdir) / "doc2.txt"
        file2.write_text("RAG systems combine retrieval and generation.\nThey use embeddings for search.")
        
        raglet = RAGlet.from_files([str(file1), str(file2)])
        print(f"   ✓ Created with {len(raglet.chunks)} chunks")
        
        # Step 2: Save to memory
        print("\n2️⃣ Saving to memory...")
        memory_file = Path(tmpdir) / "memory.sqlite"
        raglet.save(str(memory_file))
        print("   ✓ Saved")
        
        # Step 3: Load memory later
        print("\n3️⃣ Loading memory...")
        raglet = RAGlet.load(str(memory_file))
        print(f"   ✓ Loaded {len(raglet.chunks)} chunks")
        
        # Step 4: Add new files
        print("\n4️⃣ Adding new files...")
        file3 = Path(tmpdir) / "doc3.txt"
        file3.write_text("Machine learning uses algorithms.\nDeep learning uses neural networks.")
        
        file4 = Path(tmpdir) / "doc4.txt"
        file4.write_text("Vector databases store embeddings.\nFAISS is a popular vector search library.")
        
        raglet.add_files([str(file3), str(file4)])
        print(f"   ✓ Added files, now have {len(raglet.chunks)} chunks")
        
        # Step 5: Search new content
        print("\n5️⃣ Searching new content...")
        results = raglet.search("neural networks")
        print(f"   ✓ Found {len(results)} results")
        if results:
            print(f"   Top result: {results[0].text[:60]}...")
        
        # Step 6: Save incrementally
        print("\n6️⃣ Saving incrementally...")
        raglet.save(str(memory_file), incremental=True)
        print("   ✓ Saved")
        
        # Step 7: Verify persistence
        print("\n7️⃣ Verifying persistence...")
        final = RAGlet.load(str(memory_file))
        print(f"   ✓ Final count: {len(final.chunks)} chunks")
        
        # Verify all files are represented
        sources = {chunk.source for chunk in final.chunks}
        print(f"   ✓ Sources: {len(sources)} files")
        assert str(file1) in sources
        assert str(file2) in sources
        assert str(file3) in sources
        assert str(file4) in sources
        
    print("\n✅ Example complete! add_files() works correctly.")


if __name__ == "__main__":
    main()
