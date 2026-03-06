"""Simple test script to verify raglet save/load works."""

import tempfile
from pathlib import Path

from raglet.core.rag import RAGlet


def main():
    """Simple test of raglet persistence."""
    print("🧪 Testing raglet save/load...\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Create RAGlet from files
        print("1️⃣ Creating RAGlet from files...")
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text(
            "Python is a programming language.\n"
            "Machine learning uses algorithms.\n"
            "RAG systems combine retrieval and generation."
        )
        
        raglet1 = RAGlet.from_files([str(test_file)])
        print(f"   ✓ Created with {len(raglet1.chunks)} chunks")
        print(f"   ✓ Embeddings shape: {raglet1.embeddings.shape}")
        
        # Step 2: Save to SQLite
        print("\n2️⃣ Saving to SQLite...")
        db_path = Path(tmpdir) / "test.sqlite"
        raglet1.save(str(db_path))
        print(f"   ✓ Saved to {db_path}")
        
        # Step 3: Load from SQLite
        print("\n3️⃣ Loading from SQLite...")
        raglet2 = RAGlet.load(str(db_path))
        print(f"   ✓ Loaded {len(raglet2.chunks)} chunks")
        print(f"   ✓ Embeddings shape: {raglet2.embeddings.shape}")
        
        # Step 4: Verify data integrity
        print("\n4️⃣ Verifying data integrity...")
        assert len(raglet2.chunks) == len(raglet1.chunks), "Chunk count mismatch!"
        assert raglet2.embeddings.shape == raglet1.embeddings.shape, "Embedding shape mismatch!"
        print("   ✓ Chunk count matches")
        print("   ✓ Embedding shape matches")
        
        # Step 5: Test search
        print("\n5️⃣ Testing search functionality...")
        results = raglet2.search("programming language")
        print(f"   ✓ Found {len(results)} results")
        assert len(results) > 0, "No search results!"
        print(f"   ✓ Top result: {results[0].text[:50]}...")
        
        # Step 6: Test incremental updates
        print("\n6️⃣ Testing incremental updates...")
        from raglet.core.chunk import Chunk
        
        new_chunks = [
            Chunk(text="New information added.", source="test", index=0)
        ]
        raglet2.add_chunks(new_chunks)
        print(f"   ✓ Added {len(new_chunks)} new chunks")
        print(f"   ✓ Total chunks: {len(raglet2.chunks)}")
        
        # Step 7: Save incrementally
        print("\n7️⃣ Saving incrementally...")
        raglet2.save(str(db_path), incremental=True)
        print("   ✓ Saved incrementally")
        
        # Step 8: Load again and verify
        print("\n8️⃣ Loading again to verify...")
        raglet3 = RAGlet.load(str(db_path))
        print(f"   ✓ Loaded {len(raglet3.chunks)} chunks")
        assert len(raglet3.chunks) == len(raglet2.chunks), "Incremental save failed!"
        print("   ✓ Incremental save verified")
        
        print("\n✅ All tests passed! raglet persistence works correctly.")


if __name__ == "__main__":
    main()
