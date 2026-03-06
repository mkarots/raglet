"""Performance experiment to measure storage format performance.

Tests both SQLite and Directory formats:
- Load time (storage + FAISS index rebuild)
- Save time (full save)
- Incremental save time (adding chunks)
- Search time
- Various chunk counts (100, 1000, 10000, 50000)

Compares:
- SQLite format (.sqlite file)
- Directory format (.raglet/ directory)
"""

import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# CRITICAL: Import order matters for threading compatibility
# PyTorch (via sentence-transformers) must initialize BEFORE FAISS
# to prevent OpenMP threading conflicts. Import raglet at module level
# to ensure correct initialization order.
try:
    # Import raglet - this triggers sentence-transformers and faiss imports
    # in the correct order (PyTorch before FAISS)
    import raglet  # noqa: F401
    
    from raglet import RAGlet
    from raglet.core.chunk import Chunk
    from raglet.config.config import RAGletConfig
except ImportError:
    print("Warning: raglet not installed. This is a template script.")
    print("Install raglet or adjust imports based on your setup.")


def create_test_chunks(num_chunks: int) -> List[Chunk]:
    """Create test chunks with realistic content.
    
    Args:
        num_chunks: Number of chunks to create
        
    Returns:
        List of Chunk objects
    """
    chunks = []
    base_text = (
        "This is a test chunk with some realistic content. "
        "It contains multiple sentences to make it more representative of real data. "
        "The chunk includes various words and concepts that might appear in actual documents. "
        "This helps ensure our performance measurements are realistic."
    )
    
    for i in range(num_chunks):
        chunks.append(Chunk(
            text=f"{base_text} Chunk number {i}.",
            source=f"test_file_{i // 100}.txt",
            index=i,
            metadata={"chunk_id": i, "file_id": i // 100}
        ))
    
    return chunks


def measure_save_time(
    raglet: RAGlet, 
    file_path: str, 
    storage_backend: Optional[str] = None
) -> Dict[str, float]:
    """Measure time to save raglet to storage.
    
    Args:
        raglet: RAGlet instance to save
        file_path: Path to save file/directory
        storage_backend: Optional backend name ("sqlite" or "directory")
        
    Returns:
        Dictionary with timing metrics
    """
    start = time.perf_counter()
    
    # Use specific backend if provided
    if storage_backend == "sqlite":
        from raglet.storage.sqlite_backend import SQLiteStorageBackend
        backend = SQLiteStorageBackend()
        raglet.save(file_path, storage_backend=backend)
    elif storage_backend == "directory":
        from raglet.storage.directory_backend import DirectoryStorageBackend
        backend = DirectoryStorageBackend()
        raglet.save(file_path, storage_backend=backend)
    else:
        # Auto-detect
        raglet.save(file_path)
    
    save_time = time.perf_counter() - start
    
    # Calculate size (file or directory)
    path = Path(file_path)
    if path.is_file():
        file_size = path.stat().st_size
    elif path.is_dir():
        # Sum all files in directory
        file_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        file_size = 0
    
    return {
        "save_time_ms": save_time * 1000,
        "file_size_bytes": file_size,
        "chunk_count": len(raglet.chunks)
    }


def measure_load_time(
    file_path: str,
    storage_backend: Optional[str] = None
) -> Dict[str, float]:
    """Measure time to load raglet from storage and rebuild index.
    
    Args:
        file_path: Path to load file/directory
        storage_backend: Optional backend name ("sqlite" or "directory")
        
    Returns:
        Dictionary with timing metrics
    """
    start = time.perf_counter()
    
    # Use specific backend if provided
    if storage_backend == "sqlite":
        from raglet.storage.sqlite_backend import SQLiteStorageBackend
        backend = SQLiteStorageBackend()
        raglet = RAGlet.load(file_path, storage_backend=backend)
    elif storage_backend == "directory":
        from raglet.storage.directory_backend import DirectoryStorageBackend
        backend = DirectoryStorageBackend()
        raglet = RAGlet.load(file_path, storage_backend=backend)
    else:
        # Auto-detect
        raglet = RAGlet.load(file_path)
    
    load_time = time.perf_counter() - start
    
    return {
        "load_time_ms": load_time * 1000,
        "chunk_count": len(raglet.chunks),
        "embedding_dim": raglet.embedding_generator.get_dimension()
    }


def measure_incremental_save(
    raglet: RAGlet, 
    file_path: str, 
    new_chunks: List[Chunk],
) -> Dict[str, float]:
    """Measure time to incrementally add chunks.
    
    Args:
        raglet: RAGlet instance
        file_path: Path to save file/directory
        new_chunks: New chunks to add
        
    Returns:
        Dictionary with timing metrics
    """
    start = time.perf_counter()
    
    # Add chunks with automatic save (add_chunks handles incremental save internally)
    # Pass file_path to add_chunks so it saves automatically
    raglet.add_chunks(new_chunks, file_path=file_path)
    
    save_time = time.perf_counter() - start
    
    return {
        "incremental_save_time_ms": save_time * 1000,
        "new_chunks": len(new_chunks),
        "total_chunks": len(raglet.chunks)
    }


def measure_search_time(raglet: RAGlet, query: str, top_k: int = 5) -> Dict[str, float]:
    """Measure time to perform search.
    
    Args:
        raglet: RAGlet instance
        query: Search query
        top_k: Number of results
        
    Returns:
        Dictionary with timing metrics
    """
    print(f"        DEBUG: raglet has {len(raglet.chunks)} chunks")
    print(f"        DEBUG: embeddings shape: {raglet.embeddings.shape if len(raglet.embeddings) > 0 else 'empty'}")
    print(f"        DEBUG: vector_store has {raglet.vector_store.get_count()} vectors")
    
    start = time.perf_counter()
    print(f"        DEBUG: About to call raglet.search()...")
    results = raglet.search(query, top_k=top_k)
    print(f"        DEBUG: Search returned {len(results)} results")
    search_time = time.perf_counter() - start
    
    return {
        "search_time_ms": search_time * 1000,
        "results_count": len(results),
        "top_k": top_k
    }


def run_experiment(
    test_sizes: List[int], 
    formats: List[str] = ["sqlite", "directory"],
    output_file: str = "performance_results.json"
):
    """Run performance experiment with various chunk counts and formats.
    
    Args:
        test_sizes: List of chunk counts to test
        formats: List of formats to test ("sqlite", "directory", or both)
        output_file: Path to save results JSON
    """
    results = []
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Performance Experiment: Storage Format Comparison")
    print("=" * 80)
    print(f"Testing formats: {', '.join(formats)}")
    print(f"Testing sizes: {', '.join(str(s) for s in test_sizes)}")
    print()
    
    for size in test_sizes:
        print(f"{'='*80}")
        print(f"Testing with {size:,} chunks")
        print(f"{'='*80}")
        
        # Create test raglet once
        print(f"Creating raglet...")
        chunks = create_test_chunks(size)
        config = RAGletConfig()
        
        create_start = time.perf_counter()
        # Create empty RAGlet and add chunks properly
        raglet = RAGlet.from_files([], config=config)
        # Use add_chunks to properly add chunks with embeddings and vector store
        raglet.add_chunks(chunks)
        create_time = time.perf_counter() - create_start
        
        print(f"Created in {create_time*1000:.2f}ms\n")
        
        for format_name in formats:
            print(f"  Format: {format_name.upper()}")
            
            # Determine file path based on format
            if format_name == "sqlite":
                file_path = test_dir / f"test_{size}.sqlite"
            elif format_name == "directory":
                file_path = test_dir / f"test_{size}_dir"
            else:
                print(f"    ✗ Unknown format: {format_name}")
                continue
            
            # Clean up existing files
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                else:
                    shutil.rmtree(file_path)
            
            try:
                # Measure save
                print(f"    Saving...")
                save_result = measure_save_time(raglet, str(file_path), storage_backend=format_name)
                print(f"    Saved in {save_result['save_time_ms']:.2f}ms ({save_result['file_size_bytes']/1024/1024:.2f}MB)")
                
                # Measure load (includes index rebuild)
                print(f"    Loading (rebuilding index)...")
                load_result = measure_load_time(str(file_path), storage_backend=format_name)
                print(f"    Loaded in {load_result['load_time_ms']:.2f}ms")
                
                # Measure incremental save (add 10 chunks)
                print(f"    Incremental save (adding 10 chunks)...")
                new_chunks = create_test_chunks(10)
                # Reload to get fresh instance
                if format_name == "sqlite":
                    from raglet.storage.sqlite_backend import SQLiteStorageBackend
                    backend = SQLiteStorageBackend()
                    loaded_raglet = RAGlet.load(str(file_path), storage_backend=backend)
                elif format_name == "directory":
                    from raglet.storage.directory_backend import DirectoryStorageBackend
                    backend = DirectoryStorageBackend()
                    loaded_raglet = RAGlet.load(str(file_path), storage_backend=backend)
                else:
                    loaded_raglet = RAGlet.load(str(file_path))
                
                incremental_result = measure_incremental_save(
                    loaded_raglet, str(file_path), new_chunks
                )
                print(f"    Incremental save in {incremental_result['incremental_save_time_ms']:.2f}ms")
                
                # Measure search performance
                print(f"    Testing search...")
                try:
                    if format_name == "sqlite":
                        from raglet.storage.sqlite_backend import SQLiteStorageBackend
                        print(f"      Creating SQLiteStorageBackend instance...")
                        backend = SQLiteStorageBackend()
                        print(f"      SQLiteStorageBackend instance created")
                        print(f"      Loading raglet for search from {file_path}...")
                        print(f"      File size: {Path(file_path).stat().st_size / 1024 / 1024:.2f}MB")
                        print(f"      About to call RAGlet.load()...")
                        reloaded_raglet = RAGlet.load(str(file_path), storage_backend=backend)
                        print(f"      RAGlet.load() completed successfully")
                        print(f"      Loaded raglet: {len(reloaded_raglet.chunks)} chunks, {reloaded_raglet.embeddings.shape if len(reloaded_raglet.embeddings) > 0 else 'no embeddings'}")
                    elif format_name == "directory":
                        from raglet.storage.directory_backend import DirectoryStorageBackend
                        backend = DirectoryStorageBackend()
                        print(f"      Loading raglet for search...")
                        reloaded_raglet = RAGlet.load(str(file_path), storage_backend=backend)
                        print(f"      Loaded raglet: {len(reloaded_raglet.chunks)} chunks, {reloaded_raglet.embeddings.shape if len(reloaded_raglet.embeddings) > 0 else 'no embeddings'}")
                    else:
                        print(f"      Loading raglet for search...")
                        reloaded_raglet = RAGlet.load(str(file_path))
                        print(f"      Loaded raglet: {len(reloaded_raglet.chunks)} chunks, {reloaded_raglet.embeddings.shape if len(reloaded_raglet.embeddings) > 0 else 'no embeddings'}")
                    
                    print(f"      Performing search query...")
                    search_result = measure_search_time(reloaded_raglet, "test chunk content", top_k=5)
                    print(f"    Search in {search_result['search_time_ms']:.2f}ms")
                except Exception as e:
                    print(f"    ✗ Search failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create a dummy search result so we don't break the results structure
                    search_result = {
                        "search_time_ms": 0.0,
                        "results_count": 0,
                        "top_k": 5
                    }
                
                result = {
                    "chunk_count": size,
                    "format": format_name,
                    "create_time_ms": create_time * 1000,
                    **save_result,
                    **load_result,
                    **incremental_result,
                    **search_result
                }
                results.append(result)
                
                print(f"    ✓ Completed {format_name} format")
                print()
                
            except Exception as e:
                print(f"    ✗ Error testing {format_name}: {e}")
                import traceback
                traceback.print_exc()
                print()
                # Save partial results before continuing
                try:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"    Partial results saved to {output_file}")
                except Exception as save_error:
                    print(f"    Failed to save partial results: {save_error}")
                continue
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved successfully ({len(results)} entries)")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary tables
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    # Summary by format
    for format_name in formats:
        format_results = [r for r in results if r.get('format') == format_name]
        if not format_results:
            continue
            
        print(f"\n{format_name.upper()} Format:")
        print(f"{'Chunks':>10} | {'Save (ms)':>10} | {'Load (ms)':>10} | {'Incremental (ms)':>15} | {'Search (ms)':>10} | {'Size (MB)':>10}")
        print("-" * 80)
        for r in format_results:
            print(
                f"{r['chunk_count']:>10,} | "
                f"{r['save_time_ms']:>10.2f} | "
                f"{r['load_time_ms']:>10.2f} | "
                f"{r['incremental_save_time_ms']:>15.2f} | "
                f"{r['search_time_ms']:>10.2f} | "
                f"{r['file_size_bytes']/1024/1024:>10.2f}"
            )
    
    # Comparison table
    if len(formats) > 1:
        print(f"\n{'='*80}")
        print("Format Comparison (by chunk count)")
        print("=" * 80)
        
        for size in test_sizes:
            print(f"\n{size:,} chunks:")
            size_results = [r for r in results if r['chunk_count'] == size]
            
            print(f"{'Metric':<20} | {'SQLite':>15} | {'Directory':>15} | {'Difference':>15}")
            print("-" * 70)
            
            for metric in ['save_time_ms', 'load_time_ms', 'incremental_save_time_ms', 'search_time_ms', 'file_size_bytes']:
                sqlite_val = next((r[metric] for r in size_results if r.get('format') == 'sqlite'), None)
                dir_val = next((r[metric] for r in size_results if r.get('format') == 'directory'), None)
                
                if sqlite_val is not None and dir_val is not None:
                    diff = dir_val - sqlite_val
                    diff_pct = (diff / sqlite_val * 100) if sqlite_val > 0 else 0
                    
                    # Format values
                    if metric == 'file_size_bytes':
                        sqlite_str = f"{sqlite_val/1024/1024:.2f} MB"
                        dir_str = f"{dir_val/1024/1024:.2f} MB"
                        diff_str = f"{diff/1024/1024:+.2f} MB ({diff_pct:+.1f}%)"
                    else:
                        sqlite_str = f"{sqlite_val:.2f} ms"
                        dir_str = f"{dir_val:.2f} ms"
                        diff_str = f"{diff:+.2f} ms ({diff_pct:+.1f}%)"
                    
                    print(f"{metric:<20} | {sqlite_str:>15} | {dir_str:>15} | {diff_str:>15}")
    
    print()
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance test for raglet storage formats")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="Chunk counts to test (default: 100 1000 10000)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        choices=["sqlite", "directory"],
        default=["sqlite", "directory"],
        help="Formats to test (default: sqlite directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_results.json",
        help="Output JSON file (default: performance_results.json)"
    )
    
    args = parser.parse_args()
    
    try:
        run_experiment(args.sizes, args.formats, args.output)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error in experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
