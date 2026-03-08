"""Simple performance test for raglet operations.

Tests performance based on total text content size (KB to MB).

Tests:
- from_files (create from files)
- search
- save
- load
- cleanup
"""

import sys
import time
import json
import shutil
from pathlib import Path
from typing import List

import raglet  # noqa: F401
from raglet import RAGlet
from raglet.config.config import RAGletConfig


def create_test_files(target_size_mb: float, output_dir: Path) -> List[str]:
    """Create test files with approximately target_size_mb of text content.
    
    Args:
        target_size_mb: Target total file size in MB
        output_dir: Directory to write files to
        
    Returns:
        List of file paths created
    """
    # Clean up directory first
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Realistic paragraph text (~500 bytes)
    paragraph = (
        "This is a test paragraph with realistic content. "
        "It contains multiple sentences to make it more representative of real data. "
        "The paragraph includes various words and concepts that might appear in actual documents. "
        "This helps ensure our performance measurements are realistic and meaningful. "
        "We want to test how raglet handles different amounts of text content from kilobytes to megabytes."
    ) * 2  # ~1000 bytes per paragraph
    
    target_size_bytes = int(target_size_mb * 1024 * 1024)
    file_paths = []
    current_size = 0
    file_num = 0
    
    while current_size < target_size_bytes:
        file_path = output_dir / f"test_file_{file_num:05d}.txt"
        file_paths.append(str(file_path))
        
        with open(file_path, "w") as f:
            # Write paragraphs until we're close to target
            while current_size < target_size_bytes:
                f.write(paragraph + "\n\n")
                current_size += len(paragraph) + 2  # +2 for newlines
                
                # Stop if this file would exceed reasonable file size (10MB per file max)
                if current_size - (target_size_bytes - len(paragraph) - 2) > 10 * 1024 * 1024:
                    break
        
        file_num += 1
        
        # If we've written enough, break
        if current_size >= target_size_bytes:
            break
    
    actual_size_mb = current_size / (1024 * 1024)
    return file_paths, actual_size_mb


def run_experiment(
    test_sizes_mb: List[float],
    formats: List[str] = ["sqlite", "directory"],
    output_file: str = "performance_results.json",
):
    """Run simple performance experiment based on text content size.
    
    Args:
        test_sizes_mb: List of text sizes to test in MB (e.g., [0.1, 1.0, 10.0])
        formats: List of formats to test ("sqlite", "directory")
        output_file: Path to save results JSON
    """
    results = []
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Performance Test: from_files → search → save → load")
    print("=" * 80)
    print(f"Formats: {', '.join(formats)}")
    print(f"Text sizes: {', '.join(f'{s:.2f}MB' for s in test_sizes_mb)}\n")
    
    for size_mb in test_sizes_mb:
        print(f"{'='*80}")
        print(f"Testing {size_mb:.2f}MB of text content")
        print(f"{'='*80}\n")
        
        # Create test files with target size
        test_files_dir = test_dir / f"test_files_{size_mb}mb"
        file_paths, actual_size_mb = create_test_files(size_mb, test_files_dir)
        
        print(f"Created {len(file_paths)} test files ({actual_size_mb:.2f}MB total)\n")
        
        for format_name in formats:
            print(f"  Format: {format_name.upper()}")
            
            # Determine output path
            if format_name == "sqlite":
                output_path = test_dir / f"test_{size_mb}mb.sqlite"
            elif format_name == "directory":
                output_path = test_dir / f"test_{size_mb}mb_dir"
            else:
                print(f"    ✗ Unknown format: {format_name}")
                continue
            
            # Clean up existing files
            if output_path.exists():
                if output_path.is_file():
                    output_path.unlink()
                else:
                    shutil.rmtree(output_path)
            
            try:
                # 1. from_files
                print(f"    1. from_files...", flush=True)
                start = time.perf_counter()
                config = RAGletConfig()
                
                try:
                    raglet = RAGlet.from_files(file_paths, config=config)
                except Exception as e:
                    print(f"       ✗ ERROR during from_files: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise
                
                from_files_time = time.perf_counter() - start
                print(f"       ✓ {from_files_time*1000:.2f}ms", flush=True)
                
                # 2. search
                print(f"    2. search...")
                start = time.perf_counter()
                results_search = raglet.search("test chunk content", top_k=5)
                search_time = time.perf_counter() - start
                print(f"       ✓ {search_time*1000:.2f}ms ({len(results_search)} results)")
                
                # 3. save
                print(f"    3. save...")
                start = time.perf_counter()
                raglet.save(str(output_path))
                save_time = time.perf_counter() - start
                
                # Calculate file size
                if output_path.is_file():
                    file_size = output_path.stat().st_size
                else:
                    file_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
                
                print(f"       ✓ {save_time*1000:.2f}ms ({file_size/1024/1024:.2f}MB)")
                
                # 4. load
                print(f"    4. load...")
                start = time.perf_counter()
                loaded_raglet = RAGlet.load(str(output_path))
                load_time = time.perf_counter() - start
                print(f"       ✓ {load_time*1000:.2f}ms")
                
                # Capture chunk count before cleanup
                chunk_count = len(raglet.chunks)
                
                # 5. cleanup
                print(f"    5. cleanup...")
                raglet.close()
                loaded_raglet.close()
                del raglet
                del loaded_raglet
                print(f"       ✓ Done\n")
                
                result = {
                    "text_size_mb": actual_size_mb,
                    "chunk_count": chunk_count,  # Actual chunks created
                    "format": format_name,
                    "from_files_time_ms": from_files_time * 1000,
                    "search_time_ms": search_time * 1000,
                    "save_time_ms": save_time * 1000,
                    "load_time_ms": load_time * 1000,
                    "file_size_bytes": file_size,
                }
                results.append(result)
                
            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Clean up test files
        shutil.rmtree(test_files_dir)
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved {len(results)} results\n")
    
    # Print summary
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    for format_name in formats:
        format_results = [r for r in results if r.get('format') == format_name]
        if not format_results:
            continue
        
        print(f"\n{format_name.upper()}:")
        print(f"{'Text (MB)':>10} | {'Chunks':>10} | {'from_files':>12} | {'search':>10} | {'save':>10} | {'load':>10} | {'Size (MB)':>12}")
        print("-" * 90)
        for r in format_results:
            print(
                f"{r['text_size_mb']:>10.2f} | "
                f"{r['chunk_count']:>10,} | "
                f"{r['from_files_time_ms']:>12.2f} | "
                f"{r['search_time_ms']:>10.2f} | "
                f"{r['save_time_ms']:>10.2f} | "
                f"{r['load_time_ms']:>10.2f} | "
                f"{r['file_size_bytes']/1024/1024:>12.2f}"
            )
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple performance test for raglet")
    parser.add_argument(
        "--sizes",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="Text sizes to test in MB (default: 0.1 1.0 10.0)"
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
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
