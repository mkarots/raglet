#!/usr/bin/env python3
"""Parse and display performance test results for GitHub Actions."""

import json
import sys
from pathlib import Path


def main():
    """Parse performance results and print formatted summary."""
    results_file = Path("performance_results.json")
    
    if not results_file.exists():
        print("No results file found.")
        sys.exit(0)
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if not results:
            print("No results found.")
            sys.exit(0)
        
        # Group by format
        formats = {}
        for r in results:
            fmt = r.get('format', 'unknown')
            if fmt not in formats:
                formats[fmt] = []
            formats[fmt].append(r)
        
        # Print summary table
        print(f"{'Chunks':>10} | {'Save (ms)':>12} | {'Load (ms)':>12} | {'Incremental (ms)':>18} | {'Search (ms)':>12} | {'Size (MB)':>12}")
        print("-" * 90)
        
        for fmt in sorted(formats.keys()):
            print(f"\n{fmt.upper()} Format:")
            for r in sorted(formats[fmt], key=lambda x: x['chunk_count']):
                print(
                    f"{r['chunk_count']:>10,} | "
                    f"{r['save_time_ms']:>12.2f} | "
                    f"{r['load_time_ms']:>12.2f} | "
                    f"{r['incremental_save_time_ms']:>18.2f} | "
                    f"{r['search_time_ms']:>12.2f} | "
                    f"{r['file_size_bytes']/1024/1024:>12.2f}"
                )
        
        # Comparison if multiple formats
        if len(formats) > 1:
            print("\n\nComparison:")
            sizes = sorted(set(r['chunk_count'] for r in results))
            for size in sizes:
                size_results = [r for r in results if r['chunk_count'] == size]
                sqlite = next((r for r in size_results if r.get('format') == 'sqlite'), None)
                directory = next((r for r in size_results if r.get('format') == 'directory'), None)
                
                if sqlite and directory:
                    print(f"\n{size:,} chunks:")
                    print(f"  Save:      SQLite {sqlite['save_time_ms']:.2f}ms vs Directory {directory['save_time_ms']:.2f}ms")
                    print(f"  Load:      SQLite {sqlite['load_time_ms']:.2f}ms vs Directory {directory['load_time_ms']:.2f}ms")
                    print(f"  Incremental: SQLite {sqlite['incremental_save_time_ms']:.2f}ms vs Directory {directory['incremental_save_time_ms']:.2f}ms")
                    print(f"  Search:    SQLite {sqlite['search_time_ms']:.2f}ms vs Directory {directory['search_time_ms']:.2f}ms")
                    print(f"  Size:      SQLite {sqlite['file_size_bytes']/1024/1024:.2f}MB vs Directory {directory['file_size_bytes']/1024/1024:.2f}MB")
    
    except Exception as e:
        print(f"Error parsing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
