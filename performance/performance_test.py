"""Benchmark raglet build (extract + chunk + embed + index).

Generates synthetic text calibrated to produce a target chunk count,
then times RAGlet.from_files() for each target.
"""

import json
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

print = partial(print, flush=True)  # type: ignore[assignment]

from raglet import RAGlet
from raglet.config.config import ChunkingConfig, EmbeddingConfig, RAGletConfig
from raglet.processing.chunker import SentenceAwareChunker

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Meanwhile the cat sat on the mat watching birds fly south.",
    "A gentle breeze carried the scent of wildflowers across the meadow.",
    "Dark clouds gathered on the horizon as evening approached slowly.",
    "The old lighthouse keeper trimmed the wick by candlelight.",
    "Waves crashed against the rocky shore sending spray into the air.",
    "Children laughed and played in the park until the sun went down.",
    "The train rumbled through the valley echoing off the canyon walls.",
    "Fresh snow covered the mountain peaks in a blanket of white.",
    "Fireflies danced above the tall grass on warm summer nights.",
]


def generate_test_files(
    target_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
    output_dir: Path,
) -> list[str]:
    """Generate files calibrated to produce approximately target_chunks."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    chunker = SentenceAwareChunker(ChunkingConfig(size=chunk_size, overlap=chunk_overlap))
    sample_text = "\n".join(SENTENCES * 100)
    sample_chunks = chunker.chunk(sample_text, metadata={"source": "calibration"})
    if not sample_chunks:
        raise RuntimeError("Calibration produced 0 chunks")

    bytes_per_chunk = len(sample_text.encode()) / len(sample_chunks)
    total_bytes = int(bytes_per_chunk * target_chunks * 1.05)

    max_file_bytes = 2 * 1024 * 1024
    file_paths: list[str] = []
    written = 0
    file_idx = 0
    si = 0

    while written < total_bytes:
        path = output_dir / f"doc_{file_idx:05d}.txt"
        file_paths.append(str(path))
        file_written = 0
        with open(path, "w") as f:
            while file_written < max_file_bytes and written < total_bytes:
                line = SENTENCES[si % len(SENTENCES)] + "\n"
                f.write(line)
                file_written += len(line)
                written += len(line)
                si += 1
        file_idx += 1

    return file_paths


def run(
    chunk_counts: list[int],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    device: Optional[str] = None,
    fp16: bool = False,
    output_file: str = "performance/performance_results.json",
    keep: Optional[str] = None,
) -> None:
    """Benchmark build across different chunk counts."""
    from raglet.config.config import _select_device

    resolved_device = device or _select_device()

    print(f"Device: {resolved_device} | fp16: {fp16} | chunk_size: {chunk_size} | overlap: {chunk_overlap}")
    print(f"Targets: {chunk_counts}\n")

    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    results = []

    for target in chunk_counts:
        print(f"--- {target:,} chunks ---")

        files_dir = test_dir / f"files_{target}"
        file_paths = generate_test_files(target, chunk_size, chunk_overlap, files_dir)
        text_mb = sum(Path(p).stat().st_size for p in file_paths) / (1024 * 1024)
        print(f"  {len(file_paths)} files, {text_mb:.1f} MB text")

        config = RAGletConfig(
            chunking=ChunkingConfig(size=chunk_size, overlap=chunk_overlap),
            embedding=EmbeddingConfig(device=resolved_device, use_fp16=fp16),
        )

        try:
            t0 = time.perf_counter()
            rag = RAGlet.from_files(file_paths, config=config)
            build_s = time.perf_counter() - t0
            actual = len(rag.chunks)
            chunks_per_s = actual / build_s if build_s > 0 else 0

            print(f"  {build_s:.1f}s  ({actual:,} chunks, {chunks_per_s:,.0f} chunks/s)")

            results.append({
                "target": target,
                "actual": actual,
                "text_mb": round(text_mb, 2),
                "build_s": round(build_s, 2),
                "chunks_per_s": round(chunks_per_s),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "device": resolved_device,
                "fp16": fp16,
            })

            if keep:
                save_path = str(Path(keep) / f"raglet_{actual}.sqlite")
                rag.save(save_path)
                print(f"  saved: {save_path}")

            rag.close()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            shutil.rmtree(files_dir, ignore_errors=True)

        print()

    if test_dir.exists() and not any(test_dir.iterdir()):
        test_dir.rmdir()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'Chunks':>10} | {'Text':>8} | {'Build':>10} | {'Throughput':>14}")
    print("-" * 52)
    for r in results:
        print(
            f"{r['actual']:>10,} | "
            f"{r['text_mb']:>6.1f}MB | "
            f"{r['build_s']:>8.1f}s | "
            f"{r['chunks_per_s']:>10,} chunks/s"
        )
    print(f"\nResults: {output_file}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Benchmark raglet build")
    p.add_argument("--chunks", type=int, nargs="+", default=[1000, 5000, 10000],
                   help="Target chunk counts (default: 1000 5000 10000)")
    p.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens (default: 512)")
    p.add_argument("--chunk-overlap", type=int, default=50, help="Overlap in tokens (default: 50)")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None, help="Inference device")
    p.add_argument("--fp16", action="store_true", help="Enable fp16 (MPS/CUDA only)")
    p.add_argument("--output", default="performance/performance_results.json", help="Output JSON path")
    p.add_argument("--keep", type=str, default=None, help="Directory to save built raglet files into")

    args = p.parse_args()

    try:
        run(args.chunks, args.chunk_size, args.chunk_overlap, args.device, args.fp16, args.output, args.keep)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
