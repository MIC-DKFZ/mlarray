#!/usr/bin/env python
"""
Benchmark random patch read throughput across array formats.

Stage 1: Convert a NIfTI/NRRD dataset to npy, npz, zarr, hdf5, blosc2, mlarray.
Stage 2: Benchmark N random patch reads (open + read + close) per format.

The dataset_dir must contain a 'source' subdirectory with NIfTI/NRRD files.
Converted formats are written as siblings of 'source' inside dataset_dir.

Usage:
    python bench/benchmark_formats.py /path/to/dataset_dir \\
        --n_reads 1000 \\
        --patch_size 64 64 8 \\
        --seed 42

    # Skip conversion if already done:
    python bench/benchmark_formats.py /path/to/dataset_dir --skip_convert

    # Benchmark only specific formats:
    python bench/benchmark_formats.py /path/to/dataset_dir \\
        --formats mlarray blosc2 hdf5
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import blosc2
import h5py
import nibabel as nib
import numpy as np
import zarr
from rich.console import Console
from rich.table import Table
from rich import box
from tqdm import tqdm
from tqdmp import tqdmp

from mlarray import MLArray

ALL_FORMATS = ["npy", "npz", "zarr", "hdf5", "blosc2", "mlarray"]


# ---------------------------------------------------------------------------
# Source file loading
# ---------------------------------------------------------------------------

def load_source_array(filepath: Path) -> np.ndarray:
    """Load a NIfTI/NRRD file as a contiguous float32 numpy array."""
    img = nib.load(str(filepath))
    arr = np.asarray(img.dataobj)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def discover_source_files(source_dir: Path) -> list[Path]:
    files = []
    for ext in ("**/*.nii.gz", "**/*.nii", "**/*.nrrd"):
        files.extend(source_dir.glob(ext))
    return sorted(files)


def flat_stem(src: Path, source_dir: Path) -> str:
    """Unique flat filename stem: relative path with separators replaced by '__'."""
    rel = src.relative_to(source_dir)
    parts = list(rel.parts)
    # Strip all NIfTI/NRRD suffixes from the last part
    name = parts[-1]
    for suffix in (".nii.gz", ".nii", ".nrrd"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts[-1] = name
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Stage 1: Conversion
# ---------------------------------------------------------------------------

def convert_npy(arr: np.ndarray, out_path: Path) -> None:
    np.save(out_path, arr)


def convert_npz(arr: np.ndarray, out_path: Path) -> None:
    np.savez_compressed(out_path, data=arr)


def convert_zarr(arr: np.ndarray, out_path: Path) -> None:
    """Zarr v3 with auto chunk/codec selection (BytesCodec + ZstdCodec at level 0)."""
    if out_path.exists():
        shutil.rmtree(out_path)
    z = zarr.open(str(out_path), mode="w", shape=arr.shape, dtype=arr.dtype)
    z[:] = arr


def convert_hdf5(arr: np.ndarray, out_path: Path) -> None:
    """HDF5 with h5py auto chunking and gzip compression (level 4)."""
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=arr, chunks=True, compression="gzip", compression_opts=4)


def convert_blosc2(arr: np.ndarray, out_path: Path) -> None:
    """Blosc2 NDArray with fully default chunk/block/codec settings."""
    if out_path.exists():
        out_path.unlink()
    blosc2.asarray(arr, urlpath=str(out_path), mode="w")


def convert_mlarray(arr: np.ndarray, out_path: Path) -> None:
    """MLArray with patch-size optimized chunk/block sizes via comp_blosc2_params."""
    if out_path.exists():
        out_path.unlink()
    mla = MLArray.asarray(arr)
    mla.save(out_path)


CONVERTERS = {
    "npy": (convert_npy, ".npy"),
    "npz": (convert_npz, ".npz"),
    "zarr": (convert_zarr, ".zarr"),
    "hdf5": (convert_hdf5, ".h5"),
    "blosc2": (convert_blosc2, ".b2nd"),
    "mlarray": (convert_mlarray, ".mla"),
}


def _convert_one(src: Path, output_dir: Path, source_dir: Path, formats: list[str]) -> None:
    """Convert a single source file to all requested formats. Top-level for pickling."""
    arr = load_source_array(src)
    stem = flat_stem(src, source_dir)
    for fmt in formats:
        converter, suffix = CONVERTERS[fmt]
        out_path = output_dir / fmt / (stem + suffix)
        if out_path.exists():
            continue
        converter(arr, out_path)


def run_conversion(
    source_files: list[Path],
    source_dir: Path,
    output_dir: Path,
    formats: list[str],
    num_workers: int,
) -> None:
    for fmt in formats:
        (output_dir / fmt).mkdir(parents=True, exist_ok=True)

    fn = partial(_convert_one, output_dir=output_dir, source_dir=source_dir, formats=formats)
    tqdmp(fn, source_files, num_workers, desc="  Converting")


# ---------------------------------------------------------------------------
# Stage 2: Benchmark readers
# ---------------------------------------------------------------------------

def slices_for(starts: tuple, patch_size: tuple) -> tuple:
    return tuple(slice(s, s + p) for s, p in zip(starts, patch_size))


def random_starts(shape: tuple, patch_size: tuple, rng: random.Random) -> tuple:
    """Return per-axis start indices such that patch fits within shape."""
    return tuple(
        rng.randint(0, max(sh - ps, 0))
        for sh, ps in zip(shape, patch_size)
    )


def effective_patch(shape: tuple, patch_size: tuple) -> tuple:
    """Clamp patch size to image dimensions."""
    return tuple(min(ps, sh) for ps, sh in zip(patch_size, shape))


def read_patch_npy(filepath: Path, sl: tuple) -> np.ndarray:
    arr = np.load(filepath, mmap_mode="r")
    return arr[sl].copy()


def read_patch_npz(filepath: Path, sl: tuple) -> np.ndarray:
    data = np.load(filepath)["data"]
    return data[sl].copy()


def read_patch_zarr(filepath: Path, sl: tuple) -> np.ndarray:
    z = zarr.open(str(filepath), mode="r")
    return np.asarray(z[sl])


def read_patch_hdf5(filepath: Path, sl: tuple) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        return f["data"][sl]


def read_patch_blosc2(filepath: Path, sl: tuple) -> np.ndarray:
    arr = blosc2.open(str(filepath), mode="r", mmap_mode="r", dparams={"nthreads": 1})
    patch = arr[sl]
    del arr
    return patch


def read_patch_mlarray(filepath: Path, sl: tuple) -> np.ndarray:
    arr = MLArray.open(str(filepath))
    patch = arr[sl]
    arr.close()
    return patch


READERS = {
    "npy": read_patch_npy,
    "npz": read_patch_npz,
    "zarr": read_patch_zarr,
    "hdf5": read_patch_hdf5,
    "blosc2": read_patch_blosc2,
    "mlarray": read_patch_mlarray,
}


def collect_files(output_dir: Path, fmt: str) -> list[Path]:
    _, suffix = CONVERTERS[fmt]
    fmt_dir = output_dir / fmt
    return sorted(fmt_dir.glob(f"*{suffix}"))


def get_shape(filepath: Path, fmt: str) -> tuple:
    if fmt == "npy":
        arr = np.load(filepath, mmap_mode="r")
        return arr.shape
    if fmt == "npz":
        data = np.load(filepath)
        return data["data"].shape
    if fmt == "zarr":
        z = zarr.open(str(filepath), mode="r")
        return z.shape
    if fmt == "hdf5":
        with h5py.File(filepath, "r") as f:
            return f["data"].shape
    if fmt == "blosc2":
        arr = blosc2.open(str(filepath), mode="r", mmap_mode="r")
        shape = arr.shape
        del arr
        return shape
    if fmt == "mlarray":
        arr = MLArray.open(str(filepath))
        shape = arr.shape
        arr.close()
        return shape
    raise ValueError(fmt)


def disk_size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6


def total_disk_mb(output_dir: Path, fmt: str) -> float:
    fmt_dir = output_dir / fmt
    return sum(disk_size_mb(p) for p in fmt_dir.iterdir())


def run_benchmark(
    output_dir: Path,
    formats: list[str],
    n_reads: int,
    patch_size: tuple,
    rng: random.Random,
) -> dict:
    results = {}

    for fmt in formats:
        reader = READERS[fmt]
        files = collect_files(output_dir, fmt)
        if not files:
            print(f"  [{fmt}] No converted files found, skipping.")
            continue

        print(f"  [{fmt}] Loading shapes ...", end=" ", flush=True)
        file_shapes = [(f, get_shape(f, fmt)) for f in files]
        print("done")

        times_ms = []
        patch_bytes = None

        with tqdm(total=n_reads, desc=f"  [{fmt}]", unit="read", leave=True, file=sys.stdout) as pbar:
            for _ in range(n_reads):
                filepath, shape = rng.choice(file_shapes)
                ep = effective_patch(shape, patch_size)
                starts = random_starts(shape, ep, rng)
                sl = slices_for(starts, ep)

                if patch_bytes is None:
                    patch_bytes = math.prod(ep) * 4  # float32

                t0 = time.perf_counter()
                _ = reader(filepath, sl)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
                pbar.update(1)

        times_arr = np.array(times_ms)
        throughput_mbs = (patch_bytes / 1e6) / (np.mean(times_arr) / 1000.0)

        results[fmt] = {
            "n_reads": n_reads,
            "patch_bytes": patch_bytes,
            "disk_mb": round(total_disk_mb(output_dir, fmt), 2),
            "mean_ms": round(float(np.mean(times_arr)), 3),
            "median_ms": round(float(np.median(times_arr)), 3),
            "std_ms": round(float(np.std(times_arr)), 3),
            "min_ms": round(float(np.min(times_arr)), 3),
            "max_ms": round(float(np.max(times_arr)), 3),
            "p95_ms": round(float(np.percentile(times_arr, 95)), 3),
            "throughput_mbs": round(throughput_mbs, 2),
        }

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results_table(results: dict, source_disk_mb: float) -> None:
    best_mean   = min(r["mean_ms"]        for r in results.values())
    best_mb     = min(r["disk_mb"]        for r in results.values())
    best_tput   = max(r["throughput_mbs"] for r in results.values())

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="[bold]Random Patch Read Benchmark[/bold]",
        title_style="bold white",
        caption=(
            f"Ratio = {source_disk_mb:.1f} MB uncompressed / format disk size  •  "
            "Times include open + decompress + close per read  •  single process"
        ),
        caption_style="dim",
    )

    table.add_column("Format",      style="bold",    justify="left",  no_wrap=True)
    table.add_column("Disk (MB)",                    justify="right")
    table.add_column("Ratio",                        justify="right")
    table.add_column("Mean (ms)",                    justify="right")
    table.add_column("Median (ms)",                  justify="right")
    table.add_column("Std (ms)",                     justify="right")
    table.add_column("P95 (ms)",                     justify="right")
    table.add_column("Max (ms)",                     justify="right")
    table.add_column("MB/s",        style="bold",    justify="right")

    for fmt, r in results.items():
        ratio = source_disk_mb / r["disk_mb"] if r["disk_mb"] > 0 else float("nan")

        disk_str = f"[green]{r['disk_mb']:.1f}[/green]" if r["disk_mb"] == best_mb   else f"{r['disk_mb']:.1f}"
        mean_str = f"[green]{r['mean_ms']:.2f}[/green]" if r["mean_ms"] == best_mean else f"{r['mean_ms']:.2f}"
        tput_str = f"[green]{r['throughput_mbs']:.1f}[/green]" if r["throughput_mbs"] == best_tput else f"{r['throughput_mbs']:.1f}"

        table.add_row(
            fmt,
            disk_str,
            f"{ratio:.2f}×",
            mean_str,
            f"{r['median_ms']:.2f}",
            f"{r['std_ms']:.2f}",
            f"{r['p95_ms']:.2f}",
            f"{r['max_ms']:.2f}",
            tput_str,
        )

    console = Console()
    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark random patch read throughput across array formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset root directory. Must contain a 'source' subdirectory with NIfTI/NRRD files. Converted formats are written as siblings of 'source'.",
    )
    parser.add_argument("--n_reads", type=int, default=1000, help="Number of random patch reads per format. Default: 1000.")
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=[64, 64, 8],
        help="Patch size per axis (must match array ndim). Default: 64 64 8.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=ALL_FORMATS,
        default=ALL_FORMATS,
        help=f"Formats to benchmark. Default: all ({', '.join(ALL_FORMATS)}).",
    )
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to use. The full list is shuffled then sliced. Default: all images.")
    parser.add_argument("--skip_convert", action="store_true", help="Skip Stage 1 (conversion).")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Parallel workers for Stage 1 conversion. Default: all CPU cores.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    source_dir = dataset_dir / "source"
    if not source_dir.is_dir():
        raise RuntimeError(f"Expected a 'source' subdirectory in {dataset_dir}")
    output_dir = dataset_dir
    patch_size = tuple(args.patch_size)
    rng = random.Random(args.seed)

    print(f"Dataset dir: {dataset_dir}")
    print(f"Source dir : {source_dir}")
    print(f"Formats    : {args.formats}")
    print(f"Patch size : {patch_size}")
    print(f"N reads    : {args.n_reads}")
    print(f"Max images : {args.max_images if args.max_images is not None else 'all'}")
    print(f"Num workers: {args.num_workers}")
    print(f"Seed       : {args.seed}")

    all_source_files = discover_source_files(source_dir)
    if not all_source_files:
        raise RuntimeError(f"No NIfTI/NRRD files found under {source_dir}")
    print(f"\nFound {len(all_source_files)} source file(s).")

    rng.shuffle(all_source_files)
    source_files = all_source_files[: args.max_images] if args.max_images is not None else all_source_files
    if len(source_files) < len(all_source_files):
        print(f"Using {len(source_files)} image(s) (--max_images={args.max_images}).")

    source_disk_mb = sum(load_source_array(f).nbytes / 1e6 for f in source_files)
    print(f"Total uncompressed data: {source_disk_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Stage 1: Conversion
    # ------------------------------------------------------------------
    if not args.skip_convert:
        print("\n=== Stage 1: Converting ===")
        run_conversion(source_files, source_dir, output_dir, args.formats, args.num_workers)
        print("Conversion complete.")
    else:
        print("\nStage 1 skipped (--skip_convert).")

    # ------------------------------------------------------------------
    # Stage 2: Benchmark
    # ------------------------------------------------------------------
    print("\n=== Stage 2: Benchmarking ===")
    print(
        "Note: results reflect OS page cache state. Files will warm in cache during the run."
    )
    results = run_benchmark(output_dir, args.formats, args.n_reads, patch_size, rng)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print_results_table(results, source_disk_mb)

    results_path = output_dir / "results.json"
    payload = {
        "dataset_dir": str(dataset_dir),
        "source_dir": str(source_dir),
        "n_images": len(source_files),
        "n_reads": args.n_reads,
        "patch_size": list(patch_size),
        "seed": args.seed,
        "source_disk_mb": round(source_disk_mb, 2),
        "formats": results,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
