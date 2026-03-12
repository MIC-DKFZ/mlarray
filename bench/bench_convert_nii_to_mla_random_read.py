#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from medvol import MedVol
from tqdmp import tqdmp
from tqdm import tqdm

from mlarray import MLArray

try:
    from mlarray.blosc2_layout_strategies import (
        comp_blosc2_params_baseline,
        comp_blosc2_params_generalized,
        comp_blosc2_params_spatial_only,
        comp_blosc2_params_spatial_only_magnitude,
    )
except ModuleNotFoundError:
    from mlarray.blosc2_layout_strategies import (
        comp_blosc2_params_baseline,
        comp_blosc2_params_generalized,
        comp_blosc2_params_spatial_only,
        comp_blosc2_params_spatial_only_magnitude,
    )


LAYOUT_ALGOS: dict[str, Callable[..., tuple[list[int], list[int]]]] = {
    "baseline": comp_blosc2_params_baseline,
    "generalized": comp_blosc2_params_generalized,
    "spatial_only": comp_blosc2_params_spatial_only,
    "spatial_only_magnitude": comp_blosc2_params_spatial_only_magnitude,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all .nii.gz files in a dataset directory to .mla with a selected "
            "Blosc2 layout algorithm, then run random-read benchmarking."
        )
    )
    parser.add_argument(
        "--layout-algo",
        required=True,
        choices=sorted(LAYOUT_ALGOS.keys()),
        help="Layout algorithm name.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Input dataset directory containing .nii.gz files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for .mla files and benchmark results.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=192,
        help="Isotropic spatial patch size used for layout and random-read benchmarking.",
    )
    parser.add_argument(
        "--convert-processes",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="Number of processes for conversion via tqdmp.",
    )
    parser.add_argument(
        "--random-reads",
        type=int,
        default=10000,
        help=(
            "Total number of random read operations. "
            "Each operation samples one random file and reads one random patch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for random-read benchmarking. If omitted, randomness is nondeterministic.",
    )
    parser.add_argument(
        "--overwrite",
        "--recreate",
        dest="overwrite",
        action="store_true",
        help=(
            "Force fresh conversion of all files (overwrite existing .mla files). "
            "By default, valid existing outputs are reused."
        ),
    )
    parser.add_argument(
        "--mmap-mode",
        default="r",
        choices=["r", "c", "r+", "none"],
        help="mmap_mode used by MLArray.open during random-read benchmark.",
    )
    parser.add_argument(
        "--open-mode",
        default="r",
        choices=["r", "a"],
        help="mode used by MLArray.open during random-read benchmark.",
    )
    parser.add_argument(
        "--dparams-nthreads",
        type=int,
        default=1,
        help="Blosc2 dparams nthreads for MLArray.open in read benchmark.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def infer_spatial_axis_mask(shape: tuple[int, ...]) -> list[bool]:
    ndims = len(shape)
    if ndims == 2:
        return [True, True]
    if ndims == 3:
        return [True, True, True]
    if ndims == 4:
        # MedVol convention: channel/modality-first for 4D.
        return [False, True, True, True]
    raise RuntimeError(f"Unsupported dimensionality for this script: {ndims}")


def output_path_for_input(input_dir: Path, output_dir: Path, input_path: Path) -> Path:
    rel = input_path.relative_to(input_dir)
    rel_str = rel.as_posix()
    if rel_str.endswith(".nii.gz"):
        rel_out = rel_str[:-7] + ".mla"
    else:
        rel_out = str(rel.with_suffix(".mla"))
    return output_dir / rel_out


def convert_one(
    job: Union[str, dict[str, Any]],
    *,
    input_dir: str,
    output_dir: str,
    layout_algo_name: str,
    patch_size: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if isinstance(job, dict):
        input_path = Path(job["input_path"])
        preview_print = bool(job.get("preview_print", False))
    else:
        input_path = Path(job)
        preview_print = False
    input_dir_p = Path(input_dir)
    output_dir_p = Path(output_dir)
    output_path = output_path_for_input(input_dir_p, output_dir_p, input_path)

    try:
        med = MedVol(input_path)
        array = med.array
        shape = tuple(int(v) for v in array.shape)
        spatial_axis_mask = infer_spatial_axis_mask(shape)
        spatial_ndims = sum(1 for v in spatial_axis_mask if v)
        patch = tuple(int(patch_size) for _ in range(spatial_ndims))

        layout_func = LAYOUT_ALGOS[layout_algo_name]
        chunk_size, block_size = layout_func(
            image_size=shape,
            patch_size=patch,
            spatial_axis_mask=spatial_axis_mask,
            bytes_per_pixel=array.dtype.itemsize,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = MLArray(
            array=array,
            patch_size=None,
            chunk_size=chunk_size,
            block_size=block_size,
        )
        image.save(output_path)
        image.close()
        if preview_print:
            print(
                f"converted image={list(shape)} chunk={list(chunk_size)} block={list(block_size)} "
                f"path={output_path}",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "success": True,
            "skipped_existing": False,
            "validated_existing": False,
            "shape": list(shape),
            "dtype": str(array.dtype),
            "spatial_axis_mask": spatial_axis_mask,
            "patch_size": list(patch),
            "chunk_size": [int(v) for v in chunk_size],
            "block_size": [int(v) for v in block_size],
            "seconds": elapsed,
            "error": "",
        }
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "success": False,
            "skipped_existing": False,
            "validated_existing": False,
            "seconds": elapsed,
            "error": f"{type(e).__name__}: {e}",
        }


def check_existing_mla(path: Path) -> tuple[bool, str]:
    image = None
    try:
        image = MLArray.open(path, mode="r", mmap_mode="r", dparams={"nthreads": 1})
        _ = image.shape
        _ = image.dtype
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass


def random_read_slices(
    shape: tuple[int, ...],
    spatial_axis_mask: list[bool],
    spatial_patch_size: int,
    rng: np.random.Generator,
) -> tuple[slice, ...]:
    slices: list[slice] = []
    for axis_len, is_spatial in zip(shape, spatial_axis_mask):
        axis_len = int(axis_len)
        if is_spatial:
            patch_len = min(axis_len, int(spatial_patch_size))
            start_max = axis_len - patch_len
            start = 0 if start_max <= 0 else int(rng.integers(0, start_max + 1))
            slices.append(slice(start, start + patch_len))
        else:
            # Read across all modalities / channels.
            slices.append(slice(0, axis_len))
    return tuple(slices)


def bench_random_read_one(
    image: MLArray,
    filepath: Path,
    *,
    patch_size: int,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    spatial_axis_mask: list[bool],
    rng: np.random.Generator,
    read_idx: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    slc = random_read_slices(shape, spatial_axis_mask, patch_size, rng)
    patch = image[slc]
    patch_data = patch.to_numpy() if hasattr(patch, "to_numpy") else np.asarray(patch)

    elapsed = time.perf_counter() - t0
    bytes_total = int(np.prod(patch_data.shape)) * dtype.itemsize
    throughput_bps = float(bytes_total) / elapsed if elapsed > 0 else 0.0

    return {
        "read_idx": int(read_idx),
        "output_path": str(filepath),
        "success": True,
        "seconds": elapsed,
        "bytes_per_read": bytes_total,
        "bytes_total": bytes_total,
        "throughput_Bps": throughput_bps,
        "throughput_GiBps": throughput_bps / (1024**3),
        "shape": list(shape),
        "dtype": str(dtype),
        "spatial_axis_mask": spatial_axis_mask,
        "slice": str(slc),
        "error": "",
    }


def run_random_read_benchmark(
    output_paths: list[str],
    *,
    patch_size: int,
    random_reads: int,
    seed: Optional[int],
    mmap_mode: Optional[str],
    open_mode: str,
    dparams_nthreads: int,
    disable_progress: bool,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    opened: list[dict[str, Any]] = []

    for out_path in tqdm(output_paths, desc="Opening files"):
        path = Path(out_path)
        image = MLArray.open(
            path,
            mode=open_mode,
            mmap_mode=mmap_mode,
            dparams={"nthreads": dparams_nthreads},
        )
        shape = tuple(int(v) for v in image.shape)
        dtype = np.dtype(image.dtype)
        spatial_axis_mask = infer_spatial_axis_mask(shape)
        opened.append(
            {
                "path": path,
                "image": image,
                "shape": shape,
                "dtype": dtype,
                "spatial_axis_mask": spatial_axis_mask,
            }
        )

    n_files = len(opened)
    if n_files == 0:
        return rows

    warmup_reads = int(int(random_reads) * 0.1)

    for read_idx in tqdm(
        range(int(random_reads)),
        desc="Random read benchmark",
        disable=disable_progress,
    ):
        picked_idx = int(rng.integers(0, n_files))
        picked = opened[picked_idx]
        row = bench_random_read_one(
            picked["image"],
            picked["path"],
            patch_size=patch_size,
            shape=picked["shape"],
            dtype=picked["dtype"],
            spatial_axis_mask=picked["spatial_axis_mask"],
            rng=rng,
            read_idx=read_idx,
        )
        row["is_warmup"] = read_idx < warmup_reads
        row["in_measurement"] = not row["is_warmup"]
        rows.append(row)
    return rows


def summarize_conversion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success_rows = [r for r in rows if r["success"]]
    fail_rows = [r for r in rows if not r["success"]]
    skipped = sum(1 for r in success_rows if r.get("skipped_existing", False))
    secs = [float(r["seconds"]) for r in success_rows]
    return {
        "total": total,
        "successful": len(success_rows),
        "failed": len(fail_rows),
        "skipped_existing": skipped,
        "mean_seconds_success": float(statistics.mean(secs)) if secs else 0.0,
        "median_seconds_success": float(statistics.median(secs)) if secs else 0.0,
    }


def summarize_random_read(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    warmup_total = sum(1 for r in rows if bool(r.get("is_warmup", False)))
    measured_rows = [r for r in rows if bool(r.get("in_measurement", True))]
    success_rows = [r for r in measured_rows if r["success"]]
    fail_rows = [r for r in measured_rows if not r["success"]]

    throughputs = [float(r["throughput_GiBps"]) for r in success_rows]
    total_bytes = int(sum(int(r["bytes_total"]) for r in success_rows))
    total_seconds = float(sum(float(r["seconds"]) for r in success_rows))
    overall_gibps = (
        (float(total_bytes) / total_seconds) / (1024**3)
        if total_seconds > 0
        else 0.0
    )

    return {
        "total_reads": total,
        "warmup_reads": warmup_total,
        "measured_reads": len(measured_rows),
        "successful": len(success_rows),
        "failed": len(fail_rows),
        "overall_GiBps": overall_gibps,
        "mean_GiBps": float(statistics.mean(throughputs)) if throughputs else 0.0,
        "median_GiBps": float(statistics.median(throughputs)) if throughputs else 0.0,
        "min_GiBps": float(min(throughputs)) if throughputs else 0.0,
        "max_GiBps": float(max(throughputs)) if throughputs else 0.0,
        "total_bytes": total_bytes,
        "total_seconds": total_seconds,
    }


def print_summary(
    conversion_summary: dict[str, Any],
    read_summary: dict[str, Any],
) -> None:
    print("\n=== Conversion Summary ===")
    print(
        f"total={conversion_summary['total']} "
        f"success={conversion_summary['successful']} "
        f"failed={conversion_summary['failed']} "
        f"skipped_existing={conversion_summary['skipped_existing']}"
    )
    print(
        f"mean_s={conversion_summary['mean_seconds_success']:.3f} "
        f"median_s={conversion_summary['median_seconds_success']:.3f}"
    )

    print("\n=== Random Read Summary ===")
    print(
        f"total={read_summary['total_reads']} "
        f"warmup={read_summary['warmup_reads']} "
        f"measured={read_summary['measured_reads']} "
        f"success={read_summary['successful']} "
        f"failed={read_summary['failed']}"
    )
    print(
        f"overall_GiB/s={read_summary['overall_GiBps']:.3f} "
        f"mean_GiB/s={read_summary['mean_GiBps']:.3f} "
        f"median_GiB/s={read_summary['median_GiBps']:.3f} "
        f"min_GiB/s={read_summary['min_GiBps']:.3f} "
        f"max_GiB/s={read_summary['max_GiBps']:.3f}"
    )


def main() -> None:
    args = parse_args()
    if args.convert_processes < 0:
        raise ValueError("Process counts must be >= 0")
    if args.patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if args.random_reads <= 0:
        raise ValueError("random_reads must be > 0")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input dir does not exist or is not a directory: {input_dir}")

    nii_paths = sorted(input_dir.rglob("*.nii.gz"))
    if not nii_paths:
        raise RuntimeError(f"No .nii.gz files found in {input_dir}")

    print(f"Found {len(nii_paths)} .nii.gz files")
    print(f"Layout algorithm: {args.layout_algo}")
    print("Scanning existing outputs for resumable conversion...")

    conversion_rows: list[dict[str, Any]] = []
    pending_inputs: list[str] = []
    reused_existing = 0
    corrupted_existing = 0

    for input_path in nii_paths:
        output_path = output_path_for_input(input_dir, output_dir, input_path)
        if args.overwrite:
            pending_inputs.append(str(input_path))
            continue

        if output_path.exists():
            ok, err = check_existing_mla(output_path)
            if ok:
                reused_existing += 1
                conversion_rows.append(
                    {
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "success": True,
                        "skipped_existing": True,
                        "validated_existing": True,
                        "seconds": 0.0,
                        "error": "",
                    }
                )
                continue
            corrupted_existing += 1
            pending_inputs.append(str(input_path))
        else:
            pending_inputs.append(str(input_path))

    print(
        f"Resume scan: reuse_valid={reused_existing} "
        f"reconvert={len(pending_inputs)} "
        f"invalid_existing={corrupted_existing} "
        f"overwrite={bool(args.overwrite)}"
    )

    if pending_inputs:
        pending_jobs = [
            {"input_path": p, "preview_print": i < 10}
            for i, p in enumerate(pending_inputs)
        ]
        converted_rows = tqdmp(
            convert_one,
            pending_jobs,
            num_processes=args.convert_processes,
            desc="Convert NIfTI -> MLArray",
            disable=args.disable_progress,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            layout_algo_name=args.layout_algo,
            patch_size=int(args.patch_size),
        )
        conversion_rows.extend(converted_rows)

    converted_success = [row for row in conversion_rows if row["success"]]
    read_paths = [row["output_path"] for row in converted_success]
    if not read_paths:
        raise RuntimeError("No successful conversions available for random-read benchmark.")

    mmap_mode = None if args.mmap_mode == "none" else args.mmap_mode
    read_rows = run_random_read_benchmark(
        read_paths,
        patch_size=int(args.patch_size),
        random_reads=int(args.random_reads),
        seed=args.seed,
        mmap_mode=mmap_mode,
        open_mode=args.open_mode,
        dparams_nthreads=int(args.dparams_nthreads),
        disable_progress=bool(args.disable_progress),
    )

    conversion_summary = summarize_conversion(conversion_rows)
    read_summary = summarize_random_read(read_rows)
    print_summary(conversion_summary, read_summary)

    out_json = output_dir / f"bench_convert_nii_to_mla_random_read__{args.layout_algo}.json"
    payload = {
        "config": {
            "layout_algo": args.layout_algo,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "patch_size": int(args.patch_size),
            "convert_processes": int(args.convert_processes),
            "random_reads": int(args.random_reads),
            "seed": args.seed,
            "overwrite": bool(args.overwrite),
            "open_mode": args.open_mode,
            "mmap_mode": mmap_mode,
            "dparams_nthreads": int(args.dparams_nthreads),
            "random_read_strategy": (
                "serial; each read picks one random file and one random patch"
            ),
            "random_read_warmup_fraction": 0.1,
        },
        "conversion_summary": conversion_summary,
        "random_read_summary": read_summary,
        "conversion_rows": conversion_rows,
        "random_read_rows": read_rows,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote benchmark results: {out_json}")


if __name__ == "__main__":
    main()
