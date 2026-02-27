#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from mlarray import MLArray


# ---------------------------------------------------------------------------
# Baseline layout method copied from MLArray.comp_blosc2_params (do not call
# MLArray.comp_blosc2_params in this benchmark; this is a local baseline copy).
# ---------------------------------------------------------------------------
def comp_blosc2_params_baseline(
    image_size: tuple[int, ...],
    patch_size: tuple[int, ...],
    spatial_axis_mask: Optional[list[bool]] = None,
    bytes_per_pixel: int = 4,
    l1_cache_size_per_core_in_bytes: int = 32768,
    l3_cache_size_per_core_in_bytes: int = 1441792,
    safety_factor: float = 0.8,
) -> tuple[list[int], list[int]]:
    def _move_index_list(a, src, dst):
        a = list(a)
        x = a.pop(src)
        a.insert(dst, x)
        return a

    num_squeezes = 0
    if len(image_size) == 2:
        image_size = (1, 1, *image_size)
        num_squeezes = 2
    elif len(image_size) == 3:
        image_size = (1, *image_size)
        num_squeezes = 1

    non_spatial_axis = None
    if spatial_axis_mask is not None:
        non_spatial_axis_mask = [not b for b in spatial_axis_mask]
        if sum(non_spatial_axis_mask) > 1:
            raise RuntimeError(
                "Automatic blosc2 optimization currently only supports one "
                "non-spatial axis. Please set chunk and block size manually."
            )
        non_spatial_axis = next((i for i, v in enumerate(non_spatial_axis_mask) if v), None)
        if non_spatial_axis is not None:
            image_size = _move_index_list(image_size, non_spatial_axis + num_squeezes, 0)

    if len(image_size) != 4:
        raise RuntimeError("Image size must be 4D.")

    if not (len(patch_size) == 2 or len(patch_size) == 3):
        raise RuntimeError("Patch size must be 2D or 3D.")

    non_spatial_size = image_size[0]
    if len(patch_size) == 2:
        patch_size = [1, *patch_size]
    patch_size = np.array(patch_size)
    block_size = np.array(
        (non_spatial_size, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size])
    )

    # Shrink block until it fits into L1 target.
    estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
    while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
        axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
        idx = 0
        picked_axis = axis_order[idx]
        while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
            idx += 1
            picked_axis = axis_order[idx]
        block_size[picked_axis + 1] = 2 ** (
            max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1)))
        )
        block_size[picked_axis + 1] = min(
            block_size[picked_axis + 1], image_size[picked_axis + 1]
        )
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

    block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

    # Grow chunks from blocks toward L3 target.
    chunk_size = block_size.copy()
    estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
    while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
        if patch_size[0] == 1 and all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
            break
        if all([i == j for i, j in zip(chunk_size, image_size)]):
            break
        axis_order = np.argsort(chunk_size[1:] / block_size[1:])
        idx = 0
        picked_axis = axis_order[idx]
        while (
            chunk_size[picked_axis + 1] == image_size[picked_axis + 1]
            or patch_size[picked_axis] == 1
        ):
            idx += 1
            picked_axis = axis_order[idx]
        chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
        chunk_size[picked_axis + 1] = min(
            chunk_size[picked_axis + 1], image_size[picked_axis + 1]
        )
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
            chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
            break
    chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

    if non_spatial_axis is not None:
        block_size = _move_index_list(block_size, 0, non_spatial_axis + num_squeezes)
        chunk_size = _move_index_list(chunk_size, 0, non_spatial_axis + num_squeezes)

    block_size = block_size[num_squeezes:]
    chunk_size = chunk_size[num_squeezes:]
    return [int(v) for v in chunk_size], [int(v) for v in block_size]


@dataclass
class CaseDef:
    tier: str
    name: str
    shape: tuple[int, ...]
    spatial_axis_mask: list[bool]
    patch_sizes: list[tuple[int, ...]]


def build_cases() -> list[CaseDef]:
    # Includes 2D, 3D, and 4D-total arrays (3D spatial + one non-spatial axis).
    return [
        # Small
        CaseDef("small", "2d_spatial", (2048, 2048), [True, True], [(64, 64), (128, 128), (192, 192)]),
        CaseDef("small", "3d_2dsp_plus_nonsp", (4, 1024, 1024), [False, True, True], [(64, 64), (128, 128), (192, 192)]),
        CaseDef("small", "3d_spatial", (128, 128, 128), [True, True, True], [(32, 32, 32), (64, 64, 64), (96, 96, 96)]),
        CaseDef("small", "4d_3dsp_plus_nonsp", (4, 96, 96, 96), [False, True, True, True], [(32, 32, 32), (64, 64, 64), (80, 80, 80)]),
        # Medium
        CaseDef("medium", "2d_spatial", (4096, 4096), [True, True], [(64, 64), (128, 128), (256, 256)]),
        CaseDef("medium", "3d_2dsp_plus_nonsp", (8, 2048, 2048), [False, True, True], [(64, 64), (128, 128), (256, 256)]),
        CaseDef("medium", "3d_spatial", (256, 256, 256), [True, True, True], [(32, 32, 32), (64, 64, 64), (128, 128, 128)]),
        CaseDef("medium", "4d_3dsp_plus_nonsp", (4, 192, 192, 192), [False, True, True, True], [(32, 32, 32), (64, 64, 64), (96, 96, 96)]),
        # Large
        CaseDef("large", "2d_spatial", (8192, 8192), [True, True], [(128, 128), (192, 192), (256, 256)]),
        CaseDef("large", "3d_2dsp_plus_nonsp", (8, 4096, 4096), [False, True, True], [(128, 128), (192, 192), (256, 256)]),
        CaseDef("large", "3d_spatial", (512, 512, 512), [True, True, True], [(64, 64, 64), (96, 96, 96), (128, 128, 128)]),
        CaseDef("large", "4d_3dsp_plus_nonsp", (4, 384, 384, 384), [False, True, True, True], [(64, 64, 64), (96, 96, 96), (128, 128, 128)]),
        # Very large (includes 2048^3-equivalent scale)
        CaseDef("very_large", "2d_spatial", (16384, 16384), [True, True], [(192, 192), (256, 256), (384, 384)]),
        CaseDef("very_large", "3d_2dsp_plus_nonsp", (8, 8192, 8192), [False, True, True], [(192, 192), (256, 256), (384, 384)]),
        CaseDef("very_large", "3d_spatial_2048", (2048, 2048, 2048), [True, True, True], [(128, 128, 128), (192, 192, 192), (256, 256, 256)]),
        CaseDef("very_large", "4d_3dsp_plus_nonsp_equiv", (4, 1024, 1024, 2048), [False, True, True, True], [(128, 128, 128), (192, 192, 192), (256, 256, 256)]),
    ]


def parse_dtype(dtype_name: str) -> np.dtype:
    dt = np.dtype(dtype_name)
    if dt.kind not in ("f", "i", "u"):
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use numeric dtype.")
    return dt


def should_test_write(mode: str, mmap_mode: Optional[str]) -> bool:
    if mode == "a" and mmap_mode is None:
        return True
    if mmap_mode in ("r+", "w+"):
        return True
    return False


def mmap_label(mmap_mode: Optional[str]) -> str:
    return "None" if mmap_mode is None else mmap_mode


def patch_shape_for_case(
    shape: tuple[int, ...], spatial_axis_mask: list[bool], patch_size: tuple[int, ...]
) -> tuple[int, ...]:
    out = []
    sp_i = 0
    for axis_len, is_spatial in zip(shape, spatial_axis_mask):
        if is_spatial:
            out.append(min(axis_len, patch_size[sp_i]))
            sp_i += 1
        else:
            out.append(1)
    return tuple(out)


def random_patch_slices(
    shape: tuple[int, ...],
    spatial_axis_mask: list[bool],
    patch_size: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[slice, ...]:
    slices: list[slice] = []
    sp_i = 0
    for axis_len, is_spatial in zip(shape, spatial_axis_mask):
        if is_spatial:
            patch_len = min(axis_len, patch_size[sp_i])
            sp_i += 1
            start_max = axis_len - patch_len
            start = 0 if start_max == 0 else int(rng.integers(0, start_max + 1))
            slices.append(slice(start, start + patch_len))
        else:
            idx = 0 if axis_len == 1 else int(rng.integers(0, axis_len))
            slices.append(slice(idx, idx + 1))
    return tuple(slices)


def create_patch_buffer(
    shape: tuple[int, ...], dtype: np.dtype, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape, dtype=np.float32).astype(dtype, copy=False)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 127, size=shape, dtype=dtype)
    raise ValueError(f"Unsupported dtype for patch buffer: {dtype}")


def drop_linux_caches() -> None:
    # Cold-cache read runs: drop page cache after file creation and before open/read.
    if os.name != "posix":
        raise RuntimeError("Cold cache dropping is currently Linux/POSIX only.")
    if os.geteuid() != 0:
        raise PermissionError(
            "Cold cache dropping requires root (run via sudo or as root)."
        )
    subprocess.run(["sync"], check=True)
    with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as f:
        f.write("3\n")


def initialize_dataset(
    filepath: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunk_size: list[int],
    block_size: list[int],
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        filepath.unlink()
    image = MLArray.create(
        filepath,
        shape=shape,
        dtype=dtype,
        patch_size=None,
        chunk_size=chunk_size,
        block_size=block_size,
        mmap_mode="w+",
        dparams={"nthreads": 1},
    )
    image[...] = 0
    image.close()


def run_operation(
    filepath: Path,
    *,
    mode: str,
    mmap_mode: Optional[str],
    operation: str,
    patch_reads: int,
    patch_writes: int,
    shape: tuple[int, ...],
    spatial_axis_mask: list[bool],
    patch_size: tuple[int, ...],
    patch_buffer: np.ndarray,
    dtype: np.dtype,
    nthreads: int,
    seed: int,
) -> tuple[bool, float, int, Optional[str]]:
    start = time.perf_counter()
    bytes_processed = 0
    image = None
    rng = np.random.default_rng(seed)
    try:
        image = MLArray.open(
            filepath,
            mode=mode,
            mmap_mode=mmap_mode,
            dparams={"nthreads": nthreads},
        )
        if operation == "read_full":
            _ = image[...]
            bytes_processed = int(np.prod(shape)) * dtype.itemsize
        elif operation == "read_patch_random":
            for _ in range(patch_reads):
                slc = random_patch_slices(shape, spatial_axis_mask, patch_size, rng)
                _ = image[slc]
            bytes_processed = patch_reads * int(np.prod(patch_buffer.shape)) * dtype.itemsize
        elif operation == "write_patch_random":
            for _ in range(patch_writes):
                slc = random_patch_slices(shape, spatial_axis_mask, patch_size, rng)
                image[slc] = patch_buffer
            bytes_processed = patch_writes * int(np.prod(patch_buffer.shape)) * dtype.itemsize
        else:
            raise ValueError(f"Unknown operation '{operation}'.")
        elapsed = time.perf_counter() - start
        return True, elapsed, bytes_processed, None
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return False, elapsed, bytes_processed, f"{type(e).__name__}: {e}"
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass


def run_operation_worker(worker_spec_json: str) -> int:
    try:
        spec = json.loads(worker_spec_json)
        shape = tuple(spec["shape"])
        spatial_axis_mask = list(spec["spatial_axis_mask"])
        patch_size = tuple(spec["patch_size"])
        dtype = np.dtype(spec["dtype"])
        patch_shape = patch_shape_for_case(shape, spatial_axis_mask, patch_size)
        patch_buffer = create_patch_buffer(
            patch_shape, dtype=dtype, seed=int(spec["patch_seed"])
        )

        success, seconds, bytes_processed, error = run_operation(
            filepath=Path(spec["filepath"]),
            mode=spec["mode"],
            mmap_mode=spec["mmap_mode"],
            operation=spec["operation"],
            patch_reads=int(spec["patch_reads"]),
            patch_writes=int(spec["patch_writes"]),
            shape=shape,
            spatial_axis_mask=spatial_axis_mask,
            patch_size=patch_size,
            patch_buffer=patch_buffer,
            dtype=dtype,
            nthreads=int(spec["nthreads"]),
            seed=int(spec["seed"]),
        )
        print(
            json.dumps(
                {
                    "success": success,
                    "seconds": seconds,
                    "bytes_processed": bytes_processed,
                    "error": error or "",
                }
            )
        )
        return 0
    except Exception as e:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "success": False,
                    "seconds": 0.0,
                    "bytes_processed": 0,
                    "error": f"worker_exception: {type(e).__name__}: {e}",
                }
            )
        )
        return 1


def run_operation_isolated(
    *,
    timeout_seconds: int,
    **kwargs: Any,
) -> tuple[bool, float, int, Optional[str]]:
    # Run each measurement in a subprocess so native crashes are isolated
    # and recorded as failed rows instead of aborting the whole benchmark.
    dtype = kwargs["dtype"]
    patch_buffer = kwargs["patch_buffer"]
    spec = {
        "filepath": str(kwargs["filepath"]),
        "mode": kwargs["mode"],
        "mmap_mode": kwargs["mmap_mode"],
        "operation": kwargs["operation"],
        "patch_reads": kwargs["patch_reads"],
        "patch_writes": kwargs["patch_writes"],
        "shape": list(kwargs["shape"]),
        "spatial_axis_mask": list(kwargs["spatial_axis_mask"]),
        "patch_size": list(kwargs["patch_size"]),
        "dtype": str(dtype),
        "nthreads": int(kwargs["nthreads"]),
        "seed": int(kwargs["seed"]),
        # Reconstruct patch buffer deterministically in worker.
        "patch_seed": 12345 + int(np.prod(patch_buffer.shape)),
    }
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-spec-json",
        json.dumps(spec),
    ]
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, 0.0, 0, f"worker_timeout_after_{timeout_seconds}s"

    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "").strip().splitlines()[-1:] or [""]
        return (
            False,
            0.0,
            0,
            f"worker_exitcode_{completed.returncode}: {stderr_tail[0]}".strip(),
        )

    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        return False, 0.0, 0, "worker_no_output"
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False, 0.0, 0, "worker_invalid_json"

    return (
        bool(payload.get("success", False)),
        float(payload.get("seconds", 0.0)),
        int(payload.get("bytes_processed", 0)),
        payload.get("error") or None,
    )


def is_schunk_open_null_error(error: Optional[str]) -> bool:
    if not error:
        return False
    return "blosc2_schunk_open_offset" in error and "returned NULL" in error


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not row["success"]:
            continue
        key = (
            row["tier"],
            row["case"],
            row["shape"],
            row["patch_size"],
            row["layout_method"],
            row["operation"],
            row["mode"],
            row["mmap_mode"],
            row["cache_state"],
        )
        if key not in agg:
            agg[key] = {"n": 0, "seconds": 0.0, "bytes": 0}
        agg[key]["n"] += 1
        agg[key]["seconds"] += row["seconds"]
        agg[key]["bytes"] += row["bytes_processed"]

    out = []
    for key, v in agg.items():
        tier, case, shape, patch, method, op, mode, mmap_mode, cache_state = key
        throughput = float(v["bytes"]) / float(v["seconds"]) if v["seconds"] > 0 else 0.0
        out.append(
            {
                "tier": tier,
                "case": case,
                "shape": shape,
                "patch_size": patch,
                "layout_method": method,
                "operation": op,
                "mode": mode,
                "mmap_mode": mmap_mode,
                "cache_state": cache_state,
                "runs": v["n"],
                "mean_seconds": v["seconds"] / v["n"],
                "throughput_Bps": throughput,
                "throughput_GiBps": throughput / (1024**3),
            }
        )
    out.sort(
        key=lambda r: (
            r["tier"],
            r["case"],
            r["operation"],
            r["mode"],
            r["mmap_mode"],
            r["cache_state"],
        )
    )
    return out


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\n=== Benchmark Summary (successful runs) ===")
    if not summary_rows:
        print("No successful runs.")
        return
    header = (
        "tier case shape patch method op mode mmap cache runs mean_s GiB/s"
    )
    print(header)
    for r in summary_rows:
        print(
            f"{r['tier']:>10} "
            f"{r['case']:<24} "
            f"{r['shape']:<18} "
            f"{r['patch_size']:<14} "
            f"{r['layout_method']:<10} "
            f"{r['operation']:<18} "
            f"{r['mode']:<4} "
            f"{r['mmap_mode']:<5} "
            f"{r['cache_state']:<5} "
            f"{r['runs']:<4d} "
            f"{r['mean_seconds']:<7.3f} "
            f"{r['throughput_GiBps']:<8.3f}"
        )


def build_final_summary(
    rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    total_runs = len(rows)
    successful_runs = sum(1 for row in rows if row["success"])
    failed_runs = total_runs - successful_runs
    skipped_runs = sum(
        1
        for row in rows
        if isinstance(row.get("error"), str) and row["error"].startswith("skipped_")
    )
    success_rate = (
        float(successful_runs) / float(total_runs) if total_runs > 0 else 0.0
    )

    by_operation: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_runs": 0, "successful_runs": 0, "throughputs_gibps": []}
    )
    for row in rows:
        op = row["operation"]
        by_operation[op]["total_runs"] += 1
        if row["success"]:
            by_operation[op]["successful_runs"] += 1
            by_operation[op]["throughputs_gibps"].append(float(row["throughput_GiBps"]))

    by_operation_out: dict[str, Any] = {}
    for op, stats in sorted(by_operation.items()):
        values = stats["throughputs_gibps"]
        total = int(stats["total_runs"])
        successful = int(stats["successful_runs"])
        by_operation_out[op] = {
            "total_runs": total,
            "successful_runs": successful,
            "failed_runs": total - successful,
            "success_rate": float(successful) / float(total) if total > 0 else 0.0,
            "mean_throughput_GiBps": float(np.mean(values)) if values else 0.0,
            "median_throughput_GiBps": float(np.median(values)) if values else 0.0,
            "max_throughput_GiBps": float(np.max(values)) if values else 0.0,
        }

    best_by_operation: dict[str, Any] = {}
    for op in sorted({row["operation"] for row in summary_rows}):
        op_rows = [row for row in summary_rows if row["operation"] == op]
        if not op_rows:
            best_by_operation[op] = None
            continue
        best = max(op_rows, key=lambda row: row["throughput_GiBps"])
        best_by_operation[op] = {
            "throughput_GiBps": best["throughput_GiBps"],
            "mean_seconds": best["mean_seconds"],
            "runs": best["runs"],
            "tier": best["tier"],
            "case": best["case"],
            "shape": best["shape"],
            "patch_size": best["patch_size"],
            "layout_method": best["layout_method"],
            "mode": best["mode"],
            "mmap_mode": best["mmap_mode"],
            "cache_state": best["cache_state"],
        }

    best_overall = None
    if summary_rows:
        best = max(summary_rows, key=lambda row: row["throughput_GiBps"])
        best_overall = {
            "throughput_GiBps": best["throughput_GiBps"],
            "mean_seconds": best["mean_seconds"],
            "runs": best["runs"],
            "tier": best["tier"],
            "case": best["case"],
            "shape": best["shape"],
            "patch_size": best["patch_size"],
            "layout_method": best["layout_method"],
            "operation": best["operation"],
            "mode": best["mode"],
            "mmap_mode": best["mmap_mode"],
            "cache_state": best["cache_state"],
        }

    return {
        "totals": {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "skipped_runs": skipped_runs,
            "success_rate": success_rate,
        },
        "by_operation": by_operation_out,
        "best_by_operation": best_by_operation,
        "best_overall": best_overall,
    }


def print_final_summary(summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    print("\n=== Final Run Summary ===")
    print(
        f"runs={totals['total_runs']} success={totals['successful_runs']} "
        f"failed={totals['failed_runs']} skipped={totals['skipped_runs']} "
        f"success_rate={totals['success_rate'] * 100.0:.1f}%"
    )

    by_operation = summary["by_operation"]
    if by_operation:
        print("\nPer operation:")
        for op, stats in by_operation.items():
            print(
                f"  {op:<18} runs={stats['total_runs']:<4d} "
                f"success={stats['successful_runs']:<4d} "
                f"mean_GiB/s={stats['mean_throughput_GiBps']:.3f} "
                f"median_GiB/s={stats['median_throughput_GiBps']:.3f} "
                f"max_GiB/s={stats['max_throughput_GiBps']:.3f}"
            )

    best_by_operation = summary["best_by_operation"]
    if best_by_operation:
        print("\nBest aggregated config per operation (by GiB/s):")
        for op, best in best_by_operation.items():
            if best is None:
                print(f"  {op:<18} no successful runs")
                continue
            print(
                f"  {op:<18} {best['throughput_GiBps']:.3f} GiB/s "
                f"({best['tier']}:{best['case']}, patch={best['patch_size']}, "
                f"method={best['layout_method']}, mode={best['mode']}, "
                f"mmap={best['mmap_mode']}, cache={best['cache_state']})"
            )

    best_overall = summary["best_overall"]
    if best_overall is not None:
        print(
            "\nBest overall: "
            f"{best_overall['throughput_GiBps']:.3f} GiB/s "
            f"({best_overall['operation']} | {best_overall['tier']}:{best_overall['case']} | "
            f"patch={best_overall['patch_size']} | method={best_overall['layout_method']} | "
            f"mode={best_overall['mode']} mmap={best_overall['mmap_mode']} cache={best_overall['cache_state']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark IO throughput across Blosc2 open mode/mmap_mode combinations "
            "for different chunk/block layouts derived from comp_blosc2_params methods."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("bench/data"),
        help="Directory for generated .mla datasets.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("bench/results"),
        help="Directory for CSV/JSON benchmark outputs.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["small", "medium", "large", "very_large"],
        choices=["small", "medium", "large", "very_large"],
        help="Size tiers to include.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per configuration.",
    )
    parser.add_argument(
        "--patch-reads",
        type=int,
        default=16,
        help="Number of random patch reads in read_patch_random.",
    )
    parser.add_argument(
        "--patch-writes",
        type=int,
        default=16,
        help="Number of random patch writes in write_patch_random.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["warm", "cold", "both"],
        default="both",
        help="Run warm-cache only, cold-cache only, or both.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="Blosc2 dparams nthreads used for MLArray.open(...).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Array dtype for datasets (default: float32).",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=134_217_728,
        help=(
            "Skip cases with more than this many elements. "
            "Increase this to include very large cases (e.g. 2048^3)."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate datasets even if target files already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--isolate-runs",
        action="store_true",
        help=(
            "Run each measured operation in a subprocess. "
            "This is slower, but isolates native crashes/segfaults per run."
        ),
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=300,
        help="Timeout per isolated run when --isolate-runs is enabled.",
    )
    parser.add_argument(
        "--_worker-spec-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args._worker_spec_json is not None:
        raise SystemExit(run_operation_worker(args._worker_spec_json))

    dtype = parse_dtype(args.dtype)
    bytes_per_pixel = dtype.itemsize

    layout_methods: dict[str, Callable[..., tuple[list[int], list[int]]]] = {
        "baseline": comp_blosc2_params_baseline,
    }

    mode_values = ("r", "a")
    mmap_values: tuple[Optional[str], ...] = (None, "r", "r+", "c")
    operations = ("read_full", "read_patch_random", "write_patch_random")
    cache_states = ("warm", "cold") if args.cache_mode == "both" else (args.cache_mode,)

    all_cases = [c for c in build_cases() if c.tier in set(args.tiers)]
    timestamp = int(time.time())
    rows: list[dict[str, Any]] = []

    print("Preparing benchmark matrix...")
    for case in all_cases:
        n_elements = int(np.prod(case.shape))
        if n_elements > args.max_elements:
            print(
                f"SKIP case={case.name} shape={case.shape} "
                f"(elements={n_elements} > max={args.max_elements})"
            )
            continue

        for patch_size in case.patch_sizes:
            for method_name, method_func in layout_methods.items():
                try:
                    chunk_size, block_size = method_func(
                        image_size=case.shape,
                        patch_size=patch_size,
                        spatial_axis_mask=case.spatial_axis_mask,
                        bytes_per_pixel=bytes_per_pixel,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"SKIP layout failure case={case.name} patch={patch_size} "
                        f"method={method_name}: {type(e).__name__}: {e}"
                    )
                    continue

                dataset_name = (
                    f"{case.tier}__{case.name}__shape-{'x'.join(map(str, case.shape))}"
                    f"__patch-{'x'.join(map(str, patch_size))}__{method_name}.mla"
                )
                dataset_path = args.data_dir / dataset_name

                if args.recreate or not dataset_path.exists():
                    print(f"Create dataset: {dataset_path}")
                    initialize_dataset(
                        dataset_path,
                        shape=case.shape,
                        dtype=dtype,
                        chunk_size=chunk_size,
                        block_size=block_size,
                    )

                file_size = dataset_path.stat().st_size
                patch_shape = patch_shape_for_case(
                    case.shape, case.spatial_axis_mask, patch_size
                )
                patch_buffer = create_patch_buffer(
                    patch_shape, dtype=dtype, seed=args.seed + 123
                )

                for operation in operations:
                    for mode in mode_values:
                        for mmap_mode in mmap_values:
                            for cache_state in cache_states:
                                if (
                                    operation == "write_patch_random"
                                    and not should_test_write(mode, mmap_mode)
                                ):
                                    row = {
                                        "timestamp": timestamp,
                                        "tier": case.tier,
                                        "case": case.name,
                                        "shape": str(case.shape),
                                        "spatial_axis_mask": str(case.spatial_axis_mask),
                                        "patch_size": str(patch_size),
                                        "layout_method": method_name,
                                        "chunk_size": str(chunk_size),
                                        "block_size": str(block_size),
                                        "dtype": str(dtype),
                                        "nthreads": args.nthreads,
                                        "mode": mode,
                                        "mmap_mode": mmap_label(mmap_mode),
                                        "cache_state": cache_state,
                                        "operation": operation,
                                        "run_idx": -1,
                                        "success": False,
                                        "seconds": 0.0,
                                        "bytes_processed": 0,
                                        "throughput_Bps": 0.0,
                                        "throughput_GiBps": 0.0,
                                        "file_size_bytes": file_size,
                                        "error": "skipped_non_writable_mode",
                                    }
                                    rows.append(row)
                                    continue

                                if cache_state == "warm" and operation.startswith("read"):
                                    _ = run_operation(
                                        dataset_path,
                                        mode=mode,
                                        mmap_mode=mmap_mode,
                                        operation=operation,
                                        patch_reads=args.patch_reads,
                                        patch_writes=args.patch_writes,
                                        shape=case.shape,
                                        spatial_axis_mask=case.spatial_axis_mask,
                                        patch_size=patch_size,
                                        patch_buffer=patch_buffer,
                                        dtype=dtype,
                                        nthreads=args.nthreads,
                                        seed=args.seed + 7,
                                    )

                                for run_idx in range(args.runs):
                                    if cache_state == "cold" and operation.startswith("read"):
                                        try:
                                            # Important: drop caches after dataset is on disk and
                                            # right before open/read benchmarking.
                                            drop_linux_caches()
                                        except Exception as e:  # noqa: BLE001
                                            row = {
                                                "timestamp": timestamp,
                                                "tier": case.tier,
                                                "case": case.name,
                                                "shape": str(case.shape),
                                                "spatial_axis_mask": str(case.spatial_axis_mask),
                                                "patch_size": str(patch_size),
                                                "layout_method": method_name,
                                                "chunk_size": str(chunk_size),
                                                "block_size": str(block_size),
                                                "dtype": str(dtype),
                                                "nthreads": args.nthreads,
                                                "mode": mode,
                                                "mmap_mode": mmap_label(mmap_mode),
                                                "cache_state": cache_state,
                                                "operation": operation,
                                                "run_idx": run_idx,
                                                "success": False,
                                                "seconds": 0.0,
                                                "bytes_processed": 0,
                                                "throughput_Bps": 0.0,
                                                "throughput_GiBps": 0.0,
                                                "file_size_bytes": file_size,
                                                "error": f"cache_drop_failed: {type(e).__name__}: {e}",
                                            }
                                            rows.append(row)
                                            continue

                                    run_kwargs = {
                                        "filepath": dataset_path,
                                        "mode": mode,
                                        "mmap_mode": mmap_mode,
                                        "operation": operation,
                                        "patch_reads": args.patch_reads,
                                        "patch_writes": args.patch_writes,
                                        "shape": case.shape,
                                        "spatial_axis_mask": case.spatial_axis_mask,
                                        "patch_size": patch_size,
                                        "patch_buffer": patch_buffer,
                                        "dtype": dtype,
                                        "nthreads": args.nthreads,
                                        "seed": args.seed + run_idx + 1000,
                                    }
                                    if args.isolate_runs:
                                        success, seconds, bytes_processed, error = run_operation_isolated(
                                            timeout_seconds=args.worker_timeout_seconds,
                                            **run_kwargs,
                                        )
                                    else:
                                        success, seconds, bytes_processed, error = run_operation(
                                            **run_kwargs
                                        )
                                    if (not success) and is_schunk_open_null_error(error):
                                        print(
                                            f"Detected invalid dataset file, recreating and retrying once: {dataset_path}"
                                        )
                                        initialize_dataset(
                                            dataset_path,
                                            shape=case.shape,
                                            dtype=dtype,
                                            chunk_size=chunk_size,
                                            block_size=block_size,
                                        )
                                        if args.isolate_runs:
                                            success, seconds, bytes_processed, error = run_operation_isolated(
                                                timeout_seconds=args.worker_timeout_seconds,
                                                **run_kwargs,
                                            )
                                        else:
                                            success, seconds, bytes_processed, error = run_operation(
                                                **run_kwargs
                                            )
                                    throughput_bps = (
                                        float(bytes_processed) / seconds
                                        if success and seconds > 0
                                        else 0.0
                                    )
                                    row = {
                                        "timestamp": timestamp,
                                        "tier": case.tier,
                                        "case": case.name,
                                        "shape": str(case.shape),
                                        "spatial_axis_mask": str(case.spatial_axis_mask),
                                        "patch_size": str(patch_size),
                                        "layout_method": method_name,
                                        "chunk_size": str(chunk_size),
                                        "block_size": str(block_size),
                                        "dtype": str(dtype),
                                        "nthreads": args.nthreads,
                                        "mode": mode,
                                        "mmap_mode": mmap_label(mmap_mode),
                                        "cache_state": cache_state,
                                        "operation": operation,
                                        "run_idx": run_idx,
                                        "success": success,
                                        "seconds": seconds,
                                        "bytes_processed": int(bytes_processed),
                                        "throughput_Bps": throughput_bps,
                                        "throughput_GiBps": throughput_bps / (1024**3),
                                        "file_size_bytes": file_size,
                                        "error": error or "",
                                    }
                                    rows.append(row)
                                    print(
                                        f"{case.tier:>10} {case.name:<24} patch={patch_size!s:<14} "
                                        f"{operation:<18} mode={mode:<2} mmap={mmap_label(mmap_mode):<4} "
                                        f"cache={cache_state:<4} run={run_idx:<2} "
                                        f"ok={success!s:<5} "
                                        f"GiB/s={row['throughput_GiBps']:.3f}"
                                        + (f" err={error}" if error else "")
                                    )
                                    if args.isolate_runs and error and error.startswith(
                                        ("worker_exitcode_", "worker_timeout_")
                                    ):
                                        print(
                                            f"Recreate dataset after worker failure: {dataset_path}"
                                        )
                                        initialize_dataset(
                                            dataset_path,
                                            shape=case.shape,
                                            dtype=dtype,
                                            chunk_size=chunk_size,
                                            block_size=block_size,
                                        )

    summary_rows = summarize_rows(rows)
    print_summary(summary_rows)
    final_summary = build_final_summary(rows, summary_rows)
    print_final_summary(final_summary)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "bench_io_blosc2_layouts.csv"
    json_path = args.results_dir / "bench_io_blosc2_layouts.json"
    summary_json_path = args.results_dir / "bench_io_blosc2_layouts_summary.json"
    write_rows_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "tiers": args.tiers,
                    "runs": args.runs,
                    "patch_reads": args.patch_reads,
                    "patch_writes": args.patch_writes,
                    "cache_mode": args.cache_mode,
                    "nthreads": args.nthreads,
                    "isolate_runs": args.isolate_runs,
                    "worker_timeout_seconds": args.worker_timeout_seconds,
                    "dtype": str(dtype),
                    "max_elements": args.max_elements,
                    "seed": args.seed,
                },
                "rows": rows,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "tiers": args.tiers,
                    "runs": args.runs,
                    "patch_reads": args.patch_reads,
                    "patch_writes": args.patch_writes,
                    "cache_mode": args.cache_mode,
                    "nthreads": args.nthreads,
                    "isolate_runs": args.isolate_runs,
                    "worker_timeout_seconds": args.worker_timeout_seconds,
                    "dtype": str(dtype),
                    "max_elements": args.max_elements,
                    "seed": args.seed,
                },
                "aggregated_rows": summary_rows,
                "final_summary": final_summary,
            },
            f,
            indent=2,
        )
    print(f"\nWrote CSV:  {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
