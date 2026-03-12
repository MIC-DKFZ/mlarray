#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mlarray import MLArray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a directory recursively for .mla files and print image/chunk/block "
            "for the first N files."
        )
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Root directory to scan for .mla files.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of files to inspect (default: 10).",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort file paths before selecting the first N.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_files <= 0:
        raise ValueError("--num-files must be > 0")

    root = args.dir.resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Directory does not exist or is not a directory: {root}")

    mla_files = list(root.rglob("*.mla"))
    if args.sort:
        mla_files = sorted(mla_files)

    if not mla_files:
        print(f"No .mla files found in: {root}")
        return

    selected = mla_files[: args.num_files]
    print(f"Found {len(mla_files)} .mla files. Inspecting {len(selected)} file(s).")

    for idx, path in enumerate(selected, start=1):
        image = None
        try:
            image = MLArray.open(path, mode="r", mmap_mode="r", dparams={"nthreads": 1})
            shape = list(image.shape)
            chunk_size = list(image.meta.blosc2.chunk_size)
            block_size = list(image.meta.blosc2.block_size)
            itemsize = np.dtype(image.dtype).itemsize
            chunk_bytes = int(np.prod(chunk_size)) * itemsize
            block_bytes = int(np.prod(block_size)) * itemsize
            print(
                f"{idx:>3d}. image={shape} "
                f"chunk={chunk_size} chunk_bytes={chunk_bytes} "
                f"block={block_size} block_bytes={block_bytes}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"{idx:>3d}. ERROR path={path} error={type(e).__name__}: {e}")
        finally:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
