"""Blosc2 storage layout: chunk and block sizes, and the compression filter.

MLArray uses nnU-Net's layout heuristic (``comp_blosc2_params`` in
``nnunetv2/training/dataloading/nnunet_dataset.py``) generalized to any number
of spatial and non-spatial axes, and nnU-Net's per-array filter choice. For 2
and 3 spatial axes the layout is identical to nnU-Net's.

Why this layout: in a benchmark of nnU-Net-style patch loading (12 loader
processes, network storage, data not in the page cache), nnU-Net's files read
with blosc2's read-ahead were as fast as every alternative layout we tested
(smaller chunk caps, other codec levels and read modes all within 2 points),
and they write fastest. See ``docs/optimization.md``.
"""

import math
from typing import Optional, Sequence
from warnings import warn

import blosc2
import numpy as np

DEFAULT_CODEC = blosc2.Codec.LZ4HC
DEFAULT_CLEVEL = 5
DEFAULT_COMPRESS_NTHREADS = 4
DEFAULT_MAX_BLOCK_NBYTES = 128 * 1024
DEFAULT_MAX_CHUNK_NBYTES = 6 * 1024**2
DEFAULT_MAX_CHUNK_TO_PATCH_RATIO = 1.5


def comp_blosc2_params(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    spatial_axis_mask: Optional[Sequence[bool]] = None,
    bytes_per_pixel: int = 4,
    max_block_nbytes: int = DEFAULT_MAX_BLOCK_NBYTES,
    max_chunk_nbytes: int = DEFAULT_MAX_CHUNK_NBYTES,
    max_chunk_to_patch_ratio_per_axis: Optional[float] = DEFAULT_MAX_CHUNK_TO_PATCH_RATIO,
    grow_singleton_patch_axes: bool = False,
) -> tuple[list[int], list[int]]:
    """Chunk and block sizes for patch-wise reading (nnU-Net's heuristic).

    * Non-spatial axes (channels, time, ...) get chunk and block size 1, so
      reading one channel touches only its own data.
    * Blocks start at the next power of two of the patch per spatial axis and
      are halved on the axis most over-represented relative to its target
      extent until they fit ``max_block_nbytes``.
    * Chunks grow by whole blocks on the least covered spatial axis until
      ``max_chunk_nbytes``, the image extent, or
      ``max_chunk_to_patch_ratio_per_axis`` times the patch per axis.
    * A patch entry of 1 (e.g. ``(1, 256, 256)`` for 2D slices of a 3D image)
      keeps that axis at 1 unless ``grow_singleton_patch_axes`` is set.

    For 2 and 3 spatial axes the result equals nnU-Net's ``comp_blosc2_params``
    (2D images as ``(1, y, x)``); any other number of spatial axes follows the
    same rules.

    Args:
        image_size: Full array shape.
        patch_size: Patch size over the spatial axes (one entry per spatial axis).
        spatial_axis_mask: For every array axis whether it is spatial. All
            axes are spatial if omitted.
        bytes_per_pixel: Bytes per array element.
        max_block_nbytes: Largest block in bytes.
        max_chunk_nbytes: Largest chunk in bytes.
        max_chunk_to_patch_ratio_per_axis: Largest chunk extent per axis
            relative to the patch, or None for no limit.
        grow_singleton_patch_axes: Let axes with patch size 1 grow in chunks.

    Returns:
        Chunk size and block size per array axis.
    """
    image_size = tuple(int(v) for v in image_size)
    patch_size = tuple(int(v) for v in patch_size)
    if spatial_axis_mask is None:
        spatial_axis_mask = [True] * len(image_size)
    if len(spatial_axis_mask) != len(image_size):
        raise RuntimeError("spatial_axis_mask length must match image_size length.")
    spatial = [i for i, is_spatial in enumerate(spatial_axis_mask) if is_spatial]
    if not spatial:
        raise RuntimeError("At least one spatial axis is required.")
    if len(patch_size) != len(spatial):
        raise RuntimeError("Patch size must have one entry per spatial axis.")
    if any(p <= 0 for p in patch_size) or any(s <= 0 for s in image_size):
        raise RuntimeError("Image and patch size entries must be positive.")
    if bytes_per_pixel <= 0 or max_block_nbytes <= 0 or max_chunk_nbytes <= 0:
        raise RuntimeError("bytes_per_pixel and the byte caps must be positive.")
    if max_chunk_to_patch_ratio_per_axis is not None and max_chunk_to_patch_ratio_per_axis < 1.0:
        raise RuntimeError("max_chunk_to_patch_ratio_per_axis must be >= 1.0 or None.")

    block, chunk = _spatial_layout(
        [image_size[i] for i in spatial], patch_size, bytes_per_pixel, max_block_nbytes,
        max_chunk_nbytes, max_chunk_to_patch_ratio_per_axis, grow_singleton_patch_axes,
    )
    chunk_size = [1] * len(image_size)
    block_size = [1] * len(image_size)
    for axis, c, b in zip(spatial, chunk, block):
        chunk_size[axis] = int(c)
        block_size[axis] = int(b)
    return chunk_size, block_size


def _spatial_layout(
    image_spatial: Sequence[int],
    patch: Sequence[int],
    bytes_per_pixel: int,
    max_block_nbytes: int,
    max_chunk_nbytes: int,
    max_chunk_to_patch_ratio_per_axis: Optional[float],
    grow_singleton_patch_axes: bool,
) -> tuple[list[int], list[int]]:
    """Block and chunk over the spatial axes only (see ``comp_blosc2_params``)."""

    def ceil_pow2(n: int) -> int:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()

    def prev_pow2_below(n: int) -> int:
        return 1 if n <= 1 else 1 << ((n - 1).bit_length() - 1)

    def nbytes(shape: Sequence[int]) -> int:
        return math.prod(int(i) for i in shape) * int(bytes_per_pixel)

    n = len(image_spatial)
    if max_chunk_to_patch_ratio_per_axis is None:
        target = list(image_spatial)
    else:
        target = [min(image_spatial[ax], int(math.floor(max_chunk_to_patch_ratio_per_axis * patch[ax])))
                  for ax in range(n)]
    target = [max(1, int(t)) for t in target]

    block = [min(ceil_pow2(p), img) for p, img in zip(patch, image_spatial)]
    effective_block_cap = min(max_block_nbytes, max_chunk_nbytes)
    while nbytes(block) > effective_block_cap:
        axes = [ax for ax in range(n) if block[ax] > 1]
        if not axes:
            raise ValueError(f"Cannot fit minimum block {tuple(block)} into {effective_block_cap} bytes "
                             f"with bytes_per_pixel={bytes_per_pixel}.")
        # Shrink the most over-represented axis; ties go to later axes.
        ax = max(axes, key=lambda a: (block[a] / target[a], a))
        block[ax] = max(1, prev_pow2_below(block[ax]))

    chunk = block.copy()
    while True:
        candidates = []
        for ax in range(n):
            if patch[ax] == 1 and not grow_singleton_patch_axes:
                continue
            if chunk[ax] >= image_spatial[ax] or chunk[ax] >= target[ax]:
                continue
            grown = chunk.copy()
            grown[ax] = min(grown[ax] + block[ax], image_spatial[ax], target[ax])
            if grown[ax] == chunk[ax] or nbytes(grown) > max_chunk_nbytes:
                continue
            # Grow the most under-covered axis; ties go to the larger image axis.
            candidates.append((chunk[ax] / target[ax], -image_spatial[ax], ax, -nbytes(grown), grown))
        if not candidates:
            break
        chunk = min(candidates, key=lambda c: c[:4])[4]
    return block, chunk


def select_blosc2_filter(
    arr: np.ndarray,
    chunk_size: Sequence[int],
    block_size: Sequence[int],
    codec: blosc2.Codec = DEFAULT_CODEC,
    clevel: int = DEFAULT_CLEVEL,
) -> blosc2.Filter:
    """nnU-Net's per-array filter choice: NOFILTER or SHUFFLE, whichever
    compresses a centered, chunk-sized slab better. Ties and failures give
    NOFILTER.

    The better filter depends on the data: interpolated (resampled)
    intensities favour SHUFFLE, quantized (not resampled) ones NOFILTER.
    """
    try:
        shape = tuple(int(s) for s in arr.shape)
        # A centered slab of at most one chunk: cheap and usually on foreground.
        slab_shape = [min(int(c), s) for c, s in zip(chunk_size, shape)]
        slices = tuple(slice((s - ss) // 2, (s - ss) // 2 + ss) for s, ss in zip(shape, slab_shape))
        slab = np.ascontiguousarray(arr[slices])
        trial_blocks = tuple(max(1, min(int(b), ss)) for b, ss in zip(block_size, slab_shape))

        best_filter, best_bytes = blosc2.Filter.NOFILTER, None
        for f in (blosc2.Filter.NOFILTER, blosc2.Filter.SHUFFLE):
            cparams = {"codec": codec, "clevel": clevel, "nthreads": DEFAULT_COMPRESS_NTHREADS, "filters": [f]}
            comp = blosc2.asarray(slab, chunks=tuple(slab_shape), blocks=trial_blocks, cparams=cparams)
            cb = comp.schunk.cbytes
            if best_bytes is None or cb < best_bytes:
                best_bytes, best_filter = cb, f
        return best_filter
    except Exception as e:
        warn(f"Filter selection failed ({e!r}); falling back to NOFILTER.")
        return blosc2.Filter.NOFILTER


__all__ = [
    "DEFAULT_CODEC",
    "DEFAULT_CLEVEL",
    "DEFAULT_COMPRESS_NTHREADS",
    "comp_blosc2_params",
    "select_blosc2_filter",
]
