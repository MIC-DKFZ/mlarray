"""nnU-Net's own layout function, verbatim, as the reference for MLArray's layout.

Copied from nnunetv2/training/dataloading/nnunet_dataset.py (MIC-DKFZ nnU-Net
master, commit ded2aa3a). Only used by the tests.
"""

import math
from typing import Optional, Sequence, Tuple


def _nnunet_comp_blosc2_params(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    bytes_per_pixel: int = 4,
    max_block_nbytes: int = 128 * 1024,
    max_chunk_nbytes: int = 6 * 1024**2,
    max_chunk_to_patch_ratio_per_axis: Optional[float] = 1.5,
    grow_singleton_patch_axes: bool = False,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Compute Blosc2 block and chunk shapes for 4D arrays with layout:

        image_size = (c, x, y, z)

    patch_size is spatial only:

        patch_size = (x, y, z) for 3D patches
        patch_size = (y, z)    for 2D patches, internally promoted to (1, y, z)

    The channel axis is always kept at 1 for blocks and chunks.

    max_block_nbytes and max_chunk_nbytes are hard caps on the logical useful
    payload, not on any internally padded Blosc2 extent.

    Block and chunk shapes are chosen to roughly follow the image aspect ratio,
    subject to:
    - the patch-size-based per-axis chunk limit
    - the block byte cap
    - the chunk byte cap
    """

    def ceil_pow2(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def prev_pow2_below(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << ((n - 1).bit_length() - 1)

    def nbytes(shape: Sequence[int]) -> int:
        return math.prod(int(i) for i in shape) * int(bytes_per_pixel)

    if len(image_size) != 4:
        raise ValueError(f"image_size must be 4D, i.e. (c, x, y, z). Got {image_size}.")

    if len(patch_size) not in (2, 3):
        raise ValueError(f"patch_size must be 2D or 3D spatial shape. Got {patch_size}.")

    if bytes_per_pixel <= 0:
        raise ValueError("bytes_per_pixel must be positive.")

    if max_block_nbytes <= 0:
        raise ValueError("max_block_nbytes must be positive.")

    if max_chunk_nbytes <= 0:
        raise ValueError("max_chunk_nbytes must be positive.")

    if (
        max_chunk_to_patch_ratio_per_axis is not None
        and max_chunk_to_patch_ratio_per_axis < 1.0
    ):
        raise ValueError("max_chunk_to_patch_ratio_per_axis must be >= 1.0 or None.")

    image_size = tuple(int(i) for i in image_size)

    if any(i <= 0 for i in image_size):
        raise ValueError(f"All image_size entries must be positive. Got {image_size}.")

    is_2d_patch = len(patch_size) == 2

    if is_2d_patch:
        patch_spatial = (1, int(patch_size[0]), int(patch_size[1]))
    else:
        patch_spatial = tuple(int(i) for i in patch_size)

    if any(i <= 0 for i in patch_spatial):
        raise ValueError(f"All patch_size entries must be positive. Got {patch_size}.")

    image_spatial = image_size[1:]

    if max_chunk_to_patch_ratio_per_axis is None:
        target_spatial = image_spatial
    else:
        target_spatial = tuple(
            min(
                image_spatial[ax],
                int(math.floor(max_chunk_to_patch_ratio_per_axis * patch_spatial[ax])),
            )
            for ax in range(3)
        )

    target_spatial = tuple(max(1, int(i)) for i in target_spatial)

    # --------------------
    # Block shape
    # --------------------
    block = [1]
    for p, img in zip(patch_spatial, image_spatial):
        block.append(min(ceil_pow2(p), img))

    effective_block_cap = min(max_block_nbytes, max_chunk_nbytes)

    while nbytes(block) > effective_block_cap:
        candidate_axes = [ax for ax in range(3) if block[ax + 1] > 1]

        if not candidate_axes:
            raise ValueError(
                f"Cannot fit minimum block {tuple(block)} into "
                f"{effective_block_cap} bytes with bytes_per_pixel={bytes_per_pixel}."
            )

        # Shrink the most over-represented axis relative to the target extent.
        # Tie-break toward later axes to preserve the old isotropic default:
        # (1, 64, 64, 32) for 128^3 patches with a 512 KiB block cap.
        picked_axis = max(
            candidate_axes,
            key=lambda ax: (block[ax + 1] / target_spatial[ax], ax),
        )

        block[picked_axis + 1] = max(1, prev_pow2_below(block[picked_axis + 1]))

    # --------------------
    # Chunk shape
    # --------------------
    chunk = block.copy()

    while True:
        candidates = []

        for ax in range(3):
            if is_2d_patch and ax == 0:
                continue

            if patch_spatial[ax] == 1 and not grow_singleton_patch_axes:
                continue

            if chunk[ax + 1] >= image_spatial[ax]:
                continue

            if chunk[ax + 1] >= target_spatial[ax]:
                continue

            candidate = chunk.copy()

            # Grow by one block step, but allow clamping to image boundary
            # or to the target extent.
            candidate[ax + 1] = min(
                candidate[ax + 1] + block[ax + 1],
                image_spatial[ax],
                target_spatial[ax],
            )

            if candidate[ax + 1] == chunk[ax + 1]:
                continue

            candidate_nbytes = nbytes(candidate)

            # Hard cap on logical useful chunk payload.
            if candidate_nbytes > max_chunk_nbytes:
                continue

            # Grow the most under-covered axis relative to the target extent.
            # Tie-break toward the larger original image axis, so for
            # image_spatial=(123, 423, 645), z grows before y when both are tied.
            coverage_ratio = chunk[ax + 1] / target_spatial[ax]

            candidates.append(
                (
                    coverage_ratio,
                    -image_spatial[ax],
                    ax,
                    -candidate_nbytes,
                    candidate,
                )
            )

        if not candidates:
            break

        _, _, _, _, chunk = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )

    return tuple(int(i) for i in block), tuple(int(i) for i in chunk)
