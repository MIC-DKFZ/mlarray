from __future__ import annotations

import math
from typing import Optional

import numpy as np


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


def comp_blosc2_params_generalized(
    image_size: tuple[int, ...],
    patch_size: tuple[int, ...],
    spatial_axis_mask: Optional[list[bool]] = None,
    bytes_per_pixel: int = 4,
    l1_cache_size_per_core_in_bytes: int = 32768,
    l3_cache_size_per_core_in_bytes: int = 1441792,
    safety_factor: float = 0.8,
) -> tuple[list[int], list[int]]:
    if not (len(patch_size) == 2 or len(patch_size) == 3):
        raise RuntimeError("Patch size must be 2D or 3D.")

    image_size = tuple(int(v) for v in image_size)
    if any(v <= 0 for v in image_size):
        raise RuntimeError("Image size values must be positive.")

    if spatial_axis_mask is None:
        if len(image_size) != len(patch_size):
            raise RuntimeError(
                "When spatial_axis_mask is omitted, patch_size dimensionality "
                "must match image_size dimensionality."
            )
        spatial_axis_mask = [True] * len(image_size)
    else:
        if len(spatial_axis_mask) != len(image_size):
            raise RuntimeError("spatial_axis_mask length must match image_size length.")

    spatial_indices = [i for i, is_spatial in enumerate(spatial_axis_mask) if is_spatial]
    non_spatial_indices = [i for i, is_spatial in enumerate(spatial_axis_mask) if not is_spatial]
    if len(spatial_indices) != len(patch_size):
        raise RuntimeError(
            "Patch size dimensionality must match the number of spatial axes."
        )

    ordered_indices = [*non_spatial_indices, *spatial_indices]
    ordered_image = np.array([image_size[i] for i in ordered_indices], dtype=np.int64)
    first_spatial_local = len(non_spatial_indices)
    spatial_sizes = [int(ordered_image[i]) for i in range(first_spatial_local, len(ordered_indices))]

    non_spatial_factor = int(
        np.prod([int(ordered_image[i]) for i in range(first_spatial_local)], dtype=np.int64)
    )
    non_spatial_factor = max(non_spatial_factor, 1)

    l1_target_bytes = int(l1_cache_size_per_core_in_bytes * safety_factor)
    l3_target_bytes = int(l3_cache_size_per_core_in_bytes * safety_factor)
    if l1_target_bytes <= 0 or l3_target_bytes <= 0:
        raise RuntimeError("Cache targets must be positive after applying safety_factor.")

    block_spatial_limit = l1_target_bytes // (bytes_per_pixel * non_spatial_factor)
    if block_spatial_limit < 1:
        raise RuntimeError(
            "Cannot fit any block while fully stretching non-spatial axes into L1 target."
        )

    def _pref_score(values: tuple[int, ...], pref: tuple[int, ...]) -> float:
        score = 0.0
        for v, p in zip(values, pref):
            pp = max(int(p), 1)
            score += abs(math.log(float(v) / float(pp)))
        return score

    def _maximize_product_under_limit(
        bounds: list[int],
        limit: int,
        pref: Optional[tuple[int, ...]] = None,
    ) -> tuple[int, ...]:
        if not bounds:
            return tuple()
        if limit < 1:
            raise RuntimeError("No feasible solution under product limit < 1.")

        d = len(bounds)
        for b in bounds:
            if b < 1:
                raise RuntimeError("Axis bound must be >= 1.")

        best_prod = 0
        best_vals: Optional[tuple[int, ...]] = None
        best_score = float("inf")

        if d == 1:
            v0 = min(bounds[0], limit)
            return (int(v0),)

        if d == 2:
            b0, b1 = bounds
            for v0 in range(1, b0 + 1):
                max_v1 = min(b1, limit // v0)
                if max_v1 < 1:
                    continue
                cand = (v0, max_v1)
                prod = v0 * max_v1
                score = _pref_score(cand, pref) if pref is not None else 0.0
                if (
                    prod > best_prod
                    or (prod == best_prod and score < best_score)
                    or (
                        prod == best_prod
                        and abs(score - best_score) < 1e-12
                        and (best_vals is None or cand > best_vals)
                    )
                ):
                    best_prod = prod
                    best_vals = cand
                    best_score = score
        elif d == 3:
            b0, b1, b2 = bounds
            for v0 in range(1, b0 + 1):
                lim01 = limit // v0
                if lim01 < 1:
                    continue
                for v1 in range(1, b1 + 1):
                    max_v2 = min(b2, lim01 // v1)
                    if max_v2 < 1:
                        continue
                    cand = (v0, v1, max_v2)
                    prod = v0 * v1 * max_v2
                    score = _pref_score(cand, pref) if pref is not None else 0.0
                    if (
                        prod > best_prod
                        or (prod == best_prod and score < best_score)
                        or (
                            prod == best_prod
                            and abs(score - best_score) < 1e-12
                            and (best_vals is None or cand > best_vals)
                        )
                    ):
                        best_prod = prod
                        best_vals = cand
                        best_score = score
        else:
            raise RuntimeError("Only up to 3 spatial dimensions are supported.")

        if best_vals is None:
            raise RuntimeError("No feasible layout found under cache constraint.")
        return best_vals

    block_pref = tuple(
        min(spatial_sizes[i], max(1, int(patch_size[i])))
        for i in range(len(spatial_sizes))
    )
    block_spatial = _maximize_product_under_limit(
        bounds=spatial_sizes,
        limit=int(block_spatial_limit),
        pref=block_pref,
    )
    block_spatial_prod = int(np.prod(block_spatial, dtype=np.int64))
    block_size = ordered_image.copy()
    for i, v in enumerate(block_spatial, start=first_spatial_local):
        block_size[i] = v
    block_bytes = int(np.prod(block_size, dtype=np.int64)) * bytes_per_pixel
    if block_bytes > l1_target_bytes:
        raise RuntimeError("Internal error: computed block exceeds L1 target.")

    chunk_spatial_limit = l3_target_bytes // (bytes_per_pixel * non_spatial_factor)
    if chunk_spatial_limit < block_spatial_prod:
        raise RuntimeError(
            "Cannot fit a chunk (>= block) into L3 target with fully stretched non-spatial axes."
        )

    q_limit = chunk_spatial_limit // block_spatial_prod
    q_bounds = [
        max(1, int(spatial_sizes[i] // max(1, int(block_spatial[i]))))
        for i in range(len(spatial_sizes))
    ]
    chunk_pref = tuple(
        max(
            1,
            min(
                q_bounds[i],
                int(math.ceil(float(max(1, int(patch_size[i]))) / float(max(1, int(block_spatial[i]))))),
            ),
        )
        for i in range(len(spatial_sizes))
    )
    q_factors = _maximize_product_under_limit(
        bounds=q_bounds,
        limit=max(1, int(q_limit)),
        pref=chunk_pref,
    )
    chunk_spatial = tuple(
        int(block_spatial[i]) * int(q_factors[i]) for i in range(len(spatial_sizes))
    )
    chunk_size = ordered_image.copy()
    for i, v in enumerate(chunk_spatial, start=first_spatial_local):
        chunk_size[i] = v
    chunk_bytes = int(np.prod(chunk_size, dtype=np.int64)) * bytes_per_pixel
    if chunk_bytes > l3_target_bytes:
        raise RuntimeError("Internal error: computed chunk exceeds L3 target.")

    block_out = np.zeros(len(image_size), dtype=np.int64)
    chunk_out = np.zeros(len(image_size), dtype=np.int64)
    for local_idx, original_idx in enumerate(ordered_indices):
        block_out[original_idx] = block_size[local_idx]
        chunk_out[original_idx] = chunk_size[local_idx]

    return [int(v) for v in chunk_out.tolist()], [int(v) for v in block_out.tolist()]


def comp_blosc2_params_spatial_only(
    image_size: tuple[int, ...],
    patch_size: tuple[int, ...],
    spatial_axis_mask: Optional[list[bool]] = None,
    bytes_per_pixel: int = 4,
    l1_cache_size_per_core_in_bytes: int = 32768,  # 32768
    l3_cache_size_per_core_in_bytes: int = 1441792,  # 1441792
    safety_factor: float = 0.9,
) -> tuple[list[int], list[int]]:
    image_size = tuple(int(v) for v in image_size)
    if any(v <= 0 for v in image_size):
        raise RuntimeError("Image size values must be positive.")

    if spatial_axis_mask is None:
        if len(image_size) != len(patch_size):
            raise RuntimeError(
                "When spatial_axis_mask is omitted, patch_size dimensionality "
                "must match image_size dimensionality."
            )
        spatial_axis_mask = [True] * len(image_size)
    else:
        if len(spatial_axis_mask) != len(image_size):
            raise RuntimeError("spatial_axis_mask length must match image_size length.")

    spatial_indices = [i for i, is_spatial in enumerate(spatial_axis_mask) if is_spatial]
    non_spatial_indices = [i for i, is_spatial in enumerate(spatial_axis_mask) if not is_spatial]
    if len(spatial_indices) == 0:
        raise RuntimeError("At least one spatial axis is required.")
    if len(spatial_indices) != len(patch_size):
        raise RuntimeError(
            "Patch size dimensionality must match the number of spatial axes."
        )
    if bytes_per_pixel <= 0:
        raise RuntimeError("bytes_per_pixel must be positive.")

    spatial_upper = [int(image_size[i]) for i in spatial_indices]
    d = len(spatial_upper)
    last_spatial_local = d - 1

    l1_target_bytes = int(l1_cache_size_per_core_in_bytes * safety_factor)
    l3_target_bytes = int(l3_cache_size_per_core_in_bytes * safety_factor)
    if l1_target_bytes <= 0 or l3_target_bytes <= 0:
        raise RuntimeError("Cache targets must be positive after applying safety_factor.")

    block_limit_elems = l1_target_bytes // bytes_per_pixel
    chunk_limit_elems = l3_target_bytes // bytes_per_pixel
    if block_limit_elems < 1:
        raise RuntimeError("Cannot fit any block into L1 target.")
    if chunk_limit_elems < 1:
        raise RuntimeError("Cannot fit any chunk into L3 target.")

    def _prod(values: list[int]) -> int:
        p = 1
        for v in values:
            p *= int(v)
        return int(p)

    def _grow_isotropic(
        lower: list[int],
        upper: list[int],
        limit_elems: int,
        *,
        last_axis: int,
    ) -> list[int]:
        sizes = [int(v) for v in lower]
        for lo, hi in zip(sizes, upper):
            if lo < 1 or hi < 1 or lo > hi:
                raise RuntimeError("Invalid lower/upper bounds for spatial growth.")
        if _prod(sizes) > limit_elems:
            raise RuntimeError("Lower bound already exceeds cache target.")

        # Isotropy-first growth: repeatedly lift the currently smallest axes.
        while True:
            growable = [i for i in range(len(sizes)) if sizes[i] < upper[i]]
            if not growable:
                break
            smallest = min(sizes[i] for i in growable)
            lift_axes = [i for i in growable if sizes[i] == smallest]
            candidate = sizes.copy()
            for i in lift_axes:
                candidate[i] += 1
            if _prod(candidate) <= limit_elems:
                sizes = candidate
                continue
            break

        # Final boundary approach rule:
        # if the next isotropic step would exceed, grow the last spatial axis.
        other_prod = 1
        for i, v in enumerate(sizes):
            if i == last_axis:
                continue
            other_prod *= int(v)
        max_last = min(int(upper[last_axis]), int(limit_elems // max(other_prod, 1)))
        if max_last > sizes[last_axis]:
            # Prefer even sizes when possible, otherwise use the maximum.
            if max_last % 2 == 0:
                pick = max_last
            else:
                even_pick = max_last - 1
                pick = even_pick if even_pick >= sizes[last_axis] else max_last
            sizes[last_axis] = int(pick)

        if _prod(sizes) > limit_elems:
            raise RuntimeError("Internal error: constructed layout exceeds cache target.")
        return sizes

    block_spatial = _grow_isotropic(
        lower=[1] * d,
        upper=spatial_upper,
        limit_elems=int(block_limit_elems),
        last_axis=last_spatial_local,
    )
    block_elems = _prod(block_spatial)
    if block_elems < 1:
        raise RuntimeError("Failed to construct a valid block.")
    if int(chunk_limit_elems) < int(block_elems):
        raise RuntimeError("Cannot fit a chunk (>= block) into L3 target.")

    def _chunk_from_multipliers(multipliers: list[int]) -> list[int]:
        out: list[int] = []
        for i in range(d):
            raw = int(block_spatial[i]) * int(multipliers[i])
            out.append(min(int(spatial_upper[i]), raw))
        return out

    def _grow_chunk_block_multiple(
        *,
        limit_elems: int,
        last_axis: int,
    ) -> list[int]:
        multipliers = [1] * d
        chunk_sizes = _chunk_from_multipliers(multipliers)
        if _prod(chunk_sizes) > limit_elems:
            raise RuntimeError("Block already exceeds chunk cache target.")

        # Isotropy-first growth in block-sized steps.
        while True:
            growable = [i for i in range(d) if chunk_sizes[i] < spatial_upper[i]]
            if not growable:
                break
            smallest = min(chunk_sizes[i] for i in growable)
            lift_axes = [i for i in growable if chunk_sizes[i] == smallest]
            candidate_multipliers = multipliers.copy()
            for i in lift_axes:
                candidate_multipliers[i] += 1
            candidate_chunk = _chunk_from_multipliers(candidate_multipliers)
            if _prod(candidate_chunk) <= limit_elems:
                multipliers = candidate_multipliers
                chunk_sizes = candidate_chunk
                continue
            break

        # Final boundary approach rule:
        # if next isotropic step would exceed, grow the last spatial axis.
        other_prod = 1
        for i, v in enumerate(chunk_sizes):
            if i == last_axis:
                continue
            other_prod *= int(v)
        max_last = min(
            int(spatial_upper[last_axis]),
            int(limit_elems // max(other_prod, 1)),
        )
        if max_last > chunk_sizes[last_axis]:
            block_last = int(block_spatial[last_axis])
            # Prefer exact block-multiple. Allow non-multiple only when clipped at bound.
            if max_last >= int(spatial_upper[last_axis]):
                target_last = int(spatial_upper[last_axis])
            else:
                target_last = int((max_last // max(block_last, 1)) * block_last)
                if target_last < chunk_sizes[last_axis]:
                    target_last = chunk_sizes[last_axis]

            if target_last > chunk_sizes[last_axis]:
                if target_last == int(spatial_upper[last_axis]):
                    multipliers[last_axis] = int(math.ceil(target_last / max(block_last, 1)))
                else:
                    multipliers[last_axis] = int(target_last // max(block_last, 1))
                chunk_sizes = _chunk_from_multipliers(multipliers)

        if _prod(chunk_sizes) > limit_elems:
            raise RuntimeError("Internal error: constructed chunk exceeds L3 target.")
        return chunk_sizes

    chunk_spatial = _grow_chunk_block_multiple(
        limit_elems=int(chunk_limit_elems),
        last_axis=last_spatial_local,
    )

    block_out = [1] * len(image_size)
    chunk_out = [1] * len(image_size)
    for local_i, axis_i in enumerate(spatial_indices):
        block_out[axis_i] = int(block_spatial[local_i])
        chunk_out[axis_i] = int(chunk_spatial[local_i])

    for axis_i in non_spatial_indices:
        block_out[axis_i] = 1
        chunk_out[axis_i] = 1

    if int(np.prod(block_out, dtype=np.int64)) * bytes_per_pixel > l1_target_bytes:
        raise RuntimeError("Internal error: computed block exceeds L1 target.")
    if int(np.prod(chunk_out, dtype=np.int64)) * bytes_per_pixel > l3_target_bytes:
        raise RuntimeError("Internal error: computed chunk exceeds L3 target.")
    return chunk_out, block_out


def comp_blosc2_params_spatial_only_magnitude(
    image_size: tuple[int, ...],
    patch_size: tuple[int, ...],
    spatial_axis_mask: Optional[list[bool]] = None,
    bytes_per_pixel: int = 4,
    l1_cache_size_per_core_in_bytes: int = 32768,  # 32768, 65536
    l3_cache_size_per_core_in_bytes: int = 1441792,  # 1441792, 5592405
    safety_factor: float = 0.8,
) -> tuple[list[int], list[int]]:
    image_size = tuple(int(v) for v in image_size)
    if any(v <= 0 for v in image_size):
        raise RuntimeError("Image size values must be positive.")

    if spatial_axis_mask is None:
        if len(image_size) != len(patch_size):
            raise RuntimeError(
                "When spatial_axis_mask is omitted, patch_size dimensionality "
                "must match image_size dimensionality."
            )
        spatial_axis_mask = [True] * len(image_size)
    else:
        if len(spatial_axis_mask) != len(image_size):
            raise RuntimeError("spatial_axis_mask length must match image_size length.")

    spatial_indices = [i for i, is_spatial in enumerate(spatial_axis_mask) if is_spatial]
    non_spatial_indices = [i for i, is_spatial in enumerate(spatial_axis_mask) if not is_spatial]
    if len(spatial_indices) == 0:
        raise RuntimeError("At least one spatial axis is required.")
    if len(spatial_indices) != len(patch_size):
        raise RuntimeError(
            "Patch size dimensionality must match the number of spatial axes."
        )
    if bytes_per_pixel <= 0:
        raise RuntimeError("bytes_per_pixel must be positive.")

    spatial_upper = [int(image_size[i]) for i in spatial_indices]
    patch_vals = [max(1, int(v)) for v in patch_size]
    min_patch = min(patch_vals)
    magnitudes = [float(v) / float(min_patch) for v in patch_vals]

    l1_target_bytes = int(l1_cache_size_per_core_in_bytes * safety_factor)
    l3_target_bytes = int(l3_cache_size_per_core_in_bytes * safety_factor)
    if l1_target_bytes <= 0 or l3_target_bytes <= 0:
        raise RuntimeError("Cache targets must be positive after applying safety_factor.")

    block_limit_elems = l1_target_bytes // bytes_per_pixel
    chunk_limit_elems = l3_target_bytes // bytes_per_pixel
    if block_limit_elems < 1:
        raise RuntimeError("Cannot fit any block into L1 target.")
    if chunk_limit_elems < 1:
        raise RuntimeError("Cannot fit any chunk into L3 target.")

    d = len(spatial_upper)

    def _prod(values: list[int]) -> int:
        p = 1
        for v in values:
            p *= int(v)
        return int(p)

    def _is_clipped(v: int, upper: int) -> bool:
        return int(v) == int(upper)

    def _ensure_even_or_clipped(v: int, upper: int) -> Optional[int]:
        v = int(v)
        upper = int(upper)
        if v > upper:
            return None
        if _is_clipped(v, upper):
            return v
        if v % 2 == 0:
            return v
        cand = v + 1
        if cand <= upper:
            return cand
        cand = v - 1
        if cand >= 1:
            return cand
        return None

    # Weighted deterministic growth step per axis.
    block_axis_steps = [max(1, int(round(m))) for m in magnitudes]
    block_axis_steps = [2 * s for s in block_axis_steps]  # prefer even increments

    # Start with minimum even (or clipped) block sizes.
    block_spatial: list[int] = []
    for upper in spatial_upper:
        if upper == 1:
            block_spatial.append(1)
        else:
            block_spatial.append(2)
    if _prod(block_spatial) > block_limit_elems:
        raise RuntimeError("Cannot fit any even non-clipped block into L1 target.")

    # Grow block sizes: best cache fill ratio wins; tie -> axis from last to first.
    while True:
        best_axis = None
        best_candidate = None
        best_prod = -1

        for axis in range(d - 1, -1, -1):
            cur = int(block_spatial[axis])
            upper = int(spatial_upper[axis])
            if cur >= upper:
                continue
            target = cur + int(block_axis_steps[axis])
            if target >= upper:
                cand_axis = upper
            else:
                cand_axis = target
                fixed = _ensure_even_or_clipped(cand_axis, upper)
                if fixed is None or fixed <= cur:
                    continue
                cand_axis = fixed

            cand = block_spatial.copy()
            cand[axis] = int(cand_axis)
            p = _prod(cand)
            if p > int(block_limit_elems):
                continue
            if p > best_prod:
                best_prod = p
                best_axis = axis
                best_candidate = cand

        if best_axis is None or best_candidate is None:
            break
        block_spatial = best_candidate

    block_elems = _prod(block_spatial)
    if block_elems < 1:
        raise RuntimeError("Failed to construct a valid block.")
    if block_elems > int(block_limit_elems):
        raise RuntimeError("Internal error: block exceeds L1 target.")
    if int(chunk_limit_elems) < int(block_elems):
        raise RuntimeError("Cannot fit a chunk (>= block) into L3 target.")

    # Chunk growth in block multiples, magnitude-weighted.
    chunk_spatial = block_spatial.copy()
    chunk_axis_stop = [
        bool(chunk_spatial[i] > (1.5 * float(patch_vals[i])))
        for i in range(d)
    ]
    chunk_mul_steps = [max(1, int(round(m))) for m in magnitudes]

    while True:
        best_axis = None
        best_candidate = None
        best_prod = -1

        for axis in range(d - 1, -1, -1):
            if chunk_axis_stop[axis]:
                continue

            cur = int(chunk_spatial[axis])
            upper = int(spatial_upper[axis])
            if cur >= upper:
                continue

            step_mul = int(chunk_mul_steps[axis])
            raw = cur + step_mul * int(block_spatial[axis])
            if raw >= upper:
                cand_axis = upper  # clipped; multiple and even constraints relaxed
            else:
                # Must be multiple of block on non-clipped axes.
                cand_axis = raw
                if cand_axis % int(block_spatial[axis]) != 0:
                    cand_axis = (
                        (cand_axis // int(block_spatial[axis]))
                        * int(block_spatial[axis])
                    )
                if cand_axis <= cur:
                    cand_axis = cur + int(block_spatial[axis])
                if cand_axis >= upper:
                    cand_axis = upper
                if cand_axis < upper:
                    # Evenness rule on non-clipped axes.
                    if cand_axis % 2 != 0:
                        up = cand_axis + int(block_spatial[axis])
                        down = cand_axis - int(block_spatial[axis])
                        if up < upper and (up % 2 == 0):
                            cand_axis = up
                        elif down > cur and (down % 2 == 0):
                            cand_axis = down
                        else:
                            continue
                    if cand_axis % int(block_spatial[axis]) != 0:
                        continue

            if cand_axis <= cur:
                continue

            cand = chunk_spatial.copy()
            cand[axis] = int(cand_axis)
            p = _prod(cand)
            if p > int(chunk_limit_elems):
                continue
            if p > best_prod:
                best_prod = p
                best_axis = axis
                best_candidate = cand

        if best_axis is None or best_candidate is None:
            break

        chunk_spatial = best_candidate
        if chunk_spatial[best_axis] > (1.5 * float(patch_vals[best_axis])):
            chunk_axis_stop[best_axis] = True

    # Build full outputs.
    block_out = [1] * len(image_size)
    chunk_out = [1] * len(image_size)
    for local_i, axis_i in enumerate(spatial_indices):
        block_out[axis_i] = int(block_spatial[local_i])
        chunk_out[axis_i] = int(chunk_spatial[local_i])
    for axis_i in non_spatial_indices:
        block_out[axis_i] = 1
        chunk_out[axis_i] = 1

    # Final hard checks.
    if int(np.prod(block_out, dtype=np.int64)) * bytes_per_pixel > l1_target_bytes:
        raise RuntimeError("Internal error: computed block exceeds L1 target.")
    if int(np.prod(chunk_out, dtype=np.int64)) * bytes_per_pixel > l3_target_bytes:
        raise RuntimeError("Internal error: computed chunk exceeds L3 target.")

    for axis_i in spatial_indices:
        if int(chunk_out[axis_i]) < int(block_out[axis_i]):
            raise RuntimeError("Internal error: chunk is smaller than block on a spatial axis.")
        upper = int(image_size[axis_i])
        b = int(block_out[axis_i])
        c = int(chunk_out[axis_i])
        if b != upper and (b % 2 != 0):
            raise RuntimeError("Internal error: block axis must be even unless clipped.")
        if c != upper and (c % 2 != 0):
            raise RuntimeError("Internal error: chunk axis must be even unless clipped.")
        if c != upper and (c % max(1, b) != 0):
            raise RuntimeError(
                "Internal error: chunk axis must be multiple of block axis unless clipped."
            )

    return chunk_out, block_out


__all__ = [
    "comp_blosc2_params_baseline",
    "comp_blosc2_params_generalized",
    "comp_blosc2_params_spatial_only",
    "comp_blosc2_params_spatial_only_magnitude",
]
