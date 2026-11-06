"""Blosc2 storage helpers: compression parameters, layout metadata, file IO.

These functions hold no MLArray instance state; ``MLArray`` uses them for
opening, loading, saving and pickling its Blosc2 store.
"""

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import blosc2
import numpy as np

from mlarray import blosc2_layout_strategies as _layout
from mlarray.meta import MetaBlosc2

MLARRAY_SUFFIX = "mla"
MLARRAY_VERSION = "v0"
MLARRAY_DEFAULT_PATCH_SIZE = 192


def check_suffix(filepath) -> None:
    """Raise unless ``filepath`` ends with the MLArray extension."""
    if not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
        raise RuntimeError(f"MLArray requires '.{MLARRAY_SUFFIX}' as extension.")


# ----------------------------------------------------------------------------
# Compression and decompression parameters
# ----------------------------------------------------------------------------

def default_cparams() -> Dict:
    """MLArray's default compression parameters (nnU-Net's: LZ4HC level 5, 4 threads)."""
    return {
        "codec": _layout.DEFAULT_CODEC,
        "clevel": _layout.DEFAULT_CLEVEL,
        "nthreads": _layout.DEFAULT_COMPRESS_NTHREADS,
    }


def is_default_cparams(cparams) -> bool:
    """True for the default compression parameters without a chosen filter."""
    if cparams is None:
        return True
    if isinstance(cparams, blosc2.CParams) or not hasattr(cparams, "keys"):
        return False
    cparams = dict(cparams)
    default = default_cparams()
    if set(cparams) != set(default):
        return False
    try:
        return (blosc2.Codec(cparams["codec"]) == default["codec"]
                and cparams["clevel"] == default["clevel"]
                and cparams["nthreads"] == default["nthreads"])
    except (ValueError, TypeError):
        return False


def resolve_cparams(cparams: Optional[Union[Dict, blosc2.CParams]]) -> Union[Dict, blosc2.CParams]:
    """Resolve compression params with MLArray defaults."""
    if cparams is None:
        return default_cparams()
    if isinstance(cparams, dict):
        cparams = dict(cparams)
        if "codec" in cparams and not isinstance(cparams["codec"], blosc2.Codec):
            cparams["codec"] = blosc2.Codec(cparams["codec"])
        if "splitmode" in cparams and not isinstance(cparams["splitmode"], blosc2.SplitMode):
            cparams["splitmode"] = blosc2.SplitMode(cparams["splitmode"])
        if "tuner" in cparams and not isinstance(cparams["tuner"], blosc2.Tuner):
            cparams["tuner"] = blosc2.Tuner(cparams["tuner"])
        if "filters" in cparams and isinstance(cparams["filters"], (list, tuple)):
            cparams["filters"] = [
                f if isinstance(f, blosc2.Filter) else blosc2.Filter(f)
                for f in cparams["filters"]
            ]
    return cparams


def portable_cparams(cparams: Union[Dict, blosc2.CParams]) -> Dict:
    """Compression parameters as recorded in the metadata.

    Machine-specific fields are left out so a file does not carry them to
    other machines: ``typesize`` and ``blocksize`` always, and the
    ``nthreads`` of a ``blosc2.CParams`` object, which blosc2 fills with the
    number of cores. Thread counts given in a dict are kept.
    """
    if isinstance(cparams, blosc2.CParams):
        portable = dict(vars(cparams))
        portable.pop("nthreads", None)
    else:
        portable = dict(cparams)
    portable.pop("typesize", None)
    portable.pop("blocksize", None)
    return portable


def resolve_dparams(dparams: Optional[Union[Dict, blosc2.DParams]]) -> Union[Dict, blosc2.DParams]:
    """Resolve decompression params with MLArray defaults."""
    if dparams is None:
        return {"nthreads": 1}
    return dparams


# ----------------------------------------------------------------------------
# Shapes and layout
# ----------------------------------------------------------------------------

def normalize_shape(shape: Union[int, List, Tuple, np.ndarray]) -> Tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(v) for v in shape)


def normalize_patch_size(patch_size, num_spatial_axes: int) -> Optional[List[int]]:
    """Expand an int patch size and validate that all entries are positive ints."""
    if patch_size is None:
        return None
    if isinstance(patch_size, (int, np.integer)) and not isinstance(patch_size, (bool, np.bool_)):
        patch_size = [patch_size] * num_spatial_axes
    elif isinstance(patch_size, (list, tuple, np.ndarray)):
        patch_size = list(patch_size)
    else:
        raise ValueError(f"patch_size must be 'default', None, an int or a sequence of ints, got {patch_size!r}.")
    for value in patch_size:
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError(f"patch_size values must be positive ints, got {patch_size!r}.")
    return [int(value) for value in patch_size]


def comp_blosc2_meta(meta_blosc2, patch_size, chunk_size, block_size, shape, dtype_itemsize, spatial_axis_mask, cparams, dparams, array=None) -> Tuple[MetaBlosc2, Union[Dict, blosc2.CParams]]:
    """Compute and validate Blosc2 chunk/block metadata.

    Args:
        meta_blosc2 (Optional[MetaBlosc2]): Existing Blosc2 metadata to use
            as defaults.
        patch_size (Optional[Union[int, List, Tuple, str]]): Patch size hint
            or "default". See ``open``/``save`` for expected shapes.
        chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
        block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
        shape (Union[List, Tuple, np.ndarray]): Full array shape including non-spatial axes.
        dtype_itemsize (int): Number of bytes per array element.
        spatial_axis_mask (Optional[list[bool]]): Mask indicating for every axis whether it is spatial or not.
        cparams: Compression parameters, or None for the defaults.
        dparams: Decompression parameters, or None for the defaults.
        array (Optional[np.ndarray]): The data to be compressed, if known.
            With the default compression parameters, the filter
            (NOFILTER or SHUFFLE) is then chosen per array as nnU-Net
            does; without data, blosc2's default (SHUFFLE) applies.

    Returns:
        Tuple[MetaBlosc2, Union[Dict, blosc2.CParams]]: The validated Blosc2
        metadata, which records the portable compression parameters (see
        :func:`portable_cparams`), and the full compression parameters to
        create the store with.
    """
    num_spatial_axes = sum(spatial_axis_mask)
    is_default_patch_size = isinstance(patch_size, str) and patch_size == "default"
    if patch_size is not None and not is_default_patch_size and (chunk_size is not None or block_size is not None):
        raise RuntimeError("patch_size and chunk_size / block_size cannot both be explicitly set.")
    if (chunk_size is not None and block_size is None) or (chunk_size is None and block_size is not None):
        raise RuntimeError("If either chunk/block size is used then both must be set.")

    if is_default_patch_size:
        if meta_blosc2 is not None and meta_blosc2.patch_size is not None:  # Use previously loaded patch size, when patch size is not explicitly set and a patch size from a previously loaded image exists
            patch_size = meta_blosc2.patch_size
        elif meta_blosc2 is not None and (meta_blosc2.chunk_size is not None):
            chunk_size = meta_blosc2.chunk_size
            block_size = meta_blosc2.block_size
        elif num_spatial_axes > 0:  # Use default patch size, when patch size is not explicitly set and no patch size from a previously loaded image exists
            patch_size = [MLARRAY_DEFAULT_PATCH_SIZE] * num_spatial_axes
        else:  # No spatial axis to optimize for: let Blosc2 choose the layout
            patch_size = None
    if chunk_size is not None or block_size is not None:
        patch_size = None

    patch_size = normalize_patch_size(patch_size, num_spatial_axes)

    if patch_size is not None and num_spatial_axes == 0:
        raise RuntimeError(
            "Automatic patch-size optimization requires at least one spatial axis. "
            "Set patch_size=None and provide chunk_size/block_size manually, "
            "or let Blosc2 determine the layout."
        )

    if patch_size is not None and 0 in shape:
        # An array without elements has no layout to optimize; Blosc2 chooses one.
        chunk_size, block_size, patch_size = None, None, None
    elif patch_size is not None:
        chunk_size, block_size = _layout.comp_blosc2_params(
            image_size=tuple(int(v) for v in shape),
            patch_size=tuple(int(v) for v in patch_size),
            spatial_axis_mask=spatial_axis_mask,
            bytes_per_pixel=dtype_itemsize,
        )

    choose_filter = array is not None and chunk_size is not None and is_default_cparams(cparams)
    cparams = resolve_cparams(cparams)
    dparams = resolve_dparams(dparams)
    if choose_filter:
        cparams = dict(cparams)
        cparams["filters"] = [_layout.select_blosc2_filter(
            array, chunk_size, block_size, codec=cparams["codec"], clevel=cparams["clevel"],
        )]

    meta_blosc2 = MetaBlosc2(
        chunk_size=chunk_size,
        block_size=block_size,
        patch_size=patch_size,
        cparams=portable_cparams(cparams),
        dparams=dparams,
    )
    meta_blosc2._validate_and_cast(ndims=len(shape), spatial_ndims=num_spatial_axes)
    return meta_blosc2, cparams


# ----------------------------------------------------------------------------
# File IO
# ----------------------------------------------------------------------------

def load_into_memory(filepath, dparams) -> blosc2.NDArray:
    """Read a whole ``.mla`` file into an in-memory Blosc2 array, chunks and vlmeta included."""
    ondisk = blosc2.open(str(filepath), dparams=dparams, mode="r")
    in_memory = blosc2.empty(
        ondisk.shape,
        dtype=ondisk.dtype,
        chunks=ondisk.chunks,
        blocks=ondisk.blocks,
        cparams=ondisk.schunk.cparams,
        dparams=ondisk.schunk.dparams,
    )
    for i in range(ondisk.schunk.nchunks):
        in_memory.schunk.update_chunk(i, ondisk.schunk.get_chunk(i))
    for key, value in ondisk.schunk.vlmeta.getall().items():
        in_memory.schunk.vlmeta[key] = value
    return in_memory


def save_atomically(store, filepath, write_metadata: Callable[[Any], None]) -> None:
    """Save ``store`` to ``filepath`` through a temporary file next to it.

    ``write_metadata`` receives the vlmeta of the temporary file. The target
    is replaced with ``os.replace`` only once the file is complete; on any
    error the temporary file is removed and the target is left untouched.
    """
    target_dir = os.path.dirname(os.path.abspath(str(filepath)))
    fd, tmp_filepath = tempfile.mkstemp(prefix=".", suffix=f".{MLARRAY_SUFFIX}", dir=target_dir)
    os.close(fd)
    os.remove(tmp_filepath)
    try:
        store.save(tmp_filepath)
        saved = blosc2.open(urlpath=tmp_filepath, mode="a")
        write_metadata(saved.schunk.vlmeta)
        del saved
        os.replace(tmp_filepath, str(filepath))
    except BaseException:
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        raise


# ----------------------------------------------------------------------------
# Pickling
# ----------------------------------------------------------------------------

def reopen_modes(mode, mmap_mode) -> Tuple[Optional[str], Optional[str]]:
    """Modes to reopen a file with, so that a file from ``create`` is not truncated."""
    return {'w': 'a'}.get(mode, mode), {'w+': 'r+'}.get(mmap_mode, mmap_mode)


def in_memory_state(store) -> Dict[str, Any]:
    """Picklable state of an in-memory Blosc2 array."""
    return {
        "cframe": store.to_cframe(),
        "cparams": store.schunk.cparams,
        "dparams": store.schunk.dparams,
    }


def restore_in_memory(state: Dict[str, Any]) -> blosc2.NDArray:
    """Rebuild an in-memory Blosc2 array from :func:`in_memory_state`."""
    store = blosc2.ndarray_from_cframe(state["cframe"])
    store.schunk.cparams = state["cparams"]
    store.schunk.dparams = state["dparams"]
    return store
