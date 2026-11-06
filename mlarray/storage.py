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
from mlarray.metadata import Storage

MLARRAY_SUFFIX = "mla"
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


def compression_record(cparams: Union[Dict, blosc2.CParams]) -> Dict[str, Any]:
    """Portable compression settings as recorded in ``storage.compression`` (spec 8).

    Codec and filters are stored by name; only fields given in ``cparams``
    are recorded (filters left to blosc2's default stay unrecorded), and
    ``NOFILTER`` slots are dropped; ``["NOFILTER"]`` records "no filter".
    """
    portable = portable_cparams(cparams)
    record: Dict[str, Any] = {}
    if "codec" in portable:
        record["codec"] = blosc2.Codec(portable["codec"]).name
    if "clevel" in portable:
        record["level"] = int(portable["clevel"])
    if "filters" in portable:
        filters = [blosc2.Filter(f).name for f in portable["filters"] if blosc2.Filter(f) != blosc2.Filter.NOFILTER]
        record["filters"] = filters or ["NOFILTER"]  # explicitly no filter, unlike an absent entry
    return record


def cparams_from_compression(compression, nthreads: Optional[int] = None) -> blosc2.CParams:
    """Compression parameters for a recorded ``storage.compression``.

    Missing codec or level fall back to MLArray's defaults (LZ4HC, 5); missing
    filters to blosc2's default. ``nthreads`` defaults to MLArray's 4.
    """
    compression = compression or {}
    default = default_cparams()
    kwargs: Dict[str, Any] = {
        "codec": blosc2.Codec[compression["codec"]] if "codec" in compression else default["codec"],
        "clevel": compression.get("level", default["clevel"]),
        "nthreads": default["nthreads"] if nthreads is None else nthreads,
    }
    if "filters" in compression:
        filters = [blosc2.Filter[name] for name in compression["filters"]]
        kwargs["filters"] = filters
        kwargs["filters_meta"] = [0] * len(filters)
    return blosc2.CParams(**kwargs)


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


def comp_layout(previous: Optional[Storage], patch_size, chunk_size, block_size, shape, dtype_itemsize, spatial_axis_mask, cparams, array=None) -> Tuple[Storage, Union[Dict, blosc2.CParams]]:
    """Compute the blosc2 layout and compression of a new store.

    Args:
        previous (Optional[Storage]): Layout recorded before (e.g. of a loaded
            file); used when ``patch_size`` is ``"default"``: its patch size
            first, then its chunk/block size, then the default patch size.
        patch_size (Optional[Union[int, List, Tuple, str]]): Patch size hint
            over the spatial axes, ``"default"`` or None.
        chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
        block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
        shape (Union[List, Tuple, np.ndarray]): Full array shape including non-spatial axes.
        dtype_itemsize (int): Number of bytes per array element.
        spatial_axis_mask (list[bool]): For every array axis whether it is spatial.
        cparams: Compression parameters, or None for the defaults.
        array (Optional[np.ndarray]): The data to be compressed, if known.
            With the default compression parameters, the filter
            (NOFILTER or SHUFFLE) is then chosen per array as nnU-Net
            does; without data, blosc2's default (SHUFFLE) applies.

    Returns:
        Tuple[Storage, Union[Dict, blosc2.CParams]]: The layout to record
        (chunk and block size may be None when blosc2 chooses them; the
        caller fills them in from the store) and the full compression
        parameters to create the store with.
    """
    num_spatial_axes = sum(spatial_axis_mask)
    is_default_patch_size = isinstance(patch_size, str) and patch_size == "default"
    if patch_size is not None and not is_default_patch_size and (chunk_size is not None or block_size is not None):
        raise RuntimeError("patch_size and chunk_size / block_size cannot both be explicitly set.")
    if (chunk_size is not None and block_size is None) or (chunk_size is None and block_size is not None):
        raise RuntimeError("If either chunk/block size is used then both must be set.")

    if is_default_patch_size:
        if previous is not None and previous.patch_size is not None:  # Reuse the patch size recorded before
            patch_size = list(previous.patch_size)
        elif previous is not None and previous.chunk_size is not None and previous.block_size is not None:
            chunk_size = list(previous.chunk_size)
            block_size = list(previous.block_size)
        elif num_spatial_axes > 0:  # Default patch size
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
    if patch_size is not None and len(patch_size) != num_spatial_axes:
        raise ValueError(f"patch_size needs one entry per spatial axis ({num_spatial_axes}), got {patch_size}.")

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
    if choose_filter:
        cparams = dict(cparams)
        cparams["filters"] = [_layout.select_blosc2_filter(
            array, chunk_size, block_size, codec=cparams["codec"], clevel=cparams["clevel"],
        )]

    for name, value in (("chunk_size", chunk_size), ("block_size", block_size)):
        if value is not None:
            value = [value] * len(shape) if isinstance(value, (int, np.integer)) else list(value)
            if len(value) != len(shape):
                raise ValueError(f"{name} needs one entry per array axis ({len(shape)}), got {value}.")
        if name == "chunk_size":
            chunk_size = value
        else:
            block_size = value

    storage = Storage.from_dict({
        "chunk_size": chunk_size,
        "block_size": block_size,
        "patch_size": patch_size,
        "compression": compression_record(cparams),
    })
    return storage, cparams


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
    # Raw copy: decoding vlmeta through blosc2 is unsafe for untrusted files.
    for key in ondisk.schunk.vlmeta.keys():
        in_memory.schunk.vlmeta.set_vlmeta(key, ondisk.schunk.vlmeta.get_vlmeta(key), typesize=1)
    return in_memory


def save_atomically(store, filepath, write_metadata: Callable[[Any], None]) -> None:
    """Save ``store`` to ``filepath`` through a temporary file next to it.

    ``write_metadata`` receives the opened temporary file. The target
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
        write_metadata(saved)
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
    store = blosc2.ndarray_from_cframe(state["cframe"], copy=True)   # own the buffer (needed in older blosc2)
    store.schunk.cparams = state["cparams"]
    store.schunk.dparams = state["dparams"]
    return store
