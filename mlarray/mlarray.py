from copy import deepcopy
import numpy as np
import blosc2
import math
from typing import Any, Dict, Optional, Union, List, Tuple
from pathlib import Path
import os
import tempfile
from mlarray.meta import (
    Meta,
    MetaBlosc2,
    AxisLabel,
    _spatial_axis_mask,
    _meta_internal_write,
)
from mlarray import blosc2_layout_strategies as _layout
from mlarray.utils import is_serializable
import pickle
import gzip

MLARRAY_SUFFIX = "mla"
MLARRAY_VERSION = "v0"
MLARRAY_DEFAULT_PATCH_SIZE = 192
MLARRAY_DEFAULT_READ_AHEAD = 4

_read_ahead_configured = False


def set_read_ahead(nthreads: int = MLARRAY_DEFAULT_READ_AHEAD) -> int:
    """Set blosc2's read-ahead for this process (``blosc2.set_prefetch``).

    With read-ahead, reading a slice of an on-disk array fetches the chunks it
    needs with ``nthreads`` background threads in parallel, instead of one
    small request after another. For data that is not in the page cache this
    makes patch reads several times faster, most on network storage (in our
    benchmark with 12 nnU-Net-style loader processes on network storage, 3D
    patches took 64-82% less time). For data already in the page cache it can
    cost a little (up to about 17% per read with a single reader).

    MLArray turns it on with ``MLARRAY_DEFAULT_READ_AHEAD`` (4) threads the
    first time a file is opened, unless this function was called before or
    the ``BLOSC_PREFETCH`` environment variable is set. The setting applies to
    the whole process, including other blosc2 arrays it reads; each reader
    thread gets its own read-ahead threads. Call ``set_read_ahead(0)`` to turn
    it off. Without read-ahead support in the installed blosc2 this does
    nothing.

    Args:
        nthreads (int): Read-ahead threads per slice being read; 0 turns
            read-ahead off.

    Returns:
        int: The previous number of read-ahead threads (0 if unsupported).
    """
    global _read_ahead_configured
    _read_ahead_configured = True
    if not hasattr(blosc2, "set_prefetch"):
        return 0
    return int(blosc2.set_prefetch(int(nthreads)))


def _apply_default_read_ahead() -> None:
    """Turn on read-ahead once per process unless the user configured it."""
    global _read_ahead_configured
    if _read_ahead_configured:
        return
    _read_ahead_configured = True
    if "BLOSC_PREFETCH" in os.environ or not hasattr(blosc2, "set_prefetch"):
        return
    blosc2.set_prefetch(MLARRAY_DEFAULT_READ_AHEAD)


class MLArray:
    def __init__(
            self,
            array: Optional[Union[np.ndarray, str, Path]] = None,
            spacing: Optional[Union[List, Tuple, np.ndarray]] = None,
            origin: Optional[Union[List, Tuple, np.ndarray]] = None,
            direction: Optional[Union[List, Tuple, np.ndarray]] = None,
            affine: Optional[Union[List, Tuple, np.ndarray]] = None,
            meta: Optional[Union[Dict, Meta]] = None,
            axis_labels: Optional[List[Union[str, AxisLabel]]] = None,
            copy: Optional['MLArray'] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ) -> None:
        """Initializes a MLArray instance.

        The MLArray file format (".mla") is a Blosc2-compressed container
        with standardized metadata support for N-dimensional medical images.

        Args:
            array (Optional[Union[np.ndarray, str, Path]]): Input data or file
                path. Use a numpy ndarray for in-memory arrays, or a string/Path
                to load a ".mla" file. If None, an empty MLArray
                instance is created.
            spacing (Optional[Union[List, Tuple, np.ndarray]]): Spacing per
                spatial axis. Provide a list/tuple/ndarray with length equal to
                the number of spatial dimensions (e.g., [sx, sy, sz]).
            origin (Optional[Union[List, Tuple, np.ndarray]]): Origin per axis.
                Provide a list/tuple/ndarray with length equal to the number of
                spatial dimensions.
            direction (Optional[Union[List, Tuple, np.ndarray]]): Direction
                cosine matrix. Provide a 2D list/tuple/ndarray with shape
                (ndims, ndims) for spatial dimensions.
            affine (Optional[Union[List, Tuple, np.ndarray]]): Homogeneous
                affine matrix. Provide a 2D list/tuple/ndarray with shape
                (spatial_ndims + 1, spatial_ndims + 1).
            meta (Optional[Dict | Meta]): Free-form metadata dictionary or Meta
                instance. Must be JSON-serializable when saving. 
                If meta is passed as a Dict, it is internally converted into a
                ``Meta`` object with the dict stored as ``meta.source``.
            axis_labels (Optional[List[Union[str, AxisLabel]]]): Per-axis labels or roles. Length must match ndims. If None, the array
                is treated as purely spatial.
            copy (Optional[MLArray]): Another MLArray instance to copy metadata
                fields from. If provided, its metadata overrides any metadata
                set via arguments.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters used when creating in-memory compressed array data.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used for chunk access.
        """
        self.filepath = None
        self.support_metadata = None
        self.mode = None
        self.mmap_mode = None
        self._meta = None
        self._store = None
        self._backend = None
        if isinstance(array, (str, Path)) and (
            spacing is not None
            or origin is not None
            or direction is not None
            or affine is not None
            or meta is not None
            or axis_labels is not None
            or copy is not None
            or patch_size != "default"
            or chunk_size is not None
            or block_size is not None
            or cparams is not None
            or dparams is not None
        ):
            raise RuntimeError(
                "Spacing, origin, direction, affine, meta, axis_labels, copy, patch_size, "
                "chunk_size, block_size, cparams or dparams cannot be set when "
                "array is a filepath."
            )
        if isinstance(array, (str, Path)):
            self._load(array, compressed=compressed)
        else:
            self._validate_and_add_meta(
                meta,
                spacing=spacing,
                origin=origin,
                direction=direction,
                affine=affine,
                axis_labels=axis_labels,
                has_array=False,
                validate=False,
            )
            if array is not None:
                self._asarray(
                    array,
                    meta=self.meta,
                    patch_size=patch_size,
                    chunk_size=chunk_size,
                    block_size=block_size,
                    cparams=cparams,
                    dparams=dparams,
                    compressed=compressed,
                )
                has_array = True
            else:
                if compressed:
                    self._store = blosc2.empty((0,))
                    self._backend = "blosc2"
                else:
                    self._store = np.empty((0,))
                    self._backend = "numpy"
                has_array = False
            if copy is not None:
                with _meta_internal_write():
                    self.meta.copy_from(copy.meta)
            self._validate_and_add_meta(self.meta, has_array=has_array, validate=True)

    @classmethod
    def open(
            cls,
            filepath: Union[str, Path],
            mode: str = 'r',
            mmap_mode: str = 'r',
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Open an existing MLArray file with memory mapping.

        The first open in a process turns on blosc2's read-ahead with 4
        threads, which makes reads of data not in the page cache several
        times faster; see :func:`mlarray.set_read_ahead` to change or turn it
        off.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be either 'r' (default) or 'a'. Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'r' (default), 'r+', 'c' or None. Leave at default if unsure.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for opening.
        """
        class_instance = cls()
        class_instance._open(filepath, mode, mmap_mode, dparams)
        return class_instance

    @classmethod
    def create(            
            cls,
            filepath: Union[str, Path],
            shape: Union[List, Tuple, np.ndarray],
            dtype: np.dtype,
            meta: Optional[Union[Dict, Meta]] = None,
            mode: str = 'w',
            mmap_mode: str = 'w+',
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Create a new MLArray file with memory mapping.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            shape (Union[List, Tuple, np.ndarray]): Shape of the array
                to create. If provided, a new file is created. Length must match
                the full array dimensionality (including non-spatial axes if present).
            dtype (np.dtype): Numpy dtype for a newly created array.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be 'w' (default). Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'w+' (default) or None. Leave at default if unsure.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Provide an int for isotropic sizes or
                a list/tuple with length equal to the number of spatial
                dimensions. Use "default" to use the default patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.LZ4HC,
                'clevel': 5, 'nthreads': 4}`` with blosc2's default filter
                (SHUFFLE); the data is not known yet, so no filter is chosen
                from it.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for creation.
        """
        class_instance = cls()
        class_instance._create(filepath, shape, dtype, meta, mode, mmap_mode, patch_size, chunk_size, block_size, cparams, dparams)
        return class_instance

    @classmethod
    def load(
            cls,
            filepath: Union[str, Path], 
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Load a MLArray file as a whole without memory mapping.

        Args:
            filepath (Union[str, Path]): Path to the MLArray file to be loaded.
                The filepath needs to have the extension ".mla".
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when loading/accessing compressed
                chunks. If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is not ".mla".
        """
        class_instance = cls()
        class_instance._load(filepath, dparams, compressed=compressed)
        return class_instance

    @classmethod
    def empty(
            cls,
            shape: Union[int, List, Tuple, np.ndarray],
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray with uninitialized values.

        Args:
            shape (Union[int, List, Tuple, np.ndarray]): Shape of the output
                array.
            dtype (np.dtype): Numpy dtype for the output array. If None, uses
                ``blosc2.DEFAULT_FLOAT``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters used when writing chunks.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when reading chunks.

        Returns:
            MLArray: A newly created in-memory MLArray instance.
        """
        dtype = blosc2.DEFAULT_FLOAT if dtype is None else dtype
        class_instance = cls()
        class_instance._construct_in_memory(
            constructor="empty",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
        )
        return class_instance

    @classmethod
    def zeros(
            cls,
            shape: Union[int, List, Tuple, np.ndarray],
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray filled with zeros.

        Args:
            shape (Union[int, List, Tuple, np.ndarray]): Shape of the output
                array.
            dtype (np.dtype): Numpy dtype for the output array. If None, uses
                ``blosc2.DEFAULT_FLOAT``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.
        """
        dtype = blosc2.DEFAULT_FLOAT if dtype is None else dtype
        class_instance = cls()
        class_instance._construct_in_memory(
            constructor="zeros",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
        )
        return class_instance

    @classmethod
    def ones(
            cls,
            shape: Union[int, List, Tuple, np.ndarray],
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray filled with ones.

        Args:
            shape (Union[int, List, Tuple, np.ndarray]): Shape of the output
                array.
            dtype (np.dtype): Numpy dtype for the output array. If None, uses
                ``blosc2.DEFAULT_FLOAT``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.
        """
        dtype = blosc2.DEFAULT_FLOAT if dtype is None else dtype
        class_instance = cls()
        class_instance._construct_in_memory(
            constructor="ones",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
        )
        return class_instance

    @classmethod
    def full(
            cls,
            shape: Union[int, List, Tuple, np.ndarray],
            fill_value: Union[bytes, int, float, bool],
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray filled with ``fill_value``.

        Args:
            shape (Union[int, List, Tuple, np.ndarray]): Shape of the output
                array.
            fill_value (Union[bytes, int, float, bool]): Fill value used for all
                elements in the output.
            dtype (np.dtype): Numpy dtype for the output array. If None, inferred
                from ``fill_value``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.
        """
        if dtype is None:
            if isinstance(fill_value, bytes):
                dtype = np.dtype(f"S{len(fill_value)}")
            else:
                dtype = np.dtype(type(fill_value))
        class_instance = cls()
        class_instance._construct_in_memory(
            constructor="full",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
            constructor_kwargs={"fill_value": fill_value},
        )
        return class_instance

    @classmethod
    def arange(
            cls,
            start: Union[int, float],
            stop: Optional[Union[int, float]] = None,
            step: Optional[Union[int, float]] = 1,
            dtype: np.dtype = None,
            shape: Optional[Union[int, List, Tuple, np.ndarray]] = None,
            c_order: bool = True,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = None,
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray with evenly spaced values.

        Behavior mirrors :func:`blosc2.arange` while also applying MLArray
        metadata and chunk/block optimization settings.

        Args:
            start (Union[int, float]): Start of interval. If ``stop`` is None,
                this value is treated as stop and start becomes 0.
            stop (Optional[Union[int, float]]): End of interval (exclusive).
            step (Optional[Union[int, float]]): Spacing between values.
            dtype (np.dtype): Output dtype. If None, inferred similarly to
                ``blosc2.arange``.
            shape (Optional[Union[int, List, Tuple, np.ndarray]]): Target output
                shape. If None, shape is inferred from start/stop/step.
            c_order (bool): Store in C order (row-major) if True.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Defaults to None for this method.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.

        Raises:
            ValueError: If ``shape`` is inconsistent with
                ``start``/``stop``/``step``.
        """
        if step is None:
            step = 1
        if stop is None:
            stop = start
            start = 0

        num = max(math.ceil((stop - start) / step), 0)
        if shape is None:
            shape = (num,)
        else:
            shape = cls._normalize_shape(shape)
            if math.prod(shape) != num:
                raise ValueError(
                    "The shape is not consistent with the start, stop and step values"
                )

        if dtype is None:
            dtype = (
                blosc2.DEFAULT_FLOAT
                if np.any([np.issubdtype(type(d), np.floating) for d in (start, stop, step)])
                else blosc2.DEFAULT_INT
            )

        class_instance = cls()
        class_instance._construct_in_memory(
            constructor="arange",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
            constructor_kwargs={
                "start": start,
                "stop": stop,
                "step": step,
                "c_order": c_order,
            },
        )
        return class_instance

    @classmethod
    def linspace(
            cls,
            start: Union[int, float, complex],
            stop: Union[int, float, complex],
            num: Optional[int] = None,
            dtype: np.dtype = None,
            endpoint: bool = True,
            shape: Optional[Union[int, List, Tuple, np.ndarray]] = None,
            c_order: bool = True,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = None,
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray with evenly spaced samples.

        Behavior mirrors :func:`blosc2.linspace` while also applying MLArray
        metadata and chunk/block optimization settings.

        Args:
            start (Union[int, float, complex]): Start value of the sequence.
            stop (Union[int, float, complex]): End value of the sequence.
            num (Optional[int]): Number of samples. Required when ``shape`` is
                None.
            dtype (np.dtype): Output dtype. If None, inferred similarly to
                ``blosc2.linspace``.
            endpoint (bool): Whether ``stop`` is included.
            shape (Optional[Union[int, List, Tuple, np.ndarray]]): Target output
                shape. If None, inferred from ``num``.
            c_order (bool): Store in C order (row-major) if True.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Defaults to None for this method.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.

        Raises:
            ValueError: If neither ``shape`` nor ``num`` is specified, or if the
                provided ``shape`` and ``num`` are inconsistent.
        """
        if shape is None:
            if num is None:
                raise ValueError("Either `shape` or `num` must be specified.")
            shape = (num,)
        else:
            shape = cls._normalize_shape(shape)
            num = math.prod(shape) if num is None else num

        if math.prod(shape) != num or num < 0:
            msg = f"Shape is not consistent with the specified num value {num}."
            if num < 0:
                msg += "num must be nonnegative."
            raise ValueError(msg)

        if dtype is None:
            dtype = (
                blosc2.DEFAULT_COMPLEX
                if np.any([np.issubdtype(type(d), complex) for d in (start, stop)])
                else blosc2.DEFAULT_FLOAT
            )

        class_instance = cls()
        class_instance._construct_in_memory(
            constructor="linspace",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
            constructor_kwargs={
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "c_order": c_order,
            },
        )
        return class_instance

    @classmethod
    def asarray(
            cls,
            array: Union[np.ndarray],
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Convert a NumPy array into an in-memory Blosc2-backed MLArray.

        Args:
            array (Union[np.ndarray]): Input array to convert to MLArray.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Provide an int for isotropic sizes or
                a list/tuple with length equal to the number of dimensions.
                Use "default" to use the default patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Cannot be combined with an
                explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Cannot be combined with an
                explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when creating the in-memory Blosc2
                container (for example codec, compression level, and filters).
                Controls how data is compressed when stored. If None, defaults
                to ``{'codec': blosc2.Codec.LZ4HC, 'clevel': 5, 'nthreads': 4}``,
                with the filter (NOFILTER or SHUFFLE) chosen per array from its
                data as nnU-Net does.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.
            compressed (bool): If True (default), the data is compressed into
                an in-memory Blosc2 array, which is a copy. If False, the
                MLArray wraps ``array`` without copying when it is already
                C-contiguous, so writes through either object are visible in
                the other. Pass ``array.copy()`` to avoid that.

        Returns:
            MLArray: A newly created MLArray instance.

        Raises:
            TypeError: If ``array`` is not a NumPy ndarray.
            ValueError: If ``meta`` is not None, dict, or Meta.
            RuntimeError: If patch/chunk/block arguments are inconsistent.
            NotImplementedError: If automatic patch optimization is not
                implemented for the provided dimensionality.
        """
        class_instance = cls()
        class_instance._asarray(
            array,
            meta,
            patch_size,
            chunk_size,
            block_size,
            cparams,
            dparams,
            compressed=compressed,
        )
        return class_instance

    @classmethod
    def empty_like(
            cls,
            x,
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray with the same shape as ``x``.

        Args:
            x: Source object. Can be an ``MLArray`` or any array-like object
                exposing ``shape`` and ``dtype``.
            dtype (np.dtype): Output dtype. If None, inferred from ``x``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. If ``x`` is an ``MLArray`` and ``meta``
                is None, metadata is copied from ``x``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.

        Raises:
            TypeError: If ``x`` is not an ``MLArray`` and has no ``shape``/``dtype``.
        """
        class_instance = cls()
        shape, dtype, meta = class_instance._resolve_like_input(x, dtype, meta)
        class_instance._construct_in_memory(
            constructor="empty",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
        )
        return class_instance

    @classmethod
    def zeros_like(
            cls,
            x,
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray of zeros with the same shape as ``x``.

        Args:
            x: Source object. Can be an ``MLArray`` or any array-like object
                exposing ``shape`` and ``dtype``.
            dtype (np.dtype): Output dtype. If None, inferred from ``x``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. If ``x`` is an ``MLArray`` and ``meta``
                is None, metadata is copied from ``x``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.

        Raises:
            TypeError: If ``x`` is not an ``MLArray`` and has no ``shape``/``dtype``.
        """
        class_instance = cls()
        shape, dtype, meta = class_instance._resolve_like_input(x, dtype, meta)
        class_instance._construct_in_memory(
            constructor="zeros",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
        )
        return class_instance

    @classmethod
    def ones_like(
            cls,
            x,
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray of ones with the same shape as ``x``.

        Args:
            x: Source object. Can be an ``MLArray`` or any array-like object
                exposing ``shape`` and ``dtype``.
            dtype (np.dtype): Output dtype. If None, inferred from ``x``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. If ``x`` is an ``MLArray`` and ``meta``
                is None, metadata is copied from ``x``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.

        Raises:
            TypeError: If ``x`` is not an ``MLArray`` and has no ``shape``/``dtype``.
        """
        class_instance = cls()
        shape, dtype, meta = class_instance._resolve_like_input(x, dtype, meta)
        class_instance._construct_in_memory(
            constructor="ones",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
        )
        return class_instance

    @classmethod
    def full_like(
            cls,
            x,
            fill_value: Union[bool, int, float, complex],
            dtype: np.dtype = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Create an in-memory MLArray filled with ``fill_value`` and shape of ``x``.

        Args:
            x: Source object. Can be an ``MLArray`` or any array-like object
                exposing ``shape`` and ``dtype``.
            fill_value (Union[bool, int, float, complex]): Fill value used for
                all elements in the output.
            dtype (np.dtype): Output dtype. If None, inferred from ``x``.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. If ``x`` is an ``MLArray`` and ``meta``
                is None, metadata is copied from ``x``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.

        Raises:
            TypeError: If ``x`` is not an ``MLArray`` and has no ``shape``/``dtype``.
        """
        class_instance = cls()
        shape, dtype, meta = class_instance._resolve_like_input(x, dtype, meta)
        class_instance._construct_in_memory(
            constructor="full",
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
            constructor_kwargs={"fill_value": fill_value},
        )
        return class_instance

    def compress(
            self,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
        ) -> None:
        """Compress this MLArray in-place using the Blosc2 backend.

        The instance is detached from any previously opened file state and
        becomes an in-memory object. Persist changes by calling ``save()``.

        Args:
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Raises:
            TypeError: If no array data is loaded.
        """
        if self._store is None or self.meta._has_array.has_array is False:
            raise TypeError("MLArray has no array data loaded.")

        source_meta = deepcopy(self.meta)
        converted = MLArray.asarray(
            self.to_numpy(),
            meta=None,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=True,
        )
        converted.meta.copy_from(source_meta, overwrite=True)
        self._replace_with_converted(converted)

    def decompress(self) -> None:
        """Decompress this MLArray in-place to a NumPy-backed array.

        The instance is detached from any previously opened file state and
        becomes an in-memory object. Persist changes by calling ``save()``.
        ``meta.blosc2`` metadata is cleared.

        Raises:
            TypeError: If no array data is loaded.
        """
        if self._store is None or self.meta._has_array.has_array is False:
            raise TypeError("MLArray has no array data loaded.")

        source_meta = deepcopy(self.meta)
        converted = MLArray.asarray(
            self.to_numpy(),
            meta=None,
            patch_size=None,
            chunk_size=None,
            block_size=None,
            cparams=None,
            dparams=None,
            compressed=False,
        )
        converted.meta.copy_from(source_meta, overwrite=True)
        with _meta_internal_write():
            converted.meta.blosc2 = MetaBlosc2()
        self._replace_with_converted(converted)

    def save(
            self,
            filepath: Union[str, Path],
        ):
        """Save the array to a MLArray file.

        The file is written to a temporary file next to the target and then
        atomically moved into place, so an existing target is only replaced
        once the new file is complete. The source file of an opened MLArray
        is never modified. If an opened MLArray is saved to its own file, it
        is reopened afterwards with the same mode and mmap_mode.

        Args:
            filepath (Union[str, Path]): Path to save the file. Must end with
                ".mla".

        Raises:
            RuntimeError: If the file extension is not ".mla".
        """
        if not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.{MLARRAY_SUFFIX}' as extension.")

        self.support_metadata = True

        is_file_backed = self.mode is not None or self.mmap_mode is not None
        saves_onto_source = (
            is_file_backed
            and self.filepath is not None
            and Path(filepath).is_file()
            and os.path.samefile(self.filepath, filepath)
        )

        store, meta = self._blosc2_store_for_save()

        target_dir = os.path.dirname(os.path.abspath(str(filepath)))
        fd, tmp_filepath = tempfile.mkstemp(prefix=".", suffix=f".{MLARRAY_SUFFIX}", dir=target_dir)
        os.close(fd)
        os.remove(tmp_filepath)
        try:
            store.save(tmp_filepath)
            saved = blosc2.open(urlpath=tmp_filepath, mode="a")
            saved.schunk.vlmeta["mlarray"] = self._serialize_metadata(meta)
            del saved
            os.replace(tmp_filepath, str(filepath))
        except BaseException:
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            raise

        if saves_onto_source:
            # The open store still refers to the replaced file; rebind it to the new one.
            reopen_mode = {'w': 'a'}.get(self.mode, self.mode)
            reopen_mmap_mode = {'w+': 'r+'}.get(self.mmap_mode, self.mmap_mode)
            self._open(filepath, reopen_mode, reopen_mmap_mode, self._store.schunk.dparams)

    def close(self):
        """Flush metadata and close the underlying store.

        After closing, the MLArray instance no longer has an attached array.
        """
        self._write_metadata()
        self._store = None
        self._backend = None
        self.filepath = None
        self.support_metadata = None
        self.mode = None  
        self.mmap_mode = None
        self._meta = None

    def flush(self):
        """Write pending metadata changes to the opened file.

        Array data written through a writable memory map is persisted by
        Blosc2 directly; metadata changes are only written on ``flush()`` or
        ``close()``. Does nothing for read-only or in-memory arrays.
        """
        self._write_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __getstate__(self):
        """Pickle support, e.g. for spawn/forkserver DataLoader workers.

        - Opened files are pickled by reference (filepath, mode, mmap_mode,
          dparams) and reopened on unpickling. Files created with ``create``
          are reopened with ``mode='a'`` / ``mmap_mode='r+'`` so they are not
          truncated. Metadata in memory, including changes not yet flushed,
          is carried over; array changes made through a copy-on-write map
          (``mmap_mode='c'``) are not.
        - In-memory Blosc2 arrays are pickled as a compressed frame.
        - NumPy-backed arrays pickle the NumPy array.
        """
        state = self.__dict__.copy()
        store = state.pop("_store")
        if store is None or self._backend == "numpy":
            state["_store_state"] = ("value", store)
        elif self.mode is not None or self.mmap_mode is not None:
            state["_store_state"] = ("file", {
                "mode": {'w': 'a'}.get(self.mode, self.mode),
                "mmap_mode": {'w+': 'r+'}.get(self.mmap_mode, self.mmap_mode),
                "dparams": store.schunk.dparams,
            })
        else:
            state["_store_state"] = ("cframe", {
                "cframe": store.to_cframe(),
                "cparams": store.schunk.cparams,
                "dparams": store.schunk.dparams,
            })
        return state

    def __setstate__(self, state):
        kind, payload = state.pop("_store_state")
        self.__dict__.update(state)
        if kind == "value":
            self._store = payload
        elif kind == "cframe":
            store = blosc2.ndarray_from_cframe(payload["cframe"])
            store.schunk.cparams = payload["cparams"]
            store.schunk.dparams = payload["dparams"]
            self._store = store
        else:
            meta = self._meta
            self._store = None
            self._open(self.filepath, payload["mode"], payload["mmap_mode"], payload["dparams"])
            self._meta = meta

    def to_numpy(self):
        """Return the underlying data as a NumPy array.

        Returns:
            np.ndarray: A NumPy view or copy of the stored array data.

        Raises:
            TypeError: If no array data is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        return self._store[...]

    def __array__(self, dtype=None, copy=None):
        """NumPy array interface for implicit conversion.

        Args:
            dtype (Optional[np.dtype]): Optional dtype to cast to.

        Returns:
            np.ndarray: The underlying data as a NumPy array.

        Raises:
            TypeError: If no array data is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        if self._backend == "numpy":
            return np.array(self._store, dtype=dtype, copy=copy)
        if copy is False:
            raise ValueError("A Blosc2-backed MLArray cannot be converted to a NumPy array without copying.")
        arr = self._store[...]
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, key):
        """Return a slice or element from the underlying array.

        Args:
            key (Any): Any valid NumPy/Blosc2 indexing key (slices, ints, tuples,
                boolean arrays).

        Returns:
            Any: The indexed value or subarray.

        Raises:
            TypeError: If no array data is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        return self._store[key]

    def __setitem__(self, key, value):
        """Assign to a slice or element in the underlying array.

        Args:
            key (Any): Any valid NumPy/Blosc2 indexing key.
            value (Any): Value(s) to assign. Must be broadcastable to the
                selected region.

        Raises:
            TypeError: If no array data is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        self._store[key] = value

    def __iter__(self):
        """Iterate over the first axis of the underlying array.

        Returns:
            Iterator: Iterator over the array's first dimension.

        Raises:
            TypeError: If no array data is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        return iter(self._store)

    def __len__(self):
        """Return the length of the first array dimension.

        Returns:
            int: Size of axis 0, or 0 if no array is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return 0
        return len(self._store)

    @property
    def meta(self) -> Optional[Meta]:
        return self._meta

    @property
    def spacing(self):
        """Returns the image spacing.

        Returns:
            list: Spacing per spatial axis with length equal to the number of
            spatial dimensions.
        """
        return self.meta.spatial.spacing

    @spacing.setter
    def spacing(self, value):
        self.meta.spatial.spacing = value

    @property
    def origin(self):
        """Returns the image origin.

        Returns:
            list: Origin per spatial axis with length equal to the number of
            spatial dimensions.
        """
        return self.meta.spatial.origin

    @origin.setter
    def origin(self, value):
        self.meta.spatial.origin = value

    @property
    def direction(self):
        """Returns the image direction.

        Returns:
            list: Direction cosine matrix with shape (ndims, ndims).
        """
        return self.meta.spatial.direction

    @direction.setter
    def direction(self, value):
        self.meta.spatial.direction = value

    @property
    def affine(self) -> np.ndarray:
        """Computes the affine transformation matrix for the image.

        Returns:
            list: Affine matrix with shape (ndims + 1, ndims + 1), or None if
                no array is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        if self.meta.spatial.affine is not None:
            return self.meta.spatial.affine
        spacing  = np.array(self.spacing) if self.spacing is not None else np.ones(self.spatial_ndim)
        origin  = np.array(self.origin) if self.origin is not None else np.zeros(self.spatial_ndim)
        direction = np.array(self.direction) if self.direction is not None else np.eye(self.spatial_ndim)
        affine = np.eye(self.spatial_ndim + 1)
        affine[:self.spatial_ndim, :self.spatial_ndim] = direction @ np.diag(spacing)
        affine[:self.spatial_ndim, self.spatial_ndim] = origin
        return affine.tolist()

    @affine.setter
    def affine(self, value):
        self.meta.spatial.affine = value

    @property
    def translation(self):
        """Extracts the translation vector from the affine matrix.

        Returns:
            list: Translation vector with length equal to the number of spatial
                dimensions, or None if no array is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        return np.array(self.affine)[:-1, -1].tolist()

    @property
    def scale(self):
        """Extracts the scaling factors from the affine matrix.

        Returns:
            list: Scaling factors per axis with length equal to the number of
                spatial dimensions, or None if no array is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        scales = np.linalg.norm(np.array(self.affine)[:-1, :-1], axis=0)
        return scales.tolist()

    @property
    def rotation(self):
        """Extracts the rotation matrix from the affine matrix.

        Returns:
            list: Rotation matrix with shape (ndims, ndims), or None if no array
                is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        rotation_matrix = np.array(self.affine)[:-1, :-1] / np.array(self.scale)
        return rotation_matrix.tolist()

    @property
    def shear(self):
        """Computes the shear matrix from the affine matrix.

        Returns:
            list: Shear matrix with shape (ndims, ndims), or None if no array is
                loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        scales = np.array(self.scale)
        rotation_matrix = np.array(self.rotation)
        shearing_matrix = np.dot(rotation_matrix.T, np.array(self.affine)[:-1, :-1]) / scales[:, None]
        return shearing_matrix.tolist()

    @property
    def shape(self):
        """Returns the shape of the array.

        Returns:
            tuple: Shape of the underlying array, or None if no array is loaded.
        """
        if self.meta is None or self.meta._has_array.has_array == False:
            return None
        shape = self.meta.spatial.shape
        return tuple(shape) if shape is not None else None

    @property
    def dtype(self):
        """Returns the dtype of the array.

        Returns:
            np.dtype: Dtype of the underlying array, or None if no array is
                loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        return self._store.dtype

    @property
    def ndim(self) -> int:
        """Returns the number of spatial and non-spatial dimensions of the array.

        Returns:
            int: Number of dimensions, or None if not determinable.
        """
        if self.meta is None:
            return None
        spatial = self.meta.spatial
        if spatial.shape is not None:
            return len(spatial.shape)
        if spatial._num_spatial_axes is not None:
            return spatial._num_spatial_axes + spatial._num_non_spatial_axes
        for attr in (spatial.spacing, spatial.origin, spatial.direction):
            if attr is not None:
                return len(attr)
        if spatial.affine is not None:
            return len(spatial.affine) - 1
        return None

    @property
    def spatial_ndim(self) -> int:
        """Returns the number of spatial dimensions.

        Returns:
            int: Number of spatial dimensions, or None if not determinable.
        """
        if self.meta is None:
            return None
        spatial = self.meta.spatial
        if spatial._num_spatial_axes is not None:
            return spatial._num_spatial_axes
        return self.ndim

    @classmethod
    def comp_blosc2_params(
            cls,
            image_size: Tuple[int, ...],
            patch_size: Tuple[int, ...],
            spatial_axis_mask: Optional[list[bool]] = None,
            bytes_per_pixel: int = 4,
            max_block_nbytes: int = _layout.DEFAULT_MAX_BLOCK_NBYTES,
            max_chunk_nbytes: int = _layout.DEFAULT_MAX_CHUNK_NBYTES,
        ):
        """
        Compute recommended Blosc2 chunk and block sizes from a patch-size hint.

        Uses nnU-Net's layout heuristic, generalized to any number of spatial
        and non-spatial axes (:func:`mlarray.blosc2_layout_strategies.comp_blosc2_params`):

            1. Non-spatial axes (channels, time, ...) get chunk and block
               size 1, so reading one channel touches only its own data.
            2. Blocks start at the next power of two of the patch size per
               spatial axis and are halved until they fit ``max_block_nbytes``
               (128 KiB).
            3. Chunks grow by whole blocks on the least covered spatial axis
               until they reach ``max_chunk_nbytes`` (6 MiB), the image extent,
               or 1.5x the patch size per axis.

        For 2 and 3 spatial axes the layout is identical to nnU-Net's.

        Args:
            image_size (Tuple[int, ...]): Full array shape.
            patch_size (Tuple[int, ...]): Patch size over spatial axes only.
                A 1 keeps that axis at chunk size 1 (e.g. ``(1, 256, 256)``
                for 2D slices of a 3D image).
            spatial_axis_mask (Optional[list[bool]]): Mask indicating for every
                array axis whether it is spatial. If omitted, all axes are
                treated as spatial.
            bytes_per_pixel (int): Number of bytes per element. Defaults to 4
                for float32.
            max_block_nbytes (int): Largest block in bytes.
            max_chunk_nbytes (int): Largest chunk in bytes.

        Returns:
            Tuple[List[int], List[int]]: Recommended chunk size and block size.
        """
        return _layout.comp_blosc2_params(
            image_size=tuple(int(v) for v in image_size),
            patch_size=tuple(int(v) for v in patch_size),
            spatial_axis_mask=spatial_axis_mask,
            bytes_per_pixel=bytes_per_pixel,
            max_block_nbytes=max_block_nbytes,
            max_chunk_nbytes=max_chunk_nbytes,
        )

    def _open(
            self,
            filepath: Union[str, Path],
            mode: str = 'r',
            mmap_mode: str = 'r',
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Internal open method. Open an existing MLArray file with memory mapping.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be either 'r' (default) or 'a'. Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'r' (default), 'r+', 'c' or None. Leave at default if unsure.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for opening.
        """
        if not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.{MLARRAY_SUFFIX}' as extension.")
        self.filepath = str(filepath)
        
        if not Path(filepath).is_file():
            raise RuntimeError(f"No MLArray file exists under {filepath}.")
        if mode not in ('r', 'a'):
            raise RuntimeError("mode must be one of the following: 'r', 'a'")
        if mmap_mode not in ('r', 'r+', 'c', None):
            raise RuntimeError("mmap_mode must be one of the following: 'r', 'r+', 'c', None")
        
        self.support_metadata = True

        dparams = MLArray._resolve_dparams(dparams)
        
        _apply_default_read_ahead()
        self._store = blosc2.open(urlpath=str(filepath), dparams=dparams, mode=mode, mmap_mode=mmap_mode)
        self._backend = "blosc2"
        self._read_meta()
        self._update_blosc2_meta()
        self.mode = mode
        self.mmap_mode = mmap_mode

    def _create(            
            self,
            filepath: Union[str, Path],
            shape: Union[List, Tuple, np.ndarray],
            dtype: np.dtype,
            meta: Optional[Union[Dict, Meta]] = None,
            mode: str = 'w',
            mmap_mode: str = 'w+',
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Internal create method. Create a new MLArray file with memory mapping.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            shape (Union[List, Tuple, np.ndarray]): Shape of the array
                to create. If provided, a new file is created. Length must match
                the full array dimensionality (including non-spatial axes if present).
            dtype (np.dtype): Numpy dtype for a newly created array.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be 'w' (default). Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'w+' (default) or None. Leave at default if unsure.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Provide an int for isotropic sizes or
                a list/tuple with length equal to the number of spatial
                dimensions. Use "default" to use the default patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.LZ4HC,
                'clevel': 5, 'nthreads': 4}`` with blosc2's default filter
                (SHUFFLE); the data is not known yet, so no filter is chosen
                from it.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for creation.
        """
        if not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.{MLARRAY_SUFFIX}' as extension.")
        self.filepath = str(filepath)

        if mode != 'w':
            raise RuntimeError("mode must be 'w'.")
        if mmap_mode not in ('w+', None):
            raise RuntimeError("mmap_mode must be one of the following: 'w+', None")
    
        shape = self._normalize_shape(shape)
        self._validate_and_add_meta(meta, has_array=True)
        spatial_axis_mask = [True] * len(shape) if self.meta.spatial.axis_labels is None else _spatial_axis_mask(self.meta.spatial.axis_labels)
        with _meta_internal_write():
            self.meta.blosc2 = self._comp_and_validate_blosc2_meta(self.meta.blosc2, patch_size, chunk_size, block_size, shape, np.dtype(dtype).itemsize, spatial_axis_mask, cparams, dparams)
            self.meta._has_array.has_array = True
        
        self.support_metadata = True
        
        self._store = blosc2.empty(shape=shape, dtype=np.dtype(dtype), urlpath=str(filepath), chunks=self.meta.blosc2.chunk_size, blocks=self.meta.blosc2.block_size, cparams=MLArray._resolve_cparams(self.meta.blosc2.cparams), dparams=MLArray._resolve_dparams(self.meta.blosc2.dparams), mode=mode, mmap_mode=mmap_mode)
        self._backend = "blosc2"
        self.mode = mode
        self.mmap_mode = mmap_mode
        self._validate_and_add_meta(self.meta)
        self._update_blosc2_meta()
        self._write_metadata()

    def _load(
            self,
            filepath: Union[str, Path], 
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Internal MLArray load method. Load a MLArray file without memory mapping.

        Args:
            filepath (Union[str, Path]): Path to the MLArray file to be loaded.
                The filepath needs to have the extension ".mla".
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when loading/accessing compressed
                chunks. If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is not ".mla".
        """
        if not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.{MLARRAY_SUFFIX}' as extension.")
        self.filepath = str(filepath)
        self.support_metadata = True
        dparams = MLArray._resolve_dparams(dparams)
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
        self._store = in_memory
        self._backend = "blosc2"
        self.mode = None
        self.mmap_mode = None
        self._read_meta()        
        self._update_blosc2_meta()
        if not compressed:
            self._store = np.asarray(self._store[...])
            self._backend = "numpy"

    def _asarray(
            self,
            array: Union[np.ndarray],
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Internal MLArray asarray method.

        Args:
            array (Union[np.ndarray]): Input array to convert to MLArray.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Provide an int for isotropic sizes or
                a list/tuple with length equal to the number of dimensions.
                Use "default" to use the default patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Cannot be combined with an
                explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Cannot be combined with an
                explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when creating the in-memory Blosc2
                container (for example codec, compression level, and filters).
                Controls how data is compressed when stored. If None, defaults
                to ``{'codec': blosc2.Codec.LZ4HC, 'clevel': 5, 'nthreads': 4}``,
                with the filter (NOFILTER or SHUFFLE) chosen per array from its
                data as nnU-Net does.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            TypeError: If ``array`` is not a NumPy ndarray.
            ValueError: If ``meta`` is not None, dict, or Meta.
            RuntimeError: If patch/chunk/block arguments are inconsistent.
            NotImplementedError: If automatic patch optimization is not
                implemented for the provided dimensionality.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")
        self._construct_in_memory(
            constructor="asarray",
            source_array=np.require(array, requirements="C"),
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=compressed,
        )

    def _construct_in_memory(
            self,
            constructor: str,
            shape: Optional[Union[int, List, Tuple, np.ndarray]] = None,
            dtype: Optional[np.dtype] = None,
            source_array: Optional[np.ndarray] = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
            constructor_kwargs: Optional[dict[str, Any]] = None,
        ):
        """Internal generic constructor for in-memory Blosc2-backed MLArrays.

        Args:
            shape (Optional[Union[int, List, Tuple, np.ndarray]]): Target
                array shape, required when ``source_array`` is None.
            dtype (Optional[np.dtype]): Target dtype, required when
                ``source_array`` is None.
            constructor (str): Constructor operation name.
            source_array (Optional[np.ndarray]): Source array that should be
                converted into an in-memory Blosc2-backed store.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters. If None, defaults to
                ``{'codec': blosc2.Codec.LZ4HC, 'clevel': 5, 'nthreads': 4}``.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters. If None, defaults to
                ``{'nthreads': 1}``.

        Raises:
            ValueError: If constructor inputs are inconsistent.
        """
        constructor_kwargs = {} if constructor_kwargs is None else dict(constructor_kwargs)

        if source_array is not None:
            if shape is not None or dtype is not None:
                raise ValueError(
                    "shape/dtype must not be set when source_array is provided."
                )
            shape = self._normalize_shape(source_array.shape)
            dtype = np.dtype(source_array.dtype)
        else:
            if shape is None or dtype is None:
                raise ValueError(
                    "shape and dtype must be provided when source_array is None."
                )
            shape = self._normalize_shape(shape)
            dtype = np.dtype(dtype)

        self._validate_and_add_meta(meta, has_array=True)
        spatial_axis_mask = (
            [True] * len(shape)
            if self.meta.spatial.axis_labels is None
            else _spatial_axis_mask(self.meta.spatial.axis_labels)
        )

        with _meta_internal_write():
            self.meta.blosc2 = self._comp_and_validate_blosc2_meta(
                self.meta.blosc2,
                patch_size,
                chunk_size,
                block_size,
                shape,
                dtype.itemsize,
                spatial_axis_mask,
                cparams,
                dparams,
                array=source_array,
            )
            self.meta._has_array.has_array = True

        backend_lib = blosc2 if compressed else np
        blosc2_storage_kwargs = {}
        if compressed:
            blosc2_storage_kwargs = {
                "chunks": self.meta.blosc2.chunk_size,
                "blocks": self.meta.blosc2.block_size,
                "cparams": MLArray._resolve_cparams(self.meta.blosc2.cparams),
                "dparams": MLArray._resolve_dparams(self.meta.blosc2.dparams),
            }

        if constructor == "asarray":
            if compressed:
                self._store = blosc2.asarray(source_array, **blosc2_storage_kwargs)
            else:
                self._store = np.asarray(source_array, dtype=dtype)
        else:
            call_kwargs = self._build_constructor_call_kwargs(
                constructor=constructor,
                shape=shape,
                dtype=dtype,
                constructor_kwargs=constructor_kwargs,
            )

            if not compressed and constructor in ("arange", "linspace"):
                self._store = self._construct_numpy_range(constructor, call_kwargs)
            else:
                func = getattr(backend_lib, constructor, None)
                if func is None:
                    raise ValueError(f"Unknown constructor '{constructor}'.")
                if compressed:
                    call_kwargs.update(blosc2_storage_kwargs)
                self._store = func(**call_kwargs)

        self._backend = "blosc2" if compressed else "numpy"

        self.support_metadata = True

        self._validate_and_add_meta(self.meta)
        self._update_blosc2_meta()

    @staticmethod
    def _build_constructor_call_kwargs(
            constructor: str,
            shape: Tuple[int, ...],
            dtype: np.dtype,
            constructor_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
        """Build constructor kwargs shared by NumPy and Blosc2 backends."""
        if constructor in ("empty", "zeros", "ones"):
            return {"shape": shape, "dtype": dtype}
        if constructor == "full":
            return {
                "shape": shape,
                "fill_value": constructor_kwargs["fill_value"],
                "dtype": dtype,
            }
        if constructor == "arange":
            return {
                "start": constructor_kwargs["start"],
                "stop": constructor_kwargs["stop"],
                "step": constructor_kwargs["step"],
                "dtype": dtype,
                "shape": shape,
                "c_order": constructor_kwargs.get("c_order", True),
            }
        if constructor == "linspace":
            return {
                "start": constructor_kwargs["start"],
                "stop": constructor_kwargs["stop"],
                "num": constructor_kwargs["num"],
                "dtype": dtype,
                "shape": shape,
                "endpoint": constructor_kwargs.get("endpoint", True),
                "c_order": constructor_kwargs.get("c_order", True),
            }
        raise ValueError(f"Unknown constructor '{constructor}'.")

    @staticmethod
    def _construct_numpy_range(
            constructor: str,
            call_kwargs: dict[str, Any],
        ) -> np.ndarray:
        """Construct NumPy arange/linspace arrays and apply shape/order reshaping."""
        shape = call_kwargs.pop("shape")
        c_order = call_kwargs.pop("c_order", True)
        func = getattr(np, constructor)
        arr = func(**call_kwargs)
        return np.reshape(arr, shape, order="C" if c_order else "F")

    def _comp_and_validate_blosc2_meta(self, meta_blosc2, patch_size, chunk_size, block_size, shape, dtype_itemsize, spatial_axis_mask, cparams, dparams, array=None):
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
            MetaBlosc2: Validated Blosc2 metadata instance.
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

        patch_size = MLArray._normalize_patch_size(patch_size, num_spatial_axes)

        if patch_size is not None and num_spatial_axes == 0:
            raise RuntimeError(
                "Automatic patch-size optimization requires at least one spatial axis. "
                "Set patch_size=None and provide chunk_size/block_size manually, "
                "or let Blosc2 determine the layout."
            )

        if patch_size is not None:
            chunk_size, block_size = MLArray.comp_blosc2_params(shape, patch_size, spatial_axis_mask, bytes_per_pixel=dtype_itemsize)

        choose_filter = array is not None and chunk_size is not None and MLArray._is_default_cparams(cparams)
        cparams = MLArray._resolve_cparams(cparams)
        dparams = MLArray._resolve_dparams(dparams)
        if choose_filter:
            cparams = dict(cparams)
            cparams["filters"] = [_layout.select_blosc2_filter(
                array, chunk_size, block_size, codec=cparams["codec"], clevel=cparams["clevel"],
            )]

        meta_blosc2 = MetaBlosc2(
            chunk_size=chunk_size,
            block_size=block_size,
            patch_size=patch_size,
            cparams=cparams,
            dparams=dparams,
        )
        meta_blosc2._validate_and_cast(ndims=len(shape), spatial_ndims=num_spatial_axes)
        return meta_blosc2

    def _validate_and_add_meta(
            self,
            meta,
            spacing=None,
            origin=None,
            direction=None,
            affine=None,
            axis_labels=None,
            has_array=None,
            validate=True,
        ):
        """Validate and attach metadata to the MLArray instance.

        Args:
            meta (Optional[Union[dict, Meta]]): Metadata to attach. Dicts are
                interpreted as ``meta.source`` fields.
            spacing (Optional[Union[List, Tuple, np.ndarray]]): Spacing per
                spatial axis.
            origin (Optional[Union[List, Tuple, np.ndarray]]): Origin per
                spatial axis.
            direction (Optional[Union[List, Tuple, np.ndarray]]): Direction
                cosine matrix with shape (ndims, ndims).
            affine (Optional[Union[List, Tuple, np.ndarray]]): Homogeneous
                affine matrix with shape (spatial_ndims + 1, spatial_ndims + 1).
            axis_labels (Optional[List[Union[str, AxisLabel]]]): Per-axis labels or roles. Length must match ndims.
            has_array (Optional[bool]): Explicitly set whether array data is
                present. When True, metadata is validated with array-dependent
                shape information.

        Raises:
            ValueError: If ``meta`` is not None, dict, or Meta.
        """
        if meta is not None:
            if not isinstance(meta, (dict, Meta)):
                raise ValueError("Meta must be None, a dict or a Meta object.")
            if isinstance(meta, dict):
                meta = Meta(source=meta)
        else:
            meta = Meta()
        self._meta = meta
        with _meta_internal_write():
            self.meta._mlarray_version = MLARRAY_VERSION

        if affine is not None and (
            spacing is not None or origin is not None or direction is not None
        ):
            raise ValueError(
                "affine cannot be provided together with spacing, origin, or direction."
            )

        if spacing is not None:
            self.meta.spatial.spacing = spacing
        if origin is not None:
            self.meta.spatial.origin = origin
        if direction is not None:
            self.meta.spatial.direction = direction
        if affine is not None:
            self.meta.spatial.affine = affine
        if axis_labels is not None:
            self.meta.spatial.axis_labels = axis_labels
        if has_array == True:
            with _meta_internal_write():
                self.meta._has_array.has_array = True
        if self.meta._has_array.has_array:
            with _meta_internal_write():
                if self._store is not None and self._store.shape != (0,):
                    self.meta.spatial.shape = list(self._store.shape)
        if validate:
            with _meta_internal_write():
                self.meta.spatial._validate_and_cast(ndims=self.ndim, spatial_ndims=self.spatial_ndim)

    def _update_blosc2_meta(self):
        """Sync Blosc2 chunk and block sizes into metadata.

        Updates ``self.meta.blosc2`` from the underlying store when the array
        is present.
        """
        if self._backend != "blosc2":
            return
        if self.support_metadata and self.meta._has_array.has_array == True:
            with _meta_internal_write():
                self.meta.blosc2.chunk_size = list(self._store.chunks)
                self.meta.blosc2.block_size = list(self._store.blocks)
                self.meta.blosc2._validate_and_cast(
                    ndims=self.ndim,
                    spatial_ndims=self.spatial_ndim,
                )

    def _read_meta(self):
        """Read MLArray metadata from the underlying store, if available."""
        meta = Meta()
        if self.support_metadata and self._backend == "blosc2":
            meta = self._store.vlmeta["mlarray"]
            meta = pickle.loads(gzip.decompress(meta))
            meta = Meta.from_mapping(meta)
        self._validate_and_add_meta(meta)

    def _write_metadata(self):
        """Write MLArray metadata to the underlying store if it is writable."""
        is_writable = False
        if self.support_metadata:
            if self.mode in ('a', 'w') and self.mmap_mode is None:
                is_writable = True
            elif self.mmap_mode in ('r+', 'w+'):
                is_writable = True

        if not is_writable:
            return

        if self._backend != "blosc2":
            return

        self._store.vlmeta["mlarray"] = self._serialize_metadata()

    def _serialize_metadata(self, meta: Optional[Meta] = None) -> bytes:
        """Serialize MLArray metadata (default: ``self.meta``) into the bytes stored in vlmeta."""
        meta = (self.meta if meta is None else meta).to_mapping()
        if not is_serializable(meta):
            raise RuntimeError("Metadata is not serializable.")
        return gzip.compress(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _blosc2_store_for_save(self) -> Tuple[Any, Meta]:
        """Return the Blosc2 store and metadata to write, without changing this instance.

        Blosc2-backed arrays are saved as they are. NumPy-backed arrays are
        compressed into a temporary store whose layout follows the recorded
        ``meta.blosc2`` (patch size first, then chunk/block size, then the
        default patch size); the instance stays NumPy-backed.
        """
        if self._store is None or not self.meta._has_array.has_array:
            return blosc2.empty((0,)), self.meta

        if self._backend == "blosc2":
            self._update_blosc2_meta()
            return self._store, self.meta

        array = np.ascontiguousarray(self._store)
        meta = deepcopy(self.meta)
        shape = self._normalize_shape(array.shape)
        spatial_axis_mask = (
            [True] * len(shape)
            if meta.spatial.axis_labels is None
            else _spatial_axis_mask(meta.spatial.axis_labels)
        )
        with _meta_internal_write():
            meta.blosc2 = self._comp_and_validate_blosc2_meta(
                meta.blosc2,
                patch_size="default",
                chunk_size=None,
                block_size=None,
                shape=shape,
                dtype_itemsize=np.dtype(array.dtype).itemsize,
                spatial_axis_mask=spatial_axis_mask,
                cparams=meta.blosc2.cparams,
                dparams=meta.blosc2.dparams,
                array=array,
            )
        store = blosc2.asarray(
            array,
            chunks=meta.blosc2.chunk_size,
            blocks=meta.blosc2.block_size,
            cparams=MLArray._resolve_cparams(meta.blosc2.cparams),
            dparams=MLArray._resolve_dparams(meta.blosc2.dparams),
        )
        with _meta_internal_write():
            meta.blosc2.chunk_size = list(store.chunks)
            meta.blosc2.block_size = list(store.blocks)
            meta.blosc2._validate_and_cast(ndims=len(shape), spatial_ndims=sum(spatial_axis_mask))
        return store, meta

    @staticmethod
    def _normalize_patch_size(patch_size, num_spatial_axes: int) -> Optional[List[int]]:
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

    @staticmethod
    def _normalize_shape(shape: Union[int, List, Tuple, np.ndarray]) -> Tuple[int, ...]:
        if isinstance(shape, (int, np.integer)):
            return (int(shape),)
        return tuple(int(v) for v in shape)

    def _replace_with_converted(self, other: "MLArray") -> None:
        """Replace in-memory state from another MLArray and detach file state."""
        old_store = self._store
        self._store = other._store
        self._backend = other._backend
        self._meta = other._meta
        self.support_metadata = other.support_metadata
        self.filepath = None
        self.mode = None
        self.mmap_mode = None
        del old_store

    def _resolve_like_input(self, x, dtype, meta):
        if isinstance(x, MLArray):
            if x._store is None or x.meta is None or x.meta._has_array.has_array is False:
                raise TypeError("Input MLArray has no array data loaded.")
            shape = self._normalize_shape(x.shape)
            src_dtype = x.dtype
            if meta is None:
                meta = deepcopy(x.meta)
        elif hasattr(x, "shape") and hasattr(x, "dtype"):
            shape = self._normalize_shape(x.shape)
            src_dtype = x.dtype
        else:
            raise TypeError(
                "x must be an MLArray or an array-like object with shape and dtype."
            )

        dtype = src_dtype if dtype is None else dtype
        return shape, np.dtype(dtype), meta

    @staticmethod
    def _default_cparams() -> Dict:
        """MLArray's default compression parameters (nnU-Net's: LZ4HC level 5, 4 threads)."""
        return {
            "codec": _layout.DEFAULT_CODEC,
            "clevel": _layout.DEFAULT_CLEVEL,
            "nthreads": _layout.DEFAULT_COMPRESS_NTHREADS,
        }

    @staticmethod
    def _is_default_cparams(cparams) -> bool:
        """True for the default compression parameters without a chosen filter."""
        if cparams is None:
            return True
        if isinstance(cparams, blosc2.CParams) or not hasattr(cparams, "keys"):
            return False
        cparams = dict(cparams)
        default = MLArray._default_cparams()
        if set(cparams) != set(default):
            return False
        try:
            return (blosc2.Codec(cparams["codec"]) == default["codec"]
                    and cparams["clevel"] == default["clevel"]
                    and cparams["nthreads"] == default["nthreads"])
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _resolve_cparams(cparams: Optional[Union[Dict, blosc2.CParams]]) -> Union[Dict, blosc2.CParams]:
        """Resolve compression params with MLArray defaults."""
        if cparams is None:
            return MLArray._default_cparams()
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

    @staticmethod
    def _resolve_dparams(dparams: Optional[Union[Dict, blosc2.DParams]]) -> Union[Dict, blosc2.DParams]:
        """Resolve decompression params with MLArray defaults."""
        if dparams is None:
            return {"nthreads": 1}
        return dparams
