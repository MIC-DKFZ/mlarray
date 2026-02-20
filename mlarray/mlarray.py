from copy import deepcopy
import numpy as np
import blosc2
import math
from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path
import os
from mlarray.meta import Meta, MetaBlosc2, AxisLabel, _spatial_axis_mask
from mlarray.utils import is_serializable

MLARRAY_SUFFIX = "mla"
MLARRAY_VERSION = "v0"
MLARRAY_DEFAULT_PATCH_SIZE = 192


class MLArray:
    def __init__(
            self,
            array: Optional[Union[np.ndarray, str, Path]] = None,
            spacing: Optional[Union[List, Tuple, np.ndarray]] = None,
            origin: Optional[Union[List, Tuple, np.ndarray]] = None,
            direction: Optional[Union[List, Tuple, np.ndarray]] = None,
            meta: Optional[Union[Dict, Meta]] = None,
            axis_labels: Optional[List[Union[str, AxisLabel]]] = None,
            copy: Optional['MLArray'] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
        ) -> None:
        """Initializes a MLArray instance.

        The MLArray file format (".mla") is a Blosc2-compressed container
        with standardized metadata support for N-dimensional medical images.

        Args:
            array (Optional[Union[np.ndarray, str, Path]]): Input data or file
                path. Use a numpy ndarray for in-memory arrays, or a string/Path
                to load a ".b2nd" or ".mla" file. If None, an empty MLArray
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
                Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Ignored when ``patch_size`` is provided.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters used when creating in-memory compressed array data.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used for chunk access.
        """
        self.filepath = None
        self.support_metadata = None
        self.mode = None
        self.mmap_mode = None
        self.meta = None
        self._store = None
        if isinstance(array, (str, Path)) and (
            spacing is not None
            or origin is not None
            or direction is not None
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
                "Spacing, origin, direction, meta, axis_labels, copy, patch_size, "
                "chunk_size, block_size, cparams or dparams cannot be set when "
                "array is a filepath."
            )
        if isinstance(array, (str, Path)):
            self._load(array)
        else:
            self._validate_and_add_meta(meta, spacing, origin, direction, axis_labels, False, validate=False)
            if array is not None:
                self._asarray(
                    array,
                    meta=self.meta,
                    patch_size=patch_size,
                    chunk_size=chunk_size,
                    block_size=block_size,
                    cparams=cparams,
                    dparams=dparams,
                )
                has_array = True
            else:
                self._store = blosc2.empty((0,))
                has_array = False
            if copy is not None:
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

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".b2nd" or ".mla".
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

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".b2nd" or ".mla".
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
                dimensions. Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Ignored when ``patch_size`` is provided.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.LZ4HC,
                'clevel': 8}``.
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
        ):
        """Loads a MLArray file as a whole. Does not use memory-mapping. Both MLArray ('.mla') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Path to the MLArray file to be loaded.
                The filepath needs to have the extension ".b2nd" or ".mla".
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when loading/accessing compressed
                chunks. If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mla".
        """
        class_instance = cls()
        class_instance._load(filepath, dparams)
        return class_instance

    @classmethod
    def empty(
            cls,
            shape: Union[int, List, Tuple, np.ndarray],
            dtype: np.dtype,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
        ):
        """Create an in-memory MLArray with uninitialized values.

        Args:
            shape (Union[int, List, Tuple, np.ndarray]): Shape of the output
                array.
            dtype (np.dtype): Numpy dtype for the output array.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Ignored when ``patch_size`` is provided.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters used when writing chunks.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when reading chunks.

        Returns:
            MLArray: A newly created in-memory MLArray instance.
        """
        class_instance = cls()
        class_instance._construct_in_memory(
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.empty(**kwargs),
        )
        return class_instance

    @classmethod
    def zeros(
            cls,
            shape: Union[int, List, Tuple, np.ndarray],
            dtype: np.dtype,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
        ):
        """Create an in-memory MLArray filled with zeros.

        Args:
            shape (Union[int, List, Tuple, np.ndarray]): Shape of the output
                array.
            dtype (np.dtype): Numpy dtype for the output array.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Ignored when ``patch_size`` is provided.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Returns:
            MLArray: A newly created in-memory MLArray instance.
        """
        class_instance = cls()
        class_instance._construct_in_memory(
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.zeros(**kwargs),
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
                Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Ignored when ``patch_size`` is provided.
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
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.ones(**kwargs),
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
                Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Ignored when ``patch_size`` is provided.
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
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.full(fill_value=fill_value, **kwargs),
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

        num = int((stop - start) / step)
        if shape is None:
            shape = (max(num, 0),)
        else:
            shape = cls._normalize_shape(shape)
            if math.prod(shape) != num:
                raise ValueError(
                    "The shape is not consistent with the start, stop and step values"
                )

        if dtype is None:
            dtype = (
                blosc2.DEFAULT_FLOAT
                if np.any([np.issubdtype(type(d), float) for d in (start, stop, step)])
                else blosc2.DEFAULT_INT
            )

        class_instance = cls()
        class_instance._construct_in_memory(
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.arange(
                start=start,
                stop=stop,
                step=step,
                c_order=c_order,
                **kwargs,
            ),
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
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.linspace(
                start=start,
                stop=stop,
                num=num,
                endpoint=endpoint,
                c_order=c_order,
                **kwargs,
            ),
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
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
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
                of dimensions, or None to let Blosc2 decide. Ignored when
                patch_size is not None.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Ignored when
                patch_size is not None.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when creating the in-memory Blosc2
                container (for example codec, compression level, and filters).
                Controls how data is compressed when stored. If None, defaults
                to ``{'codec': blosc2.Codec.LZ4HC, 'clevel': 8}``.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

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
        class_instance._asarray(array, meta, patch_size, chunk_size, block_size, cparams, dparams)
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
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.empty(**kwargs),
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
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.zeros(**kwargs),
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
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.ones(**kwargs),
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
            shape=shape,
            dtype=dtype,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.full(fill_value=fill_value, **kwargs),
        )
        return class_instance

    def save(
            self,
            filepath: Union[str, Path],
        ):
        """Saves the array to a MLArray file. Both MLArray ('.mla') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when saving.

        Args:
            filepath (Union[str, Path]): Path to save the file. Must end with
                ".b2nd" or ".mla".

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mla".
        """
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")

        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")

        if Path(filepath).is_file():
            os.remove(str(filepath))
        
        self._write_metadata(force=True)
        self._store.save(str(filepath))
        self._update_blosc2_meta()
        self.mode = None
        self.mmap_mode = None

    def close(self):
        """Flush metadata and close the underlying store.

        After closing, the MLArray instance no longer has an attached array.
        """
        self._write_metadata()
        self._store = None
        self.filepath = None
        self.support_metadata = None
        self.mode = None  
        self.mmap_mode = None
        self.meta = None

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

    def __array__(self, dtype=None):
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
        arr = np.asarray(self._store)
        if dtype is not None:
            return arr.astype(dtype)
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
    def spacing(self):
        """Returns the image spacing.

        Returns:
            list: Spacing per spatial axis with length equal to the number of
            spatial dimensions.
        """
        return self.meta.spatial.spacing

    @property
    def origin(self):
        """Returns the image origin.

        Returns:
            list: Origin per spatial axis with length equal to the number of
            spatial dimensions.
        """
        return self.meta.spatial.origin

    @property
    def direction(self):
        """Returns the image direction.

        Returns:
            list: Direction cosine matrix with shape (ndims, ndims).
        """
        return self.meta.spatial.direction

    @property
    def affine(self) -> np.ndarray:
        """Computes the affine transformation matrix for the image.

        Returns:
            list: Affine matrix with shape (ndims + 1, ndims + 1), or None if
                no array is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        spacing  = np.array(self.spacing) if self.spacing is not None else np.ones(self.spatial_ndim)
        origin  = np.array(self.origin) if self.origin is not None else np.zeros(self.spatial_ndim)
        direction = np.array(self.direction) if self.direction is not None else np.eye(self.spatial_ndim)
        affine = np.eye(self.spatial_ndim + 1)
        affine[:self.spatial_ndim, :self.spatial_ndim] = direction @ np.diag(spacing)
        affine[:self.spatial_ndim, self.spatial_ndim] = origin
        return affine.tolist()

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
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        return self._store.shape

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
            int: Number of dimensions, or None if no array is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        return len(self._store.shape)

    @property
    def spatial_ndim(self) -> int:
        """Returns the number of spatial dimensions.

        Returns:
            int: Number of spatial dimensions, or None if no array is loaded.
        """
        if self._store is None or self.meta._has_array.has_array == False:
            return None
        ndim = len(self._store.shape)
        if self.meta.spatial._num_spatial_axes is not None:
            ndim = self.meta.spatial._num_spatial_axes
        return ndim

    @classmethod
    def comp_blosc2_params(
            cls,
            image_size: Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]],
            patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
            spatial_axis_mask: Optional[list[bool]] = None,
            bytes_per_pixel: int = 4,  # 4 byte are float32
            l1_cache_size_per_core_in_bytes: int = 32768,  # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
            l3_cache_size_per_core_in_bytes: int = 1441792, # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
            safety_factor: float = 0.8  # we dont will the caches to the brim. 0.8 means we target 80% of the caches
        ):
        """
        Computes a recommended block and chunk size for saving arrays with Blosc v2.

        Blosc2 NDIM documentation:
        "Having a second partition allows for greater flexibility in fitting different partitions to different CPU cache levels. 
        Typically, the first partition (also known as chunks) should be sized to fit within the L3 cache, 
        while the second partition (also known as blocks) should be sized to fit within the L2 or L1 caches, 
        depending on whether the priority is compression ratio or speed." 
        (Source: https://www.blosc.org/posts/blosc2-ndim-intro/)

        Our approach is not fully optimized for this yet. 
        Currently, we aim to fit the uncompressed block within the L1 cache, accepting that it might occasionally spill over into L2, which we consider acceptable.

        Note: This configuration is specifically optimized for nnU-Net data loading, where each read operation is performed by a single core, so multi-threading is not an option.

        The default cache values are based on an older Intel 4110 CPU with 32KB L1, 128KB L2, and 1408KB L3 cache per core. 
        We haven't further optimized for modern CPUs with larger caches, as our data must still be compatible with the older systems.

        Args:
            image_size (Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]):
                Image shape. Use a 2D, 3D, or 4D size; 2D/3D inputs are
                internally expanded to 4D (with non-spatial axes first).
            patch_size (Union[Tuple[int, int], Tuple[int, int, int]]): Patch
                size for spatial dimensions. Use a 2-tuple (x, y) or 3-tuple
                (x, y, z).
            spatial_axis_mask (Optional[list[bool]]): Mask indicating for every axis whether it is spatial or not.
            bytes_per_pixel (int): Number of bytes per element. Defaults to 4
                for float32.
            l1_cache_size_per_core_in_bytes (int): L1 cache per core in bytes.
            l3_cache_size_per_core_in_bytes (int): L3 cache per core in bytes.
            safety_factor (float): Safety factor to avoid filling caches.

        Returns:
            Tuple[List[int], List[int]]: Recommended chunk size and block size.
        """
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
                raise RuntimeError("Automatic blosc2 optimization currently only supports one non-spatial axis. Please set chunk and block size manually.")
            non_spatial_axis = next((i for i, v in enumerate(non_spatial_axis_mask) if v), None)
            if non_spatial_axis is not None:
                image_size = _move_index_list(image_size, non_spatial_axis+num_squeezes, 0)

        if len(image_size) != 4:
            raise RuntimeError("Image size must be 4D.")
        
        if not (len(patch_size) == 2 or len(patch_size) == 3):
            raise RuntimeError("Patch size must be 2D or 3D.")

        non_spatial_size = image_size[0]
        if len(patch_size) == 2:
            patch_size = [1, *patch_size]
        patch_size = np.array(patch_size)
        block_size = np.array((non_spatial_size, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            if patch_size[0] == 1 and all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
                # chunk size should not exceed patch size * 1.5 on average
                chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
                break
        # better safe than sorry
        chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

        if non_spatial_axis is not None:
            block_size = _move_index_list(block_size, 0, non_spatial_axis+num_squeezes)
            chunk_size = _move_index_list(chunk_size, 0, non_spatial_axis+num_squeezes)

        block_size = block_size[num_squeezes:]
        chunk_size = chunk_size[num_squeezes:]

        return [int(value) for value in chunk_size], [int(value) for value in block_size]

    def _open(
            self,
            filepath: Union[str, Path],
            mode: str = 'r',
            mmap_mode: str = 'r',
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Internal open method. Open an existing MLArray file with memory mapping.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".b2nd" or ".mla".
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
        self.filepath = str(filepath)
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")
        
        if not Path(filepath).is_file():
            raise RuntimeError(f"No MLArray file exists under {filepath}.")
        if mode not in ('r', 'a'):
            raise RuntimeError("mode must be one of the following: 'r', 'a'")
        if mmap_mode not in ('r', 'r+', 'c', None):
            raise RuntimeError("mmap_mode must be one of the following: 'r', 'r+', 'c', None")
        
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")

        dparams = MLArray._resolve_dparams(dparams)
        
        self._store = blosc2.open(urlpath=str(filepath), dparams=dparams, mode=mode, mmap_mode=mmap_mode)
        self._read_meta()
        self._update_blosc2_meta()
        self.mode = mode
        self.mmap_mode = mmap_mode
        self._write_metadata()

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

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".b2nd" or ".mla".
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
                dimensions. Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Ignored when ``patch_size`` is provided.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.LZ4HC,
                'clevel': 8}``.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for creation.
        """
        self.filepath = str(filepath)
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")

        if mode != 'w':
            raise RuntimeError("mode must be 'w'.")
        if mmap_mode not in ('w+', None):
            raise RuntimeError("mmap_mode must be one of the following: 'w+', None")
    
        self._validate_and_add_meta(meta, has_array=True)
        spatial_axis_mask = [True] * len(shape) if self.meta.spatial.axis_labels is None else _spatial_axis_mask(self.meta.spatial.axis_labels)
        self.meta.blosc2 = self._comp_and_validate_blosc2_meta(self.meta.blosc2, patch_size, chunk_size, block_size, shape, np.dtype(dtype).itemsize, spatial_axis_mask)   
        self.meta._has_array.has_array = True
        
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")

        cparams = MLArray._resolve_cparams(cparams)
        dparams = MLArray._resolve_dparams(dparams)
        
        self._store = blosc2.empty(shape=shape, dtype=np.dtype(dtype), urlpath=str(filepath), chunks=self.meta.blosc2.chunk_size, blocks=self.meta.blosc2.block_size, cparams=cparams, dparams=dparams, mmap_mode=mmap_mode)
        self._update_blosc2_meta()
        self.mode = mode
        self.mmap_mode = mmap_mode
        self._validate_and_add_meta(self.meta)
        self._write_metadata()

    def _load(
            self,
            filepath: Union[str, Path], 
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
        ):
        """Internal MLArray load method. Loads a MLArray file. Both MLArray ('.mla') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Path to the MLArray file to be loaded.
                The filepath needs to have the extension ".b2nd" or ".mla".
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when loading/accessing compressed
                chunks. If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mla".
        """
        self.filepath = str(filepath)
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")
        dparams = MLArray._resolve_dparams(dparams)
        ondisk = blosc2.open(str(filepath), dparams=dparams, mode="r")
        cframe = ondisk.to_cframe()
        self._store = blosc2.ndarray_from_cframe(cframe, copy=True)
        self.mode = None
        self.mmap_mode = None
        self._read_meta()        
        self._update_blosc2_meta()

    def _asarray(
            self,
            array: Union[np.ndarray],
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
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
                of dimensions, or None to let Blosc2 decide. Ignored when
                patch_size is not None.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Ignored when
                patch_size is not None.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when creating the in-memory Blosc2
                container (for example codec, compression level, and filters).
                Controls how data is compressed when stored. If None, defaults
                to ``{'codec': blosc2.Codec.LZ4HC, 'clevel': 8}``.
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
            source_array=array,
            meta=meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            store_builder=lambda **kwargs: blosc2.asarray(
                kwargs["source_array"],
                chunks=kwargs["chunks"],
                blocks=kwargs["blocks"],
                cparams=kwargs["cparams"],
                dparams=kwargs["dparams"],
            ),
        )

    def _construct_in_memory(
            self,
            store_builder,
            shape: Optional[Union[int, List, Tuple, np.ndarray]] = None,
            dtype: Optional[np.dtype] = None,
            source_array: Optional[np.ndarray] = None,
            meta: Optional[Union[Dict, Meta]] = None,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
        ):
        """Internal generic constructor for in-memory Blosc2-backed MLArrays.

        Args:
            shape (Optional[Union[int, List, Tuple, np.ndarray]]): Target
                array shape, required when ``source_array`` is None.
            dtype (Optional[np.dtype]): Target dtype, required when
                ``source_array`` is None.
            store_builder (Callable): Callable receiving normalized kwargs and
                returning a Blosc2 NDArray.
            source_array (Optional[np.ndarray]): Source array that should be
                converted into an in-memory Blosc2-backed store.
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Ignored when ``patch_size`` is provided.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Ignored when ``patch_size`` is provided.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters. If None, defaults to
                ``{'codec': blosc2.Codec.LZ4HC, 'clevel': 8}``.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters. If None, defaults to
                ``{'nthreads': 1}``.

        Raises:
            ValueError: If constructor inputs are inconsistent.
        """
        if source_array is not None:
            if shape is not None or dtype is not None:
                raise ValueError(
                    "shape/dtype must not be set when source_array is provided."
                )
            source_array = np.ascontiguousarray(source_array[...])
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
        self.meta.blosc2 = self._comp_and_validate_blosc2_meta(
            self.meta.blosc2,
            patch_size,
            chunk_size,
            block_size,
            shape,
            dtype.itemsize,
            spatial_axis_mask,
        )
        self.meta._has_array.has_array = True

        cparams = MLArray._resolve_cparams(cparams)
        dparams = MLArray._resolve_dparams(dparams)

        builder_kwargs = dict(
            shape=shape,
            dtype=dtype,
            chunks=self.meta.blosc2.chunk_size,
            blocks=self.meta.blosc2.block_size,
            cparams=cparams,
            dparams=dparams,
        )
        if source_array is not None:
            builder_kwargs["source_array"] = source_array

        self._store = store_builder(**builder_kwargs)

        self.support_metadata = True

        self._update_blosc2_meta()
        self._validate_and_add_meta(self.meta)

    def _comp_and_validate_blosc2_meta(self, meta_blosc2, patch_size, chunk_size, block_size, shape, dtype_itemsize, spatial_axis_mask):
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

        Returns:
            MetaBlosc2: Validated Blosc2 metadata instance.
        """
        num_spatial_axes = sum(spatial_axis_mask)
        num_non_spatial_axes = sum([not b for b in spatial_axis_mask])
        if patch_size is not None and patch_size != "default" and (num_spatial_axes == 1 or num_spatial_axes > 3 or num_non_spatial_axes > 1):
            raise NotImplementedError("Chunk and block size optimization based on patch size is only implemented for 2D and 3D spatial images with at most one further non-spatial axis. Please set the chunk and block size manually or set to None for blosc2 to determine a chunk and block size.")
        if patch_size is not None and patch_size != "default" and (chunk_size is not None or block_size is not None):
            raise RuntimeError("patch_size and chunk_size / block_size cannot both be explicitly set.")
        if (chunk_size is not None and block_size is None) or (chunk_size is None and block_size is not None):
            raise RuntimeError("If either chunk/block size is used then both must be set.")

        if patch_size == "default": 
            if meta_blosc2 is not None and meta_blosc2.patch_size is not None:  # Use previously loaded patch size, when patch size is not explicitly set and a patch size from a previously loaded image exists
                patch_size = meta_blosc2.patch_size
            elif meta_blosc2 is not None and (meta_blosc2.chunk_size is not None):
                chunk_size = meta_blosc2.chunk_size
                block_size = meta_blosc2.block_size
            else:  # Use default patch size, when patch size is not explicitly set and no patch size from a previously loaded image exists
                patch_size = [MLARRAY_DEFAULT_PATCH_SIZE] * num_spatial_axes
        if chunk_size is not None or block_size is not None:
            patch_size = None

        patch_size = [patch_size] * len(shape) if isinstance(patch_size, int) else patch_size

        if patch_size is not None:
            chunk_size, block_size = MLArray.comp_blosc2_params(shape, patch_size, spatial_axis_mask, bytes_per_pixel=dtype_itemsize)

        meta_blosc2 = MetaBlosc2(chunk_size, block_size, patch_size)
        meta_blosc2._validate_and_cast(ndims=len(shape), spatial_ndims=num_spatial_axes)
        return meta_blosc2

    def _validate_and_add_meta(self, meta, spacing=None, origin=None, direction=None, axis_labels=None, has_array=None, validate=True):
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
        self.meta = meta
        self.meta._mlarray_version = MLARRAY_VERSION
        if spacing is not None:
            self.meta.spatial.spacing = spacing
        if origin is not None:
            self.meta.spatial.origin = origin
        if direction is not None:
            self.meta.spatial.direction = direction
        if axis_labels is not None:
            self.meta.spatial.axis_labels = axis_labels
        if has_array == True:
            self.meta._has_array.has_array = True
        if self.meta._has_array.has_array:
            self.meta.spatial.shape = self.shape
        if validate:
            self.meta.spatial._validate_and_cast(ndims=self.ndim, spatial_ndims=self.spatial_ndim)

    def _update_blosc2_meta(self):
        """Sync Blosc2 chunk and block sizes into metadata.

        Updates ``self.meta.blosc2`` from the underlying store when the array
        is present.
        """
        if self.support_metadata and self.meta._has_array.has_array == True:
            self.meta.blosc2.chunk_size = list(self._store.chunks)
            self.meta.blosc2.block_size = list(self._store.blocks)

    def _read_meta(self):
        """Read MLArray metadata from the underlying store, if available."""
        meta = Meta()
        if self.support_metadata:
            meta = self._store.vlmeta["mlarray"]
            meta = Meta.from_mapping(meta)
        self._validate_and_add_meta(meta)

    def _write_metadata(self, force=False):
        """Write MLArray metadata to the underlying store if supported.

        Args:
            force (bool): If True, write even when mmap_mode is read-only.
        """
        is_writable = False
        if self.support_metadata:
            if self.mode in ('a', 'w') and self.mmap_mode is None:
                is_writable = True
            elif self.mmap_mode in ('r+', 'w+'):
                is_writable = True
            elif force:
                is_writable = True
        
        if not is_writable:
            return
        
        metadata = self.meta.to_mapping()
        if not is_serializable(metadata):
            raise RuntimeError("Metadata is not serializable.")
        self._store.vlmeta["mlarray"] = metadata

    @staticmethod
    def _normalize_shape(shape: Union[int, List, Tuple, np.ndarray]) -> Tuple[int, ...]:
        if isinstance(shape, (int, np.integer)):
            return (int(shape),)
        return tuple(int(v) for v in shape)

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
    def _resolve_cparams(cparams: Optional[Union[Dict, blosc2.CParams]]) -> Union[Dict, blosc2.CParams]:
        """Resolve compression params with MLArray defaults."""
        if cparams is None:
            return {"codec": blosc2.Codec.LZ4HC, "clevel": 8}
        return cparams

    @staticmethod
    def _resolve_dparams(dparams: Optional[Union[Dict, blosc2.DParams]]) -> Union[Dict, blosc2.DParams]:
        """Resolve decompression params with MLArray defaults."""
        if dparams is None:
            return {"nthreads": 1}
        return dparams
