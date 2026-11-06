"""In-memory constructors of MLArray (``empty``, ``zeros``, ``asarray``, ``*_like`` ...).

``MLArray`` inherits them from :class:`MLArrayConstructors`.
"""

from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import blosc2
import numpy as np

from mlarray import storage as _storage
from mlarray.meta import Meta, _meta_internal_write, _spatial_axis_mask


class MLArrayConstructors:
    """Constructors that build an in-memory MLArray (Blosc2- or NumPy-backed)."""

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
                "cparams": _storage.resolve_cparams(self.meta.blosc2.cparams),
                "dparams": _storage.resolve_dparams(self.meta.blosc2.dparams),
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

    def _resolve_like_input(self, x, dtype, meta):
        if isinstance(x, MLArrayConstructors):
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
