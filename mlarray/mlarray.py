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
            copy: Optional['MLArray'] = None) -> None:
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
        """
        self.filepath = None
        self.support_metadata = None
        self.mode = None
        self.mmap_mode = None
        self.meta = None
        if isinstance(array, (str, Path)) and (spacing is not None or origin is not None or direction is not None or meta is not None or axis_labels is not None or copy is not None):
            raise ("Spacing, origin, direction, meta, axis_labels or copy cannot be set when array is a filepath.")
        if isinstance(array, (str, Path)):
            self._load(array)
        else:
            self._store = array
            has_array = array is not None
            self._validate_and_add_meta(meta, spacing, origin, direction, axis_labels, has_array)        
            if copy is not None:
                self.meta.copy_from(copy.meta)

    @classmethod
    def open(
            cls,
            filepath: Union[str, Path],
            shape: Optional[Union[List, Tuple, np.ndarray]] = None,
            dtype: Optional[np.dtype] = None,
            axis_labels: Optional[List[Union[str, AxisLabel]]] = None,
            mode: str = 'r',
            mmap_mode: str = 'r',
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            num_threads: int = 1,
            cparams: Optional[Dict] = None,
            dparams: Optional[Dict] = None
        ):
        """Open an existing Blosc2 file or create a new one with memory mapping.

        This method supports both MLArray (".mla") and plain Blosc2 (".b2nd")
        files. When creating a new file, both ``shape`` and ``dtype`` must be
        provided.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".b2nd" or ".mla".
            shape (Optional[Union[List, Tuple, np.ndarray]]): Shape of the array
                to create. If provided, a new file is created. Length must match
                the full array dimensionality (including non-spatial axes if present).
            dtype (Optional[np.dtype]): Numpy dtype for a newly created array.
            axis_labels (Optional[List[Union[str, AxisLabel]]]): Per-axis labels or roles. Length must match ndims. If None, the array
                is treated as purely spatial. Used for patch/chunk/block calculations.
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                One of "r", "a", "w". Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. One of "r", "r+", "w+", "c". Leave at default if unsure.
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
            num_threads (int): Number of threads for Blosc2 operations.
            cparams (Optional[Dict]): Blosc2 compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.ZSTD,
                'clevel': 8}``.
            dparams (Optional[Dict]): Blosc2 decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': num_threads}``.

        Returns:
            MLArray: A newly created MLArray instance.

        Raises:
            RuntimeError: If the file extension is invalid, if shape/dtype are
                inconsistent, or if mmap_mode is invalid for creation.
        """
        class_instance = cls()
        class_instance._open(filepath, shape, dtype, axis_labels, mode, mmap_mode, patch_size, chunk_size, block_size, num_threads, cparams, dparams)
        return class_instance

    def _open(
            self,
            filepath: Union[str, Path],
            shape: Optional[Union[List, Tuple, np.ndarray]] = None,
            dtype: Optional[np.dtype] = None,
            axis_labels: Optional[list[Union[str, AxisLabel]]] = None,
            mode: str = 'r',
            mmap_mode: str = 'r',
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            num_threads: int = 1,
            cparams: Optional[Dict] = None,
            dparams: Optional[Dict] = None
        ):
        """Internal open method. Open an existing Blosc2 file or create a new one with memory mapping.

        This method supports both MLArray (".mla") and plain Blosc2 (".b2nd")
        files. When creating a new file, both ``shape`` and ``dtype`` must be
        provided.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".b2nd" or ".mla".
            shape (Optional[Union[List, Tuple, np.ndarray]]): Shape of the array
                to create. If provided, a new file is created. Length must match
                the full array dimensionality (including non-spatial axes if present).
            dtype (Optional[np.dtype]): Numpy dtype for a newly created array.
            axis_labels (Optional[List[Union[str, AxisLabel]]]): Per-axis labels or roles. Length must match ndims. If None, the array
                is treated as purely spatial.
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                One of "r", "a", "w". Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. One of "r", "r+", "w+", "c". Leave at default if unsure.
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
            num_threads (int): Number of threads for Blosc2 operations.
            cparams (Optional[Dict]): Blosc2 compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.ZSTD,
                'clevel': 8}``.
            dparams (Optional[Dict]): Blosc2 decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': num_threads}``.

        Raises:
            RuntimeError: If the file extension is invalid, if shape/dtype are
                inconsistent, or if mmap_mode is invalid for creation.
        """
        self.filepath = str(filepath)
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")

        if Path(filepath).is_file() and (shape is not None or dtype is not None):
            raise RuntimeError("Cannot create a new file as a file exists already under that path. Explicitly set shape and dtype only if you intent to create a new file.")
        if (shape is not None and dtype is None) or (shape is None and dtype is not None):
            raise RuntimeError("Both shape and dtype must be set if you intend to create a new file.")
        if shape is not None and mmap_mode != 'w+':
            raise RuntimeError("mmap_mode must be 'w+' (create/overwrite) if you intend to write a new file. Explicitly set shape and dtype only if you intent to create a new file.")
        if (shape is None or dtype is None) and mmap_mode == 'w+':
            raise RuntimeError("Shape and dtype must be set explicitly when mmap_mode is 'w+'. Explicitly set shape and dtype only if you intent to create a new file.")
        if mode not in ('r', 'a', 'w'):
            raise RuntimeError("mode must be one of the following: 'r', 'a', 'w'")
        if mmap_mode not in ('r', 'r+', 'w+', 'c'):
            raise RuntimeError("mmap_mode must be one of the following: 'r', 'r+', 'w+', 'c'")
        
        create_array = mmap_mode == 'w+'
    
        if create_array:
            spatial_axis_mask = [True] * len(shape) if axis_labels is None else _spatial_axis_mask(axis_labels)
            self.meta._blosc2 = self._comp_and_validate_blosc2_meta(self.meta._blosc2, patch_size, chunk_size, block_size, shape, np.dtype(dtype).itemsize, spatial_axis_mask)   
            self.meta._has_array.has_array = True
        
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")

        blosc2.set_nthreads(num_threads)
        if cparams is None:
            cparams = {'codec': blosc2.Codec.ZSTD, 'clevel': 8,}
        if dparams is None:
            dparams = {'nthreads': num_threads}
        
        if create_array:
            self._store = blosc2.empty(shape=shape, dtype=np.dtype(dtype), urlpath=str(filepath), chunks=self.meta._blosc2.chunk_size, blocks=self.meta._blosc2.block_size, cparams=cparams, dparams=dparams, mmap_mode=mmap_mode)
        else:
            self._store = blosc2.open(urlpath=str(filepath), dparams=dparams, mode=mode, mmap_mode=mmap_mode)
            self._read_meta()
        self._update_blosc2_meta()
        self.mode = mode
        self.mmap_mode = mmap_mode
        self._write_metadata()

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

    @classmethod
    def asarray(
            cls,
            array: Union[np.ndarray],
            memory_compressed: bool = False,
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            meta: Optional[Union[Dict, Meta]] = None,
            cparams: Optional[Dict] = None,
            dparams: Optional[Dict] = None
        ):
        """Converts the array to an MLArray. The MLArray can optionally be in-memory compressed.

        Args:
            array (Union[np.ndarray]): Input array to convert to MLArray.
            memory_compressed (bool): If True, convert ``array`` into an in-memory
                Blosc2 container. If False, keep the underlying store as the
                original NumPy array.
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
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            cparams (Optional[Dict]): Blosc2 compression parameters used when
                ``memory_compressed=True`` and data is written into an in-memory
                Blosc2 container (for example codec, compression level, and
                filters). Controls how data is compressed when stored. If None,
                defaults to ``{'codec': blosc2.Codec.ZSTD, 'clevel': 8}``.
                Ignored when ``memory_compressed=False``.
            dparams (Optional[Dict]): Blosc2 decompression parameters used when
                accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``. Ignored when
                ``memory_compressed=False``.

        Returns:
            MLArray: A newly created MLArray instance.

        Raises:
            ValueError: If ``meta`` is not None, dict, or Meta.
            RuntimeError: If patch/chunk/block arguments are inconsistent.
            NotImplementedError: If automatic patch optimization is not
                implemented for the provided dimensionality.
        """
        class_instance = cls()
        class_instance._asarray(array, memory_compressed, patch_size, chunk_size, block_size, meta, cparams, dparams)
        return class_instance

    def _asarray(
            self,
            array: Union[np.ndarray],
            memory_compressed: bool = False,
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            meta: Optional[Union[Dict, Meta]] = None,
            cparams: Optional[Dict] = None,
            dparams: Optional[Dict] = None
        ):
        """Internal MLArray asarray method. Converts the array to an MLArray. The MLArray can optionally be in-memory compressed.

        Args:
            array (Union[np.ndarray]): Input array to convert to MLArray.
            memory_compressed (bool): If True, convert ``array`` into an in-memory
                Blosc2 container. If False, keep the underlying store as the
                original NumPy array.
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
            meta (Optional[Union[Dict, Meta]]): Optional metadata attached to
                the created ``MLArray``. Dict values are stored as
                ``meta.source``.
            cparams (Optional[Dict]): Blosc2 compression parameters used when
                ``memory_compressed=True`` and data is written into an in-memory
                Blosc2 container (for example codec, compression level, and
                filters). Controls how data is compressed when stored. If None,
                defaults to ``{'codec': blosc2.Codec.ZSTD, 'clevel': 8}``.
                Ignored when ``memory_compressed=False``.
            dparams (Optional[Dict]): Blosc2 decompression parameters used when
                accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``. Ignored when
                ``memory_compressed=False``.

        Raises:
            ValueError: If ``meta`` is not None, dict, or Meta.
            RuntimeError: If patch/chunk/block arguments are inconsistent.
            NotImplementedError: If automatic patch optimization is not
                implemented for the provided dimensionality.
        """
        self._store = array
        has_array = True
        self._validate_and_add_meta(meta, has_array=has_array)
        spatial_axis_mask = [True] * self.ndim if self.meta.spatial.axis_labels is None else _spatial_axis_mask(self.meta.spatial.axis_labels)
        self.meta._blosc2 = self._comp_and_validate_blosc2_meta(self.meta._blosc2, patch_size, chunk_size, block_size, self._store.shape, self._store.dtype.itemsize, spatial_axis_mask)
        self.meta._has_array.has_array = True

        num_threads = dparams.get('nthreads', 1) if dparams is not None else 1
        blosc2.set_nthreads(num_threads)
        if cparams is None:
            cparams = {'codec': blosc2.Codec.ZSTD, 'clevel': 8,}
        if dparams is None:
            dparams = {'nthreads': num_threads}

        if memory_compressed:
            array = np.ascontiguousarray(array[...])
            self._store = blosc2.asarray(array, chunks=self.meta._blosc2.chunk_size, blocks=self.meta._blosc2.block_size, cparams=cparams, dparams=dparams)
        else:
            self._store = array

    @classmethod
    def load(
            cls,
            filepath: Union[str, Path], 
            num_threads: int = 1,
        ):
        """Loads a Blosc2-compressed file as a whole. Does not use memory-mapping. Both MLArray ('.mla') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Path to the Blosc2 file to be loaded.
                The filepath needs to have the extension ".b2nd" or ".mla".
            num_threads (int): Number of threads to use for loading the file.

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mla".
        """
        class_instance = cls()
        class_instance._load(filepath, num_threads)
        return class_instance
    
    def _load(
            self,
            filepath: Union[str, Path], 
            num_threads: int = 1,
        ):
        """Internal MLArray load method. Loads a Blosc2-compressed file. Both MLArray ('.mla') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Path to the Blosc2 file to be loaded.
                The filepath needs to have the extension ".b2nd" or ".mla".
            num_threads (int): Number of threads to use for loading the file.

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mla".
        """
        self.filepath = str(filepath)
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")
        blosc2.set_nthreads(num_threads)
        dparams = {'nthreads': num_threads}
        self._store = blosc2.open(urlpath=str(filepath), cdparams=dparams, mode='r')
        self.mode = None
        self.mmap_mode = None
        self._read_meta()        
        self._update_blosc2_meta()
        self._store = np.ascontiguousarray(self._store[...])

    def save(
            self,
            filepath: Union[str, Path],
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            num_threads: int = 1,
            cparams: Optional[Dict] = None,
            dparams: Optional[Dict] = None
        ):
        """Saves the array to a Blosc2-compressed file. Both MLArray ('.mla') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when saving.

        Args:
            filepath (Union[str, Path]): Path to save the file. Must end with
                ".b2nd" or ".mla".
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
            num_threads (int): Number of threads to use for saving the file.
            cparams (Optional[Dict]): Blosc2 compression parameters used when
                writing array data to disk (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.ZSTD,
                'clevel': 8}``.
            dparams (Optional[Dict]): Blosc2 decompression parameters used when
                accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': num_threads}``.

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mla".
        """
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")
    
        if self._store is not None:
            spatial_axis_mask = [True] * self.ndim if self.meta.spatial.axis_labels is None else _spatial_axis_mask(self.meta.spatial.axis_labels)
            self.meta._blosc2 = self._comp_and_validate_blosc2_meta(self.meta._blosc2, patch_size, chunk_size, block_size, self._store.shape, self._store.dtype.itemsize, spatial_axis_mask)
            self.meta._has_array.has_array = True
    
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")

        blosc2.set_nthreads(num_threads)
        if cparams is None:
            cparams = {'codec': blosc2.Codec.ZSTD, 'clevel': 8,}
        if dparams is None:
            dparams = {'nthreads': num_threads}

        if Path(filepath).is_file():
            os.remove(str(filepath))
        
        if self._store is not None:
            array = np.ascontiguousarray(self._store[...])
            self._store = blosc2.asarray(array, urlpath=str(filepath), chunks=self.meta._blosc2.chunk_size, blocks=self.meta._blosc2.block_size, cparams=cparams, dparams=dparams)
        else:
            array = np.empty((0,))
            self._store = blosc2.asarray(array, urlpath=str(filepath), chunks=self.meta._blosc2.chunk_size, blocks=self.meta._blosc2.block_size, cparams=cparams, dparams=dparams)
        self._update_blosc2_meta()
        self.mode = None
        self.mmap_mode = None
        self._write_metadata(force=True)

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
        spacing  = np.array(self.spacing) if self.spacing is not None else np.ones(self._spatial_ndim)
        origin  = np.array(self.origin) if self.origin is not None else np.zeros(self._spatial_ndim)
        direction = np.array(self.direction) if self.direction is not None else np.eye(self._spatial_ndim)
        affine = np.eye(self._spatial_ndim + 1)
        affine[:self._spatial_ndim, :self._spatial_ndim] = direction @ np.diag(spacing)
        affine[:self._spatial_ndim, self._spatial_ndim] = origin
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
    def _spatial_ndim(self) -> int:
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

        if patch_size == "default": 
            if meta_blosc2 is not None and meta_blosc2.patch_size is not None:  # Use previously loaded patch size, when patch size is not explicitly set and a patch size from a previously loaded image exists
                patch_size = meta_blosc2.patch_size
            else:  # Use default patch size, when patch size is not explicitly set and no patch size from a previously loaded image exists
                patch_size = [MLARRAY_DEFAULT_PATCH_SIZE] * num_spatial_axes

        patch_size = [patch_size] * len(shape) if isinstance(patch_size, int) else patch_size

        if patch_size is not None:
            chunk_size, block_size = MLArray.comp_blosc2_params(shape, patch_size, spatial_axis_mask, bytes_per_pixel=dtype_itemsize)

        meta_blosc2 = MetaBlosc2(chunk_size, block_size, patch_size)
        meta_blosc2._validate_and_cast(ndims=len(shape), spatial_ndims=num_spatial_axes)
        return meta_blosc2
    
    def _read_meta(self):
        """Read MLArray metadata from the underlying store, if available."""
        meta = Meta()
        if self.support_metadata and isinstance(self._store, blosc2.ndarray.NDArray):
            meta = self._store.vlmeta["mlarray"]
            meta = Meta.from_mapping(meta)
        self._validate_and_add_meta(meta)

    def _write_metadata(self, force=False):
        """Write MLArray metadata to the underlying store if supported.

        Args:
            force (bool): If True, write even when mmap_mode is read-only.
        """
        if self.support_metadata and isinstance(self._store, blosc2.ndarray.NDArray) and (self.mmap_mode in ('r+', 'w+') or force):
            metadata = self.meta.to_mapping()
            if not is_serializable(metadata):
                raise RuntimeError("Metadata is not serializable.")
            self._store.vlmeta["mlarray"] = metadata
    
    def _validate_and_add_meta(self, meta, spacing=None, origin=None, direction=None, axis_labels=None, has_array=None):
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
        self.meta.spatial._validate_and_cast(ndims=self.ndim, spatial_ndims=self._spatial_ndim)

    def _update_blosc2_meta(self):
        """Sync Blosc2 chunk and block sizes into metadata.

        Updates ``self.meta._blosc2`` from the underlying store when the array
        is present.
        """
        if self.meta._has_array.has_array == True:
            self.meta._blosc2.chunk_size = list(self._store.chunks)
            self.meta._blosc2.block_size = list(self._store.blocks)
