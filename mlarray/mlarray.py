from copy import deepcopy
import numpy as np
import blosc2
import math
from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path
import os
from mlarray.meta import Meta, MetaBlosc2
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
            channel_axis: Optional[int] = None,
            num_threads: int = 1,
            copy: Optional['MLArray'] = None) -> None:
        """Initializes a MLArray instance.

        The MLArray file format (".mla") is a Blosc2-compressed container
        with standardized metadata support for N-dimensional medical images.

        Args:
            array (Union[np.ndarray, str, Path]): Input data or file path. Use
                a numpy ndarray for in-memory arrays. Use a string or Path to
                load a ".b2nd" or ".mla" file.
            spacing (Optional[Union[List, Tuple, np.ndarray]]): Spacing per
                axis. Provide a list/tuple/ndarray with length equal to the
                number of dimensions (e.g., [sx, sy, sz]).
            origin (Optional[Union[List, Tuple, np.ndarray]]): Origin per axis.
                Provide a list/tuple/ndarray with length equal to the number of
                dimensions.
            direction (Optional[Union[List, Tuple, np.ndarray]]): Direction
                cosine matrix. Provide a 2D list/tuple/ndarray with shape
                (ndims, ndims).
            meta (Optional[Dict | Meta]): Free-form metadata dictionary or Meta
                instance. Must be JSON-serializable when saving. 
                If meta is passed as a Dict, it will internally be converted into a Meta object with the dict being interpreted as meta.image metadata.
            num_threads (int): Number of threads for Blosc2 operations.
            mode (str): Blosc2 open mode
                - 'r': read-only, must exist (Default)
                - 'a': read/write, create if doesn't exist (Currently not supported)
                - 'w': create, overwrite if it exists (Currently not supported)
            copy (Optional[MLArray]): Another MLArray instance to copy
                metadata fields from.
        """    
        self.filepath = None
        self.support_metadata = None   
        self.mmap = None
        if isinstance(array, (str, Path)):
            self.load(array, num_threads)
        else:
            self._store = array
            self._validate_and_add_meta(meta, spacing, origin, direction, channel_axis)
        
        if copy is not None:
            self.meta.copy_from(copy.meta)

    def open(
            self,
            filepath: Union[str, Path],
            shape: Optional[Union[List, Tuple, np.ndarray]] = None,
            dtype: Optional[np.dtype] = None,
            channel_axis: Optional[int] = None,
            mmap: str = 'r',
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            num_threads: int = 1,
            cparams: Optional[Dict] = None,
            dparams: Optional[Dict] = None
        ):
        self.filepath = str(filepath)
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")

        if Path(filepath).is_file() and (shape is not None or dtype is not None):
            raise RuntimeError("Cannot create a new file as a file exists already under that path. Explicitly set shape and dtype only if you intent to create a new file.")
        if (shape is not None and dtype is None) or (shape is None and dtype is not None):
            raise RuntimeError("Both shape and dtype must be set if you intend to create a new file.")
        if shape is not None and mmap == 'r':
            raise RuntimeError("mmap_mode cannot be 'r' (read-only) if you intend to write a new file. Explicitly set shape and dtype only if you intent to create a new file.")
        if mmap not in ('r', 'r+', 'w+', 'c'):
            raise RuntimeError("mmap_mode must be one of the following: 'r', 'r+', 'w+', 'c'")
        
        create_array = shape is not None
    
        if create_array:
            self.meta._blosc2 = self._comp_and_validate_blosc2_meta(self.meta._blosc2, patch_size, chunk_size, block_size, shape, channel_axis)   
            self.meta._has_array = True     
        
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")

        blosc2.set_nthreads(num_threads)
        if cparams is None:
            cparams = {'codec': blosc2.Codec.ZSTD, 'clevel': 8,}
        if dparams is None:
            dparams = {'nthreads': num_threads}
        
        if create_array:
            self._store = blosc2.empty(shape=shape, dtype=dtype, urlpath=str(filepath), chunks=self.meta._blosc2.chunk_size, blocks=self.meta._blosc2.block_size, cparams=cparams, dparams=dparams, mmap_mode=mmap)
        else:
            self._store = blosc2.open(urlpath=str(filepath), dparams=dparams, mmap_mode=mmap)
            self._read_meta()
        if self.meta._has_array == True:
            self.meta._blosc2.chunk_size = list(self._store.chunks)
            self.meta._blosc2.block_size = list(self._store.blocks)
        self.mmap = mmap
        self._write_metadata()
        return self

    def close(self):
        self._write_metadata()
        self._store = None
        self.filepath = None
        self.support_metadata = None   
        self.mmap = None
        
    def load(
            self,
            filepath: Union[str, Path], 
            num_threads: int = 1,
        ):
        """Loads a Blosc2-compressed file. Both MLArray ('.mla') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MLArray supports both ".b2nd" and ".mla" files. The MLArray
            format standard and standardized metadata are honored only for
            ".mla". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Path to the Blosc2 file to be loaded.
                The filepath needs to have the extension ".b2nd" or ".mla".
            num_threads (int): Number of threads to use for loading the file.
            mode (str): Blosc2 open mode (e.g., "r", "a").

        Returns:
            Tuple[blosc2.ndarray, dict]: Loaded data and its metadata.

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
        self.mmap = None
        self._read_meta()        
        if self.meta._has_array == True:
            self.meta._blosc2.chunk_size = list(self._store.chunks)
            self.meta._blosc2.block_size = list(self._store.blocks)

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

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mla".
        """
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MLARRAY_SUFFIX}"):
            raise RuntimeError(f"MLArray requires '.b2nd' or '.{MLARRAY_SUFFIX}' as extension.")
    
        if self._store is not None:
            self.meta._blosc2 = self._comp_and_validate_blosc2_meta(self.meta._blosc2, patch_size, chunk_size, block_size, self._store.shape, self.meta.spatial.channel_axis)
            self.meta._has_array = True
        else:
            self.meta._has_array = False
    
        self.support_metadata = str(filepath).endswith(f".{MLARRAY_SUFFIX}")

        blosc2.set_nthreads(num_threads)
        if cparams is None:
            cparams = {'codec': blosc2.Codec.ZSTD, 'clevel': 8,}
        if dparams is None:
            dparams = {'nthreads': num_threads}
        
        if self._store is not None:
            array = np.ascontiguousarray(self._store[...])
            self._store = blosc2.asarray(array, urlpath=str(filepath), chunks=self.meta._blosc2.chunk_size, blocks=self.meta._blosc2.block_size, cparams=cparams, dparams=dparams)
        else:
            array = np.empty((0,))
            self._store = blosc2.asarray(array, urlpath=str(filepath), chunks=self.meta._blosc2.chunk_size, blocks=self.meta._blosc2.block_size, cparams=cparams, dparams=dparams)
        if self.meta._has_array == True:
            self.meta._blosc2.chunk_size = list(self._store.chunks)
            self.meta._blosc2.block_size = list(self._store.blocks)
        self.mmap = None
        self._write_metadata(force=True)

    def to_numpy(self):
        if self._store is None or self.meta._has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        return self._store[...]

    def __getitem__(self, key):
        if self._store is None or self.meta._has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        return self._store[key]

    def __setitem__(self, key, value):
        if self._store is None or self.meta._has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        self._store[key] = value

    def __iter__(self):
        if self._store is None or self.meta._has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        return iter(self._store)

    def __len__(self):
        if self._store is None or self.meta._has_array == False:
            return 0
        return len(self._store)

    def __array__(self, dtype=None):
        if self._store is None or self.meta._has_array == False:
            raise TypeError("MLArray has no array data loaded.")
        arr = np.asarray(self._store)
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    @property
    def spacing(self):
        """Returns the image spacing.

        Returns:
            list: The image spacing with length equal to the number of
            dimensions.
        """
        return self.meta.spatial.spacing
    
    @property
    def origin(self):
        """Returns the image origin.

        Returns:
            list: The image origin with length equal to the number of
            dimensions.
        """
        return self.meta.spatial.origin
    
    @property
    def direction(self):
        """Returns the image direction.

        Returns:
            list: The image direction with shape (ndims, ndims).
        """
        return self.meta.spatial.direction

    @property
    def affine(self) -> np.ndarray:
        """Computes the affine transformation matrix for the image.

        Returns:
            list: Affine matrix with shape (ndims + 1, ndims + 1).
        """
        if self._store is None or self.meta._has_array == False:
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
            list: Translation vector with length equal to the number of
            dimensions.
        """
        if self._store is None or self.meta._has_array == False:
            return None
        return np.array(self.affine)[:-1, -1].tolist()

    @property
    def scale(self):
        """Extracts the scaling factors from the affine matrix.

        Returns:
            list: Scaling factors per axis with length equal to the number of
            dimensions.
        """
        if self._store is None or self.meta._has_array == False:
            return None
        scales = np.linalg.norm(np.array(self.affine)[:-1, :-1], axis=0)
        return scales.tolist()

    @property
    def rotation(self):
        """Extracts the rotation matrix from the affine matrix.

        Returns:
            list: Rotation matrix with shape (ndims, ndims).
        """
        if self._store is None or self.meta._has_array == False:
            return None
        rotation_matrix = np.array(self.affine)[:-1, :-1] / np.array(self.scale)
        return rotation_matrix.tolist()

    @property
    def shear(self):
        """Computes the shear matrix from the affine matrix.

        Returns:
            list: Shear matrix with shape (ndims, ndims).
        """
        if self._store is None or self.meta._has_array == False:
            return None
        scales = np.array(self.scale)
        rotation_matrix = np.array(self.rotation)
        shearing_matrix = np.dot(rotation_matrix.T, np.array(self.affine)[:-1, :-1]) / scales[:, None]
        return shearing_matrix.tolist()
    
    @property
    def shape(self):
        """Returns the shape of the array.

        Returns:
            tuple: Shape of the underlying array.
        """
        if self._store is None or self.meta._has_array == False:
            return None
        return self._store.shape

    @property
    def dtype(self):
        """Returns the dtype of the array."""
        if self._store is None or self.meta._has_array == False:
            return None
        return self._store.dtype
    
    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the image."""
        if self._store is None or self.meta._has_array == False:
            return None
        return len(self._store.shape)
    
    @property
    def _spatial_ndim(self) -> int:
        """Returns the number of dimensions of the image."""
        if self._store is None or self.meta._has_array == False:
            return None
        ndim = len(self._store.shape)
        if self.meta.spatial.channel_axis is not None:
            ndim -= 1
        return ndim

    def comp_blosc2_params(
            self,
            image_size: Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]],
            patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
            channel_axis: Optional[int] = None,
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
            image_size (Tuple[int, int, int, int]): Image shape. Use a 2D, 3D,
                or 4D size; 2D/3D inputs are internally expanded.
            patch_size (Union[Tuple[int, int], Tuple[int, int, int]]): Patch
                size for spatial dimensions. Use a 2-tuple (x, y) or 3-tuple
                (x, y, z).
            bytes_per_pixel (int): Number of bytes per element. Defaults to 4
                for float32.
            l1_cache_size_per_core_in_bytes (int): L1 cache per core in bytes.
            l3_cache_size_per_core_in_bytes (int): L3 cache per core in bytes.
            safety_factor (float): Safety factor to avoid filling caches.

        Returns:
            Tuple[Tuple[int, ...], Tuple[int, ...]]: Recommended chunk size and block size.
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

        if channel_axis is not None:
            image_size = _move_index_list(image_size, channel_axis+num_squeezes, 0)

        if len(image_size) != 4:
            raise RuntimeError("Image size must be 4D.")
        
        if not (len(patch_size) == 2 or len(patch_size) == 3):
            raise RuntimeError("Patch size must be 2D or 3D.")

        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = [1, *patch_size]
        patch_size = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

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

        if channel_axis is not None:
            block_size = _move_index_list(block_size, 0, channel_axis+num_squeezes)
            chunk_size = _move_index_list(chunk_size, 0, channel_axis+num_squeezes)

        block_size = block_size[num_squeezes:]
        chunk_size = chunk_size[num_squeezes:]

        return [int(value) for value in chunk_size], [int(value) for value in block_size]
    
    def _comp_and_validate_blosc2_meta(self, meta_blosc2, patch_size, chunk_size, block_size, shape, channel_axis):
        if patch_size is not None and patch_size != "default" and not ((len(shape) == 2 and channel_axis is None) or (len(shape) == 3 and channel_axis is None) or (len(shape) == 4 and channel_axis is not None) or (len(shape) == 4 and channel_axis is not None)):
            raise NotImplementedError("Chunk and block size optimization based on patch size is only implemented for 2D and 3D images. Please set the chunk and block size manually or set to None for blosc2 to determine a chunk and block size.")
        if patch_size is not None and patch_size != "default" and (chunk_size is not None or block_size is not None):
            raise RuntimeError("patch_size and chunk_size / block_size cannot both be explicitly set.")

        ndims = len(shape) if channel_axis is None else len(shape) - 1
        if patch_size == "default": 
            if meta_blosc2 is not None and meta_blosc2.patch_size is not None:  # Use previously loaded patch size, when patch size is not explicitly set and a patch size from a previously loaded image exists
                patch_size = meta_blosc2.patch_size
            else:  # Use default patch size, when patch size is not explicitly set and no patch size from a previously loaded image exists
                patch_size = [MLARRAY_DEFAULT_PATCH_SIZE] * ndims

        patch_size = [patch_size] * len(shape) if isinstance(patch_size, int) else patch_size

        if patch_size is not None:
            chunk_size, block_size = self.comp_blosc2_params(shape, patch_size, channel_axis)

        meta_blosc2 = MetaBlosc2(chunk_size, block_size, patch_size)
        meta_blosc2._validate_and_cast(len(shape), channel_axis)
        return meta_blosc2
    
    def _read_meta(self):
        meta = Meta()
        if self.support_metadata and isinstance(self._store, blosc2.ndarray.NDArray):
            meta = self._store.vlmeta["mlarray"]
            meta = Meta.from_dict(meta)
        self._validate_and_add_meta(meta)

    def _write_metadata(self, force=False):
        if self.support_metadata and isinstance(self._store, blosc2.ndarray.NDArray) and (self.mmap in ('r+', 'w+') or force):
            metadata = self.meta.to_dict()
            if not is_serializable(metadata):
                raise RuntimeError("Metadata is not serializable.")
            self._store.vlmeta["mlarray"] = metadata
    
    def _validate_and_add_meta(self, meta, spacing=None, origin=None, direction=None, channel_axis=None):
        if meta is not None:
            if not isinstance(meta, (dict, Meta)):
                raise ValueError("Meta must be None, a dict or a Meta object.")
            if isinstance(meta, dict):
                meta = Meta(image=meta)
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
        if channel_axis is not None:
            self.meta.spatial.channel_axis = channel_axis
        self.meta.spatial.shape = self.shape
        self.meta.spatial._validate_and_cast(self._spatial_ndim)

