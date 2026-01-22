# MLArray Public API

This document lists the public API surface of `MLArray`.

## Class: `MLArray`

### Constructor

```python
MLArray(
    array: Union[np.ndarray, str, Path],
    spacing: Optional[Union[List, Tuple, np.ndarray]] = None,
    origin: Optional[Union[List, Tuple, np.ndarray]] = None,
    direction: Optional[Union[List, Tuple, np.ndarray]] = None,
    meta: Optional[Union[Dict, Meta]] = None,
    mmap: bool = False,
    num_threads: int = 1,
    mode: str = "r",
    copy: Optional["MLArray"] = None,
)
```

| argument | type | description |
| --- | --- | --- |
| array | Union[np.ndarray, str, Path] | Input data or file path. Use a numpy ndarray for in-memory arrays. Use a string or Path to load a ".b2nd" or ".mla" file. |
| spacing | Optional[Union[List, Tuple, np.ndarray]] | Spacing per axis. Provide a list/tuple/ndarray with length equal to the number of dimensions (e.g., [sx, sy, sz]). |
| origin | Optional[Union[List, Tuple, np.ndarray]] | Origin per axis. Provide a list/tuple/ndarray with length equal to the number of dimensions. |
| direction | Optional[Union[List, Tuple, np.ndarray]] | Direction cosine matrix. Provide a 2D list/tuple/ndarray with shape (ndims, ndims). |
| meta | Optional[Union[Dict, Meta]] | Free-form metadata dictionary or Meta instance. Must be JSON-serializable when saving. If meta is passed as a Dict, it will internally be converted into a Meta object with the dict being interpreted as meta.image metadata. |
| mmap | bool | Whether to keep the loaded array memory-mapped when loading from disk. If true, MLArray.array will be an blosc2.ndarray.NDArray, else np.ndarray. |
| num_threads | int | Number of threads for Blosc2 operations. |
| mode | str | Blosc2 open mode: 'r' read-only (default), 'a' read/write create if doesn't exist (not supported), 'w' create overwrite if exists (not supported). |
| copy | Optional[MLArray] | Another MLArray instance to copy metadata fields from. |

### Properties

| name | type | description |
| --- | --- | --- |
| spacing | Optional[List[float]] | Image spacing per axis. |
| origin | Optional[List[float]] | Image origin per axis. |
| direction | Optional[List[List[float]]] | Direction cosine matrix. |
| affine | List[List[float]] | Affine transform matrix. |
| translation | List[float] | Translation vector from the affine. |
| scale | List[float] | Scale factors from the affine. |
| rotation | List[List[float]] | Rotation matrix from the affine. |
| shear | List[List[float]] | Shear matrix from the affine. |
| shape | Tuple[int, ...] | Shape of the underlying array. |
| ndims | int | Number of array dimensions. |

### Methods

| name | signature | description |
| --- | --- | --- |
| save | `save(filepath, patch_size="default", chunk_size=None, block_size=None, clevel=8, codec=blosc2.Codec.ZSTD, num_threads=1)` | Save to `.b2nd` or `.mla`. |
| comp_blosc2_params | `comp_blosc2_params(image_size, patch_size, bytes_per_pixel=4, l1_cache_size_per_core_in_bytes=32768, l3_cache_size_per_core_in_bytes=1441792, safety_factor=0.8)` | Compute recommended chunk/block sizes. |

#### save arguments

| argument | type | description |
| --- | --- | --- |
| filepath | Union[str, Path] | Path to save the file. Must end with ".b2nd" or ".mla". |
| patch_size | Optional[Union[int, List, Tuple]] | Patch size hint for chunk/block optimization. Provide an int for isotropic sizes or a list/tuple with length equal to the number of dimensions. Use "default" to use the default patch size of 192. |
| chunk_size | Optional[Union[int, List, Tuple]] | Explicit chunk size. Provide an int or a tuple/list with length equal to the number of dimensions, or None to let Blosc2 decide. Ignored when patch_size is not None. |
| block_size | Optional[Union[int, List, Tuple]] | Explicit block size. Provide an int or a tuple/list with length equal to the number of dimensions, or None to let Blosc2 decide. Ignored when patch_size is not None. |
| clevel | int | Compression level from 0 (no compression) to 9 (maximum compression). |
| codec | blosc2.Codec | Compression codec to use. |
| num_threads | int | Number of threads to use for saving the file. |

#### comp_blosc2_params arguments

| argument | type | description |
| --- | --- | --- |
| image_size | Tuple[int, int, int, int] | Image shape. Use a 2D, 3D, or 4D size; 2D/3D inputs are internally expanded. |
| patch_size | Union[Tuple[int, int], Tuple[int, int, int]] | Patch size for spatial dimensions. Use a 2-tuple (x, y) or 3-tuple (x, y, z). |
| bytes_per_pixel | int | Number of bytes per element. Defaults to 4 for float32. |
| l1_cache_size_per_core_in_bytes | int | L1 cache per core in bytes. |
| l3_cache_size_per_core_in_bytes | int | L3 cache per core in bytes. |
| safety_factor | float | Safety factor to avoid filling caches. |
