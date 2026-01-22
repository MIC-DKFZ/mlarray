# MLArray Public API

This document lists the public API surface of `MLArray`.

## Class: `MLArray`

### Constructor

```python
MLArray(
    array: Optional[Union[np.ndarray, str, Path]] = None,
    spacing: Optional[Union[List, Tuple, np.ndarray]] = None,
    origin: Optional[Union[List, Tuple, np.ndarray]] = None,
    direction: Optional[Union[List, Tuple, np.ndarray]] = None,
    meta: Optional[Union[Dict, Meta]] = None,
    channel_axis: Optional[int] = None,
    num_threads: int = 1,
    copy: Optional["MLArray"] = None,
)
```

| argument | type | description |
| --- | --- | --- |
| array | Optional[Union[np.ndarray, str, Path]] | Input data or file path. Use a numpy ndarray for in-memory arrays, or a string/Path to load a ".b2nd" or ".mla" file. If None, an empty MLArray instance is created. |
| spacing | Optional[Union[List, Tuple, np.ndarray]] | Spacing per spatial axis. Provide a list/tuple/ndarray with length equal to the number of spatial dimensions. |
| origin | Optional[Union[List, Tuple, np.ndarray]] | Origin per spatial axis. Provide a list/tuple/ndarray with length equal to the number of spatial dimensions. |
| direction | Optional[Union[List, Tuple, np.ndarray]] | Direction cosine matrix. Provide a 2D list/tuple/ndarray with shape (ndims, ndims). |
| meta | Optional[Union[Dict, Meta]] | Free-form metadata dictionary or Meta instance. Must be JSON-serializable when saving. Dicts are interpreted as `meta.image`. |
| channel_axis | Optional[int] | Axis index that represents channels in the array (e.g., 0 for CHW or -1 for HWC). If None, the array is treated as purely spatial. |
| num_threads | int | Number of threads for Blosc2 operations. |
| copy | Optional[MLArray] | Another MLArray instance to copy metadata fields from. |

### Properties

| name | type | description |
| --- | --- | --- |
| spacing | Optional[List[float]] | Spacing per spatial axis. |
| origin | Optional[List[float]] | Origin per spatial axis. |
| direction | Optional[List[List[float]]] | Direction cosine matrix. |
| affine | Optional[List[List[float]]] | Affine transform matrix, shape (ndims + 1, ndims + 1). |
| translation | Optional[List[float]] | Translation vector from the affine. |
| scale | Optional[List[float]] | Scale factors from the affine. |
| rotation | Optional[List[List[float]]] | Rotation matrix from the affine. |
| shear | Optional[List[List[float]]] | Shear matrix from the affine. |
| shape | Optional[Tuple[int, ...]] | Shape of the underlying array. |
| dtype | Optional[np.dtype] | Dtype of the underlying array. |
| ndim | Optional[int] | Number of array dimensions. |

### Methods

| name | signature | description |
| --- | --- | --- |
| open | `open(filepath, shape=None, dtype=None, channel_axis=None, mmap='r', patch_size='default', chunk_size=None, block_size=None, num_threads=1, cparams=None, dparams=None)` | Open an existing file or create a new one with memory mapping. |
| close | `close()` | Flush metadata and close the underlying store. |
| load | `load(filepath, num_threads=1)` | Load a ".b2nd" or ".mla" file into the instance. |
| save | `save(filepath, patch_size='default', chunk_size=None, block_size=None, num_threads=1, cparams=None, dparams=None)` | Save to ".b2nd" or ".mla". |
| to_numpy | `to_numpy()` | Return the underlying data as a NumPy array. |
| comp_blosc2_params | `comp_blosc2_params(image_size, patch_size, channel_axis=None, bytes_per_pixel=4, l1_cache_size_per_core_in_bytes=32768, l3_cache_size_per_core_in_bytes=1441792, safety_factor=0.8)` | Compute recommended chunk/block sizes. |

#### open arguments

| argument | type | description |
| --- | --- | --- |
| filepath | Union[str, Path] | Target file path. Must end with ".b2nd" or ".mla". |
| shape | Optional[Union[List, Tuple, np.ndarray]] | Shape of the array to create. Length must match full array dimensionality (including channels if present). |
| dtype | Optional[np.dtype] | Numpy dtype for a newly created array. |
| channel_axis | Optional[int] | Axis index for channels in the array. Used for patch/chunk/block calculations. |
| mmap | str | Blosc2 mmap mode. One of "r", "r+", "w+", "c". |
| patch_size | Optional[Union[int, List, Tuple]] | Patch size hint for chunk/block optimization. Provide an int for isotropic sizes or a list/tuple with length equal to the number of spatial dimensions. Use "default" to use the default patch size of 192. |
| chunk_size | Optional[Union[int, List, Tuple]] | Explicit chunk size. Provide an int or tuple/list with length equal to the array dimensions. Ignored when patch_size is provided. |
| block_size | Optional[Union[int, List, Tuple]] | Explicit block size. Provide an int or tuple/list with length equal to the array dimensions. Ignored when patch_size is provided. |
| num_threads | int | Number of threads for Blosc2 operations. |
| cparams | Optional[Dict] | Blosc2 compression parameters. |
| dparams | Optional[Dict] | Blosc2 decompression parameters. |

#### load arguments

| argument | type | description |
| --- | --- | --- |
| filepath | Union[str, Path] | Path to the file to be loaded. Must end with ".b2nd" or ".mla". |
| num_threads | int | Number of threads to use for loading the file. |

#### save arguments

| argument | type | description |
| --- | --- | --- |
| filepath | Union[str, Path] | Path to save the file. Must end with ".b2nd" or ".mla". |
| patch_size | Optional[Union[int, List, Tuple]] | Patch size hint for chunk/block optimization. Provide an int for isotropic sizes or a list/tuple with length equal to the number of spatial dimensions. Use "default" to use the default patch size of 192. |
| chunk_size | Optional[Union[int, List, Tuple]] | Explicit chunk size. Provide an int or a tuple/list with length equal to the array dimensions, or None to let Blosc2 decide. Ignored when patch_size is provided. |
| block_size | Optional[Union[int, List, Tuple]] | Explicit block size. Provide an int or a tuple/list with length equal to the array dimensions, or None to let Blosc2 decide. Ignored when patch_size is provided. |
| num_threads | int | Number of threads to use for saving the file. |
| cparams | Optional[Dict] | Blosc2 compression parameters. |
| dparams | Optional[Dict] | Blosc2 decompression parameters. |

#### comp_blosc2_params arguments

| argument | type | description |
| --- | --- | --- |
| image_size | Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]] | Image shape. Use a 2D, 3D, or 4D size; 2D/3D inputs are internally expanded. |
| patch_size | Union[Tuple[int, int], Tuple[int, int, int]] | Patch size for spatial dimensions. Use a 2-tuple (x, y) or 3-tuple (x, y, z). |
| channel_axis | Optional[int] | Axis index for channels in the original array. |
| bytes_per_pixel | int | Number of bytes per element. Defaults to 4 for float32. |
| l1_cache_size_per_core_in_bytes | int | L1 cache per core in bytes. |
| l3_cache_size_per_core_in_bytes | int | L3 cache per core in bytes. |
| safety_factor | float | Safety factor to avoid filling caches. |
