# Usage

This section shows common usage patterns for **creating**, **saving**, and **loading** MLArray files, as well as working with **memory mapping** and **metadata**.

MLArray is designed to feel natural in Python workflows: you can construct an `MLArray` directly from a NumPy array, save it to disk, and later load it again with a single line. For large images, MLArray also supports **memory-mapped access**, allowing you to read and modify *only small regions* of an image without loading the full array into RAM. In addition, MLArray exposes a consistent metadata interface, so you can store and retrieve both **standardized metadata** (e.g., spacing, origin) and **arbitrary custom metadata** (e.g., raw DICOM tags, experiment info).

Below are practical examples that cover the most common workflows.

---

## Default usage

The simplest workflow: create an `MLArray` from a NumPy array, save it to disk, and load it back later.

```python
import numpy as np
from mlarray import MLArray

array = np.random.random((128, 256, 256))
image = MLArray(array)  # Create MLArray image
image.save("sample.mla")

image = MLArray("sample.mla")  # Loads image
```

---

## Memory-mapped usage

Memory mapping allows you to access large arrays on disk *without loading everything into memory*. This is ideal for patch-based training, interactive visualization, or working with multi-GB/ TB-scale volumes.

```python
from mlarray import MLArray
import numpy as np

# read-only, partial access (default)
image = MLArray.open("sample.mla", mmap_mode='r')  
crop = image[10:20, 50:60]  # Read crop

# read/write, partial access
image = MLArray.open("sample.mla", mmap_mode='r+')  
image[10:20, 50:60] *= 5  # Modify crop in memory and disk

# read/write, partial access, create/overwrite
array = np.random.random((128, 256, 256))
image = MLArray.create("sample.mla", shape=array.shape, dtype=array.dtype, mmap_mode='w+')
image[...] = array  # Modify image in memory and disk
```

---

## Metadata inspection and manipulation

MLArray provides first-class support for common image metadata (spacing, origin, direction), and also lets you attach arbitrary metadata from the source image source via `meta=...` (e.g., raw DICOM fields, acquisition parameters, dataset identifiers).

```python
import numpy as np
from mlarray import MLArray

array = np.random.random((64, 128, 128))
image = MLArray(
    array,
    spacing=(1.0, 1.0, 1.5),
    origin=(10.0, 10.0, 30.0),
    direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    meta={"patient_id": "123", "modality": "CT"},  # Any metadata from the original image source (for example raw DICOM metadata)
)

print(image.spacing)  # [1.0, 1.0, 1.5]
print(image.origin)  # [10.0, 10.0, 30.0]
print(image.meta.source)  # {"patient_id": "123", "modality": "CT"}

image.spacing[1] = 5.3
image.meta.source["study_id"] = "study-001"
image.save("with-metadata.mla")

# Open memory-mapped
image = MLArray.open("with-metadata.mla", mmap_mode='r+')  
image.meta.source["study_id"] = "new-study"  # Modify metadata
image.close()  # Close and save metadata, only necessary to save modified metadata
```

---

## Copy metadata with overrides

This pattern is useful when you want to generate derived data (e.g., predictions, augmentations, resampled images) while keeping most metadata consistent with a reference image, but selectively overriding specific fields like spacing.

```python
import numpy as np
from mlarray import MLArray

base = MLArray("sample.mla")
array = np.random.random(base.shape)

image = MLArray(
    array,
    spacing=(0.8, 0.8, 1.0),
    copy=base,  # Copies all non-explicitly set arguments from base
)

image.save("copied-metadata.mla")
```

---

## Standardized metadata usage

For structured workflows, MLArray supports a standardized metadata container via `Meta`. This makes metadata access explicit and predictable, while still allowing flexible extensions when needed.

```python
import numpy as np
from mlarray import MLArray, Meta

array = np.random.random((64, 128, 128))
image = MLArray(
    array,
    meta=Meta(source={"patient_id": "123", "modality": "CT"}, is_seg=True),  # Add metadata in a pre-defined format
)

print(image.meta.source)  # {"patient_id": "123", "modality": "CT"}
print(image.meta.is_seg)  # True

image.meta.source["study_id"] = "study-001"
image.meta.is_seg = False
image.save("with-metadata.mla")
```

---

## Non-spatial data usage

Use `axis_labels` to mark which axes are spatial and which are non-spatial
(channels, temporal, components, etc.). Spatial metadata (`spacing`, `origin`,
`direction`) is specified only for the spatial axes, while the full array shape
includes both spatial and non-spatial axes.

```python
import numpy as np
from mlarray import MLArray, MetaSpatial

# Example shape: (time, z, y, x, channels)
array = np.random.random((2, 6, 4, 4, 3, 2))

axis_labels = [
    MetaSpatial.AxisLabel.temporal,
    MetaSpatial.AxisLabel.spatial_z,
    MetaSpatial.AxisLabel.spatial_y,
    "spatial_x",  # Possible to pass predefined labels as strings as well
    MetaSpatial.AxisLabel.channel,
    "some-other-type"  # Possible to pass arbitrary strings as well
]

image = MLArray(
    array,
    spacing=(2.5, 0.7, 0.7),  # spatial axes only (z, y, x)
    origin=(0.0, 0.0, 0.0),
    axis_labels=axis_labels,
)

# Optional per-axis units (length = full array ndims)
image.meta.spatial.axis_units = ["s", "mm", "mm", "mm", ""]
image.save("time-series.mla", patch_size=None)
```



## Patch size variants

MLArray stores arrays in a chunked layout to enable efficient partial reads. You can control how data is chunked using `patch_size` (recommended in most cases), or manually specify chunk and block sizes when you need full control.

### Default patch size (192)

Uses the default patch configuration, optimized for typical ML patch-based access patterns.

```python
from mlarray import MLArray

image = MLArray("sample.mla")
image.save("default-patch.mla")  # Default patch_size is 'default' -> Isotropic patch size of 192 pixels
image.save("default-patch.mla", patch_size='default')
```

### Custom isotropic patch size (512)

A larger patch size can improve throughput when you typically load large regions at once (at the cost of slightly less granular random access).

```python
from mlarray import MLArray

image = MLArray("sample.mla")
image.save("patch-512.mla", patch_size=512)
```

### Custom non-isotropic patch size

Non-isotropic patches are useful when one axis behaves differently (e.g., fewer slices in Z, anisotropic voxel spacing, or slice-wise training).

```python
from mlarray import MLArray

image = MLArray("sample.mla")
image.save("patch-non-iso.mla", patch_size=(128, 192, 256))
```

### Manual chunk/block size

For advanced use cases, you can explicitly define the chunk and block size used by the storage backend.

```python
from mlarray import MLArray

image = MLArray("sample.mla")
image.save("manual-chunk-block.mla", chunk_size=(1, 128, 128), block_size=(1, 32, 32))
```

### Let Blosc2 itself configure chunk/block size

If you disable MLArray patch sizing, Blosc2 can choose chunk and block sizes automatically. This can be helpful when experimenting or when you want to rely entirely on backend heuristics.

```python
from mlarray import MLArray

image = MLArray("sample.mla")
# If patch_size, chunk_size and block_size are all None, Blosc2 will auto-configure chunk and block size
image.save("manual-chunk-block.mla", patch_size=None)
```
