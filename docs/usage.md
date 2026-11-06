# Usage

This section shows common usage patterns for **creating**, **saving**, and **loading** MLArray files, as well as working with **memory mapping** and **metadata**.

MLArray is designed to feel natural in Python workflows: you can construct an `MLArray` directly from a NumPy array, save it to disk, and later load it again with a single line. For large images, MLArray also supports **memory-mapped access**, allowing you to read and modify *only small regions* of an image without loading the full array into RAM. In addition, MLArray stores explicit metadata: what each axis is, where the array sits in physical space, what its values mean, optional domain profiles, and your own metadata.

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

## Metadata

MLArray metadata is explicit and domain-neutral (full definition: [Schema](schema.md) and the
[v1 spec](design/spec.md)):

| part | meaning |
|---|---|
| `axes` | one entry per array axis, in array order: a name and a type (`space`, `time`, `channel`, `component`, `other`) |
| `spaces` | coordinate systems (e.g. the patient frame) and the matrix that maps array indices into each |
| `values` | what the numbers mean: `kind` (measurement, label, ...), unit, labels |
| `profiles` | optional domain metadata, e.g. `medical` (DICOM keywords) and `ml` (statistics, boxes, split) |
| `user` | arbitrary metadata of your own, never interpreted |
| `source` | the original metadata of converted files, read-only |

### Geometry

Geometry is never implied. An array without `spaces` lives in index space only. To place it in a
physical frame, give `spacing`, `origin` and `direction` **together with the space they refer to**:

```python
import numpy as np
from mlarray import MLArray

array = np.random.random((64, 128, 128))
image = MLArray(
    array,
    axes=["z", "y", "x"],             # names only; they say nothing about direction
    spacing=(2.5, 0.7, 0.7),          # per array axis, in array order (z, y, x)
    origin=(-170.1, -201.3, -87.0),   # centre of index (0, 0, 0) in space coordinates
    direction=[[0, 0, 1],             # column c = unit vector of array axis c in the space;
               [0, 1, 0],             # rows = space axes x, y, z
               [1, 0, 0]],
    space="patient",                  # DICOM patient frame: x to the left, y posterior, z superior (LPS)
)
image.save("sample.mla")

image = MLArray("sample.mla")
print(image.spacing)         # (2.5, 0.7, 0.7)
print(image.origin)          # (-170.1, -201.3, -87.0)
print(image.matrix())        # 4x4 from-array matrix; an integer index is a voxel centre
print(image.letter_codes())  # ('S', 'P', 'L'): the direction each array axis increases toward
```

`spacing`, `origin` and `direction` are derived from the stored matrix and are read-only. Change them
with `set_geometry`; values you leave out are kept:

```python
image.set_geometry(spacing=(3.0, 0.7, 0.7))
```

Without `space=` the geometry is refused, because spacing and orientation mean nothing until the frame
is known. For full control, pass `spaces=[...]` with an explicit `from_array` matrix (see
[Schema](schema.md#spaces)).

### Values, profiles and user metadata

```python
import numpy as np
from mlarray import MLArray

seg = MLArray(
    np.zeros((64, 128, 128), dtype=np.uint8),
    values={"kind": "label", "labels": {"0": {"name": "background"}, "1": {"name": "liver", "color": [221, 130, 101]}}},
    profiles={"ml": {"split": "train", "boxes": {"frame": "index", "axes": ["axis_0", "axis_1", "axis_2"],
                                                 "items": [{"min": [10, 50, 60], "max": [40, 90, 120], "label": 1}]}}},
    user={"annotator": "A", "notes": ["checked"]},
)
seg.user["study_id"] = "study-001"     # user metadata is a plain dict
seg.profiles["ml"]["split"] = "test"   # profiles too; they are validated when written
seg.save("seg.mla")

seg = MLArray.open("seg.mla")          # opening reads only the small core metadata
print(seg.values.kind, dict(seg.values.labels))
print(seg.user)                        # user and source are decoded on first access
```

Core parts (`axes`, `spaces`, `values`) are immutable values. Replace them as a whole; every change is
checked against the array shape and leaves the metadata unchanged when it fails:

```python
seg.axes = ["z", "y", "x"]   # raises: the ml boxes still refer to axis_0 ... axis_2
```

### Derived images

`copy=` takes the metadata of another image: geometry, values, profiles and user metadata. It does not
take its history, its `ml` statistics and boxes, or its `source` (pass `copy_source=True` for that).
The `*_like` constructors do the same and also drop label semantics.

```python
import numpy as np
from mlarray import MLArray

ct = MLArray("sample.mla")
prediction = MLArray(np.zeros(ct.shape, dtype=np.uint8), copy=ct, values={"kind": "label"})
empty = MLArray.zeros_like(ct)
```

## Non-spatial axes

Declare what each axis is. Only axes of type `space` take part in `spaces` and in the layout
optimisation; channels, time steps and other axes get chunk size 1.

```python
import numpy as np
from mlarray import MLArray

array = np.random.random((4, 16, 64, 64, 3))   # (time, z, y, x, channels)
image = MLArray(
    array,
    axes=[
        {"name": "t", "type": "time", "start": 0.0, "step": 1.5, "unit": "second"},
        "z", "y", "x",
        {"name": "c", "type": "channel", "labels": ["T1", "T2", "FLAIR"]},
    ],
    spacing=(1.0, 0.8, 0.8),   # for the space axes z, y, x
    space="patient",
)
image.save("multi-axis.mla")
```

---

## Patch size variants

MLArray stores arrays in a chunked layout to enable efficient partial reads. You can control how data is chunked using `patch_size` (recommended in most cases), or manually specify chunk and block sizes when you need full control.

### Default patch size (192)

Uses the default patch configuration, optimized for typical ML patch-based access patterns.

```python
from mlarray import MLArray

image = MLArray("sample.mla")
image.save("default-patch.mla")  # Keeps existing layout metadata

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size='default')
image.save("default-patch-relayout.mla")  # Uses constructor patch_size='default' (192)
```

### Custom isotropic patch size (512)

A larger patch size can improve throughput when you typically load large regions at once (at the cost of slightly less granular random access).

```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size=512)
image.save("patch-512.mla")
```

### Custom non-isotropic patch size

Non-isotropic patches are useful when one axis behaves differently (e.g., fewer slices in Z, anisotropic voxel spacing, or slice-wise training).

```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size=(128, 192, 256))
image.save("patch-non-iso.mla")
```

### Manual chunk/block size

For advanced use cases, you can explicitly define the chunk and block size used by the storage backend.

```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(
    loaded.to_numpy(),
    patch_size=None,
    chunk_size=(1, 128, 128),
    block_size=(1, 32, 32),
)
image.save("manual-chunk-block.mla")
```

### Let Blosc2 itself configure chunk/block size

If you disable MLArray patch sizing, Blosc2 can choose chunk and block sizes automatically. This can be helpful when experimenting or when you want to rely entirely on backend heuristics.

```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size=None)
# If patch_size, chunk_size and block_size are all None, Blosc2 will auto-configure chunk and block size
image.save("blosc2-auto.mla")
```
