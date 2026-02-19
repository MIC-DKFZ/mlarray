# ML Optimization

MLArray is designed around a simple goal: **make training-time I/O fast and predictable**, especially for large N-D images where reading full volumes is impractical. Most ML pipelines repeatedly sample **small patches/crops** (e.g., nnU-Net-style random patch sampling) and the storage layout should match that access pattern.

To achieve this, MLArray builds on **Blosc2 ND arrays**, which store data in a two-level tiled layout:

* **Chunks**: larger partitions that are typically sized to fit higher-level CPU caches (and amortize overhead).
* **Blocks**: smaller partitions inside chunks that are typically sized to fit lower-level CPU caches and improve decompression speed.

On top of that, MLArray supports **memory-mapped access** via Blosc2, so reading `image[x0:x1, y0:y1, ...]` can touch *only the required on-disk regions*, instead of loading the whole array.

The key challenge is that choosing good `chunk_size` and `block_size` is **hard**:

* the optimal values depend on CPU cache sizes, dtype, dimensionality, and—most importantly—your **training patch size**,
* and the wrong choices can silently tank throughput.

### Patch-size-driven layout optimization

Instead of requiring users to be storage experts, MLArray introduces a **patch size optimization**:

1. You tell MLArray the **patch size** you expect to sample during training (e.g., `192³`).
2. MLArray derives **block_size** and **chunk_size** automatically to match this access pattern.
3. Internally, the heuristic considers:

   * element size (bytes per pixel),
   * CPU cache sizes (L1 / L3 per core),
   * your patch size (2D or 3D),
   * non-spatial axes layout (via `axis_labels`),
   * and then chooses block/chunk sizes that aim to keep decompression and reads cache-friendly.

Practically: this means *reading a training patch should tend to require as few chunk/block touches as possible*, while keeping the decompressed working set aligned with CPU caches.

---

## When should I care?

* If you train with **patch sampling** (most medical imaging pipelines do): set `patch_size`.
* If you do mostly **full-volume reads**: patch sizing matters less; larger chunks may be fine.
* If you have a very specific access pattern or hardware constraint: set `chunk_size` / `block_size` manually.

---

## Usage patterns

Below are common end-to-end workflows. The examples show the important knobs and what they do.

### 1) “Just do the right thing” (recommended)

Use the default patch size optimization. If you don’t specify anything, MLArray uses an isotropic default patch size of **192** (per spatial axis) and derives chunk/block sizes automatically.

```python
import numpy as np
from mlarray import MLArray

array = np.random.random((128, 256, 256))
image = MLArray(array)

# Uses patch_size='default' (192) -> auto-derives chunk/block sizes
image.save("default-opt.mla")
```

When to use:

* you want good performance without tuning,
* your training patch size is close to ~192 (common in many 3D pipelines).

---

### 2) Optimize explicitly for your training patch size

If you know your sampler will draw patches of a specific size, set `patch_size` accordingly. This makes the on-disk layout match your training-time reads more closely.

```python
import numpy as np
from mlarray import MLArray

array = np.random.random((128, 256, 256))
image = MLArray(array, patch_size=(128, 192, 256))

# Optimize storage layout for 3D patches of 128×192×256 (spatial axes)
image.save("patch-non-iso.mla")
```

When to use:

* your patch sampling is strongly anisotropic (common with anisotropic spacing),
* you have a fixed patch size in your training config and want to match it.

---

### 3) Memory-mapped patch reads (training-style access)

For large files, you typically want **mmap reads** so random patches don’t require loading the entire array into RAM.

```python
from mlarray import MLArray

# read-only mmap: fast random access without loading the full volume
image = MLArray.open("patch-non-iso.mla", mmap_mode='r')

patch = image[10:20, 50:60]  # Read a crop/patch (partial read)
```

When to use:

* dataset is too large to fit in RAM,
* you do random access reads (patch sampling, interactive slicing).

---

### 4) Memory-mapped in-place modification (advanced)

You can modify regions in-place with `mmap_mode='r+'`. This is useful for workflows like:

* writing derived arrays (e.g., post-processing outputs),
* patch-wise updates,
* annotation edits (careful with concurrency).

```python
from mlarray import MLArray

image = MLArray.open("patch-non-iso.mla", mmap_mode='r+')
image[10:20, 50:60] *= 5  # Modify crop in memory and on disk
image.close()
```

---

### 5) Create a new memory-mapped file (streaming write)

If you want to create a file on disk and then fill it (without holding the full array in memory), use `create(..., shape=..., dtype=..., mmap_mode='w+')`. MLArray will compute and store the optimized layout up front.

```python
import numpy as np
from mlarray import MLArray

shape = (128, 256, 256)
dtype = np.float32

image = MLArray.create(
    "streamed-write.mla",
    shape=shape,
    dtype=dtype,
    mmap_mode='w+',
    patch_size=192,  # optimize for your training patch size
)

# Fill incrementally if you want (here we write everything at once)
image[...] = np.random.random(shape).astype(dtype)
image.close()
```

When to use:

* you generate data on the fly,
* you want to avoid a full in-memory intermediate array.

---

### 6) Manual chunk/block sizing (experts only)

If you already know what you’re doing (or want to reproduce a very specific layout), you can override the automatic optimization. Note that in MLArray, `patch_size` and `chunk_size`/`block_size` are mutually exclusive.

```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(
    loaded.to_numpy(),
    patch_size=None,
    chunk_size=(1, 128, 128),
    block_size=(1, 32, 32),
)
image.save("manual-layout.mla")
```

When to use:

* you benchmarked and found a better layout for your hardware/access pattern,
* you need strict reproducibility across environments.

---

### 7) Let Blosc2 auto-configure chunk/block sizes

If you set `patch_size=None` (and don’t provide chunk/block sizes), Blosc2 will choose chunk/block sizes itself. This can be useful for experimentation or as a baseline.

```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size=None)

# If patch_size, chunk_size and block_size are all None, Blosc2 auto-configures
image.save("blosc2-auto.mla")
```

When to use:

* you want to compare MLArray’s patch optimization against Blosc2 defaults,
* you don’t have a meaningful patch size (non-ML access patterns).

---

## Notes and practical tips

* **Patch optimization is currently implemented for 2D and 3D images** (with at most one further non-spatial axis). If your data falls outside that, you can still set `chunk_size`/`block_size` manually or let Blosc2 decide.
* The best patch size to use is usually the **patch size your dataloader requests most often** (training patch, not necessarily inference tile size).
* If you’re unsure: start with the default (`patch_size='default'`) and only tune if profiling shows I/O bottlenecks.
