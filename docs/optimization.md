# ML Optimization

MLArray is designed around a simple goal: **make training-time I/O fast and predictable**, especially for large N-D images where reading full volumes is impractical. Most ML pipelines repeatedly sample **small patches/crops** (e.g., nnU-Net-style random patch sampling) and the storage layout should match that access pattern.

To achieve this, MLArray builds on **Blosc2 ND arrays**, which store data in a two-level tiled layout:

* **Chunks**: the unit stored on disk and fetched from storage.
* **Blocks**: the unit decompressed; a read decompresses only the blocks it touches.

On top of that, MLArray supports **memory-mapped access** via Blosc2, so reading `image[x0:x1, y0:y1, ...]` touches *only the required on-disk regions*, instead of loading the whole array.

MLArray's defaults come from a benchmark of nnU-Net-style patch loading on a compute cluster (random patches, 1 to 12 loader processes, network and node-local storage, data in and not in the page cache, several datasets). In short: **nnU-Net's layout and codec, `mmap_mode='r'` and blosc2's read-ahead** were as fast as every alternative we tested, and they write fastest.

### Patch-size-driven layout

You tell MLArray the **patch size** you expect to sample during training (e.g., `192³`), and it derives `chunk_size` and `block_size` with nnU-Net's heuristic, generalized to any number of axes:

1. **Non-spatial axes** (channels, time, ... from `axis_labels`) get chunk and block size 1, so reading one channel touches only its own data.
2. **Blocks** start at the next power of two of the patch size per spatial axis and are halved until they fit **128 KiB**.
3. **Chunks** grow by whole blocks on the least covered axis until they reach **6 MiB**, the image extent, or **1.5x the patch size** per axis.
4. A patch entry of 1 (e.g. `(1, 256, 256)` for 2D slices of a 3D image) keeps that axis at chunk size 1.

For 2D and 3D images the layout is identical to nnU-Net's. In the benchmark, smaller chunks (1.5 MiB) with LZ4HC level 9 read 4-10% faster than nnU-Net's files without read-ahead; with read-ahead the difference was gone (within 2 points), while they wrote up to 3x slower.

### Compression

By default MLArray compresses with **LZ4HC level 5** and **4 compression threads**, and chooses the **filter per array** as nnU-Net does: it compresses a centered chunk-sized slab with and without SHUFFLE and keeps the better one (resampled intensities usually favour SHUFFLE, integer-valued ones no filter). Without data to look at (`MLArray.create`, `empty`, ...), blosc2's default SHUFFLE applies. Pass `cparams` to use anything else, e.g. `{"codec": blosc2.Codec.ZSTD, "clevel": 5}` for smaller, slower-to-read archives.

### Reading: memory mapping and read-ahead

* **`mmap_mode='r'`** (the default of `MLArray.open`) with **1 decompression thread** per reader. With read-ahead on, plain reads (`mmap_mode=None`) were no faster for uncached data and slower for cached data.
* **Read-ahead**: the first `MLArray.open` in a process turns on blosc2's read-ahead with **4 threads** (`blosc2.set_prefetch(4)`). Reading a slice then fetches all chunks it needs in parallel instead of one small request after another. In the benchmark, with 12 loader processes reading 3D patches that were not in the page cache from network storage, this cut the time per read by 64-82% (3-5x faster), for larger-than-memory datasets by about 70-78% in every epoch. Reader threads in one process each get their own read-ahead threads, so threaded readers gain as much as processes.
* For data already in the page cache, read-ahead costs a little: about 0-8% per patch with 12 processes and up to about 17% with a single reader.
* Change or turn it off with `mlarray.set_read_ahead(n)` (`0` = off) before opening files, or set the `BLOSC_PREFETCH` environment variable, which MLArray leaves alone. The setting applies to the whole process. It needs a blosc2 with read-ahead support (`blosc2.set_prefetch`); with older versions nothing changes.

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

* you want to compare MLArray’s layout against Blosc2 defaults,
* you don’t have a meaningful patch size (non-ML access patterns).

---

## Notes and practical tips

* **Patch optimization works for any number of spatial and non-spatial axes**; for 2D and 3D images it gives nnU-Net's layout. You can still set `chunk_size`/`block_size` manually or let Blosc2 decide.
* **Reading from network storage or datasets larger than RAM:** keep read-ahead on (the default). If your data always fits in RAM and is read many times by a single process, `mlarray.set_read_ahead(0)` can save a few percent per read.
* The best patch size to use is usually the **patch size your dataloader requests most often** (training patch, not necessarily inference tile size).
* If you’re unsure: start with the default (`patch_size='default'`) and only tune if profiling shows I/O bottlenecks.
