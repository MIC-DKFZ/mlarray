<p align="center">
  <img src="https://raw.githubusercontent.com/MIC-DKFZ/mlarray/main/assets/banner.png" alt="{ML Array} banner" width="700" />
</p>

<p align="center">
  <a href="https://pypi.org/project/mlarray/"><img src="https://img.shields.io/pypi/v/mlarray?logo=pypi&color=brightgreen&cacheSeconds=300&v" alt="PyPI" align="middle" /></a>
  <a href="https://pypi.org/project/mlarray/"><img src="https://img.shields.io/pypi/pyversions/mlarray?logo=python&cacheSeconds=300&v" alt="Python Version" align="middle" /></a>
  <a href="https://github.com/MIC-DKFZ/mlarray/actions"><img src="https://img.shields.io/github/actions/workflow/status/MIC-DKFZ/mlarray/workflow.yml?branch=main&logo=github" alt="Tests" align="middle" /></a>
  <a href="https://MIC-DKFZ.github.io/mlarray/"><img src="https://img.shields.io/badge/docs-mlarray-blue?logo=readthedocs&logoColor=white" alt="Docs" align="middle" /></a>
  <a href="https://github.com/MIC-DKFZ/mlarray/blob/main/LICENSE"><img src="https://img.shields.io/github/license/MIC-DKFZ/mlarray" alt="License" align="middle" /></a>
</p>

**tl;dr:** Working with large medical or scientific images for machine learning? -> Use MLArray.

MLArray is a purpose-built file format for *N*-dimensional medical and scientific array data in machine learning workflows. It replaces the usual patchwork of source formats and late-stage conversions to NumPy/Zarr/Blosc2 by layering **standardized metadata** on top of a **Blosc2-backed** storage layout, so the same files work reliably across training, analysis, and visualization tools (including [Napari](https://napari.org) and [MITK](https://www.mitk.org/wiki/The_Medical_Imaging_Interaction_Toolkit_%28MITK%29)).

## Installation

You can install mlarray via [pip](https://pypi.org/project/mlarray/):
```bash
pip install mlarray
```

To enable the `mlarray_convert` CLI command, install MLArray with the necessary extra dependencies:
```bash
pip install mlarray[all]
```

## Documentaion

See the [documentation](https://MIC-DKFZ.github.io/mlarray/) for the [API reference](https://MIC-DKFZ.github.io/mlarray/api/), the [metadata schema](https://MIC-DKFZ.github.io/mlarray/schema/), [usage examples](https://MIC-DKFZ.github.io/mlarray/usage/) or [CLI usage](https://MIC-DKFZ.github.io/mlarray/cli/).

## Usage

Below are common usage patterns for loading, saving, and working with metadata.

### Default usage

```python
import numpy as np
from mlarray import MLArray

array = np.random.random((128, 256, 256))
image = MLArray(array)  # Create MLArray image
image.save("sample.mla")

image = MLArray("sample.mla")  # Loads image
```

### Memory-mapped usage

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

### Metadata inspection and manipulation

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

### Copy metadata with overrides

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

### Standardized metadata usage

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

### Patch size variants

Default patch size (192):
```python
from mlarray import MLArray

image = MLArray("sample.mla")  # Existing file
image.save("default-patch.mla")  # Keeps existing layout metadata

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size='default')
image.save("default-patch-relayout.mla")  # Uses constructor patch_size='default' (192)
```

Custom isotropic patch size (512):
```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size=512)
image.save("patch-512.mla")
```

Custom non-isotropic patch size:
```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size=(128, 192, 256))
image.save("patch-non-iso.mla")
```

Manual chunk/block size:
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

Let Blosc2 itself configure chunk/block size:
```python
from mlarray import MLArray

loaded = MLArray("sample.mla")
image = MLArray(loaded.to_numpy(), patch_size=None)
# If patch_size, chunk_size and block_size are all None, Blosc2 will auto-configure chunk and block size
image.save("blosc2-auto.mla")
```

## CLI

### mlarray_header

Print the metadata header from a `.mla` file.

```bash
mlarray_header sample.mla
```

### mlarray_convert

Convert a NIfTI or NRRD file to MLArray and copy metadata.

```bash
mlarray_convert sample.nii.gz output.mla
```

## Contributing

Contributions are welcome! Please open a pull request with clear changes and add tests when appropriate.

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.
