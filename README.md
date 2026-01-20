MedBlosc2
=========

![PyPI](https://img.shields.io/pypi/v/med-blosc2?logo=pypi&color=brightgreen)
![Python Version](https://img.shields.io/pypi/pyversions/med-blosc2?logo=python)
![Tests](https://img.shields.io/github/actions/workflow/status/Karol-G/med-blosc2/workflow.yml?branch=main&logo=github)
![Copier Template](https://img.shields.io/badge/copier-template-blue?logo=jinja)
![License](https://img.shields.io/github/license/Karol-G/med-blosc2)

A standardized Blosc2 image reader and writer for medical images. The MedBlosc2
file format (".mb2nd") is a Blosc2-compressed container with standardized
metadata support for N-dimensional medical images. Plain ".b2nd" files are also
supported, but they do not participate in the MedBlosc2 metadata standard.

## Installation

You can install med-blosc2 via [pip](https://pypi.org/project/med-blosc2/):
```bash
pip install med-blosc2
```

To enable the `medblosc2_convert` CLI command, install MedBlosc2 with the necessary extra dependencies:
```bash
pip install med-blosc2[all]
```

## API

See [API.md](API.md) for the full MedBlosc2 api, including argument
descriptions and types.

## Metadata schema

See [SCHEMA.md](SCHEMA.md) for the full MedBlosc2 metadata schema, including field
descriptions and types.

## Usage

Below are common usage patterns for loading, saving, and working with metadata.

### Default usage

```python
import numpy as np
from med_blosc2 import MedBlosc2

array = np.random.random((128, 256, 256))
image = MedBlosc2(array)  # Create MedBlosc2 image
image.save("sample.mb2nd")

image = MedBlosc2("sample.mb2nd")  # Loads image
```

### Memory-mapped usage

```python
from med_blosc2 import MedBlosc2
import numpy as np

# read-only, partial access (default)
image = MedBlosc2().open("sample.mb2nd", mmap='r')  
crop = image.array[10:20, 50:60]  # Read crop

# read/write, partial access
image = MedBlosc2().open("sample.mb2nd", mmap='r+')  
image.array[10:20, 50:60] *= 5  # Modify crop in memory and disk

# read/write, partial access, create/overwrite
array = np.random.random((128, 256, 256))
image = MedBlosc2().open("sample.mb2nd", shape=array.shape, dtype=array.dtype, mmap='w+')  
image.array[...] = array  # Modify image in memory and disk
```

### Metadata inspection and manipulation

```python
import numpy as np
from med_blosc2 import MedBlosc2

array = np.random.random((64, 128, 128))
image = MedBlosc2(
    array,
    spacing=(1.0, 1.0, 1.5),
    origin=(10.0, 10.0, 30.0),
    direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    meta={"patient_id": "123", "modality": "CT"},  # Any image metadata (for example raw DICOM metadata)
)

print(image.spacing)  # [1.0, 1.0, 1.5]
print(image.origin)  # [10.0, 10.0, 30.0]
print(image.meta.image)  # {"patient_id": "123", "modality": "CT"}

image.spacing[1] = 5.3
image.meta.image["study_id"] = "study-001"
image.save("with-metadata.mb2nd")
```

### Copy metadata with overrides

```python
import numpy as np
from med_blosc2 import MedBlosc2

base = MedBlosc2("sample.mb2nd")
array = np.random.random(base.shape)

image = MedBlosc2(
    array,
    spacing=(0.8, 0.8, 1.0),
    copy=base,  # Copies all non-explicitly set arguments from base
)

image.save("copied-metadata.mb2nd")
```

### Standardized metadata usage

```python
import numpy as np
from med_blosc2 import MedBlosc2, Meta

array = np.random.random((64, 128, 128))
image = MedBlosc2(
    array,
    meta=Meta(image={"patient_id": "123", "modality": "CT"}, is_seg=True),  # Add metadata in a pre-defined format
)

print(image.meta.image)  # {"patient_id": "123", "modality": "CT"}
print(image.meta.is_seg)  # True

image.meta.image["study_id"] = "study-001"
image.meta.is_seg = False
image.save("with-metadata.mb2nd")
```

### Patch size variants

Default patch size (192):
```python
from med_blosc2 import MedBlosc2

image = MedBlosc2("sample.mb2nd")
image.save("default-patch.mb2nd")  # Default patch_size is 'default' -> Isotropic patch size of 192 pixels
image.save("default-patch.mb2nd", patch_size='default')
```

Custom isotropic patch size (512):
```python
from med_blosc2 import MedBlosc2

image = MedBlosc2("sample.mb2nd")
image.save("patch-512.mb2nd", patch_size=512)
```

Custom non-isotropic patch size:
```python
from med_blosc2 import MedBlosc2

image = MedBlosc2("sample.mb2nd")
image.save("patch-non-iso.mb2nd", patch_size=(128, 192, 256))
```

Manual chunk/block size:
```python
from med_blosc2 import MedBlosc2

image = MedBlosc2("sample.mb2nd")
image.save("manual-chunk-block.mb2nd", chunk_size=(1, 128, 128), block_size=(1, 32, 32))
```

Let Blosc2 itself configure chunk/block size:
```python
from med_blosc2 import MedBlosc2

image = MedBlosc2("sample.mb2nd")
# If patch_size, chunk_size and block_size are all None, Blosc2 will auto-configure chunk and block size
image.save("manual-chunk-block.mb2nd", patch_size=None)
```

## CLI

### medblosc2_header

Print the metadata header from a `.mb2nd` or `.b2nd` file.

```bash
medblosc2_header sample.mb2nd
```

### medblosc2_convert

Convert a NIfTI or NRRD file to MedBlosc2 and copy metadata.

```bash
medblosc2_convert sample.nii.gz output.mb2nd
```

## Contributing

Contributions are welcome! Please open a pull request with clear changes and add tests when appropriate.

## Issues

Found a bug or have a request? Open an issue at https://github.com/Karol-G/med-blosc2/issues.

## License

Distributed under the MIT license. See `LICENSE` for details.
