# MedBlosc2 Metadata Schema

This document defines the MedBlosc2 metadata schema.

## Meta

Top-level metadata container.

### image

- Description: Arbitrary JSON-serializable dictionary for image-level metadata.
  Stores information from medical image sources such as DICOM, NIfTI, NRRD,
  or other medical imaging formats.
- Dataclass: None (plain dict).

### spatial

- Description: Spatial metadata for the image.
- Dataclass: `MetaSpatial`.

| field | type | description |
| --- | --- | --- |
| shape | Optional[List[float]] | The shape of the image, length = `ndims`. |
| spacing | Optional[List[float]] | Voxel spacing per axis, length = `ndims`. |
| origin | Optional[List[float]] | Origin per axis, length = `ndims`. |
| direction | Optional[List[List[float]]] | Direction matrix, shape `[ndims][ndims]`. |

### stats

- Description: Summary statistics for the image.
- Dataclass: `MetaStatistics`.

| field | type | description |
| --- | --- | --- |
| min | Optional[float] | Minimum value. |
| max | Optional[float] | Maximum value. |
| mean | Optional[float] | Mean value. |
| median | Optional[float] | Median value. |
| std | Optional[float] | Standard deviation. |
| percentile_min | Optional[float] | Minimum within a selected percentile range. |
| percentile_max | Optional[float] | Maximum within a selected percentile range. |
| percentile_mean | Optional[float] | Mean within a selected percentile range. |
| percentile_median | Optional[float] | Median within a selected percentile range. |
| percentile_std | Optional[float] | Standard deviation within a selected percentile range. |

### bbox

- Description: Bounding boxes for objects/regions in the image.
- Dataclass: `MetaBbox`.
- Structure: List of bboxes, each bbox is a list with length equal to image `ndims`, and each entry is `[min, max]`.

| field | type | description |
| --- | --- | --- |
| bboxes | Optional[List[List[List[int]]]] | Bounding boxes shaped `[num_bboxes][ndims][2]` (min/max), ints only. |

### is_seg

- Description: Whether the image is a segmentation mask.
- Dataclass: None (boolean).

### _blosc2

- Description: Blosc2 layout parameters.
- Dataclass: `MetaBlosc2`.

| field | type | description |
| --- | --- | --- |
| chunk_size | Optional[List[float]] | Chunk size per axis, length = `ndims`. |
| block_size | Optional[List[float]] | Block size per axis, length = `ndims`. |
| patch_size | Optional[List[float]] | Patch size per axis, length = `ndims`. |

### _med_blosc2_version

- Description: MedBlosc2 version string used to write the file.
- Dataclass: None (string).

### extra

- Description: Flexible container for arbitrary, JSON-serializable metadata
  when no schema exists. Intended for experimental or application-specific
  fields that are not part of the standard.
- Dataclass: None (plain dict).
