# MLArray Metadata Schema

This section defines the **MLArray metadata schema**: the standardized structure used to store and retrieve metadata alongside array data.

The schema is designed around a few core goals:

* **Interoperability:** tools can reliably access common fields (e.g., spacing, orientation, statistics) without guessing conventions.
* **Flexibility:** users can still store arbitrary metadata (including raw metadata from existing formats) without being forced into a rigid structure.
* **Format preservation:** MLArray can act as a storage and ML-optimized alternative to existing image formats (e.g., DICOM, NIfTI, NRRD) while retaining their metadata in a consistent place.
* **Practicality for ML workflows:** fields like `is_seg`, `bbox`, and `_blosc2` directly support common training, preprocessing, and patch-based access patterns.

All fields in the schema are **JSON-serializable** unless otherwise noted. Fields marked as `Optional[...]` may be omitted if unknown or not applicable.

---

## Meta

Top-level metadata container.

### Overview

`Meta` is the root object that groups all metadata into well-defined namespaces. Some namespaces are **standardized** (e.g., `spatial`, `stats`), while others are intentionally **free-form** (`image`, `extra`) to support arbitrary metadata and long-term extensibility.

---

### image

* **Description:** Arbitrary JSON-serializable dictionary for image-level metadata.
  Stores information from medical image sources such as DICOM, NIfTI, NRRD,
  or other medical imaging formats.
* **Dataclass:** None (plain dict).

---

### extra

* **Description:** Flexible container for arbitrary, JSON-serializable metadata
  when no schema exists. Intended for experimental or application-specific
  fields that are not part of the standard.
* **Dataclass:** None (plain dict).

---

### spatial

* **Description:** Spatial metadata for the image.
* **Dataclass:** `MetaSpatial`.

This section stores the information needed to interpret the array in physical space (e.g., voxel spacing, coordinate origin, and orientation). It also optionally captures array shape and channel layout to make downstream consumers more robust.

| field        | type                        | description                                                                              |
| ------------ | --------------------------- | ---------------------------------------------------------------------------------------- |
| spacing      | Optional[List[float]]       | Voxel spacing per spatial axis, length = `ndims`.                                        |
| origin       | Optional[List[float]]       | Origin per spatial axis, length = `ndims`.                                               |
| direction    | Optional[List[List[float]]] | Direction matrix, shape `[ndims][ndims]`.                                                |
| shape        | Optional[List[float]]       | Array shape. If `channel_axis` is set, length = `ndims + 1`, otherwise length = `ndims`. |
| channel_axis | Optional[int]               | Index of channel dimension in the full array, if present.                                |

---

### is_seg

* **Description:** Whether the image is a segmentation mask.
* **Dataclass:** None (boolean).

---

### stats

* **Description:** Summary statistics for the image.
* **Dataclass:** `MetaStatistics`.

This section stores precomputed global statistics for the array, which can be useful for normalization, QA, dataset inspection, and visualization defaults.

| field             | type            | description                                            |
| ----------------- | --------------- | ------------------------------------------------------ |
| min               | Optional[float] | Minimum value.                                         |
| max               | Optional[float] | Maximum value.                                         |
| mean              | Optional[float] | Mean value.                                            |
| median            | Optional[float] | Median value.                                          |
| std               | Optional[float] | Standard deviation.                                    |
| percentile_min    | Optional[float] | Minimum within a selected percentile range.            |
| percentile_max    | Optional[float] | Maximum within a selected percentile range.            |
| percentile_mean   | Optional[float] | Mean within a selected percentile range.               |
| percentile_median | Optional[float] | Median within a selected percentile range.             |
| percentile_std    | Optional[float] | Standard deviation within a selected percentile range. |

---

### bbox

* **Description:** Bounding boxes for objects/regions in the image.
* **Dataclass:** `MetaBbox`.
* **Structure:** List of bboxes, each bbox is a list with length equal to image `ndims`,
  and each entry is `[min, max]`.

Bounding boxes are stored in a normalized, axis-aligned representation that works across dimensionalities (2D, 3D, …). This is especially useful for detection-style workflows, ROI cropping, dataset summaries, and interactive visualization.

| field  | type                            | description                                                          |
| ------ | ------------------------------- | -------------------------------------------------------------------- |
| bboxes | Optional[List[List[List[int]]]] | Bounding boxes shaped `[num_bboxes][ndims][2]` (min/max), ints only. |

---

### _has_array

* **Description:** Whether this metadata instance represents an on-disk array.
* **Dataclass:** None (boolean).

---

### _image_meta_format

* **Description:** Source format identifier for the `image` metadata (e.g., "dicom",
  "nifti", "nrrd"). This is advisory and application-defined.
* **Dataclass:** None (string).

---

### _blosc2

* **Description:** Blosc2 layout parameters.
* **Dataclass:** `MetaBlosc2`.

This section records how the array was laid out on disk (chunking, blocking, patching). It is primarily intended for reproducibility, debugging, and performance introspection.

| field      | type                  | description                                                            |
| ---------- | --------------------- | ---------------------------------------------------------------------- |
| chunk_size | Optional[List[float]] | Chunk size per axis, length = full array `ndims` (including channels). |
| block_size | Optional[List[float]] | Block size per axis, length = full array `ndims` (including channels). |
| patch_size | Optional[List[float]] | Patch size per spatial axis, length = `ndims` (channels excluded).     |

---

### _mlarray_version

* **Description:** MLArray version string used to write the file.
* **Dataclass:** None (string).
