# MLArray Metadata Schema

This section defines the **MLArray metadata schema**: the standardized structure used to store and retrieve metadata alongside array data.

The schema is designed around a few core goals:

* **Interoperability:** tools can reliably access common fields (e.g., spacing, orientation, statistics) without guessing conventions.
* **Flexibility:** users can still store arbitrary metadata (including raw metadata from existing formats) without being forced into a rigid structure.
* **Format preservation:** MLArray can act as a storage and ML-optimized alternative to existing image formats (e.g., DICOM, NIfTI, NRRD) while retaining their metadata in a consistent place.
* **Practicality for ML workflows:** fields like `is_seg`, `bbox`, and `_blosc2` directly support common training, preprocessing, and patch-based access patterns.

All fields in the schema are **JSON-serializable** unless otherwise noted. Fields marked as `Optional[...]` may be omitted if unknown or not applicable.

### Representation notes

Many namespaces are implemented as **single-key dataclasses** (subclasses of `SingleKeyBaseMeta`). In Python, these behave like their wrapped value (e.g., `meta.is_seg` is a `bool`, `meta.source` is a `dict`). When serialized via `Meta.to_mapping()`, they appear as a one-field object keyed by their internal field name (e.g., `is_seg: {"is_seg": true}` or `source: {"data": {...}}`). `Meta.from_mapping()` accepts either the one-field object or the raw value and will coerce it to the correct class.

---

## Meta

Top-level metadata container.

### Overview

`Meta` is the root object that groups all metadata into well-defined namespaces. Some namespaces are **standardized** (e.g., `spatial`, `stats`), while others are intentionally **free-form** (`source`, `extra`) to support arbitrary metadata and long-term extensibility. Several entries are single-key dataclasses that wrap a primitive value while still allowing schema-aware validation.

---

### source

* **Description:** Arbitrary JSON-serializable dictionary for metadata from the original image source.
  Stores information from image sources such as DICOM, NIfTI, NRRD,
  or other imaging formats.
* **Dataclass:** `MetaSource` (single-key wrapper).

| field | type                  | description                                  |
| ----- | --------------------- | -------------------------------------------- |
| data  | Dict[str, Any]        | JSON-serializable metadata from the source. |

---

### extra

* **Description:** Flexible container for arbitrary, JSON-serializable metadata
  when no schema exists. Intended for experimental or application-specific
  fields that are not part of the standard.
* **Dataclass:** `MetaExtra` (single-key wrapper).

| field | type           | description                                  |
| ----- | -------------- | -------------------------------------------- |
| data  | Dict[str, Any]  | JSON-serializable metadata for extra fields. |

---

### spatial

* **Description:** Spatial metadata for the image.
* **Dataclass:** `MetaSpatial`.

This section stores the information needed to interpret the array in physical space (e.g., voxel spacing, coordinate origin, and orientation). It also optionally captures array shape and axes layout to make downstream consumers more robust.

| field        | type                        | description                                                                              |
| ------------ | --------------------------- | ---------------------------------------------------------------------------------------- |
| spacing      | Optional[List[float]]       | Voxel spacing per spatial axis, length = `ndims`.                                        |
| origin       | Optional[List[float]]       | Origin per spatial axis, length = `ndims`.                                               |
| direction    | Optional[List[List[float]]] | Direction matrix, shape `[ndims][ndims]`.                                                |
| shape        | Optional[List[int]]         | Full array shape, length = spatial + non-spatial axes.                                   |
| axis_labels  | Optional[List[str,AxisLabel]]   | Per-axis labels or roles, length = full array `ndims`.                                   |
| axis_units   | Optional[List[str]]         | Per-axis units, length = full array `ndims`.                                             |

---

#### AxisLabel

Axis labels describe the semantic role of each axis. They may be provided as strings or enum values.

| value         | description                                              |
| ------------- | -------------------------------------------------------- |
| spatial       | Generic spatial axis (used when no axis-specific label). |
| spatial_x     | Spatial axis representing X.                             |
| spatial_y     | Spatial axis representing Y.                             |
| spatial_z     | Spatial axis representing Z.                             |
| non_spatial   | Generic non-spatial axis.                                |
| channel       | Channel axis (e.g., color channels or feature maps).     |
| temporal      | Time axis.                                               |
| continuous    | Continuous-valued axis (non-spatial).                    |
| components    | Component axis (e.g., vector components).                |

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
| percentile_min_key    | Optional[float] | Minimum percentile key used to determine percentile_min (for example 0.05).            |
| percentile_max_key    | Optional[float] | Maximum percentile key used to determine percentile_max (for example 0.95).            |

---

### bbox

* **Description:** Bounding boxes for objects/regions in the image.
* **Dataclass:** `MetaBbox`.
* **Structure:** List of bboxes, each bbox is a list with length equal to image `ndims`,
  and each entry is `[min, max]`.

Bounding boxes are stored in a normalized, axis-aligned representation that works across dimensionalities (2D, 3D, …). This is especially useful for detection-style workflows, ROI cropping, dataset summaries, and interactive visualization.

| field  | type                                            | description                                                                 |
| ------ | ----------------------------------------------- | --------------------------------------------------------------------------- |
| bboxes | Optional[List[List[List[Union[int, float]]]]]   | Bounding boxes shaped `[num_bboxes][ndims][2]` (min/max), ints or floats.   |
| scores | Optional[List[Union[int, float]]]               | Optional confidence scores aligned with `bboxes`.                           |
| labels | Optional[List[Union[str, int, float]]]          | Optional labels aligned with `bboxes`.                                      |

---

### is_seg

* **Description:** Whether the image is a segmentation mask.
* **Dataclass:** `MetaIsSeg` (single-key wrapper).

| field  | type           | description                                     |
| ------ | -------------- | ----------------------------------------------- |
| is_seg | Optional[bool] | True/False when known, None when unknown.       |

---

### _blosc2

* **Description:** Blosc2 layout parameters.
* **Dataclass:** `MetaBlosc2`.

This section records how the array was laid out on disk (chunking, blocking, patching). It is primarily intended for reproducibility, debugging, and performance introspection.

| field      | type                  | description                                                            |
| ---------- | --------------------- | ---------------------------------------------------------------------- |
| chunk_size | Optional[List[float]] | Chunk size per axis, length = full array `ndims` (including non-spatial axes). |
| block_size | Optional[List[float]] | Block size per axis, length = full array `ndims` (including non-spatial axes). |
| patch_size | Optional[List[float]] | Patch size per spatial axis, length = `ndims` (non-spatial axes excluded).     |

---

### _has_array

* **Description:** Whether this metadata instance represents an on-disk array.
* **Dataclass:** `MetaHasArray` (single-key wrapper).

| field     | type | description                           |
| --------- | ---- | ------------------------------------- |
| has_array | bool | True when an array payload is stored. |

---

### _image_meta_format

* **Description:** Source format identifier for the `image` metadata (e.g., "dicom",
  "nifti", "nrrd"). This is advisory and application-defined.
* **Dataclass:** `MetaImageFormat` (single-key wrapper).

| field             | type           | description                                   |
| ----------------- | -------------- | --------------------------------------------- |
| image_meta_format | Optional[str]  | Identifier for the source metadata format. |

---

### _mlarray_version

* **Description:** MLArray version string used to write the file.
* **Dataclass:** `MetaVersion` (single-key wrapper).

| field           | type           | description                          |
| --------------- | -------------- | ------------------------------------ |
| mlarray_version | Optional[str]  | Version string for the writer.       |
