# MLArray v1 — format and metadata specification (draft)

Status: **draft for discussion, not implemented.** Branch `rework/v1`. Last update: 2026-11-06.
Companion spec: MedVol v2 converter (`medvol` repo, branch `rework/v2`, `docs/design/spec.md`).

Keywords **MUST**, **SHOULD** and **MAY** are normative. Everything marked *informative* is explanation.

---

## 1. Goals and non-goals

**Goals**

1. One array format for scientific images of any domain: medical, digital pathology, microscopy, astronomy,
   earth observation, physics, biology, natural images.
2. A metadata model that is **small, explicit and readable**, with optional domain depth and room for
   arbitrary metadata.
3. Geometry that cannot be misread. Every convention that has historically caused bugs is stated in the
   file, never implied: array axis order, world frame, axis direction, voxel centre. (Evidence:
   `~/projects/image-orientation-kb`.)
4. Faithful round trips. Converting format X → MLArray → X (via MedVol) reproduces X's metadata and array,
   to the accuracy expected of a file format (§9).
5. Fast access: metadata needed to use the array loads in about a millisecond, and large original metadata
   loads only on demand (§3).

**Non-goals**

- De-identification or any other modification of metadata. Metadata is stored as given.
- Backwards compatibility with v0 files. There is no migration; v0 files must be re-converted from their
  sources.
- Reading or writing other formats inside `mlarray`. That is MedVol's job. `mlarray` depends on numpy and
  blosc2 only.

## 2. Terminology (informative)

| Term | Meaning |
|---|---|
| array axis | one dimension of the stored array, in memory order (`array.shape` order, C order, last axis fastest) |
| index | integer position along the array axes; index `(0,…,0)` is the first element |
| space | a named physical or reference coordinate system: patient, microscope stage, slide, sky, map projection, … |
| from-array transform | the mapping from array index to coordinates in one space |
| core | metadata every MLArray reader must understand (§4) |
| profile | optional, versioned domain metadata (§5) |
| source | the original metadata of the file(s) the array was converted from (§6) |

**Style rules for all keys and vocabulary.** `snake_case`, full English words, no abbreviations except
`id`, and no domain jargon in the core. Terms are chosen so that a pathologist, an astronomer and a
radiologist read them the same way.

## 3. Container

- An `.mla` file is a blosc2 NDArray file (the array payload) with metadata in blosc2 variable-length
  metadata (`vlmeta`).
- Metadata is stored in **three `vlmeta` keys**. Each value is a msgpack map, written through blosc2,
  which serialises with msgpack and compresses each key separately:

| key | content | loaded |
|---|---|---|
| `mlarray` | `version`, `axes`, `spaces`, `values`, `profiles`, `history`, `storage` | on open (small) |
| `mlarray.source` | `source` (§6), may be very large | on first access |
| `mlarray.user` | `user` (§7), arbitrary, may be very large | on first access |

- Metadata **MUST NOT** use pickle or any other executable serialisation.
- Values **MUST** be msgpack-native types: map, array, string, integer, float, boolean, nil, binary.
  Map key order is preserved.

*Informative. Benchmark (blosc2 4.14.1, 2026-11-06), synthetic DICOM-like headers:*

| original headers | blosc2 metadata on disk | open file | read `mlarray` key | read `mlarray.source` key | current gzip+pickle |
|---|---|---|---|---|---|
| 300 slice headers, 0.58 MB | 0.30 MB | 0.2 ms | 0.8 ms | 6.5 ms | 155 ms |
| 3000 slice headers, 5.85 MB | 3.02 MB | 0.5 ms | 0.8 ms | 55 ms | 1793 ms |

Opening the file stays well under 1 ms whatever the size of the source metadata. The test data was
random, so real headers compress better than the ~2× shown.

## 4. Core metadata

### 4.1 `version`

```yaml
version: "1.0"            # metadata spec version (semver). Required.
writer: "mlarray 1.0.0"   # software that wrote the file. Required.
```

Readers **MUST** accept any `1.x` and **MUST** keep keys they don't know when rewriting a file.

### 4.2 `axes` — the array axes

One entry per array axis, in array order. `len(axes) == array.ndim`. Required.

| field | required | meaning |
|---|---|---|
| `name` | yes | unique name of the array axis, e.g. `z`, `y`, `x`, `t`, `c`. A name only: it says nothing about direction in space. |
| `type` | yes | `space` · `time` · `channel` · `component` (vector or tensor components) · `other` (any other categorical or indexed axis: echo, b-value, batch, …) |
| `labels` | no | one label per index for categorical axes: `["T1", "T2", "FLAIR"]`, `["red", "green", "blue"]` |
| `start`, `step`, `unit` | no | uniform sampling of a non-space axis: coordinate of index *i* = `start + i·step` in `unit`, e.g. time in seconds |
| `coordinates`, `unit` | no | explicit coordinate per index for non-uniform sampling: timestamps, b-values, wavelengths |

Space axes carry no spacing or unit here. Their geometry lives in `spaces` (§4.3), so it has exactly one
source of truth.

### 4.3 `spaces` — coordinate systems and how the array sits in them

Optional. Without `spaces` the array lives in index space only, which is valid for photos, ML tensors and
the like.

```yaml
spaces:
  - name: patient
    reference: {scheme: dicom_frame_of_reference, id: "1.2.826.0.1.3680043.8.498.1"}
    axes:
      - {name: x, unit: millimeter, increases_toward: left}
      - {name: y, unit: millimeter, increases_toward: posterior}
      - {name: z, unit: millimeter, increases_toward: superior}
    from_array:
      array_axes: [z, y, x]
      matrix:                              # rows: space axes x, y, z (+1)
        - [ 0.0, 0.0, 0.7, -170.1]         # columns: array_axes z, y, x (+ translation)
        - [ 0.0, 0.7, 0.0, -201.3]
        - [-2.5, 0.0, 0.0,  -87.0]
        - [ 0.0, 0.0, 0.0,    1.0]
```

| field | required | meaning |
|---|---|---|
| `name` | yes | unique space name, chosen freely (`patient`, `stage`, `slide`, `sky`, `atlas`, `map`) |
| `reference` | no | identifies the physical frame, so files can be compared: `{scheme, id}`. Two spaces are the **same frame iff `scheme` and `id` are equal**. Examples: `dicom_frame_of_reference`, `atlas` (`MNI152NLin2009cAsym`), `epsg` (`32632`), `celestial` (`ICRS`) |
| `axes` | yes | the space's own axes, in order: `name`, `unit`, optional `increases_toward` |
| `from_array` | yes | how array indices map into this space (below) |

**`increases_toward`** gives the direction in which a coordinate grows, as a plain word from a vocabulary:
- medical: `left` / `right` / `anterior` / `posterior` / `superior` / `inferior`;
- earth observation: `east` / `west` / `north` / `south` / `up` / `down`;
- profiles may add more.

It replaces letter codes such as `RAS`/`LPS`/`RAI`, whose meaning differs between tools. The example above
is the DICOM patient frame (LPS). Libraries **MAY** derive letter codes for display, but they **MUST NOT**
store them.

**`from_array`** rules:
1. `array_axes` lists the array axes that map into this space, by name, in the order of the matrix columns.
   Each listed axis must have `type: space`.
2. `matrix` is a homogeneous matrix with shape `(len(space.axes) + 1) × (len(array_axes) + 1)`.
   - Row *r* gives space coordinate *r*.
   - Column *c* multiplies array axis `array_axes[c]`.
   - The last column is the translation, and the last row is `[0, …, 0, 1]`.
   - Rotation, oblique axes, mirrored grids and shear (e.g. gantry-tilted CT kept exactly as acquired) are
     all allowed.
3. Shorthand for axis-aligned grids: `scale: [...]` and `translation: [...]` (per array axis, in
   `array_axes` order) instead of `matrix`. Exactly one form **MUST** be present.
4. **An integer index is the centre of its voxel or pixel.** `matrix · [index, 1]` is the coordinate of the
   voxel centre.
5. Profiles **MAY** define non-linear mappings (e.g. astronomy sky projections) under
   `from_array.profile_mapping`. A reader that doesn't support it still uses `matrix` as the linear
   approximation, which **MUST** then be present.
6. `spacing`, `origin`, `direction` and letter codes are derived by libraries on request and are never
   stored.

Several spaces are allowed, e.g. `patient` plus `atlas` after registration, or `stage` plus `slide`. The
first space is the default.

### 4.4 `values` — what the numbers mean

Optional. Defaults to `{kind: measurement}`.

| field | meaning |
|---|---|
| `kind` | `measurement` (physical or intensity values) · `label` (integer class per element) · `instance` (integer object id per element) · `probability` · `vector` · `other` |
| `unit` | unit of the physical value: `hounsfield_unit`, `counts`, `kelvin`, `reflectance`, `arbitrary`, … |
| `scale`, `offset` | physical value = `scale · stored + offset`. Stored values are never pre-applied (keeps exact round trips, §9). |
| `fill` | value meaning "no data / outside the field of view" |
| `labels` | for `label`/`instance`/`probability`: map from value (or channel index) to `{name, color, code}` |
| `label_layout` | `index` (one label value per element, default) or `channel` (one channel per label: one-hot or probabilities, using the `channel` axis) |

```yaml
values:
  kind: label
  labels:
    "0": {name: background}
    "1": {name: liver, color: [221, 130, 101], code: {scheme: SCT, value: "10200004", meaning: Liver}}
```

`code` is an optional ontology reference: scheme, value and human-readable meaning. For medical
segmentations it carries the DICOM SEG coded concept.

## 5. Profiles

```yaml
profiles:
  medical: {profile_version: "1.0", ...}
  ml:      {profile_version: "1.0", ...}
```

1. A profile adds domain metadata. It **MUST NOT** change the meaning of core fields.
2. Profiles are optional, independent and combinable.
3. Readers **MUST** keep profiles they don't understand.
4. Official profiles are published with the spec as JSON Schemas. Third-party profiles use a reverse-domain
   name (`org_example_cryoem`).
5. **One fact, one place:**
   - A profile **MUST NOT** copy what the core states (geometry, units, labels).
   - It **SHOULD NOT** copy what `source` already contains verbatim; libraries expose source fields
     directly instead (§6.4).
   - A profile field that does overlap a source field is a deliberate, explicit override. Read precedence is
     **core > profile > source**.

### 5.1 `medical` profile

Vocabulary: the **DICOM data dictionary**. Keys are DICOM keywords, values are plain msgpack values.

```yaml
medical:
  profile_version: "1.0"
  dicom:                       # DICOM keywords not (or differently) present in source
    Modality: MR
    BodyPartExamined: BRAIN
    RepetitionTime: 2300
    EchoTime: 2.98
```

- Typical use:
  - curated or user-provided medical metadata for files without a DICOM source, e.g. public NIfTI
    datasets;
  - deliberate overrides of DICOM source values.
- Files converted from DICOM normally need no `medical.dicom` entries: their full DICOM metadata is in
  `source`, addressable by keyword.
- Recommended world frame for medical images: the **DICOM patient frame** (x increases toward left, y toward
  posterior, z toward superior; LPS), with `reference.scheme: dicom_frame_of_reference` when the
  FrameOfReferenceUID is known. MedVol writes this by default.
- Recommended array layout: the acquisition grid as stored by the source, in C order (no reorientation).

### 5.2 `ml` profile

```yaml
ml:
  profile_version: "1.0"
  statistics: {min: -1024, max: 3071, mean: 40.2, standard_deviation: 410.0,
               percentiles: {"0.5": -1000, "99.5": 1500}}
  boxes:
    frame: index              # "index" (array axes below) or a space name (space coordinates)
    axes: [z, y, x]
    items:
      - {min: [10, 50, 60], max: [40, 90, 120], label: 1, score: 0.97}
  patch_size: [128, 128, 128]
  split: train
```

### 5.3 Further profiles (informative, defined later)

| profile | adds | geometry usage |
|---|---|---|
| `microscopy` | per-channel excitation and emission wavelength, fluorophore, objective (numerical aperture, magnification, immersion) | `stage` space, micrometer |
| `pathology` | stain, objective power, scanner, specimen and slide identifiers | `slide` space, micrometer; microns per pixel = scale |
| `astronomy` | telescope, instrument, exposure, FITS WCS keywords | `sky` space, non-linear projection via `profile_mapping` |
| `earth_observation` | bands (name, centre wavelength), acquisition time, sensor | `map` space with `reference.scheme: epsg` |
| `natural` | camera and lens (EXIF subset), colour space | usually none |
| `physics` | quantity names, solver, per-component units | SI-unit space, `component` axis |

## 6. Source — the original metadata

### 6.1 Purpose and rules

`source` keeps the original metadata **readable, authentic and complete**, so that MedVol can write the
original format back faithfully and users can inspect everything the source contained.

1. `source` is **read-only** in the library API. To change metadata, edit the core or a profile.
2. **The core owns everything it describes.** When exporting, MedVol regenerates core-owned source fields
   from the core: geometry, dimensions, data type, value scale/offset, units, labels. They can therefore
   never contradict the core. On an unchanged round trip they regenerate to the original values (§9).
3. **Data-dependent fields** (e.g. NIfTI `cal_min`, DICOM `LargestPixelValue`, per-slice `SliceLocation`)
   are kept only while the array still matches the `fingerprint` (§6.3). Otherwise MedVol recomputes or
   drops them and records that in `history`.
4. Images derived from another image (predictions, crops, resamples) **MUST NOT** inherit its `source`
   implicitly. Copying source is an explicit opt-in.
5. Which fields are core-owned or data-dependent is defined per format in the MedVol spec, not in the file.

### 6.2 Structure

```yaml
source:
  - format: nifti                       # nifti | nrrd | metaimage | mgh | dicom | mla | ...
    format_version: "1"
    files: ["sub-01_T1w.nii.gz"]        # original file name(s), relative, informative
    reader: {name: medvol, version: "2.0.0"}
    index_map:                          # how MLArray indices relate to the source file's own indices
      axis_order: [2, 1, 0]             # source axis feeding each MLArray axis
      flipped: [false, false, false]
      offset: [0, 0, 0]
    fingerprint: {shape: [176, 256, 256], dtype: int16, geometry: "<hash>", data: "<xxh3 hash>"}
    header: { ... readable original metadata, §6.4 ... }
```

`index_map` keeps raw geometry fields (NIfTI `srow_*`, DICOM `ImagePositionPatient`, NRRD
`space directions`) interpretable. They refer to the source's own index order, and `index_map` says how that
relates to the stored array.

### 6.3 Fingerprint

Written at conversion time:
- `shape` and `dtype`;
- a hash of the core geometry (axes and spaces, after rounding to the accuracy of §9);
- a fast hash of the array data (xxh3).

MedVol compares it on export to decide whether data-dependent fields are still valid.

### 6.4 Readable header representation

The header is stored as nested maps with native types, not as raw bytes:

- **Numbers are numbers.** Integers stay integers. Floats are stored as the shortest decimal value that
  converts back to the original binary value: a float32 `0.7` is stored as `0.7`, not
  `0.699999988079071`.
- **Text is text.** Fixed-length or null-padded strings are stored stripped, and MedVol re-pads on export.
- **Field order is preserved.**
- **Binary content** (NIfTI extensions with binary payload, DICOM OB/OW/UN elements) is stored as msgpack
  binary. Viewers display such values as `<binary, N bytes>`.
- **DICOM:**
  - Public elements are keyed by keyword (`Modality`, `ImagePositionPatient`). Private elements are keyed by
    tag `"(0029,1010)"` and carry `{vr, value}`, because their type is not in the dictionary.
  - Sequences are lists of maps.
  - A multi-file series is stored as `shared` (identical in every file, stored once) plus `per_file` (a list
    with only the fields that differ, in file order) plus `file_names`.
- **Format-layout fields** (NIfTI `vox_offset`, NRRD `encoding`, DICOM transfer syntax) are kept like any
  other field. MedVol reuses them on export unless told otherwise.

## 7. `user`

Arbitrary msgpack-compatible metadata, never interpreted or validated by MLArray. It replaces v0's `extra`
and lives in its own lazily loaded key.

## 8. `history` and `storage`

```yaml
history:
  - {action: converted, tool: "medvol 2.0.0", time: "2026-11-06T10:12:00Z",
     from: nifti, settings: {nifti_form: nibabel},
     warnings: ["qform and sform disagree by 4.9 mm; used sform (sform_code=2)"]}
storage:
  chunk_size: [1, 256, 256]
  block_size: [1, 64, 64]
  patch_size: [128, 128, 128]
```

`history` is an append-only list of conversions and operations with tool, version, settings and warnings.
`storage` describes the blosc2 layout; it is an implementation detail with no scientific meaning.

## 9. Accuracy standard

The guarantee is "what a careful file format guarantees", not byte identity:

| item | guarantee |
|---|---|
| array values | bit-identical. Stored values are kept in the original dtype, with no rescaling or reorientation unless explicitly requested. |
| integers, strings, booleans, binary | identical |
| floating-point header values | identical binary value when the source stores binary floats (NIfTI, MGH, DICOM FL/FD). For text-encoded numbers (NRRD, MetaImage, DICOM DS/IS) the numeric value is identical, but the textual spelling may differ (`0.69999999999999996` ↔ `0.7`, `1.0` ↔ `1`). |
| geometry through `from_array.matrix` | float64; round-trip error ≤ 1e-9 relative. Below every format's own precision (NIfTI srow is float32; DICOM DS has 16 characters). |
| field order and set of fields | identical for same-format round trips, except fields the format itself defines as derived (e.g. NIfTI `vox_offset` when the extension size changes) |
| file bytes | not guaranteed: compression level, gzip timestamps, padding and whitespace may differ |

Cross-format conversion (X → MLArray → Y, Y ≠ X) preserves the core. MedVol reports which source fields Y
cannot represent.

## 10. Validation

The normative definitions are JSON Schemas shipped in the repo: `schemas/mlarray-1.0.json`,
`schemas/profile-medical-1.0.json`, `schemas/profile-ml-1.0.json`. They are applied to the msgpack content
as if it were JSON, with binary allowed in `source`/`user`. The Python model implements the schemas.
Third-party readers (SlicerMLArray, napari-mlarray, future C++/JS) validate against the same files.

## 11. Python API (informative sketch)

```python
img = MLArray("ct.mla")                       # opens; loads key "mlarray" only
img.axes                                      # [Axis(name="z", type="space"), ...]
img.space()                                   # default space; img.space("atlas")
img.matrix()                                  # from-array matrix of the default space
img.spacing, img.origin, img.direction        # derived, read-only
img.letter_codes()                            # derived display helper, e.g. ("L","P","S"), points-to
img.values.kind, img.values.labels
img.profiles["medical"]["dicom"]
img.source                                    # lazy; read-only view
img.source[0].header["RepetitionTime"]        # DICOM keyword access
img.user                                      # lazy

MLArray(array, axes=[...], spaces=[...], values=...)        # explicit construction
MLArray(array, spacing=..., origin=..., direction=..., space="patient")  # convenience; composed into matrix
```

## 12. Open questions

See the companion discussion. Items still open in this spec:

1. Unit vocabulary: a curated list of readable names (proposed), with free strings allowed, and optionally an
   additional machine code (UCUM) per unit?
2. Multi-resolution pyramids (pathology, microscopy): part of v1, or a later extension?
3. More than one array per file (e.g. image + segmentation): out of scope (proposed)?
