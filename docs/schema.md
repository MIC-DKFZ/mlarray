# MLArray Metadata Schema

This page describes the metadata stored in an `.mla` file (format version 1). The normative definition is
the [v1 specification](design/spec.md) and the JSON Schemas shipped with the package
(`mlarray/schemas/mlarray-1.0.json`, `profile-ml-1.0.json`, `profile-medical-1.0.json`).

## Storage

An `.mla` file is a blosc2 NDArray. Its metadata lives in three blosc2 `vlmeta` keys, each a msgpack map
that blosc2 compresses separately:

| key | content | read |
|---|---|---|
| `mlarray` | `version`, `writer`, `axes`, `spaces`, `values`, `profiles`, `history`, `storage` | on open |
| `mlarray.source` | `{"source": [...]}`, original metadata of converted files | on first access |
| `mlarray.user` | `{"user": {...}}`, arbitrary user metadata | on first access |

- Only plain msgpack types are stored: maps, arrays, strings, integers, floats, booleans, nil, binary.
  Nothing is pickled, and readers decode the raw bytes with plain msgpack, never through blosc2's own
  vlmeta decoder.
- Opening a file reads only the `mlarray` key, so it stays fast however large `source` is.
- Unknown keys are kept and written back at every level.
- Files written by MLArray before v1 are not readable; convert them again from their original files.

Inspect a file with `mlarray_header file.mla` (add `--source` and `--user` for the lazy keys).

## `version` and `writer`

```yaml
version: "1.0"            # metadata format version; readers accept any 1.x
writer: "mlarray 1.0.0"   # software that wrote the file
```

## `axes`

One entry per array axis, in array (C) order. Required.

| field | meaning |
|---|---|
| `name` | unique name, e.g. `z`, `y`, `x`, `t`, `c`. A name only: it says nothing about direction. Defaults: `axis_0`, `axis_1`, ... |
| `type` | `space` · `time` · `channel` · `component` · `other` |
| `labels` | one label per index, for categorical axes: `["T1", "T2", "FLAIR"]` |
| `start`, `step`, `unit`, `unit_code` | uniform sampling of a non-space axis: coordinate of index *i* = `start + i·step` |
| `coordinates`, `unit`, `unit_code` | explicit coordinate per index (timestamps, b-values, wavelengths) |

Space axes carry no sampling or unit here: their geometry lives in `spaces`.

## `spaces`

Optional. Without spaces the array lives in index space only (photos, ML tensors). Each space is a named
coordinate system and the matrix that maps array indices into it:

```yaml
spaces:
  - name: patient
    reference: {scheme: dicom_frame_of_reference, id: "1.2.826.0.1.3680043.8.498.1"}
    axes:
      - {name: x, unit: millimeter, unit_code: mm, increases_toward: left}
      - {name: y, unit: millimeter, unit_code: mm, increases_toward: posterior}
      - {name: z, unit: millimeter, unit_code: mm, increases_toward: superior}
    from_array:
      array_axes: [z, y, x]
      matrix:                              # rows: space axes x, y, z (+1)
        - [ 0.0, 0.0, 0.7, -170.1]         # columns: array_axes z, y, x (+ translation)
        - [ 0.0, 0.7, 0.0, -201.3]
        - [-2.5, 0.0, 0.0,  -87.0]
        - [ 0.0, 0.0, 0.0,    1.0]
```

- `reference` identifies the physical frame; two spaces are the same frame if `scheme` and `id` match.
- `increases_toward` says in which direction a coordinate grows: `left`/`right`, `anterior`/`posterior`,
  `superior`/`inferior`, `east`/`west`, `north`/`south`, `up`/`down`. It replaces letter codes such as
  `RAS` or `LPS`, which mean different things in different tools. The example is the DICOM patient frame
  (LPS), the recommended world frame for medical images (`space="patient"` in the Python API).
- `from_array.matrix` has one row per space axis plus `[0, ..., 0, 1]`, and one column per entry of
  `array_axes` plus the translation. Rotations, oblique and mirrored grids and shear are allowed.
  Axis-aligned grids may use `scale` and `translation` (per array axis) instead.
- **An integer index is the centre of its voxel:** `matrix · [index, 1]` is the coordinate of the voxel
  centre.
- Spacing, origin, direction and letter codes are derived (`MLArray.spacing`, `.origin`, `.direction`,
  `.letter_codes()`), never stored.
- Several spaces are allowed (e.g. `patient` and `atlas`); the first is the default.

## `values`

What the stored numbers mean. Defaults to `{kind: measurement}`.

| field | meaning |
|---|---|
| `kind` | `measurement` · `label` · `instance` · `probability` · `vector` · `other` |
| `unit`, `unit_code` | unit of the values, e.g. `hounsfield_unit` |
| `scale`, `offset` | physical value = `scale · stored + offset`; stored values are never pre-scaled |
| `fill` | value meaning "no data" |
| `labels` | for label kinds: map from value (as a decimal string) to `{name, color, code}`; `code` is an ontology reference `{scheme, value, meaning}` |
| `label_layout` | `index` (one label value per element) or `channel` (one channel per label) |

## Units

Units are a readable name plus an optional UCUM code. Use a name from the curated list
(`mlarray/schemas/units-1.0.json`: nanometer ... kilometer, second, degree, hounsfield_unit, counts,
percent, kelvin, arbitrary, ...); MLArray fills in `unit_code` for listed names. Free text is allowed
when no listed unit fits.

## `profiles`

Optional, independent domain metadata. Profiles never repeat what the core states.

```yaml
profiles:
  medical:
    profile_version: "1.0"
    dicom: {Modality: MR, BodyPartExamined: BRAIN, RepetitionTime: 2300}   # DICOM keywords
  ml:
    profile_version: "1.0"
    statistics: {min: -1024, max: 3071, mean: 40.2, standard_deviation: 410.0, percentiles: {"99.5": 1500}}
    boxes:
      frame: index            # "index" (array axes) or a space name (space coordinates)
      axes: [z, y, x]
      items: [{min: [10, 50, 60], max: [40, 90, 120], label: 1, score: 0.97}]
    patch_size: [128, 128, 128]
    split: train
```

- `medical.dicom` holds curated medical metadata for files without a DICOM source, or deliberate
  overrides of source values. Files converted from DICOM keep their full DICOM metadata in `source`.
- Profiles from third parties use a reverse-domain name (`org_example_cryoem`) and are kept as they are.

## `history` and `storage`

```yaml
history:
  - {action: converted, tool: "medvol 2.0.0", time: "2026-11-06T10:12:00Z", from: nifti, warnings: [...]}
storage:
  chunk_size: [1, 256, 256]
  block_size: [1, 64, 64]
  patch_size: [128, 128, 128]
  compression: {codec: LZ4HC, level: 5, filters: [SHUFFLE]}
```

`storage` records only portable layout settings. Thread counts, typesize and similar runtime settings
are never stored; they come from the library defaults (see [ML Optimization](optimization.md)).
`meta.storage.to_cparams()` returns the recorded compression as `blosc2.CParams`.

## `source` and `user`

`source` keeps the original metadata of converted files readable and complete, so a converter can write
the original format back. Each entry has `format`, `files`, `reader`, an `index_map` (how the stored
array relates to the source file's own index order), a `fingerprint` (shape, dtype and hashes of the
geometry and data) and the `header` as nested native values. It is read-only, and images derived from
another image do not inherit it unless asked to (`copy_source=True`).

`user` holds any msgpack-compatible metadata of your own and is never interpreted.
