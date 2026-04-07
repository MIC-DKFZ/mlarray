# CLI

MLArray includes a small command-line interface for common tasks such as **inspecting file headers** and **converting between MLArray and existing image formats**. This is especially useful when you want to quickly verify metadata, debug a dataset, or batch-convert files without writing Python code.

The CLI currently focuses on core workflows (header inspection and conversion). Support for converting a wider range of image formats will be added over time.

---

## `mlarray_header`

Print the metadata header from a `.mla` file.

This command is useful for quickly checking spatial metadata, stored schemas, and other file-level information without loading the full array into memory.

```bash
mlarray_header sample.mla
```

---

## `mlarray_convert`

Convert between MLArray and NIfTI/NRRD files.

This provides an easy way to move medical imaging data into or out of an MLArray-based workflow.

When converting from NIfTI/NRRD to MLArray, the source header is copied into `meta.source`.

When converting from NIfTI/NRRD to MLArray, `meta.spatial.coord_system` is
propagated conservatively:

- For NRRD input, the converter looks for explicit NRRD space metadata in the
  header (`space`, falling back to `NRRD_space` from the SimpleITK path). It
  maps `right-anterior-superior` to `RAS`, `left-posterior-superior` to `LPS`,
  and preserves other explicit NRRD space strings verbatim. If no explicit
  space metadata is available, `coord_system` is left unset.
- For NIfTI input, the converter sets `coord_system` to `LPS`. This is based on
  the current `MedVol -> SimpleITK/ITK` import path: the imported geometry is
  represented in the ITK physical-space convention, and MedVol only reindexes
  axes to match NumPy array layout. It is not a statement that raw NIfTI
  qform/sform metadata is preserved verbatim.

When converting from MLArray to NIfTI/NRRD, only `meta.source` is copied into
the output header. Spatial metadata (`spacing`, `origin`, and `direction`) is
written explicitly from `meta.spatial`.

- For NRRD output, `coord_system="RAS"` and `coord_system="LPS"` are written as
  the corresponding NRRD `space` strings. A small set of already-valid NRRD
  space strings is also preserved. Unsupported arbitrary `coord_system` strings
  are not written as NRRD `space` metadata.
- For NIfTI output, geometry is preserved through the existing affine-related
  path, but `coord_system` is not exported as an explicit NIfTI metadata field
  because the current conversion path does not expose a reliable direct
  representation for it.

```bash
mlarray_convert sample.nii.gz output.mla
mlarray_convert sample.mla output.nrrd
```
