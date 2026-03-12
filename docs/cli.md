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

When converting from MLArray to NIfTI/NRRD, only `meta.source` is copied into the output header. Other MLArray metadata is ignored, while `spacing`, `origin`, and `direction` are written explicitly from `meta.spatial`.

```bash
mlarray_convert sample.nii.gz output.mla
mlarray_convert sample.mla output.nrrd
```
