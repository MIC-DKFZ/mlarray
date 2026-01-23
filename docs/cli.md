# CLI

MLArray includes a small command-line interface for common tasks such as **inspecting file headers** and **converting existing image formats** into MLArray. This is especially useful when you want to quickly verify metadata, debug a dataset, or batch-convert files without writing Python code.

The CLI currently focuses on core workflows (header inspection and conversion). Support for converting a wider range of image formats will be added over time.

---

## `mlarray_header`

Print the metadata header from a `.mla` or `.b2nd` file.

This command is useful for quickly checking spatial metadata, stored schemas, and other file-level information without loading the full array into memory.

```bash
mlarray_header sample.mla
```

---

## `mlarray_convert`

Convert a NIfTI or NRRD file to MLArray and copy metadata.

This provides an easy way to bring existing medical imaging data into an MLArray-based workflow while preserving the original metadata for downstream analysis and visualization.

```bash
mlarray_convert sample.nii.gz output.mla
```
