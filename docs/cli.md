# CLI

MLArray provides CLI commands for common operations such as header inspection or file conversion. We intend to support conversion of a wide range of image formats to MLArray in the future.

## mlarray_header

Print the metadata header from a `.mla` or `.b2nd` file.

```bash
mlarray_header sample.mla
```

## mlarray_convert

Convert a NIfTI or NRRD file to MLArray and copy metadata.

```bash
mlarray_convert sample.nii.gz output.mla
```
