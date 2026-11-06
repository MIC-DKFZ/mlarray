# CLI

MLArray includes a small command-line interface for common tasks such as **inspecting file headers** and **converting between MLArray and existing image formats**. This is especially useful when you want to quickly verify metadata, debug a dataset, or batch-convert files without writing Python code.

The CLI currently focuses on core workflows (header inspection and conversion). Support for converting a wider range of image formats will be added over time.

---

## `mlarray_header`

Print the metadata of a `.mla` file as JSON, without loading the array.

```bash
mlarray_header sample.mla                    # core metadata: axes, spaces, values, profiles, history, storage
mlarray_header sample.mla --source --user    # also the original metadata of converted files and user metadata
```

The output is strict JSON (`jq` accepts it): non-finite numbers are printed as the strings `"NaN"`,
`"Infinity"` and `"-Infinity"`, and binary values as `"<binary, N bytes>"`.

---

## `mlarray_convert`

Convert between MLArray (`.mla`) and NIfTI (`.nii`, `.nii.gz`, `.hdr/.img`), NRRD (`.nrrd`, `.nhdr`), MetaImage
(`.mha`, `.mhd`), MGH/MGZ and DICOM. Conversion is done by [MedVol 2](https://github.com/MIC-DKFZ/MedVol); install
it with `pip install "mlarray[all]"`.

```bash
mlarray_convert sample.nii.gz sample.mla
mlarray_convert sample.mla roundtrip.nii.gz
mlarray_convert dicom_dir/ ct.mla
mlarray_convert ct.mla out_dir/ --target-format dicom
```

What a conversion guarantees (MedVol spec):

- The array keeps its stored values, dtype and storage order. Nothing is reoriented or rescaled; scaling such as
  NIfTI `scl_slope` or DICOM `RescaleSlope` becomes `values.scale`/`offset`.
- The geometry comes from the file's own fields and is stored as an explicit space (DICOM patient frame, LPS, by
  default; `--world-frame ras` for RAS). Voxel positions agree between input and output to the formats'
  precision.
- The original header is kept, readable, in `source`. Converting back to the same format reproduces it; other
  formats get only the fields they can represent.
- Every ambiguity is an explicit policy, and every choice, warning and dropped field is printed and stored in the
  file's `history`.

Policy flags (defaults first): `--nifti-form {nibabel,sform,qform,itk}`, `--nifti-form-disagreement
{warn,error,ignore}`, `--nifti-no-codes {error,spec,nibabel,itk}`, `--world-frame {dicom_patient,ras}`,
`--dicom-tilt {exact,resample,error}`, `--dicom-irregular-spacing {error,split,resample}`, `--dicom-stacks
{split,largest,error}`, `--vector-frame {from_file,lps,ras,error}`, `--on-loss {warn,error}`, `--layout
{source,default}`; `--source-format` and `--target-format` name a format that cannot be detected.
