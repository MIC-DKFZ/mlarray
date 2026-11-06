# Follow-ups after MLArray 1.0

MLArray 1.0 changes the metadata model completely (`docs/design/spec.md`) and cannot read v0 `.mla` files.
The libraries below still use the v0 API and need their own rework. Do them one at a time, each on a rework
branch, with the same working rules as `rework-plan.md`.

## Release path (CD)

For all of these repos: local push → GitLab remote (`codebase.helmholtz.cloud`) → automatic sync to the GitHub
remote → a GitHub Actions workflow on a `v*` tag publishes to PyPI.

| package | PyPI trusted publisher | notes |
|---|---|---|
| mlarray | yes | released as 1.0.0 from `main` |
| MedVol | yes | released as 1.0.0 from `main` (trusted publishing, tests on Python 3.10-3.13 before publishing) |
| napari-mlarray | yes | check the workflow uses trusted publishing |
| SlicerMLArray | no | released through the Slicer extension index, not PyPI (check) |
| napari-nifti | no | publishes with an API token, or set up trusted publishing |

## TODO

### MedVol 1.0 release
- [x] Trusted publishing, `mlarray>=1.0.0`, version 1.0.0, `rework/v2` merged into `main`, tagged `v1.0.0` (on PyPI 2026-11-06).
- [ ] Remaining gaps (mlarray `rework-plan.md` §6): Siemens mosaic (needs dcm2niix), RTSTRUCT writing, vector frame in MetaImage/MGH writers.

### napari-mlarray
- [ ] Read the v1 core: `axes`, `spaces` (from-array matrix), `values` (labels → napari Labels layer with names/colours), profiles.
- [ ] Correct on-screen orientation with the axis-order recipe from the orientation KB (ch06 §6.3).
- [ ] Writer: build v1 metadata instead of the `_original_*` keys and `meta.spatial`.
- [ ] Tests against files written by mlarray 1.0 and MedVol 1.0.

### SlicerMLArray
- [ ] Map the v1 space matrix to Slicer's RAS IJKToRAS (via `increases_toward`, never letter codes); `values.labels` → segmentation with names, colours and codes; boxes in `profiles.ml`.
- [ ] Remove the v0 helpers (`coord_system`, `axis_labels`, affine-based bbox conversion).
- [ ] Export: write v1 metadata.

### napari-nifti
- [ ] Rework on top of MedVol 1.0 (faithful geometry, policies, reports) instead of its own NIfTI handling; same orientation recipe as napari-mlarray.
- [ ] Decide on PyPI publishing (token or trusted publisher).
