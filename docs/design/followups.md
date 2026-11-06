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
| MedVol | yes | `rework/v2`, version 1.0.0.dev0; release after mlarray 1.0.0 is on PyPI (needs `mlarray>=1.0`). Its workflow (`test_and_deploy.yml`) still uploads with a twine token and must be switched to trusted publishing (`id-token: write`, `pypa/gh-action-pypi-publish`) |
| napari-mlarray | yes | check the workflow uses trusted publishing |
| SlicerMLArray | no | released through the Slicer extension index, not PyPI (check) |
| napari-nifti | no | publishes with an API token, or set up trusted publishing |

## TODO

### MedVol 1.0 release
- [ ] Switch `test_and_deploy.yml` to trusted publishing; check that the workflow file name matches the PyPI registration.
- [ ] Pin `mlarray>=1.0.0` (currently `>=1.0.0.dev0`), set the version to 1.0.0, merge `rework/v2` into `main`, tag `v1.0.0`.
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
