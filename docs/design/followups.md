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

### nnunetv2 and nnunetv3: check that they load data correctly
- [x] **nnunetv3 ported to v1** (2026-11-06): branch `fix/mlarray-v1` (worktree `~/projects/nnunetv3-mlarray-v1`,
  commit `7d769d48`, based on `project/nnunetv3`, merges cleanly into `feat/config-schema`; not pushed, not merged).
  `mlarray_geometry()` in `nnunet_dataset.py` turns nnU-Net's array-aligned geometry (world rows S,P,L) into the
  patient space (L,P,S); segmentations are `values.kind = "label"`; the layout is now nnU-Net's own (the old L1/L3
  budgets of the preprocessor are gone); pin `mlarray>=1.0.0,<2`. Test: `nnunetv3/tests/test_mlarray_dataset.py`
  (checks the matrix against nnU-Net's own NIfTI affine). The `nnunetv3` conda env still has mlarray v0: upgrade
  it when the branch is merged. Preprocessed `.mla` folders written with v0 must be preprocessed again.
- [ ] **nnunetv3:** check that loading keeps the layout and geometry: patch reads, segmentation labels, spacing
  used for resampling (now derived from the space matrix, per array axis in array order), and that the storage
  defaults (nnU-Net layout, LZ4HC-5, read-ahead) still apply (`docs/optimization.md`).
- [ ] **nnunetv2:** it reads blosc2 `.b2nd` files directly and uses neither mlarray nor MedVol. Check its image
  IO (SimpleITK/nibabel readers) against the orientation KB phantom (`~/projects/image-orientation-kb`,
  KB ch05) and decide whether it should move to MedVol/mlarray.

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
