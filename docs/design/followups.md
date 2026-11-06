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
| MedVol | yes | 1.0.1 from `main` (tests on Python 3.10-3.13 before publishing; docs on GitHub Pages) |
| napari-mlarray | yes | workflow uses trusted publishing (`id-token: write`) |
| napari-bboxes | yes | GitHub only (no GitLab mirror); tag `v*` publishes |
| SlicerMLArray | no | released through the Slicer extension index, not PyPI (check) |
| napari-nifti | yes | trusted publishing since 1.0.0 (old token workflow replaced) |

## TODO

### MedVol 1.0 release
- [x] Trusted publishing, `mlarray>=1.0.0`, version 1.0.0, `rework/v2` merged into `main`, tagged `v1.0.0` (on PyPI 2026-11-06).
- [ ] Remaining gaps (mlarray `rework-plan.md` §6): Siemens mosaic (needs dcm2niix), RTSTRUCT writing, vector frame in MetaImage/MGH writers.

### nnunetv2 and nnunetv3: check that they load data correctly

> The nnunetv3 items are **out of scope here**: a dedicated nnunetv3 agent does them (merge into
> `feat/config-schema`, push, end-to-end run on real data).

- [x] **nnunetv3 ported to v1** (2026-11-06): branch `fix/mlarray-v1` (worktree `~/projects/nnunetv3-mlarray-v1`,
  commit `7d769d48`, merged into `project/nnunetv3` (worktree `~/projects/nnunetv3-main`, not pushed).
  `mlarray_geometry()` in `nnunet_dataset.py` turns nnU-Net's array-aligned geometry (world rows S,P,L) into the
  patient space (L,P,S); segmentations are `values.kind = "label"`; the layout is now nnU-Net's own (the old L1/L3
  budgets of the preprocessor are gone); pin `mlarray>=1.0.0,<2`. Test: `nnunetv3/tests/test_mlarray_dataset.py`
  (checks the matrix against nnU-Net's own NIfTI affine). The `nnunetv3` conda env has mlarray 1.0 (editable,
  `~/projects/mlarray`). Preprocessed `.mla` folders written with v0 must be preprocessed again.
- [ ] **nnunetv3:** merge `project/nnunetv3` (or `fix/mlarray-v1`) into `feat/config-schema`: the env's nnunetv3
  install points at `~/projects/nnunetv3` (that branch), which still uses the v0 API. Push `project/nnunetv3`.
- [ ] **nnunetv3:** check that loading keeps the layout and geometry: patch reads, segmentation labels, spacing
  used for resampling (now derived from the space matrix, per array axis in array order), and that the storage
  defaults (nnU-Net layout, LZ4HC-5, read-ahead) still apply (`docs/optimization.md`).
- [ ] **nnunetv2:** it reads blosc2 `.b2nd` files directly and uses neither mlarray nor MedVol. Check its image
  IO (SimpleITK/nibabel readers) against the orientation KB phantom (`~/projects/image-orientation-kb`,
  KB ch05) and decide whether it should move to MedVol/mlarray.

### napari-mlarray
- [x] Rewritten (branch `rework/v1`, merged into `main` at `ce3b141`, pushed; CI green on Linux/macOS/Windows, Python 3.10-3.13; **released as 1.0.0 on 2026-11-06**). Patient spaces are shown in the
  world `I, P, L` (radiological; `I, P, R` neurological, 3-D camera depth flipped), data reordered so the affine has
  no permutation (napari cannot slice permuting affines in 2-D), oblique grids resampled/snapped/kept, labels with
  the label table, channels split, lazy reads above 1 GiB, ml boxes as box-shell Labels (3-D) or rectangles (2-D),
  writer keeps all metadata and writes new layers in LPS. Tests: `tests/test_reader_writer.py` (28); on-screen
  check: `scripts/verify_orientation.py` (26 checks pass under Xvfb, docker image `orient-viewers`, env
  `napari-mlarray`, napari 0.10).
- [x] Box drawing moved to the new package **napari-bboxes** (`~/projects/napari-bboxes`, GitHub
  `Karol-G/napari-bboxes`, pushed directly to GitHub, CI green, trusted publisher configured); napari-mlarray
  `74bdaf9` depends on `napari-bboxes>=0.1`.
- [x] napari-bboxes 0.1.0 and 0.1.1 released (2026-11-06; 0.1.1: 3-D wireframe, slice-following texts, cut outlines).
- [x] napari-mlarray 1.0.0 released (trusted publisher; version from the git tag via setuptools_scm).
- [ ] napari-nifti should reuse `napari_mlarray._display` (the same convention) on top of MedVol.

### SlicerMLArray
- [x] Rewritten for MLArray 1.0 (2026-11-06; `~/projects/SlicerMLArray`, merged into `main` and pushed to GitLab): Slicer
  extension with a pure-Python core; v1 geometry via `increases_toward`, labels as segmentations (names, colours,
  codes), boxes as ROIs, channels/vector/time, saving keeps metadata. 36 pytest tests; in Slicer 5.12.4
  (`~/Slicer-5.12.4-linux-amd64`) `scripts/verify_in_slicer.py` passes 95/95. Plan and status:
  `~/projects/SlicerMLArray/docs/design/rework-plan.md`.
- [x] GitHub mirror live; submitted to the Slicer Extensions Index: https://github.com/Slicer/ExtensionsIndex/pull/2444 (2026-11-06).
- [ ] After the merge and green preview builds: PR to the index's `5.12` branch.

### napari-nifti
- [x] Reworked (`~/projects/napari-nifti`, merged into `main` at `93118f9`, pushed, CI green): reads every MedVol format (NIfTI,
  NRRD, MetaImage, MGH/MGZ, DICOM files and folders) and displays through napari-mlarray's
  `layers_from_meta` (same convention); writes Image/Labels layers through `medvol.write` (`layer_to_array`).
  Integer images whose values look like class ids show as Labels. 14 tests; on-screen check of the KB letter
  phantoms (NIfTI ras/lps/sag/obl25, NRRD) passes. Version 1.0.0, needs napari-mlarray 1.1.0 (`6231d39`).
- [x] napari-mlarray 1.1.0 released; napari-nifti is a PyPI trusted publisher now.
- [x] napari-nifti 1.0.0 released (2026-11-06).
