# MLArray / MedVol rework — process and handoff

This document lets any agent or developer continue the rework without prior context. Read it fully before
touching code. Keep it current: when a stage step is done, tick it and add the commit hash.

Last update: 2026-11-06 (stages 1-4 done and released; follow-up work in `followups.md`).

## Start here (current state)

- **Released:** mlarray 1.0.0 and MedVol 1.0.0 are on PyPI (2026-11-06). `pip install "mlarray[all]"` installs
  both; `mlarray_convert` delegates to MedVol.
- **Branches:** mlarray `main` == `rework/v1`; MedVol `main` == `rework/v2` (both pushed). Work on `main` or a new
  feature branch from now on.
- **Next work:** `docs/design/followups.md`. The most urgent item is nnunetv3, which still uses the v0 API.
  Open gaps and decisions are in §6 below.
- **Envs:** `mlarray` (mlarray dev; its `mlarray` editable install points at the old review worktree, so run tests
  with `python -m pytest` from the repo root) and `medvol-v2` (both repos installed editable; use it for MedVol and
  for mlarray's CLI tests).
- **Tests:** `cd ~/projects/mlarray && ~/miniforge3/envs/medvol-v2/bin/python -m pytest -q -p no:cacheprovider`
  (163 passed) and `cd ~/projects/medvol && ~/miniforge3/envs/medvol-v2/bin/python -m pytest -q -p no:cacheprovider tests/`
  (170 passed).
- **Releasing (CD):** local push → GitLab (`codebase.helmholtz.cloud`) → automatic sync to GitHub → GitHub Actions
  publishes `v*` tags to PyPI by trusted publishing (mlarray: `.github/workflows/workflow.yml`; MedVol:
  `test_and_deploy.yml`, tests on Python 3.10-3.13 first). Version: mlarray from the git tag (setuptools_scm),
  MedVol from `medvol/__init__.py`. Pushing a tag is a public action: ask the user to run `git push origin vX.Y.Z`
  themselves. Check on Python 3.10 too: pip resolves blosc2 4.3.3 there, which behaves differently.
  Docs (GitHub Pages, `gh-pages`) deploy with each `v*` tag, or on demand with
  `gh workflow run workflow.yml -R MIC-DKFZ/mlarray --ref main`.
  MedVol's docs (https://mic-dkfz.github.io/MedVol/, sources in `medvol/docs/*.md` + `mkdocs.yml`) deploy the same
  way (`gh workflow run test_and_deploy.yml -R MIC-DKFZ/MedVol --ref main`).
- **MedVol 1.0.1** (2026-11-06) adds the `medvol` command line (1.0.0 declared it without `medvol/cli.py`);
  released to PyPI and docs deployed.

---

## 0. Orientation

### What the rework is

1. MLArray gets a new metadata model: domain-neutral core + profiles + readable `source` + `user`, plain
   msgpack, explicit geometry.
2. MedVol becomes a faithful N:N converter built on that model.
3. Before that, the existing storage and array code is cleaned up and its known bugs are fixed, without
   changing user-facing behaviour or storage optimisation.

### Normative documents (read these first)

| document | where | content |
|---|---|---|
| MLArray v1 spec | `mlarray` repo, branch `rework/v1`, `docs/design/spec.md` | format, metadata model, accuracy standard, decisions log (§12) |
| MedVol v2 spec | `medvol` repo, branch `rework/v2`, `docs/design/spec.md` | converter architecture, field ownership, policies, per-format rules, decisions log (§10) |
| Storage defaults | `mlarray` `rework/v1`, `docs/optimization.md` | why the nnU-Net layout heuristic, LZ4HC-5 and read-ahead are the defaults |
| Review findings | branch `review/findings`, worktree `~/projects/mlarray-review`, `FINDINGS.md` | open and fixed findings against old `main` (`c95290b`); finding IDs (C1, H2, M12 …) are used below |
| Orientation knowledge base | `~/projects/image-orientation-kb` (start with `README.md`, `docs/00-pitfall-index.md`) | verified behaviour of DICOM/NIfTI/NRRD, libraries and viewers; the evidence behind many spec rules ("KB") |

The spec is the source of truth. If implementation shows that the spec is wrong or incomplete:
- stop;
- write the problem and a proposed change into §6 of this document;
- ask the user before changing the spec.

### Repositories and branches

| repo | path | working branch | notes |
|---|---|---|---|
| mlarray | `~/projects/mlarray` | `rework/v1` | **all of stages 1–3 happen here** (user decision: no intermediate branch or release) |
| mlarray review | `~/projects/mlarray-review` (worktree) | `review/findings` | reference only, **never merge it**; cherry-pick or re-implement |
| medvol | `~/projects/medvol` | `rework/v2` | stage 4 |
| consumers | `~/projects/SlicerMLArray`, `~/projects/napari-mlarray` | — | adapt after stage 3 (§5) |

`design/metadata-v1` (mlarray) holds an earlier proposal that the spec supersedes. Ignore it.

### Environment and commands

- Conda env `mlarray` (`~/miniforge3/envs/mlarray/bin/python`): Python 3.12, blosc2 4.14.1, numpy 2.5.4,
  medvol 0.0.22, nibabel, pynrrd.
- Tests: `cd ~/projects/mlarray && ~/miniforge3/envs/mlarray/bin/python -m pytest -q -p no:cacheprovider`.
  - **Careful:** the env's editable install of `mlarray` points at `~/projects/mlarray-review`. `python -m pytest`
    from the repo root imports the local package, but scripts run elsewhere import the review code unless
    `PYTHONPATH=~/projects/mlarray` is set.
  - **Baseline on `rework/v1` @ `8f0511c` + spec commits: 111 passed, 3038 subtests passed.**
- For MedVol work, use the experiment envs of the KB (`orient-kb` etc., see `~/projects/image-orientation-kb/TOOLS.md`) to cross-check against other libraries.

### Working rules (from the user)

- Commit after every sensible, complete, working change. Use separate, focused commits per logical change.
- Commit messages and docs **must not** mention AI, Claude or any assistant, and must not contain
  `Co-Authored-By` trailers.
- Don't push unless asked.
- **No backwards compatibility** with v0 `.mla` files and no migration code.
- **No de-identification** and no other modification of metadata.
- Functionality, user-facing usage and storage optimisation must not change in stages 1–2. Any bug fix that
  changes observable behaviour must be listed in §6 and confirmed by the user before it lands.
- **Invariant through all stages:** `tests/test_layout.py` and `tests/test_read_ahead.py` stay unchanged and
  green (one agreed exception: in stage 3, `test_layout.py`'s metadata accessors change, see §3). They pin
  the storage defaults:
  - the nnU-Net chunk/block heuristic in `mlarray/blosc2_layout_strategies.py`;
  - LZ4HC level 5 with 4 compression threads, and nnU-Net's per-array filter choice;
  - blosc2 read-ahead with 4 threads, switched on by the first `MLArray.open` (`mlarray.set_read_ahead()`).

---

## 1. Stage 1 — port surviving review fixes onto `rework/v1`

Goal: bring over every fix from `review/findings` that concerns code surviving the rework, each with its
tests. Prefer `git cherry-pick -x <hash>` when it applies cleanly. Otherwise re-implement by hand and copy
the tests. `mlarray/mlarray.py` changed on `rework/v1`, so expect conflicts there. Drop `FINDINGS.md` hunks.

| done | review commit | content | findings | notes |
|---|---|---|---|---|
| ☑ `12dc805` | `6c52edd` | atomic `save()`: write to temp file, write metadata into it, `os.replace`; never modify the source file; reopen with the same mode when saving onto itself | C1, H1, M2 | brings `tests/test_save.py`. Keep the metadata writing behind one function so stage 3 can swap the serialisation |
| ☑ `4ea3528` | `2ce1242` | `flush()` and context manager (`with MLArray.open(...) as a:`) | M3 | |
| ☑ `8e9859b` | `ac433cb` | `arange` length via `ceil`; float detection via `np.floating` | H2, H3 | |
| ☑ `e267669` | `4f6604b` | 0-d arrays stay 0-d; `create(mmap_mode=None)` overwrites; `__array__(copy=)`; opening `r+`/`a` doesn't rewrite metadata; int `shape` in `create`; `patch_size` validation; numpy ints in `patch_size` | M5 (part), M6, L1, L2, L3, L5, M9 (part) | its `tests/test_optimization.py` hunk: keep only what still applies to the new layout heuristic |
| ☑ `7bc9eed` | `012c896` | packaging and CI: CI on Python 3.10–3.13, real pytest exit code, banner not installed as data files, keywords | L16, L18, Nit | the `medvol>=0.0.22` pin (L17) is obsolete: in stage 4 mlarray's CLI delegates to MedVol v2. Pin the new version then |
| ☑ `b4239a6` | `1ab1641` | benchmark measurement fixes | M19–M22 | **only the `bench/benchmark_formats.py` and `bench/README.md` parts**. The other two scripts were deleted on `rework/v1` |
| ☑ `efafaf1` | `bed5c54` | examples write to a temp dir instead of the cwd | Nit | skip the `docs/schema.md` and `docs/optimization.md` hunks: schema docs are rewritten in stage 3, and optimisation docs were rewritten on `rework/v1` |

**Not ported** (handled by the redesign instead):
- `45c3ff4`, `35b613c`, `1ac9bcc` (msgpack storage, lazy source/extra decoding): stage 3 implements the
  spec's 3-key storage. Reuse their ideas and tests as a reference.
- `33776c6`, `d7b0daa`, `f035c55` (metadata fixes in `meta.py`): `meta.py` is replaced in stage 3. Their
  findings become stage-3 requirements (§3).

**Moot** (code no longer exists on `rework/v1`):
- H9: fixed by the new layout heuristic;
- M24, L15: old layout strategies, now deleted;
- H13 remainder, L13, L14: deleted benchmark scripts.

**Done when:** every row is ticked, the full suite is green (baseline tests + ported tests), and each port is
its own commit.

**Result (2026-11-06):** all 7 ported in the review branch's order (`4f6604b` before `2ce1242`, because
`2ce1242`'s tests build on it). Suite: 127 passed, 3056 subtests.

---

## 2. Stage 2 — fix the open findings that survive, then restructure the code

### 2a. Fixes (one commit per finding, with a regression test)

| done | finding | what to do |
|---|---|---|
| ☑ `28b61ab` | **M1** not picklable | `__getstate__`/`__setstate__`. Opened files store `(filepath, mode, mmap_mode, dparams)` and reopen; in-memory blosc2 stores serialise via `to_cframe()`; numpy-backed arrays pickle the array. Test with `multiprocessing` spawn and a torch-free DataLoader-style worker. |
| ☑ `bc0e993` | **M4** `save()` on numpy-backed arrays converts the instance to blosc2 and loses `patch_size` | save from a temporary blosc2 store; the instance stays unchanged |
| ☑ `9f6a2ba` | **M5** remainder: empty arrays and zero-length dimensions | stop using the `(0,)` placeholder as the "no array" marker; use an explicit flag. Layout: handle zero-length axes without calling the heuristic, or clamp |
| ☑ `549e5b0` | **M12** machine-specific `cparams` persisted | persist only portable settings (codec, level, filters; spec §8 `storage`). Thread counts come from runtime defaults. Must keep `tests/test_layout.py`/`test_read_ahead.py` green |
| ☑ `b94ae9b` | **L4** `load(compressed=False)` keeps `meta.blosc2` while `decompress()` clears it | pick one rule (keep the layout record) and apply it in both |
| ☑ `e004e6c` `b5e87eb` `a936880` | Nits | consistent optional `dtype` for `empty`/`zeros`; document memory sharing of `asarray(compressed=False)`; arrays with only non-spatial axes work without `patch_size=None` (check if the new heuristic already solves this); fix docstrings ("ignored when patch_size is provided") |

Not in stage 2:
- metadata findings M10, M11, M13, L7, L9: they belong to stage 3;
- CLI findings H10–H12, M14–M18, M25, L11, L12 and the CLI nits: they belong to stage 4.

### 2b. Restructure (behaviour-preserving)

Target layout (adjust if the code suggests better boundaries; record deviations here):

```
mlarray/
  __init__.py          public API re-exports (unchanged names)
  array.py             MLArray class: construction, slicing, numpy interop, constructors (zeros, arange, *_like …)
  storage.py           blosc2 IO: open/create/load/save (atomic), flush, pickling, compressed/uncompressed backends
  layout.py            chunk/block heuristic + filter selection (today blosc2_layout_strategies.py; keep the public import path working)
  readahead.py         set_read_ahead and the default
  metadata_io.py       the ONLY place that serialises/deserialises metadata (stage 3 swaps its internals)
  meta.py              current metadata model (replaced in stage 3)
  cli.py               current CLI (replaced in stage 4)
```

Rules:
- Keep the public API and import paths, e.g. `from mlarray import MLArray, Meta, set_read_ahead`. Check
  `mlarray/__init__.py` and `docs/api.md`.
- Pure moves first (one commit), then internal simplifications (separate commits).
- The full test suite passes after every commit. Don't edit existing tests except for import paths.

**Result of 2b (2026-11-06)**, commits `3a0bce7` `2c63460` `da11b87` `47914c1`. The layout above was adjusted
because the invariant test `tests/test_read_ahead.py` does `import mlarray.mlarray as mlmod`, resets
`mlmod._read_ahead_configured` and patches `mlmod.blosc2`:

```
mlarray/
  mlarray.py           MLArray class: __init__, open/create/load/save/close/flush, pickling, numpy interop,
                       properties, metadata glue; set_read_ahead and its state (pinned here by test_read_ahead.py)
  constructors.py      MLArrayConstructors mixin: empty, zeros, ones, full, arange, linspace, asarray, *_like
  storage.py           stateless blosc2 helpers: cparams/dparams, comp_blosc2_meta (layout metadata), whole-file
                       load, atomic save, pickling helpers; file constants (suffix, version, default patch size)
  metadata_io.py       the ONLY place that encodes/decodes metadata (stage 3 swaps its internals)
  blosc2_layout_strategies.py   unchanged (invariant tests and docs/optimization.md reference it; no layout.py)
  meta.py, cli.py      unchanged until stages 3 and 4
```

Benchmark (`bench/benchmark_formats.py`, 8 synthetic CT-like int16 NIfTI volumes 256×256×160–216, patch
128×128×64, 2000 reads, warm cache, 3 alternating rounds per version, blosc2 4.14.1):

| version | mlarray MB/s (3 rounds) | blosc2 control MB/s | mlarray disk |
|---|---|---|---|
| stage 1 (`d853721`) | 718, 721, 703 | 300, 305, 306 | 116.29 MB |
| stage 2 (`47914c1`) | 684, 709, 712 | 303, 299, 297 | 116.29 MB |

No regression beyond run-to-run noise. Decoded data, metadata, layout and cparams of the written files are
identical. Compressed chunk bytes differ even between two runs of the same code (multi-threaded blosc2
compression is not byte-deterministic), which spec §9 allows.

**Done when:**
- all 2a rows are ticked;
- `mlarray.py` is split;
- metadata serialisation goes through `metadata_io.py` only;
- the suite is green;
- the benchmark (`bench/benchmark_formats.py`) shows no throughput regression against stage 1. Run it on
  synthetic data and note the numbers here.

---

## 3. Stage 3 — metadata rework (replace `meta.py`) per MLArray spec

Implement `docs/design/spec.md` §3–§11. Suggested commit sequence:

1. ☑ `d6feea3` JSON Schemas: `schemas/mlarray-1.0.json`, `schemas/profile-medical-1.0.json`,
   `schemas/profile-ml-1.0.json`, `schemas/units-1.0.json` (curated unit names + UCUM codes; spec §4.5).
2. ☑ `c70a82e` Storage in `metadata_io.py` (spec §3):
   - three vlmeta keys `mlarray`, `mlarray.source`, `mlarray.user`;
   - write msgpack;
   - **read raw bytes and decode with plain msgpack** (never blosc2's vlmeta decoder);
   - lazy decoding of `source`/`user`;
   - reference: review commits `45c3ff4`, `35b613c` and `tests/test_metadata_storage.py`.
3. ☑ `1846be9` New model classes: `Axis`, `Space` (+ `from_array`), `Values`, `Profiles`, `SourceEntry`, `History`,
   `Storage`. Validation against the schemas. Unknown keys and unknown profiles are preserved (spec §4.1,
   §5).
4. ☑ `1846be9` `c70a82e` Derived geometry API (spec §11):
   - `matrix()`, read-only `spacing`/`origin`/`direction`, `letter_codes()`;
   - convenience constructor (`spacing=, origin=, direction=, space=`) composed into a matrix with the
     documented column convention;
   - world geometry without a declared space is refused;
   - fixes the affine-derived property nits: return types, division by zero, shear for non-orthogonal
     matrices.
5. ☑ `1846be9` `values`: `kind`, `unit`/`unit_code`, `scale`/`offset`, `fill`, `labels`, `label_layout`. This replaces
   `is_seg`.
6. ☑ `1846be9` `ml` profile: `statistics`, `boxes` with an explicit `frame` (replaces `stats`, `bbox`), `patch_size`,
   `split`. `medical` profile skeleton.
7. ☑ `1846be9` `source`: read-only API, `index_map`, `fingerprint` (xxh3 data hash; add `xxhash` as a dependency, or use
   blosc2/numpy hashing if adequate; decide and record here).
8. ☑ `c70a82e` `9f612db` Constructors, `copy=` and `*_like` follow the new rules:
   - geometry is copied;
   - `source` is **not** inherited unless `copy_source=True`;
   - statistics, boxes and label semantics are not copied by `*_like` (fixes M13).
9. ☑ `c70a82e` `mlarray_header` prints valid JSON (NaN → `null` or a string; fixes L12) and shows `source`/`user` only on
   request.
10. ☑ `a79be97` Docs: rewrite `docs/schema.md` and `docs/usage.md`, update README examples.

**Requirements carried over from the review** (each needs a test in the new model):

| from | requirement |
|---|---|
| H4 | metadata equality compares values |
| H5, H6, L8 | copying never shares nested objects; a `Meta` passed in is copied, never aliased; input lists are copied |
| M7, L7 | a failed edit or merge leaves the previous state untouched (transactional) |
| H7, H8 | replacing a sub-part (e.g. `spaces`) is validated against the array shape; derived counts are never stale |
| M8 | read-only containers can't be mutated (`|=`, item assignment) |
| M9 | numpy scalars are accepted and converted; `bool` is not accepted as a number |
| M10 | user dicts are never unwrapped or reinterpreted |
| M11 | validate: matrix shape and last row, finite values, positive scale, axis-name references, box dimensions, min ≤ max |
| L9 | unknown keys are preserved, not rejected |
| L10 | unserialisable values give a clear `TypeError` |
| M12 rest | the compression settings recorded in `storage` can be turned back into blosc2 parameters (stage 2 could not make `blosc2.CParams(**meta.blosc2.cparams)` work: CParams rewrites the read-only filter list in place) |

**Agreed change to the invariant test (user decision 2026-11-06):** `tests/test_layout.py` reads
`arr.meta.blosc2.{cparams,chunk_size,block_size}`. In stage 3, change **only** these accessors to the new
`storage` section, and check the 4 compression threads on the written store's cparams instead of the
metadata (spec §8 persists no thread counts). All layout, codec, level, filter and read-ahead assertions stay.

**Result (2026-11-06).** Suite: 159 passed, 3109 subtests. Modules: `metadata.py` (model), `metadata_io.py`
(3 keys, raw msgpack, lazy parts), `fingerprint.py`, `units.py`, `schemas/` (inside the package).
`meta.py` and `utils.py` are removed. Implementation decisions where spec §11 is only a sketch:

| topic | decision |
|---|---|
| model shape | core parts are frozen dataclasses (`Axis`, `SpaceAxis`, `FromArray`, `Space`, `Values`, `Storage`, `SourceEntry`); `Meta` replaces them via validated setters / `Meta.update(...)` (all-or-nothing). `profiles` and `user` are plain dicts (profiles validated on write and on `validate()`); `source` and `history` are read-only (`FrozenMap`, tuples) |
| `MLArray(...)` | keyword-only after `array`: `axes, spaces, values, profiles, user, source, spacing, origin, direction, space, meta, copy, copy_source` + layout args. Constructors (`asarray`, `zeros`, `create`, ...) take `meta=` (a `Meta` or its dict form) |
| default axes | `axis_0`, `axis_1`, ... of type `space` (neutral names; nothing implies a direction) |
| convenience geometry | `spacing`/`direction` per array space axis in array order (`direction[:, c]` = unit vector of array axis `c`), `origin` = centre of index 0; requires `space=` (preset `"patient"` = DICOM LPS, mm) or a space description. If a space of that name exists, values not given are kept. `set_geometry(...)` does the same after construction |
| removed v0 API | `affine`, `translation`, `scale`, `rotation`, `shear`, `spatial_ndim`, geometry setters, `axis_labels`, `is_seg`, `meta.blosc2`, `meta.source` as free dict (now `user`) |
| derived geometry | `spacing` = column norms (tuple), `origin` = translation (tuple), `direction` = normalised columns (ndarray; unit vectors also for shear), `letter_codes()` = points-to letters of the dominant space axis, `"?"` without `increases_toward`. Zero matrix columns are invalid |
| storage | `storage.compression` records codec/level by name and filters by name, only what was given or chosen; `["NOFILTER"]` = explicitly no filter. `Storage.to_cparams()` converts back (4 threads by default) |
| fingerprint | dependency `xxhash`. `data = "xxh3_128:<hex>"` of C-order bytes (streamed for blosc2 arrays), `geometry = "xxh3_64:<hex>"` of msgpack(axes, spaces) with floats rounded to 10 significant digits, `dtype` = numpy name. `mlarray.fingerprint.compute/matches`, `MLArray.fingerprint()` |
| copy rules | `copy=`: geometry, values, profiles, user; not history, `ml.statistics`/`ml.boxes`, source (unless `copy_source=True`), storage kept as layout hint. `*_like`: same and label semantics dropped |
| no-array instances | an `MLArray` without data is saved as an empty array of shape `(0,) * len(axes)` (a valid v1 file); it loads as that empty array |
| lazy parts | `source`/`user` stay undecoded until accessed; unchanged parts are skipped when rewriting their own file and copied as raw bytes otherwise; pickling an opened file re-reads them from the file |

**Done when:**
- the spec examples (§4–§8) round-trip through save/load;
- the schemas validate the files written;
- all carried-over requirements have tests;
- stage-1/2 tests still pass, except tests that assert v0 metadata behaviour. Rewrite or delete those and
  list them in the commit message;
- `tests/test_layout.py` and `tests/test_read_ahead.py` are unchanged and green.

---

## 4. Stage 4 — MedVol v2 and the CLI

Work in `~/projects/medvol` on `rework/v2`, following its `docs/design/spec.md`.

1. ☑ medvol `84d1626` Skeleton:
   - `Image`, `Report`, `Policy`;
   - plugin registry via entry points, with `sniff`/`read`/`write`, capabilities and field tables (spec
     §3–§5).
2. ☑ medvol `84d1626` `.mla` plugin (via mlarray).
3. ☑ medvol `cc4a9a4` Conformance phantom and test harness (spec §9). Port the generators from
   `~/projects/image-orientation-kb/experiments/` (e.g. `05_python_libraries/synth.py`,
   `02_dicom/_synth.py`, `verify_dicom_real/_data.py`).
4. ☑ NIfTI `b54a8c5`, MGH/MGZ `0b9960a`, MetaImage `c2d9459`, NRRD `47bd5b5` (medvol); conformance matrix `a983099`. Each comes with the round-trip suite and the
   field-ownership regeneration test.
5. ☑ DICOM read phase 1 `54c4403`, phases 2–3 `d5a251a`; write: re-emit `b579767` (multi-frame re-emit and SEG in `d5a251a`).
6. ☑ mlarray `d1a626f` `0626299` mlarray CLI (`mlarray/cli.py`): `mlarray_convert` delegates to `medvol.convert`. Rewrite
   `tests/test_cli.py` against real MedVol with small nibabel/pynrrd fixtures (fixes M25). CI installs the
   extras.
7. ☑ medvol `3677dbc` (H10, H11, M14, M17, L11) + `test_nrrd.py` (H12, M15, M16, M18), mlarray `0626299` (CLI nits), `d1a626f` (CLI docs): Close the deferred review findings by verifying them against the new code: H10, H11, H12, M14–M18, L11,
   the CLI nits, and the stale CLI docs (`README.md`, `docs/cli.md`).
8. ☑ mlarray `8dd5775` Pin `medvol>=2` (`>=2.0.0.dev0`) in mlarray's `[all]` extra.

**Environment for stage 4:** conda env `medvol-v2` (Python 3.12, blosc2 4.14.1, nibabel 5.4.2, pynrrd 1.1.3,
pydicom 3.0.2, highdicom 0.28.2; `mlarray` and `medvol` installed editable from the rework branches; install
medvol with `--no-deps` because mlarray's version is not 1.0 yet). Tests:
`cd ~/projects/medvol && ~/miniforge3/envs/medvol-v2/bin/python -m pytest -q -p no:cacheprovider tests/`.

**Implementation notes (stage 4):**
- MedVol v1 (RAS canonicalisation, backends, snapping), its tests and examples are deleted; packaging is
  `pyproject.toml` with the plugin entry-point group `medvol.formats`.
- Phantom (`tests/phantom.py`): every voxel stores its own Fortran-order file index, so `world_error` checks the
  world position of every voxel against an oblique, mirrored LPS truth (KB ch05 geometry).
- Exactness rule used by all writers: geometry-bound source fields are reused bit-identically while the source
  fingerprint's geometry hash matches the current core; otherwise they are regenerated from the core (spec §4's
  regeneration test then holds to float32 rounding, e.g. NIfTI quaternions to 1e-6).
- NIfTI: header re-read as stored (nibabel moves scaling into the data proxy); header, extensions and data written
  by MedVol through nibabel's header objects and `array_to_file`; `.hdr/.img` need nibabel's *Pair* header
  classes (`is_single` is a class attribute).

**Done when:**
- every reader × writer pair passes the phantom test (< 1e-4 mm);
- same-format round trips meet the accuracy standard (MLArray spec §9);
- the deferred findings are closed with tests.

---

## 5. After the stages

The follow-up work (consumers, MedVol release, CD notes) is tracked in `docs/design/followups.md`.

- SlicerMLArray and napari-mlarray read the v1 core. napari should use the axis-order recipe from KB ch06
  §6.3 for correct on-screen orientation.
- Pyramids (spec §4.6, storage still to design) and lazy or streaming IO are later versions.
- Non-medical format plugins (OME-TIFF, FITS, GeoTIFF) are later; the model already covers them (spec
  §5.3).

---

## 6. Log: decisions, deviations, questions for the user

Append entries; never delete them.

| date | stage | entry | status |
|---|---|---|---|
| 2026-11-06 | — | plan written; specs agreed with the user (see spec decision logs) | — |
| 2026-11-06 | 1 | `012c896` ported with its `medvol>=0.0.22` pin: the current CLI still needs it; stage 4 step 8 replaces it | — |
| 2026-11-06 | 1 | stage 1 done, 127 passed / 3056 subtests | — |
| 2026-11-06 | 2a | Landed bug fixes that change observable behaviour (listed for review; revert on request): M4 (`save()` leaves a NumPy-backed instance NumPy-backed, `patch_size` kept), M5 (empty arrays report their shape instead of `None`; zero-length axes save, Blosc2 picks their layout and no patch size is recorded), nits (`dtype` optional for `empty`/`zeros`; arrays with only non-spatial axes no longer raise with the implied default patch size, Blosc2 picks the layout) | landed |
| 2026-11-06 | 2a | **Question L4:** `decompress()` clears `meta.blosc2`, documented and asserted by `tests/test_compress_decompress.py::test_decompress_in_place_clears_blosc2_and_detaches`. The plan's rule "keep the layout record in both" changes that behaviour and needs that test changed. | asked |
| 2026-11-06 | 2a | **Question M12:** `tests/test_layout.py:193` (invariant) asserts `meta.blosc2.cparams["nthreads"] == 4` after save, so "persist no thread counts" (plan, spec §8) cannot land in stage 2 without editing it. Proposal for stage 2: drop only machine-derived fields (`typesize`, `blocksize`, and `nthreads` of `blosc2.CParams` objects, which blosc2 fills with the core count); keep dict input as given; fix `blosc2.CParams(**meta.blosc2.cparams)` failing on the frozen filter list. | asked |
| 2026-11-06 | 3 | **Question (spec vs invariant):** `tests/test_layout.py` reads `arr.meta.blosc2.{cparams,chunk_size,block_size}`. Spec v1 replaces `meta.blosc2` with `storage` without thread counts, so the test cannot stay unchanged through stage 3. Proposal: in stage 3 change only its metadata accessors (`meta.blosc2.*` → `meta.storage.*`) and check the 4 compression threads on the written store's cparams instead of the metadata; all layout, codec, level, filter and read-ahead assertions stay. | asked |
| 2026-11-06 | 2b | Restructure done with the adjusted layout above; benchmark shows no regression | — |
| 2026-11-06 | 2a | User decisions: L4 keep the layout record in both (`b94ae9b`); M12 drop machine fields only (`549e5b0`); stage 3 changes only `test_layout.py`'s metadata accessors; keep the landed M4/M5/nit fixes | answered |
| 2026-11-06 | 2a | M12 sub-item `blosc2.CParams(**meta.blosc2.cparams)` not fixed in stage 2: blosc2 rewrites the filter list in place, which the frozen metadata forbids, and making the stored values enums breaks `is_serializable`. Moved to the stage-3 requirements (§3) | deferred |
| 2026-11-06 | 2 | stage 2 done: 147 passed, 3074 subtests; invariant tests unchanged | — |
| 2026-11-06 | 3 | Implementation decisions recorded in §3 "Result" (API shape, convenience geometry, copy rules, storage record, fingerprint encoding) | for review |
| 2026-11-06 | 3 | **Question (spec text):** the schemas live in `mlarray/schemas/` (installed with the package; `units.py` reads the unit list at runtime) instead of a top-level `schemas/` as spec §4.5/§10 say. Proposal: change the spec's paths | answered: spec changed |
| 2026-11-06 | 3 | **Question (spec gap):** the spec has no metadata-only files (v0 had them). Implemented: an instance without data is saved as an empty array `(0,) * len(axes)`. Proposal: add one sentence to spec §3 | answered: spec changed |
| 2026-11-06 | 3 | **Question (spec text):** spec §1 says mlarray depends on numpy and blosc2 only; v1 adds `msgpack` (already a blosc2 dependency) and `xxhash` (fingerprint, as the plan allowed). Proposal: say "numpy, blosc2, msgpack and xxhash" | answered: spec changed |
| 2026-11-06 | 3 | `mlarray_convert` raises until stage 4 (it was built on MedVol v1 and the v0 metadata); its fake-MedVol tests were removed | — |
| 2026-11-06 | 3 | blosc2 4.14.1: `blosc2.open(p).schunk.vlmeta.get_vlmeta(k)` on a temporary array segfaults (array freed while vlmeta is used). The library always keeps the store referenced; tests use helpers that do the same | note |
| 2026-11-06 | 3 | stage 3 done: 159 passed, 3109 subtests; `test_read_ahead.py` unchanged, `test_layout.py` changed only in its metadata accessors (agreed) | — |
| 2026-11-06 | 4 | stage 4 done. medvol: 127 passed, 213 subtests (reader x writer matrix over nifti, nrrd, metaimage, mgh, dicom, mla); mlarray: 163 passed in env `medvol-v2` (161 + 2 skipped without MedVol 2). NRRD, MetaImage, MGH and DICOM phases 2-3 were implemented in parallel and reviewed/integrated | — |
| 2026-11-06 | 4 | **Open (not implemented):** policy `reorient` (accepted, not applied); policy `vector_frame` and applying the NRRD `measurement frame` to vector components (spec 7.2 says "applied"; currently kept verbatim, values unchanged); `resample` options of `dicom_tilt` / `dicom_irregular_spacing` (raise NotImplementedError); DICOM phases 4 (RTSTRUCT) and 5 (mosaic); per-slice/per-frame differing DICOM rescale (refused); regenerating geometry when re-emitting enhanced multi-frame DICOM (refused) | answered: implemented below |
| 2026-11-06 | 4 | **Question (spec):** §3.3 says a NIfTI `descrip` written to NRRD is "kept as a key-value pair", §6/M18 forbid copying fields across formats. Implemented: nothing mapped. Proposal: drop the §3.3 example or define explicit cross-format mappings (e.g. descrip <-> NRRD content) | answered: example dropped, spec changed (medvol `b99327a`) |
| 2026-11-06 | 4 | Decisions for review: NRRD `space dimension` without `space` -> non-patient space `nrrd` (ITK assumes LPS); `scanner-xyz`/`3D-*-handed` read as LPS (ITK/Slicer) with a warning; NIfTI time series -> NRRD `left-posterior-superior-time`; MGH shear stored exactly (Mdc non-orthonormal, KB 04 4.4); MetaImage shear written as closest orthonormal direction (reported); MGH goodRASFlag < 0 treated like 0 (only 0 verified in KB); fractional SEG -> values.kind probability with scale 1/MaximumFractionalValue; layout='default' changes only the encoding (all plugins) | for review |
| 2026-11-06 | 4 | CI note: mlarray CI cannot install MedVol 2 until it is released; its CLI conversion tests skip without it | note |
| 2026-11-06 | 4 | User: "go with your recommendations, implement what you can". Done in medvol: `reorient` (`85c419f`), `vector_frame` incl. NRRD measurement frame (`c1ef2c1`), `dicom_rescale` policy + apply (`703cde9`, `6fef8a0`), DICOM tilt/irregular `resample` (`6fef8a0`), enhanced multi-frame geometry/rescale rewrite (`6fef8a0`), RTSTRUCT reading (`6fef8a0`, `medvol.read(..., reference=)` `c009a56`), foreign source headers reported on write (`022cea9`), spec decisions recorded (`b99327a`). medvol: 170 passed, 250 subtests | done |
| 2026-11-06 | 4 | **Still open:** Siemens mosaic (needs dcm2niix, not installed); writing RTSTRUCT (out of scope); resampling SEG label maps (refused); RTSTRUCT contour-plane tolerance is a fixed 0.01 mm; MetaImage/MGH writers do not report the vector frame; rotated NRRD measurement frames are applied on read, so they do not round-trip byte-exactly (reported); DICOM DS values whose original text is longer than 16 characters cannot be reproduced | open |
