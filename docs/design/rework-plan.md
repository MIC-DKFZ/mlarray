# MLArray / MedVol rework — process and handoff

This document lets any agent or developer continue the rework without prior context. Read it fully before
touching code. Keep it current: when a stage step is done, tick it and add the commit hash.

Last update: 2026-11-06 (stage 1 done; stage 2 in progress).

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
  green. They pin the storage defaults:
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

| ☐ | finding | what to do |
|---|---|---|
| ☐ | **M1** not picklable | `__getstate__`/`__setstate__`. Opened files store `(filepath, mode, mmap_mode, dparams)` and reopen; in-memory blosc2 stores serialise via `to_cframe()`; numpy-backed arrays pickle the array. Test with `multiprocessing` spawn and a torch-free DataLoader-style worker. |
| ☐ | **M4** `save()` on numpy-backed arrays converts the instance to blosc2 and loses `patch_size` | save from a temporary blosc2 store; the instance stays unchanged |
| ☐ | **M5** remainder: empty arrays and zero-length dimensions | stop using the `(0,)` placeholder as the "no array" marker; use an explicit flag. Layout: handle zero-length axes without calling the heuristic, or clamp |
| ☐ | **M12** machine-specific `cparams` persisted | persist only portable settings (codec, level, filters; spec §8 `storage`). Thread counts come from runtime defaults. Must keep `tests/test_layout.py`/`test_read_ahead.py` green |
| ☐ | **L4** `load(compressed=False)` keeps `meta.blosc2` while `decompress()` clears it | pick one rule (keep the layout record) and apply it in both |
| ☐ | Nits | consistent optional `dtype` for `empty`/`zeros`; document memory sharing of `asarray(compressed=False)`; arrays with only non-spatial axes work without `patch_size=None` (check if the new heuristic already solves this); fix docstrings ("ignored when patch_size is provided") |

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

1. ☐ JSON Schemas: `schemas/mlarray-1.0.json`, `schemas/profile-medical-1.0.json`,
   `schemas/profile-ml-1.0.json`, `schemas/units-1.0.json` (curated unit names + UCUM codes; spec §4.5).
2. ☐ Storage in `metadata_io.py` (spec §3):
   - three vlmeta keys `mlarray`, `mlarray.source`, `mlarray.user`;
   - write msgpack;
   - **read raw bytes and decode with plain msgpack** (never blosc2's vlmeta decoder);
   - lazy decoding of `source`/`user`;
   - reference: review commits `45c3ff4`, `35b613c` and `tests/test_metadata_storage.py`.
3. ☐ New model classes: `Axis`, `Space` (+ `from_array`), `Values`, `Profiles`, `SourceEntry`, `History`,
   `Storage`. Validation against the schemas. Unknown keys and unknown profiles are preserved (spec §4.1,
   §5).
4. ☐ Derived geometry API (spec §11):
   - `matrix()`, read-only `spacing`/`origin`/`direction`, `letter_codes()`;
   - convenience constructor (`spacing=, origin=, direction=, space=`) composed into a matrix with the
     documented column convention;
   - world geometry without a declared space is refused;
   - fixes the affine-derived property nits: return types, division by zero, shear for non-orthogonal
     matrices.
5. ☐ `values`: `kind`, `unit`/`unit_code`, `scale`/`offset`, `fill`, `labels`, `label_layout`. This replaces
   `is_seg`.
6. ☐ `ml` profile: `statistics`, `boxes` with an explicit `frame` (replaces `stats`, `bbox`), `patch_size`,
   `split`. `medical` profile skeleton.
7. ☐ `source`: read-only API, `index_map`, `fingerprint` (xxh3 data hash; add `xxhash` as a dependency, or use
   blosc2/numpy hashing if adequate; decide and record here).
8. ☐ Constructors, `copy=` and `*_like` follow the new rules:
   - geometry is copied;
   - `source` is **not** inherited unless `copy_source=True`;
   - statistics, boxes and label semantics are not copied by `*_like` (fixes M13).
9. ☐ `mlarray_header` prints valid JSON (NaN → `null` or a string; fixes L12) and shows `source`/`user` only on
   request.
10. ☐ Docs: rewrite `docs/schema.md` and `docs/usage.md`, update README examples.

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

1. ☐ Skeleton:
   - `Image`, `Report`, `Policy`;
   - plugin registry via entry points, with `sniff`/`read`/`write`, capabilities and field tables (spec
     §3–§5).
2. ☐ `.mla` plugin (via mlarray).
3. ☐ Conformance phantom and test harness (spec §9). Port the generators from
   `~/projects/image-orientation-kb/experiments/` (e.g. `05_python_libraries/synth.py`,
   `02_dicom/_synth.py`, `verify_dicom_real/_data.py`).
4. ☐ NIfTI plugin, then NRRD, MetaImage, MGH/MGZ. Each comes with the round-trip suite and the
   field-ownership regeneration test.
5. ☐ DICOM read, phases 1–3 (spec §7.5), then DICOM write (re-emit original series, SEG; spec §7.6).
6. ☐ mlarray CLI (`mlarray/cli.py`): `mlarray_convert` delegates to `medvol.convert`. Rewrite
   `tests/test_cli.py` against real MedVol with small nibabel/pynrrd fixtures (fixes M25). CI installs the
   extras.
7. ☐ Close the deferred review findings by verifying them against the new code: H10, H11, H12, M14–M18, L11,
   the CLI nits, and the stale CLI docs (`README.md`, `docs/cli.md`).
8. ☐ Pin `medvol>=2` in mlarray's `[all]` extra.

**Done when:**
- every reader × writer pair passes the phantom test (< 1e-4 mm);
- same-format round trips meet the accuracy standard (MLArray spec §9);
- the deferred findings are closed with tests.

---

## 5. After the stages

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
