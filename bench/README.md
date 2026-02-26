# Benchmark Scripts

This folder contains benchmarking scripts for MLArray IO/layout experiments.

## `bench_io_blosc2_layouts.py`

Benchmarks IO throughput across:

- layout method(s) based on `comp_blosc2_params` (currently baseline copy only)
- image size tiers (`small`, `medium`, `large`, `very_large`)
- 2D / 3D / 4D-total array cases with spatial and optional non-spatial axis
- multiple patch sizes (2D and 3D patch vectors)
- `MLArray.open(...)` mode/mmap combinations
- operations:
  - `read_full`
  - `read_patch_random`
  - `write_patch_random`
- warm and cold cache runs

Outputs are printed to console and written to:

- `bench/results/bench_io_blosc2_layouts.csv`
- `bench/results/bench_io_blosc2_layouts.json`

### Example

```bash
python bench/bench_io_blosc2_layouts.py \
  --tiers small medium \
  --runs 3 \
  --cache-mode both \
  --nthreads 1
```

If you hit native segfaults in Blosc2 during long runs, isolate each measured run
in a subprocess (slower, but robust):

```bash
python bench/bench_io_blosc2_layouts.py \
  --tiers small medium \
  --runs 3 \
  --cache-mode both \
  --nthreads 1 \
  --isolate-runs
```

### Cold cache note (Linux)

For cold-cache read measurements, the script drops Linux page cache **after the dataset has been created on disk and immediately before measured open/read runs**.

This requires root:

- run as root, or
- run via `sudo`

If cache dropping fails, those runs are recorded with error status in results.
