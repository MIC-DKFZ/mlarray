# Benchmark Scripts

- `benchmark_formats.py`: compares MLArray with other image file formats.

The benchmark behind MLArray's storage defaults (layout, codec, read mode and
read-ahead, measured on a compute cluster with nnU-Net-style loading) lives on
the `feat/bench-suite` branch under `bench/`; `docs/optimization.md` summarizes
its results.
