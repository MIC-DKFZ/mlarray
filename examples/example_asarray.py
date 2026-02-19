import gc
import os
import numpy as np
import time
from mlarray import MLArray


def to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def get_process_rss_bytes():
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    # Linux fallback without extra dependency.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass

    return None


def get_process_peak_rss_bytes():
    # Linux fallback for peak resident set size.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass
    return None


def format_bytes_mib(num_bytes):
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes} ({to_mib(num_bytes):.2f} MiB)"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rss_start = get_process_rss_bytes()

    # Build a sparse array with mostly zeros.
    shape = (512, 512, 512)
    density = 0.01  # 1% non-zero values
    total = int(np.prod(shape))
    nnz = int(total * density)

    array = np.zeros(shape, dtype=np.float32)
    non_zero_indices = rng.choice(total, size=nnz, replace=False)
    array.flat[non_zero_indices] = rng.random(nnz, dtype=np.float32)
    rss_after_numpy = get_process_rss_bytes()

    # NumPy in-memory payload size.
    numpy_bytes = array.nbytes

    # Convert to in-memory compressed MLArray.
    start_time = time.time()
    image = MLArray.asarray(array)
    print("MLArray conversion duration: ", time.time() - start_time)
    rss_after_compressed = get_process_rss_bytes()

    # Compressed payload size stored by Blosc2 in RAM.
    compressed_bytes = image._store.schunk.cbytes
    uncompressed_bytes = image._store.schunk.nbytes

    # Verify data integrity.
    equal = bool(np.allclose(array, image.to_numpy()))

    saved_bytes = numpy_bytes - compressed_bytes
    saved_pct = 100.0 * saved_bytes / numpy_bytes
    ratio = numpy_bytes / compressed_bytes

    # Optional: drop the original NumPy array and force GC, then re-check RSS.
    del array
    gc.collect()
    rss_after_drop_numpy = get_process_rss_bytes()
    peak_rss = get_process_peak_rss_bytes()

    rss_saved_bytes = None
    rss_saved_pct = None
    rss_saved_workload_bytes = None
    rss_saved_workload_pct = None
    if rss_after_numpy is not None and rss_after_drop_numpy is not None and rss_after_numpy > 0:
        rss_saved_bytes = rss_after_numpy - rss_after_drop_numpy
        rss_saved_pct = 100.0 * rss_saved_bytes / rss_after_numpy
    if (
        rss_start is not None
        and rss_after_numpy is not None
        and rss_after_drop_numpy is not None
        and (rss_after_numpy - rss_start) > 0
    ):
        rss_saved_workload_bytes = (rss_after_numpy - rss_start) - (rss_after_drop_numpy - rss_start)
        rss_saved_workload_pct = 100.0 * rss_saved_workload_bytes / (rss_after_numpy - rss_start)

    rss_numpy_consumption = None
    rss_mlarray_compressed_consumption = None
    rss_compression_ratio = None
    if rss_start is not None and rss_after_numpy is not None:
        rss_numpy_consumption = rss_after_numpy - rss_start
    if rss_start is not None and rss_after_drop_numpy is not None:
        rss_mlarray_compressed_consumption = rss_after_drop_numpy - rss_start
    if (
        rss_numpy_consumption is not None
        and rss_mlarray_compressed_consumption is not None
        and rss_mlarray_compressed_consumption > 0
    ):
        rss_compression_ratio = rss_numpy_consumption / rss_mlarray_compressed_consumption

    print("Sparse array compression demo (in-memory)")
    print(f"shape:                {shape}")
    print(f"density (non-zero):   {density:.2%}")
    print(f"numpy bytes:          {numpy_bytes} ({to_mib(numpy_bytes):.2f} MiB)")
    print(f"mlarray cbytes:       {compressed_bytes} ({to_mib(compressed_bytes):.2f} MiB)")
    print(f"mlarray nbytes:       {uncompressed_bytes} ({to_mib(uncompressed_bytes):.2f} MiB)")
    print(f"compression ratio:    {ratio:.2f}x")
    print(f"memory saved:         {saved_bytes} ({to_mib(saved_bytes):.2f} MiB, {saved_pct:.2f}%)")
    print(f"roundtrip equal:      {equal}")
    print()
    print("Process RSS snapshots (real memory in RAM):")
    print(f"rss start:            {format_bytes_mib(rss_start)}")
    print(f"rss after numpy:      {format_bytes_mib(rss_after_numpy)}")
    print(f"rss after compressed: {format_bytes_mib(rss_after_compressed)}")
    print(f"rss after del numpy:  {format_bytes_mib(rss_after_drop_numpy)}")
    print(f"rss peak (VmHWM):     {format_bytes_mib(peak_rss)}")
    if rss_saved_bytes is not None and rss_saved_pct is not None:
        print(
            f"rss saved (raw):      {rss_saved_bytes} ({to_mib(rss_saved_bytes):.2f} MiB, {rss_saved_pct:.2f}%)"
        )
    else:
        print("rss saved (raw):      n/a")
    if rss_saved_workload_bytes is not None and rss_saved_workload_pct is not None:
        print(
            "rss saved (workload): "
            f"{rss_saved_workload_bytes} ({to_mib(rss_saved_workload_bytes):.2f} MiB, {rss_saved_workload_pct:.2f}%)"
        )
    else:
        print("rss saved (workload): n/a")
    print()
    print("RSS-derived memory consumption summary:")
    if rss_numpy_consumption is not None:
        print(
            f"rss numpy memory consumption:              {rss_numpy_consumption} ({to_mib(rss_numpy_consumption):.2f} MiB)"
        )
    else:
        print("rss numpy memory consumption:              n/a")
    if rss_mlarray_compressed_consumption is not None:
        print(
            "rss mlarray compressed memory consumption: "
            f"{rss_mlarray_compressed_consumption} ({to_mib(rss_mlarray_compressed_consumption):.2f} MiB)"
        )
    else:
        print("rss mlarray compressed memory consumption: n/a")
    if rss_compression_ratio is not None:
        print(f"rss compression ratio:                     {rss_compression_ratio:.2f}x")
    else:
        print("rss compression ratio:                     n/a")
