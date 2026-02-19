from pathlib import Path

import numpy as np

from mlarray import MLArray


def main():
    shape = (2, 4, 4)

    a_empty = MLArray.empty(
        shape=shape,
        dtype=np.float32,
        patch_size=None,
        chunk_size=(1, 4, 4),
        block_size=(1, 2, 2),
    )
    a_empty[...] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    a_zeros = MLArray.zeros(
        shape=shape,
        dtype=np.float32,
        patch_size=None,
        chunk_size=(1, 4, 4),
        block_size=(1, 2, 2),
    )
    a_ones = MLArray.ones(
        shape=shape,
        dtype=np.float32,
        patch_size=None,
        chunk_size=(1, 4, 4),
        block_size=(1, 2, 2),
    )
    a_full = MLArray.full(
        shape=shape,
        fill_value=5,
        dtype=np.float32,
        patch_size=None,
        chunk_size=(1, 4, 4),
        block_size=(1, 2, 2),
    )

    a_arange = MLArray.arange(0, 16, 1, shape=(4, 4), patch_size=None)
    a_linspace = MLArray.linspace(
        0.0, 1.0, num=8, shape=(2, 4), endpoint=False, patch_size=None
    )

    a_empty_like = MLArray.empty_like(
        a_full, patch_size=None, chunk_size=(1, 4, 4), block_size=(1, 2, 2)
    )
    a_empty_like[...] = 11
    a_zeros_like = MLArray.zeros_like(
        a_full, patch_size=None, chunk_size=(1, 4, 4), block_size=(1, 2, 2)
    )
    a_ones_like = MLArray.ones_like(
        a_full, patch_size=None, chunk_size=(1, 4, 4), block_size=(1, 2, 2)
    )
    a_full_like = MLArray.full_like(
        a_full,
        fill_value=9,
        patch_size=None,
        chunk_size=(1, 4, 4),
        block_size=(1, 2, 2),
    )

    print("empty sum:", float(a_empty[...].sum()))
    print("zeros sum:", float(a_zeros[...].sum()))
    print("ones sum:", float(a_ones[...].sum()))
    print("full sum:", float(a_full[...].sum()))
    print("arange:", a_arange[...])
    print("linspace:", a_linspace[...])
    print("empty_like unique:", np.unique(a_empty_like[...]))
    print("zeros_like sum:", float(a_zeros_like[...].sum()))
    print("ones_like sum:", float(a_ones_like[...].sum()))
    print("full_like sum:", float(a_full_like[...].sum()))

    out_path = Path("example_in_memory_constructors_output.mla")
    a_full_like.save(out_path)
    loaded = MLArray(out_path)
    print("saved shape:", loaded.shape, "dtype:", loaded.dtype)


if __name__ == "__main__":
    main()
