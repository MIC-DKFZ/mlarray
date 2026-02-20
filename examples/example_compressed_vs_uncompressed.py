from pathlib import Path

import numpy as np

from mlarray import MLArray


def main():
    array = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)

    compressed_img = MLArray.asarray(
        array,
        compressed=True,
        patch_size=None,
        chunk_size=(1, 4, 4),
        block_size=(1, 2, 2),
    )
    uncompressed_img = MLArray.asarray(array, compressed=False)

    compressed_path = Path("example_compressed_output.mla")
    uncompressed_path = Path("example_uncompressed_output.mla")

    print("compressed in-memory backend (before save):", type(compressed_img._store))
    print(
        "uncompressed in-memory backend (before save):",
        type(uncompressed_img._store),
    )

    compressed_img.save(compressed_path)
    uncompressed_img.save(uncompressed_path)

    print("compressed in-memory backend (after save):", type(compressed_img._store))
    print(
        "uncompressed in-memory backend (after save):",
        type(uncompressed_img._store),
    )
    print("saved compressed file:", compressed_path)
    print("saved uncompressed->compressed file:", uncompressed_path)


if __name__ == "__main__":
    main()
