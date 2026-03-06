import json
import os
from pathlib import Path

import numpy as np

from mlarray import MLArray, Meta


if __name__ == "__main__":
    print("Creating array with metadata...")
    rng = np.random.default_rng(7)
    array = rng.random((16, 32, 32), dtype=np.float32)
    filepath = "tmp_compress_decompress.mla"

    if Path(filepath).is_file():
        os.remove(filepath)

    image = MLArray(
        array,
        spacing=(1.0, 1.5, 2.0),
        origin=(10.0, 20.0, 30.0),
        direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        meta=Meta(source={"patient_id": "p-001", "study": "demo"}),
    )
    print("Initial backend:", image._backend)
    print("Initial blosc2 meta:", json.dumps(image.meta.blosc2.to_plain(include_none=True), indent=2))

    print("\nDecompressing in-place...")
    image.decompress()
    print("Backend after decompress:", image._backend)
    print("blosc2 meta after decompress:", json.dumps(image.meta.blosc2.to_plain(include_none=True), indent=2))
    print("Metadata still present:", image.meta.source.to_plain())
    print("Spacing still present:", image.spacing)

    print("\nCompressing in-place with manual chunk/block...")
    image.compress(
        patch_size=None,
        chunk_size=(1, 16, 16),
        block_size=(1, 8, 8),
        dparams={"nthreads": 1},
    )
    print("Backend after compress:", image._backend)
    print("blosc2 meta after compress:", json.dumps(image.meta.blosc2.to_plain(include_none=True), indent=2))
    print("Metadata still present:", image.meta.source.to_plain())

    print("\nSaving and reloading...")
    image.save(filepath)
    loaded = MLArray(filepath)
    print("Reloaded shape:", loaded.shape)
    print("Reloaded source meta:", loaded.meta.source.to_plain())
    print("Reloaded blosc2 meta:", json.dumps(loaded.meta.blosc2.to_plain(include_none=True), indent=2))
    print("Arrays equal:", np.allclose(loaded.to_numpy(), array))

    if Path(filepath).is_file():
        os.remove(filepath)
