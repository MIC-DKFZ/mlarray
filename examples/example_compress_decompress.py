import json
import tempfile
from pathlib import Path

import numpy as np

from mlarray import MLArray


if __name__ == "__main__":
    print("Creating array with metadata...")
    rng = np.random.default_rng(7)
    array = rng.random((16, 32, 32), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "tmp_compress_decompress.mla"

        image = MLArray(
            array,
            spacing=(1.0, 1.5, 2.0),
            origin=(10.0, 20.0, 30.0),
            space="patient",
            user={"patient_id": "p-001", "study": "demo"},
        )
        print("Initial backend:", image._backend)
        print("Initial storage:", json.dumps(image.meta.storage.to_dict(), indent=2))

        print("\nDecompressing in-place...")
        image.decompress()
        print("Backend after decompress:", image._backend)
        print("Storage after decompress:", json.dumps(image.meta.storage.to_dict(), indent=2))
        print("Metadata still present:", image.user)
        print("Spacing still present:", image.spacing)

        print("\nCompressing in-place with manual chunk/block...")
        image.compress(
            patch_size=None,
            chunk_size=(1, 16, 16),
            block_size=(1, 8, 8),
            dparams={"nthreads": 1},
        )
        print("Backend after compress:", image._backend)
        print("Storage after compress:", json.dumps(image.meta.storage.to_dict(), indent=2))

        print("\nSaving and reloading...")
        image.save(filepath)
        loaded = MLArray(filepath)
        print("Reloaded shape:", loaded.shape)
        print("Reloaded user meta:", loaded.user)
        print("Reloaded storage:", json.dumps(loaded.meta.storage.to_dict(), indent=2))
        print("Arrays equal:", np.allclose(loaded.to_numpy(), array))
