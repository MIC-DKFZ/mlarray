import numpy as np
import os
from pathlib import Path
from mlarray import MLArray


if __name__ == "__main__":
    print("Creating input array...")
    array = np.random.random((16, 64, 64)).astype(np.float32)

    filepath_plain = "tmp_asarray_plain.mla"
    filepath_compressed = "tmp_asarray_compressed.mla"

    for filepath in (filepath_plain, filepath_compressed):
        if Path(filepath).is_file():
            os.remove(filepath)

    print("Converting with MLArray.asarray(memory_compressed=False)...")
    image_plain = MLArray.asarray(
        array,
        memory_compressed=False,
        patch_size=128,
    )
    image_plain.save(filepath_plain)

    print("Converting with MLArray.asarray(memory_compressed=True)...")
    image_compressed = MLArray.asarray(
        array,
        memory_compressed=True,
        patch_size=128,
    )
    image_compressed.save(filepath_compressed)

    print("Loading saved files...")
    reloaded_plain = MLArray(filepath_plain)
    reloaded_compressed = MLArray(filepath_compressed)

    print("Plain mean:      ", float(np.mean(reloaded_plain.to_numpy())))
    print("Compressed mean: ", float(np.mean(reloaded_compressed.to_numpy())))
    print(
        "Arrays equal:",
        bool(np.allclose(reloaded_plain.to_numpy(), reloaded_compressed.to_numpy())),
    )
    print("Some array data:\n", reloaded_compressed[:2, :2, :2])

    for filepath in (filepath_plain, filepath_compressed):
        if Path(filepath).is_file():
            os.remove(filepath)
