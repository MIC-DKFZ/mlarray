import json
import tempfile
from pathlib import Path

import numpy as np

from mlarray import MLArray


if __name__ == '__main__':
    print("Creating array...")
    array = np.random.random((2, 6, 4, 4, 3, 2))
    axes = [
        {"name": "t", "type": "time", "start": 0.0, "step": 1.5, "unit": "second"},
        "z",
        "y",
        "x",
        {"name": "c", "type": "channel", "labels": ["red", "green", "blue"]},
        {"name": "echo", "type": "other", "coordinates": [10.0, 20.0], "unit": "millisecond"},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "tmp.mla"

        print("Initializing image...")
        image = MLArray(array, axes=axes, spacing=(2, 2.5, 4), space="patient", patch_size=None)
        print("Saving image...")
        image.save(filepath)

        print("Loading image...")
        image = MLArray(filepath)
        print(json.dumps(image.meta.to_dict(), indent=2))
        print("Image mean value: ", np.mean(image.to_numpy()))
        print("Some array data: \n", image[:2, :2, 0])
