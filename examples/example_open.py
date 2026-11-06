import json
import tempfile
from pathlib import Path

import numpy as np

from mlarray import MLArray


if __name__ == '__main__':
    print("Creating array...")
    array = np.random.randint(0, 3, size=(32, 64, 64)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "tmp.mla"

        print("Creating file and writing into it...")
        image = MLArray.create(filepath, array.shape, array.dtype, meta={"axes": ["z", "y", "x"]})
        image[...] = array
        image.set_geometry(spacing=(2, 2.5, 4), origin=(1, 1, 1), space="patient")
        image.values = {"kind": "label", "labels": {"0": {"name": "background"}, "1": {"name": "liver"}, "2": {"name": "tumor"}}}
        image.user["tmp"] = {"test1": 16.4587, "test2": [1, 2, 3, 4, 5, 6]}
        image.close()

        print("Opening image...")
        image = MLArray.open(filepath)
        print(json.dumps(image.meta.to_dict(include_lazy=True), indent=2))
        print("Labels:", dict(image.values.labels))
        print("Some array data: \n", image[:2, :2, 0])
