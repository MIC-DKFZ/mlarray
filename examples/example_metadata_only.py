import numpy as np
import os
from pathlib import Path
from mlarray import MLArray, Meta, MetaBbox
import json


if __name__ == '__main__':
    print("Creating array...")
    array = np.random.random((32, 64, 64))
    spacing = np.array((2, 2.5, 4))
    origin = (1, 1, 1)
    direction = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    image_meta = {"tmp1": "This is an image", "tmp2": 5, "tmp3": {"test1": 16.4587, "test2": [1, 2, 3, 4, 5, 6]}}
    bboxes = [[[0, 1], [0, 1], [0, 1]]]
    filepath = "tmp.mla"

    if Path(filepath).is_file():
        os.remove(filepath)

    print("Initializing image...")
    image = MLArray(spacing=spacing, origin=origin, direction=direction, meta=Meta(original=image_meta, bbox=MetaBbox(bboxes)))
    print("Saving image...")
    image.save(filepath)

    print("Loading image...")
    image = MLArray(filepath)
    print(json.dumps(image.meta.to_mapping(), indent=2, sort_keys=True))

    if Path(filepath).is_file():
        os.remove(filepath)