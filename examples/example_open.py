import numpy as np
import os
from pathlib import Path
from mlarray import MLArray, Meta
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
    image = MLArray(spacing=spacing, origin=origin, direction=direction, meta=Meta(image=image_meta, bbox=bboxes))
    image.open(filepath, shape=array.shape, dtype=array.dtype, mmap='w+')
    print("Saving image...")
    image[...] = array
    image.meta.is_seg = True
    image.close()

    print("Loading image...")
    image = MLArray()
    image.open(filepath)
    print(json.dumps(image.meta.to_dict(), indent=2, sort_keys=True))
    print("Image mean value: ", np.mean(image.to_numpy()))
    print("Some array data: \n", image[:2, :2, 0])

    if Path(filepath).is_file():
        os.remove(filepath)