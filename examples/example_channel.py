import json
import tempfile
from pathlib import Path

import numpy as np

from mlarray import MLArray


if __name__ == '__main__':
    print("Creating array...")
    array = np.random.random((32, 64, 64, 3))
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "tmp.mla"

        print("Initializing image...")
        image = MLArray(
            array,
            axes=["z", "y", "x", {"name": "c", "type": "channel", "labels": ["T1", "T2", "FLAIR"]}],
            spacing=(2, 2.5, 4),  # for the space axes z, y, x
            space="patient",
            user={"note": "multi-channel MR"},
        )
        print("Saving image...")
        image.save(filepath)

        print("Loading image...")
        image = MLArray(filepath)
        print(json.dumps(image.meta.to_dict(), indent=2))
        print("Channel labels:", image.axes[3].labels)
        print("Image mean value: ", np.mean(image.to_numpy()))
        print("Some array data: \n", image[:2, :2, 0])
