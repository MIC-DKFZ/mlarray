import json
import tempfile
from pathlib import Path

import numpy as np

from mlarray import MLArray


if __name__ == '__main__':
    print("Creating array...")
    array = np.random.random((32, 64, 64))
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "tmp.mla"

        print("Initializing image...")
        image = MLArray(
            array,
            axes=["z", "y", "x"],
            # Geometry per array axis (z, y, x) in the DICOM patient frame (LPS):
            # x runs toward the patient's left, y toward posterior, z toward superior.
            spacing=(2.5, 0.7, 0.7),
            origin=(-170.1, -201.3, -87.0),
            direction=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            space="patient",
            values={"kind": "measurement", "unit": "hounsfield_unit"},
            profiles={"ml": {"profile_version": "1.0", "split": "train",
                             "boxes": {"frame": "index", "axes": ["z", "y", "x"],
                                       "items": [{"min": [0, 0, 0], "max": [1, 1, 1], "label": 1}]}}},
            user={"note": "This is an image", "numbers": [1, 2, 3]},
        )
        print("Saving image...")
        image.save(filepath)

        print("Loading image...")
        image = MLArray(filepath)
        image = MLArray.load(filepath)  # Just testing both load methods
        print(json.dumps(image.meta.to_dict(include_lazy=True), indent=2))
        print("Spacing:", image.spacing, "Origin:", image.origin, "Letters:", image.letter_codes())
        print("Image mean value: ", np.mean(image.to_numpy()))
        print("Some array data: \n", image[:2, :2, 0])
