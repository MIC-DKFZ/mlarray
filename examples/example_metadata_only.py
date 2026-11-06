import json
import tempfile
from pathlib import Path

from mlarray import MLArray


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "tmp.mla"

        print("Initializing metadata without array data...")
        image = MLArray(axes=["z", "y", "x"], spacing=(2, 2.5, 4), origin=(1, 1, 1), space="patient",
                        user={"note": "metadata only"})
        print("Saving image (stored as an empty array)...")
        image.save(filepath)

        print("Loading image...")
        image = MLArray(filepath)
        print("Shape:", image.shape)
        print(json.dumps(image.meta.to_dict(include_lazy=True), indent=2))
