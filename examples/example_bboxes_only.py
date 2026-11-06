import json
import tempfile
from pathlib import Path

import numpy as np

from mlarray import MLArray


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "tmp_bboxes.mla"

        # Five example 3D boxes in index coordinates along the array axes z, y, x.
        boxes = [
            ([12, 40, 55], [28, 75, 90], "tumor", 0.97),
            ([48, 18, 10], [70, 36, 24], "lymph_node", 0.84),
            ([20, 80, 44], [33, 108, 73], "lesion", 0.76),
            ([72, 52, 88], [95, 81, 118], "organ_part", 0.91),
            ([5, 12, 100], [16, 30, 126], "cyst", 0.68),
        ]
        image = MLArray(
            np.zeros((100, 128, 128), dtype=np.uint8),
            axes=["z", "y", "x"],
            profiles={"ml": {"profile_version": "1.0", "boxes": {
                "frame": "index",
                "axes": ["z", "y", "x"],
                "items": [{"min": lo, "max": hi, "label": label, "score": score} for lo, hi, label, score in boxes],
            }}},
            user={"case_id": "case-001", "modality": "CT"},
        )
        image.save(filepath)

        loaded = MLArray(filepath)
        print("Saved MLArray with 3D boxes in the ml profile:")
        print(json.dumps(loaded.profiles["ml"], indent=2))
