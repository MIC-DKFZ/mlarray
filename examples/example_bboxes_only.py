import json
import os
from pathlib import Path

from mlarray import MLArray, Meta, MetaBbox


if __name__ == "__main__":
    filepath = "tmp_metadata_bboxes_only.mla"

    if Path(filepath).is_file():
        os.remove(filepath)

    # Five example 3D bounding boxes in [z, y, x] order.
    bboxes = [
        [[12, 28], [40, 75], [55, 90]],
        [[48, 70], [18, 36], [10, 24]],
        [[20, 33], [80, 108], [44, 73]],
        [[72, 95], [52, 81], [88, 118]],
        [[5, 16], [12, 30], [100, 126]],
    ]
    scores = [0.97, 0.84, 0.76, 0.91, 0.68]
    labels = ["tumor", "lymph_node", "lesion", "organ_part", "cyst"]

    meta = Meta(
        source={"case_id": "case-001", "modality": "CT"},
        bbox=MetaBbox(bboxes=bboxes, scores=scores, labels=labels)
    )

    # No array data: this MLArray stores metadata only.
    image = MLArray(meta=meta)
    image.save(filepath)

    loaded = MLArray(filepath)
    print("Saved metadata-only MLArray with 3D bboxes:")
    print(json.dumps(loaded.meta.to_mapping(), indent=2, sort_keys=True))
