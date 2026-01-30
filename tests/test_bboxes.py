import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlarray import MLArray
from mlarray.meta import Meta, MetaBbox


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestMetaBbox(unittest.TestCase):
    def test_bbox_accepts_ints_and_floats(self):
        bbox = MetaBbox(bboxes=[[[0, 1.5], [2, 3], [4.0, 5]]])
        self.assertEqual(bbox.bboxes, [[[0, 1.5], [2, 3], [4.0, 5]]])

    def test_bbox_scores_and_labels(self):
        bbox = MetaBbox(
            bboxes=[[[0, 1], [2, 3]]],
            scores=[0.9],
            labels=["lesion"],
        )
        self.assertEqual(bbox.scores, [0.9])
        self.assertEqual(bbox.labels, ["lesion"])

    def test_bbox_scores_labels_length_mismatch(self):
        with self.assertRaises(ValueError):
            MetaBbox(bboxes=[[[0, 1], [2, 3]]], scores=[0.9, 0.8])
        with self.assertRaises(ValueError):
            MetaBbox(bboxes=[[[0, 1], [2, 3]]], labels=["a", "b"])

    def test_bbox_labels_type_validation(self):
        with self.assertRaises(TypeError):
            MetaBbox(bboxes=[[[0, 1], [2, 3]]], labels=[True])

    def test_bbox_casts_numpy_and_tuple_inputs(self):
        bboxes = np.array([[[0, 1], [2, 3]]], dtype=np.int64)
        bbox = MetaBbox(bboxes=bboxes, scores=(0.5,), labels=("a",))
        self.assertEqual(bbox.bboxes, [[[0, 1], [2, 3]]])
        self.assertEqual(bbox.scores, [0.5])
        self.assertEqual(bbox.labels, ["a"])

    def test_bbox_roundtrip_mlarray(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            meta = Meta(
                bbox=MetaBbox(
                    bboxes=[[[0, 1], [2, 3], [4, 5]]],
                    scores=[0.7],
                    labels=[1],
                )
            )
            image = MLArray(array, meta=meta)

            path = Path(tmpdir) / "bbox.mla"
            image.save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.meta.bbox.to_plain(), meta.bbox.to_plain())


if __name__ == "__main__":
    unittest.main()
