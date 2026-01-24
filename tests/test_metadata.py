import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlarray import MLArray
from mlarray.meta import Meta, MetaBbox, MetaStatistics


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestMetadataStorage(unittest.TestCase):
    def test_metadata_roundtrip_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            meta_dict = {
                "patient_id": "p-001",
                "modality": "CT",
                "nested": {"a": 1, "b": [1, 2, 3]},
            }
            image = MLArray(array, meta=meta_dict)

            path = Path(tmpdir) / "meta-dict.mla"
            image.save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.meta.image, meta_dict)

    def test_metadata_roundtrip_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            stats = MetaStatistics(min=0.0, max=1.0, mean=0.5)
            bbox = MetaBbox(bboxes=[[[0, 3], [1, 5], [2, 7]]])
            meta = Meta(
                image={"patient_id": "p-002"},
                stats=stats,
                bbox=bbox,
                is_seg=True,
                extra={"pipeline": "v1"},
            )
            image = MLArray(array, meta=meta)

            path = Path(tmpdir) / "meta-class.mla"
            image.save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.meta.image, {"patient_id": "p-002"})
            self.assertTrue(loaded.meta.is_seg)
            self.assertEqual(loaded.meta.stats.to_dict(), stats.to_dict())
            self.assertEqual(loaded.meta.bbox.to_list(), bbox.to_list())
            self.assertEqual(loaded.meta.extra, {"pipeline": "v1"})

    def test_metadata_mmap_readonly_no_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "readonly.mla"
            image = MLArray(array, meta={"a": 1})
            image.save(path)

            opened = MLArray.open(path, mmap="r")
            opened.meta.image["a"] = 2
            opened.close()

            reloaded = MLArray(path)
            self.assertEqual(reloaded.meta.image["a"], 1)

    def test_metadata_mmap_readwrite_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "readwrite.mla"
            image = MLArray(array, meta={"a": 1})
            image.save(path)

            opened = MLArray.open(path, mmap="r+")
            opened.meta.image["a"] = 2
            opened.close()

            reloaded = MLArray(path)
            self.assertEqual(reloaded.meta.image["a"], 2)

    def test_metadata_mmap_copy_mode_no_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "copy.mla"
            image = MLArray(array, meta={"a": 1})
            image.save(path)

            opened = MLArray.open(path, mmap="c")
            opened.meta.image["a"] = 3
            opened.close()

            reloaded = MLArray(path)
            self.assertEqual(reloaded.meta.image["a"], 1)

    def test_metadata_open_create_write_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "create.mla"
            shape = (8, 16, 16)
            dtype = np.float32

            opened = MLArray.open(path, shape=shape, dtype=dtype, mmap="w+")
            opened.meta.extra["created"] = True
            opened.close()

            reloaded = MLArray(path)
            self.assertEqual(reloaded.meta.extra["created"], True)


if __name__ == "__main__":
    unittest.main()
