import tempfile
import unittest
from pathlib import Path

import numpy as np

from med_blosc2 import MedBlosc2, MED_BLOSC2_DEFAULT_PATCH_SIZE
from med_blosc2.meta import Meta


def _make_array(shape=(16, 32, 32), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestUsage(unittest.TestCase):
    def test_default_usage_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            image = MedBlosc2(array)

            path = Path(tmpdir) / "sample.mb2nd"
            image.save(path)

            loaded = MedBlosc2(path)
            self.assertEqual(loaded.shape, array.shape)

    def test_mmap_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "sample.mb2nd"
            MedBlosc2(array).save(path)

            loaded = MedBlosc2(array)
            loaded.open(path, mmap_mode="r")
            self.assertFalse(isinstance(loaded.array, np.ndarray))

    def test_loading_and_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            src = Path(tmpdir) / "sample.mb2nd"
            dst = Path(tmpdir) / "copy.mb2nd"

            MedBlosc2(array).save(src)
            MedBlosc2(src).save(dst)

            self.assertTrue(dst.exists())

    def test_metadata_inspection_and_manipulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            image = MedBlosc2(
                array,
                spacing=(1.0, 1.0, 1.5),
                origin=(10.0, 10.0, 30.0),
                direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                meta=Meta(image={"patient_id": "123", "modality": "CT"}, is_seg=False),
            )
            image.spacing[1] = 5.3
            image.meta.image["study_id"] = "study-001"

            path = Path(tmpdir) / "with-metadata.mb2nd"
            image.save(path)

            loaded = MedBlosc2(path)
            self.assertEqual(loaded.spacing, [1.0, 5.3, 1.5])
            self.assertEqual(loaded.origin, [10.0, 10.0, 30.0])
            self.assertEqual(loaded.meta.image["study_id"], "study-001")

    def test_copy_metadata_with_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            base = MedBlosc2(
                array,
                spacing=(1.0, 1.0, 1.0),
                origin=(1.0, 2.0, 3.0),
                direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                meta=Meta(image={"source": "base"}, is_seg=True),
            )
            base_path = Path(tmpdir) / "base.mb2nd"
            base.save(base_path)

            base_loaded = MedBlosc2(base_path)
            array2 = _make_array(seed=1)
            image = MedBlosc2(array2, spacing=(0.8, 0.8, 1.0), copy=base_loaded)

            self.assertEqual(image.spacing, [0.8, 0.8, 1.0])
            self.assertEqual(image.origin, base_loaded.origin)
            self.assertEqual(image.direction, base_loaded.direction)
            self.assertEqual(image.meta.is_seg, base_loaded.meta.is_seg)
            self.assertEqual(image.meta.image, base_loaded.meta.image)

    def test_patch_size_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "default-patch.mb2nd"
            MedBlosc2(array).save(path)

            loaded = MedBlosc2(path)
            self.assertEqual(
                loaded.meta._blosc2.patch_size,
                [MED_BLOSC2_DEFAULT_PATCH_SIZE,] * 3,
            )

    def test_patch_size_isotropic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "patch-64.mb2nd"
            MedBlosc2(array).save(path, patch_size=64)

            loaded = MedBlosc2(path)
            self.assertEqual(loaded.meta._blosc2.patch_size, [64, 64, 64])

    def test_patch_size_non_isotropic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "patch-non-iso.mb2nd"
            MedBlosc2(array).save(path, patch_size=[16, 24, 32])

            loaded = MedBlosc2(path)
            self.assertEqual(loaded.meta._blosc2.patch_size, [16, 24, 32])

    def test_manual_chunk_and_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "manual-chunk-block.mb2nd"
            MedBlosc2(array).save(
                path,
                patch_size=None,
                chunk_size=(1, 16, 16),
                block_size=(1, 8, 8),
            )

            loaded = MedBlosc2(path)
            self.assertEqual(loaded.meta._blosc2.chunk_size, [1, 16, 16])
            self.assertEqual(loaded.meta._blosc2.block_size, [1, 8, 8])

    def test_b2nd_metadata_ignored_on_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "plain.b2nd"
            image = MedBlosc2(
                array,
                spacing=(1.0, 2.0, 3.0),
                origin=(1.0, 2.0, 3.0),
                direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                meta=Meta(image={"tag": "value"}, is_seg=True),
            )
            image.save(path)

            loaded = MedBlosc2(path)
            self.assertIsNone(loaded.spacing)
            self.assertIsNone(loaded.origin)
            self.assertIsNone(loaded.direction)


if __name__ == "__main__":
    unittest.main()
