import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlarray import MLArray


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestCompressDecompress(unittest.TestCase):
    def test_compress_in_place_switches_backend_and_applies_layout(self):
        array = _make_array(seed=1)
        image = MLArray(array, compressed=False, user={"case": "compress"})
        self.assertTrue(isinstance(image._store, np.ndarray))

        image.compress(
            patch_size=None,
            chunk_size=(1, 8, 8),
            block_size=(1, 4, 4),
        )

        self.assertFalse(isinstance(image._store, np.ndarray))
        self.assertTrue(np.allclose(image.to_numpy(), array))
        self.assertEqual(image.user, {"case": "compress"})
        self.assertEqual(image.meta.storage.chunk_size, (1, 8, 8))
        self.assertEqual(image.meta.storage.block_size, (1, 4, 4))
        self.assertIsNone(image.filepath)
        self.assertIsNone(image.mode)
        self.assertIsNone(image.mmap_mode)

    def test_decompress_in_place_keeps_blosc2_and_detaches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=2)
            path = Path(tmpdir) / "switch-source.mla"
            MLArray(array, user={"case": "decompress"}).save(path)

            image = MLArray.open(path, mode="a", mmap_mode="r+")
            self.assertEqual(image.mode, "a")
            self.assertEqual(image.mmap_mode, "r+")
            self.assertEqual(image.filepath, str(path))
            self.assertIsNotNone(image.meta.storage.chunk_size)
            blosc2_before = image.meta.storage.to_dict()

            image.decompress()

            self.assertTrue(isinstance(image._store, np.ndarray))
            self.assertTrue(np.allclose(image.to_numpy(), array))
            self.assertEqual(image.user, {"case": "decompress"})
            self.assertEqual(image.meta.storage.to_dict(), blosc2_before)
            self.assertIsNone(image.filepath)
            self.assertIsNone(image.mode)
            self.assertIsNone(image.mmap_mode)

            # Decompression is detached until explicitly saved.
            reloaded = MLArray(path)
            self.assertIsNotNone(reloaded.meta.storage.chunk_size)

    def test_roundtrip_preserves_data_and_metadata(self):
        array = _make_array(seed=3)
        image = MLArray(
            array,
            spacing=(0.8, 0.9, 1.2), space="patient",
            origin=(1.0, 2.0, 3.0),
            user={"patient_id": "p-123"},
        )

        image.decompress()
        self.assertTrue(np.allclose(image.to_numpy(), array))
        self.assertEqual(image.spacing, (0.8, 0.9, 1.2))
        self.assertEqual(image.origin, (1.0, 2.0, 3.0))
        self.assertEqual(image.user["patient_id"], "p-123")

        image.compress(patch_size=64)
        self.assertTrue(np.allclose(image.to_numpy(), array))
        self.assertEqual(image.spacing, (0.8, 0.9, 1.2))
        self.assertEqual(image.origin, (1.0, 2.0, 3.0))
        self.assertEqual(image.user["patient_id"], "p-123")
        self.assertEqual(image.meta.storage.patch_size, (64, 64, 64))

    def test_compress_and_decompress_raise_without_array(self):
        image = MLArray(_make_array(seed=4))
        image.close()

        with self.assertRaises(TypeError):
            image.compress()
        with self.assertRaises(TypeError):
            image.decompress()

    def test_decompress_matches_uncompressed_load_and_save_reuses_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=5)
            path = Path(tmpdir) / "layout.mla"
            MLArray(array, patch_size=(4, 8, 8)).save(path)

            decompressed = MLArray(path)
            decompressed.decompress()
            loaded = MLArray.load(path, compressed=False)

            self.assertEqual(decompressed.meta.storage.to_dict(), loaded.meta.storage.to_dict())
            self.assertEqual(decompressed.meta.storage.patch_size, (4, 8, 8))

            resaved = Path(tmpdir) / "resaved.mla"
            decompressed.save(resaved)
            self.assertEqual(MLArray(resaved).meta.storage.patch_size, (4, 8, 8))


if __name__ == "__main__":
    unittest.main()
