import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlarray import MLArray, MLARRAY_DEFAULT_PATCH_SIZE


def _make_array(shape=(16, 32, 32), seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=dtype)


class TestOptimizationExamples(unittest.TestCase):
    def test_default_patch_optimization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "default-opt.mla"

            MLArray(array).save(path)
            loaded = MLArray(path)

            self.assertEqual(
                loaded.meta.blosc2.patch_size,
                [MLARRAY_DEFAULT_PATCH_SIZE] * 3,
            )

    def test_explicit_patch_size_optimization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "patch-non-iso.mla"

            MLArray(array, patch_size=(8, 12, 16)).save(path)
            loaded = MLArray(path)

            self.assertEqual(loaded.meta.blosc2.patch_size, [8, 12, 16])

    def test_mmap_patch_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(shape=(32, 64, 64))
            path = Path(tmpdir) / "patch-read.mla"

            MLArray(array, patch_size=(8, 16, 16)).save(path)

            image = MLArray.open(path, mmap_mode="r")
            patch = image[10:20, 5:15, 7:17]

            self.assertEqual(patch.shape, (10, 10, 10))

    def test_mmap_in_place_modification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(shape=(16, 32, 32))
            path = Path(tmpdir) / "patch-write.mla"

            MLArray(array, patch_size=(8, 16, 16)).save(path)

            image = MLArray.open(path, mmap_mode="r+")
            image[0:2, 0:2, 0:2] *= 0.0
            image.close()

            reloaded = MLArray(path)
            self.assertTrue(np.allclose(reloaded[0:2, 0:2, 0:2], 0.0))

    def test_streamed_write_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shape = (16, 32, 32)
            dtype = np.float32
            path = Path(tmpdir) / "streamed-write.mla"

            image = MLArray.create(
                path,
                shape=shape,
                dtype=dtype,
                mmap_mode="w+",
                patch_size=8,
            )
            data = _make_array(shape=shape, seed=1, dtype=dtype)
            image[...] = data
            image.close()

            loaded = MLArray(path)
            self.assertEqual(loaded.shape, shape)
            self.assertTrue(np.allclose(loaded[...], data))

    def test_manual_chunk_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "manual-layout.mla"

            MLArray(
                array,
                patch_size=None,
                chunk_size=(1, 16, 16),
                block_size=(1, 8, 8),
            ).save(path)
            loaded = MLArray(path)

            self.assertEqual(loaded.meta.blosc2.chunk_size, [1, 16, 16])
            self.assertEqual(loaded.meta.blosc2.block_size, [1, 8, 8])

    def test_blosc2_auto_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "blosc2-auto.mla"

            MLArray(array, patch_size=None).save(path)
            loaded = MLArray(path)

            self.assertIsNone(loaded.meta.blosc2.patch_size)
            self.assertIsNotNone(loaded.meta.blosc2.chunk_size)
            self.assertIsNotNone(loaded.meta.blosc2.block_size)


if __name__ == "__main__":
    unittest.main()
